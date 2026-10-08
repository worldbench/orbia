"""Bake an ORBIA trajectory into a CineCamera level sequence."""

import json
import os
from pathlib import Path

import unreal


spec_path = Path(os.environ["ORBIA_TRAJECTORY_SPEC"])
sequence_path = os.environ.get(
    "ORBIA_SEQUENCE_PATH", "/Game/ORBIA/Sequences/Trajectory"
)
camera_label = os.environ.get("ORBIA_CAMERA_LABEL", "ORBIA_Camera")
spec = json.loads(spec_path.read_text(encoding="utf-8"))
frame_count = int(spec["frame_count"])
fps = int(spec["fps"])
if frame_count < 2 or fps < 1 or len(spec["frames"]) != frame_count:
    raise RuntimeError("Invalid trajectory frame count or clock")

camera_config = spec["camera"]
binding_mode = os.environ.get("ORBIA_CAMERA_BINDING_MODE", "possessable").strip().lower()
if binding_mode not in {"possessable", "spawnable"}:
    raise RuntimeError(
        "ORBIA_CAMERA_BINDING_MODE must be 'possessable' or 'spawnable', got "
        f"{binding_mode!r}"
    )

package_path, asset_name = sequence_path.rsplit("/", 1)
if unreal.EditorAssetLibrary.does_asset_exist(sequence_path):
    raise RuntimeError(
        f"Refusing to overwrite existing sequence {sequence_path}; use a new versioned path"
    )
asset_tools = unreal.AssetToolsHelpers.get_asset_tools()
sequence = asset_tools.create_asset(
    asset_name, package_path, unreal.LevelSequence, unreal.LevelSequenceFactoryNew()
)
sequence.set_display_rate(unreal.FrameRate(fps, 1))
sequence.set_tick_resolution_directly(unreal.FrameRate(fps, 1))
sequence.set_playback_start(0)
sequence.set_playback_end(frame_count)

actor_subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
first = spec["frames"][0]["ue_world"]
qx, qy, qz, qw = first["quaternion_xyzw"]
first_location = unreal.Vector(*[float(v) for v in first["location_cm"]])
first_rotation = unreal.Quat(float(qx), float(qy), float(qz), float(qw)).rotator()


def _configure_camera(camera_actor):
    component = camera_actor.get_cine_camera_component()
    if component is None:
        raise RuntimeError("CineCameraActor has no CineCameraComponent")
    component.set_editor_property(
        "current_focal_length", float(camera_config["focal_length_mm"])
    )
    filmback = component.get_editor_property("filmback")
    filmback.sensor_width = float(camera_config["sensor_width_mm"])
    filmback.sensor_height = float(camera_config["sensor_height_mm"])
    component.set_editor_property("filmback", filmback)
    try:
        focus = component.get_editor_property("focus_settings")
        focus.focus_method = unreal.CameraFocusMethod.DISABLE
        component.set_editor_property("focus_settings", focus)
    except Exception as error:
        unreal.log_warning(f"Could not disable camera depth of field: {error}")


temporary_camera = None
if binding_mode == "possessable":
    cameras = [
        actor
        for actor in actor_subsystem.get_all_level_actors()
        if isinstance(actor, unreal.CineCameraActor)
        and actor.get_actor_label() == camera_label
    ]
    if len(cameras) > 1:
        raise RuntimeError(
            f"Expected at most one CineCameraActor labelled {camera_label}, "
            f"found {len(cameras)}"
        )
    camera_actor = cameras[0] if cameras else actor_subsystem.spawn_actor_from_class(
        unreal.CineCameraActor, first_location, first_rotation
    )
    if camera_actor is None:
        raise RuntimeError("Could not create the possessable CineCameraActor")
    camera_actor.set_actor_label(camera_label)
    _configure_camera(camera_actor)
    binding = sequence.add_possessable(camera_actor)
else:
    # UE 5.7 class-only spawnables can lose the configured component.  Copy a
    # configured temporary instance so this opt-in mode preserves the lens.
    temporary_camera = actor_subsystem.spawn_actor_from_class(
        unreal.CineCameraActor, first_location, first_rotation
    )
    if temporary_camera is None:
        raise RuntimeError("Could not create the temporary CineCameraActor")
    temporary_camera.set_actor_label(camera_label)
    _configure_camera(temporary_camera)
    binding = sequence.add_spawnable_from_instance(temporary_camera)
try:
    binding.set_display_name(camera_label)
except Exception:
    pass
track = binding.add_track(unreal.MovieScene3DTransformTrack)
section = track.add_section()
section.set_range(0, frame_count)
channels = section.get_all_channels()
if len(channels) < 9:
    raise RuntimeError(f"Expected at least 9 transform channels, found {len(channels)}")


def _unwrap(previous, current):
    if previous is None:
        return current
    while current - previous > 180.0:
        current -= 360.0
    while current - previous < -180.0:
        current += 360.0
    return current


previous_rotation = [None, None, None]
for frame in spec["frames"]:
    index = int(frame["frame_index"])
    location = frame["ue_world"]["location_cm"]
    qx, qy, qz, qw = frame["ue_world"]["quaternion_xyzw"]
    rotator = unreal.Quat(qx, qy, qz, qw).rotator()
    # MovieScene3DTransform channel order: translation XYZ, rotation XYZ,
    # scale XYZ. Rotation X/Y/Z correspond to roll/pitch/yaw.
    rotation = [float(rotator.roll), float(rotator.pitch), float(rotator.yaw)]
    rotation = [_unwrap(previous_rotation[i], rotation[i]) for i in range(3)]
    previous_rotation = rotation
    values = [float(v) for v in location] + rotation + [1.0, 1.0, 1.0]
    for channel, value in zip(channels[:9], values):
        channel.add_key(unreal.FrameNumber(index), value)

# UE 5.7 exposes the camera cut as a sequence track rather than a master track.
cut_track = sequence.add_track(unreal.MovieSceneCameraCutTrack)
cut_section = cut_track.add_section()
cut_section.set_range(0, frame_count)
cut_section.set_camera_binding_id(sequence.get_binding_id(binding))

unreal.EditorAssetLibrary.save_loaded_asset(sequence)
if temporary_camera is not None:
    actor_subsystem.destroy_actor(temporary_camera)
unreal.log(
    f"ORBIA baked {frame_count} camera keys into {sequence_path}; "
    f"camera_binding={binding_mode}"
)
if binding_mode == "possessable":
    unreal.log(
        f"ORBIA retained {camera_label} in the current level. Save the level "
        "before closing the editor or reopening the sequence in another process."
    )
