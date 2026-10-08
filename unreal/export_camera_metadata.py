"""Export evaluated CineCamera poses and lens settings for the capture clock."""

import json
import os
from pathlib import Path

import unreal


spec = json.loads(Path(os.environ["ORBIA_TRAJECTORY_SPEC"]).read_text(encoding="utf-8"))
frame_count = int(spec["frame_count"])
fps = int(spec["fps"])
resolution = list(map(int, spec["camera"]["resolution_px"]))
sequence_path = os.environ["ORBIA_SEQUENCE_PATH"]
camera_label = os.environ.get("ORBIA_CAMERA_LABEL", "ORBIA_Camera")
capture_root = Path(os.environ["ORBIA_CAPTURE_ROOT"])
sequence = unreal.load_asset(sequence_path)
if sequence is None:
    raise RuntimeError(f"Could not load sequence {sequence_path}")

actor_subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
camera = next(
    (
        actor
        for actor in actor_subsystem.get_all_level_actors()
        if isinstance(actor, unreal.CineCameraActor)
        and actor.get_actor_label() == camera_label
    ),
    None,
)
if camera is None:
    raise RuntimeError(f"Could not find CineCameraActor labelled {camera_label}")

unreal.LevelSequenceEditorBlueprintLibrary.open_level_sequence(sequence)
component = camera.get_cine_camera_component()
frames = []
for frame_index in range(frame_count):
    # UE 5.7 accepts an int32 frame here; a FrameNumber fails nativization.
    unreal.LevelSequenceEditorBlueprintLibrary.set_current_time(
        frame_index
    )
    # The camera is an unparented possessable, so its actor transform is the
    # evaluated camera-to-world transform.
    transform = camera.get_actor_transform()
    location = transform.translation
    rotation = transform.rotation
    frames.append(
        {
            "frame_index": frame_index,
            "timestamp_s": frame_index / float(fps),
            "location_cm": [float(location.x), float(location.y), float(location.z)],
            "quaternion_xyzw": [
                float(rotation.x),
                float(rotation.y),
                float(rotation.z),
                float(rotation.w),
            ],
            "focal_length_mm": float(component.current_focal_length),
            "sensor_width_mm": float(component.filmback.sensor_width),
            "sensor_height_mm": float(component.filmback.sensor_height),
            "resolution_px": resolution,
        }
    )

capture_root.mkdir(parents=True, exist_ok=True)
(capture_root / "camera_frames.json").write_text(
    json.dumps(
        {
            "contract": "orbia.unreal.camera-frames.v1",
            "source": "final editor-evaluated CineCameraComponent",
            "frames": frames,
        },
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)
unreal.log(f"ORBIA wrote {frame_count} evaluated camera frames to {capture_root}")
