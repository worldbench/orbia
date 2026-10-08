"""Queue a synchronized RGB, WorldDepth, and actor Object-ID MRQ render."""

import json
import os
from pathlib import Path

import unreal


SPEC = json.loads(Path(os.environ["ORBIA_TRAJECTORY_SPEC"]).read_text(encoding="utf-8"))
FRAME_COUNT = int(SPEC["frame_count"])
FPS = int(SPEC["fps"])
WIDTH, HEIGHT = map(int, SPEC["camera"]["resolution_px"])
ACTIVE_EXECUTOR = None
HIDDEN_DYNAMIC_ACTORS = []


def _on_finished(executor, success):
    global ACTIVE_EXECUTOR, HIDDEN_DYNAMIC_ACTORS
    unreal.log(f"ORBIA synchronized ground truth finished: success={bool(success)}")
    for actor in HIDDEN_DYNAMIC_ACTORS:
        try:
            actor.set_actor_hidden_in_game(False)
        except Exception as error:
            unreal.log_warning(f"Could not restore dynamic actor {actor}: {error}")
    HIDDEN_DYNAMIC_ACTORS = []
    ACTIVE_EXECUTOR = None
    unreal.EditorPythonScripting.set_keep_python_script_alive(False)


def _required_env(name):
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"Required environment variable {name} is not set")
    return value


def _set(setting, name, value):
    """Set an editor property while producing a useful UE-version error."""
    try:
        setting.set_editor_property(name, value)
    except Exception as exc:
        raise RuntimeError(
            f"This Unreal version does not expose {setting.get_class().get_name()}."
            f"{name}; adapt only this MRQ bridge and keep the output contract"
        ) from exc


def _world_depth_material():
    # The mount point is stable, while get_path_name() differs across UE builds.
    candidates = (
        "/MovieRenderPipeline/Materials/MovieRenderQueue_WorldDepth",
        "/Engine/Plugins/MovieScene/MovieRenderPipeline/Content/Materials/"
        "MovieRenderQueue_WorldDepth.MovieRenderQueue_WorldDepth",
    )
    for path in candidates:
        material = unreal.load_asset(path)
        if material is not None:
            return material
    raise RuntimeError(
        "Could not load MovieRenderQueue_WorldDepth. Enable Movie Render "
        "Pipeline, restart Unreal Editor, and verify the engine plugin content."
    )


def _require_object_id_plugin():
    render_pass_class = getattr(unreal, "MoviePipelineObjectIdRenderPass", None)
    id_type_class = getattr(unreal, "MoviePipelineObjectIdPassIdType", None)
    if render_pass_class is None or id_type_class is None:
        raise RuntimeError(
            "MoviePipelineObjectIdRenderPass is unavailable. Enable 'Movie "
            "Render Queue Additional Render Passes', restart Unreal Editor, "
            "and run this script again."
        )
    return render_pass_class, id_type_class


sequence_path = _required_env("ORBIA_SEQUENCE_PATH")
capture_root = Path(_required_env("ORBIA_CAPTURE_ROOT")).expanduser().resolve()
exr_dir = capture_root / "multilayer_exr"
partial_manifest_path = capture_root / "capture_manifest.partial.json"

if exr_dir.exists() and any(exr_dir.iterdir()):
    raise RuntimeError(
        f"Refusing to overwrite non-empty MRQ output directory: {exr_dir}. "
        "Use a new versioned case/run path."
    )
if partial_manifest_path.exists():
    raise RuntimeError(
        f"Refusing to overwrite existing render provenance: {partial_manifest_path}. "
        "Use a new versioned case/run path."
    )
exr_dir.mkdir(parents=True, exist_ok=True)

sequence = unreal.load_asset(sequence_path)
if sequence is None:
    raise RuntimeError(f"Could not load Level Sequence {sequence_path}")

world = unreal.EditorLevelLibrary.get_editor_world()
if world is None:
    raise RuntimeError("No editor world is loaded")

subsystem = unreal.get_editor_subsystem(unreal.MoviePipelineQueueSubsystem)
queue = subsystem.get_queue()
existing_jobs = list(queue.get_jobs())
if existing_jobs:
    names = ", ".join(str(job.job_name) for job in existing_jobs)
    raise RuntimeError(
        "Movie Render Queue already contains jobs. Clear or render them first "
        f"so this script cannot accidentally execute unrelated work: {names}"
    )

object_id_class, object_id_type_class = _require_object_id_plugin()
id_type_name = os.environ.get("ORBIA_OBJECT_ID_TYPE", "ACTOR_WITH_HIERARCHY").strip().upper()
id_type = getattr(object_id_type_class, id_type_name, None)
if id_type is None:
    choices = "FULL, MATERIAL, ACTOR, ACTOR_WITH_HIERARCHY, FOLDER, LAYER"
    raise RuntimeError(f"Unsupported ORBIA_OBJECT_ID_TYPE={id_type_name}; choose {choices}")
spatial_samples = int(os.environ.get("ORBIA_SPATIAL_SAMPLES", "8"))
if spatial_samples < 1:
    raise RuntimeError("ORBIA_SPATIAL_SAMPLES must be at least 1")
random_seed = int(os.environ.get("ORBIA_RANDOM_SEED", "0"))
exposure_offset_ev = float(os.environ.get("ORBIA_EXPOSURE_OFFSET_EV", "0.0"))
disable_dynamic_actors = os.environ.get("ORBIA_DISABLE_DYNAMIC_ACTORS", "1") == "1"
world_depth_material = _world_depth_material()

if disable_dynamic_actors:
    actor_subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    for actor in actor_subsystem.get_all_level_actors():
        if actor.get_class().get_name() == "NiagaraActor":
            actor.set_actor_hidden_in_game(True)
            HIDDEN_DYNAMIC_ACTORS.append(actor)

job = queue.allocate_new_job(unreal.MoviePipelineExecutorJob)
job.job_name = f"ORBIA_GT_{sequence.get_name()}"
job.sequence = unreal.SoftObjectPath(sequence.get_path_name())
job.map = unreal.SoftObjectPath(world.get_path_name())
configuration = job.get_configuration()

deferred = configuration.find_or_add_setting_by_class(
    unreal.MoviePipelineDeferredPassBase
)
_set(deferred, "disable_multisample_effects", True)
# UE 5.7 removed this reflected switch; multilayer EXR output remains floating
# point, so only set it on builds that still expose it.
if hasattr(deferred, "use_32_bit_post_process_materials"):
    _set(deferred, "use_32_bit_post_process_materials", True)
world_depth = unreal.MoviePipelinePostProcessPass()
world_depth.enabled = True
world_depth.material = world_depth_material
_set(deferred, "additional_post_process_materials", [world_depth])

object_ids = configuration.find_or_add_setting_by_class(object_id_class)
_set(object_ids, "id_type", id_type)
if hasattr(object_ids, "include_translucent_objects"):
    _set(
        object_ids,
        "include_translucent_objects",
        os.environ.get("ORBIA_INCLUDE_TRANSLUCENT", "0") == "1",
    )

exr_output = configuration.find_or_add_setting_by_class(
    unreal.MoviePipelineImageSequenceOutput_EXR
)
_set(exr_output, "multilayer", True)
if hasattr(unreal, "EXRCompressionFormat") and hasattr(
    unreal.EXRCompressionFormat, "PIZ"
):
    _set(exr_output, "compression", unreal.EXRCompressionFormat.PIZ)

output = configuration.find_or_add_setting_by_class(unreal.MoviePipelineOutputSetting)
output.output_directory.path = str(exr_dir)
output.file_name_format = "{frame_number_rel}"
output.output_resolution = unreal.IntPoint(WIDTH, HEIGHT)
output.use_custom_frame_rate = True
output.output_frame_rate = unreal.FrameRate(FPS, 1)
output.use_custom_playback_range = True
output.custom_start_frame = 0
# MRQ playback end is exclusive.
output.custom_end_frame = FRAME_COUNT
output.zero_pad_frame_numbers = 6

anti_aliasing = configuration.find_or_add_setting_by_class(
    unreal.MoviePipelineAntiAliasingSetting
)
_set(anti_aliasing, "override_anti_aliasing", True)
_set(anti_aliasing, "anti_aliasing_method", unreal.AntiAliasingMethod.AAM_NONE)
_set(anti_aliasing, "spatial_sample_count", spatial_samples)
_set(anti_aliasing, "temporal_sample_count", 1)

console = configuration.find_or_add_setting_by_class(
    unreal.MoviePipelineConsoleVariableSetting
)
console_variables = {
    "r.MotionBlurQuality": 0.0,
    "r.DepthOfFieldQuality": 0.0,
    # Preserve the level's fixed exposure configuration.
    "r.DefaultFeature.AutoExposure": 1.0,
    "r.EyeAdaptationQuality": 2.0,
    "r.ExposureOffset": exposure_offset_ev,
    "r.DynamicRes.OperationMode": 0.0,
    "FX.FreezeParticleSimulation": 1.0,
    "FX.FreezeGPUSimulation": 1.0,
    "r.Water.FreezeWaves": 1.0,
    "r.Test.OverrideTimeMaterialExpressions": 0.0,
}
for key, value in console_variables.items():
    if hasattr(console, "add_or_update_console_variable"):
        console.add_or_update_console_variable(key, value)
    else:
        console.console_variables = console_variables
        break

partial_manifest = {
    "contract": "orbia.unreal.capture-partial.v1",
    "project": Path(unreal.Paths.get_project_file_path()).stem,
    "map": world.get_path_name(),
    "ue_version": str(unreal.SystemLibrary.get_engine_version()),
    "sequence": sequence_path,
    "source_asset_identifier": os.environ.get("ORBIA_ASSET_SOURCE"),
    "source_asset_license": os.environ.get("ORBIA_ASSET_LICENSE"),
    "random_seed": random_seed,
    "enabled_capture_plugins": [
        "Movie Render Pipeline",
        "Movie Render Queue Additional Render Passes",
    ],
    "resolution_px": [WIDTH, HEIGHT],
    "fps": FPS,
    "frame_count": FRAME_COUNT,
    "mrq": {
        "output": "multilayer OpenEXR",
        "layers": ["FinalImage", "MovieRenderQueue_WorldDepth", "ObjectIds"],
        "object_id_type": id_type_name,
        "disable_multisample_effects": True,
        "spatial_samples": spatial_samples,
        "temporal_samples": 1,
        "anti_aliasing_method": "None",
        "motion_blur": False,
        "depth_of_field": False,
        "exposure_mode": "level_post_process_volume",
        "automatic_exposure": False,
        "exposure_offset_ev": exposure_offset_ev,
        "dynamic_resolution": False,
        "dynamic_policy": {
            "niagara_hidden": disable_dynamic_actors,
            "particle_simulation_frozen": True,
            "gpu_particle_simulation_frozen": True,
            "water_waves_frozen": True,
            "material_time_s": 0.0,
        },
    },
}
partial_manifest_path.write_text(
    json.dumps(partial_manifest, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)

unreal.log(
    f"ORBIA queued synchronized ground truth: {sequence_path} -> {exr_dir} "
    f"({WIDTH}x{HEIGHT}, {FPS} FPS, frames 0..{FRAME_COUNT - 1}, "
    f"{spatial_samples} spatial samples)"
)
if os.environ.get("ORBIA_MRQ_QUEUE_ONLY", "0") == "1":
    unreal.log("ORBIA_MRQ_QUEUE_ONLY=1: job configured but render was not started")
    for actor in HIDDEN_DYNAMIC_ACTORS:
        actor.set_actor_hidden_in_game(False)
    HIDDEN_DYNAMIC_ACTORS = []
else:
    ACTIVE_EXECUTOR = unreal.MoviePipelinePIEExecutor(subsystem)
    ACTIVE_EXECUTOR.on_executor_finished_delegate.add_callable_unique(_on_finished)
    unreal.EditorPythonScripting.set_keep_python_script_alive(True)
    subsystem.render_queue_with_executor_instance(ACTIVE_EXECUTOR)
    unreal.log(
        "MRQ is asynchronous. Wait for all output EXRs before running "
        "python tools/build_ue.py extract."
    )
