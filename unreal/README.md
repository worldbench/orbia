# Build Cases in Unreal Engine

The UE pipeline plans a camera trajectory, renders RGB, depth and object IDs in Unreal Editor, and converts the render into ORBIA cases. It was developed with UE 5.7. You need your own UE project and scenes.

## Setup

```bash
bash tools/install_ue.sh
conda activate orbia-ue
```

In Unreal Editor, enable the **Python Editor Script Plugin**, **Movie Render Queue** and **Movie Render Queue Additional Render Passes** plugins.

## 1. Plan the trajectory

Edit [configs/ue.json](../configs/ue.json):

| Field | Description |
| --- | --- |
| `work_dir` | Directory for trajectories and renders |
| `output_dir` | Output dataset directory |
| `map` | UE map to load |
| `origin_ue` | Start camera position (cm) and rotation |
| `camera` | Resolution, focal length and sensor size |
| `cases` | Case ID, template, query pose, caption and anchor actors |

```bash
python tools/build_ue.py plan --config configs/ue.json
```

This writes `trajectory_spec.json` for each case (e.g. `work_dir/ue_forward/`) and a `plan_index.json`. If `long_case_id` is set, a 961-frame version is planned as well.

## 2. Render in Unreal Editor

Load the map, set the post-process volume to manual exposure, and disable moving actors. Then run in the Unreal Python console (replace the paths):

```python
import os, runpy
os.environ['ORBIA_PLAN_INDEX'] = '/path/to/work_dir/plan_index.json'
os.environ['ORBIA_PROBE_OUTPUT'] = '/path/to/work_dir/clearance.json'
os.environ['ORBIA_TRAJECTORY_SPEC'] = '/path/to/work_dir/ue_forward/trajectory_spec.json'
os.environ['ORBIA_CAPTURE_ROOT'] = '/path/to/work_dir/ue_forward'
os.environ['ORBIA_SEQUENCE_PATH'] = '/Game/ORBIA/Sequences/Forward'

runpy.run_path('/path/to/orbia/unreal/probe_trajectory_clearance.py', run_name='__main__')  # collision check
runpy.run_path('/path/to/orbia/unreal/import_trajectory_to_sequence.py')                   # create Level Sequence
runpy.run_path('/path/to/orbia/unreal/export_camera_metadata.py')                          # export cameras
runpy.run_path('/path/to/orbia/unreal/render_ground_truth_mrq.py')                         # render with MRQ
```

RGB, depth and object IDs are rendered into one multilayer EXR per frame, with motion blur and depth of field disabled. Wait for Movie Render Queue to finish before the next step. For the long case, repeat with its own `trajectory_spec.json`, capture directory and sequence path.

## 3. Extract

```bash
python tools/build_ue.py extract --config configs/ue.json --case-id ue_forward
```

This writes `rgb/`, `depth/` (metric Z depth), `instance_mask/` and `instances.json` (object ID → actor name).

## 4. Export

Choose the anchor actors from `instances.json` and add them to the case in `configs/ue.json`:

```json
{"anchor_id": "A1", "label": "building entrance", "actor_label": "Entrance"}
```

```bash
python tools/build_ue.py convert --config configs/ue.json --case-id ue_forward
```

The query frame and revisit pairs are selected on the rendered video using depth and anchor masks (`curation` in the config). The selection and alternatives are saved to `<capture>/curation.json` for review. The output has the same format as the released dataset ([docs/data.md](../docs/data.md)).
