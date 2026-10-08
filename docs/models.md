# Run Your Model on ORBIA

ORBIA only evaluates the output videos, so any model can be added. `src.models` converts each case into the control your model takes (camera poses, actions or text), runs generation in windows if needed, and saves the videos.

```bash
pip install -r requirements/models.txt
```

## Inputs

```python
from src.models import load_model_case

case = load_model_case('data/orbia', 'case_0001')
case.image        # uint8 [H, W, 3] input image
case.prompt       # scene caption
case.poses_c2w    # [N, 4, 4] OpenCV camera-to-world, relative to the input view
case.K            # [3, 3] intrinsics of case.image
case.num_frames   # 357 (short) or 961 (long)
case.template     # trajectory template
case.events       # frame indices of the input, query, revisits and return
```

The model must output one video per case with exactly `case.num_frames` frames, where frame *i* follows the *i*-th pose. Any resolution is fine. The trajectories are sampled at 16 FPS, and `generate.py` saves videos at 16 FPS. A model with another native frame rate (e.g. 24 FPS) still produces one frame per pose; for image+text-to-video models, set `native_fps` so the times in the prompt match the model's clock.

## Camera-conditioned models

```python
from src.models import CameraModel, camera

class MyCameraModel(CameraModel):
    window_frames = 81     # frames per call (None: whole video in one call)
    overlap_frames = 1     # frames shared between consecutive windows

    def generate_with_poses(self, image, prompt, poses_c2w, K, *, case, history=None):
        image, K = camera.resize_and_crop(image, K, (832, 480))
        w2c = camera.c2w_to_w2c(camera.opencv_to_opengl(poses_c2w))
        return self.pipeline(image, prompt, w2c, K, num_frames=len(poses_c2w))
```

For long videos, the trajectory is split into windows. Each window's poses are relative to its first pose and the next window starts from the last generated frame (`history` holds all previous frames). `src.models.camera` has helpers for coordinate conversion, intrinsics scaling, cropping and Plücker rays. See [examples/camera_model.py](../examples/camera_model.py).

## Action-conditioned models

Camera poses are converted into `W A S D` (move) and `I J K L` (look) keys:

```python
from src.models import ActionModel
from src.models.actions import BlockActionSpec

class MyActionModel(ActionModel):
    action_mode = 'block'
    block_spec = BlockActionSpec(first_block_frames=9, block_frames=12,
                                 yaw_step_deg=20.0, pitch_step_deg=17.0)

    def generate_with_actions(self, image, prompt, plan, *, case):
        return self.stream(image, prompt, plan['actions'])  # one 8-dim key vector per block
```

`action_mode = 'block'` gives one key vector per generation block. `action_mode = 'timed'` gives key-press intervals in seconds, which suits interactive products controlled through a browser or SDK. Set the turning speed (`yaw_step_deg`, `pitch_step_deg`) to match your model: generate a few clips holding one look key and measure the rotation with `evaluate.py --stage geometry`.

Action-conditioned models are scored on Move/Look accuracy only. See [examples/action_model.py](../examples/action_model.py).

## Image+text-to-video models

The trajectory is described as a timed text prompt, e.g. move forward for 3 seconds, then turn left by 60 degrees. The prompt is appended to the scene caption.

```python
from src.models import TextModel

class MyTextModel(TextModel):
    native_fps = 24.0       # prompt timing follows the model's frame rate
    segment_frames = 121    # frames per request (None: one request)
    context_frames = 1      # frames passed on to the next request

    def generate_segment(self, context, prompt, num_frames, *, case, index):
        return self.api(image=context[-1], prompt=prompt, frames=num_frames)
```

To see the prompt of a case:

```python
from src.models import camera_text
print(camera_text.camera_prompt(case, native_fps=24)['prompt'])
```

See [examples/text_model.py](../examples/text_model.py).

## Generate and evaluate

```bash
python generate.py --model examples.camera_model:MyCameraModel \
  --dataset data/orbia --output videos/my-model
python evaluate.py --videos videos/my-model/videos.jsonl
```

To use several GPUs, start one process per GPU with `--shard i --num-shards n`, then merge:

```bash
CUDA_VISIBLE_DEVICES=0 python generate.py --model ... --output videos/my-model --shard 0 --num-shards 8
# ... shards 1-7
python generate.py --output videos/my-model --merge
```

Finished cases are skipped when the command is rerun. `static-camera`, `static-action` and `static-text` are dummy models that repeat the input frame, useful for testing the pipeline.

Output:

```text
videos/my-model/
├── videos/<case_id>.mp4
├── conditions/<case_id>.json   # the control given to the model
└── videos.jsonl                # video list for evaluate.py
```

If you generate videos with your own code, you can also put them in a folder as `<case_id>.mp4` and evaluate the folder directly.
