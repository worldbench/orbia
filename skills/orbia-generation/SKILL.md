---
name: orbia-generation
description: Run a video world model on the ORBIA benchmark cases and produce one video per case. Use when the user wants to add, wrap or run a camera-conditioned, action-conditioned or image+text-to-video model on ORBIA data.
---

# Generate ORBIA videos with a model

Goal: one video per case, `videos/<case_id>.mp4`, with exactly 357 frames (short) or 961 frames (long). Frame *i* must follow the *i*-th requested camera pose. Run commands from the ORBIA repository root.

## 1. Setup

```bash
pip install -r requirements/models.txt          # in the model's own environment
hf download Orbia/Orbia --repo-type dataset --local-dir data/download
python tools/extract_dataset.py --archives data/download --output data/orbia
```

## 2. Pick the interface

Look at what the model takes as control and copy the matching template:

| Model control | Template | Implement |
| --- | --- | --- |
| Camera poses / extrinsics | `examples/camera_model.py` | `generate_with_poses(image, prompt, poses_c2w, K, *, case, history)` |
| Keyboard actions (WASD, arrows, IJKL) | `examples/action_model.py` | `generate_with_actions(image, prompt, plan, *, case)` |
| Image + text prompt | `examples/text_model.py` | `generate_segment(context, prompt, num_frames, *, case, index)` |

Load weights in `setup()`. Return a list of uint8 RGB frames `[H, W, 3]`. Any resolution is fine.

## 3. Match the model's conventions

- **Camera**: ORBIA poses are OpenCV camera-to-world, relative to the input view. Convert with `src.models.camera` (`opencv_to_opengl`, `c2w_to_w2c`, `rebase`, `local_increments`, `plucker_rays`). If the model resizes or crops the image, update `K` with `camera.resize_and_crop` / `scale_intrinsics`. For models with a fixed clip length, set `window_frames` and `overlap_frames`; windows are chained automatically.
- **Action**: set `block_spec` (frames per block, degrees per look step) for streaming models, or `action_mode = 'timed'` with `timed_spec` for SDK/browser products. Calibrate the turning rate first: hold one look key, then measure rotation with `python evaluate.py --videos <clips> --stage geometry`.
- **Image+text**: set `native_fps` to the model's frame rate so prompt times match its clock, and `segment_frames` to its maximum clip length.
- Read the model's official inference code for its pose format, image size and frame count before writing the adapter.

## 4. Test, then run

```bash
# quick test on two cases
python generate.py --model my_model:MyModel --dataset data/orbia \
  --output videos/my-model --cases case_0001 case_0002

# all cases, one process per GPU
CUDA_VISIBLE_DEVICES=0 python generate.py --model my_model:MyModel \
  --dataset data/orbia --output videos/my-model --shard 0 --num-shards 8
python generate.py --output videos/my-model --merge
```

Check a few output videos visually: the camera should follow the requested path and return to the start. Rerunning skips finished cases. The output folder contains `videos/`, `conditions/` (the control given to the model) and `videos.jsonl` for evaluation.

See `docs/models.md` for details.
