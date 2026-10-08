# Evaluation

## Usage

```bash
python evaluate.py --videos videos/my-model
```

`videos/my-model/` holds one `<case_id>.mp4` per case. Short cases need 357 frames and long cases 961 frames. Evaluation aligns frames by index, so frame *i* must follow the *i*-th requested pose; 16 FPS is recommended but the stored frame rate is not used. Any resolution works. The model name defaults to the folder name.

| Option | Default | Description |
| --- | --- | --- |
| `--videos` | | Video folder, or a `.jsonl` video list |
| `--dataset` | `data/orbia` | Dataset root |
| `--config` | `configs/evaluation.local.yaml` | Model paths, written by `download_weights.py` |
| `--output` | `results` | Output directory |
| `--tiers` | all | Any of `T1 … T6` |
| `--devices` | `0` | GPU IDs, e.g. `0,1,2,3` |
| `--model` | folder name | Model name in the score table |
| `--group` | `world` | `world` or `ti2v` |
| `--control-profile` | `pose_and_action` | Use `action_only` for action-conditioned models |
| `--stage` | `full` | `geometry` only recovers cameras and depth |

Running the same command again resumes from where it stopped. Running tiers separately (e.g. `--tiers T1` and later `--tiers T3`) with the same `--output` adds to the same results.

## Multiple models

To evaluate several models at once, or videos with other file names, pass a `.jsonl` file with one line per video. Relative paths are resolved against the file's directory.

```json
{"model": "model-a", "case_id": "case_0001", "video": "model-a/case_0001.mp4", "group": "world"}
{"model": "model-b", "case_id": "case_0001", "video": "model-b/case_0001.mp4", "group": "ti2v", "control_profile": "action_only"}
```

```bash
python evaluate.py --videos videos/all.jsonl
```

`generate.py` writes this file automatically as `videos.jsonl`.

## Outputs

```text
results/
├── scores/scores.csv              # T1–T6, Overall and ranks per model
├── scores/scores.json
└── <model>/results/<case_id>/report.json   # per-case measurements
```

Overall is computed once all six tiers are available. T6 needs both short and long videos.

## Scoring

Each measurement is normalized per case with fixed endpoints ([src/metrics/scoring.json](../src/metrics/scoring.json)), averaged over cases, and combined into tier scores. Overall weights T1–T6 by 25/25/10/10/10/20%.

- **T1 Control Alignment**: translation and rotation error after global scale alignment, plus Move/Look accuracy. Action-conditioned models are scored on Move/Look only.
- **T2 Visual Quality**: MUSIQ, LAION aesthetic, flicker, AMT smoothness and HPSv3.
- **T3 Input Preservation**: return (30%), query (40%) and anchor identity (30%). Return and query compare scene and anchor regions against the registered references with DINO, PSNR, SSIM, LPIPS and depth AbsRel.
- **T4 Generated-Content Persistence**: compares each revisit against the first visit in appearance and depth, plus a surface F-score.
- **T5 3D Self-Consistency**: cross-view depth and DINO consistency, regional scale agreement on a 4×4 grid, and Gaussian reconstruction from 64 context views rendered at 16 held-out views.
- **T6 Temporal Stability**: quality, control and local geometry in 5-second windows. Each indicator combines its level (30%) and retention relative to the first window (70%); short and long videos are weighted 30% and 70%.

Cameras and depth of generated videos are recovered with DA3 (DA3 Streaming for long videos).
