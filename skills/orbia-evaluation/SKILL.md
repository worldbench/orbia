---
name: orbia-evaluation
description: Evaluate generated videos on the ORBIA benchmark and read the T1–T6 and Overall scores. Use when the user has ORBIA videos and wants scores, selected tiers, or per-case results.
---

# Evaluate videos on ORBIA

Run commands from the ORBIA repository root.

## 1. Setup (once)

```bash
bash tools/install_evaluation.sh
conda activate orbia-eval
python tools/download_weights.py --workflow evaluation
python tools/verify_install.py --workflow evaluation --cuda
```

The dataset should be at `data/orbia` (otherwise pass `--dataset`).

## 2. Check the videos

Videos are either a folder of `<case_id>.mp4` or a `videos.jsonl` written by `generate.py`. Each video needs 357 frames (short case) or 961 frames (long case); frame *i* is matched to the *i*-th requested pose. Any resolution works. A wrong frame count makes that case fail.

## 3. Run

```bash
# camera-conditioned model
python evaluate.py --videos videos/my-model --devices 0,1,2,3

# action-conditioned model
python evaluate.py --videos videos/my-model --control-profile action_only --devices 0,1,2,3

# image+text-to-video model
python evaluate.py --videos videos/my-model --group ti2v --devices 0,1,2,3
```

- `--tiers T3 T4` runs only some tiers; later runs with the same `--output` add to the same results.
- Rerunning the same command resumes unfinished cases.
- `--model NAME` sets the model name (default: folder name).
- Keep the default config and checkpoints, otherwise scores are not comparable with the leaderboard.

## 4. Read the results

- `results/scores/scores.csv`: T1–T6, Overall and ranks per model. Overall needs all six tiers; T6 needs both short and long videos.
- `results/<model>/results/<case_id>/report.json`: per-case measurements, useful for finding failed or outlier cases.

Tiers: T1 Control, T2 Visual Quality, T3 Input Preservation, T4 Generated-Content Persistence, T5 3D Self-Consistency, T6 Temporal Stability. Overall = 25/25/10/10/10/20%. See `docs/evaluation.md` for details.
