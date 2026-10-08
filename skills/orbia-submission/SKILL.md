---
name: orbia-submission
description: Prepare and check a model submission for the ORBIA leaderboard. Use when the user wants to submit a model's ORBIA videos or scores to the leaderboard.
---

# Submit a model to the ORBIA leaderboard

Run commands from the ORBIA repository root.

## 1. Build the folder

```text
my-model/
├── meta.json
├── videos/<case_id>.mp4   # all 720 cases
├── conditions/            # optional, copied from generate.py output
└── results/               # only if self-evaluated
```

Videos come from the `orbia-generation` skill (copy `videos/` and `conditions/` from its output). Every case in `data/orbia/public/` needs exactly one video with 357 or 961 frames.

`meta.json`:

```json
{
  "model_name": "my-model",
  "display_name": "My Model 1.0",
  "type": "camera",
  "tags": ["5B", "4-step"],
  "org": "My Lab",
  "contact": "you@example.com",
  "url": "https://github.com/my-lab/my-model"
}
```

`type` is `camera`, `action` or `text`. At most 4 tags of up to 16 characters (model size, steps, variant). Ask the user for the display name, organization and contact instead of guessing.

## 2. Evaluate (optional, Path A)

Follow the `orbia-evaluation` skill with `--videos my-model/videos --model <model_name> --output my-model/results`, adding `--control-profile action_only` for `action` or `--group ti2v` for `text`. Without results, the maintainers run the evaluation (Path B).

## 3. Check

```bash
python tools/check_submission.py my-model
```

Fix every `x` issue. `note:` lines (e.g. a frame rate other than 16 FPS) are fine.

## 4. Send

Upload the folder to Hugging Face (preferred), Google Drive, OneDrive or Baidu Netdisk. Draft an email for the user to send to dongyue.lu@u.nus.edu and guianfang@u.nus.edu:

- Subject: `[ORBIA Submission] <display_name>`
- Model name, type, organization, one-line description
- Path A (self-evaluated) or Path B (videos only)
- Download link
- Attach `meta.json`, plus `results/scores/scores.json` for Path A

Do not send the email or upload files without the user's confirmation. See `docs/submission.md` for details.
