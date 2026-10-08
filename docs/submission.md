# Submit to the ORBIA Leaderboard

Send your model's videos to us by email and we will add it to the leaderboard on the ORBIA project page. You can either evaluate the videos yourself (Path A), or send only the videos and let us run the evaluation (Path B).

| | Path A: self-evaluation | Path B: evaluated by us |
| --- | --- | --- |
| You run | Generation and `evaluate.py` | Generation only |
| You send | Videos, `meta.json`, `results/` | Videos, `meta.json` |
| We do | Re-run a subset of cases to verify the scores | Run the full evaluation |
| Turnaround | Faster | Processed in batches |

## 1. Generate videos

Generate one video for every case in the dataset: all 720 cases (480 short, 240 long). Short cases have 357 frames and long cases 961 frames, one frame per requested pose. 16 FPS is recommended. Any resolution is fine. See [models.md](models.md) for how to run your model.

## 2. Prepare the folder

```text
my-model/
├── meta.json
├── videos/
│   ├── case_0001.mp4
│   └── ...
└── results/          # Path A only
```

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

- `type` is `camera`, `action` or `text` (image+text-to-video), following the conditioning your model uses.
- `tags` are optional, at most 4 short tags (up to 16 characters each) about model size, inference steps or variants, e.g. `5B`, `4-step`, `distilled`, `realtime`.
- `url` is optional: a paper, project page or code link.

If you generated the videos with `generate.py`, also include its `conditions/` folder. It records the exact control given to the model and helps us check the setup.

## 3. Evaluate (Path A only)

```bash
# camera-conditioned
python evaluate.py --videos my-model/videos --model my-model --output my-model/results

# action-conditioned
python evaluate.py --videos my-model/videos --model my-model --output my-model/results \
  --control-profile action_only

# image+text-to-video
python evaluate.py --videos my-model/videos --model my-model --output my-model/results \
  --group ti2v
```

This writes the scores to `my-model/results/scores/` and per-case reports to `my-model/results/my-model/results/`. Do not change the evaluation config or model checkpoints.

## 4. Check the folder

```bash
python tools/check_submission.py my-model
```

This checks `meta.json`, that every case has exactly one video, and the frame count and frame rate of each video.

## 5. Send it

Upload the folder to Hugging Face (preferred), Google Drive, OneDrive or Baidu Netdisk, and send an email to **dongyue.lu@u.nus.edu** and **guianfang@u.nus.edu** with:

- Subject: `[ORBIA Submission] <display_name>`
- Model name, type, organization and a one-line description
- Path A or Path B
- The download link
- `meta.json` (and `results/scores/scores.json` for Path A) as attachments

Please keep the link available until the model is listed. If something fails our checks, we will reply with the list of issues.
