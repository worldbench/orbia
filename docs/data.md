# Dataset Format

```text
orbia/
├── manifest.json            # case list: source, template, frame count, parent case
├── public/<case_id>/        # model inputs
│   ├── q0.png               # input image
│   ├── prompt.txt           # scene caption
│   ├── case.json            # intrinsics, units, events, revisit pairs
│   └── camera.npz           # poses_c2w [N, 4, 4], timestamps_s [N]
└── references/<case_id>/    # used only by the evaluator
    ├── reference.json
    ├── q0/, qe/             # rgb.png, depth.npy, valid.png
    ├── anchors/<anchor_id>/ # q0.png, qe.png anchor masks
    ├── evaluation_regions/  # scene.png and <anchor_id>.png in the query view
    └── observability/       # input_supported.png, invalid.png
```

Models only read `public/`. `references/` is used for evaluation.

## Loading a case

```python
from src.data import load_case

case = load_case("data/orbia", "case_0001", references=True)
case.q0_rgb                  # input image, uint8 [H, W, 3]
case.prompt                  # scene caption
case.poses_c2w               # [N, 4, 4] camera poses
case.timestamps_s            # [N]
case.frames["qe"].depth      # query reference depth
case.anchor_masks            # anchor masks
```

## Conventions

- **Cameras**: OpenCV camera-to-world (X right, Y down, Z forward), relative to the input view.
- **Depth**: camera-axis Z. Invalid pixels are marked by the valid mask.
- **Units**: `pose_unit` / `depth_unit` is `metre` for ScanNet++ and Unreal Engine, `source_pose_unit` for SpatialVID's released geometry, and `DA3_estimated_unit` for DA3-estimated geometry. The control score aligns a global scale, so non-metric units are fine.
- **Timing**: short cases have 357 frames and long cases 961 frames, with trajectories sampled at 16 FPS. `case.json` lists the frame index and time of each event (input, query, first visit, revisit, return). Long cases link to their short case through `parent_case_id`.
- **Evaluation regions**: `scene.png` marks the part of the query view visible from the input view; `<anchor_id>.png` is its intersection with each anchor. Query-view appearance is measured inside these regions.
