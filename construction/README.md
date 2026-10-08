# Build Cases from Real Videos

This pipeline builds ORBIA cases from SpatialVID, Sekai, MiraData, ScanNet++ and self-recorded videos. Each case pairs two registered RGB-D views (input and query) with a synthesized trajectory that passes through both. The pipeline follows the four steps in the paper:

1. **Select reference views**: read or estimate cameras and depth, then pick input/query pairs that fit a trajectory template.
2. **Discover and verify anchors**: Qwen3.5 proposes landmarks, SAM 3.1 segments them in both views, and tracking checks that both masks show the same object. For ScanNet++, anchors are rendered from the semantic mesh.
3. **Synthesize trajectories and revisits**: fit the template through the two views, check camera clearance, select revisits, and create the minute-long version.
4. **Rank and review**: filter scenes with a VLM, rank candidates, review them in a browser, and export the accepted cases.

## Setup

```bash
bash tools/install_construction.sh
conda activate orbia-construction
python tools/download_weights.py --workflow construction --config configs/construction/spatialvid.json
```

This writes `configs/construction/spatialvid.local.json`. Set the source paths and frames in it.

| Source | Config | Input | Geometry |
| --- | --- | --- | --- |
| SpatialVID | [spatialvid.json](../configs/construction/spatialvid.json) | Official release (video, cameras, depth) | Provided |
| Sekai | [sekai.json](../configs/construction/sekai.json) | Video clip | DA3 or ViPE |
| MiraData | [mira.json](../configs/construction/mira.json) | Video clip | DA3 or ViPE |
| Self-recorded | [self_captured.json](../configs/construction/self_captured.json) | Video clip | DA3 or ViPE |
| ScanNet++ | [scannetpp.json](../configs/construction/scannetpp.json) | iPhone RGB-D, COLMAP cameras, semantic mesh | Sensor depth |
| Annotated RGB-D | [construction.json](../configs/construction.json) | Your own RGB-D frames, masks and caption | Provided |

## Run

```bash
CONFIG=configs/construction/spatialvid.local.json

python tools/build_dataset.py --config $CONFIG --stage geometry     # cameras and depth
python tools/build_dataset.py --config $CONFIG --stage candidates   # input/query pairs
python tools/build_dataset.py --config $CONFIG --stage annotate     # anchors, trajectories, revisits
python tools/build_dataset.py --config $CONFIG --stage review       # scene review, ranking, review pages
```

Open `work_dir/review/<case_id>/index.html` to check the anchors, projections, trajectory and revisits of each case. In `work_dir/cases.json`, set `"accepted": true` for the cases to keep, and optionally `"selected_anchor_ids"` to keep only some anchors. Then export:

```bash
python tools/build_dataset.py --config $CONFIG --stage export
```

The output has the same format as the released dataset ([docs/data.md](../docs/data.md)). Each stage caches its results, so it can be rerun.

## Stages

**geometry.** `geometry.mode` chooses where cameras and depth come from:

- `provided`: use the cameras and depth that come with the source (SpatialVID release, ScanNet++, prepared RGB-D).
- `da3`: estimate cameras, intrinsics and depth jointly from the selected frames with [Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3). Default for Sekai, MiraData and self-recorded videos.
- `vipe`: run [ViPE](https://github.com/nv-tlabs/vipe) on the whole video and read the selected frames. Set `geometry.output` to reuse an existing ViPE output directory.

**candidates.** Enumerates frame pairs and keeps those whose relative motion fits the template, e.g. mostly forward translation for `forward_backward` or mostly rotation for `yaw_return` ([pose_gates.json](../configs/construction/pose_gates.json)). Pairs are ranked by viewpoint change and visible overlap, and saved to `work_dir/cases.json`, which can also be edited by hand.

**annotate.** This stage does most of the work for each pair:

- *Anchors.* Qwen3.5 looks at five frames between the two views, checks that the scene is usable and proposes static, bounded landmarks. SAM 3.1 segments each landmark in both views. Masks are filtered by area (0.3–50% of the image), connectivity and holes, and duplicates are merged. For SpatialVID, Sekai and MiraData, the input mask is tracked to the query view and must match the query mask (IoU ≥ 0.8). Self-recorded videos skip tracking, since the anchor is chosen during recording. ScanNet++ renders anchor masks from the semantic mesh instead.
- *Shared surfaces.* The input anchor is back-projected with depth into the query view. The part of the query anchor that is visible in the input becomes the evaluation region.
- *Trajectory.* The template is fitted through the two views and back to the start (357 frames). Camera clearance is checked against the fused depth, or against the mesh for ScanNet++, and failed routes are retried with other parameters.
- *Revisits.* Pairs of temporally separated visits are selected where both views see the same surfaces that are not visible in the input.
- *Long version.* With `include_long`, the trajectory is extended to 961 frames with three revisit pairs, keeping the same input, query and anchors.
- *Caption.* Qwen3.5 writes a factual scene caption from the source frames.

**review.** Qwen3.5 reviews each remaining candidate for shot changes, prominent people or moving objects, poor visibility, bad weather and subtitles or watermarks (`annotation.scene_review`). The rest are ranked with a 70-point score covering image sharpness, anchor quality, trajectory quality and revisit quality, and a review page is written for each case.

**export.** Writes the accepted cases. Short and long cases share the same input image and references.

Thresholds can be changed in each config under `annotation`, `visibility`, `trajectory_qa` and `revisits`.
