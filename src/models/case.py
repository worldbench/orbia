"""Model-facing view of one benchmark case."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src.data.case_loader import load_case

BENCHMARK_FPS = 16
SHORT_FRAMES = 357
LONG_FRAMES = 961


@dataclass(frozen=True)
class ModelCase:
    """Everything a world model may consume for one case."""

    case_id: str
    image: np.ndarray
    image_path: Path
    prompt: str
    poses_c2w: np.ndarray
    timestamps_s: np.ndarray
    K: np.ndarray
    width: int
    height: int
    template: str
    events: Mapping[str, int]
    pose_unit: str = 'unknown'
    parent_case_id: str | None = None
    anchor_label: str | None = None
    document: Mapping[str, Any] = field(default_factory=dict, repr=False)

    @property
    def num_frames(self) -> int:
        return int(len(self.poses_c2w))

    @property
    def fps(self) -> int:
        return BENCHMARK_FPS

    @property
    def is_long(self) -> bool:
        return self.num_frames >= LONG_FRAMES

    @property
    def duration_s(self) -> float:
        return float(self.timestamps_s[-1] - self.timestamps_s[0])

    @property
    def event_frames(self) -> list[int]:
        return sorted(set(int(i) for i in self.events.values()))


def load_model_case(dataset_root: str | Path, case_id: str) -> ModelCase:
    """Load the public inputs of ``case_id`` from an extracted ORBIA dataset."""
    case = load_case(dataset_root, case_id, references=False)
    doc = case.document
    poses = np.linalg.inv(case.poses_c2w[0]) @ case.poses_c2w
    intrinsics = doc['intrinsics']
    return ModelCase(
        case_id=case.case_id,
        image=case.q0_rgb,
        image_path=Path(case.dataset_root) / 'public' / case.case_id / 'q0.png',
        prompt=case.prompt,
        poses_c2w=poses,
        timestamps_s=np.asarray(case.timestamps_s, dtype=np.float64),
        K=np.asarray(intrinsics['K'], dtype=np.float64),
        width=int(intrinsics['width']),
        height=int(intrinsics['height']),
        template=doc['template'],
        events={name: int(e['frame_index']) for name, e in doc['events'].items()},
        pose_unit=doc['pose_unit'],
        parent_case_id=doc['parent_case_id'],
        document=doc,
    )


def list_case_ids(dataset_root: str | Path, horizon: str | None = None) -> list[str]:
    """Return public case IDs, optionally filtered to ``short`` or ``long``."""
    root = Path(dataset_root) / 'public'
    ids = sorted(p.name for p in root.iterdir() if (p / 'case.json').is_file())
    if horizon is None:
        return ids
    if horizon not in ('short', 'long'):
        raise ValueError("horizon must be 'short', 'long' or None")
    import json
    keep = []
    for case_id in ids:
        count = json.loads((root / case_id / 'case.json').read_text())['frame_count']
        if (count >= LONG_FRAMES) == (horizon == 'long'):
            keep.append(case_id)
    return keep
