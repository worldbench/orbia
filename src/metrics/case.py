"""Typed evaluator view of the unified public/reference bundle."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src.utils.camera import CameraIntrinsics


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def camera_from_document(value: Mapping[str, Any], *, name: str) -> CameraIntrinsics:
    try:
        return CameraIntrinsics(
            width=int(value["width"]),
            height=int(value["height"]),
            K=np.asarray(value["K"], dtype=np.float64),
            model=str(value.get("model", "PINHOLE")),
            distortion=tuple(float(item) for item in value.get("distortion", ())),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid {name} camera document") from exc


@dataclass(frozen=True)
class HiddenFrame:
    role: str
    rgb: np.ndarray
    depth_m: np.ndarray
    depth_valid: np.ndarray
    rgb_camera: CameraIntrinsics
    depth_camera: CameraIntrinsics
    camera_document: Mapping[str, Any]


@dataclass(frozen=True)
class EvaluationCase:
    root: Path
    case_id: str
    q0_rgb: np.ndarray
    canonical_camera: CameraIntrinsics
    native_to_canonical: np.ndarray
    requested_poses: np.ndarray
    timestamps_s: np.ndarray
    events: Mapping[str, int]
    completion_pairs: Mapping[str, Any]
    revisit_pairs: Mapping[str, Mapping[str, Any]]
    q0: HiddenFrame
    qe: HiddenFrame
    observability_input_supported_depth: np.ndarray
    observability_invalid_depth: np.ndarray
    q0_anchor_masks_rgb: Mapping[str, np.ndarray]
    qe_anchor_masks_rgb: Mapping[str, np.ndarray]
    anchor_dirs: tuple[Path, ...]
    template: str = ""
    source: str = ""
    appearance_support_rgb: np.ndarray | None = None
    appearance_anchor_masks_rgb: Mapping[str, np.ndarray] | None = None

    @property
    def frame_count(self) -> int:
        return len(self.requested_poses)



def load_evaluation_case(root: Path | str) -> EvaluationCase:
    """Read a public/<case_id> directory from the unified reference bundle."""
    from src.data.case_loader import load_case
    root = Path(root).resolve()
    return load_case(root.parent.parent, root.name).to_evaluation_case()
