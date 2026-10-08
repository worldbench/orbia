"""Stable evaluator records and JSON-safe result helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import sys
from typing import Any, Mapping


VALID_STATUSES = frozenset(
    {"ok", "not_applicable", "not_available", "not_covered", "invalid_input", "evaluator_failed"}
)


@dataclass(frozen=True)
class MetricResult:
    """One independently resumable metric result."""

    name: str
    status: str
    applicable: bool
    version: str = "v0"
    coverage: Mapping[str, Any] = field(default_factory=dict)
    raw: Mapping[str, Any] = field(default_factory=dict)
    conditional_score: float | Mapping[str, float] | None = None
    effective_score: float | Mapping[str, float] | None = None
    failure: str | None = None
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in VALID_STATUSES:
            raise ValueError(f"unsupported evaluator status {self.status!r}")
        if self.status == "ok" and not self.applicable:
            raise ValueError("an ok metric result must be applicable")

    def to_dict(self) -> dict[str, Any]:
        result = {
            "metric_version": f"orbia.{self.name}.{self.version}",
            "status": self.status,
            "applicable": self.applicable,
            "coverage": dict(self.coverage),
            "raw": dict(self.raw),
            "reason": self.failure,
            "provenance": dict(self.provenance),
        }
        if self.conditional_score is not None:
            result["conditional_score"] = self.conditional_score
        if self.effective_score is not None:
            result["effective_score"] = self.effective_score
        return result


@dataclass(frozen=True)
class EvaluationPaths:
    case_root: Path
    video_path: Path
    output_root: Path
    pose_cache: Path | None
    depth_root: Path | None


@dataclass(frozen=True)
class EvaluationOptions:
    """Evaluation backends, sampling settings, and validity thresholds."""

    feature_backend: str = "dinov2"
    dino_repo: Path | None = None
    dino_checkpoint: Path | None = None
    lpips_checkpoint: Path | None = None
    musiq_checkpoint: Path | None = None
    aesthetic_clip_checkpoint: Path | None = None
    aesthetic_checkpoint: Path | None = None
    amt_config: Path | None = None
    amt_checkpoint: Path | None = None
    vbench_root: Path | None = None
    torch_home: Path | None = None
    da3_python: Path | None = field(default_factory=lambda: Path(sys.executable))
    da3_repo: Path | None = None
    da3_checkpoint: Path | None = None
    da3_streaming_python: Path | None = field(default_factory=lambda: Path(sys.executable))
    da3_streaming_repo: Path | None = None
    da3_streaming_checkpoint: Path | None = None
    da3_streaming_salad_checkpoint: Path | None = None
    da3_joint_max_frames: int = 240
    da3_geometry_resolution: int = 504
    da3_streaming_chunk_size: int = 120
    da3_streaming_overlap: int = 30
    da3_streaming_loop_enable: bool = True
    da3_context_views: int = 64
    da3_heldout_views: int = 16
    da3_input_resolution: int = 504
    da3_timeout_s: int = 1800
    device: str = "cpu"
    return_neighborhood_frames: int = 0
    qe_neighborhood_frames: int = 0
    pose_translation_threshold_normalized: float = 0.08
    pose_rotation_threshold_deg: float = 8.0
    pose_rotation_cost_weight: float = 1.0 / 45.0
    completion_translation_threshold_normalized: float = 0.10
    completion_rotation_threshold_deg: float = 10.0
    geometry_short_baseline_frames: int = 8
    geometry_medium_baseline_frames: int = 32
    geometry_max_short_pairs: int = 48
    geometry_max_medium_pairs: int = 16
    geometry_pixel_stride: int = 4
    geometry_min_depth: float = 1e-4
    geometry_max_depth: float = 1000.0
    geometry_visibility_relative_tolerance: float = 0.10
    geometry_feature_max_pairs: int = 16
    geometry_pointcloud_frame_stride: int = 4
    geometry_pointcloud_pixel_stride: int = 8
    geometry_pointcloud_relative_threshold: float = 0.05
    geometry_pointcloud_min_threshold: float = 0.02
    tier2_quality_frame_stride: int = 4
    tier2_musiq_batch_size: int = 16
    tier2_aesthetic_batch_size: int = 32
    tier2_amt_batch_size: int = 8
    tier2_amt_triplet_stride: int = 8
    geometry_frame_stride: int = 2
    control_frame_stride: int = 2
    anchor_tracking_frame_stride: int = 2
    anchor_tracking_stride: int = 4
    anchor_tracking_max_points: int = 64
    anchor_tracking_ratio_threshold: float = 0.7
    anchor_tracking_min_matches: int = 4
    anchor_tracking_visibility: Path | None = None
    anchor_sam_python: Path | None = field(default_factory=lambda: Path(sys.executable))
    anchor_sam_repo: Path | None = None
    anchor_sam_checkpoint: Path | None = None
    anchor_sam_config: str = 'sam3.1_multiplex_annotation_mask_seed'
    anchor_tracking_timeout_s: int = 1200
    anchor_identity_min_pixels: int = 16
    temporal_window_s: float = 5.0
    force: bool = False

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "EvaluationOptions":
        if not value:
            return cls()
        permitted = {field.name for field in cls.__dataclass_fields__.values()}
        unknown = sorted(set(value) - permitted)
        if unknown:
            raise ValueError(f"unknown evaluator config keys: {unknown}")
        normalized = dict(value)
        for key in ('da3_python', 'da3_streaming_python', 'anchor_sam_python'):
            if normalized.get(key) is None:
                normalized[key] = sys.executable
        normalized.setdefault("control_frame_stride", normalized.get("geometry_frame_stride", cls().geometry_frame_stride))
        for key in (
            "anchor_tracking_visibility", "anchor_sam_python", "anchor_sam_repo", "anchor_sam_checkpoint",
            "dino_repo",
            "dino_checkpoint",
            "lpips_checkpoint",
            "musiq_checkpoint",
            "aesthetic_clip_checkpoint",
            "aesthetic_checkpoint",
            "amt_config",
            "amt_checkpoint",
            "vbench_root",
            "torch_home",
            "da3_python",
            "da3_repo",
            "da3_checkpoint",
            "da3_streaming_python",
            "da3_streaming_repo",
            "da3_streaming_checkpoint",
            "da3_streaming_salad_checkpoint",
        ):
            if normalized.get(key) is not None:
                normalized[key] = Path(normalized[key])
        return cls(**normalized)
