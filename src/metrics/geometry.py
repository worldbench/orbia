"""Camera-aware estimator depth diagnostics for ORBIA Tier 3."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.spatial import cKDTree

from src.metrics.case import EvaluationCase
from src.metrics.contract import EvaluationOptions, MetricResult
from src.metrics.features import ImageComparator
from src.metrics.reconstruction import load_vipe_depths
from src.metrics.pose import PoseCache
from src.metrics.video import VideoInfo


@dataclass(frozen=True)
class GeometryBundle:
    """Estimator output in its native pose/depth coordinate system."""

    case: EvaluationCase
    info: VideoInfo
    poses_by_index: Mapping[int, np.ndarray]
    depths_by_index: Mapping[int, np.ndarray]
    depth_root: Path
    # Per-frame K matrices in decoded-video pixel coordinates (from DA3).
    intrinsics_by_index: Mapping[int, np.ndarray] | None = None

    @property
    def indices(self) -> tuple[int, ...]:
        return tuple(sorted(set(self.poses_by_index) & set(self.depths_by_index)))

    def camera_matrix(self, index: int) -> np.ndarray:
        depth = _depth_image(self.depths_by_index[index])
        height, width = depth.shape
        if self.intrinsics_by_index is not None and index in self.intrinsics_by_index:
            source_K = np.asarray(self.intrinsics_by_index[index], dtype=np.float64)
            source_width, source_height = self.info.width, self.info.height
        else:
            source_K = self.case.canonical_camera.K
            source_width = self.case.canonical_camera.width
            source_height = self.case.canonical_camera.height
        scale = np.array(
            [[width / source_width, 0.0, 0.0],
             [0.0, height / source_height, 0.0],
             [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        return scale @ source_K


@dataclass(frozen=True)
class ReprojectionCorrespondence:
    source_index: int
    target_index: int
    source_rows: np.ndarray
    source_columns: np.ndarray
    target_rows: np.ndarray
    target_columns: np.ndarray
    projected_depth: np.ndarray
    target_depth: np.ndarray
    attempted_source_samples: int
    failure_reasons: tuple[str, ...] = ()
    target_height: int | None = None
    target_width: int | None = None

    @property
    def count(self) -> int:
        return int(len(self.projected_depth))

    @property
    def coverage(self) -> float:
        return float(self.count / max(self.attempted_source_samples, 1))

    @property
    def available(self) -> bool:
        return not any(reason.startswith("missing_") for reason in self.failure_reasons)


def _depth_image(value: np.ndarray) -> np.ndarray:
    depth = np.asarray(value, dtype=np.float64)
    if depth.ndim == 3:
        depth = depth[..., 0]
    if depth.ndim != 2:
        raise ValueError("estimator depth must be HxW or HxWx1")
    return depth


def infer_depth_root(pose_cache: Path | None, depth_root: Path | None) -> Path | None:
    if depth_root is not None:
        return depth_root
    if pose_cache is None or not pose_cache.is_dir():
        return None
    candidate = pose_cache.parent / "depth"
    return candidate if candidate.is_dir() else None


def load_geometry_bundle(
    *, case: EvaluationCase, info: VideoInfo, cache: PoseCache | None, depth_root: Path | None,
) -> GeometryBundle | None:
    if cache is None or depth_root is None:
        return None
    depths = load_vipe_depths(depth_root)
    poses = {int(index): pose for pose, index in zip(cache.poses_c2w, cache.frame_indices)}
    intrinsics = (
        None
        if cache.intrinsics is None
        else {
            int(index): intrinsic
            for intrinsic, index in zip(cache.intrinsics, cache.frame_indices)
        }
    )
    return GeometryBundle(
        case=case,
        info=info,
        poses_by_index=poses,
        depths_by_index={int(index): _depth_image(depth) for index, depth in depths.items()},
        depth_root=depth_root.resolve(),
        intrinsics_by_index=intrinsics,
    )


def _subsample(values: Sequence[tuple[int, int]], maximum: int) -> tuple[tuple[int, int], ...]:
    if maximum < 1:
        raise ValueError("geometry pair maximum must be positive")
    if len(values) <= maximum:
        return tuple(values)
    positions = np.linspace(0, len(values) - 1, maximum, dtype=np.int64)
    return tuple(values[int(position)] for position in positions)


def pair_schedule(bundle: GeometryBundle, options: EvaluationOptions) -> dict[str, tuple[tuple[int, int], ...]]:
    # The candidate schedule depends only on the protocol, not estimator success.
    if hasattr(bundle, 'case') and hasattr(bundle.case, 'events'):
        from src.metrics.sampling import geometry_sampling_plan
        pool = geometry_sampling_plan(bundle.case, bundle.info, options, include_gs=False)['frame_indices']
    else:
        pool = list(range(0, bundle.info.frame_count, options.geometry_frame_stride))
    def build(delta: int, maximum: int) -> tuple[tuple[int, int], ...]:
        if delta < 1:
            raise ValueError("geometry baselines must be positive frame counts")
        candidates = []
        for position, index in enumerate(pool[:-1]):
            target = index + delta
            if target > pool[-1]:
                continue
            # Deterministic closest sampled timestamp; ties use the earlier frame.
            right = min(pool[position+1:], key=lambda j: (abs(j-target), j))
            candidates.append((index, right))
        base = _subsample(candidates, maximum)
        return tuple(direction for left, right in base for direction in ((left, right), (right, left)))

    medium_frames = getattr(
        options, "geometry_medium_baseline_frames",
        getattr(options, "geometry_long_baseline_frames", 32),
    )
    max_medium_pairs = getattr(
        options, "geometry_max_medium_pairs",
        getattr(options, "geometry_max_long_pairs", 16),
    )

    return {
        "short": build(options.geometry_short_baseline_frames, options.geometry_max_short_pairs),
        "medium": build(medium_frames, max_medium_pairs),
    }


def _empty_correspondence(
    source_index: int, target_index: int, *reasons: str, attempted_source_samples: int = 0,
) -> ReprojectionCorrespondence:
    return ReprojectionCorrespondence(
        source_index=source_index, target_index=target_index,
        source_rows=np.empty(0, dtype=np.int64), source_columns=np.empty(0, dtype=np.int64),
        target_rows=np.empty(0, dtype=np.int64), target_columns=np.empty(0, dtype=np.int64),
        projected_depth=np.empty(0, dtype=np.float64), target_depth=np.empty(0, dtype=np.float64),
        attempted_source_samples=attempted_source_samples, failure_reasons=tuple(reasons),
    )


def correspondence_for_pair(
    bundle: GeometryBundle,
    source_index: int,
    target_index: int,
    *, pixel_stride: int,
    min_depth: float,
    max_depth: float,
    visibility_relative_tolerance: float,
) -> ReprojectionCorrespondence:
    if (
        pixel_stride < 1
        or min_depth <= 0.0
        or max_depth <= min_depth
        or visibility_relative_tolerance < 0.0
    ):
        raise ValueError("invalid Tier-3 depth sampling limits")
    source = bundle.depths_by_index[source_index]
    target = bundle.depths_by_index[target_index]
    source_height, source_width = source.shape
    target_height, target_width = target.shape
    source_K, target_K = bundle.camera_matrix(source_index), bundle.camera_matrix(target_index)
    rows, columns = np.mgrid[0:source_height:pixel_stride, 0:source_width:pixel_stride]
    source_depth = source[rows, columns]
    valid = np.isfinite(source_depth) & (source_depth >= min_depth) & (source_depth <= max_depth)
    rows, columns, source_depth = rows[valid], columns[valid], source_depth[valid]
    attempted = int(len(source_depth))
    if not attempted:
        return _empty_correspondence(source_index, target_index, "no_valid_source_depth")
    camera = np.stack([
        (columns - source_K[0, 2]) * source_depth / source_K[0, 0],
        (rows - source_K[1, 2]) * source_depth / source_K[1, 1],
        source_depth,
        np.ones_like(source_depth),
    ], axis=1)
    source_c2w, target_c2w = bundle.poses_by_index[source_index], bundle.poses_by_index[target_index]
    target_camera = (np.linalg.inv(target_c2w) @ source_c2w @ camera.T).T
    z = target_camera[:, 2]
    positive = z > 1e-8
    target_camera, rows, columns, z = target_camera[positive], rows[positive], columns[positive], z[positive]
    u = np.rint(target_K[0, 0] * target_camera[:, 0] / z + target_K[0, 2]).astype(np.int64)
    v = np.rint(target_K[1, 1] * target_camera[:, 1] / z + target_K[1, 2]).astype(np.int64)
    inside = (u >= 0) & (u < target_width) & (v >= 0) & (v < target_height)
    rows, columns, z, u, v = rows[inside], columns[inside], z[inside], u[inside], v[inside]
    if not len(z):
        return _empty_correspondence(source_index, target_index, "zero_visibility", attempted_source_samples=attempted)
    # Keep only the front-most source point for each target pixel.
    linear = v * target_width + u
    order = np.lexsort((z, linear))
    keep = np.r_[True, linear[order][1:] != linear[order][:-1]]
    selected = order[keep]
    rows, columns, z, u, v = rows[selected], columns[selected], z[selected], u[selected], v[selected]
    observed = target[v, u]
    valid_observed = (
        np.isfinite(observed)
        & (observed >= min_depth)
        & (observed <= max_depth)
    )
    # Visibility and consistency are separate.  Only a projected point behind
    # a closer target surface is occluded; a projected point in front remains
    # an evaluable depth inconsistency.
    occluded = valid_observed & (z > observed * (1.0 + visibility_relative_tolerance))
    co_visible = valid_observed & ~occluded
    reasons = () if np.any(co_visible) else ("zero_visibility",)
    return ReprojectionCorrespondence(
        source_index=source_index, target_index=target_index,
        source_rows=rows[co_visible].astype(np.int64), source_columns=columns[co_visible].astype(np.int64),
        target_rows=v[co_visible].astype(np.int64), target_columns=u[co_visible].astype(np.int64),
        projected_depth=z[co_visible].astype(np.float64), target_depth=observed[co_visible].astype(np.float64),
        attempted_source_samples=attempted, failure_reasons=reasons,
        target_height=target_height, target_width=target_width,
    )


def build_correspondences(
    bundle: GeometryBundle, pairs: Mapping[str, Iterable[tuple[int, int]]], options: EvaluationOptions,
) -> dict[str, tuple[ReprojectionCorrespondence, ...]]:
    grouped: dict[str, tuple[ReprojectionCorrespondence, ...]] = {}
    for name, values in pairs.items():
        items: list[ReprojectionCorrespondence] = []
        for source, target in values:
            reasons = tuple(reason for reason, missing in (
                ("missing_source_pose", source not in bundle.poses_by_index),
                ("missing_target_pose", target not in bundle.poses_by_index),
                ("missing_source_depth", source not in bundle.depths_by_index),
                ("missing_target_depth", target not in bundle.depths_by_index),
            ) if missing)
            if reasons:
                items.append(_empty_correspondence(source, target, *reasons))
            else:
                items.append(correspondence_for_pair(
                    bundle, source, target, pixel_stride=options.geometry_pixel_stride,
                    min_depth=options.geometry_min_depth, max_depth=options.geometry_max_depth,
                    visibility_relative_tolerance=options.geometry_visibility_relative_tolerance,
                ))
        grouped[name] = tuple(items)
    return grouped


def _failure_counts(items: Iterable[ReprojectionCorrespondence]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        for reason in item.failure_reasons:
            counts[reason] = counts.get(reason, 0) + 1
    return counts


def _coverage_counts(items: Iterable[ReprojectionCorrespondence], *, evaluated: int) -> dict[str, Any]:
    values = tuple(items)
    return {
        "scheduled_pair_count": len(values),
        "available_pair_count": sum(item.available for item in values),
        "usable_pair_count": sum(item.count > 0 for item in values),
        "evaluated_pair_count": int(evaluated),
        "failure_reasons": _failure_counts(values),
    }


def _summary(values: Iterable[float]) -> dict[str, float | None]:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        return {"mean": None, "median": None, "p95": None}
    return {"mean": float(array.mean()), "median": float(np.median(array)), "p95": float(np.percentile(array, 95))}


def _pair_depth_errors(item: ReprojectionCorrespondence) -> dict[str, Any]:
    if not item.count:
        return {
            "source_frame_index": item.source_index, "target_frame_index": item.target_index,
            "frame_delta": item.target_index - item.source_index,
            "correspondence_count": 0, "coverage": item.coverage,
            "abs_rel": None, "abs_rel_median": None, "abs_rel_p95": None,
            "abs_log": None, "scale_ratio": None, "failure_reasons": list(item.failure_reasons),
        }
    residual = np.abs(item.projected_depth - item.target_depth)
    abs_rel = residual / np.maximum(item.target_depth, 1e-8)
    ratio = item.target_depth / np.maximum(item.projected_depth, 1e-8)
    return {
        "source_frame_index": item.source_index, "target_frame_index": item.target_index,
            "frame_delta": item.target_index - item.source_index,
        "correspondence_count": item.count,
        "coverage": item.coverage,
        "abs_rel": float(np.mean(abs_rel)),
        "abs_rel_median": float(np.median(abs_rel)),
        "abs_rel_p95": float(np.percentile(abs_rel, 95)),
        "abs_log": float(np.median(np.abs(np.log(np.maximum(ratio, 1e-8))))),
        "scale_ratio": float(np.median(ratio)),
        "failure_reasons": [],
    }


def depth_temporal_consistency_metric(
    correspondences: Mapping[str, Iterable[ReprojectionCorrespondence]], *, provenance: Mapping[str, Any],
) -> MetricResult:
    raw: dict[str, Any] = {}
    total_pairs = total_available = total_usable = total_points = 0
    total_failures: dict[str, int] = {}
    for name, items in correspondences.items():
        items = tuple(items)
        values = [_pair_depth_errors(item) for item in items]
        usable = [value for value in values if value["correspondence_count"]]
        total_pairs += len(values)
        total_available += sum(item.available for item in items)
        total_usable += len(usable)
        total_points += sum(int(value["correspondence_count"]) for value in usable)
        for reason, count in _failure_counts(items).items():
            total_failures[reason] = total_failures.get(reason, 0) + count
        raw[name] = {
            **_coverage_counts(items, evaluated=len(usable)),
            "correspondence_count": sum(int(value["correspondence_count"]) for value in usable),
            "abs_rel": _summary(float(value["abs_rel"]) for value in usable if value["abs_rel"] is not None),
            "abs_rel_median": _summary(float(value["abs_rel_median"]) for value in usable if value["abs_rel_median"] is not None),
            "abs_rel_p95": _summary(float(value["abs_rel_p95"]) for value in usable if value["abs_rel_p95"] is not None),
            "abs_log": _summary(float(value["abs_log"]) for value in usable if value["abs_log"] is not None),
            "co_visible_coverage": _summary(float(value["coverage"]) for value in values),
            "pairs": values,
        }
    status = "ok" if total_usable else "not_covered"
    return MetricResult(
        name="depth_temporal_consistency", status=status, applicable=True,
        coverage={"scheduled_pair_count": total_pairs, "available_pair_count": total_available,
                  "usable_pair_count": total_usable, "evaluated_pair_count": total_usable,
                  "correspondence_count": total_points, "failure_reasons": total_failures},
        raw=raw, failure=None if total_usable else "no co-visible finite estimator depth correspondences",
        provenance=provenance,
    )


def covisible_reprojection_metric(
    correspondences: Mapping[str, Iterable[ReprojectionCorrespondence]], *, frames: Mapping[int, np.ndarray],
    comparator: ImageComparator, max_pairs: int, provenance: Mapping[str, Any],
) -> MetricResult:
    grouped: dict[str, dict[str, Any]] = {}
    all_details: list[dict[str, Any]] = []
    total_scheduled = total_available = total_usable = total_selected = 0
    failure_counts: dict[str, int] = {}
    backend_failure = comparator.official_spatial_dino_unavailable_reason
    for group, values in correspondences.items():
        values = tuple(values)
        # Sample the registered schedule, not only successful pairs: failed
        # pairs cannot be silently replaced by easier later pairs.
        selected = {index for index, _ in _subsample(
            [(index, index) for index in range(len(values))], max_pairs
        )} if values else set()
        details: list[dict[str, Any]] = []
        pair_records: list[dict[str, Any]] = []
        for index, item in enumerate(values):
            reasons = list(item.failure_reasons)
            is_selected = index in selected
            if is_selected and item.count and item.source_index not in frames:
                reasons.append("missing_source_rgb")
            if is_selected and item.count and item.target_index not in frames:
                reasons.append("missing_target_rgb")
            if is_selected and item.count and backend_failure is not None:
                reasons.append("dino_backend_unavailable")
            record: dict[str, Any] = {
                "group": group, "source_frame_index": item.source_index,
                "target_frame_index": item.target_index,
                "frame_delta": item.target_index - item.source_index,
                "available": item.available, "usable": bool(item.count),
                "selected_for_feature_evaluation": is_selected, "evaluated": False,
                "correspondence_count": item.count, "co_visible_coverage": item.coverage,
                "failure_reasons": reasons,
            }
            if not is_selected or reasons or not item.count:
                pair_records.append(record)
                continue
            value = comparator.compare_corresponding_pixels(
                frames[item.source_index], frames[item.target_index],
                source_rows=item.source_rows,
                source_columns=item.source_columns,
                target_rows=item.target_rows,
                target_columns=item.target_columns,
                source_cache_key=("frame", item.source_index),
                target_cache_key=("frame", item.target_index),
            )
            if value["feature_backend"] != "dinov2":
                raise RuntimeError("official Tier-3 DINO metric received a non-DINO descriptor")
            record.update({"evaluated": True, **value})
            pair_records.append(record)
            details.append(record)
        all_details.extend(details)
        group_failures: dict[str, int] = {}
        for record in pair_records:
            for reason in record["failure_reasons"]:
                group_failures[reason] = group_failures.get(reason, 0) + 1
                failure_counts[reason] = failure_counts.get(reason, 0) + 1
        coverage = _coverage_counts(values, evaluated=len(details))
        coverage.update(selected_pair_count=len(selected), failure_reasons=group_failures)
        total_scheduled += len(values)
        total_available += sum(item.available for item in values)
        total_usable += sum(item.count > 0 for item in values)
        total_selected += len(selected)
        grouped[group] = {
            **coverage,
            "correspondence_count": sum(
                int(item["correspondence_count"]) for item in details
            ),
            "photometric_l1": _summary(
                float(item["photometric_l1"]) for item in details
            ),
            "feature_reprojection_similarity": _summary(
                float(item["feature_reprojection_similarity"]) for item in details
            ),
            "dino_reprojection_similarity": _summary(
                float(item["feature_reprojection_similarity"]) for item in details
            ),
            "edge_alignment": _summary(
                float(item["edge_alignment"])
                for item in details
                if item["edge_alignment"] is not None
            ),
            "co_visible_coverage": _summary(
                float(item["co_visible_coverage"]) for item in details
            ),
            "source_patch_coverage": _summary(float(item["source_patch_coverage"]) for item in details),
            "target_patch_coverage": _summary(float(item["target_patch_coverage"]) for item in details),
            "unique_patch_pair_count": sum(int(item["unique_patch_pair_count"]) for item in details),
            "pairs": pair_records,
        }
    status = "not_available" if backend_failure is not None else ("ok" if all_details else "not_covered")
    return MetricResult(
        name="covisible_reprojection", status=status, applicable=True,
        version="v0.1",
        coverage={
            "scheduled_pair_count": total_scheduled,
            "available_pair_count": total_available,
            "usable_pair_count": total_usable,
            "selected_pair_count": total_selected,
            "evaluated_pair_count": len(all_details),
            "correspondence_count": sum(
                int(item["correspondence_count"]) for item in all_details
            ),
            "failure_reasons": failure_counts,
        },
        raw={
            **grouped,
            "all": {
                "photometric_l1": _summary(
                    float(item["photometric_l1"]) for item in all_details
                ),
                "feature_reprojection_similarity": _summary(
                    float(item["feature_reprojection_similarity"])
                    for item in all_details
                ),
                "dino_reprojection_similarity": _summary(
                    float(item["feature_reprojection_similarity"]) for item in all_details
                ),
                "edge_alignment": _summary(
                    float(item["edge_alignment"])
                    for item in all_details
                    if item["edge_alignment"] is not None
                ),
                "co_visible_coverage": _summary(
                    float(item["co_visible_coverage"]) for item in all_details
                ),
                "source_patch_coverage": _summary(float(item["source_patch_coverage"]) for item in all_details),
                "target_patch_coverage": _summary(float(item["target_patch_coverage"]) for item in all_details),
                "unique_patch_pair_count": sum(int(item["unique_patch_pair_count"]) for item in all_details),
            },
        },
        failure=(backend_failure if backend_failure is not None else
                 (None if all_details else "no co-visible depth correspondences with decoded RGB frames")),
        provenance={
            **dict(provenance),
            **comparator.provenance,
            "protocol": "short and medium baselines; bidirectional, schedule-first DINO reprojection",
            "maximum_pairs_per_group": max_pairs,
        },
    )


def _reprojection_group_summary(
    items: Iterable[ReprojectionCorrespondence],
    *,
    bundle: GeometryBundle,
    options: EvaluationOptions,
    frames: Mapping[int, np.ndarray],
    comparator: ImageComparator,
    metadata: Mapping[tuple[int, int], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    details: list[dict[str, Any]] = []
    for item in items:
        record: dict[str, Any] = {
            "source_frame_index": item.source_index,
            "target_frame_index": item.target_index,
            "correspondence_count": item.count,
            "covisible_coverage": item.coverage if item.available else None,
            **dict((metadata or {}).get((item.source_index, item.target_index), {})),
        }
        if (
            not item.count
            or item.source_index not in frames
            or item.target_index not in frames
        ):
            reasons = list(item.failure_reasons)
            if item.source_index not in frames:
                reasons.append("missing_source_rgb")
            if item.target_index not in frames:
                reasons.append("missing_target_rgb")
            record.update({"status": "not_covered", "failure_reasons": sorted(set(reasons))})
            details.append(record)
            continue
        value = comparator.compare_corresponding_pixels(
            frames[item.source_index],
            frames[item.target_index],
            source_rows=item.source_rows,
            source_columns=item.source_columns,
            target_rows=item.target_rows,
            target_columns=item.target_columns,
            source_cache_key=("frame", item.source_index),
            target_cache_key=("frame", item.target_index),
        )
        depth_abs_rel = float(np.mean(
            np.abs(item.projected_depth - item.target_depth)
            / np.maximum(item.target_depth, 1e-8)
        ))
        cloud = _completion_cloud_result(
            bundle, (item.source_index,), (item.target_index,), options,
        )
        record.update({
            "status": "ok",
            "dino_similarity": value["feature_reprojection_similarity"],
            "depth_abs_rel": depth_abs_rel,
            "local_pointcloud_fscore": cloud.get("fscore"),
            "covisible_coverage": item.coverage,
        })
        details.append(record)
    evaluated = [item for item in details if item.get("status") == "ok"]
    return {
        "status": "ok" if evaluated else "not_covered",
        "pair_count": len(details),
        "evaluated_pair_count": len(evaluated),
        "correspondence_count": sum(
            int(item["correspondence_count"]) for item in evaluated
        ),
        "dino_similarity": _summary(
            float(item["dino_similarity"]) for item in evaluated
        ),
        "depth_abs_rel": _summary(
            float(item["depth_abs_rel"]) for item in evaluated
        ),
        "local_pointcloud_fscore": _summary(
            float(item["local_pointcloud_fscore"])
            for item in evaluated
            if item["local_pointcloud_fscore"] is not None
        ),
        "covisible_coverage": _summary(
            float(item["covisible_coverage"]) for item in details
            if item.get("covisible_coverage") is not None
        ),
        "pairs": details,
    }


def completion_reprojection_metric(
    *,
    canonical: Iterable[ReprojectionCorrespondence],
    nearest_pose: Iterable[ReprojectionCorrespondence],
    nearest_metadata: Mapping[tuple[int, int], Mapping[str, Any]],
    expected_pair_count: int,
    pose_estimate_available: bool,
    bundle: GeometryBundle,
    options: EvaluationOptions,
    frames: Mapping[int, np.ndarray],
    comparator: ImageComparator,
    provenance: Mapping[str, Any],
) -> MetricResult:
    """Generated first/revisit persistence after estimator pose/depth reprojection."""

    canonical_items = tuple(canonical)
    nearest_items = tuple(nearest_pose)
    canonical_value = _reprojection_group_summary(
        canonical_items, bundle=bundle, options=options,
        frames=frames, comparator=comparator,
    )
    nearest_value = _reprojection_group_summary(
        nearest_items, bundle=bundle, options=options,
        frames=frames, comparator=comparator, metadata=nearest_metadata,
    )
    if not pose_estimate_available:
        nearest_value["status"] = "not_available"
    status = "ok" if any(
        value["status"] == "ok" for value in (canonical_value, nearest_value)
    ) else "not_covered"
    return MetricResult(
        name="completion_revisit",
        status=status,
        applicable=True,
        version="v0.2",
        coverage={
            "expected_pair_count": expected_pair_count,
            "canonical_pair_count": len(canonical_items),
            "canonical_evaluated_pair_count": canonical_value["evaluated_pair_count"],
            "nearest_pose_pair_count": len(nearest_items),
            "nearest_pose_evaluated_pair_count": nearest_value["evaluated_pair_count"],
            "pose_estimate_available": pose_estimate_available,
        },
        raw={
            "canonical": canonical_value,
            "nearest_pose": nearest_value,
        },
        failure=None if status == "ok" else "no usable completion reprojection correspondences",
        provenance={
            **dict(provenance),
            **comparator.provenance,
            "protocol": (
                "first-depth backprojection, recovered-pose target projection, "
                "target-depth co-visibility and z-buffer; no same-pixel SSIM"
            ),
            "completion_region_policy": (
                "all generated co-visible surfaces; schema-0.5 has no "
                "per-completion-frame unseen-region image mask"
            ),
        },
    )


def regional_scale_inconsistency_metric(
    correspondences: Mapping[str, Iterable[ReprojectionCorrespondence]], *,
    provenance: Mapping[str, Any], grid_size: int = 4,
) -> MetricResult:
    """Raw spatial depth-scale diagnostic, not a temporal-drift score."""

    records: list[dict[str, Any]] = []
    all_items: list[ReprojectionCorrespondence] = []
    for group, values in correspondences.items():
        for item in values:
            all_items.append(item)
            if not item.count:
                continue
            ratio = item.target_depth / np.maximum(item.projected_depth, 1e-8)
            local_ratios: list[float] = []
            if item.target_height is None or item.target_width is None:
                raise ValueError("regional scale diagnostic requires target depth dimensions")
            for row in range(grid_size):
                for column in range(grid_size):
                    mask = (
                        (item.target_rows >= row * item.target_height / grid_size)
                        & (item.target_rows < (row + 1) * item.target_height / grid_size)
                        & (item.target_columns >= column * item.target_width / grid_size)
                        & (item.target_columns < (column + 1) * item.target_width / grid_size)
                    )
                    if int(mask.sum()) >= 16:
                        local_ratios.append(float(np.median(ratio[mask])))
            local_logs = np.log(np.maximum(local_ratios, 1e-8))
            records.append({
                "group": group, "source_frame_index": item.source_index, "target_frame_index": item.target_index,
            "frame_delta": item.target_index - item.source_index,
                "correspondence_count": item.count,
                "local_cell_count": len(local_ratios),
                "regional_scale_inconsistency": (
                    float(np.median(np.abs(local_logs - np.median(local_logs))))
                    if len(local_ratios) >= 2 else None
                ),
            })
    evaluated = [item for item in records if item["regional_scale_inconsistency"] is not None]
    if not evaluated:
        return MetricResult(
            name="regional_scale_inconsistency", status="not_covered", applicable=True,
            coverage=_coverage_counts(all_items, evaluated=0),
            raw={"regional_scale_inconsistency": _summary([]), "local_grid_size": grid_size, "pairs": records},
            failure="fewer than two valid depth-grid cells in every usable pair",
            provenance={**dict(provenance), "role": "raw_diagnostic_only"},
        )
    return MetricResult(
        name="regional_scale_inconsistency", status="ok", applicable=True,
        coverage={**_coverage_counts(all_items, evaluated=len(evaluated)), "local_grid_size": grid_size},
        raw={
            "regional_scale_inconsistency": _summary(
                float(item["regional_scale_inconsistency"]) for item in evaluated
            ),
            "local_grid_size": grid_size,
            "pairs": records,
        },
        provenance={**dict(provenance), "role": "raw_diagnostic_only"},
    )


def _points_for_indices(bundle: GeometryBundle, indices: Iterable[int], *, frame_stride: int, pixel_stride: int, min_depth: float, max_depth: float) -> tuple[np.ndarray, float]:
    selected = sorted(set(int(index) for index in indices if index in bundle.poses_by_index and index in bundle.depths_by_index))
    selected = selected[::frame_stride]
    values: list[np.ndarray] = []
    depth_values: list[np.ndarray] = []
    for index in selected:
        depth = bundle.depths_by_index[index]
        K = bundle.camera_matrix(index)
        height, width = depth.shape
        rows, columns = np.mgrid[0:height:pixel_stride, 0:width:pixel_stride]
        z = depth[rows, columns]
        valid = np.isfinite(z) & (z >= min_depth) & (z <= max_depth)
        rows, columns, z = rows[valid], columns[valid], z[valid]
        if not len(z):
            continue
        camera = np.stack([
            (columns - K[0, 2]) * z / K[0, 0],
            (rows - K[1, 2]) * z / K[1, 1], z, np.ones_like(z),
        ], axis=1)
        values.append((bundle.poses_by_index[index] @ camera.T).T[:, :3])
        depth_values.append(z)
    if not values:
        return np.empty((0, 3), dtype=np.float64), float("nan")
    return np.concatenate(values), float(np.median(np.concatenate(depth_values)))


def _voxel_downsample(points: np.ndarray, voxel_size: float) -> np.ndarray:
    if not len(points):
        return points
    key = np.floor(points / voxel_size).astype(np.int64)
    _, indices = np.unique(key, axis=0, return_index=True)
    return points[np.sort(indices)]


def _cloud_overlap(left: np.ndarray, right: np.ndarray, threshold: float) -> dict[str, float | int | None]:
    if not len(left) or not len(right):
        return {"precision": None, "recall": None, "fscore": None, "trimmed_chamfer": None}
    left_distances = cKDTree(right).query(left, k=1)[0]
    right_distances = cKDTree(left).query(right, k=1)[0]
    precision, recall = float(np.mean(left_distances <= threshold)), float(np.mean(right_distances <= threshold))
    fscore = float(2 * precision * recall / max(precision + recall, 1e-12))
    return {
        "precision": precision, "recall": recall, "fscore": fscore,
        "trimmed_chamfer": float((np.mean(np.minimum(left_distances, np.percentile(left_distances, 95))) + np.mean(np.minimum(right_distances, np.percentile(right_distances, 95)))) / 2.0),
    }


def _completion_cloud_result(bundle: GeometryBundle, first: Iterable[int], revisit: Iterable[int], options: EvaluationOptions) -> dict[str, Any]:
    left, left_depth = _points_for_indices(bundle, first, frame_stride=options.geometry_pointcloud_frame_stride, pixel_stride=options.geometry_pointcloud_pixel_stride, min_depth=options.geometry_min_depth, max_depth=options.geometry_max_depth)
    right, right_depth = _points_for_indices(bundle, revisit, frame_stride=options.geometry_pointcloud_frame_stride, pixel_stride=options.geometry_pointcloud_pixel_stride, min_depth=options.geometry_min_depth, max_depth=options.geometry_max_depth)
    median_depth = float(np.nanmedian([left_depth, right_depth]))
    threshold = max(options.geometry_pointcloud_min_threshold, options.geometry_pointcloud_relative_threshold * median_depth) if np.isfinite(median_depth) else options.geometry_pointcloud_min_threshold
    left, right = _voxel_downsample(left, threshold / 2.0), _voxel_downsample(right, threshold / 2.0)
    return {
        "first_point_count": int(len(left)), "revisit_point_count": int(len(right)),
        "distance_threshold": threshold, **_cloud_overlap(left, right, threshold),
    }
