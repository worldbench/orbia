"""Evaluate one generated video against one benchmark case."""

from __future__ import annotations

from src.metrics.sampling import geometry_sampling_plan

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable, Mapping

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation, Slerp

from src.metrics.case import EvaluationCase, load_evaluation_case
from src.metrics.da3 import (
    DA3Adapter,
    da3_artifact_complete,
    context_heldout_indices,
    prepare_da3_request,
    run_da3_metric,
    score_da3_artifact,
)
from src.metrics.contract import EvaluationOptions, EvaluationPaths, MetricResult
from src.metrics.features import ImageComparator
from src.metrics.geometry import (
    build_correspondences,
    completion_reprojection_metric,
    covisible_reprojection_metric,
    depth_temporal_consistency_metric,
    infer_depth_root,
    load_geometry_bundle,
    regional_scale_inconsistency_metric,
    pair_schedule,
)
from src.metrics.io import write_json
from src.metrics.pose import (
    PoseCache,
    align_pose_track_umeyama,
    fit_positive_scale,
    load_pose_cache,
    nearest_cached_pose,
    rotation_error_deg,
    scale_pose_translations,
    pose_track_error_summary,
)
from src.metrics.reconstruction import list_vipe_depth_indices
from src.metrics.registry import (
    DEFAULT_METRICS,
    KNOWN_METRICS,
    TIER5_METRICS,
)
from src.metrics.tier2 import (
    aesthetic_quality_metric,
    imaging_quality_metric,
    motion_smoothness_metric,
)
from src.metrics.anchor_identity import sample_indices, anchor_identity_metric
from src.metrics.anchor_tracking import tracking_windows, anchor_trajectory_metric
from src.metrics.tier6 import long_horizon_metric, reused_temporal_diagnostics
from src.metrics.transforms import (
    canonical_to_output_transform,
    depth_mask_to_native_rgb,
    depth_to_native_rgb,
    qe_reference_and_mask_in_output,
    warp_image,
)
from src.metrics.video import VideoInfo, inspect_video, read_video_frames


@dataclass(frozen=True)
class AlignedTrack:
    cache: PoseCache
    poses: np.ndarray
    executed_relative_unscaled: np.ndarray
    scale: float
    requested_path_scale: float
    requested_reference_pose: np.ndarray


@dataclass(frozen=True)
class Tier0Inspection:
    """Coverage-only inputs collected once for fast and full evaluation."""

    info: VideoInfo | None
    cache: PoseCache | None
    depth_root: Path | None
    validity: MetricResult


INCOMPLETE_METRIC_STATUSES = frozenset(
    {"not_available", "invalid_input", "evaluator_failed"}
)


def _multi_revisit_metric(
    *,
    canonical_by_pair: Mapping[str, Iterable[Any]],
    revisit_pairs: Mapping[str, Mapping[str, Any]],
    nearest_pose: Iterable[Any],
    nearest_metadata: Mapping[tuple[int, int], Mapping[str, Any]],
    pose_estimate_available: bool,
    bundle: Any,
    options: EvaluationOptions,
    frames: Mapping[int, np.ndarray],
    comparator: ImageComparator,
    provenance: Mapping[str, Any],
    exact_frame_contract: bool = False,
) -> MetricResult:
    """Revisit metrics per registered pair; long cases average their pairs equally."""
    results = {}
    multiple = exact_frame_contract or len(revisit_pairs) > 1
    for name, pair in revisit_pairs.items():
        results[name] = completion_reprojection_metric(
            canonical=canonical_by_pair.get(name, ()),
            nearest_pose=() if multiple else nearest_pose,
            nearest_metadata={} if multiple else nearest_metadata,
            expected_pair_count=len(pair["first_window_indices"]),
            pose_estimate_available=pose_estimate_available,
            bundle=bundle, options=options, frames=frames,
            comparator=comparator, provenance=provenance,
        )
    primary = results["C"]
    if not multiple:
        return primary
    groups = [result.raw["canonical"] for result in results.values()]
    complete = all(group["status"] == "ok" for group in groups)
    aggregate = {
        "status": "ok" if complete else "not_covered",
        "pair_count": sum(group["pair_count"] for group in groups),
        "evaluated_pair_count": sum(group["evaluated_pair_count"] for group in groups),
        "correspondence_count": sum(group["correspondence_count"] for group in groups),
    }
    for field in ("dino_similarity", "depth_abs_rel", "local_pointcloud_fscore", "covisible_coverage"):
        values = [group[field]["mean"] for group in groups
                  if group[field].get("mean") is not None]
        aggregate[field] = {
            "mean": float(np.mean(values)) if len(values) == len(groups) else None,
            "conditional_mean": float(np.mean(values)) if values else None,
            "count": len(values), "expected_count": len(groups),
        }
    status = next((result.status for result in results.values()
                   if result.status in INCOMPLETE_METRIC_STATUSES), aggregate["status"])
    return replace(
        primary, version="v0.3", status=status,
        coverage={
            "expected_pair_count": sum(len(pair["first_window_indices"]) for pair in revisit_pairs.values()),
            "canonical_pair_count": aggregate["pair_count"],
            "canonical_evaluated_pair_count": aggregate["evaluated_pair_count"],
            "registered_revisit_pair_count": len(groups),
            "evaluated_revisit_pair_count": sum(group["status"] == "ok" for group in groups),
            "registered_revisit_pairs": sorted(revisit_pairs),
            "pose_estimate_available": pose_estimate_available,
        },
        raw={
            "canonical": aggregate,
            "nearest_pose": {"status": "not_applicable"},
            "revisit_pairs": {
                name: {"status": result.status, "coverage": result.coverage, "raw": result.raw,
                       "first_event": revisit_pairs[name]["first_event"],
                       "revisit_event": revisit_pairs[name]["revisit_event"]}
                for name, result in results.items()
            },
        },
        failure=None if status == "ok" else "registered revisit measurements are incomplete",
        provenance={**primary.provenance,
                    "multi_revisit_protocol": "equal-weight registered exact-frame pairs; incomplete means withheld"},
    )


def evaluation_run_status(results: Mapping[str, MetricResult]) -> str:
    """Summarise whether every requested evaluator output is publishable."""

    return (
        "incomplete"
        if any(result.status in INCOMPLETE_METRIC_STATUSES for result in results.values())
        else "complete"
    )


def _interpolate_poses(timestamps: np.ndarray, poses: np.ndarray, query: np.ndarray) -> np.ndarray:
    if np.any(query < timestamps[0] - 1e-9) or np.any(query > timestamps[-1] + 1e-9):
        raise ValueError("query timestamps outside requested trajectory")
    translation = np.stack([np.interp(query, timestamps, poses[:, axis, 3]) for axis in range(3)], axis=1)
    rotations = Slerp(timestamps, Rotation.from_matrix(poses[:, :3, :3]))(query).as_matrix()
    result = np.repeat(np.eye(4, dtype=np.float64)[None], len(query), axis=0)
    result[:, :3, :3] = rotations
    result[:, :3, 3] = translation
    return result


def _requested_path_scale(case: EvaluationCase) -> float:
    increments = np.linalg.norm(np.diff(case.requested_poses[:, :3, 3], axis=0), axis=1)
    return max(float(increments.sum()), 1e-6)


def _align_track(case: EvaluationCase, info: VideoInfo, cache: PoseCache | None, control_stride: int = 1) -> AlignedTrack | None:
    if cache is None:
        return None
    cache_timestamps = cache.frame_indices.astype(np.float64) / info.fps
    valid = (cache_timestamps >= case.timestamps_s[0] - 1e-9) & (cache_timestamps <= case.timestamps_s[-1] + 1e-9)
    if valid.sum() < 2:
        return None
    cache_subset = cache.subset(valid)
    q0_video_index = _event_video_index(case, info, "q0")
    q0_matches = np.flatnonzero(cache_subset.frame_indices == q0_video_index)
    if q0_video_index is None or len(q0_matches) != 1:
        # Tier 0 completeness is the normal prerequisite.  This guard prevents
        # a direct Tier-1 call from silently re-anchoring to a later frame.
        return None
    q0_pose = cache_subset.poses_c2w[int(q0_matches[0])]
    executed_relative = np.linalg.inv(q0_pose)[None] @ cache_subset.poses_c2w
    expected = _interpolate_poses(case.timestamps_s, case.requested_poses, cache_subset.frame_indices.astype(np.float64) / info.fps)
    requested_reference_pose = case.requested_poses[case.events["q0"]]
    expected_relative = np.linalg.inv(requested_reference_pose)[None] @ expected
    from src.metrics.sampling import primary_pose_selection
    primary = primary_pose_selection(cache_subset.frame_indices, control_stride)
    if primary.sum() < 2:
        return None
    scale = fit_positive_scale(executed_relative[primary, :3, 3], expected_relative[primary, :3, 3])
    return AlignedTrack(
        cache=PoseCache(
            scale_pose_translations(executed_relative, scale),
            cache_subset.frame_indices,
            cache_subset.provenance,
            cache_subset.intrinsics,
        ),
        poses=expected_relative,
        executed_relative_unscaled=executed_relative,
        scale=scale,
        requested_path_scale=_requested_path_scale(case),
        requested_reference_pose=requested_reference_pose,
    )


def _event_video_index(case: EvaluationCase, info: VideoInfo, event_name: str) -> int | None:
    return info.timestamp_to_video_index(float(case.timestamps_s[case.events[event_name]]))


def _canonical_video_index(case: EvaluationCase, info: VideoInfo, canonical_index: int) -> int | None:
    return info.timestamp_to_video_index(float(case.timestamps_s[canonical_index]))


def _window_video_indices(case: EvaluationCase, info: VideoInfo, center: int, radius: int) -> list[int]:
    # Only the registered event frame is evaluated; ``radius`` is unused.
    index = _canonical_video_index(case, info, center)
    return [] if index is None else [index]


def _important_node_record(
    *,
    case: EvaluationCase,
    info: VideoInfo,
    track: AlignedTrack,
    event_name: str,
    candidate_video_indices: Iterable[int],
    rotation_cost_weight: float,
) -> dict[str, Any]:
    """Return the closest recovered pose in one event window."""

    candidates = sorted(set(int(index) for index in candidate_video_indices))
    available = set(int(index) for index in track.cache.frame_indices)
    candidate_pose_count = len(available.intersection(candidates))
    target = (
        np.linalg.inv(track.requested_reference_pose)
        @ case.requested_poses[case.events[event_name]]
    )
    nearest = nearest_cached_pose(
        track.cache,
        target_pose=target,
        candidate_video_indices=candidates,
        path_scale=track.requested_path_scale,
        rotation_cost_weight=rotation_cost_weight,
    )
    if nearest is None:
        return {
            "nearest_frame_index": None,
            "translation_error_normalized": None,
            "rotation_error_deg": None,
            "candidate_pose_count": candidate_pose_count,
        }
    index, translation, rotation = nearest
    return {
        "nearest_frame_index": index,
        "translation_error_normalized": translation,
        "rotation_error_deg": rotation,
        "candidate_pose_count": candidate_pose_count,
    }


def indexed_relative_pose_errors(indices, estimated, requested, deltas=(1, 8, 32)):
    lookup = {int(i): k for k,i in enumerate(indices)}
    result = {}
    for delta in deltas:
        errors = []
        for i,k in lookup.items():
            if i+delta not in lookup:
                continue
            j = lookup[i+delta]
            a = np.linalg.inv(estimated[k]) @ estimated[j]
            b = np.linalg.inv(requested[k]) @ requested[j]
            errors.append((float(np.linalg.norm(a[:3,3]-b[:3,3])), rotation_error_deg(a,b)))
        result[str(delta)] = {'pair_count':len(errors), 'translation_mean':float(np.mean([e[0] for e in errors])) if errors else None,
                              'rotation_mean_deg':float(np.mean([e[1] for e in errors])) if errors else None}
    return result


def _control_metric(
    case: EvaluationCase,
    info: VideoInfo,
    track: AlignedTrack | None,
    options: EvaluationOptions | None = None,
) -> MetricResult:
    options = options or EvaluationOptions()
    if track is None:
        return MetricResult(
            name="control", status="not_available", applicable=True,
            failure="no usable estimator pose cache overlaps the generated video",
        )
    translation_errors = np.linalg.norm(track.cache.poses_c2w[:, :3, 3] - track.poses[:, :3, 3], axis=1) / track.requested_path_scale
    rotation_errors = np.asarray([
        rotation_error_deg(left, right) for left, right in zip(track.cache.poses_c2w, track.poses)
    ])
    from src.metrics.sampling import primary_pose_selection
    primary = primary_pose_selection(track.cache.frame_indices, options.control_frame_stride)
    required = set(range(0, min(info.frame_count, int(np.floor(case.timestamps_s[-1]*info.fps))+1), options.control_frame_stride))
    if not required.issubset(set(map(int,track.cache.frame_indices))):
        raise ValueError('pose cache lacks the required uniform control timestamps')
    if primary.sum() < 2:
        raise ValueError('fewer than two primary control samples')
    per_pose_errors = [dict(frame_index=int(i), translation_error_normalized=float(t), rotation_error_deg=float(r))
                       for i,t,r in zip(track.cache.frame_indices, translation_errors, rotation_errors)]
    translation_errors = translation_errors[primary]
    rotation_errors = rotation_errors[primary]
    indexed = {int(index): (pose, expected) for index, pose, expected in zip(track.cache.frame_indices, track.cache.poses_c2w, track.poses)}
    events: dict[str, Any] = {}
    covered = 0
    for name, canonical_index in case.events.items():
        video_index = _canonical_video_index(case, info, canonical_index)
        if video_index is not None and video_index in indexed:
            pose, expected = indexed[video_index]
            translation = float(np.linalg.norm(pose[:3, 3] - expected[:3, 3]) / track.requested_path_scale)
            rotation = rotation_error_deg(pose, expected)
            events[name] = {"video_frame_index": video_index, "covered": True, "translation_error_normalized": translation, "rotation_error_deg": rotation}
            covered += 1
        else:
            events[name] = {"video_frame_index": video_index, "covered": False}
    anchored_ate_m = float(
        np.sqrt(
            np.mean(
                np.linalg.norm(
                    track.cache.poses_c2w[:, :3, 3] - track.poses[:, :3, 3],
                    axis=1,
                )
                ** 2
            )
        )
    )
    se3_track, se3_alignment = align_pose_track_umeyama(
        track.cache.poses_c2w, track.poses, with_scale=False
    )
    sim3_track, sim3_alignment = align_pose_track_umeyama(
        track.executed_relative_unscaled, track.poses, with_scale=True
    )
    first_window = [
        index
        for canonical in case.completion_pairs["first_window_indices"]
        if (index := _canonical_video_index(case, info, int(canonical))) is not None
    ]
    revisit_window = [
        index
        for canonical in case.completion_pairs["revisit_window_indices"]
        if (index := _canonical_video_index(case, info, int(canonical))) is not None
    ]
    important_nodes = {
        "QE": _important_node_record(
            case=case,
            info=info,
            track=track,
            event_name="qe1",
            candidate_video_indices=_window_video_indices(
                case, info, case.events["qe1"], options.qe_neighborhood_frames
            ),
            rotation_cost_weight=options.pose_rotation_cost_weight,
        ),
        "C1": _important_node_record(
            case=case,
            info=info,
            track=track,
            event_name="qc_first",
            candidate_video_indices=first_window,
            rotation_cost_weight=options.pose_rotation_cost_weight,
        ),
        "C2": _important_node_record(
            case=case,
            info=info,
            track=track,
            event_name="qc_revisit",
            candidate_video_indices=revisit_window,
            rotation_cost_weight=options.pose_rotation_cost_weight,
        ),
        "Return": _important_node_record(
            case=case,
            info=info,
            track=track,
            event_name="q0_return",
            candidate_video_indices=_window_video_indices(
                case,
                info,
                case.events["q0_return"],
                options.return_neighborhood_frames,
            ),
            rotation_cost_weight=options.pose_rotation_cost_weight,
        ),
    }
    raw = {
        "translation_mean_normalized": float(translation_errors.mean()),
        "rotation_mean_deg": float(rotation_errors.mean()),
        "important_nodes": important_nodes,
        "primary_frame_indices": track.cache.frame_indices[primary].tolist(),
        "per_pose_errors": per_pose_errors,
        "diagnostics": {
            "translation_rmse_normalized": float(
                np.sqrt(np.mean(translation_errors**2))
            ),
            "ate_first_frame_scale_aligned_m": anchored_ate_m,
            "ate_first_frame_scale_aligned_normalized": float(
                anchored_ate_m / track.requested_path_scale
            ),
            "ate_umeyama_se3_after_scale": {
                **se3_alignment,
                **pose_track_error_summary(se3_track, track.poses),
                "metric_scale_policy": "one global scale was fitted before SE3 alignment",
            },
            "ate_umeyama_sim3": {
                **sim3_alignment,
                **pose_track_error_summary(sim3_track, track.poses),
                "metric_scale_policy": "Sim3 maps recovered camera centres into requested metres",
            },
            "rpe_first_frame_scale_aligned": indexed_relative_pose_errors(track.cache.frame_indices[primary], track.cache.poses_c2w[primary], track.poses[primary], deltas=tuple(d*options.control_frame_stride for d in (1,8,32))),
            "rotation_p95_deg": float(np.percentile(rotation_errors, 95)),
            "path_progress_ratio": float(
                np.linalg.norm(
                    np.diff(track.cache.poses_c2w[:, :3, 3], axis=0), axis=1
                ).sum()
                / max(
                    np.linalg.norm(
                        np.diff(track.poses[:, :3, 3], axis=0), axis=1
                    ).sum(),
                    1e-6,
                )
            ),
            "path_length_ratio_before_scale": float(
                np.linalg.norm(
                    np.diff(
                        track.executed_relative_unscaled[:, :3, 3], axis=0
                    ),
                    axis=1,
                ).sum()
                / max(
                    np.linalg.norm(
                        np.diff(track.poses[:, :3, 3], axis=0), axis=1
                    ).sum(),
                    1e-6,
                )
            ),
            "global_translation_scale": track.scale,
            "events": events,
        },
    }
    important_node_coverage = float(
        sum(node["candidate_pose_count"] > 0 for node in important_nodes.values())
        / len(important_nodes)
    )
    coverage = {
        "pose_usable_ratio": float(len(track.cache.poses_c2w) / info.frame_count),
        "event_pose_coverage": float(covered / len(case.events)),
        "key_node_coverage": float(covered / len(case.events)),
        "important_node_pose_coverage": important_node_coverage,
    }
    return MetricResult(
        name="control",
        status="ok",
        applicable=True,
        version="v0.2",
        coverage=coverage,
        raw=raw,
        provenance={
            **track.cache.provenance,
            "alignment_protocols": {
                "primary": "Q0-relative plus one non-negative global translation scale; no free rotation",
                "se3_diagnostic": "Umeyama SE3 after the primary global scale",
                "sim3_diagnostic": "Umeyama Sim3 from the unscaled recovered relative track",
                "rpe": "original video frame deltas on the primary aligned track; no per-pair fit",
            },
        },
    )


def _nearest_event(
    *,
    case: EvaluationCase,
    info: VideoInfo,
    track: AlignedTrack | None,
    event_name: str,
    radius: int,
    options: EvaluationOptions,
) -> tuple[int | None, dict[str, Any] | None]:
    if track is None:
        return None, None
    candidates = _window_video_indices(case, info, case.events[event_name], radius)
    # Both recovered and requested tracks are expressed relative to Q0.
    target = (
        np.linalg.inv(track.requested_reference_pose)
        @ case.requested_poses[case.events[event_name]]
    )
    result = nearest_cached_pose(
        track.cache, target_pose=target, candidate_video_indices=candidates,
        path_scale=track.requested_path_scale, rotation_cost_weight=options.pose_rotation_cost_weight,
    )
    if result is None:
        return None, None
    index, translation, rotation = result
    canonical_index = case.events[event_name]
    event_index = _event_video_index(case, info, event_name)
    return index, {
        "status": "ok", "frame_index": index, "canonical_event_video_frame": event_index,
        "time_offset_s": float(index / info.fps - case.timestamps_s[canonical_index]),
        "translation_error_normalized": translation, "rotation_error_deg": rotation,
        "candidate_count": len(candidates),
    }


def _event_frame_metadata(
    *,
    case: EvaluationCase,
    track: AlignedTrack | None,
    event_name: str,
    frame_index: int | None,
    options: EvaluationOptions,
) -> dict[str, Any]:
    if frame_index is None:
        return {"status": "not_covered", "frame_index": None}
    if track is None:
        return {"status": "not_available", "frame_index": frame_index}
    target = (
        np.linalg.inv(track.requested_reference_pose)
        @ case.requested_poses[case.events[event_name]]
    )
    match = nearest_cached_pose(
        track.cache,
        target_pose=target,
        candidate_video_indices=(frame_index,),
        path_scale=track.requested_path_scale,
        rotation_cost_weight=options.pose_rotation_cost_weight,
    )
    if match is None:
        return {"status": "not_available", "frame_index": frame_index}
    _, translation, rotation = match
    return {
        "status": "ok",
        "frame_index": frame_index,
        "translation_error_normalized": translation,
        "rotation_error_deg": rotation,
    }


def _warp_masks(
    masks: Mapping[str, np.ndarray],
    transform: np.ndarray,
    width: int,
    height: int,
) -> dict[str, np.ndarray]:
    return {
        name: warp_image(mask.astype(np.uint8), transform, width=width, height=height, mask=True)
        for name, mask in masks.items()
    }


def _write_review_pair(path: Path, left: np.ndarray, right: np.ndarray, *, title: str) -> None:
    from PIL import ImageDraw
    height, width = left.shape[:2]
    canvas = Image.new("RGB", (width * 2, height + 28), "white")
    canvas.paste(Image.fromarray(left), (0, 28))
    canvas.paste(Image.fromarray(right), (width, 28))
    ImageDraw.Draw(canvas).text((6, 6), title, fill="black")
    canvas.save(path)


def _metric_path(output_root: Path, name: str) -> Path:
    return output_root / "metrics" / f"{name}.json"


def _inspect_tier0(
    paths: EvaluationPaths,
    *,
    expected_frame_count: int,
    video_info: VideoInfo | None = None,
) -> Tier0Inspection:
    """Check that recovered poses and depths cover the expected frame indices."""

    video_error: str | None = None
    if video_info is not None:
        if video_info.path.resolve() != paths.video_path.resolve():
            raise ValueError("preflight VideoInfo path differs from evaluator video")
        info = video_info
    else:
        try:
            info = inspect_video(paths.video_path)
        except Exception as exc:
            info = None
            video_error = f"{type(exc).__name__}: {exc}"
    decoded_frame_count = info.frame_count if info is not None else 0
    estimator_frame_count = expected_frame_count

    pose_error: str | None = None
    cache: PoseCache | None = None
    if paths.pose_cache is not None:
        try:
            cache = load_pose_cache(
                paths.pose_cache, video_frame_count=estimator_frame_count
            )
        except Exception as exc:
            pose_error = f"{type(exc).__name__}: {exc}"
    pose_indices = set(int(index) for index in cache.frame_indices) if cache else set()
    expected_estimator_indices = set(range(estimator_frame_count))
    if cache is not None and cache.provenance.get('sampling_plan'):
        declared = cache.provenance['sampling_plan']
        indices = declared['frame_indices']
        if (declared.get('video_frame_count') != estimator_frame_count or not indices
                or any(type(i) is not int or not 0 <= i < estimator_frame_count for i in indices)
                or indices != sorted(set(indices)) or indices[0] != 0):
            raise ValueError('invalid geometry sampling plan')
        expected_estimator_indices = set(indices)
    pose_in_range = pose_indices.intersection(expected_estimator_indices)

    resolved_depth_root = infer_depth_root(paths.pose_cache, paths.depth_root)
    depth_error: str | None = None
    depth_indices: set[int] = set()
    if resolved_depth_root is not None:
        try:
            depth_indices = set(list_vipe_depth_indices(resolved_depth_root))
        except Exception as exc:
            depth_error = f"{type(exc).__name__}: {exc}"
    depth_in_range = depth_indices.intersection(expected_estimator_indices)

    video_complete = bool(
        info is not None
        and decoded_frame_count == expected_frame_count
        and np.isclose(info.decoded_frame_ratio, 1.0)
    )
    pose_complete = bool(
        paths.pose_cache is not None
        and pose_error is None
        and pose_indices == expected_estimator_indices
    )
    depth_complete = bool(
        resolved_depth_root is not None
        and depth_error is None
        and depth_indices == expected_estimator_indices
    )
    denominator = max(len(expected_estimator_indices), 1)
    validity = MetricResult(
        name="validity",
        status="ok",
        applicable=True,
        version="v0.3",
        coverage={
            "video_decode_ratio": float(
                min(decoded_frame_count, expected_frame_count)
                / max(expected_frame_count, 1)
            ),
            "pose_output_ratio": float(len(pose_in_range) / denominator),
            "depth_output_ratio": float(len(depth_in_range) / denominator),
        },
        raw={
            "evaluator_status": "ok",
            "expected_frame_count": expected_frame_count,
            "expected_geometry_frame_count": len(expected_estimator_indices),
            "estimator_complete": bool(
                video_complete and pose_complete and depth_complete
            ),
            "video": {
                "complete": video_complete,
                "decoded_frame_count": decoded_frame_count,
                "fps": info.fps if info is not None else None,
                "width": info.width if info is not None else None,
                "height": info.height if info is not None else None,
                "container_decode_ratio": (
                    info.decoded_frame_ratio if info is not None else 0.0
                ),
                "error": video_error,
            },
            "pose": {
                "provided": paths.pose_cache is not None,
                "complete": pose_complete,
                "output_frame_count": len(pose_in_range),
                "out_of_range_frame_count": len(pose_indices - pose_in_range),
                "error": pose_error,
            },
            "depth": {
                "provided": resolved_depth_root is not None,
                "complete": depth_complete,
                "output_frame_count": len(depth_in_range),
                "out_of_range_frame_count": len(depth_indices - depth_in_range),
                "error": depth_error,
            },
        },
        provenance={
            "case_root": str(paths.case_root),
            "video_path": str(paths.video_path),
            "pose_cache": str(paths.pose_cache) if paths.pose_cache else None,
            "depth_root": str(resolved_depth_root) if resolved_depth_root else None,
            "depth_policy": "artifact frame-index scan only; no depth pixel decode",
            "quality_policy": "coverage only; pose/depth accuracy is not evaluated",
        },
    )
    return Tier0Inspection(info, cache, resolved_depth_root, validity)


def _fast_validity_report(paths: EvaluationPaths) -> dict[str, Any]:
    """Run the Tier-0 artifact coverage check without downstream evaluators."""

    from src.metrics.precompute import load_precompute_case
    case = load_precompute_case(paths.case_root)
    case_id = case.case_id
    expected_frame_count = case.frame_count

    inspection = _inspect_tier0(paths, expected_frame_count=expected_frame_count)
    info, validity = inspection.info, inspection.validity
    decoded_frame_count = info.frame_count if info is not None else 0
    paths.output_root.mkdir(parents=True, exist_ok=True)
    write_json(_metric_path(paths.output_root, "validity"), validity.to_dict())
    run_status = evaluation_run_status({"validity": validity})
    report = {
        "metric_schema_version": "0.2.0",
        "status": run_status,
        "case_id": case_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "video": {
            "path": str(paths.video_path),
            "frame_count": decoded_frame_count,
            "fps": info.fps if info is not None else None,
            "resolution": [info.width, info.height] if info is not None else None,
        },
        "pose_cache": str(paths.pose_cache) if paths.pose_cache else None,
        "metrics": {"validity": validity.to_dict()},
    }
    write_json(paths.output_root / "report.json", report)
    write_json(
        paths.output_root / "run.json",
        {
            "status": run_status,
            "case_root": str(paths.case_root),
            "case_id": case_id,
            "evaluation_profile": "tier0_fast",
        },
    )
    return report


def prepare_da3_case(
    *,
    case_root: Path | str,
    video_path: Path | str,
    output_root: Path | str,
    pose_cache: Path | str,
    config: Mapping[str, Any] | None = None,
    video_info: VideoInfo | None = None,
    decoded_frames: Mapping[int, np.ndarray] | None = None,
    context_images: dict[int, np.ndarray] | None = None,
    preloaded_case=None,
) -> Path:
    """Prepare one DA3-GS request from the active geometry cache."""

    options = EvaluationOptions.from_mapping(config)
    case = preloaded_case if preloaded_case is not None else load_evaluation_case(Path(case_root).resolve())
    info = video_info or inspect_video(video_path)
    cache = load_pose_cache(
        Path(pose_cache).resolve(), video_frame_count=case.frame_count
    )
    context, heldout = context_heldout_indices(
        info.frame_count,
        context_count=options.da3_context_views,
        heldout_count=options.da3_heldout_views,
        candidate_indices=geometry_sampling_plan(case, info, options)["frame_indices"],
    )
    selected = (*context, *heldout)
    frames = {index: decoded_frames[index] for index in selected} if decoded_frames is not None else read_video_frames(info, selected)
    poses = {
        int(index): pose
        for pose, index in zip(cache.poses_c2w, cache.frame_indices)
    }
    missing_pose = sorted(set(selected) - set(poses))
    if missing_pose:
        raise ValueError(
            f"geometry pose cache lacks DA3 frames: {missing_pose}"
        )
    transform = canonical_to_output_transform(
        canonical_width=case.canonical_camera.width,
        canonical_height=case.canonical_camera.height,
        output_width=info.width,
        output_height=info.height,
    )
    output_k = transform @ case.canonical_camera.K
    cached_intrinsics = (
        {}
        if cache.intrinsics is None
        else {
            int(index): intrinsic
            for intrinsic, index in zip(cache.intrinsics, cache.frame_indices)
        }
    )
    return prepare_da3_request(
        frames=frames,
        context_images=context_images,
        poses_c2w=poses,
        intrinsics={
            index: cached_intrinsics.get(index, output_k)
            for index in selected
        },
        frame_count=info.frame_count,
        output_root=Path(output_root).resolve() / "da3",
        context_count=options.da3_context_views,
        heldout_count=options.da3_heldout_views,
        candidate_indices=geometry_sampling_plan(case, info, options)["frame_indices"],
        resolution=options.da3_input_resolution,
    )


def evaluate_case(
    *,
    case_root: Path | str,
    video_path: Path | str,
    output_root: Path | str,
    pose_cache: Path | str | None = None,
    depth_root: Path | str | None = None,
    config: Mapping[str, Any] | None = None,
    metrics: Iterable[str] | None = None,
    force: bool = False,
    image_comparator: ImageComparator | None = None,
    require_precomputed_da3: bool = False,
    precomputed_da3_root: Path | str | None = None,
    video_info: VideoInfo | None = None,
) -> dict[str, Any]:
    """Evaluate one generated video without modifying the immutable case."""

    del force  # selected metric files are always rewritten

    evaluation_started = perf_counter()
    stage_timings = {}
    options = EvaluationOptions.from_mapping(config)
    paths = EvaluationPaths(
        Path(case_root).resolve(), Path(video_path).resolve(), Path(output_root).resolve(),
        Path(pose_cache).resolve() if pose_cache else None,
        Path(depth_root).resolve() if depth_root else None,
    )
    selected_metrics = tuple(metrics or DEFAULT_METRICS)
    unknown = sorted(set(selected_metrics) - KNOWN_METRICS)
    if unknown:
        raise ValueError(f"unknown evaluator metrics: {unknown}")
    if selected_metrics == ("validity",):
        return _fast_validity_report(paths)
    case = load_evaluation_case(paths.case_root)
    inspection = _inspect_tier0(
        paths,
        expected_frame_count=case.frame_count,
        video_info=video_info,
    )
    info = inspection.info
    if info is None:
        # The coverage-only fast path records decode errors without raising.
        # Downstream evaluators cannot operate without decoded video metadata.
        raise ValueError(inspection.validity.raw["video"]["error"])
    cache = inspection.cache
    resolved_depth_root = inspection.depth_root
    paths.output_root.mkdir(parents=True, exist_ok=True)
    write_json(paths.output_root / "cache" / "video.json", {
        "path": str(info.path), "frame_count": info.frame_count,
        "fps": info.fps, "width": info.width, "height": info.height, "decoded_frame_ratio": info.decoded_frame_ratio,
    })
    track = _align_track(case, info, cache, options.control_frame_stride)
    geometry_metric_names = set(TIER5_METRICS)
    output_transform = canonical_to_output_transform(
        canonical_width=case.canonical_camera.width, canonical_height=case.canonical_camera.height,
        output_width=info.width, output_height=info.height,
    )
    q0_output = warp_image(case.q0_rgb, output_transform, width=info.width, height=info.height)
    qe_reference, qe_mask = qe_reference_and_mask_in_output(
        qe_rgb_native=case.qe.rgb, input_supported_depth=case.observability_input_supported_depth,
        qe_depth_camera=case.qe.depth_camera, qe_rgb_camera=case.qe.rgb_camera,
        native_to_canonical=case.native_to_canonical, canonical_to_output=output_transform,
        output_width=info.width, output_height=info.height,
    )
    native_to_output = output_transform @ case.native_to_canonical
    q0_anchor_masks = _warp_masks(
        case.q0_anchor_masks_rgb, native_to_output, info.width, info.height,
    )
    qe_anchor_masks = _warp_masks(
        case.qe_anchor_masks_rgb, native_to_output, info.width, info.height,
    )
    qe_invalid_native = depth_mask_to_native_rgb(
        case.observability_invalid_depth,
        depth_camera=case.qe.depth_camera,
        rgb_camera=case.qe.rgb_camera,
    )
    qe_invalid = warp_image(
        qe_invalid_native.astype(np.uint8), native_to_output,
        width=info.width, height=info.height, mask=True,
    )
    q0_depth_native, q0_depth_valid_native = depth_to_native_rgb(
        case.q0.depth_m, case.q0.depth_valid,
        depth_camera=case.q0.depth_camera, rgb_camera=case.q0.rgb_camera,
    )
    qe_depth_native, qe_depth_valid_native = depth_to_native_rgb(
        case.qe.depth_m, case.qe.depth_valid,
        depth_camera=case.qe.depth_camera, rgb_camera=case.qe.rgb_camera,
    )
    q0_depth = warp_image(
        q0_depth_native, native_to_output, width=info.width, height=info.height,
        nearest=True,
    )
    q0_valid = warp_image(
        q0_depth_valid_native.astype(np.uint8), native_to_output,
        width=info.width, height=info.height, mask=True,
    )
    qe_depth = warp_image(
        qe_depth_native, native_to_output, width=info.width, height=info.height,
        nearest=True,
    )
    qe_depth_valid = warp_image(
        qe_depth_valid_native.astype(np.uint8), native_to_output,
        width=info.width, height=info.height, mask=True,
    )
    qe_mask &= qe_depth_valid
    stage_timings["reference_and_pose_prepare_s"] = perf_counter() - evaluation_started
    geometry_started = perf_counter()
    geometry_bundle = None
    geometry_correspondences: dict[str, tuple[Any, ...]] = {}
    geometry_failure: str | None = None
    if set(selected_metrics) & (
        geometry_metric_names | {"completion_revisit", "q0_return", "evidence_query", "anchor_trajectory", "long_horizon"}
    ):
        if cache is None:
            geometry_failure = "Tier-3 metrics require a usable estimator pose cache"
        elif resolved_depth_root is None:
            geometry_failure = "Tier-3 metrics require --depth-root or a sibling depth/ beside --pose-cache"
        else:
            try:
                geometry_bundle = load_geometry_bundle(case=case, info=info, cache=cache, depth_root=resolved_depth_root)
                if geometry_bundle is None:
                    geometry_failure = "Tier-3 estimator depth/pose bundle is unavailable"
                else:
                    geometry_correspondences = build_correspondences(
                        geometry_bundle, pair_schedule(geometry_bundle, options), options,
                    )
            except Exception as exc:
                geometry_failure = f"{type(exc).__name__}: {exc}"
    stage_timings["geometry_load_and_correspondences_s"] = perf_counter() - geometry_started
    frame_prepare_started = perf_counter()
    requested_indices: set[int] = {0}
    da3_sampling_failure: str | None = None
    da3_root = Path(precomputed_da3_root) if precomputed_da3_root is not None else paths.output_root / "da3"
    has_precomputed_da3 = da3_artifact_complete(
        da3_root,
        context_count=options.da3_context_views,
        heldout_count=options.da3_heldout_views,
    )
    if "da3_reconstruction" in selected_metrics:
        try:
            context_indices, heldout_indices = context_heldout_indices(
                info.frame_count,
                context_count=options.da3_context_views,
                heldout_count=options.da3_heldout_views,
                candidate_indices=geometry_sampling_plan(case, info, options)["frame_indices"],
            )
            has_precomputed_da3 = da3_artifact_complete(da3_root,
                expected_context=context_indices, expected_heldout=heldout_indices)
            if not has_precomputed_da3:
                requested_indices.update(context_indices)
                requested_indices.update(heldout_indices)
        except ValueError as exc:
            has_precomputed_da3 = False
            da3_sampling_failure = str(exc)
    for values in geometry_correspondences.values():
        for item in values:
            requested_indices.update((item.source_index, item.target_index))
    quality_indices: set[int] = set()
    if set(selected_metrics) & {"imaging_quality", "aesthetic_quality", "long_horizon"}:
        quality_indices.update(
            range(0, info.frame_count, max(options.tier2_quality_frame_stride, 1))
        )
        requested_indices.update(quality_indices)
    for name in ("q0_return", "qe1"):
        index = _event_video_index(case, info, name)
        if index is not None:
            requested_indices.add(index)
    canonical_completion_by_pair: dict[str, list[tuple[int, int]]] = {}
    for name, pair in case.revisit_pairs.items():
        first_canonical = [int(index) for index in pair["first_window_indices"]]
        revisit_canonical = [int(index) for index in pair["revisit_window_indices"]]
        canonical_completion_by_pair[name] = []
        for first, revisit in zip(first_canonical, revisit_canonical):
            first_video = _canonical_video_index(case, info, first)
            revisit_video = _canonical_video_index(case, info, revisit)
            if first_video is not None and revisit_video is not None:
                canonical_completion_by_pair[name].append((first_video, revisit_video))
                requested_indices.update((first_video, revisit_video))
    # Pair C is the primary revisit pair of every case.
    first_canonical = [int(index) for index in case.completion_pairs["first_window_indices"]]
    revisit_canonical = [int(index) for index in case.completion_pairs["revisit_window_indices"]]
    canonical_completion = canonical_completion_by_pair["C"]
    q0_conditioned_index, q0_conditioned_metadata = _nearest_event(
        case=case, info=info, track=track, event_name="q0_return", radius=options.return_neighborhood_frames, options=options
    )
    qe_conditioned_index, qe_conditioned_metadata = _nearest_event(
        case=case, info=info, track=track, event_name="qe1", radius=options.qe_neighborhood_frames, options=options
    )
    if q0_conditioned_index is not None:
        requested_indices.add(q0_conditioned_index)
    if qe_conditioned_index is not None:
        requested_indices.add(qe_conditioned_index)
    nearest_completion: list[tuple[int, int, dict[str, Any]]] = []
    if track is not None:
        first_video = [index for canonical in first_canonical if (index := _canonical_video_index(case, info, canonical)) is not None]
        revisit_video = [index for canonical in revisit_canonical if (index := _canonical_video_index(case, info, canonical)) is not None]
        requested_reference_inv = np.linalg.inv(track.requested_reference_pose)
        first_match = nearest_cached_pose(
            track.cache,
            target_pose=requested_reference_inv @ case.requested_poses[case.events["qc_first"]],
            candidate_video_indices=first_video,
            path_scale=track.requested_path_scale,
            rotation_cost_weight=options.pose_rotation_cost_weight,
        )
        revisit_match = nearest_cached_pose(
            track.cache,
            target_pose=requested_reference_inv @ case.requested_poses[case.events["qc_revisit"]],
            candidate_video_indices=revisit_video,
            path_scale=track.requested_path_scale,
            rotation_cost_weight=options.pose_rotation_cost_weight,
        )
        if first_match is not None and revisit_match is not None:
            first_index, first_translation, first_rotation = first_match
            revisit_index, revisit_translation, revisit_rotation = revisit_match
            requested_indices.update((first_index, revisit_index))
            nearest_completion.append((
                first_index,
                revisit_index,
                {
                    "first_frame_index": first_index,
                    "revisit_frame_index": revisit_index,
                    "first_translation_error_normalized": first_translation,
                    "first_rotation_error_deg": first_rotation,
                    "revisit_translation_error_normalized": revisit_translation,
                    "revisit_rotation_error_deg": revisit_rotation,
                },
            ))
    completion_correspondences: dict[
        str, tuple[Any, ...]
    ] = {}
    if geometry_bundle is not None and "completion_revisit" in selected_metrics:
        completion_correspondences = build_correspondences(
            geometry_bundle,
            {
                **canonical_completion_by_pair,
                "nearest_pose": [
                    (first, revisit)
                    for first, revisit, _metadata in nearest_completion
                ],
            },
            options,
        )
    identity_start = info.timestamp_to_video_index(float(case.timestamps_s[case.events['q0']]))
    identity_end = info.timestamp_to_video_index(float(case.timestamps_s[case.events['qe1']]))
    if 'anchor_identity' in selected_metrics and identity_start is not None and identity_end is not None and identity_end > identity_start:
        requested_indices.update(sample_indices(identity_start, identity_end))
    anchor_windows = tracking_windows(case, info, options.anchor_tracking_frame_stride) if 'anchor_trajectory' in selected_metrics else {}
    requested_indices.update(i for group in anchor_windows.values() for i in group)
    repeated_qe = {}
    if 'long_horizon' in selected_metrics:
        for event_name, canonical_index in case.events.items():
            if event_name.startswith('qe') and event_name != 'qe1':
                if np.allclose(case.requested_poses[canonical_index], case.requested_poses[case.events['qe1']], atol=1e-5):
                    repeated_qe[event_name] = info.timestamp_to_video_index(float(case.timestamps_s[canonical_index]))
        requested_indices.update(i for i in repeated_qe.values() if i is not None)
    frames = read_video_frames(info, requested_indices)
    # Tier-2 quality sampling is protocol-defined and independent of whichever
    # geometry/persistence frames happen to be loaded in the same evaluator run.
    quality_frames = {
        index: frames[index] for index in sorted(quality_indices) if index in frames
    }
    comparator = image_comparator or ImageComparator(
        backend=options.feature_backend,
        dino_repo=options.dino_repo,
        dino_checkpoint=options.dino_checkpoint,
        lpips_checkpoint=options.lpips_checkpoint,
        device=options.device,
    )
    stage_timings["sampling_and_decode_s"] = perf_counter() - frame_prepare_started
    results: dict[str, MetricResult] = {}
    validity = inspection.validity
    imaging_cached: MetricResult | None = None
    aesthetic_cached: MetricResult | None = None

    def imaging_result() -> MetricResult:
        nonlocal imaging_cached
        if imaging_cached is None:
            imaging_cached = imaging_quality_metric(quality_frames, options)
        return imaging_cached

    def aesthetic_result() -> MetricResult:
        nonlocal aesthetic_cached
        if aesthetic_cached is None:
            aesthetic_cached = aesthetic_quality_metric(quality_frames, options)
        return aesthetic_cached

    def geometry_unavailable(name: str) -> MetricResult:
        return MetricResult(
            name=name,
            status="not_available" if geometry_failure and not geometry_failure.startswith(("RuntimeError:", "ValueError:", "ImportError:")) else "evaluator_failed",
            applicable=True,
            failure=geometry_failure or "estimator pose/depth inputs are unavailable",
            provenance={
                "pose_cache": str(paths.pose_cache) if paths.pose_cache else None,
                "depth_root": str(resolved_depth_root) if resolved_depth_root else None,
            },
        )

    geometry_provenance = {
        "pose_cache": str(paths.pose_cache) if paths.pose_cache else None,
        "depth_root": str(resolved_depth_root) if resolved_depth_root else None,
        "coordinate_convention": "native estimator c2w pose/depth coordinates; no per-pair scale fit",
    }

    def output_depth(frame_index: int | None) -> np.ndarray | None:
        if geometry_bundle is None or frame_index is None:
            return None
        depth = geometry_bundle.depths_by_index.get(frame_index)
        if depth is None:
            return None
        height, width = depth.shape
        transform = np.asarray(
            [[info.width / width, 0.0, 0.0],
             [0.0, info.height / height, 0.0],
             [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        return warp_image(
            depth, transform, width=info.width, height=info.height, nearest=True,
        )

    canonical_return_index = _event_video_index(case, info, "q0_return")
    canonical_qe_index = _event_video_index(case, info, "qe1")
    canonical_return_metadata = _event_frame_metadata(
        case=case, track=track, event_name="q0_return",
        frame_index=canonical_return_index, options=options,
    )
    canonical_qe_metadata = _event_frame_metadata(
        case=case, track=track, event_name="qe1",
        frame_index=canonical_qe_index, options=options,
    )
    aligned_pose_by_index = (
        {
            int(index): pose
            for pose, index in zip(track.cache.poses_c2w, track.cache.frame_indices)
        }
        if track is not None else {}
    )
    qe_target_pose = (
        np.linalg.inv(track.requested_reference_pose)
        @ case.requested_poses[case.events["qe1"]]
        if track is not None else None
    )
    output_K = output_transform @ case.canonical_camera.K

    def da3_metric() -> MetricResult:
        required = {
            "python": options.da3_python,
            "repo": options.da3_repo,
            "checkpoint": options.da3_checkpoint,
        }
        missing = [name for name, value in required.items() if value is None]
        adapter = None if missing else DA3Adapter(
            python=options.da3_python,
            bridge_script=Path(__file__).resolve().parents[1] / "backends/run_da3_evaluator.py",
            repo=options.da3_repo,
            checkpoint=options.da3_checkpoint,
            device=options.device,
            timeout_s=options.da3_timeout_s,
        )
        if has_precomputed_da3:
            return score_da3_artifact(
                output_root=da3_root,
                comparator=comparator,
                adapter=adapter,
            )
        if require_precomputed_da3:
            raise RuntimeError(
                f"precomputed DA3 artifact is missing or incomplete: {da3_root}"
            )
        if da3_sampling_failure is not None:
            return MetricResult(
                name="da3_reconstruction",
                status="not_covered",
                applicable=True,
                version="v0.1",
                failure=da3_sampling_failure,
            )
        if cache is None:
            raise RuntimeError("DA3 reconstruction requires a usable geometry pose cache")
        if missing:
            raise RuntimeError(f"DA3 configuration is missing: {', '.join(missing)}")
        assert adapter is not None
        poses = {
            int(index): pose
            for pose, index in zip(cache.poses_c2w, cache.frame_indices)
        }
        cached_intrinsics = (
            {}
            if cache.intrinsics is None
            else {
                int(index): intrinsic
                for intrinsic, index in zip(cache.intrinsics, cache.frame_indices)
            }
        )
        intrinsics = {
            index: cached_intrinsics.get(index, output_K)
            for index in frames
        }
        return run_da3_metric(
            frames=frames,
            poses_c2w=poses,
            intrinsics=intrinsics,
            frame_count=info.frame_count,
            output_root=da3_root,
            adapter=adapter,
            comparator=comparator,
            context_count=options.da3_context_views,
            heldout_count=options.da3_heldout_views,
            candidate_indices=geometry_sampling_plan(case, info, options)["frame_indices"],
            resolution=options.da3_input_resolution,
        )

    anchor_indices = sorted({i for values in anchor_windows.values() for i in values})
    anchor_requested = dict(zip(anchor_indices, _interpolate_poses(case.timestamps_s, case.requested_poses, np.asarray(anchor_indices)/info.fps))) if anchor_indices else {}
    from src.metrics.reference_metrics import ReferenceMetrics
    reference_metrics = ReferenceMetrics(case)
    builders: dict[str, Any] = {
        "anchor_identity": lambda: anchor_identity_metric(
            reference=q0_output, masks=q0_anchor_masks, frames=frames,
            start=identity_start, end=identity_end, video=paths.video_path,
            output=paths.output_root, options=options, comparator=comparator, case_id=case.case_id),
        "anchor_trajectory": lambda: anchor_trajectory_metric(
            reference=q0_output, reference_depth=q0_depth, reference_valid=q0_valid,
            masks=q0_anchor_masks, frames=frames, requested_poses=anchor_requested,
            reference_pose=case.requested_poses[case.events['q0']], K=output_K,
            windows=anchor_windows, options=(replace(options, anchor_tracking_visibility=options.anchor_tracking_visibility/case.case_id)
                if options.anchor_tracking_visibility and (options.anchor_tracking_visibility/case.case_id).is_dir() else options),
            executed_poses=({int(i):track.requested_reference_pose @ pose
                for i,pose in zip(track.cache.frame_indices,track.cache.poses_c2w)} if track is not None else None)),
        "validity": lambda: validity,
        "control": lambda: _control_metric(case, info, track, options),
        "da3_reconstruction": da3_metric,
        "q0_return": lambda: reference_metrics.q0(
            q0_reference=q0_output,
            canonical_return=frames.get(canonical_return_index),
            canonical_metadata=canonical_return_metadata,
            nearest_pose_return=frames.get(q0_conditioned_index) if q0_conditioned_index is not None else None,
            nearest_pose_metadata=q0_conditioned_metadata,
            comparator=comparator,
            valid_mask=q0_valid,
            anchor_masks=q0_anchor_masks,
            q0_depth=q0_depth,
            canonical_depth=output_depth(canonical_return_index),
            nearest_pose_depth=output_depth(q0_conditioned_index),
        ),
        "evidence_query": lambda: reference_metrics.qe(
            reference=qe_reference,
            input_supported_mask=qe_mask,
            invalid_mask=qe_invalid,
            anchor_masks=qe_anchor_masks,
            canonical_frame=frames.get(canonical_qe_index),
            canonical_metadata=canonical_qe_metadata,
            nearest_pose_frame=frames.get(qe_conditioned_index) if qe_conditioned_index is not None else None,
            nearest_pose_metadata=qe_conditioned_metadata,
            comparator=comparator,
            qe_depth=qe_depth,
            canonical_depth=output_depth(canonical_qe_index),
            nearest_pose_depth=output_depth(qe_conditioned_index),
            nearest_pose_source_pose=aligned_pose_by_index.get(qe_conditioned_index),
            qe_target_pose=qe_target_pose,
            camera_matrix=output_K,
            nearest_pose_geometry_scale=track.scale if track is not None else 1.0,
        ),
        "completion_revisit": lambda: _multi_revisit_metric(
            canonical_by_pair=completion_correspondences,
            revisit_pairs=case.revisit_pairs,
            exact_frame_contract="pairs" in case.completion_pairs,
            nearest_pose=completion_correspondences.get("nearest_pose", ()),
            nearest_metadata={
                (first, revisit): metadata
                for first, revisit, metadata in nearest_completion
            },
            pose_estimate_available=track is not None,
            bundle=geometry_bundle,
            options=options,
            frames=frames,
            comparator=comparator,
            provenance=geometry_provenance,
        ) if geometry_bundle is not None else geometry_unavailable("completion_revisit"),
        "depth_temporal_consistency": lambda: depth_temporal_consistency_metric(
            geometry_correspondences, provenance=geometry_provenance,
        ) if geometry_bundle is not None else geometry_unavailable("depth_temporal_consistency"),
        "covisible_reprojection": lambda: covisible_reprojection_metric(
            geometry_correspondences, frames=frames, comparator=comparator,
            max_pairs=options.geometry_feature_max_pairs, provenance=geometry_provenance,
        ) if geometry_bundle is not None else geometry_unavailable("covisible_reprojection"),
        "regional_scale_inconsistency": lambda: regional_scale_inconsistency_metric(
            geometry_correspondences, provenance=geometry_provenance,
        ) if geometry_bundle is not None else geometry_unavailable("regional_scale_inconsistency"),
        "imaging_quality": imaging_result,
        "aesthetic_quality": aesthetic_result,
        "long_horizon": lambda: long_horizon_metric(
            timestamps_s=case.timestamps_s,
            events=case.events,
            revisit_pairs=case.revisit_pairs,
            video_fps=info.fps,
            quality_frame_indices=sorted(quality_frames),
            imaging=imaging_result().to_dict(),
            aesthetic=aesthetic_result().to_dict(),
            window_s=options.temporal_window_s,
        ),
        "motion_smoothness": lambda: motion_smoothness_metric(str(info.path), options),
    }
    # Temporal summaries consume the metric results produced earlier in this run.
    selected_metrics = tuple(n for n in selected_metrics if n != 'long_horizon') + (('long_horizon',) if 'long_horizon' in selected_metrics else ())
    for name in selected_metrics:
        metric_path = _metric_path(paths.output_root, name)
        metric_started = perf_counter()
        try:
            result = builders[name]()
            if name == 'completion_revisit':
                from src.metrics.revisit_direct import direct_revisit
                direct = direct_revisit(case, canonical_completion_by_pair, frames,
                    geometry_bundle.depths_by_index if geometry_bundle is not None else {}, comparator)
                result = replace(result, version='v0.4',
                    raw={**result.raw, 'geometry_status': result.status, 'direct': direct})
            if name == 'evidence_query' and repeated_qe:
                repeated = {}
                for event_name, index in repeated_qe.items():
                    item = reference_metrics.qe(reference=qe_reference, input_supported_mask=qe_mask,
                        invalid_mask=qe_invalid, anchor_masks=qe_anchor_masks,
                        canonical_frame=frames.get(index), comparator=comparator,
                        qe_depth=qe_depth, canonical_depth=output_depth(index),
                        canonical_metadata={'event':event_name,'frame_index':index})
                    repeated[event_name] = item.to_dict()
                result = replace(result, raw={**result.raw, 'repeated_events':repeated})
            if name == 'long_horizon':
                result = replace(result, raw={**result.raw, **reused_temporal_diagnostics(
                    results, fps=info.fps, duration_s=float(case.timestamps_s[-1]),
                    window_s=options.temporal_window_s)})
        except Exception as exc:  # one failing metric must not abort the others
            result = MetricResult(
                name=name, status="evaluator_failed", applicable=True,
                failure=f"{type(exc).__name__}: {exc}",
                provenance=comparator.provenance,
            )
        result = replace(
            result,
            provenance={
                **result.provenance,
                "runtime_s": perf_counter() - metric_started,
            },
        )
        write_json(metric_path, result.to_dict())
        results[name] = result
    review_started = perf_counter()
    review = paths.output_root / "review"
    review.mkdir(parents=True, exist_ok=True)
    return_index = _event_video_index(case, info, "q0_return")
    if return_index is not None and return_index in frames:
        _write_review_pair(review / "q0_return.jpg", q0_output, frames[return_index], title="q0 input | canonical q0-return")
    qe_index = _event_video_index(case, info, "qe1")
    if qe_index is not None and qe_index in frames:
        overlay = frames[qe_index].copy()
        overlay[qe_mask] = (0.5 * overlay[qe_mask] + 0.5 * np.asarray([0, 255, 0])).astype(np.uint8)
        _write_review_pair(review / "qe_evidence.jpg", qe_reference, overlay, title="qe reference | generated qe with input-supported mask")
    offset = int(case.completion_pairs["event_offset_in_window"])
    if offset < len(canonical_completion):
        first, revisit = canonical_completion[offset]
        _write_review_pair(review / "completion_event.jpg", frames[first], frames[revisit], title="completion canonical event pair")
    stage_timings["review_write_s"] = perf_counter() - review_started
    stage_timings["metrics_s"] = {n: r.provenance.get("runtime_s") for n, r in results.items()}
    stage_timings["total_s"] = perf_counter() - evaluation_started
    run_status = evaluation_run_status(results)
    report = {
        "metric_schema_version": "0.2.0",
        "timing": stage_timings,
        "status": run_status,
        "case_id": case.case_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "video": {"path": str(info.path), "frame_count": info.frame_count, "fps": info.fps, "resolution": [info.width, info.height]},
        "pose_cache": str(paths.pose_cache) if paths.pose_cache else None,
        "metrics": {name: result.to_dict() for name, result in results.items()},
    }
    write_json(paths.output_root / "report.json", report)
    write_json(paths.output_root / "run.json", {
        "status": run_status, "case_root": str(case.root), "case_id": case.case_id,
        "camera_adapter_ckpt": None, "config": dict(config or {}),
    })
    return report
