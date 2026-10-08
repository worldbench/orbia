"""ORBIA v0 metric implementations, independent from CLI plumbing."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

import numpy as np

from src.metrics.contract import MetricResult
from src.metrics.features import ImageComparator


def _finite_mean(values: Iterable[float | None]) -> float | None:
    present = [float(value) for value in values if value is not None and np.isfinite(value)]
    return float(np.mean(present)) if present else None


_APPEARANCE_KEYS = ("psnr", "ssim", "lpips", "dino_similarity")


def _appearance(
    reference: np.ndarray,
    generated: np.ndarray,
    mask: np.ndarray,
    comparator: ImageComparator,
    *,
    local_dino: bool = False,
) -> dict[str, Any]:
    if not np.any(mask):
        keys = (*_APPEARANCE_KEYS[:3], "local_dino_similarity") if local_dino else _APPEARANCE_KEYS
        return {**{key: None for key in keys}, "valid_pixel_ratio": 0.0}
    value = comparator.compare(reference, generated, mask=mask)
    result = {key: value.get(key) for key in _APPEARANCE_KEYS}
    if local_dino:
        result.pop("dino_similarity", None)
        result["local_dino_similarity"] = comparator.local_dino_similarity(
            reference, generated, mask=mask, radius=1
        )
    result["valid_pixel_ratio"] = float(np.asarray(mask, dtype=bool).mean())
    return result


def _anchor_appearance(
    reference: np.ndarray,
    generated: np.ndarray,
    masks: Mapping[str, np.ndarray],
    support: np.ndarray,
    comparator: ImageComparator,
    *,
    local_dino: bool = False,
) -> tuple[dict[str, dict[str, Any]], dict[str, float | None]]:
    anchors: dict[str, dict[str, Any]] = {}
    for name, anchor_mask in masks.items():
        mask = np.asarray(anchor_mask, dtype=bool) & support
        if np.any(mask):
            anchors[str(name)] = _appearance(
                reference, generated, mask, comparator, local_dino=local_dino
            )
    keys = (*_APPEARANCE_KEYS[:3], "local_dino_similarity") if local_dino else _APPEARANCE_KEYS
    macro = {key: _finite_mean(item.get(key) for item in anchors.values()) for key in keys}
    return anchors, macro


def scale_aligned_depth_metrics(
    reference_depth: np.ndarray,
    estimated_depth: np.ndarray,
    mask: np.ndarray | None = None,
) -> dict[str, float | int | None]:
    """Fit one positive median scale and report AbsRel on common support."""

    reference = np.asarray(reference_depth, dtype=np.float64).squeeze()
    estimated = np.asarray(estimated_depth, dtype=np.float64).squeeze()
    if reference.ndim != 2 or estimated.ndim != 2:
        raise ValueError("depth maps must be two-dimensional")
    if estimated.shape != reference.shape:
        rows = np.minimum(
            np.arange(reference.shape[0]) * estimated.shape[0] // reference.shape[0],
            estimated.shape[0] - 1,
        )
        columns = np.minimum(
            np.arange(reference.shape[1]) * estimated.shape[1] // reference.shape[1],
            estimated.shape[1] - 1,
        )
        estimated = estimated[np.ix_(rows, columns)]
    expected = (
        np.ones(reference.shape, dtype=bool)
        if mask is None else np.asarray(mask, dtype=bool)
    )
    if expected.shape != reference.shape:
        raise ValueError("depth metric mask shape does not match reference depth")
    reference_valid = expected & np.isfinite(reference) & (reference > 0.0)
    common = reference_valid & np.isfinite(estimated) & (estimated > 0.0)
    if not np.any(common):
        return {"abs_rel": None, "coverage": 0.0, "scale": None, "valid_pixel_count": 0}
    scale = float(np.median(reference[common] / estimated[common]))
    residual = np.abs(scale * estimated[common] - reference[common]) / reference[common]
    return {
        "abs_rel": float(np.mean(residual)),
        "coverage": float(common.sum() / max(int(reference_valid.sum()), 1)),
        "fitted_scale": scale,
        "valid_pixel_count": int(common.sum()),
    }


def anchor_depth_metrics(reference_depth, estimated_depth, masks, expected_mask,
                         parent_depth, support=None):
    """Tier 4.4: reference masks and ONE scale inherited from the parent branch."""
    if reference_depth is None or estimated_depth is None:
        return {"status":"not_available", "anchors":{}, "macro_abs_rel":None}
    reference = np.asarray(reference_depth, dtype=float).squeeze()
    estimated = np.asarray(estimated_depth, dtype=float).squeeze()
    if estimated.shape != reference.shape:
        rows = np.minimum(np.arange(reference.shape[0])*estimated.shape[0]//reference.shape[0], estimated.shape[0]-1)
        cols = np.minimum(np.arange(reference.shape[1])*estimated.shape[1]//reference.shape[1], estimated.shape[1]-1)
        estimated = estimated[np.ix_(rows, cols)]
    scale = (parent_depth or {}).get("fitted_scale")
    visible = np.ones(reference.shape, bool) if support is None else np.asarray(support, bool)
    anchors = {}
    for name, mask in masks.items():
        expected = np.asarray(mask, bool) & np.asarray(expected_mask, bool) & np.isfinite(reference) & (reference > 0)
        common = expected & visible & np.isfinite(estimated) & (estimated > 0)
        count, total = int(common.sum()), int(expected.sum())
        value = float(np.mean(np.abs(scale*estimated[common]-reference[common])/reference[common])) if count and scale is not None else None
        anchors[str(name)] = {"abs_rel":value, "expected_pixel_count":total,
                              "evaluated_pixel_count":count, "coverage":count/total if total else None,
                              "status":"not_applicable" if not total else "ok" if value is not None else "not_covered"}
    values = [v["abs_rel"] for v in anchors.values() if v["abs_rel"] is not None]
    expected_count = sum(v["expected_pixel_count"] > 0 for v in anchors.values())
    complete = len(values) == expected_count
    return {"status":"ok" if complete and expected_count else "not_covered",
            "anchors":anchors, "parent_fitted_scale":scale,
            "macro_abs_rel":float(np.mean(values)) if values and complete else None,
            "conditional_macro_abs_rel":float(np.mean(values)) if values else None,
            "expected_anchor_count":expected_count, "evaluated_anchor_count":len(values),
            "mask_source":"reference", "scale_policy":"shared_parent_no_anchor_refit"}


def reproject_rgbd_to_target(
    rgb: np.ndarray,
    depth: np.ndarray,
    *,
    source_pose_c2w: np.ndarray,
    target_pose_c2w: np.ndarray,
    camera_matrix: np.ndarray,
    target_shape: tuple[int, int] | None = None,
    source_depth_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Lightweight pinhole RGB-D reprojection with a nearest-pixel z-buffer."""

    source_rgb = np.asarray(rgb, dtype=np.uint8)
    source_depth = np.asarray(depth, dtype=np.float64).squeeze() * float(source_depth_scale)
    if source_rgb.shape[:2] != source_depth.shape or source_rgb.ndim != 3:
        raise ValueError("RGB and depth must share the same image plane")
    K = np.asarray(camera_matrix, dtype=np.float64)
    source_pose = np.asarray(source_pose_c2w, dtype=np.float64)
    target_pose = np.asarray(target_pose_c2w, dtype=np.float64)
    if K.shape != (3, 3) or source_pose.shape != (4, 4) or target_pose.shape != (4, 4):
        raise ValueError("reprojection requires a 3x3 K and two 4x4 poses")
    target_height, target_width = target_shape or source_depth.shape
    rows, columns = np.nonzero(np.isfinite(source_depth) & (source_depth > 0.0))
    output_rgb = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    output_depth = np.zeros((target_height, target_width), dtype=np.float32)
    output_valid = np.zeros((target_height, target_width), dtype=bool)
    if not len(rows):
        return output_rgb, output_depth, output_valid
    z = source_depth[rows, columns]
    camera = np.stack([
        (columns - K[0, 2]) * z / K[0, 0],
        (rows - K[1, 2]) * z / K[1, 1],
        z,
        np.ones_like(z),
    ], axis=1)
    target_camera = (np.linalg.inv(target_pose) @ source_pose @ camera.T).T
    positive = target_camera[:, 2] > 0.0
    target_camera = target_camera[positive]
    colours = source_rgb[rows[positive], columns[positive]]
    u = np.rint(K[0, 0] * target_camera[:, 0] / target_camera[:, 2] + K[0, 2]).astype(np.int64)
    v = np.rint(K[1, 1] * target_camera[:, 1] / target_camera[:, 2] + K[1, 2]).astype(np.int64)
    inside = (u >= 0) & (u < target_width) & (v >= 0) & (v < target_height)
    u, v = u[inside], v[inside]
    projected_depth = target_camera[inside, 2]
    colours = colours[inside]
    order = np.argsort(projected_depth)[::-1]
    output_rgb[v[order], u[order]] = colours[order]
    output_depth[v[order], u[order]] = projected_depth[order].astype(np.float32)
    output_valid[v[order], u[order]] = True
    return output_rgb, output_depth, output_valid


def _return_branch(
    *,
    reference: np.ndarray,
    generated: np.ndarray,
    valid_mask: np.ndarray,
    anchor_masks: Mapping[str, np.ndarray],
    comparator: ImageComparator,
    metadata: Mapping[str, Any] | None,
    reference_depth: np.ndarray | None,
    generated_depth: np.ndarray | None,
) -> dict[str, Any]:
    anchors, macro = _anchor_appearance(
        reference, generated, anchor_masks, valid_mask, comparator
    )
    depth_result = (scale_aligned_depth_metrics(reference_depth, generated_depth, valid_mask)
                    if reference_depth is not None and generated_depth is not None else None)
    return {
        **dict(metadata or {}),
        "full_valid_image": _appearance(reference, generated, valid_mask, comparator),
        "anchors": anchors,
        "anchor_macro": macro,
        "depth": depth_result,
        "anchor_depth": anchor_depth_metrics(reference_depth, generated_depth, anchor_masks,
                                             valid_mask, depth_result),
    }


def q0_return_metric(
    *,
    q0_reference: np.ndarray,
    canonical_return: np.ndarray | None,
    comparator: ImageComparator,
    valid_mask: np.ndarray | None = None,
    anchor_masks: Mapping[str, np.ndarray] | None = None,
    canonical_metadata: Mapping[str, Any] | None = None,
    nearest_pose_return: np.ndarray | None = None,
    nearest_pose_metadata: Mapping[str, Any] | None = None,
    q0_depth: np.ndarray | None = None,
    canonical_depth: np.ndarray | None = None,
    nearest_pose_depth: np.ndarray | None = None,
) -> MetricResult:
    full_valid = np.ones(q0_reference.shape[:2], dtype=bool) if valid_mask is None else np.asarray(valid_mask, dtype=bool)
    anchors = dict(anchor_masks or {})
    if canonical_return is None:
        return MetricResult(
            name="q0_return", status="not_covered", applicable=True,
            coverage={"canonical_return_available": False},
            failure="generated video does not cover the annotated q0-return timestamp",
            provenance=comparator.provenance,
        )
    canonical = _return_branch(
        reference=q0_reference, generated=canonical_return, valid_mask=full_valid,
        anchor_masks=anchors, comparator=comparator, metadata=canonical_metadata,
        reference_depth=q0_depth, generated_depth=canonical_depth,
    )
    raw: dict[str, Any] = {"canonical": canonical}
    coverage: dict[str, Any] = {"canonical_return_available": True}
    if nearest_pose_return is not None:
        raw["nearest_pose"] = _return_branch(
            reference=q0_reference, generated=nearest_pose_return, valid_mask=full_valid,
            anchor_masks=anchors, comparator=comparator, metadata=nearest_pose_metadata,
            reference_depth=q0_depth, generated_depth=nearest_pose_depth,
        )
        coverage["nearest_pose_available"] = True
    else:
        raw["nearest_pose"] = {**dict(nearest_pose_metadata or {}), "status": "not_available"}
        coverage["nearest_pose_available"] = False
    return MetricResult(
        name="q0_return", status="ok", applicable=True, version="v0.2",
        coverage=coverage, raw=raw,
        provenance=comparator.provenance,
    )


def qe_evidence_metric(
    *,
    reference: np.ndarray,
    input_supported_mask: np.ndarray,
    anchor_masks: Mapping[str, np.ndarray],
    canonical_frame: np.ndarray | None,
    comparator: ImageComparator,
    invalid_mask: np.ndarray | None = None,
    canonical_metadata: Mapping[str, Any] | None = None,
    nearest_pose_frame: np.ndarray | None = None,
    nearest_pose_metadata: Mapping[str, Any] | None = None,
    qe_depth: np.ndarray | None = None,
    canonical_depth: np.ndarray | None = None,
    nearest_pose_depth: np.ndarray | None = None,
    nearest_pose_source_pose: np.ndarray | None = None,
    qe_target_pose: np.ndarray | None = None,
    camera_matrix: np.ndarray | None = None,
    nearest_pose_geometry_scale: float = 1.0,
) -> MetricResult:
    evidence = np.asarray(input_supported_mask, dtype=bool).copy()
    if invalid_mask is not None:
        evidence &= ~np.asarray(invalid_mask, dtype=bool)
    valid_ratio = float(evidence.mean())
    if canonical_frame is None:
        return MetricResult(
            name="evidence_query", status="not_covered", applicable=True,
            coverage={"input_supported_pixel_ratio": valid_ratio, "canonical_qe_available": False},
            failure="generated video does not cover the annotated qe timestamp",
            provenance=comparator.provenance,
        )
    def branch(
        frame: np.ndarray,
        depth: np.ndarray | None,
        support: np.ndarray,
        metadata: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        evaluated = evidence & support
        anchors, macro = _anchor_appearance(
            reference, frame, anchor_masks, evaluated, comparator, local_dino=True
        )
        depth_result = (scale_aligned_depth_metrics(qe_depth, depth, evaluated)
                        if qe_depth is not None and depth is not None else None)
        return {
            **dict(metadata or {}),
            "expected_support": int(evidence.sum()),
            "evaluated_support": int(evaluated.sum()),
            "overall_evidence": _appearance(
                reference, frame, evaluated, comparator, local_dino=True
            ),
            "anchors": anchors,
            "anchor_macro": macro,
            "depth": depth_result,
            "anchor_depth": anchor_depth_metrics(qe_depth, depth, anchor_masks,
                                                 evidence, depth_result, support),
        }

    canonical = branch(
        canonical_frame, canonical_depth, np.ones(evidence.shape, dtype=bool), canonical_metadata
    )
    raw: dict[str, Any] = {"canonical": canonical}
    coverage = {
        "input_supported_pixel_ratio": valid_ratio,
        "canonical_qe_available": True,
        "anchor_mask_count": len(anchor_masks),
        "canonical_anchor_count": len(canonical["anchors"]),
    }
    if nearest_pose_frame is not None:
        nearest_support = np.ones(evidence.shape, dtype=bool)
        if (
            nearest_pose_depth is not None and nearest_pose_source_pose is not None
            and qe_target_pose is not None and camera_matrix is not None
        ):
            nearest_pose_frame, nearest_pose_depth, nearest_support = reproject_rgbd_to_target(
                nearest_pose_frame, nearest_pose_depth,
                source_pose_c2w=nearest_pose_source_pose, target_pose_c2w=qe_target_pose,
                camera_matrix=camera_matrix, target_shape=evidence.shape,
                source_depth_scale=nearest_pose_geometry_scale,
            )
        raw["nearest_pose"] = branch(
            nearest_pose_frame, nearest_pose_depth, nearest_support, nearest_pose_metadata
        )
        coverage["nearest_pose_available"] = True
        coverage["nearest_pose_anchor_count"] = len(raw["nearest_pose"]["anchors"])
    else:
        raw["nearest_pose"] = {**dict(nearest_pose_metadata or {}), "status": "not_available"}
        coverage["nearest_pose_available"] = False
    return MetricResult(
        name="evidence_query", status="ok", applicable=True, version="v0.2",
        coverage=coverage, raw=raw,
        provenance=comparator.provenance,
    )
