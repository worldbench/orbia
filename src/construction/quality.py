"""70-point candidate quality ranking (visual, anchor, trajectory, revisit)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image

from src.construction.anchor_policy import (
    measure_endpoint_mask,
    strict_anchor_policy_failures,
)


SCHEMA_VERSION = "orbia.quality_ranking.v1"

# Fixed knots calibrated on a 101-case validation set; the score uses the
# sharper-limited (worse) endpoint.
SHARPNESS_KNOTS = (
    (0.0, 0.0),
    (300.0, 1.0),
    (450.0, 2.0),
    (650.0, 3.25),
    (900.0, 4.25),
    (1400.0, 5.0),
)


def image_sharpness(rgb: np.ndarray) -> float:
    """Cheap deterministic edge-energy sharpness without an OpenCV dependency."""

    rgb = np.asarray(rgb, dtype=np.float32)
    gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    # Subsample only for compute; every candidate uses the same rule.
    gray = gray[::4, ::4]
    if gray.shape[0] < 2 or gray.shape[1] < 2:
        return 0.0
    dx = np.diff(gray, axis=1)
    dy = np.diff(gray, axis=0)
    return float(np.var(dx) + np.var(dy))


def _clip(value: float, low: float, high: float) -> float:
    return min(max(float(value), low), high)


def _round(value: float) -> float:
    return round(float(value), 3)


def _interp(value: float, knots: Sequence[tuple[float, float]]) -> float:
    value = float(value)
    if value <= knots[0][0]:
        return float(knots[0][1])
    for (x0, y0), (x1, y1) in zip(knots, knots[1:]):
        if value <= x1:
            if x1 == x0:
                return float(y1)
            alpha = (value - x0) / (x1 - x0)
            return float(y0 + alpha * (y1 - y0))
    return float(knots[-1][1])


def _number(value: object, default: float | None = None) -> float | None:
    if value is None:
        return default
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if np.isfinite(result) else default


def _mean(values: Sequence[float], *, default: float = 0.0) -> float:
    return float(np.mean(values)) if values else float(default)


def score_visual_quality(case: Mapping[str, Any]) -> dict[str, Any]:
    endpoints = dict(case["assets"]["raw_endpoints"])
    values: dict[str, float] = {}
    for role in ("q0", "qe"):
        with Image.open(Path(str(endpoints[role]))) as image:
            rgb = np.asarray(image.convert("RGB"))
        values[role] = float(image_sharpness(rgb))
    worst = min(values.values())
    score = _clip(_interp(worst, SHARPNESS_KNOTS), 0.0, 5.0)
    flags = []
    if score < 2.0:
        flags.append("endpoint_sharpness_poor")
    elif score < 3.25:
        flags.append("endpoint_sharpness_risk")
    return {
        "score": _round(score),
        "maximum": 5.0,
        "q0_sharpness": _round(values["q0"]),
        "qe_sharpness": _round(values["qe"]),
        "scoring_sharpness": _round(worst),
        "policy": "minimum_of_q0_and_qe_only",
        "flags": flags,
    }


def _load_mask_geometry(anchor: Mapping[str, Any], role: str) -> dict[str, Any]:
    """Measure selected independent masks, never rendered QA overlays."""

    source = dict((anchor.get("mask_sources") or {}).get(role) or {})
    path = source.get("npz")
    index = source.get("index")
    if path is None or index is None:
        return {"available": False}
    try:
        with np.load(Path(str(path)), allow_pickle=False) as archive:
            masks = np.asarray(archive["out_binary_masks"], dtype=bool)
        if masks.ndim == 2:
            masks = masks[None]
        mask = masks[int(index)]
    except (OSError, KeyError, IndexError, TypeError, ValueError):
        return {"available": False}
    return measure_endpoint_mask(mask)


def _anchor_area_score(
    anchor: Mapping[str, Any], geometry: Mapping[str, Mapping[str, Any]] | None = None
) -> tuple[float, dict[str, Any]]:
    areas = dict(anchor.get("area_percent") or {})
    q0 = max(_number(areas.get("q0"), 0.0) or 0.0, 0.0)
    qe = max(_number(areas.get("qe"), 0.0) or 0.0, 0.0)
    minimum, maximum = min(q0, qe), max(q0, qe)
    score = _interp(
        minimum,
        ((0.0, 0.0), (0.3, 2.0), (1.0, 5.0), (2.0, 8.0), (5.0, 10.0)),
    )
    if maximum > 50.0:
        score *= _interp(maximum, ((50.0, 1.0), (70.0, 0.5), (100.0, 0.0)))
    balance = minimum / maximum if maximum > 0.0 else 0.0
    if balance < 0.5:
        score *= _interp(balance, ((0.0, 0.35), (0.25, 0.65), (0.5, 1.0)))
    geometry = dict(geometry or {
        role: _load_mask_geometry(anchor, role) for role in ("q0", "qe")
    })
    position_factors = [
        float(row["position_factor"])
        for row in geometry.values()
        if bool(row.get("available"))
    ]
    position_factor = min(position_factors, default=1.0)
    score *= position_factor
    score = 1.3 * _clip(score, 0.0, 10.0)
    return _clip(score, 0.0, 13.0), {
        "q0_percent": _round(q0),
        "qe_percent": _round(qe),
        "endpoint_balance": _round(balance),
        "position_factor": _round(position_factor),
        "mask_geometry": geometry,
    }


def _endpoint_integrity_score(
    row: Mapping[str, Any], geometry: Mapping[str, Any] | None = None
) -> tuple[float, dict[str, Any]]:
    geometry = dict(geometry or {})
    largest = _number(row.get("largest_component_ratio"), 1.0) or 0.0
    holes = _number(row.get("hole_ratio"), 0.0) or 0.0
    components = int(
        _number(
            geometry.get("significant_component_count_0_2pct"),
            _number(row.get("significant_component_count"), 1.0),
        )
        or 0
    )
    border_ratio = _number(row.get("border_contact_pixel_ratio"), 0.0) or 0.0
    closing_gap = _number(geometry.get("closing_gap_ratio"), 0.0) or 0.0
    enclosed_hole = _number(geometry.get("enclosed_hole_ratio"), 0.0) or 0.0
    internal_gap = max(holes, closing_gap, enclosed_hole)
    aspect = _number(geometry.get("bbox_aspect_ratio"), 1.0) or 1.0
    solidity = _number(geometry.get("solidity"), 1.0) or 1.0
    score = _interp(
        largest,
        ((0.0, 0.0), (0.5, 1.0), (0.75, 4.0), (0.85, 7.0), (0.95, 10.0)),
    )
    score -= _interp(
        internal_gap,
        ((0.0, 0.0), (0.02, 0.0), (0.05, 1.0), (0.10, 3.0), (0.20, 6.0)),
    )
    score -= min(max(components - 1, 0) * 1.5, 4.0)
    # Merely touching a border is not a defect.  Only substantial border contact
    # adds a small diagnostic penalty; it is never a hard rejection.
    score -= _interp(border_ratio, ((0.0, 0.0), (0.03, 0.0), (0.10, 0.5), (0.25, 1.0)))
    # Long strips and irregular silhouettes are less useful than coherent,
    # compact regions, even when technically connected.
    score -= _interp(aspect, ((1.0, 0.0), (3.0, 0.0), (5.0, 1.0), (8.0, 3.0), (12.0, 5.0)))
    score -= _interp(
        solidity,
        (
            (0.0, 7.0), (0.60, 7.0), (0.80, 5.0),
            (0.90, 3.333), (0.95, 1.667), (0.99, 0.0),
        ),
    )
    if internal_gap > 0.20:
        score = min(score, 2.0)
    return _clip(score, 0.0, 10.0), {
        "largest_component_ratio": _round(largest),
        "hole_ratio": _round(holes),
        "closing_gap_ratio": _round(closing_gap),
        "enclosed_hole_ratio": _round(enclosed_hole),
        "internal_gap_ratio": _round(internal_gap),
        "significant_component_count": components,
        "detached_component_fraction": _round(
            _number(geometry.get("detached_component_fraction"), 0.0) or 0.0
        ),
        "touches_image_border": bool(row.get("touches_image_border", False)),
        "border_contact_pixel_ratio": _round(border_ratio),
        "bbox_aspect_ratio": _round(aspect),
        "solidity": _round(solidity),
    }


def _anchor_integrity_score(
    anchor: Mapping[str, Any], geometry: Mapping[str, Mapping[str, Any]] | None = None
) -> tuple[float, dict[str, Any]]:
    integrity = dict(anchor.get("integrity") or {})
    geometry = dict(geometry or {
        role: _load_mask_geometry(anchor, role) for role in ("q0", "qe")
    })
    endpoint_rows: dict[str, Any] = {}
    endpoint_scores = []
    for role in ("q0", "qe"):
        value, details = _endpoint_integrity_score(
            dict(integrity.get(role) or {}), geometry.get(role)
        )
        endpoint_scores.append(value)
        endpoint_rows[role] = {"score": _round(value), **details}
    # Preserve the endpoint diagnostic on its intuitive 0-10 scale while the
    # ranking component uses the agreed 12-point weight.
    return 1.2 * min(endpoint_scores), endpoint_rows


def _anchor_postfilter_reasons(
    anchor: Mapping[str, Any], geometry: Mapping[str, Mapping[str, Any]]
) -> list[str]:
    """Apply the strict mask/identity policy to a selected anchor."""

    ious = dict(anchor.get("tracking_iou") or {})
    areas = dict(anchor.get("area_percent") or {})
    q0_area = _number(areas.get("q0"), 0.0) or 0.0
    qe_area = _number(areas.get("qe"), 0.0) or 0.0
    integrity = dict(anchor.get("integrity") or {})
    endpoint_integrities: dict[str, dict[str, Any]] = {}
    for role in ("q0", "qe"):
        endpoint_integrities[role] = {
            **dict(integrity.get(role) or {}),
            **dict(geometry.get(role) or {}),
        }
    canonical = strict_anchor_policy_failures(
        identity_confirmed=bool(anchor.get("identity_confirmed")),
        q0_tracking_iou=_number(ious.get("q0"), 0.0) or 0.0,
        qe_tracking_iou=_number(ious.get("qe"), 0.0) or 0.0,
        q0_area_percent=q0_area,
        qe_area_percent=qe_area,
        q0_integrity=endpoint_integrities["q0"],
        qe_integrity=endpoint_integrities["qe"],
    )
    reason_names = {
        "identity_missing": "tracking_iou_below_0_8",
        "tracking_iou_below_minimum": "tracking_iou_below_0_8",
        "endpoint_area_below_minimum": "endpoint_area_below_0_3_percent",
        "endpoint_area_above_maximum": "endpoint_area_above_50_percent",
        "q0_largest_component_below_minimum": "q0_largest_component_below_0_85",
        "qe_largest_component_below_minimum": "qe_largest_component_below_0_85",
        "q0_internal_gap_above_maximum": "q0_internal_gap_above_0_20",
        "qe_internal_gap_above_maximum": "qe_internal_gap_above_0_20",
    }
    return list(dict.fromkeys(reason_names.get(reason, reason) for reason in canonical))


def _anchor_tracking_score(anchor: Mapping[str, Any]) -> tuple[float, dict[str, Any]]:
    ious = dict(anchor.get("tracking_iou") or {})
    q0 = _number(ious.get("q0"), 0.0) or 0.0
    qe = _number(ious.get("qe"), 0.0) or 0.0
    minimum = min(q0, qe)
    if not bool(anchor.get("identity_confirmed")):
        score = 0.0
    else:
        score = _interp(minimum, ((0.0, 0.0), (0.6, 0.0), (0.8, 3.0), (0.9, 5.0)))
    return _clip(score, 0.0, 5.0), {
        "identity_confirmed": bool(anchor.get("identity_confirmed")),
        "q0_iou": _round(q0),
        "qe_iou": _round(qe),
        "minimum_iou": _round(minimum),
    }


def _depth_direction_score(row: Mapping[str, Any]) -> tuple[float, dict[str, Any]]:
    within = _number(row.get("depth_residual_within_0_25_fraction"), 0.0) or 0.0
    residual = _number(row.get("median_normalized_depth_residual"), 1.0) or 1.0
    inside = _number(row.get("projected_inside_target_mask_fraction"), 0.0) or 0.0
    score = (
        _interp(within, ((0.0, 0.0), (0.5, 1.0), (0.75, 3.0), (0.9, 4.0)))
        + _interp(residual, ((0.0, 3.0), (0.10, 2.5), (0.25, 1.0), (0.50, 0.0)))
        + _interp(inside, ((0.0, 0.0), (0.5, 1.0), (0.8, 2.5), (0.95, 3.0)))
    )
    return _clip(score, 0.0, 10.0), {
        "within_0_25_fraction": _round(within),
        "median_normalized_residual": _round(residual),
        "inside_target_mask_fraction": _round(inside),
    }


def _anchor_depth_score(anchor: Mapping[str, Any]) -> tuple[float, dict[str, Any]]:
    depth = dict(anchor.get("depth_consistency") or {})
    grade = str(depth.get("grade") or "unavailable")
    direction_rows: dict[str, Any] = {}
    direction_scores = []
    for role in ("q0_to_qe", "qe_to_q0"):
        row = depth.get(role)
        if not isinstance(row, Mapping):
            continue
        value, details = _depth_direction_score(row)
        direction_scores.append(value)
        direction_rows[role] = {"score": _round(value), **details}
    if not direction_scores or grade == "unavailable":
        score = 3.0
    else:
        score = _mean(direction_scores)
        if grade == "risk":
            score = min(score, 3.0)
    return _clip(score, 0.0, 10.0), {
        "grade": grade,
        "policy": "learned_depth_diagnostic_not_ground_truth",
        "directions": direction_rows,
    }


def score_anchor_quality(case: Mapping[str, Any]) -> dict[str, Any]:
    anchors = list(case.get("anchors") or [])
    if not anchors:
        return {
            "score": 0.0,
            "maximum": 35.0,
            "breakdown": {"area_and_position": 0.0, "mask_integrity": 0.0, "depth": 0.0},
            "anchors": [],
            "rejected_anchors": [],
            "flags": ["no_selected_anchor"],
        }
    rows = []
    rejected_rows = []
    for index, anchor in enumerate(anchors, 1):
        geometry = {role: _load_mask_geometry(anchor, role) for role in ("q0", "qe")}
        rejection_reasons = _anchor_postfilter_reasons(anchor, geometry)
        if rejection_reasons:
            rejected_rows.append({
                "source_anchor_id": f"A{index}",
                "label": str(anchor.get("label") or ""),
                "reasons": rejection_reasons,
            })
            continue
        area, area_details = _anchor_area_score(anchor, geometry)
        integrity, integrity_details = _anchor_integrity_score(anchor, geometry)
        tracking, tracking_details = _anchor_tracking_score(anchor)
        depth, depth_details = _anchor_depth_score(anchor)
        anchor_score = area + integrity + depth
        rows.append({
            "anchor_id": f"A{len(rows) + 1}",
            "source_anchor_id": f"A{index}",
            "label": str(anchor.get("label") or ""),
            "score": _round(_clip(anchor_score, 0.0, 35.0)),
            "scores": {
                "area_and_position": _round(area),
                "mask_integrity": _round(integrity),
                "depth": _round(depth),
            },
            "area": area_details,
            "integrity": integrity_details,
            "tracking": tracking_details,
            "depth": depth_details,
        })
    if not rows:
        return {
            "score": 0.0,
            "maximum": 35.0,
            "breakdown": {"area_and_position": 0.0, "mask_integrity": 0.0, "depth": 0.0},
            "aggregation": "highest_scoring_reliable_anchor",
            "anchors": [],
            "rejected_anchors": rejected_rows,
            "flags": ["no_reliable_anchor"],
        }
    selected_anchor = max(rows, key=lambda row: float(row["score"]))
    breakdown = dict(selected_anchor["scores"])
    total = float(selected_anchor["score"])
    flags = []
    if breakdown["mask_integrity"] < 7.2:
        flags.append("anchor_integrity_risk")
    if breakdown["depth"] < 5.0:
        flags.append("learned_depth_consistency_risk")
    return {
        "score": _round(_clip(total, 0.0, 35.0)),
        "maximum": 35.0,
        "breakdown": breakdown,
        "aggregation": "highest_scoring_reliable_anchor",
        "selected_anchor_id": selected_anchor["anchor_id"],
        "anchors": rows,
        "rejected_anchors": rejected_rows,
        "flags": flags,
    }


def trajectory_numeric_metrics(path: Path) -> dict[str, float]:
    with np.load(path, allow_pickle=False) as archive:
        timestamps = np.asarray(archive["timestamps_s"], dtype=np.float64)
        poses = np.asarray(archive["poses_c2w"], dtype=np.float64)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4) or len(poses) < 2:
        raise ValueError(f"invalid trajectory poses: {path}: {poses.shape}")
    if timestamps.shape != (len(poses),) or not np.isfinite(poses).all():
        raise ValueError(f"invalid trajectory arrays: {path}")
    positions = poses[:, :3, 3]
    relative = positions - positions[0]
    horizontal_excursion = np.linalg.norm(relative[:, (0, 2)], axis=1)
    maximum_horizontal = float(np.max(horizontal_excursion))
    maximum_3d = float(np.max(np.linalg.norm(relative, axis=1)))
    vertical_excursion = float(np.max(np.abs(relative[:, 1])))
    vertical_ratio = vertical_excursion / max(maximum_horizontal, 1.0e-6)

    delta = np.diff(positions, axis=0)
    horizontal_step = np.linalg.norm(delta[:, (0, 2)], axis=1)
    active = horizontal_step > max(float(np.percentile(horizontal_step, 25)), 1.0e-7)
    slopes = np.abs(delta[:, 1]) / np.maximum(horizontal_step, 1.0e-8)
    p95_vertical_slope = float(np.percentile(slopes[active], 95)) if np.any(active) else 0.0

    down_axes = poses[:, :3, 1]
    down_axes /= np.maximum(np.linalg.norm(down_axes, axis=1, keepdims=True), 1.0e-12)
    reference_down = down_axes[0]
    tilt = np.degrees(np.arccos(np.clip(down_axes @ reference_down, -1.0, 1.0)))

    translation_steps = np.linalg.norm(delta, axis=1)
    translation_p90 = float(np.percentile(translation_steps, 90))
    translation_spike = float(np.max(translation_steps)) / max(translation_p90, 1.0e-8)

    relative_rotations = np.einsum(
        "nij,njk->nik", np.transpose(poses[:-1, :3, :3], (0, 2, 1)), poses[1:, :3, :3]
    )
    traces = np.trace(relative_rotations, axis1=1, axis2=2)
    angular_steps = np.degrees(np.arccos(np.clip((traces - 1.0) / 2.0, -1.0, 1.0)))
    angular_p90 = float(np.percentile(angular_steps, 90))
    angular_spike = float(np.max(angular_steps)) / max(angular_p90, 1.0e-8)

    from_q0 = np.einsum(
        "ij,njk->nik", poses[0, :3, :3].T, poses[:, :3, :3]
    )
    rotation_from_q0 = np.degrees(
        np.arccos(np.clip((np.trace(from_q0, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0))
    )
    return {
        "maximum_horizontal_excursion": maximum_horizontal,
        # Display diagnostic only; not the path length.
        "maximum_3d_distance_from_q0": maximum_3d,
        "maximum_rotation_from_q0_deg": float(np.max(rotation_from_q0)),
        "vertical_excursion_ratio": vertical_ratio,
        "p95_vertical_step_slope": p95_vertical_slope,
        "maximum_down_axis_deviation_deg": float(np.max(tilt)),
        "translation_step_spike_ratio": translation_spike,
        "angular_step_spike_ratio": angular_spike,
    }


@dataclass(frozen=True)
class MotionCalibration:
    by_template: Mapping[str, tuple[float, float, float]]


def fit_motion_calibration(cases: Sequence[Mapping[str, Any]]) -> MotionCalibration:
    grouped: dict[str, list[float]] = {}
    for case in cases:
        metrics = trajectory_numeric_metrics(Path(str(case["assets"]["trajectory_npz"])))
        value = (
            metrics["maximum_rotation_from_q0_deg"]
            if str(case["template"]) == "yaw_return"
            else metrics["maximum_horizontal_excursion"]
        )
        grouped.setdefault(str(case["template"]), []).append(value)
    return MotionCalibration({
        template: tuple(float(np.percentile(values, q)) for q in (10, 50, 90))
        for template, values in grouped.items()
    })


def _motion_score(value: float, calibration: tuple[float, float, float]) -> float:
    q10, q50, q90 = calibration
    if q90 - q10 < 1.0e-8:
        return 10.0
    # Per-template relative calibration handles unknown SpatialVID pose scale.
    # Do not use path length: return legs must not receive artificial credit.
    return _interp(value, ((0.0, 0.0), (q10, 6.0), (q50, 10.0), (q90, 15.0)))


def score_trajectory_quality(
    case: Mapping[str, Any], calibration: MotionCalibration
) -> dict[str, Any]:
    trajectory = dict((case.get("diagnostics") or {}).get("trajectory") or {})
    reported = dict(trajectory.get("metrics") or {})
    numeric = trajectory_numeric_metrics(Path(str(case["assets"]["trajectory_npz"])))

    # Template correctness remains in the report for review, but it does not
    # consume ranking points. Cases have already passed construction/validation.
    template_checks = [
        bool(trajectory.get("passed")),
        str(trajectory.get("template_name")) == str(case.get("template")),
    ]
    for key in (
        "route_geometry_valid", "global_arc_geometry_valid", "piecewise_curvature_valid",
        "outward_monotonic", "return_monotonic", "qe1_registered_binding_valid",
    ):
        if key in reported:
            template_checks.append(bool(reported[key]))
    template_diagnostic = {
        "passed_fraction": _round(_mean([float(value) for value in template_checks])),
        "all_passed": bool(all(template_checks)),
        "check_count": len(template_checks),
    }

    translation_naturalness = _interp(
        numeric["translation_step_spike_ratio"],
        ((1.0, 2.5), (1.5, 2.5), (3.0, 1.25), (6.0, 0.0)),
    )
    angular_naturalness = _interp(
        numeric["angular_step_spike_ratio"],
        ((1.0, 2.5), (1.5, 2.5), (3.0, 1.25), (6.0, 0.0)),
    )
    naturalness_score = translation_naturalness + angular_naturalness

    motion_value = (
        numeric["maximum_rotation_from_q0_deg"]
        if str(case["template"]) == "yaw_return"
        else numeric["maximum_horizontal_excursion"]
    )
    motion_score = _motion_score(motion_value, calibration.by_template[str(case["template"])])

    tilt_score = _interp(
        numeric["maximum_down_axis_deviation_deg"],
        ((0.0, 2.0), (3.0, 2.0), (8.0, 1.5), (15.0, 0.5), (30.0, 0.0)),
    )
    vertical_score = _interp(
        numeric["vertical_excursion_ratio"],
        ((0.0, 2.0), (0.03, 2.0), (0.08, 1.5), (0.15, 0.75), (0.35, 0.0)),
    )
    slope_score = _interp(
        numeric["p95_vertical_step_slope"],
        ((0.0, 1.0), (0.05, 1.0), (0.12, 0.5), (0.30, 0.0)),
    )
    upright_score = tilt_score + vertical_score + slope_score
    breakdown = {
        "effective_motion_magnitude": _round(motion_score),
        "horizontal_vertical_behavior": _round(upright_score),
    }
    total = sum(breakdown.values())
    flags = []
    if upright_score < 3.5:
        flags.append("side_or_vertical_trajectory_risk")
    if naturalness_score < 3.0:
        flags.append("trajectory_step_discontinuity_risk")
    if not template_diagnostic["all_passed"]:
        flags.append("trajectory_template_risk")
    return {
        "score": _round(_clip(total, 0.0, 20.0)),
        "maximum": 20.0,
        "breakdown": breakdown,
        "step_continuity_diagnostic": _round(naturalness_score),
        "template_compliance_diagnostic": template_diagnostic,
        "numeric_metrics": {key: _round(value) for key, value in numeric.items()},
        "motion_calibration_q10_q50_q90": [
            _round(value) for value in calibration.by_template[str(case["template"])]
        ],
        "flags": flags,
    }


def score_completion_value(case: Mapping[str, Any]) -> dict[str, Any]:
    completion = dict((case.get("diagnostics") or {}).get("completion") or {})
    grade = str(completion.get("strength_grade") or "invalid").lower()
    first = _number(completion.get("first_shared_unseen_fraction"), 0.0) or 0.0
    revisit = _number(completion.get("revisit_shared_unseen_fraction"), 0.0) or 0.0
    minimum = min(first, revisit)
    if not bool(completion.get("passed")) or grade not in {"strong", "moderate", "weak"}:
        score = 0.0
    elif grade == "strong":
        score = _interp(minimum, ((0.40, 8.0), (0.70, 10.0)))
    elif grade == "moderate":
        score = _interp(minimum, ((0.10, 4.0), (0.40, 6.0)))
        score = min(score, 6.0)
    else:
        score = _interp(minimum, ((0.0, 0.5), (0.10, 2.0)))
    translation_error = _number(completion.get("event_pose_translation_error"), 0.0) or 0.0
    rotation_error = _number(completion.get("event_pose_rotation_error_deg"), 0.0) or 0.0
    # Pose penalties are scaled to the 10-point revisit component.
    pose_penalty = _interp(translation_error, ((0.0, 0.0), (0.05, 0.0), (0.20, 1.0)))
    pose_penalty += _interp(rotation_error, ((0.0, 0.0), (2.0, 0.0), (10.0, 1.0)))
    score = _clip(score - pose_penalty, 0.0, 10.0)
    flags = [] if grade == "strong" else [f"completion_{grade}"]
    return {
        "score": _round(score),
        "maximum": 10.0,
        "strength_grade": grade,
        "first_shared_unseen_fraction": _round(first),
        "revisit_shared_unseen_fraction": _round(revisit),
        "minimum_shared_unseen_fraction": _round(minimum),
        "event_pose_penalty": _round(pose_penalty),
        "policy": "strong_8_to_10_moderate_4_to_6_weak_0_5_to_2",
        "flags": flags,
    }


def score_case(
    case: Mapping[str, Any], *, motion_calibration: MotionCalibration
) -> dict[str, Any]:
    visual = score_visual_quality(case)
    anchor = score_anchor_quality(case)
    trajectory = score_trajectory_quality(case, motion_calibration)
    completion = score_completion_value(case)
    scores = {
        "visual_quality": visual["score"],
        "anchor_quality": anchor["score"],
        "trajectory_quality": trajectory["score"],
        "completion_value": completion["score"],
    }
    total = _round(sum(scores.values()))
    flags = sorted(set(
        visual["flags"] + anchor["flags"] + trajectory["flags"] + completion["flags"]
    ))
    return {
        "case_id": str(case["case_id"]),
        "video_id": str(case["video_id"]),
        "template": str(case["template"]),
        "source_group": str(case.get("source_group") or "unspecified"),
        "scores": {**scores, "total": total},
        "visual_quality": visual,
        "anchor_quality": anchor,
        "trajectory_quality": trajectory,
        "completion_value": completion,
        "flags": flags,
    }
