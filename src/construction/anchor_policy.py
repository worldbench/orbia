"""Endpoint anchor mask integrity policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import cv2
import numpy as np


@dataclass(frozen=True)
class StrictAnchorThresholds:
    minimum_endpoint_area_percent: float = 0.3
    maximum_endpoint_area_percent: float = 50.0
    minimum_tracking_iou: float = 0.8
    significant_component_fraction: float = 0.002
    minimum_significant_component_pixels: int = 16
    maximum_significant_component_count: int = 1
    minimum_largest_component_ratio: float = 0.85
    maximum_internal_gap_ratio: float = 0.20


STRICT_ANCHOR_THRESHOLDS = StrictAnchorThresholds()


def _finite_float(value: object, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if np.isfinite(result) else default


def endpoint_internal_gap_ratio(integrity: Mapping[str, Any]) -> float:
    """Return the worst available endpoint interior-defect measurement."""

    return max(
        _finite_float(integrity.get("hole_ratio")),
        _finite_float(integrity.get("closing_gap_ratio")),
        _finite_float(integrity.get("enclosed_hole_ratio")),
        _finite_float(integrity.get("internal_gap_ratio")),
    )


def measure_endpoint_mask(
    mask: np.ndarray,
    *,
    thresholds: StrictAnchorThresholds = STRICT_ANCHOR_THRESHOLDS,
) -> dict[str, Any]:
    """Measure hard-gate integrity and soft-ranking geometry from one mask."""

    binary = np.asarray(mask, dtype=bool)
    area = int(np.count_nonzero(binary))
    if binary.ndim != 2 or area == 0:
        return {
            "available": False,
            "component_count": 0,
            "significant_component_count": 0,
            "significant_component_count_0_2pct": 0,
            "largest_component_ratio": 0.0,
            "hole_ratio": 0.0,
            "closing_gap_ratio": 0.0,
            "enclosed_hole_ratio": 0.0,
            "internal_gap_ratio": 0.0,
            "touches_image_border": False,
            "border_contact_pixel_ratio": 0.0,
        }

    height, width = binary.shape
    ys, xs = np.nonzero(binary)
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    box_width, box_height = x1 - x0, y1 - y0

    count, _, stats, _ = cv2.connectedComponentsWithStats(
        binary.astype(np.uint8), connectivity=8
    )
    component_areas = [int(value) for value in stats[1:, cv2.CC_STAT_AREA]]
    significant_floor = max(
        thresholds.minimum_significant_component_pixels,
        int(np.ceil(thresholds.significant_component_fraction * area)),
    )
    significant_count = sum(value >= significant_floor for value in component_areas)
    largest_component_ratio = max(component_areas, default=0) / max(area, 1)
    detached_fraction = 1.0 - largest_component_ratio

    contours, _ = cv2.findContours(
        binary.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    filled = np.zeros_like(binary, dtype=np.uint8)
    cv2.drawContours(filled, contours, -1, 1, thickness=cv2.FILLED)
    filled_area = int(np.count_nonzero(filled))
    enclosed_hole_ratio = max(0, filled_area - area) / max(filled_area, 1)

    kernel_size = min(51, max(3, int(round(min(box_width, box_height) * 0.05))))
    if kernel_size % 2 == 0:
        kernel_size += 1
    crop = binary[y0:y1, x0:x1].astype(np.uint8)
    pad = kernel_size
    padded = np.pad(crop, pad, mode="constant")
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    closed = cv2.morphologyEx(padded, cv2.MORPH_CLOSE, kernel)
    closed = closed[pad : pad + box_height, pad : pad + box_width].astype(bool)
    closed_area = int(np.count_nonzero(closed))
    closing_gap_ratio = int(np.count_nonzero(closed & ~crop.astype(bool))) / max(
        closed_area, 1
    )
    internal_gap_ratio = max(enclosed_hole_ratio, closing_gap_ratio)

    border = np.zeros_like(binary, dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    border_contact_pixels = int(np.count_nonzero(binary & border))

    aspect = max(box_width / max(box_height, 1), box_height / max(box_width, 1))
    hull_points = np.concatenate(contours, axis=0)
    hull = cv2.convexHull(hull_points)
    hull_area = max(float(cv2.contourArea(hull)), 1.0)
    solidity = min(float(area) / hull_area, 1.0)
    centroid_x = float(xs.mean() / max(width - 1, 1))
    centroid_y = float(ys.mean() / max(height - 1, 1))
    margin_x = min(centroid_x, 1.0 - centroid_x)
    margin_y = min(centroid_y, 1.0 - centroid_y)
    if margin_x < 0.20 and margin_y < 0.20:
        position_factor = 0.65
    elif margin_x < 0.30 and margin_y < 0.30:
        position_factor = 0.80
    elif min(margin_x, margin_y) < 0.15:
        position_factor = 0.82
    else:
        position_factor = 1.0

    return {
        "available": True,
        "integrity_policy": "strict_anchor_v1",
        "significant_component_fraction": thresholds.significant_component_fraction,
        "component_count": max(0, count - 1),
        "significant_component_count": significant_count,
        "significant_component_count_0_2pct": significant_count,
        "significant_component_floor_pixels": significant_floor,
        "largest_component_ratio": round(largest_component_ratio, 6),
        "detached_component_fraction": round(detached_fraction, 3),
        "hole_ratio": round(enclosed_hole_ratio, 6),
        "closing_gap_ratio": round(closing_gap_ratio, 3),
        "enclosed_hole_ratio": round(enclosed_hole_ratio, 3),
        "internal_gap_ratio": round(internal_gap_ratio, 3),
        "touches_image_border": border_contact_pixels > 0,
        "border_contact_pixel_ratio": round(border_contact_pixels / area, 6),
        # The fields below affect ranking only and are not strict gates.
        "bbox_aspect_ratio": round(aspect, 3),
        "solidity": round(solidity, 3),
        "centroid_x": round(centroid_x, 3),
        "centroid_y": round(centroid_y, 3),
        "position_factor": round(position_factor, 3),
    }


def strict_anchor_policy_failures(
    *,
    identity_confirmed: bool,
    q0_tracking_iou: float,
    qe_tracking_iou: float,
    q0_area_percent: float,
    qe_area_percent: float,
    q0_integrity: Mapping[str, Any] | None,
    qe_integrity: Mapping[str, Any] | None,
    thresholds: StrictAnchorThresholds = STRICT_ANCHOR_THRESHOLDS,
) -> list[str]:
    """Return the strict policy failures of one anchor."""

    failures: list[str] = []
    if not identity_confirmed:
        failures.append("identity_missing")
    elif min(q0_tracking_iou, qe_tracking_iou) < thresholds.minimum_tracking_iou:
        failures.append("tracking_iou_below_minimum")

    if min(q0_area_percent, qe_area_percent) < thresholds.minimum_endpoint_area_percent:
        failures.append("endpoint_area_below_minimum")
    if max(q0_area_percent, qe_area_percent) > thresholds.maximum_endpoint_area_percent:
        failures.append("endpoint_area_above_maximum")

    for role, integrity in (("q0", q0_integrity), ("qe", qe_integrity)):
        if not isinstance(integrity, Mapping):
            continue
        significant = int(
            integrity.get(
                "significant_component_count_0_2pct",
                integrity.get("significant_component_count", 1),
            )
        )
        if significant > thresholds.maximum_significant_component_count:
            failures.append(f"{role}_multiple_significant_components")
        if _finite_float(integrity.get("largest_component_ratio"), 1.0) < (
            thresholds.minimum_largest_component_ratio
        ):
            failures.append(f"{role}_largest_component_below_minimum")
        if endpoint_internal_gap_ratio(integrity) > thresholds.maximum_internal_gap_ratio:
            failures.append(f"{role}_internal_gap_above_maximum")
    return failures
