"""Anchor-local bidirectional learned-depth consistency diagnostics."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from src.utils.camera import CameraIntrinsics


def _resize_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    value = np.asarray(mask, dtype=np.uint8)
    if value.shape != shape:
        value = cv2.resize(
            value, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST
        )
    return value.astype(bool)


def _directional_consistency(
    source_depth: np.ndarray,
    target_depth: np.ndarray,
    source_camera: CameraIntrinsics,
    target_camera: CameraIntrinsics,
    source_pose: np.ndarray,
    target_pose: np.ndarray,
    source_mask: np.ndarray,
    target_mask: np.ndarray,
    *,
    stride: int,
) -> dict[str, Any]:
    source = np.asarray(source_depth, dtype=np.float64)
    target = np.asarray(target_depth, dtype=np.float64)
    source_mask = _resize_mask(source_mask, source.shape)
    target_mask = _resize_mask(target_mask, target.shape)
    ys, xs = np.nonzero(source_mask & np.isfinite(source) & (source > 0.0))
    if stride > 1:
        ys, xs = ys[::stride], xs[::stride]
    result: dict[str, Any] = {
        "source_mask_pixel_count": int(np.count_nonzero(source_mask)),
        "source_valid_depth_count": int(len(xs)),
        "projected_in_frame_count": 0,
        "projected_inside_target_mask_fraction": 0.0,
        "target_valid_depth_count": 0,
        "local_depth_scale_ratio": None,
        "median_normalized_depth_residual": None,
        "depth_residual_within_0_25_fraction": None,
    }
    if not len(xs):
        return result
    pixels = np.stack([xs + 0.5, ys + 0.5], axis=1)
    points_camera = source_camera.normalized_rays(pixels) * source[ys, xs, None]
    source_pose = np.asarray(source_pose, dtype=np.float64)
    target_pose = np.asarray(target_pose, dtype=np.float64)
    points_world = points_camera @ source_pose[:3, :3].T + source_pose[:3, 3]
    points_target = (points_world - target_pose[:3, 3]) @ target_pose[:3, :3]
    positive = points_target[:, 2] > 1.0e-6
    projected = np.full((len(points_target), 2), np.nan, dtype=np.float64)
    projected[positive] = target_camera.project_camera_points(points_target[positive])
    tx = np.zeros(len(projected), dtype=np.int64)
    ty = np.zeros(len(projected), dtype=np.int64)
    tx[positive] = np.rint(projected[positive, 0] - 0.5).astype(np.int64)
    ty[positive] = np.rint(projected[positive, 1] - 0.5).astype(np.int64)
    inside = (
        positive
        & (tx >= 0) & (tx < target.shape[1])
        & (ty >= 0) & (ty < target.shape[0])
    )
    result["projected_in_frame_count"] = int(np.count_nonzero(inside))
    if not np.any(inside):
        return result
    inside_indices = np.flatnonzero(inside)
    mask_hits = target_mask[ty[inside], tx[inside]]
    result["projected_inside_target_mask_fraction"] = float(np.mean(mask_hits))
    comparable_indices = inside_indices[
        mask_hits
        & np.isfinite(target[ty[inside], tx[inside]])
        & (target[ty[inside], tx[inside]] > 0.0)
    ]
    result["target_valid_depth_count"] = int(len(comparable_indices))
    if not len(comparable_indices):
        return result
    predicted = points_target[comparable_indices, 2]
    observed = target[ty[comparable_indices], tx[comparable_indices]]
    scale = float(np.median(observed / predicted))
    residual = np.abs(scale * predicted - observed) / np.maximum(observed, 1.0e-6)
    result["local_depth_scale_ratio"] = scale
    result["median_normalized_depth_residual"] = float(np.median(residual))
    result["depth_residual_within_0_25_fraction"] = float(np.mean(residual <= 0.25))
    return result


def anchor_bidirectional_depth_consistency(
    q0_depth: np.ndarray,
    qe_depth: np.ndarray,
    q0_camera: CameraIntrinsics,
    qe_camera: CameraIntrinsics,
    q0_pose_world_camera: np.ndarray,
    qe_pose_world_camera: np.ndarray,
    q0_mask: np.ndarray,
    qe_mask: np.ndarray,
    *,
    stride: int = 2,
    minimum_valid_depth_pixels: int = 32,
) -> dict[str, Any]:
    """Bidirectional anchor depth consistency, used as a ranking diagnostic."""

    q0_to_qe = _directional_consistency(
        q0_depth, qe_depth, q0_camera, qe_camera,
        q0_pose_world_camera, qe_pose_world_camera, q0_mask, qe_mask,
        stride=stride,
    )
    qe_to_q0 = _directional_consistency(
        qe_depth, q0_depth, qe_camera, q0_camera,
        qe_pose_world_camera, q0_pose_world_camera, qe_mask, q0_mask,
        stride=stride,
    )
    directions = (q0_to_qe, qe_to_q0)
    if any(row["target_valid_depth_count"] < minimum_valid_depth_pixels for row in directions):
        grade = "insufficient_support"
    elif all(
        row["projected_inside_target_mask_fraction"] >= 0.50
        and row["median_normalized_depth_residual"] is not None
        and row["median_normalized_depth_residual"] <= 0.25
        for row in directions
    ):
        grade = "consistent"
    else:
        grade = "risk"
    return {
        "gating": False,
        "grade": grade,
        "depth_source": "learned_depth",
        "scale_policy": "per_direction_anchor_local_median_alignment",
        "q0_to_qe": q0_to_qe,
        "qe_to_q0": qe_to_q0,
    }
