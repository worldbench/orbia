"""Canonical public camera stream transforms shared by model adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from src.utils.camera import CameraIntrinsics


CANONICAL_WIDTH = 1280
CANONICAL_HEIGHT = 720
SE3_HOMOGENEOUS_ATOL = 1e-9
SE3_ROTATION_ATOL = 1e-6


@dataclass(frozen=True)
class CanonicalCamera:
    """The exact model-visible centre crop, resize, and calibration."""

    camera: CameraIntrinsics
    provenance: Mapping[str, object]


def validate_se3_poses(poses: np.ndarray, *, name: str = "poses") -> np.ndarray:
    """Return float64 poses after strictly validating proper finite SE(3)."""

    value = np.asarray(poses, dtype=np.float64)
    if value.ndim != 3 or value.shape[1:] != (4, 4) or len(value) < 1:
        raise ValueError(f"{name} must have shape (N,4,4) with N >= 1")
    if not np.isfinite(value).all():
        raise ValueError(f"{name} must be finite")
    if not np.allclose(
        value[:, 3, :],
        np.asarray([0.0, 0.0, 0.0, 1.0]),
        rtol=0.0,
        atol=SE3_HOMOGENEOUS_ATOL,
    ):
        raise ValueError(f"{name} must have homogeneous [0,0,0,1] rows")
    rotations = value[:, :3, :3]
    gram = np.swapaxes(rotations, 1, 2) @ rotations
    if not np.allclose(
        gram, np.eye(3, dtype=np.float64)[None], rtol=0.0, atol=SE3_ROTATION_ATOL
    ):
        raise ValueError(f"{name} rotations must be orthonormal")
    if not np.allclose(
        np.linalg.det(rotations), 1.0, rtol=0.0, atol=SE3_ROTATION_ATOL
    ):
        raise ValueError(f"{name} rotations must be proper (determinant +1)")
    return value


def canonicalize_camera(camera: CameraIntrinsics) -> CanonicalCamera:
    """Transform calibration through the exact public 16:9 model crop."""

    source_width, source_height = camera.width, camera.height
    target_aspect = CANONICAL_WIDTH / CANONICAL_HEIGHT
    source_aspect = source_width / source_height
    if source_aspect >= target_aspect:
        crop_height = source_height
        crop_width = min(source_width, int(round(crop_height * target_aspect)))
        crop_left = (source_width - crop_width) // 2
        crop_top = 0
    else:
        crop_width = source_width
        crop_height = min(source_height, int(round(crop_width / target_aspect)))
        crop_left = 0
        crop_top = (source_height - crop_height) // 2
    crop_right = crop_left + crop_width
    crop_bottom = crop_top + crop_height
    scale_x = CANONICAL_WIDTH / crop_width
    scale_y = CANONICAL_HEIGHT / crop_height
    pixel_transform = np.asarray(
        [
            [scale_x, 0.0, -scale_x * crop_left],
            [0.0, scale_y, -scale_y * crop_top],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    target_camera = CameraIntrinsics(
        width=CANONICAL_WIDTH,
        height=CANONICAL_HEIGHT,
        K=pixel_transform @ camera.K,
        model=camera.model,
        distortion=camera.distortion,
    )
    return CanonicalCamera(
        camera=target_camera,
        provenance={
            "crop_xyxy_source_pixels": [crop_left, crop_top, crop_right, crop_bottom],
            "operation_order": ["center_crop_to_16:9", "resize_lanczos"],
            "pixel_transform_source_to_canonical": pixel_transform.tolist(),
            "source_resolution": [source_width, source_height],
            "target_resolution": [CANONICAL_WIDTH, CANONICAL_HEIGHT],
        },
    )


