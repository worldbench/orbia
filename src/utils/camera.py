"""Canonical in-memory data models shared by all dataset adapters."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


FloatArray = npt.NDArray[np.floating]


def _float_matrix(value: npt.ArrayLike, shape: tuple[int, int], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


@dataclass(frozen=True)
class CameraIntrinsics:
    """Calibrated camera model and the native image extent it applies to."""

    width: int
    height: int
    K: FloatArray
    model: str = "PINHOLE"
    distortion: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("camera dimensions must be positive")
        K = _float_matrix(self.K, (3, 3), "K")
        if K[0, 0] <= 0 or K[1, 1] <= 0:
            raise ValueError("camera focal lengths must be positive")
        if not np.allclose(K[2], [0.0, 0.0, 1.0], atol=1e-8):
            raise ValueError("K must use the canonical homogeneous last row")
        model = str(self.model).upper()
        distortion = tuple(float(x) for x in self.distortion)
        if not np.isfinite(distortion).all():
            raise ValueError("camera distortion contains non-finite values")
        if model == "PINHOLE":
            if distortion:
                raise ValueError("PINHOLE camera must not have distortion coefficients")
        elif model == "OPENCV":
            if len(distortion) != 4:
                raise ValueError(
                    "OPENCV camera requires distortion (k1, k2, p1, p2)"
                )
        else:
            raise ValueError(
                f"unsupported camera model {model!r}; supported models are "
                "PINHOLE and OPENCV"
            )
        object.__setattr__(self, "K", K)
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "distortion", distortion)

    def normalized_rays(self, pixels: npt.ArrayLike) -> np.ndarray:
        """Invert calibration for ``(..., 2)`` continuous pixel coordinates."""

        pixel_array = np.asarray(pixels, dtype=np.float64)
        if pixel_array.ndim < 1 or pixel_array.shape[-1] != 2:
            raise ValueError("pixels must have shape (..., 2)")
        if not np.isfinite(pixel_array).all():
            raise ValueError("pixels contain non-finite values")
        homogeneous = np.concatenate(
            [pixel_array, np.ones(pixel_array.shape[:-1] + (1,))], axis=-1
        )
        distorted = homogeneous @ np.linalg.inv(self.K).T
        xd = distorted[..., 0] / distorted[..., 2]
        yd = distorted[..., 1] / distorted[..., 2]
        if self.model == "PINHOLE":
            x, y = xd, yd
        else:
            k1, k2, p1, p2 = self.distortion
            x = xd.copy()
            y = yd.copy()
            # Newton's method is deterministic and converges rapidly for the
            # calibrated ScanNet++ lenses.  The determinant guard turns an
            # invalid/non-invertible calibration into an explicit error.
            for _ in range(12):
                r2 = x * x + y * y
                radial = 1.0 + k1 * r2 + k2 * r2 * r2
                radial_slope = k1 + 2.0 * k2 * r2
                radial_dx = 2.0 * x * radial_slope
                radial_dy = 2.0 * y * radial_slope
                estimate_x = (
                    x * radial + 2.0 * p1 * x * y + p2 * (3.0 * x * x + y * y)
                )
                estimate_y = (
                    y * radial + p1 * (x * x + 3.0 * y * y) + 2.0 * p2 * x * y
                )
                residual_x = estimate_x - xd
                residual_y = estimate_y - yd
                j11 = radial + x * radial_dx + 2.0 * p1 * y + 6.0 * p2 * x
                j12 = x * radial_dy + 2.0 * p1 * x + 2.0 * p2 * y
                j21 = y * radial_dx + 2.0 * p1 * x + 2.0 * p2 * y
                j22 = radial + y * radial_dy + 6.0 * p1 * y + 2.0 * p2 * x
                determinant = j11 * j22 - j12 * j21
                if np.any(np.abs(determinant) < 1e-14):
                    raise ValueError("OPENCV distortion is non-invertible at a pixel")
                step_x = (j22 * residual_x - j12 * residual_y) / determinant
                step_y = (-j21 * residual_x + j11 * residual_y) / determinant
                x -= step_x
                y -= step_y
                if max(
                    float(np.max(np.abs(step_x), initial=0.0)),
                    float(np.max(np.abs(step_y), initial=0.0)),
                ) < 1e-13:
                    break
        return np.stack([x, y, np.ones_like(x)], axis=-1)

    def project_camera_points(self, points_camera: npt.ArrayLike) -> np.ndarray:
        """Project ``(..., 3)`` camera points into this camera's pixel space."""

        points = np.asarray(points_camera, dtype=np.float64)
        if points.ndim < 1 or points.shape[-1] != 3:
            raise ValueError("points_camera must have shape (..., 3)")
        if not np.isfinite(points).all():
            raise ValueError("points_camera contains non-finite values")
        z = points[..., 2]
        if np.any(np.abs(z) < 1e-12):
            raise ValueError("cannot project a point with camera-z equal to zero")
        x = points[..., 0] / z
        y = points[..., 1] / z
        if self.model == "OPENCV":
            k1, k2, p1, p2 = self.distortion
            r2 = x * x + y * y
            radial = 1.0 + k1 * r2 + k2 * r2 * r2
            distorted_x = (
                x * radial + 2.0 * p1 * x * y + p2 * (3.0 * x * x + y * y)
            )
            distorted_y = (
                y * radial + p1 * (x * x + 3.0 * y * y) + 2.0 * p2 * x * y
            )
            x, y = distorted_x, distorted_y
        normalized = np.stack([x, y, np.ones_like(x)], axis=-1)
        homogeneous = normalized @ self.K.T
        return homogeneous[..., :2] / homogeneous[..., 2:3]

    def coordinate_policy(self) -> dict[str, object]:
        """JSON-safe declaration of the coordinate/model policy."""

        distorted = self.model == "OPENCV"
        return {
            "distortion": list(self.distortion),
            "image_coordinates": "native_distorted" if distorted else "native_pinhole",
            "model": self.model,
            "pixel_sample_location": "array_index_plus_0.5",
            "projection": (
                "opencv_brown_conrady_k1_k2_p1_p2" if distorted else "pinhole"
            ),
            "ray_unprojection": (
                "inverse_opencv_brown_conrady" if distorted else "inverse_K"
            ),
            "resolution": [self.width, self.height],
        }

    def scaled(self, width: int, height: int) -> "CameraIntrinsics":
        """Return intrinsics for a resized image from the same camera model."""

        sx = width / self.width
        sy = height / self.height
        K = self.K.copy()
        K[0, :] *= sx
        K[1, :] *= sy
        return CameraIntrinsics(
            width=width,
            height=height,
            K=K,
            model=self.model,
            distortion=self.distortion,
        )


def map_camera_pixels(
    pixels: npt.ArrayLike,
    source_camera: CameraIntrinsics,
    target_camera: CameraIntrinsics,
) -> np.ndarray:
    """Map co-located camera pixels through their calibrated ray models."""

    pixel_array = np.asarray(pixels, dtype=np.float64)
    if pixel_array.ndim < 1 or pixel_array.shape[-1] != 2:
        raise ValueError("pixels must have shape (..., 2)")
    if not np.isfinite(pixel_array).all():
        raise ValueError("pixels contain non-finite values")
    if (
        source_camera.model == target_camera.model
        and source_camera.distortion == target_camera.distortion
    ):
        # The same normalized distortion is applied on both sides, so it
        # cancels exactly. This is the ScanNet++ scaled registered-depth case
        # and avoids expanding a full native-RGB Newton solve in memory.
        homogeneous = np.concatenate(
            [pixel_array, np.ones(pixel_array.shape[:-1] + (1,))], axis=-1
        )
        mapped = (
            homogeneous
            @ np.linalg.inv(source_camera.K).T
            @ target_camera.K.T
        )
        return mapped[..., :2] / mapped[..., 2:3]
    rays = source_camera.normalized_rays(pixel_array)
    return target_camera.project_camera_points(rays)


