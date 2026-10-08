"""Calibrated evaluator-space image and mask transforms."""

from __future__ import annotations

import numpy as np

from src.utils.camera import CameraIntrinsics, map_camera_pixels


def _cv2():
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("OpenCV is required for evaluator transforms") from exc
    return cv2


def canonical_to_output_transform(
    *, canonical_width: int,
    canonical_height: int,
    output_width: int,
    output_height: int,
) -> np.ndarray:
    """Explicit default used when generator metadata provides no crop record."""

    return np.asarray(
        [[output_width / canonical_width, 0.0, 0.0], [0.0, output_height / canonical_height, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def warp_image(
    image: np.ndarray,
    transform: np.ndarray,
    *,
    width: int,
    height: int,
    mask: bool = False,
    nearest: bool = False,
) -> np.ndarray:
    cv2 = _cv2()
    interpolation = cv2.INTER_NEAREST if mask or nearest else cv2.INTER_LANCZOS4
    value = cv2.warpPerspective(
        np.asarray(image), np.asarray(transform, dtype=np.float64), (width, height), flags=interpolation,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )
    return value.astype(bool) if mask else value


def depth_mask_to_native_rgb(
    mask_depth: np.ndarray,
    *,
    depth_camera: CameraIntrinsics,
    rgb_camera: CameraIntrinsics,
) -> np.ndarray:
    """Forward-map a depth-space partition through calibrated co-located rays."""

    mask = np.asarray(mask_depth, dtype=bool)
    if mask.shape != (depth_camera.height, depth_camera.width):
        raise ValueError("depth mask does not match depth camera")
    rows, columns = np.nonzero(mask)
    target = np.zeros((rgb_camera.height, rgb_camera.width), dtype=np.uint8)
    if len(rows) == 0:
        return target.astype(bool)
    source_pixels = np.stack([columns + 0.5, rows + 0.5], axis=1)
    mapped = map_camera_pixels(source_pixels, depth_camera, rgb_camera)
    columns_out = np.rint(mapped[:, 0] - 0.5).astype(np.int64)
    rows_out = np.rint(mapped[:, 1] - 0.5).astype(np.int64)
    valid = (
        (columns_out >= 0) & (columns_out < rgb_camera.width) &
        (rows_out >= 0) & (rows_out < rgb_camera.height)
    )
    target[rows_out[valid], columns_out[valid]] = 1
    # A native RGB pixel covers many depth samples.  One pixel of conservative
    # dilation closes sampling holes but does not turn depth resize into a mask.
    cv2 = _cv2()
    target = cv2.dilate(target, np.ones((3, 3), dtype=np.uint8), iterations=1)
    return target.astype(bool)


def depth_to_native_rgb(
    depth: np.ndarray,
    valid: np.ndarray,
    *,
    depth_camera: CameraIntrinsics,
    rgb_camera: CameraIntrinsics,
) -> tuple[np.ndarray, np.ndarray]:
    """Forward-map metric depth into the co-located native RGB image plane."""

    values = np.asarray(depth, dtype=np.float32)
    support = np.asarray(valid, dtype=bool) & np.isfinite(values) & (values > 0)
    if values.shape != (depth_camera.height, depth_camera.width) or support.shape != values.shape:
        raise ValueError("depth assets do not match the depth camera")
    rows, columns = np.nonzero(support)
    target = np.full((rgb_camera.height, rgb_camera.width), np.inf, dtype=np.float32)
    if len(rows):
        pixels = np.stack([columns + 0.5, rows + 0.5], axis=1)
        mapped = map_camera_pixels(pixels, depth_camera, rgb_camera)
        columns_out = np.rint(mapped[:, 0] - 0.5).astype(np.int64)
        rows_out = np.rint(mapped[:, 1] - 0.5).astype(np.int64)
        inside = (
            (columns_out >= 0) & (columns_out < rgb_camera.width)
            & (rows_out >= 0) & (rows_out < rgb_camera.height)
        )
        np.minimum.at(
            target,
            (rows_out[inside], columns_out[inside]),
            values[rows[inside], columns[inside]],
        )
    output_valid = np.isfinite(target)
    target[~output_valid] = 0.0
    return target, output_valid


def qe_reference_and_mask_in_output(
    *,
    qe_rgb_native: np.ndarray,
    input_supported_depth: np.ndarray,
    qe_depth_camera: CameraIntrinsics,
    qe_rgb_camera: CameraIntrinsics,
    native_to_canonical: np.ndarray,
    canonical_to_output: np.ndarray,
    output_width: int,
    output_height: int,
) -> tuple[np.ndarray, np.ndarray]:
    native_mask = depth_mask_to_native_rgb(
        input_supported_depth, depth_camera=qe_depth_camera, rgb_camera=qe_rgb_camera
    )
    native_to_output = np.asarray(canonical_to_output) @ np.asarray(native_to_canonical)
    reference = warp_image(qe_rgb_native, native_to_output, width=output_width, height=output_height)
    mask = warp_image(native_mask.astype(np.uint8), native_to_output, width=output_width, height=output_height, mask=True)
    return reference, mask
