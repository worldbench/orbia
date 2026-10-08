"""Camera-convention helpers shared by camera-conditioned adapters."""
from __future__ import annotations

import numpy as np

# OpenCV (x right, y down, z forward) -> OpenGL/Blender (x right, y up, z backward).
OPENCV_TO_OPENGL = np.diag([1.0, -1.0, -1.0, 1.0])


def rebase(poses_c2w: np.ndarray, index: int = 0) -> np.ndarray:
    """Express poses relative to ``poses_c2w[index]``."""
    poses = np.asarray(poses_c2w, dtype=np.float64)
    return np.linalg.inv(poses[index]) @ poses


def c2w_to_w2c(poses_c2w: np.ndarray) -> np.ndarray:
    return np.linalg.inv(np.asarray(poses_c2w, dtype=np.float64))


def opencv_to_opengl(poses_c2w: np.ndarray) -> np.ndarray:
    """Change the camera axes of c2w poses from OpenCV to OpenGL."""
    return np.asarray(poses_c2w, dtype=np.float64) @ OPENCV_TO_OPENGL


def local_increments(poses_c2w: np.ndarray, step: int = 1) -> np.ndarray:
    """Relative motion ``inv(P[t]) @ P[t+step]`` in the current camera frame."""
    poses = np.asarray(poses_c2w, dtype=np.float64)
    return np.linalg.inv(poses[:-step]) @ poses[step:]


def rotation_6d(rotations: np.ndarray) -> np.ndarray:
    """Continuous 6D rotation representation (first two matrix columns)."""
    r = np.asarray(rotations, dtype=np.float64)
    return np.concatenate([r[..., :, 0], r[..., :, 1]], axis=-1)


def normalize_translation(poses_c2w: np.ndarray, target: float = 1.0) -> tuple[np.ndarray, float]:
    """Scale camera centres so the largest distance from the first camera equals ``target``."""
    poses = np.asarray(poses_c2w, dtype=np.float64).copy()
    extent = float(np.linalg.norm(poses[:, :3, 3] - poses[0, :3, 3], axis=1).max())
    scale = target / extent if extent > 1e-12 else 1.0
    poses[:, :3, 3] = poses[0, :3, 3] + (poses[:, :3, 3] - poses[0, :3, 3]) * scale
    return poses, scale


def scale_intrinsics(K: np.ndarray, source_wh: tuple[int, int], target_wh: tuple[int, int]) -> np.ndarray:
    """Intrinsics after resizing an image from ``source_wh`` to ``target_wh``."""
    K = np.asarray(K, dtype=np.float64).copy()
    sx, sy = target_wh[0] / source_wh[0], target_wh[1] / source_wh[1]
    K[0] *= sx
    K[1] *= sy
    return K


def crop_intrinsics(K: np.ndarray, left: float, top: float) -> np.ndarray:
    """Intrinsics after cropping ``left``/``top`` pixels from the image."""
    K = np.asarray(K, dtype=np.float64).copy()
    K[0, 2] -= left
    K[1, 2] -= top
    return K


def normalized_intrinsics(K: np.ndarray, width: int, height: int) -> np.ndarray:
    """``[fx/W, fy/H, cx/W, cy/H]``."""
    K = np.asarray(K, dtype=np.float64)
    return np.array([K[0, 0] / width, K[1, 1] / height, K[0, 2] / width, K[1, 2] / height])


def resize_and_crop(image: np.ndarray, K: np.ndarray, size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Resize to cover ``size=(W, H)`` and centre-crop, updating ``K`` consistently."""
    from PIL import Image

    h, w = image.shape[:2]
    tw, th = size
    scale = max(tw / w, th / h)
    rw, rh = int(round(w * scale)), int(round(h * scale))
    resized = np.asarray(Image.fromarray(image).resize((rw, rh), Image.BICUBIC))
    left, top = (rw - tw) // 2, (rh - th) // 2
    K = crop_intrinsics(scale_intrinsics(K, (w, h), (rw, rh)), left, top)
    return resized[top:top + th, left:left + tw], K


def pad_trajectory(poses_c2w: np.ndarray, length: int) -> np.ndarray:
    """Repeat the terminal pose until the trajectory has ``length`` poses."""
    poses = np.asarray(poses_c2w, dtype=np.float64)
    if length <= len(poses):
        return poses[:length].copy()
    tail = np.repeat(poses[-1:], length - len(poses), axis=0)
    return np.concatenate([poses, tail], axis=0)


def windows(num_frames: int, window: int, overlap: int = 1) -> list[tuple[int, int]]:
    """Split ``[0, num_frames)`` into windows that share ``overlap`` boundary frames."""
    if window <= overlap:
        raise ValueError('window must be larger than overlap')
    spans, start = [], 0
    while True:
        spans.append((start, start + window))
        if start + window >= num_frames:
            return spans
        start += window - overlap


def plucker_rays(K: np.ndarray, poses_c2w: np.ndarray, width: int, height: int) -> np.ndarray:
    """Per-pixel Plücker embeddings ``[N, H, W, 6]`` (direction, moment)."""
    K = np.asarray(K, dtype=np.float64)
    poses = np.asarray(poses_c2w, dtype=np.float64)
    u, v = np.meshgrid(np.arange(width) + 0.5, np.arange(height) + 0.5)
    pixels = np.stack([u, v, np.ones_like(u)], axis=-1) @ np.linalg.inv(K).T
    directions = np.einsum('nij,hwj->nhwi', poses[:, :3, :3], pixels)
    directions /= np.linalg.norm(directions, axis=-1, keepdims=True)
    origins = np.broadcast_to(poses[:, None, None, :3, 3], directions.shape)
    return np.concatenate([directions, np.cross(origins, directions)], axis=-1)
