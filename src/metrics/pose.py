"""Estimator-cache loading, SE(3) comparison, and constrained event matching."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.spatial.transform import Rotation

from src.data.canonical import validate_se3_poses


@dataclass(frozen=True)
class PoseCache:
    poses_c2w: np.ndarray
    frame_indices: np.ndarray
    provenance: dict[str, object]
    # Optional per-pose pinhole matrices expressed in the decoded video's
    # pixel coordinates.  DA3 predicts these jointly with its camera track;
    # without them the case calibration is used.
    intrinsics: np.ndarray | None = None

    def subset(self, selection: np.ndarray) -> "PoseCache":
        """Select synchronized pose-cache rows without dropping intrinsics."""

        selected = np.asarray(selection)
        return PoseCache(
            poses_c2w=self.poses_c2w[selected],
            frame_indices=self.frame_indices[selected],
            provenance=self.provenance,
            intrinsics=(None if self.intrinsics is None else self.intrinsics[selected]),
        )


def _validate_intrinsics(value: np.ndarray, *, pose_count: int) -> np.ndarray:
    intrinsics = np.asarray(value, dtype=np.float64)
    if intrinsics.shape == (pose_count, 4):
        matrices = np.repeat(np.eye(3, dtype=np.float64)[None], pose_count, axis=0)
        matrices[:, 0, 0] = intrinsics[:, 0]
        matrices[:, 1, 1] = intrinsics[:, 1]
        matrices[:, 0, 2] = intrinsics[:, 2]
        matrices[:, 1, 2] = intrinsics[:, 3]
        intrinsics = matrices
    if intrinsics.shape != (pose_count, 3, 3):
        raise ValueError("pose intrinsics must be Nx3x3 or Nx4 and match poses")
    if not np.isfinite(intrinsics).all():
        raise ValueError("pose intrinsics must be finite")
    if np.any(intrinsics[:, 0, 0] <= 0.0) or np.any(intrinsics[:, 1, 1] <= 0.0):
        raise ValueError("pose intrinsics focal lengths must be positive")
    expected_last_row = np.broadcast_to(
        np.asarray([0.0, 0.0, 1.0], dtype=np.float64), (pose_count, 3)
    )
    if not np.allclose(intrinsics[:, 2, :], expected_last_row, atol=1e-8):
        raise ValueError("pose intrinsics must be pinhole camera matrices")
    return intrinsics


def load_pose_cache(path: Path | str | None, *, video_frame_count: int) -> PoseCache | None:
    if path is None:
        return None
    path = Path(path)
    if path.is_dir():
        pose_path = path / "poses_c2w.npy"
        index_path = path / "frame_indices.npy"
        intrinsics_path = path / "intrinsics.npy"
        provenance_path = path / "provenance.json"
        vipe_archives = sorted((path / "pose").glob("*.npz"))
        if pose_path.is_file():
            poses = np.load(pose_path, allow_pickle=False)
            indices = np.load(index_path, allow_pickle=False) if index_path.is_file() else np.arange(len(poses))
        elif len(vipe_archives) == 1:
            archive = np.load(vipe_archives[0], allow_pickle=False)
            if "data" not in archive:
                raise ValueError("official ViPE pose archive needs data poses")
            poses = archive["data"]
            indices = archive["inds"] if "inds" in archive else np.arange(len(poses))
        elif vipe_archives:
            pose_chunks, index_chunks = [], []
            for archive_path in vipe_archives:
                archive = np.load(archive_path, allow_pickle=False)
                if "data" not in archive:
                    raise ValueError(f"official ViPE pose archive lacks data: {archive_path}")
                pose_chunks.append(archive["data"])
                index_chunks.append(archive["inds"] if "inds" in archive else np.arange(len(archive["data"])))
            poses = np.concatenate(pose_chunks)
            indices = np.concatenate(index_chunks)
        else:
            raise FileNotFoundError(f"expected {pose_path} or official ViPE pose/*.npz")
        provenance: dict[str, object] = {}
        if provenance_path.is_file():
            import json
            value = json.loads(provenance_path.read_text(encoding="utf-8"))
            if isinstance(value, dict):
                provenance = value
        intrinsics = (
            np.load(intrinsics_path, allow_pickle=False)
            if intrinsics_path.is_file()
            else None
        )
    elif path.suffix == ".npz":
        archive = np.load(path, allow_pickle=False)
        pose_key = next((key for key in ("poses_c2w", "poses_c2w_opencv_m", "data") if key in archive), None)
        if pose_key not in archive:
            raise ValueError("pose npz needs poses_c2w, poses_c2w_opencv_m, or official ViPE data")
        poses = archive[pose_key]
        indices = archive["frame_indices"] if "frame_indices" in archive else (archive["inds"] if "inds" in archive else np.arange(len(poses)))
        intrinsics = archive["intrinsics"] if "intrinsics" in archive else None
        provenance = {"source": str(path.resolve()), "format": "npz"}
    else:
        poses = np.load(path, allow_pickle=False)
        indices = np.arange(len(poses))
        intrinsics = None
        provenance = {"source": str(path.resolve()), "format": "npy"}
    poses = validate_se3_poses(np.asarray(poses, dtype=np.float64), name="estimator poses")
    indices = np.asarray(indices, dtype=np.int64)
    if indices.shape != (len(poses),) or np.any(np.diff(indices) <= 0):
        raise ValueError("estimator frame_indices must be strictly increasing and match poses")
    if len(indices) != video_frame_count or not np.array_equal(indices, np.arange(video_frame_count)):
        # A sparse cache is legal, but only recorded video indices may be used.
        if np.any(indices < 0) or np.any(indices >= video_frame_count):
            raise ValueError("estimator frame_indices are outside the video")
    validated_intrinsics = (
        None
        if intrinsics is None
        else _validate_intrinsics(intrinsics, pose_count=len(poses))
    )
    return PoseCache(
        poses_c2w=poses,
        frame_indices=indices,
        provenance=provenance,
        intrinsics=validated_intrinsics,
    )


def rotation_error_deg(left: np.ndarray, right: np.ndarray) -> float:
    relative = left[:3, :3].T @ right[:3, :3]
    return float(np.rad2deg(Rotation.from_matrix(relative).magnitude()))


def pose_error(left: np.ndarray, right: np.ndarray, *, path_scale: float) -> tuple[float, float]:
    translation = float(np.linalg.norm(left[:3, 3] - right[:3, 3]) / max(path_scale, 1e-12))
    return translation, rotation_error_deg(left, right)


def fit_positive_scale(executed: np.ndarray, requested: np.ndarray) -> float:
    left = np.asarray(executed, dtype=np.float64).reshape(-1, 3)
    right = np.asarray(requested, dtype=np.float64).reshape(-1, 3)
    denominator = float(np.sum(left * left))
    if denominator <= 1e-12:
        return 1.0
    return max(float(np.sum(left * right) / denominator), 0.0)


def scale_pose_translations(poses: np.ndarray, scale: float) -> np.ndarray:
    value = validate_se3_poses(poses).copy()
    if not np.isfinite(scale) or scale < 0.0:
        raise ValueError("pose scale must be finite and non-negative")
    value[:, :3, 3] *= float(scale)
    return value


def nearest_cached_pose(
    cache: PoseCache,
    *,
    target_pose: np.ndarray,
    candidate_video_indices: Iterable[int],
    path_scale: float,
    rotation_cost_weight: float,
) -> tuple[int, float, float] | None:
    candidate_set = set(int(index) for index in candidate_video_indices)
    best: tuple[float, int, float, float] | None = None
    for pose, index in zip(cache.poses_c2w, cache.frame_indices):
        if int(index) not in candidate_set:
            continue
        translation, rotation = pose_error(pose, target_pose, path_scale=path_scale)
        score = translation + rotation_cost_weight * rotation
        candidate = (score, int(index), translation, rotation)
        if best is None or candidate < best:
            best = candidate
    if best is None:
        return None
    _, index, translation, rotation = best
    return index, translation, rotation


def umeyama_align_points(
    source: np.ndarray,
    target: np.ndarray,
    *,
    with_scale: bool,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Least-squares Umeyama alignment from ``source`` to ``target``."""

    left = np.asarray(source, dtype=np.float64)
    right = np.asarray(target, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 2 or left.shape[1] != 3:
        raise ValueError("Umeyama inputs must be equal Nx3 arrays")
    if len(left) < 2 or not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("Umeyama alignment needs at least two finite points")
    left_mean, right_mean = left.mean(axis=0), right.mean(axis=0)
    left_centered, right_centered = left - left_mean, right - right_mean
    covariance = right_centered.T @ left_centered / len(left)
    u, singular, vt = np.linalg.svd(covariance)
    correction = np.eye(3, dtype=np.float64)
    if np.linalg.det(u @ vt) < 0.0:
        correction[-1, -1] = -1.0
    rotation = u @ correction @ vt
    if with_scale:
        variance = float(np.mean(np.sum(left_centered * left_centered, axis=1)))
        scale = (
            float(np.sum(singular * np.diag(correction)) / variance)
            if variance > 1e-12
            else 1.0
        )
    else:
        scale = 1.0
    translation = right_mean - scale * (rotation @ left_mean)
    return rotation, scale, translation


def align_pose_track_umeyama(
    poses: np.ndarray,
    target: np.ndarray,
    *,
    with_scale: bool,
) -> tuple[np.ndarray, dict[str, float]]:
    """Apply one global SE(3) or Sim(3) camera-centre alignment to a track."""

    source_poses = validate_se3_poses(poses).copy()
    target_poses = validate_se3_poses(target)
    if len(source_poses) != len(target_poses):
        raise ValueError("pose tracks must have equal length")
    rotation, scale, translation = umeyama_align_points(
        source_poses[:, :3, 3], target_poses[:, :3, 3], with_scale=with_scale
    )
    source_poses[:, :3, 3] = (
        scale * (rotation @ source_poses[:, :3, 3].T).T + translation
    )
    source_poses[:, :3, :3] = rotation[None] @ source_poses[:, :3, :3]
    return source_poses, {
        "scale": float(scale),
        "rotation_deg": float(np.rad2deg(Rotation.from_matrix(rotation).magnitude())),
        "translation_norm": float(np.linalg.norm(translation)),
    }


def pose_track_error_summary(
    predicted: np.ndarray,
    target: np.ndarray,
) -> dict[str, float]:
    """Absolute position and orientation errors for already aligned tracks."""

    left = validate_se3_poses(predicted)
    right = validate_se3_poses(target)
    if len(left) != len(right):
        raise ValueError("pose tracks must have equal length")
    translation = np.linalg.norm(left[:, :3, 3] - right[:, :3, 3], axis=1)
    rotation = np.asarray(
        [rotation_error_deg(one, two) for one, two in zip(left, right)],
        dtype=np.float64,
    )
    return {
        "translation_rmse_m": float(np.sqrt(np.mean(translation**2))),
        "translation_mean_m": float(translation.mean()),
        "translation_median_m": float(np.median(translation)),
        "translation_p95_m": float(np.percentile(translation, 95)),
        "rotation_mean_deg": float(rotation.mean()),
        "rotation_median_deg": float(np.median(rotation)),
        "rotation_p95_deg": float(np.percentile(rotation, 95)),
    }


