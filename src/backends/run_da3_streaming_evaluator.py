from __future__ import annotations


from datetime import datetime, timezone

import json

import os

from pathlib import Path

import re

import shutil





from typing import Any, Mapping, Sequence

import numpy as np


CONTRACT = "orbia.da3_geometry_case_contract.v1"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)

def _clear_contract_files(output: Path) -> tuple[Path, Path]:
    cache = output / "evaluator_pose_cache"
    depth = output / "depth"
    cache.mkdir(parents=True, exist_ok=True)
    depth.mkdir(parents=True, exist_ok=True)
    for path in (
        cache / "poses_c2w.npy",
        cache / "frame_indices.npy",
        cache / "intrinsics.npy",
        cache / "provenance.json",
        output / "complete.json",
    ):
        path.unlink(missing_ok=True)
    for path in depth.glob("frame_*.npz"):
        path.unlink()
    return cache, depth

def _normalize_to_first(poses_c2w: np.ndarray) -> np.ndarray:
    poses = np.asarray(poses_c2w, dtype=np.float64)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4) or not np.isfinite(poses).all():
        raise ValueError("DA3 camera poses must be finite Nx4x4 matrices")
    # DA3-Streaming aligns chunks with Sim(3); its exported 3x3 blocks can
    # retain a small scale/shear component.  The evaluator expects SE(3),
    # so project every camera orientation to the nearest proper rotation while
    # retaining the already globally aligned translation.
    rigid = poses.copy()
    for index, matrix in enumerate(rigid):
        left, _singular, right = np.linalg.svd(matrix[:3, :3])
        rotation = left @ right
        if np.linalg.det(rotation) < 0.0:
            left[:, -1] *= -1.0
            rotation = left @ right
        rigid[index, :3, :3] = rotation
        rigid[index, 3, :] = np.asarray([0.0, 0.0, 0.0, 1.0])
    normalized = np.linalg.inv(rigid[0])[None] @ rigid
    normalized[:, 3, :] = np.asarray([0.0, 0.0, 0.0, 1.0])
    if not np.allclose(normalized[0], np.eye(4), atol=1e-6):
        raise ValueError("failed to normalize DA3 poses to the first frame")
    return normalized

def _video_space_intrinsics(
    intrinsic: np.ndarray,
    *,
    processed_width: int,
    processed_height: int,
    video_width: int,
    video_height: int,
) -> np.ndarray:
    value = np.asarray(intrinsic, dtype=np.float64)
    if value.shape != (3, 3) or not np.isfinite(value).all():
        raise ValueError("DA3 intrinsics must be finite 3x3 matrices")
    scale = np.asarray(
        [
            [video_width / processed_width, 0.0, 0.0],
            [0.0, video_height / processed_height, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return scale @ value

def _materialize_contract(
    *,
    job: Mapping[str, Any],
    output: Path,
    poses_c2w: np.ndarray,
    intrinsics_video: np.ndarray,
    source_depths: Sequence[Path],
    provenance: Mapping[str, Any],
) -> None:
    original_frame_count = int(job["frame_count"])
    indices = np.asarray(provenance.get('sampling_plan', {}).get('frame_indices', list(range(original_frame_count))), dtype=np.int64)
    frame_count = len(indices)
    if not frame_count or indices[0] != 0 or np.any(np.diff(indices) <= 0) or indices[-1] >= original_frame_count:
        raise ValueError('invalid original-video geometry indices')
    if len(poses_c2w) != frame_count:
        raise ValueError("DA3 pose output has the wrong frame count")
    if np.asarray(intrinsics_video).shape != (frame_count, 3, 3):
        raise ValueError("DA3 intrinsic output has the wrong frame count or shape")
    if len(source_depths) != frame_count:
        raise ValueError("DA3 depth output has the wrong frame count")
    cache, depth_root = _clear_contract_files(output)
    np.save(cache / "poses_c2w.npy", _normalize_to_first(poses_c2w))
    np.save(cache / "frame_indices.npy", indices)
    np.save(cache / "intrinsics.npy", np.asarray(intrinsics_video, dtype=np.float64))
    for index, source in zip(indices, source_depths):
        destination = depth_root / f"frame_{index:06d}.npz"
        shutil.copy2(source, destination)
    cache_provenance = {
        "format": "orbia.pose_cache.v2",
        "case_id": job["case_id"],
        "implementation": "official_da3_streaming",
        "camera_convention": "opencv_c2w_q0_normalized",
        "intrinsics_coordinate_space": "source_video_pixels",
        **dict(provenance),
    }
    _write_json(cache / "provenance.json", cache_provenance)
    _write_json(
        output / "complete.json",
        {
            "format": CONTRACT,
            "status": "ok",
            "case_id": job["case_id"],
            "created_at_utc": _now(),
            "source_video": str(Path(job["video"]).resolve()),
            "pose_frame_count": frame_count,
            "depth_frame_count": frame_count,
            "pose_cache": str(cache.resolve()),
            "depth_root": str(depth_root.resolve()),
            "backend": provenance["backend"],
        },
    )

def _indexed_depths(path: Path, frame_count: int) -> list[Path]:
    by_index: dict[int, Path] = {}
    for item in path.glob("frame_*.npz"):
        match = re.fullmatch(r"frame_(\d+)", item.stem)
        if match is None:
            continue
        index = int(match.group(1))
        if index in by_index:
            raise ValueError(f"duplicate DA3 depth frame {index}")
        by_index[index] = item
    expected = set(range(frame_count))
    if set(by_index) != expected:
        missing = sorted(expected - set(by_index))
        extra = sorted(set(by_index) - expected)
        raise ValueError(
            f"DA3 depth indices do not match video; missing={missing[:8]} extra={extra[:8]}"
        )
    return [by_index[index] for index in range(frame_count)]

def _read_real_outputs(
    raw_output: Path, *, frame_count: int, video_width: int, video_height: int,
) -> tuple[np.ndarray, np.ndarray, list[Path]]:
    poses = np.loadtxt(raw_output / "camera_poses.txt", dtype=np.float64)
    poses = np.asarray(poses).reshape(-1, 4, 4)
    intrinsic_rows = np.loadtxt(raw_output / "intrinsic.txt", dtype=np.float64)
    intrinsic_rows = np.asarray(intrinsic_rows).reshape(-1, 4)
    depths = _indexed_depths(raw_output / "results_output", frame_count)
    if len(poses) != frame_count or len(intrinsic_rows) != frame_count:
        raise ValueError("DA3 camera output does not match the expected frame count")
    intrinsics = []
    for index, (row, depth_path) in enumerate(zip(intrinsic_rows, depths)):
        with np.load(depth_path, allow_pickle=False) as archive:
            if "depth" not in archive:
                raise ValueError(f"DA3 depth archive lacks depth: {depth_path}")
            depth = np.asarray(archive["depth"])
            if depth.ndim not in (2, 3):
                raise ValueError(f"DA3 depth has invalid shape at frame {index}: {depth.shape}")
            processed_height, processed_width = depth.shape[:2]
        native_K = np.asarray(
            [[row[0], 0.0, row[2]], [0.0, row[1], row[3]], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        intrinsics.append(
            _video_space_intrinsics(
                native_K,
                processed_width=processed_width,
                processed_height=processed_height,
                video_width=video_width,
                video_height=video_height,
            )
        )
    return poses, np.stack(intrinsics), depths
