"""Streaming MP4 validation and deterministic canonical-frame mapping."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

# A worker owns at most one active video. Consumers receive views of uint8 RGB
# frames; model-specific preprocessing remains in each consumer.
_ACTIVE_VIDEO = {}


def cached_video_frames(path):
    value = _ACTIVE_VIDEO.get(str(Path(path).resolve()))
    return None if value is None else value[1]


def decode_video_cached(path, expected_frames):
    path = Path(path).resolve()
    _ACTIVE_VIDEO.clear()
    cv2 = _cv2()
    capture = cv2.VideoCapture(str(path))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    declared = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()
    if len(frames) != expected_frames or not frames or not np.isfinite(fps) or fps <= 0:
        raise ValueError(f"invalid video: decoded {len(frames)}/{expected_frames} frames: {path}")
    height, width = frames[0].shape[:2]
    if any(frame.shape != frames[0].shape for frame in frames):
        raise ValueError(f"video changes resolution: {path}")
    info = VideoInfo(path, len(frames), fps, width, height, len(frames) / max(declared, 1))
    _ACTIVE_VIDEO[str(path)] = (info, frames)
    return info, frames


def clear_video_cache():
    _ACTIVE_VIDEO.clear()


def _cv2():
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("OpenCV is required for ORBIA video evaluation") from exc
    return cv2


@dataclass(frozen=True)
class VideoInfo:
    path: Path
    frame_count: int
    fps: float
    width: int
    height: int
    decoded_frame_ratio: float

    def canonical_to_video_index(self, canonical_index: int, canonical_count: int) -> int:
        if not 0 <= canonical_index < canonical_count:
            raise ValueError("canonical index is outside its frame range")
        if self.frame_count <= 0:
            raise ValueError("cannot map into an empty video")
        return int(round(canonical_index * (self.frame_count - 1) / max(canonical_count - 1, 1)))

    def timestamp_to_video_index(self, timestamp_s: float) -> int | None:
        """Map a requested timestamp without stretching a short output."""

        index = int(round(float(timestamp_s) * self.fps))
        if not 0 <= index < self.frame_count:
            return None
        if abs(index / self.fps - float(timestamp_s)) > 0.51 / self.fps:
            return None
        return index


def inspect_video(path: Path | str) -> VideoInfo:
    path = Path(path)
    cached = _ACTIVE_VIDEO.get(str(path.resolve()))
    if cached is not None:
        return cached[0]
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(path)
    cv2 = _cv2()
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"cannot decode video {path}")
    declared = int(round(float(capture.get(cv2.CAP_PROP_FRAME_COUNT))))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(round(float(capture.get(cv2.CAP_PROP_FRAME_WIDTH))))
    height = int(round(float(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    decoded = 0
    while True:
        ok, _ = capture.read()
        if not ok:
            break
        decoded += 1
    capture.release()
    if decoded == 0 or width <= 0 or height <= 0 or not np.isfinite(fps) or fps <= 0:
        raise ValueError(f"video metadata/decode is invalid: {path}")
    return VideoInfo(
        path=path.resolve(),
        frame_count=decoded,
        fps=fps,
        width=width,
        height=height,
        decoded_frame_ratio=float(decoded / max(declared, 1)),
    )


def read_video_frames(info: VideoInfo, indices: Iterable[int]) -> dict[int, np.ndarray]:
    """Decode requested unique frames in one sequential pass, as RGB uint8."""

    requested = sorted(set(int(index) for index in indices))
    if any(index < 0 or index >= info.frame_count for index in requested):
        raise ValueError("requested video frame index is outside decoded video")
    if not requested:
        return {}
    cached = cached_video_frames(info.path)
    if cached is not None:
        return {index: cached[index] for index in requested}
    cv2 = _cv2()
    capture = cv2.VideoCapture(str(info.path))
    values: dict[int, np.ndarray] = {}
    target_position = 0
    index = 0
    while target_position < len(requested):
        ok, frame = capture.read()
        if not ok:
            break
        if index == requested[target_position]:
            values[index] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            target_position += 1
        index += 1
    capture.release()
    missing = sorted(set(requested) - set(values))
    if missing:
        raise ValueError(f"video stopped before requested frames: {missing[:8]}")
    return values
