"""Profiles and construction helpers for variable-duration ORBIA routes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
from scipy.spatial.transform import Rotation



@dataclass(frozen=True)
class HorizonProfile:
    """Public sampling profile."""

    name: str
    frame_count: int
    fps: int = 16

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("horizon profile name must be non-empty")
        if self.frame_count < 2 or self.fps <= 0:
            raise ValueError("horizon profile requires at least two frames and positive fps")

    @property
    def duration_s(self) -> float:
        return float(self.frame_count) / float(self.fps)

    @property
    def end_time_s(self) -> float:
        return float(self.frame_count - 1) / float(self.fps)

    @property
    def timestamps_s(self) -> np.ndarray:
        return np.arange(self.frame_count, dtype=np.float64) / float(self.fps)


SHORT_HORIZON = HorizonProfile("short", frame_count=357)
LONG_HORIZON = HorizonProfile("long", frame_count=961)
HORIZON_PROFILES: Mapping[str, HorizonProfile] = {
    SHORT_HORIZON.name: SHORT_HORIZON,
    LONG_HORIZON.name: LONG_HORIZON,
}


@dataclass(frozen=True)
class RevisitPair:
    """One registered generated-to-generated revisit."""

    name: str
    first_event: str
    revisit_event: str

    def __post_init__(self) -> None:
        if not self.name or not self.first_event or not self.revisit_event:
            raise ValueError("revisit pair fields must be non-empty")
        if self.first_event == self.revisit_event:
            raise ValueError("a revisit pair needs two distinct events")


def horizon_profile(value: str | HorizonProfile) -> HorizonProfile:
    if isinstance(value, HorizonProfile):
        return value
    try:
        return HORIZON_PROFILES[str(value)]
    except KeyError as exc:
        raise ValueError(f"unknown horizon profile: {value!r}") from exc


def validate_revisit_excursion(poses: np.ndarray, first: int, revisit: int) -> None:
    """Check closure and non-stationarity, not unseen geometry or clearance."""
    if not 0 <= first < revisit < len(poses):
        raise ValueError("revisit indices must be ordered and in bounds")
    if not np.allclose(poses[first], poses[revisit], atol=1e-6, rtol=0):
        raise ValueError("revisit endpoints must have the same complete pose")
    relative = np.linalg.inv(poses[first]) @ poses[first:revisit + 1]
    distance = np.linalg.norm(relative[:, :3, 3], axis=1).max()
    angle = Rotation.from_matrix(relative[:, :3, :3]).magnitude().max()
    if distance <= 1e-6 and angle <= 1e-6:
        raise ValueError("revisit must include an intervening excursion")


