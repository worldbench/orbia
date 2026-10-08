"""Deterministic motion-template generation and hard geometric validators."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

import numpy as np
from scipy.interpolate import CubicHermiteSpline, CubicSpline, PchipInterpolator
from scipy.spatial.transform import Rotation, RotationSpline, Slerp

from src.utils.errors import TrajectoryFeasibilityError

from src.utils.geometry import MeshRaycaster, PathCollisionReport


FORWARD_BACKWARD_KEY_INDICES = {
    "q0": 0,
    "w1": 36,
    "qe1": 60,
    "qc_first": 84,
    "w2": 108,
    "qc_revisit": 132,
    "q0_return": 215,
}

# Metric paths are authored directly on the benchmark clock (16 FPS, 357 frames).
CANONICAL_FORWARD_BACKWARD_FPS = 16
CANONICAL_FORWARD_BACKWARD_FRAME_COUNT = 357
_FORWARD_BACKWARD_EVENT_NAMES = tuple(FORWARD_BACKWARD_KEY_INDICES)

# Every metric template uses the same clock and event names.
CANONICAL_METRIC_TRAJECTORY_FPS = CANONICAL_FORWARD_BACKWARD_FPS
CANONICAL_METRIC_TRAJECTORY_FRAME_COUNT = CANONICAL_FORWARD_BACKWARD_FRAME_COUNT
METRIC_TRAJECTORY_EVENT_ROLES = _FORWARD_BACKWARD_EVENT_NAMES

# Route-conforming trajectories (ScanNet++) use the same seven events, but
# their intermediate samples follow the registered capture route instead of an
# analytic template curve. The template name describes the route topology.
ROUTE_CONFORMING_TEMPLATE_NAMES = frozenset(
    {
        "forward_backward",
        "lateral_out_and_back",
        "arc_return",
    }
)

# Revisit events must return to the first-visit view within 15 cm / 8 degrees.
# The 48-frame window constants describe the moving windows around each event.
COMPLETION_REVISIT_HOLD_RADIUS_FRAMES = 6
COMPLETION_REVISIT_WINDOW_FRAME_COUNT = 48
COMPLETION_REVISIT_EVENT_OFFSET_IN_WINDOW = 24
MINIMUM_COMPLETION_WINDOW_SEPARATION_FRAMES = 24
COMPLETION_CONTRACT_NATURAL_MOVING = "natural_moving_48_frame_windows"
COMPLETION_CONTRACT_EXACT_HOLD = "exact_hold_windows"

# Orbit event timing is allocated from the route and dynamics limits; read
# ``trajectory.key_indices`` instead of assuming this nominal schedule.


# The requested trajectory passes exactly through the registered query pose at qe1.
REQUESTED_QE_TRANSLATION_TOLERANCE_M = 1e-6
REQUESTED_QE_ROTATION_TOLERANCE_DEG = 1e-6

# Wider tolerances for matching a model's executed camera to the query view.
EXECUTED_QUERY_TRANSLATION_TOLERANCE_M = 0.20
EXECUTED_QUERY_ROTATION_TOLERANCE_DEG = 10.0


@dataclass(frozen=True)
class TrajectoryDynamicsLimits:
    """Hard sampled-path limits in metric units at the declared frame rate."""

    maximum_linear_speed_m_s: float = 2.0
    maximum_angular_speed_deg_s: float = 120.0
    maximum_linear_acceleration_m_s2: float = 8.0
    maximum_angular_acceleration_deg_s2: float = 720.0
    maximum_linear_jerk_m_s3: float = 120.0
    maximum_angular_jerk_deg_s3: float = 10000.0

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            numeric = float(value)
            if not np.isfinite(numeric) or numeric <= 0.0:
                raise ValueError(f"trajectory dynamics limit {name} must be finite and positive")
            object.__setattr__(self, name, numeric)


DEFAULT_DYNAMICS_LIMITS = TrajectoryDynamicsLimits()


def _coerce_dynamics_limits(
    value: TrajectoryDynamicsLimits | Mapping[str, object] | None,
) -> TrajectoryDynamicsLimits:
    if value is None:
        return DEFAULT_DYNAMICS_LIMITS
    if isinstance(value, TrajectoryDynamicsLimits):
        return value
    if not isinstance(value, Mapping):
        raise TypeError("dynamics_limits must be a mapping or TrajectoryDynamicsLimits")
    return TrajectoryDynamicsLimits(**dict(value))


def _validate_trajectory_components(
    poses: np.ndarray,
    key_indices: Mapping[str, int],
    duration_s: float,
    fps: int,
) -> None:
    """Validate sampled SE(3), timing, and key-node bounds."""

    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError("trajectory poses must have shape (N, 4, 4)")
    if not np.isfinite(poses).all():
        raise ValueError("trajectory poses must be finite")
    if not np.allclose(
        poses[:, 3, :],
        np.asarray([0.0, 0.0, 0.0, 1.0]),
        rtol=0.0,
        atol=1e-9,
    ):
        raise ValueError("trajectory poses must have homogeneous [0,0,0,1] rows")
    rotations = poses[:, :3, :3]
    gram = np.swapaxes(rotations, 1, 2) @ rotations
    identity = np.eye(3, dtype=np.float64)[None]
    if not np.allclose(gram, identity, rtol=0.0, atol=1e-6):
        raise ValueError("trajectory rotations must be orthonormal")
    determinants = np.linalg.det(rotations)
    if not np.allclose(determinants, 1.0, rtol=0.0, atol=1e-6):
        raise ValueError("trajectory rotations must be proper (determinant +1)")
    if not np.isfinite(duration_s) or duration_s <= 0.0:
        raise ValueError("trajectory duration_s must be finite and positive")
    if isinstance(fps, (bool, np.bool_)) or int(fps) != fps or int(fps) <= 0:
        raise ValueError("trajectory fps must be a positive integer")
    frame_count_float = duration_s * int(fps)
    if not np.isclose(frame_count_float, round(frame_count_float), rtol=0.0, atol=1e-9):
        raise ValueError("trajectory duration_s * fps must be an integer")
    if len(poses) != int(round(frame_count_float)):
        raise ValueError("trajectory frame count must equal duration_s * fps")
    if not key_indices:
        raise ValueError("trajectory key_indices must be non-empty")
    indices: list[int] = []
    for name, index in key_indices.items():
        if not isinstance(name, str) or not name:
            raise ValueError("trajectory key-node names must be non-empty strings")
        if isinstance(index, (bool, np.bool_)) or not isinstance(index, (int, np.integer)):
            raise ValueError("trajectory key indices must be integers")
        numeric_index = int(index)
        if not 0 <= numeric_index < len(poses):
            raise ValueError("trajectory key index is outside the sampled path")
        indices.append(numeric_index)
    if len(set(indices)) != len(indices):
        raise ValueError("trajectory key indices must be unique")
    if indices != sorted(indices):
        raise ValueError("trajectory key indices must be strictly increasing")


@dataclass(frozen=True)
class Trajectory:
    template_name: str
    difficulty: str
    duration_s: float
    fps: int
    poses_world_camera: np.ndarray
    key_indices: Mapping[str, int]
    parameters: Mapping[str, object]

    def __post_init__(self) -> None:
        poses = np.array(self.poses_world_camera, dtype=np.float64, copy=True)
        duration_s = float(self.duration_s)
        fps = int(self.fps)
        key_indices = dict(self.key_indices)
        _validate_trajectory_components(poses, key_indices, duration_s, self.fps)
        if not str(self.template_name).strip():
            raise ValueError("trajectory template_name must be non-empty")
        if not str(self.difficulty).strip():
            raise ValueError("trajectory difficulty must be non-empty")
        object.__setattr__(self, "poses_world_camera", poses)
        object.__setattr__(self, "duration_s", duration_s)
        object.__setattr__(self, "fps", fps)
        object.__setattr__(self, "key_indices", key_indices)
        object.__setattr__(self, "parameters", dict(self.parameters))

    def metadata(self) -> dict:
        qe_frame_id = self.parameters.get("evidence_frame_id")
        dynamics_limits = _coerce_dynamics_limits(
            self.parameters.get("dynamics_limits")
        )
        return {
            "difficulty": self.difficulty,
            "duration_s": self.duration_s,
            "fps": self.fps,
            "frame_count": len(self.poses_world_camera),
            "key_node_ids": list(self.key_indices),
            "key_indices": dict(self.key_indices),
            "name": self.template_name,
            "parameters": dict(self.parameters),
            "dynamics_limits": asdict(dynamics_limits),
            "dynamics_units": {
                "angular_acceleration": "deg/s^2",
                "angular_jerk": "deg/s^3",
                "angular_speed": "deg/s",
                "linear_acceleration": "m/s^2",
                "linear_jerk": "m/s^3",
                "linear_speed": "m/s",
            },
            "registered_keyframes": {
                "qe1": qe_frame_id,
            },
            "executed_query_matching_tolerance": {
                "rotation_deg": EXECUTED_QUERY_ROTATION_TOLERANCE_DEG,
                "translation_m": EXECUTED_QUERY_TRANSLATION_TOLERANCE_M,
            },
        }


@dataclass(frozen=True)
class TrajectoryValidation:
    passed: bool
    template_name: str
    exact_q0_return: bool
    collision: PathCollisionReport
    metrics: Mapping[str, float | bool]

    def to_dict(self) -> dict:
        return {
            "collision": asdict(self.collision),
            "exact_q0_return": self.exact_q0_return,
            "metrics": dict(self.metrics),
            "passed": self.passed,
            "template_name": self.template_name,
        }


DENSE_MESH_CLEARANCE_CONTRACT = "dense_mesh_clearance_5cm"
SAMPLED_MESH_CLEARANCE_CONTRACT = "sampled_mesh_clearance"
COARSE_SAMPLED_MESH_CLEARANCE_CONTRACT = "coarse_sampled_mesh_clearance"


def _validate_trajectory_mesh_clearance(
    trajectory: Trajectory,
    raycaster: MeshRaycaster,
) -> PathCollisionReport:
    """Validate mesh clearance with the policy recorded in the trajectory parameters."""

    contract = str(
        trajectory.parameters.get(
            "clearance_validation_contract", DENSE_MESH_CLEARANCE_CONTRACT
        )
    )
    desired = float(trajectory.parameters["minimum_clearance_m"])
    if contract == DENSE_MESH_CLEARANCE_CONTRACT:
        return raycaster.validate_camera_path(
            trajectory.poses_world_camera[:, :3, 3], desired
        )
    if contract == SAMPLED_MESH_CLEARANCE_CONTRACT:
        sample_count = trajectory.parameters.get("clearance_uniform_sample_count", 96)
        return raycaster.validate_camera_path_sampled(
            trajectory.poses_world_camera[:, :3, 3],
            desired,
            uniform_sample_count=sample_count,
            key_position_indices=trajectory.key_indices.values(),
        )
    if contract == COARSE_SAMPLED_MESH_CLEARANCE_CONTRACT:
        sample_count = trajectory.parameters.get("clearance_uniform_sample_count", 32)
        return raycaster.validate_camera_path_coarse_sampled(
            trajectory.poses_world_camera[:, :3, 3],
            desired,
            uniform_sample_count=sample_count,
            key_position_indices=trajectory.key_indices.values(),
        )
    raise ValueError(f"unsupported clearance_validation_contract: {contract!r}")


def _quintic_smoothstep(value: np.ndarray | float) -> np.ndarray | float:
    value = np.asarray(value)
    result = value**3 * (value * (value * 6.0 - 15.0) + 10.0)
    return float(result) if result.ndim == 0 else result


def _look_at_opencv(
    camera_position: np.ndarray,
    target: np.ndarray,
    *,
    world_up: np.ndarray = np.asarray([0.0, 0.0, 1.0]),
) -> np.ndarray:
    forward = np.asarray(target, dtype=np.float64) - np.asarray(
        camera_position, dtype=np.float64
    )
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, world_up)
    if np.linalg.norm(right) < 1e-8:
        raise ValueError("look-at forward is parallel to world_up")
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    down /= np.linalg.norm(down)
    return np.stack([right, down, forward], axis=1)


def _interpolate_key_poses(
    frame_count: int,
    key_indices: list[int],
    positions: np.ndarray,
    rotations: np.ndarray,
) -> np.ndarray:
    if len(key_indices) != len(positions) or len(positions) != len(rotations):
        raise ValueError("key pose arrays must have equal lengths")
    poses = np.repeat(np.eye(4, dtype=np.float64)[None], frame_count, axis=0)
    for segment in range(len(key_indices) - 1):
        start = key_indices[segment]
        end = key_indices[segment + 1]
        frame_indices = np.arange(start, end + 1)
        alpha = (frame_indices - start) / (end - start)
        smooth = _quintic_smoothstep(alpha)
        poses[frame_indices, :3, 3] = (
            (1.0 - smooth[:, None]) * positions[segment]
            + smooth[:, None] * positions[segment + 1]
        )
        slerp = Slerp(
            [0.0, 1.0],
            Rotation.from_matrix(rotations[segment : segment + 2]),
        )
        poses[frame_indices, :3, :3] = slerp(smooth).as_matrix()
    return poses


def _normalized_translation_derivative_maxima(
    start: np.ndarray,
    end: np.ndarray,
    *,
    bow_direction: np.ndarray | None = None,
    bow_m: float = 0.0,
) -> tuple[float, float, float]:
    """Return normalized-time speed/acceleration/jerk maxima for a segment."""

    delta = np.asarray(end, dtype=np.float64) - np.asarray(start, dtype=np.float64)
    bow_m = float(bow_m)
    if bow_direction is None:
        lateral = np.zeros(3, dtype=np.float64)
    else:
        lateral = np.asarray(bow_direction, dtype=np.float64)
        norm = float(np.linalg.norm(lateral))
        if norm <= 1e-12:
            raise ValueError("bow_direction must be non-zero")
        lateral = lateral / norm
    tau = np.linspace(0.0, 1.0, 4097, dtype=np.float64)
    smooth = _quintic_smoothstep(tau)
    first = 30.0 * tau**2 * (tau - 1.0) ** 2
    second = 60.0 * tau * (2.0 * tau**2 - 3.0 * tau + 1.0)
    third = 360.0 * tau**2 - 360.0 * tau + 60.0
    phase = np.pi * smooth
    position_first = (
        delta[None] + (bow_m * np.pi * np.cos(phase))[:, None] * lateral[None]
    )
    position_second = (
        (-bow_m * np.pi**2 * np.sin(phase))[:, None] * lateral[None]
    )
    position_third = (
        (-bow_m * np.pi**3 * np.cos(phase))[:, None] * lateral[None]
    )
    velocity = position_first * first[:, None]
    acceleration = (
        position_second * first[:, None] ** 2
        + position_first * second[:, None]
    )
    jerk = (
        position_third * first[:, None] ** 3
        + 3.0
        * position_second
        * first[:, None]
        * second[:, None]
        + position_first * third[:, None]
    )
    return (
        _maximum_vector_norm(velocity),
        _maximum_vector_norm(acceleration),
        _maximum_vector_norm(jerk),
    )


def _adaptive_metric_key_indices(
    positions: np.ndarray,
    rotations: np.ndarray,
    *,
    frame_count: int,
    fps: int,
    limits: TrajectoryDynamicsLimits,
    template_name: str,
    bow_direction: np.ndarray | None = None,
    segment_bows_m: np.ndarray | None = None,
    segment_start_holds: np.ndarray | None = None,
    segment_end_holds: np.ndarray | None = None,
    segment_translation_derivative_maxima: np.ndarray | None = None,
    segment_minimum_intervals: np.ndarray | None = None,
) -> dict[str, int]:
    """Allocate fixed-clock event indices from continuous SE(3) requirements."""

    positions = np.asarray(positions, dtype=np.float64)
    rotations = np.asarray(rotations, dtype=np.float64)
    segment_count = len(positions) - 1
    if positions.shape != (len(METRIC_TRAJECTORY_EVENT_ROLES), 3):
        raise ValueError("metric template positions must define seven event roles")
    if rotations.shape != (len(METRIC_TRAJECTORY_EVENT_ROLES), 3, 3):
        raise ValueError("metric template rotations must define seven event roles")

    bows = (
        np.zeros(segment_count, dtype=np.float64)
        if segment_bows_m is None
        else np.asarray(segment_bows_m, dtype=np.float64)
    )
    start_holds = (
        np.zeros(segment_count, dtype=np.int64)
        if segment_start_holds is None
        else np.asarray(segment_start_holds, dtype=np.int64)
    )
    end_holds = (
        np.zeros(segment_count, dtype=np.int64)
        if segment_end_holds is None
        else np.asarray(segment_end_holds, dtype=np.int64)
    )
    if bows.shape != (segment_count,):
        raise ValueError("segment_bows_m must have one value per event segment")
    if start_holds.shape != (segment_count,) or end_holds.shape != (segment_count,):
        raise ValueError("segment holds must have one value per event segment")
    if np.any(start_holds < 0) or np.any(end_holds < 0):
        raise ValueError("segment holds must be non-negative")
    requested_minimum_intervals = (
        np.zeros(segment_count, dtype=np.int64)
        if segment_minimum_intervals is None
        else np.asarray(segment_minimum_intervals, dtype=np.int64)
    )
    if requested_minimum_intervals.shape != (segment_count,):
        raise ValueError(
            "segment_minimum_intervals must have one value per event segment"
        )
    if np.any(requested_minimum_intervals < 0):
        raise ValueError("segment minimum intervals must be non-negative")

    relative = rotations[:-1].transpose(0, 2, 1) @ rotations[1:]
    angles_deg = np.rad2deg(Rotation.from_matrix(relative).magnitude())
    if segment_translation_derivative_maxima is None:
        normalized_derivatives = np.asarray(
            [
                _normalized_translation_derivative_maxima(
                    start,
                    end,
                    bow_direction=bow_direction,
                    bow_m=bow,
                )
                for start, end, bow in zip(positions[:-1], positions[1:], bows)
            ],
            dtype=np.float64,
        )
    else:
        normalized_derivatives = np.asarray(
            segment_translation_derivative_maxima, dtype=np.float64
        )
        if normalized_derivatives.shape != (segment_count, 3):
            raise ValueError(
                "segment_translation_derivative_maxima must have shape "
                "(segment_count, 3)"
            )
        if not np.isfinite(normalized_derivatives).all() or np.any(
            normalized_derivatives < 0.0
        ):
            raise ValueError(
                "segment_translation_derivative_maxima must be finite and non-negative"
            )
    angular_normalized = np.stack(
        [
            1.875 * angles_deg,
            (10.0 / np.sqrt(3.0)) * angles_deg,
            60.0 * angles_deg,
        ],
        axis=1,
    )
    required_times = np.maximum.reduce(
        [
            normalized_derivatives[:, 0] / limits.maximum_linear_speed_m_s,
            np.sqrt(
                normalized_derivatives[:, 1]
                / limits.maximum_linear_acceleration_m_s2
            ),
            np.cbrt(
                normalized_derivatives[:, 2]
                / limits.maximum_linear_jerk_m_s3
            ),
            angular_normalized[:, 0] / limits.maximum_angular_speed_deg_s,
            np.sqrt(
                angular_normalized[:, 1]
                / limits.maximum_angular_acceleration_deg_s2
            ),
            np.cbrt(
                angular_normalized[:, 2]
                / limits.maximum_angular_jerk_deg_s3
            ),
        ]
    )
    moving_intervals = np.maximum(
        2, np.ceil(required_times * int(fps)).astype(np.int64) + 1
    )
    moving_intervals = np.maximum(moving_intervals, requested_minimum_intervals)
    minimum_intervals = moving_intervals + start_holds + end_holds
    available_intervals = int(frame_count) - 1
    required_intervals = int(minimum_intervals.sum())
    if required_intervals > available_intervals:
        raise TrajectoryFeasibilityError(
            f"canonical 357-frame clock cannot satisfy fixed dynamics for "
            f"{template_name} (required intervals={required_intervals}, "
            f"available={available_intervals})"
        )

    slack = available_intervals - required_intervals
    weights = (
        normalized_derivatives[:, 0]
        + angles_deg / max(limits.maximum_angular_speed_deg_s, 1e-9)
    )
    if not np.any(weights > 0.0):
        weights = np.ones_like(weights)
    quotas = slack * weights / weights.sum()
    extras = np.floor(quotas).astype(np.int64)
    remainder = slack - int(extras.sum())
    order = np.argsort(-(quotas - extras), kind="stable")
    extras[order[:remainder]] += 1
    intervals = minimum_intervals + extras
    indices = np.concatenate([[0], np.cumsum(intervals)]).astype(int)
    if indices[-1] != frame_count - 1:
        raise AssertionError("adaptive metric timing did not consume canonical clock")
    return dict(zip(METRIC_TRAJECTORY_EVENT_ROLES, indices.tolist()))


def _interpolate_metric_key_poses(
    frame_count: int,
    key_indices: Mapping[str, int],
    positions: np.ndarray,
    rotations: np.ndarray,
    *,
    bow_direction: np.ndarray | None = None,
    segment_bows_m: np.ndarray | None = None,
    segment_start_holds: np.ndarray | None = None,
    segment_end_holds: np.ndarray | None = None,
) -> np.ndarray:
    """Interpolate metric event poses with optional curved translation and holds."""

    keys = list(key_indices.values())
    segment_count = len(keys) - 1
    bows = (
        np.zeros(segment_count, dtype=np.float64)
        if segment_bows_m is None
        else np.asarray(segment_bows_m, dtype=np.float64)
    )
    start_holds = (
        np.zeros(segment_count, dtype=np.int64)
        if segment_start_holds is None
        else np.asarray(segment_start_holds, dtype=np.int64)
    )
    end_holds = (
        np.zeros(segment_count, dtype=np.int64)
        if segment_end_holds is None
        else np.asarray(segment_end_holds, dtype=np.int64)
    )
    lateral = (
        np.zeros(3, dtype=np.float64)
        if bow_direction is None
        else np.asarray(bow_direction, dtype=np.float64)
    )
    if bow_direction is not None:
        lateral = lateral / np.linalg.norm(lateral)

    poses = np.repeat(np.eye(4, dtype=np.float64)[None], frame_count, axis=0)
    for segment, (start, end) in enumerate(zip(keys[:-1], keys[1:])):
        moving_start = start + int(start_holds[segment])
        moving_end = end - int(end_holds[segment])
        if moving_end - moving_start < 2:
            raise TrajectoryFeasibilityError(
                "canonical event interval is too short after completion holds"
            )
        poses[start : moving_start + 1, :3, 3] = positions[segment]
        poses[start : moving_start + 1, :3, :3] = rotations[segment]
        poses[moving_end : end + 1, :3, 3] = positions[segment + 1]
        poses[moving_end : end + 1, :3, :3] = rotations[segment + 1]

        frame_indices = np.arange(moving_start, moving_end + 1)
        alpha = (frame_indices - moving_start) / (moving_end - moving_start)
        smooth = _quintic_smoothstep(alpha)
        translation = (
            (1.0 - smooth[:, None]) * positions[segment]
            + smooth[:, None] * positions[segment + 1]
        )
        if bows[segment] != 0.0:
            translation += (
                bows[segment] * np.sin(np.pi * smooth)
            )[:, None] * lateral[None]
        poses[frame_indices, :3, 3] = translation
        slerp = Slerp(
            [0.0, 1.0],
            Rotation.from_matrix(rotations[segment : segment + 2]),
        )
        poses[frame_indices, :3, :3] = slerp(smooth).as_matrix()
    return poses


def _monotone_work_derivatives(
    times: np.ndarray,
    work: np.ndarray,
) -> np.ndarray:
    """Return shape-preserving derivatives for a strictly increasing work path."""

    times = np.asarray(times, dtype=np.float64)
    work = np.asarray(work, dtype=np.float64)
    if (
        times.ndim != 1
        or work.ndim != 1
        or len(times) != len(work)
        or len(times) < 2
    ):
        raise ValueError("times/work must be aligned one-dimensional arrays")
    intervals = np.diff(times)
    increments = np.diff(work)
    if np.any(intervals <= 0.0) or np.any(increments <= 0.0):
        raise ValueError("phase time and path work must be strictly increasing")
    secants = increments / intervals
    derivatives = np.zeros_like(work)
    for index in range(1, len(work) - 1):
        before = secants[index - 1]
        after = secants[index]
        before_interval = intervals[index - 1]
        after_interval = intervals[index]
        weight_before = 2.0 * after_interval + before_interval
        weight_after = after_interval + 2.0 * before_interval
        derivatives[index] = (
            weight_before + weight_after
        ) / (
            weight_before / before + weight_after / after
        )
    return derivatives


def _continuous_translation_parameterization(
    translations: np.ndarray,
    key_indices: Mapping[str, int],
    *,
    stop_roles: Sequence[str],
    polar_focus_xy: np.ndarray | None = None,
) -> np.ndarray:
    """Retain a geometric path while removing auxiliary-keyframe stops."""

    source = np.asarray(translations, dtype=np.float64)
    if source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("translations must have shape (N, 3)")
    if not np.isfinite(source).all():
        raise ValueError("translations must be finite")
    focus_xy = None
    if polar_focus_xy is not None:
        focus_xy = np.asarray(polar_focus_xy, dtype=np.float64)
        if focus_xy.shape != (2,) or not np.isfinite(focus_xy).all():
            raise ValueError("polar_focus_xy must be finite XY")
    key = {name: int(index) for name, index in key_indices.items()}
    missing = [role for role in stop_roles if role not in key]
    if missing:
        raise ValueError(f"unknown continuous-time stop roles: {missing}")
    stop_indices = [key[role] for role in stop_roles]
    if stop_indices != sorted(stop_indices) or len(set(stop_indices)) != len(
        stop_indices
    ):
        raise ValueError("continuous-time stop roles must be strictly ordered")
    if stop_indices[0] != 0 or stop_indices[-1] != len(source) - 1:
        raise ValueError("continuous-time stop roles must include both endpoints")

    result = np.array(source, copy=True)
    ordered_events = sorted(key.items(), key=lambda item: item[1])
    for phase_start, phase_end in zip(stop_indices[:-1], stop_indices[1:]):
        phase_source = source[phase_start : phase_end + 1]
        step_lengths = np.linalg.norm(np.diff(phase_source, axis=0), axis=1)
        source_arc = np.concatenate([[0.0], np.cumsum(step_lengths)])
        if source_arc[-1] <= 1e-10:
            raise TrajectoryFeasibilityError(
                "continuous-time translation phase has no path length"
            )
        phase_events = [
            index
            for _, index in ordered_events
            if phase_start <= index <= phase_end
        ]
        event_times = np.asarray(phase_events, dtype=np.float64)
        event_arc = source_arc[
            np.asarray(phase_events, dtype=np.int64) - phase_start
        ]
        if np.any(np.diff(event_arc) <= 1e-10):
            raise TrajectoryFeasibilityError(
                "continuous-time auxiliary translation knots must advance "
                "strictly within each stop-to-stop phase"
            )
        derivatives = _monotone_work_derivatives(event_times, event_arc)
        target_arc = CubicHermiteSpline(
            event_times,
            event_arc,
            derivatives,
        )(np.arange(phase_start, phase_end + 1, dtype=np.float64))
        target_arc = np.clip(target_arc, event_arc[0], event_arc[-1])

        unique = np.concatenate(
            [[True], np.diff(source_arc) > 1e-12]
        )
        source_arc_unique = source_arc[unique]
        phase_unique = phase_source[unique]
        if focus_xy is None:
            for axis in range(3):
                result[phase_start : phase_end + 1, axis] = np.interp(
                    target_arc,
                    source_arc_unique,
                    phase_unique[:, axis],
                )
        else:
            relative_xy = phase_unique[:, :2] - focus_xy[None]
            source_angles = np.unwrap(
                np.arctan2(relative_xy[:, 1], relative_xy[:, 0])
            )
            source_radii = np.linalg.norm(relative_xy, axis=1)
            target_angles = np.interp(
                target_arc,
                source_arc_unique,
                source_angles,
            )
            target_radii = np.interp(
                target_arc,
                source_arc_unique,
                source_radii,
            )
            result[phase_start : phase_end + 1, 0] = (
                focus_xy[0] + target_radii * np.cos(target_angles)
            )
            result[phase_start : phase_end + 1, 1] = (
                focus_xy[1] + target_radii * np.sin(target_angles)
            )
            result[phase_start : phase_end + 1, 2] = np.interp(
                target_arc,
                source_arc_unique,
                phase_unique[:, 2],
            )

    # Interpolation arithmetic must never perturb evaluator event positions.
    for _, index in ordered_events:
        result[index] = source[index]
    return result


def _continuous_orientation_parameterization(
    frame_count: int,
    key_indices: Mapping[str, int],
    rotations: np.ndarray,
    *,
    orientation_roles: Sequence[str],
) -> np.ndarray:
    """Distribute SO(3) work globally through exact evaluator rotation knots."""

    key = {name: int(index) for name, index in key_indices.items()}
    missing = [role for role in orientation_roles if role not in key]
    if missing:
        raise ValueError(f"unknown continuous orientation roles: {missing}")
    indices = np.asarray([key[role] for role in orientation_roles], dtype=int)
    if np.any(np.diff(indices) <= 0):
        raise ValueError("continuous orientation roles must be strictly ordered")
    rotations = np.asarray(rotations, dtype=np.float64)
    if rotations.shape != (len(key), 3, 3):
        raise ValueError("rotations must align with canonical event roles")
    event_order = list(key)
    role_rotations = np.stack(
        [rotations[event_order.index(role)] for role in orientation_roles]
    )
    spline = RotationSpline(
        indices.astype(np.float64),
        Rotation.from_matrix(role_rotations),
    )
    result = spline(np.arange(frame_count, dtype=np.float64)).as_matrix()
    for role, rotation in zip(orientation_roles, role_rotations):
        result[key[role]] = rotation
    return result


def _enforce_monotone_world_yaw(
    rotations_world_camera: np.ndarray,
    key_indices: Mapping[str, int],
    key_rotations: np.ndarray,
) -> np.ndarray:
    """Remove SO(3)-spline yaw overshoot without losing exact key rotations."""

    rotations = np.asarray(rotations_world_camera, dtype=np.float64)
    key_rotations = np.asarray(key_rotations, dtype=np.float64)
    event_order = list(key_indices)
    if key_rotations.shape != (len(event_order), 3, 3):
        raise ValueError("key_rotations must align with canonical event roles")
    times = np.asarray(
        [int(key_indices[role]) for role in event_order],
        dtype=np.float64,
    )
    reference = key_rotations[0]

    def relative_yaw(values: np.ndarray) -> np.ndarray:
        relative = values @ reference.T
        yaw = np.arctan2(relative[:, 1, 0], relative[:, 0, 0])
        return np.unwrap(yaw)

    key_yaw = relative_yaw(key_rotations)
    desired_yaw = PchipInterpolator(times, key_yaw)(
        np.arange(len(rotations), dtype=np.float64)
    )
    actual_yaw = relative_yaw(rotations)
    correction = Rotation.from_rotvec(
        (desired_yaw - actual_yaw)[:, None]
        * np.asarray([0.0, 0.0, 1.0], dtype=np.float64)[None]
    ).as_matrix()
    result = correction @ rotations
    for index, rotation in zip(times.astype(int), key_rotations):
        result[index] = rotation
    return result


def _camera_relative_yaw_deg(
    rotations_world_camera: np.ndarray,
    reference_rotation_world_camera: np.ndarray,
) -> np.ndarray:
    """Return unwrapped yaw about the reference camera's local down axis."""

    rotations = np.asarray(rotations_world_camera, dtype=np.float64)
    reference = np.asarray(reference_rotation_world_camera, dtype=np.float64)
    relative = np.einsum("ij,njk->nik", reference.T, rotations)
    yaw = np.arctan2(relative[:, 0, 2], relative[:, 2, 2])
    return np.rad2deg(np.unwrap(yaw))


def _enforce_monotone_camera_yaw(
    rotations_world_camera: np.ndarray,
    key_indices: Mapping[str, int],
    key_rotations: np.ndarray,
) -> np.ndarray:
    """Remove yaw overshoot about Q0 camera-down while preserving exact knots."""

    rotations = np.asarray(rotations_world_camera, dtype=np.float64)
    key_rotations = np.asarray(key_rotations, dtype=np.float64)
    event_order = list(key_indices)
    if key_rotations.shape != (len(event_order), 3, 3):
        raise ValueError("key_rotations must align with canonical event roles")
    times = np.asarray([int(key_indices[role]) for role in event_order], dtype=float)
    reference = key_rotations[0]
    key_yaw = np.deg2rad(_camera_relative_yaw_deg(key_rotations, reference))
    desired_yaw = PchipInterpolator(times, key_yaw)(
        np.arange(len(rotations), dtype=np.float64)
    )
    actual_yaw = np.deg2rad(_camera_relative_yaw_deg(rotations, reference))
    correction = Rotation.from_rotvec(
        (desired_yaw - actual_yaw)[:, None]
        * np.asarray([0.0, 1.0, 0.0], dtype=np.float64)[None]
    ).as_matrix()
    result = rotations @ correction
    for index, rotation in zip(times.astype(int), key_rotations):
        result[index] = rotation
    return result


def _apply_continuous_canonical_parameterization(
    poses: np.ndarray,
    key_indices: Mapping[str, int],
    rotations: np.ndarray,
    *,
    stop_roles: Sequence[str],
    orientation_roles: Sequence[str],
    translation_stop_roles: Sequence[str] | None = None,
    polar_focus_xy: np.ndarray | None = None,
) -> np.ndarray:
    """Apply the shared canonical clock while preserving exact strong poses."""

    result = np.array(poses, dtype=np.float64, copy=True)
    # Pure yaw has no translation phase to parameterize. Preserve its exact
    # constant camera centre while applying the shared orientation clock.
    if np.max(np.linalg.norm(result[:, :3, 3] - result[0, :3, 3], axis=1)) > 1e-12:
        result[:, :3, 3] = _continuous_translation_parameterization(
            result[:, :3, 3],
            key_indices,
            stop_roles=(
                stop_roles
                if translation_stop_roles is None
                else translation_stop_roles
            ),
            polar_focus_xy=polar_focus_xy,
        )
    result[:, :3, :3] = _continuous_orientation_parameterization(
        len(result),
        key_indices,
        rotations,
        orientation_roles=orientation_roles,
    )
    return result


def _continuous_time_parameters(
    *,
    stop_roles: Sequence[str],
    orientation_roles: Sequence[str],
    translation_stop_roles: Sequence[str] | None = None,
    translation_clock: str = "monotone_arc_length_cubic_hermite",
    rotation_clock: str = "global_so3_rotation_spline",
) -> dict[str, object]:
    """Serialize the exact clock contract used by one metric template."""

    parameters = {
        "exact_pose_roles": ["q0", "qe1", "q0_return"],
        "orientation_knot_roles": list(orientation_roles),
        "stop_roles": list(stop_roles),
        "event_time_allocation": (
            "metric_translation_work_plus_so3_rotation_work"
        ),
        "translation_clock": str(translation_clock),
        "rotation_clock": str(rotation_clock),
    }
    if translation_stop_roles is not None:
        parameters["translation_stop_roles"] = list(translation_stop_roles)
    return parameters


def pose_error(
    actual_world_camera: np.ndarray,
    expected_world_camera: np.ndarray,
) -> tuple[float, float]:
    """Return translation metres and geodesic rotation degrees."""

    actual = np.asarray(actual_world_camera, dtype=np.float64)
    expected = np.asarray(expected_world_camera, dtype=np.float64)
    translation = float(np.linalg.norm(actual[:3, 3] - expected[:3, 3]))
    relative = expected[:3, :3].T @ actual[:3, :3]
    rotation = float(np.rad2deg(Rotation.from_matrix(relative).magnitude()))
    return translation, rotation


def _maximum_vector_norm(values: np.ndarray) -> float:
    if len(values) == 0:
        return 0.0
    return float(np.max(np.linalg.norm(values, axis=1)))


def trajectory_dynamics_metrics(
    trajectory: Trajectory,
    limits: TrajectoryDynamicsLimits | Mapping[str, object] | None = None,
) -> dict[str, float | bool]:
    """Measure sampled SE(3) continuity using shortest SO(3) increments."""

    poses = trajectory.poses_world_camera
    _validate_trajectory_components(
        poses,
        trajectory.key_indices,
        trajectory.duration_s,
        trajectory.fps,
    )
    resolved_limits = _coerce_dynamics_limits(
        limits if limits is not None else trajectory.parameters.get("dynamics_limits")
    )
    fps = float(trajectory.fps)
    positions = poses[:, :3, 3]
    rotations = poses[:, :3, :3]
    position_steps = np.diff(positions, axis=0)
    linear_velocity = position_steps * fps
    linear_acceleration = np.diff(linear_velocity, axis=0) * fps
    linear_jerk = np.diff(linear_acceleration, axis=0) * fps

    # Spatial/world-frame rotation increments make adjacent derivative vectors
    # comparable while Rotation.as_rotvec chooses the shortest geodesic branch.
    relative_world = rotations[1:] @ np.swapaxes(rotations[:-1], 1, 2)
    angular_steps_rad = Rotation.from_matrix(relative_world).as_rotvec()
    angular_velocity_deg = np.rad2deg(angular_steps_rad) * fps
    angular_acceleration_deg = np.diff(angular_velocity_deg, axis=0) * fps
    angular_jerk_deg = np.diff(angular_acceleration_deg, axis=0) * fps

    maximum_adjacent_translation_m = _maximum_vector_norm(position_steps)
    maximum_adjacent_rotation_deg = _maximum_vector_norm(
        np.rad2deg(angular_steps_rad)
    )
    maximum_linear_speed_m_s = _maximum_vector_norm(linear_velocity)
    maximum_angular_speed_deg_s = _maximum_vector_norm(angular_velocity_deg)
    maximum_linear_acceleration_m_s2 = _maximum_vector_norm(linear_acceleration)
    maximum_angular_acceleration_deg_s2 = _maximum_vector_norm(
        angular_acceleration_deg
    )
    maximum_linear_jerk_m_s3 = _maximum_vector_norm(linear_jerk)
    maximum_angular_jerk_deg_s3 = _maximum_vector_norm(angular_jerk_deg)
    checks = {
        "linear_speed_within_limit": maximum_linear_speed_m_s
        <= resolved_limits.maximum_linear_speed_m_s,
        "angular_speed_within_limit": maximum_angular_speed_deg_s
        <= resolved_limits.maximum_angular_speed_deg_s,
        "linear_acceleration_within_limit": maximum_linear_acceleration_m_s2
        <= resolved_limits.maximum_linear_acceleration_m_s2,
        "angular_acceleration_within_limit": maximum_angular_acceleration_deg_s2
        <= resolved_limits.maximum_angular_acceleration_deg_s2,
        "linear_jerk_within_limit": maximum_linear_jerk_m_s3
        <= resolved_limits.maximum_linear_jerk_m_s3,
        "angular_jerk_within_limit": maximum_angular_jerk_deg_s3
        <= resolved_limits.maximum_angular_jerk_deg_s3,
    }
    return {
        "maximum_adjacent_rotation_deg": maximum_adjacent_rotation_deg,
        "maximum_adjacent_translation_m": maximum_adjacent_translation_m,
        "maximum_angular_acceleration_deg_s2": maximum_angular_acceleration_deg_s2,
        "maximum_angular_acceleration_limit_deg_s2": resolved_limits.maximum_angular_acceleration_deg_s2,
        "maximum_angular_jerk_deg_s3": maximum_angular_jerk_deg_s3,
        "maximum_angular_jerk_limit_deg_s3": resolved_limits.maximum_angular_jerk_deg_s3,
        "maximum_angular_speed_deg_s": maximum_angular_speed_deg_s,
        "maximum_angular_speed_limit_deg_s": resolved_limits.maximum_angular_speed_deg_s,
        "maximum_linear_acceleration_m_s2": maximum_linear_acceleration_m_s2,
        "maximum_linear_acceleration_limit_m_s2": resolved_limits.maximum_linear_acceleration_m_s2,
        "maximum_linear_jerk_m_s3": maximum_linear_jerk_m_s3,
        "maximum_linear_jerk_limit_m_s3": resolved_limits.maximum_linear_jerk_m_s3,
        "maximum_linear_speed_m_s": maximum_linear_speed_m_s,
        "maximum_linear_speed_limit_m_s": resolved_limits.maximum_linear_speed_m_s,
        **checks,
        "dynamics_valid": bool(all(checks.values())),
    }


def _trajectory_positions(trajectory: Trajectory) -> np.ndarray:
    """Return validated sampled camera centres for public metric helpers."""

    if not isinstance(trajectory, Trajectory):
        raise TypeError("trajectory must be a Trajectory")
    _validate_trajectory_components(
        trajectory.poses_world_camera,
        trajectory.key_indices,
        trajectory.duration_s,
        trajectory.fps,
    )
    return trajectory.poses_world_camera[:, :3, 3]


def trajectory_total_path_length_m(trajectory: Trajectory) -> float:
    """Return the sampled translational path length in metres."""

    positions = _trajectory_positions(trajectory)
    return float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum())


trajectory_path_length_m = trajectory_total_path_length_m


def trajectory_position_pca_metrics(trajectory: Trajectory) -> dict[str, float | int]:
    """Measure sampled-position spread and PCA spectrum for route auditing."""

    positions = _trajectory_positions(trajectory)
    centered = positions - np.mean(positions, axis=0, keepdims=True)
    covariance = centered.T @ centered / float(len(centered))
    eigenvalues = np.linalg.eigvalsh(covariance)[::-1]
    eigenvalues = np.maximum(eigenvalues, 0.0)
    total = float(eigenvalues.sum())
    first = float(eigenvalues[0])
    second = float(eigenvalues[1])
    third = float(eigenvalues[2])
    effective_threshold = max(1e-12, first * 1e-6)
    effective_rank = int(np.count_nonzero(eigenvalues > effective_threshold))
    extent = np.ptp(positions, axis=0)
    return {
        "position_pca_eigenvalue_0_m2": first,
        "position_pca_eigenvalue_1_m2": second,
        "position_pca_eigenvalue_2_m2": third,
        "position_pca_effective_rank": effective_rank,
        "position_pca_linearity": (first / total) if total > 0.0 else 1.0,
        "position_pca_planarity": (
            (first + second) / total if total > 0.0 else 1.0
        ),
        "position_pca_second_to_first_ratio": (
            second / first if first > 0.0 else 0.0
        ),
        "position_pca_third_to_first_ratio": (
            third / first if first > 0.0 else 0.0
        ),
        "position_x_extent_m": float(extent[0]),
        "position_y_extent_m": float(extent[1]),
        "position_z_extent_m": float(extent[2]),
    }


def trajectory_geometry_metrics(trajectory: Trajectory) -> dict[str, float | int]:
    """Return reusable path-scale, shape, and orientation metrics."""

    positions = _trajectory_positions(trajectory)
    rotations = trajectory.poses_world_camera[:, :3, :3]
    relative_world = rotations[1:] @ np.swapaxes(rotations[:-1], 1, 2)
    adjacent_angles_deg = np.rad2deg(
        Rotation.from_matrix(relative_world).magnitude()
    )
    relative_to_q0 = np.swapaxes(rotations[0:1], 1, 2) @ rotations
    q0_angles_deg = np.rad2deg(
        Rotation.from_matrix(relative_to_q0).magnitude()
    )
    return {
        "accumulated_orientation_change_deg": float(adjacent_angles_deg.sum()),
        "maximum_displacement_from_q0_m": float(
            np.linalg.norm(positions - positions[0], axis=1).max()
        ),
        "maximum_orientation_change_from_q0_deg": float(q0_angles_deg.max()),
        "total_translation_path_length_m": float(
            np.linalg.norm(np.diff(positions, axis=0), axis=1).sum()
        ),
        **trajectory_position_pca_metrics(trajectory),
    }


def _requested_qe_binding(
    trajectory: Trajectory,
    qe_index: int,
    expected_evidence_frame_id: str,
) -> tuple[float, float, bool, bool]:
    """Validate requested ``qe1`` against its selected registered frame pose."""

    expected_frame_id = str(expected_evidence_frame_id).strip()
    if not expected_frame_id:
        raise ValueError("expected selected evidence frame ID must be non-empty")
    evidence_frame_id = str(trajectory.parameters.get("evidence_frame_id", "")).strip()
    frame_id_bound = evidence_frame_id == expected_frame_id
    expected_evidence = np.asarray(
        trajectory.parameters["evidence_pose_world_camera"], dtype=np.float64
    )
    if expected_evidence.shape != (4, 4):
        raise ValueError("trajectory evidence pose must have shape (4, 4)")
    translation_error, rotation_error = pose_error(
        trajectory.poses_world_camera[qe_index], expected_evidence
    )
    pose_bound = bool(
        translation_error <= REQUESTED_QE_TRANSLATION_TOLERANCE_M
        and rotation_error <= REQUESTED_QE_ROTATION_TOLERANCE_DEG
    )
    return translation_error, rotation_error, pose_bound, frame_id_bound


def _metric_template_axes(
    q0: np.ndarray,
    evidence: np.ndarray,
    *,
    direction_sign: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return evidence direction, deterministic lateral direction, and distance."""

    if direction_sign not in {-1, 1}:
        raise ValueError("direction_sign must be -1 or +1")
    offset = evidence[:3, 3] - q0[:3, 3]
    evidence_distance = float(np.linalg.norm(offset))
    if evidence_distance <= 0.05:
        raise TrajectoryFeasibilityError(
            "registered evidence translation must exceed 0.05m"
        )
    forward = offset / evidence_distance
    world_up = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    lateral = np.cross(world_up, forward)
    if np.linalg.norm(lateral) <= 1e-8:
        lateral = q0[:3, 0] - np.dot(q0[:3, 0], forward) * forward
    if np.linalg.norm(lateral) <= 1e-8:
        fallback = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
        lateral = fallback - np.dot(fallback, forward) * forward
    lateral /= np.linalg.norm(lateral)
    lateral *= int(direction_sign)
    return forward, lateral, evidence_distance


def _camera_right_lateral_axis(
    q0: np.ndarray,
    forward: np.ndarray,
    *,
    direction_sign: int,
) -> np.ndarray:
    """Return camera-right projected orthogonal to the evidence direction."""

    camera_right = np.asarray(q0[:3, 0], dtype=np.float64)
    lateral = camera_right - np.dot(camera_right, forward) * forward
    norm = float(np.linalg.norm(lateral))
    if norm <= 1e-8:
        raise TrajectoryFeasibilityError(
            "SpatialVID arc evidence direction is parallel to Q0 camera right"
        )
    return lateral / norm * int(direction_sign)


def _moving_completion_segment_holds() -> tuple[np.ndarray, np.ndarray]:
    """Return no artificial holds for natural moving-window templates."""

    return np.zeros(6, dtype=np.int64), np.zeros(6, dtype=np.int64)


def _natural_completion_parameters() -> dict[str, object]:
    """Return the parameters of the 48-frame moving revisit windows."""

    return {
        "completion_contract": COMPLETION_CONTRACT_NATURAL_MOVING,
        "completion_window_frame_count": COMPLETION_REVISIT_WINDOW_FRAME_COUNT,
        "completion_event_offset_in_window": (
            COMPLETION_REVISIT_EVENT_OFFSET_IN_WINDOW
        ),
        "completion_requires_distinct_event_poses": True,
    }


def _natural_completion_segment_minimum_intervals() -> np.ndarray:
    """Reserve canonical timing around the two 48-frame moving windows."""

    return np.asarray(
        [
            0,
            0,
            0,
            COMPLETION_REVISIT_EVENT_OFFSET_IN_WINDOW,
            2 * COMPLETION_REVISIT_EVENT_OFFSET_IN_WINDOW,
            0,
        ],
        dtype=np.int64,
    )


def _bind_completion_revisit(
    poses: np.ndarray, key_indices: Mapping[str, int]
) -> None:
    """Bind the exact revisit event to the first event pose."""

    poses[int(key_indices["qc_revisit"])] = poses[int(key_indices["qc_first"])]


def _turned_world_rotation(
    rotation_world_camera: np.ndarray,
    angle_deg: float,
) -> np.ndarray:
    yaw_world = Rotation.from_rotvec(
        np.asarray([0.0, 0.0, np.deg2rad(float(angle_deg))])
    ).as_matrix()
    return yaw_world @ np.asarray(rotation_world_camera, dtype=np.float64)


def _turned_camera_y_rotation(
    rotation_world_camera: np.ndarray,
    angle_deg: float,
) -> np.ndarray:
    """Turn about the starting camera's local OpenCV down axis."""

    yaw_camera = Rotation.from_rotvec(
        np.asarray([0.0, np.deg2rad(float(angle_deg)), 0.0])
    ).as_matrix()
    return np.asarray(rotation_world_camera, dtype=np.float64) @ yaw_camera


def _orbit_normalized_translation_derivative_maxima(
    start_angle: float,
    end_angle: float,
    start_radius: float,
    end_radius: float,
    start_height: float,
    end_height: float,
) -> tuple[float, float, float]:
    """Return normalized-time bounds for a quintic polar orbit segment."""

    tau = np.linspace(0.0, 1.0, 4097, dtype=np.float64)
    progress = _quintic_smoothstep(tau)
    first = 30.0 * tau**2 * (tau - 1.0) ** 2
    second = 60.0 * tau * (2.0 * tau**2 - 3.0 * tau + 1.0)
    third = 360.0 * tau**2 - 360.0 * tau + 60.0
    delta_angle = float(end_angle - start_angle)
    delta_radius = float(end_radius - start_radius)
    delta_height = float(end_height - start_height)
    angle = float(start_angle) + delta_angle * progress
    radius = float(start_radius) + delta_radius * progress
    radial = np.stack(
        [np.cos(angle), np.sin(angle), np.zeros_like(angle)], axis=1
    )
    tangential = np.stack(
        [-np.sin(angle), np.cos(angle), np.zeros_like(angle)], axis=1
    )
    height = np.asarray([0.0, 0.0, delta_height], dtype=np.float64)
    position_first = (
        delta_radius * radial
        + (radius * delta_angle)[:, None] * tangential
        + height[None]
    )
    position_second = (
        (-radius * delta_angle**2)[:, None] * radial
        + (2.0 * delta_radius * delta_angle) * tangential
    )
    position_third = (
        (-3.0 * delta_radius * delta_angle**2) * radial
        - (radius * delta_angle**3)[:, None] * tangential
    )
    velocity = position_first * first[:, None]
    acceleration = (
        position_second * first[:, None] ** 2
        + position_first * second[:, None]
    )
    jerk = (
        position_third * first[:, None] ** 3
        + 3.0
        * position_second
        * first[:, None]
        * second[:, None]
        + position_first * third[:, None]
    )
    return (
        _maximum_vector_norm(velocity),
        _maximum_vector_norm(acceleration),
        _maximum_vector_norm(jerk),
    )


def _interpolate_orbit_key_poses(
    frame_count: int,
    key_indices: Mapping[str, int],
    positions: np.ndarray,
    rotations: np.ndarray,
    angles: np.ndarray,
    radii: np.ndarray,
    heights: np.ndarray,
    focus_xy: np.ndarray,
    *,
    segment_start_holds: np.ndarray,
    segment_end_holds: np.ndarray,
) -> np.ndarray:
    """Interpolate canonical orbit SE(3) samples without cutting circle arcs."""

    poses = _interpolate_metric_key_poses(
        frame_count,
        key_indices,
        positions,
        rotations,
        segment_start_holds=segment_start_holds,
        segment_end_holds=segment_end_holds,
    )
    keys = list(key_indices.values())
    for segment, (start, end) in enumerate(zip(keys[:-1], keys[1:])):
        moving_start = start + int(segment_start_holds[segment])
        moving_end = end - int(segment_end_holds[segment])
        frame_indices = np.arange(moving_start, moving_end + 1)
        alpha = (frame_indices - moving_start) / (moving_end - moving_start)
        smooth = _quintic_smoothstep(alpha)
        segment_angles = (
            (1.0 - smooth) * angles[segment]
            + smooth * angles[segment + 1]
        )
        segment_radii = (
            (1.0 - smooth) * radii[segment]
            + smooth * radii[segment + 1]
        )
        segment_heights = (
            (1.0 - smooth) * heights[segment]
            + smooth * heights[segment + 1]
        )
        poses[frame_indices, 0, 3] = (
            float(focus_xy[0]) + segment_radii * np.cos(segment_angles)
        )
        poses[frame_indices, 1, 3] = (
            float(focus_xy[1]) + segment_radii * np.sin(segment_angles)
        )
        poses[frame_indices, 2, 3] = segment_heights
    return poses


def _raise_if_generated_dynamics_invalid(trajectory: Trajectory) -> None:
    metrics = trajectory_dynamics_metrics(trajectory)
    if bool(metrics["dynamics_valid"]):
        return
    failed = sorted(
        name
        for name, value in metrics.items()
        if name.endswith("_within_limit") and value is False
    )
    raise TrajectoryFeasibilityError(
        f"generated {trajectory.template_name} violates fixed trajectory "
        f"dynamics limits: {', '.join(failed)}"
    )


def _completion_revisit_geometry_metrics(
    trajectory: Trajectory,
) -> dict[str, float | bool]:
    """Measure the trajectory-local structure of the revisit windows."""

    contract = str(
        trajectory.parameters.get(
            "completion_contract",
            COMPLETION_CONTRACT_EXACT_HOLD,
        )
    )
    natural_contract = contract == COMPLETION_CONTRACT_NATURAL_MOVING
    frame_count = int(
        trajectory.parameters.get(
            "completion_window_frame_count",
            (
                COMPLETION_REVISIT_WINDOW_FRAME_COUNT
                if natural_contract
                else 2 * COMPLETION_REVISIT_HOLD_RADIUS_FRAMES + 1
            ),
        )
    )
    event_offset = int(
        trajectory.parameters.get(
            "completion_event_offset_in_window",
            (
                COMPLETION_REVISIT_EVENT_OFFSET_IN_WINDOW
                if natural_contract
                else COMPLETION_REVISIT_HOLD_RADIUS_FRAMES
            ),
        )
    )
    first_index = int(trajectory.key_indices["qc_first"])
    revisit_index = int(trajectory.key_indices["qc_revisit"])
    first_start = first_index - event_offset
    revisit_start = revisit_index - event_offset
    first_end = first_start + frame_count
    revisit_end = revisit_start + frame_count
    if (
        frame_count <= 1
        or not 0 <= event_offset < frame_count
        or first_start < 0
        or revisit_start < 0
        or first_end > len(trajectory.poses_world_camera)
        or revisit_end > len(trajectory.poses_world_camera)
    ):
        return {
            "completion_contract_natural_moving_windows": False,
            "completion_center_exact_revisit": False,
            "completion_exact_hold_windows": False,
            "completion_paired_moving_windows": False,
            "completion_natural_moving_windows": False,
            "completion_window_frame_count": float(frame_count),
            "completion_event_offset_in_window": float(event_offset),
            "completion_windows_nonstationary": False,
            "completion_increment_exact_replay": False,
            "completion_event_poses_distinct": False,
            "completion_first_window_rotation_deg": 0.0,
            "completion_revisit_window_rotation_deg": 0.0,
            "completion_first_window_motion_valid": False,
            "completion_revisit_window_motion_valid": False,
            "completion_window_separation_frames": -1.0,
            "completion_window_separation_valid": False,
        }
    poses = trajectory.poses_world_camera
    first_window = poses[first_start:first_end]
    revisit_window = poses[revisit_start:revisit_end]
    center_exact = bool(np.array_equal(poses[first_index], poses[revisit_index]))
    exact_windows = bool(np.array_equal(first_window, revisit_window))
    first_position_steps = np.linalg.norm(
        np.diff(first_window[:, :3, 3], axis=0), axis=1
    )
    revisit_position_steps = np.linalg.norm(
        np.diff(revisit_window[:, :3, 3], axis=0), axis=1
    )
    first_rotation_steps = Rotation.from_matrix(
        first_window[:-1, :3, :3].transpose(0, 2, 1)
        @ first_window[1:, :3, :3]
    ).magnitude()
    revisit_rotation_steps = Rotation.from_matrix(
        revisit_window[:-1, :3, :3].transpose(0, 2, 1)
        @ revisit_window[1:, :3, :3]
    ).magnitude()
    first_rotation_deg = float(np.rad2deg(first_rotation_steps).sum())
    revisit_rotation_deg = float(np.rad2deg(revisit_rotation_steps).sum())
    first_path_length = float(first_position_steps.sum())
    revisit_path_length = float(revisit_position_steps.sum())
    nonstationary = bool(
        np.all((first_position_steps > 1e-12) | (first_rotation_steps > 1e-12))
        and np.all((revisit_position_steps > 1e-12) | (revisit_rotation_steps > 1e-12))
    )
    increments_exact = bool(
        np.array_equal(
            first_window[:-1].transpose(0, 2, 1) @ first_window[1:],
            revisit_window[:-1].transpose(0, 2, 1) @ revisit_window[1:],
        )
    )
    separation = revisit_start - first_end
    separation_valid = bool(
        separation >= MINIMUM_COMPLETION_WINDOW_SEPARATION_FRAMES
        and first_end <= int(trajectory.key_indices["w2"]) <= revisit_start
    )
    event_translation, event_rotation = pose_error(
        poses[first_index],
        poses[revisit_index],
    )
    events_distinct = bool(
        event_translation > 1e-6 or event_rotation > 1e-4
    )
    # Do not introduce a trajectory-local minimum path/yaw value.  The
    # completion annotator owns the dataset-level comparison to surrounding
    # normal motion.  Here a window is valid only when every adjacent pair is
    # non-stationary in either translation or rotation.
    first_motion_valid = bool(
        np.all((first_position_steps > 1e-12) | (first_rotation_steps > 1e-12))
    )
    revisit_motion_valid = bool(
        np.all((revisit_position_steps > 1e-12) | (revisit_rotation_steps > 1e-12))
    )
    natural_moving = bool(
        natural_contract
        and frame_count == COMPLETION_REVISIT_WINDOW_FRAME_COUNT
        and event_offset == COMPLETION_REVISIT_EVENT_OFFSET_IN_WINDOW
        and nonstationary
        and first_motion_valid
        and revisit_motion_valid
        and separation_valid
        and first_start + event_offset == first_index
        and revisit_start + event_offset == revisit_index
    )
    paired_moving = bool(
        natural_contract
        and exact_windows
        and increments_exact
        and nonstationary
        and center_exact
        and first_start + event_offset == first_index
        and revisit_start + event_offset == revisit_index
    )
    exact_hold = bool(
        exact_windows and np.all(first_window == poses[first_index])
    )
    return {
        "completion_contract_natural_moving_windows": natural_contract,
        "completion_center_exact_revisit": center_exact,
        "completion_exact_hold_windows": exact_hold,
        "completion_paired_moving_windows": paired_moving,
        "completion_natural_moving_windows": natural_moving,
        "completion_window_frame_count": float(frame_count),
        "completion_event_offset_in_window": float(event_offset),
        "completion_windows_nonstationary": nonstationary,
        "completion_increment_exact_replay": increments_exact,
        "completion_event_poses_distinct": events_distinct,
        "completion_event_translation_m": event_translation,
        "completion_event_rotation_deg": event_rotation,
        "completion_first_window_path_length_m": first_path_length,
        "completion_revisit_window_path_length_m": revisit_path_length,
        "completion_first_window_rotation_deg": first_rotation_deg,
        "completion_revisit_window_rotation_deg": revisit_rotation_deg,
        "completion_first_window_motion_valid": first_motion_valid,
        "completion_revisit_window_motion_valid": revisit_motion_valid,
        "completion_window_separation_frames": float(separation),
        "completion_window_separation_valid": separation_valid,
    }


def _enforce_completion_hold_windows(
    poses: np.ndarray,
    key_indices: Mapping[str, int],
) -> None:
    """Hold the camera still around the revisit events for exact-hold templates."""

    radius = COMPLETION_REVISIT_HOLD_RADIUS_FRAMES
    first_index = int(key_indices["qc_first"])
    revisit_index = int(key_indices["qc_revisit"])
    completion_pose = np.array(poses[first_index], copy=True)
    poses[first_index - radius : first_index + radius + 1] = completion_pose
    poses[revisit_index - radius : revisit_index + radius + 1] = completion_pose


def _validate_canonical_metric_template_contract(
    trajectory: Trajectory,
    *,
    template_name: str,
) -> None:
    _validate_trajectory_components(
        trajectory.poses_world_camera,
        trajectory.key_indices,
        trajectory.duration_s,
        trajectory.fps,
    )
    if trajectory.template_name != template_name:
        raise ValueError(f"wrong template passed to {template_name} validator")
    if tuple(trajectory.key_indices) != METRIC_TRAJECTORY_EVENT_ROLES:
        raise ValueError("metric trajectory event roles do not match the template")
    if (
        trajectory.parameters.get("canonical_sampling") is not True
        or trajectory.fps != CANONICAL_METRIC_TRAJECTORY_FPS
        or len(trajectory.poses_world_camera)
        != CANONICAL_METRIC_TRAJECTORY_FRAME_COUNT
    ):
        raise ValueError("metric trajectory sampling must be 357 frames at 16 fps")
    if trajectory.key_indices["q0"] != 0 or trajectory.key_indices["q0_return"] != (
        len(trajectory.poses_world_camera) - 1
    ):
        raise ValueError("metric trajectory q0 roles must bound the sampled path")


def _coerce_route_control_poses(
    route_control_poses_world_camera: np.ndarray | Sequence[Sequence[Sequence[float]]],
    *,
    name: str,
) -> np.ndarray:
    """Validate a registered camera-pose polyline used as route evidence."""

    poses = np.asarray(route_control_poses_world_camera, dtype=np.float64)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4) or len(poses) < 2:
        raise ValueError(f"{name} must have shape (N, 4, 4) with N >= 2")
    if not np.isfinite(poses).all():
        raise ValueError(f"{name} must be finite")
    if not np.allclose(
        poses[:, 3, :],
        np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float64),
        rtol=0.0,
        atol=1e-6,
    ):
        raise ValueError(f"{name} must contain homogeneous c2w poses")
    rotations = poses[:, :3, :3]
    identity = np.eye(3, dtype=np.float64)
    if not np.allclose(
        rotations @ np.swapaxes(rotations, 1, 2),
        identity[None],
        rtol=0.0,
        atol=1e-5,
    ) or not np.allclose(
        np.linalg.det(rotations),
        np.ones(len(rotations), dtype=np.float64),
        rtol=0.0,
        atol=1e-5,
    ):
        raise ValueError(f"{name} must contain proper rotation matrices")
    return np.array(poses, copy=True)


def _coerce_route_index(
    value: int,
    *,
    name: str,
    maximum_exclusive: int,
) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    index = int(value)
    if not 0 <= index < maximum_exclusive:
        raise ValueError(f"{name} is outside the route control pose array")
    return index


def _interpolate_route_control_pose(
    start_pose: np.ndarray,
    end_pose: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Interpolate one registered-route edge without leaving its polyline."""

    alpha = float(alpha)
    if not np.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
        raise ValueError("route interpolation alpha must lie in [0, 1]")
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 3] = (
        (1.0 - alpha) * start_pose[:3, 3] + alpha * end_pose[:3, 3]
    )
    pose[:3, :3] = Slerp(
        [0.0, 1.0],
        Rotation.from_matrix(
            np.stack([start_pose[:3, :3], end_pose[:3, :3]], axis=0)
        ),
    )([alpha]).as_matrix()[0]
    return pose


def resample_route_polyline(
    route_control_poses_world_camera: np.ndarray | Sequence[Sequence[Sequence[float]]],
    sample_count: int,
) -> np.ndarray:
    """Resample a capture-route pose polyline by translation arc length."""

    route = _coerce_route_control_poses(
        route_control_poses_world_camera,
        name="route_control_poses_world_camera",
    )
    if isinstance(sample_count, (bool, np.bool_)) or not isinstance(
        sample_count, (int, np.integer)
    ):
        raise TypeError("sample_count must be an integer")
    sample_count = int(sample_count)
    if sample_count < len(route):
        raise ValueError(
            "sample_count must retain every route control pose; "
            f"got {sample_count} for {len(route)} controls"
        )

    positions = route[:, :3, 3]
    edge_lengths = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    total_length = float(edge_lengths.sum())
    if total_length <= 1e-8:
        raise TrajectoryFeasibilityError(
            "route polyline must contain non-zero translation length"
        )
    # One interval per route edge guarantees that all registered control poses
    # survive.  Allocate the remaining intervals by arc length, deterministically.
    interval_count = sample_count - 1
    edge_intervals = np.ones(len(edge_lengths), dtype=np.int64)
    extras_to_assign = interval_count - len(edge_lengths)
    if extras_to_assign:
        weights = edge_lengths / total_length
        quotas = extras_to_assign * weights
        extras = np.floor(quotas).astype(np.int64)
        remainder = extras_to_assign - int(extras.sum())
        order = np.argsort(-(quotas - extras), kind="stable")
        extras[order[:remainder]] += 1
        edge_intervals += extras

    pieces: list[np.ndarray] = []
    for edge_index, intervals in enumerate(edge_intervals):
        alpha = np.linspace(0.0, 1.0, int(intervals) + 1, dtype=np.float64)
        edge_samples = np.stack(
            [
                _interpolate_route_control_pose(
                    route[edge_index],
                    route[edge_index + 1],
                    float(value),
                )
                for value in alpha
            ],
            axis=0,
        )
        pieces.append(edge_samples if edge_index == 0 else edge_samples[1:])
    sampled = np.concatenate(pieces, axis=0)
    if len(sampled) != sample_count:
        raise AssertionError("route polyline resampling emitted the wrong sample count")
    sampled[0] = route[0]
    sampled[-1] = route[-1]
    return sampled


# A descriptive alias is useful to callers which distinguish pose-polylines
# from position-only paths.
resample_route_polyline_poses = resample_route_polyline


def _insert_route_pose_at_arc_fraction(
    route_poses: np.ndarray,
    *,
    end_index: int,
    fraction: float,
) -> tuple[np.ndarray, int, bool]:
    """Insert/locate a pose at a translation arc-length fraction of a prefix."""

    if not 0 < end_index < len(route_poses):
        raise ValueError("route prefix end_index must lie inside the route")
    fraction = float(fraction)
    if not np.isfinite(fraction) or not 0.0 < fraction < 1.0:
        raise ValueError("route arc fraction must lie strictly between zero and one")
    prefix_positions = route_poses[: end_index + 1, :3, 3]
    edge_lengths = np.linalg.norm(np.diff(prefix_positions, axis=0), axis=1)
    total_length = float(edge_lengths.sum())
    if total_length <= 1e-8:
        raise TrajectoryFeasibilityError(
            "q0-to-qe capture route must have non-zero translation length"
        )
    target = fraction * total_length
    cumulative = np.concatenate([[0.0], np.cumsum(edge_lengths)])
    vertex_matches = np.flatnonzero(np.isclose(cumulative, target, atol=1e-10))
    if len(vertex_matches):
        return np.array(route_poses, copy=True), int(vertex_matches[0]), False
    edge_index = int(np.searchsorted(cumulative, target, side="right") - 1)
    edge_index = min(max(edge_index, 0), end_index - 1)
    edge_length = float(edge_lengths[edge_index])
    if edge_length <= 1e-12:
        raise TrajectoryFeasibilityError(
            "q0-to-qe capture route contains an unusable zero-length edge"
        )
    alpha = (target - cumulative[edge_index]) / edge_length
    inserted = _interpolate_route_control_pose(
        route_poses[edge_index],
        route_poses[edge_index + 1],
        float(np.clip(alpha, 0.0, 1.0)),
    )
    insertion_index = edge_index + 1
    expanded = np.concatenate(
        [
            route_poses[:insertion_index],
            inserted[None],
            route_poses[insertion_index:],
        ],
        axis=0,
    )
    return expanded, insertion_index, True


def _pose_matches_route_control(
    actual_pose: np.ndarray,
    expected_pose: np.ndarray,
) -> tuple[float, float, bool]:
    translation_error, rotation_error = pose_error(actual_pose, expected_pose)
    valid = bool(
        translation_error <= REQUESTED_QE_TRANSLATION_TOLERANCE_M
        and rotation_error <= REQUESTED_QE_ROTATION_TOLERANCE_DEG
    )
    return translation_error, rotation_error, valid


def _require_route_control_match(
    actual_pose: np.ndarray,
    expected_pose: np.ndarray,
    *,
    name: str,
) -> None:
    translation_error, rotation_error, valid = _pose_matches_route_control(
        actual_pose, expected_pose
    )
    if not valid:
        raise TrajectoryFeasibilityError(
            f"{name} must bind the same registered route control pose "
            f"(translation_error_m={translation_error:.9f}, "
            f"rotation_error_deg={rotation_error:.9f})"
        )


def _allocate_route_polyline_frame_indices(
    route_poses: np.ndarray,
    *,
    frame_count: int,
    fps: int,
    limits: TrajectoryDynamicsLimits,
    start_holds: np.ndarray,
    end_holds: np.ndarray,
    template_name: str,
) -> np.ndarray:
    """Allocate canonical timing for every capture-route polyline edge."""

    segment_count = len(route_poses) - 1
    if start_holds.shape != (segment_count,) or end_holds.shape != (
        segment_count,
    ):
        raise ValueError("route hold arrays must have one value per route edge")
    if np.any(start_holds < 0) or np.any(end_holds < 0):
        raise ValueError("route holds must be non-negative")
    rotations = route_poses[:, :3, :3]
    relative = rotations[:-1].transpose(0, 2, 1) @ rotations[1:]
    angles_deg = np.rad2deg(Rotation.from_matrix(relative).magnitude())
    derivatives = np.asarray(
        [
            _normalized_translation_derivative_maxima(start, end)
            for start, end in zip(route_poses[:-1, :3, 3], route_poses[1:, :3, 3])
        ],
        dtype=np.float64,
    )
    angular_normalized = np.stack(
        [
            1.875 * angles_deg,
            (10.0 / np.sqrt(3.0)) * angles_deg,
            60.0 * angles_deg,
        ],
        axis=1,
    )
    required_times = np.maximum.reduce(
        [
            derivatives[:, 0] / limits.maximum_linear_speed_m_s,
            np.sqrt(
                derivatives[:, 1] / limits.maximum_linear_acceleration_m_s2
            ),
            np.cbrt(derivatives[:, 2] / limits.maximum_linear_jerk_m_s3),
            angular_normalized[:, 0] / limits.maximum_angular_speed_deg_s,
            np.sqrt(
                angular_normalized[:, 1]
                / limits.maximum_angular_acceleration_deg_s2
            ),
            np.cbrt(
                angular_normalized[:, 2] / limits.maximum_angular_jerk_deg_s3
            ),
        ]
    )
    moving_intervals = np.maximum(
        2, np.ceil(required_times * int(fps)).astype(np.int64) + 1
    )
    minimum_intervals = moving_intervals + start_holds + end_holds
    available_intervals = int(frame_count) - 1
    required_intervals = int(minimum_intervals.sum())
    if required_intervals > available_intervals:
        raise TrajectoryFeasibilityError(
            "canonical 357-frame clock cannot satisfy fixed dynamics for "
            f"route-conforming {template_name} "
            f"(required intervals={required_intervals}, "
            f"available={available_intervals})"
        )
    slack = available_intervals - required_intervals
    weights = derivatives[:, 0] + angles_deg / max(
        limits.maximum_angular_speed_deg_s, 1e-9
    )
    if not np.any(weights > 0.0):
        weights = np.ones_like(weights)
    quotas = slack * weights / weights.sum()
    extras = np.floor(quotas).astype(np.int64)
    remainder = slack - int(extras.sum())
    order = np.argsort(-(quotas - extras), kind="stable")
    extras[order[:remainder]] += 1
    indices = np.concatenate([[0], np.cumsum(minimum_intervals + extras)]).astype(
        int
    )
    if indices[-1] != frame_count - 1:
        raise AssertionError(
            "route-conforming timing did not consume the canonical clock"
        )
    return indices


def _interpolate_route_polyline_poses(
    route_poses: np.ndarray,
    control_frame_indices: np.ndarray,
    *,
    start_holds: np.ndarray,
    end_holds: np.ndarray,
) -> np.ndarray:
    """Sample a route polyline without cutting capture-route corners."""

    if control_frame_indices.ndim != 1 or len(control_frame_indices) != len(
        route_poses
    ):
        raise ValueError("route control frame indices must match route pose count")
    frame_count = int(control_frame_indices[-1]) + 1
    poses = np.repeat(np.eye(4, dtype=np.float64)[None], frame_count, axis=0)
    for segment, (start, end) in enumerate(
        zip(control_frame_indices[:-1], control_frame_indices[1:])
    ):
        start = int(start)
        end = int(end)
        moving_start = start + int(start_holds[segment])
        moving_end = end - int(end_holds[segment])
        if moving_end - moving_start < 2:
            raise TrajectoryFeasibilityError(
                "canonical route edge interval is too short after completion holds"
            )
        start_pose = route_poses[segment]
        end_pose = route_poses[segment + 1]
        poses[start : moving_start + 1] = start_pose
        poses[moving_end : end + 1] = end_pose
        frame_indices = np.arange(moving_start, moving_end + 1)
        alpha = (frame_indices - moving_start) / (moving_end - moving_start)
        smooth = _quintic_smoothstep(alpha)
        poses[frame_indices, :3, 3] = (
            (1.0 - smooth[:, None]) * start_pose[:3, 3]
            + smooth[:, None] * end_pose[:3, 3]
        )
        poses[frame_indices, :3, :3] = Slerp(
            [0.0, 1.0],
            Rotation.from_matrix(
                np.stack([start_pose[:3, :3], end_pose[:3, :3]], axis=0)
            ),
        )(smooth).as_matrix()
    return poses


def _coerce_route_polyline_positions(
    route_polyline: np.ndarray | Sequence[Sequence[float]] | None,
    trajectory: Trajectory,
) -> np.ndarray:
    if route_polyline is None:
        route_polyline = trajectory.parameters.get("route_polyline_world_m")
    if route_polyline is None:
        raise ValueError(
            "route polyline is required unless trajectory stores "
            "route_polyline_world_m"
        )
    array = np.asarray(route_polyline, dtype=np.float64)
    if array.ndim == 3 and array.shape[1:] == (4, 4):
        array = array[:, :3, 3]
    if array.ndim != 2 or array.shape[1] != 3 or len(array) < 2:
        raise ValueError("route polyline must have shape (N, 3) or (N, 4, 4)")
    if not np.isfinite(array).all():
        raise ValueError("route polyline must be finite")
    return np.array(array, copy=True)


def _point_to_polyline_distances(
    points: np.ndarray,
    polyline: np.ndarray,
) -> np.ndarray:
    """Return Euclidean distance from each point to a piecewise-linear route."""

    starts = polyline[:-1]
    edges = polyline[1:] - starts
    lengths_squared = np.einsum("ij,ij->i", edges, edges)
    deltas = points[:, None, :] - starts[None, :, :]
    projections = np.einsum("pse,se->ps", deltas, edges)
    alpha = np.divide(
        projections,
        lengths_squared[None, :],
        out=np.zeros_like(projections),
        where=lengths_squared[None, :] > 1e-12,
    )
    alpha = np.clip(alpha, 0.0, 1.0)
    closest = starts[None, :, :] + alpha[:, :, None] * edges[None, :, :]
    distances = np.linalg.norm(points[:, None, :] - closest, axis=2)
    return np.min(distances, axis=1)


def trajectory_route_corridor_metrics(
    trajectory: Trajectory,
    *,
    route_polyline_world_m: np.ndarray | Sequence[Sequence[float]] | None = None,
    corridor_radius_m: float | None = None,
) -> dict[str, float | bool]:
    """Measure whether every sampled camera centre stays in a route corridor."""

    positions = _trajectory_positions(trajectory)
    polyline = _coerce_route_polyline_positions(route_polyline_world_m, trajectory)
    if corridor_radius_m is None:
        corridor_radius_m = trajectory.parameters.get("route_corridor_radius_m")
    if corridor_radius_m is None:
        raise ValueError("corridor_radius_m is required for route-corridor metrics")
    corridor_radius_m = float(corridor_radius_m)
    if not np.isfinite(corridor_radius_m) or corridor_radius_m <= 0.0:
        raise ValueError("corridor_radius_m must be finite and positive")
    distances = _point_to_polyline_distances(positions, polyline)
    within = distances <= corridor_radius_m + 1e-9
    polyline_length = float(np.linalg.norm(np.diff(polyline, axis=0), axis=1).sum())
    return {
        "route_corridor_radius_m": corridor_radius_m,
        "route_corridor_max_distance_m": float(distances.max()),
        "route_corridor_mean_distance_m": float(distances.mean()),
        "route_corridor_within_fraction": float(np.mean(within)),
        "route_corridor_valid": bool(np.all(within)),
        "route_polyline_length_m": polyline_length,
    }


def _polyline_turning_angle_deg(positions: np.ndarray) -> float:
    steps = np.diff(positions, axis=0)
    norms = np.linalg.norm(steps, axis=1)
    directions = steps[norms > 1e-8] / norms[norms > 1e-8, None]
    if len(directions) < 2:
        return 0.0
    cosine = np.sum(directions[:-1] * directions[1:], axis=1)
    return float(np.rad2deg(np.arccos(np.clip(cosine, -1.0, 1.0))).sum())


def _route_family_axes(
    trajectory: Trajectory,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return evidence and macro forward directions plus an evidence lateral."""

    positions = _trajectory_positions(trajectory)
    q0 = positions[trajectory.key_indices["q0"]]
    qe = positions[trajectory.key_indices["qe1"]]
    remote = positions[trajectory.key_indices["w2"]]
    evidence_direction = qe - q0
    if np.linalg.norm(evidence_direction) <= 1e-8:
        raise ValueError("route family metrics require q0 and qe translation")
    evidence_direction /= np.linalg.norm(evidence_direction)
    macro_direction = remote - q0
    if np.linalg.norm(macro_direction) <= 1e-8:
        macro_direction = evidence_direction.copy()
    else:
        macro_direction /= np.linalg.norm(macro_direction)
    world_up = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    lateral = np.cross(world_up, evidence_direction)
    if np.linalg.norm(lateral) <= 1e-8:
        lateral = trajectory.poses_world_camera[0, :3, 0].copy()
        lateral -= evidence_direction * float(lateral @ evidence_direction)
    if np.linalg.norm(lateral) <= 1e-8:
        lateral = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
        lateral -= evidence_direction * float(lateral @ evidence_direction)
    lateral /= np.linalg.norm(lateral)
    return evidence_direction, macro_direction, lateral


def trajectory_route_family_metrics(
    trajectory: Trajectory,
    *,
    minimum_forward_progress_m: float = 1.5,
    minimum_lateral_range_m: float = 0.75,
    minimum_arc_bow_m: float = 0.50,
) -> dict[str, float | bool]:
    """Classify route-conforming forward/lateral/arc topology from samples."""

    thresholds = np.asarray(
        [
            minimum_forward_progress_m,
            minimum_lateral_range_m,
            minimum_arc_bow_m,
        ],
        dtype=np.float64,
    )
    if not np.isfinite(thresholds).all() or np.any(thresholds < 0.0):
        raise ValueError("route family thresholds must be finite and non-negative")
    if trajectory.template_name not in ROUTE_CONFORMING_TEMPLATE_NAMES:
        raise ValueError(
            "route family metrics support forward_backward, "
            "lateral_out_and_back, and arc_return"
        )
    positions = _trajectory_positions(trajectory)
    q0_index = int(trajectory.key_indices["q0"])
    remote_index = int(trajectory.key_indices["w2"])
    evidence_direction, macro_direction, lateral_axis = _route_family_axes(
        trajectory
    )
    origin = positions[q0_index]
    displacement = positions - origin
    evidence_progress = displacement @ evidence_direction
    macro_progress = displacement @ macro_direction
    lateral = displacement @ lateral_axis
    outbound = slice(q0_index, remote_index + 1)
    returning = slice(remote_index, len(positions))
    outbound_positions = positions[outbound]
    outbound_path_length = float(
        np.linalg.norm(np.diff(outbound_positions, axis=0), axis=1).sum()
    )
    remote_macro_progress = float(macro_progress[remote_index])
    forward_efficiency = (
        remote_macro_progress / outbound_path_length
        if outbound_path_length > 1e-8
        else 0.0
    )
    lateral_range = float(np.max(lateral) - np.min(lateral))
    outbound_max_lateral = float(np.max(lateral[outbound]))
    outbound_min_lateral = float(np.min(lateral[outbound]))
    return_max_lateral = float(np.max(lateral[returning]))
    return_min_lateral = float(np.min(lateral[returning]))
    outbound_sign = (
        1.0
        if abs(outbound_max_lateral) >= abs(outbound_min_lateral)
        else -1.0
    )
    outbound_bow = (
        outbound_max_lateral if outbound_sign > 0.0 else -outbound_min_lateral
    )
    return_opposite_bow = (
        -return_min_lateral if outbound_sign > 0.0 else return_max_lateral
    )
    forward_valid = bool(
        remote_macro_progress >= float(minimum_forward_progress_m)
        and forward_efficiency >= 0.25
    )
    lateral_valid = bool(lateral_range >= float(minimum_lateral_range_m))
    arc_valid = bool(
        outbound_bow >= float(minimum_arc_bow_m)
        and return_opposite_bow >= float(minimum_arc_bow_m)
    )
    family_valid = {
        "forward_backward": forward_valid,
        "lateral_out_and_back": lateral_valid,
        "arc_return": arc_valid,
    }[trajectory.template_name]
    return {
        "route_arc_outbound_bow_m": float(outbound_bow),
        "route_arc_outbound_side_sign": float(outbound_sign),
        "route_arc_return_opposite_bow_m": float(return_opposite_bow),
        "route_evidence_max_progress_m": float(np.max(evidence_progress)),
        "route_family_valid": bool(family_valid),
        "route_forward_efficiency": float(forward_efficiency),
        "route_macro_max_progress_m": float(np.max(macro_progress)),
        "route_macro_remote_progress_m": remote_macro_progress,
        "route_lateral_range_m": lateral_range,
        "route_outbound_turning_angle_deg": _polyline_turning_angle_deg(
            outbound_positions
        ),
        "route_return_turning_angle_deg": _polyline_turning_angle_deg(
            positions[returning]
        ),
        "route_forward_family_valid": forward_valid,
        "route_lateral_family_valid": lateral_valid,
        "route_arc_family_valid": arc_valid,
    }


def _route_control_binding_metrics(
    trajectory: Trajectory,
) -> dict[str, float | bool]:
    controls = trajectory.parameters.get("route_control_poses_world_camera")
    if not isinstance(controls, Mapping):
        raise ValueError(
            "route-conforming trajectory must store route_control_poses_world_camera"
        )
    required = ("q0", "qe1", "qc", "remote")
    expected: dict[str, np.ndarray] = {}
    for name in required:
        if name not in controls:
            raise ValueError(f"route control poses are missing {name}")
        pose = np.asarray(controls[name], dtype=np.float64)
        if pose.shape != (4, 4):
            raise ValueError(f"route control {name} must have shape (4, 4)")
        expected[name] = pose

    q0_error_t, q0_error_r, q0_valid = _pose_matches_route_control(
        trajectory.poses_world_camera[trajectory.key_indices["q0"]],
        expected["q0"],
    )
    q0_return_error_t, q0_return_error_r, q0_return_valid = (
        _pose_matches_route_control(
            trajectory.poses_world_camera[trajectory.key_indices["q0_return"]],
            expected["q0"],
        )
    )
    qe_error_t, qe_error_r, qe_valid = _pose_matches_route_control(
        trajectory.poses_world_camera[trajectory.key_indices["qe1"]],
        expected["qe1"],
    )
    qc_first_error_t, qc_first_error_r, qc_first_valid = _pose_matches_route_control(
        trajectory.poses_world_camera[trajectory.key_indices["qc_first"]],
        expected["qc"],
    )
    qc_revisit_error_t, qc_revisit_error_r, qc_revisit_valid = (
        _pose_matches_route_control(
            trajectory.poses_world_camera[trajectory.key_indices["qc_revisit"]],
            expected["qc"],
        )
    )
    remote_error_t, remote_error_r, remote_valid = _pose_matches_route_control(
        trajectory.poses_world_camera[trajectory.key_indices["w2"]],
        expected["remote"],
    )
    all_valid = bool(
        q0_valid
        and q0_return_valid
        and qe_valid
        and qc_first_valid
        and qc_revisit_valid
        and remote_valid
    )
    return {
        "q0_route_control_bound": q0_valid,
        "q0_route_control_rotation_error_deg": q0_error_r,
        "q0_route_control_translation_error_m": q0_error_t,
        "q0_return_route_control_bound": q0_return_valid,
        "q0_return_route_control_rotation_error_deg": q0_return_error_r,
        "q0_return_route_control_translation_error_m": q0_return_error_t,
        "qc_first_route_control_bound": qc_first_valid,
        "qc_first_route_control_rotation_error_deg": qc_first_error_r,
        "qc_first_route_control_translation_error_m": qc_first_error_t,
        "qc_revisit_route_control_bound": qc_revisit_valid,
        "qc_revisit_route_control_rotation_error_deg": qc_revisit_error_r,
        "qc_revisit_route_control_translation_error_m": qc_revisit_error_t,
        "qe1_route_control_bound": qe_valid,
        "qe1_route_control_rotation_error_deg": qe_error_r,
        "qe1_route_control_translation_error_m": qe_error_t,
        "remote_route_control_bound": remote_valid,
        "remote_route_control_rotation_error_deg": remote_error_r,
        "remote_route_control_translation_error_m": remote_error_t,
        "route_control_binding_valid": all_valid,
    }


def generate_route_conforming_trajectory(
    route_control_poses_world_camera: np.ndarray
    | Sequence[Sequence[Sequence[float]]],
    *,
    q0_route_index: int,
    qe_route_index: int,
    qc_route_index: int,
    remote_route_index: int,
    evidence_frame_id: str,
    template_name: str = "forward_backward",
    return_route_control_poses_world_camera: np.ndarray
    | Sequence[Sequence[Sequence[float]]]
    | None = None,
    return_qc_route_index: int | None = None,
    route_node_ids: Sequence[str] | None = None,
    return_route_node_ids: Sequence[str] | None = None,
    minimum_clearance_m: float = 0.20,
    route_corridor_radius_m: float = 0.25,
    dynamics_limits: TrajectoryDynamicsLimits | Mapping[str, object] | None = None,
    canonical_sampling: bool = True,
) -> Trajectory:
    """Generate a canonical trajectory constrained to real capture polylines."""

    template_name = str(template_name).strip()
    if template_name not in ROUTE_CONFORMING_TEMPLATE_NAMES:
        raise ValueError(
            "route-conforming template_name must be one of "
            f"{sorted(ROUTE_CONFORMING_TEMPLATE_NAMES)}"
        )
    if not isinstance(canonical_sampling, (bool, np.bool_)):
        raise TypeError("canonical_sampling must be a boolean")
    if not bool(canonical_sampling):
        raise ValueError(
            "route-conforming trajectories are defined only on the fixed "
            "357-frame/16-FPS canonical clock"
        )
    evidence_frame_id = str(evidence_frame_id).strip()
    if not evidence_frame_id:
        raise ValueError("evidence_frame_id must be non-empty")
    minimum_clearance_m = float(minimum_clearance_m)
    route_corridor_radius_m = float(route_corridor_radius_m)
    if not np.isfinite(minimum_clearance_m) or minimum_clearance_m <= 0.0:
        raise ValueError("minimum_clearance_m must be finite and positive")
    if not np.isfinite(route_corridor_radius_m) or route_corridor_radius_m <= 0.0:
        raise ValueError("route_corridor_radius_m must be finite and positive")

    source_route = _coerce_route_control_poses(
        route_control_poses_world_camera,
        name="route_control_poses_world_camera",
    )
    q0_source_index = _coerce_route_index(
        q0_route_index,
        name="q0_route_index",
        maximum_exclusive=len(source_route),
    )
    qe_source_index = _coerce_route_index(
        qe_route_index,
        name="qe_route_index",
        maximum_exclusive=len(source_route),
    )
    qc_source_index = _coerce_route_index(
        qc_route_index,
        name="qc_route_index",
        maximum_exclusive=len(source_route),
    )
    remote_source_index = _coerce_route_index(
        remote_route_index,
        name="remote_route_index",
        maximum_exclusive=len(source_route),
    )
    if not (
        q0_source_index < qe_source_index < qc_source_index < remote_source_index
    ):
        raise ValueError(
            "route controls must occur in strict outbound order q0 < qe < qc < remote"
        )
    if route_node_ids is not None and len(route_node_ids) != len(source_route):
        raise ValueError("route_node_ids must align with route_control_poses_world_camera")

    outbound = np.array(
        source_route[q0_source_index : remote_source_index + 1],
        copy=True,
    )
    qe_local = qe_source_index - q0_source_index
    qc_local = qc_source_index - q0_source_index
    outbound, w1_local, inserted_w1 = _insert_route_pose_at_arc_fraction(
        outbound,
        end_index=qe_local,
        fraction=0.5,
    )
    if inserted_w1:
        qe_local += 1
        qc_local += 1

    outbound_ids: list[str] | None = None
    if route_node_ids is not None:
        outbound_ids = [
            str(node_id)
            for node_id in route_node_ids[
                q0_source_index : remote_source_index + 1
            ]
        ]
        if inserted_w1:
            outbound_ids.insert(w1_local, "interpolated_w1")

    q0_pose = np.array(outbound[0], copy=True)
    qe_pose = np.array(outbound[qe_local], copy=True)
    qc_pose = np.array(outbound[qc_local], copy=True)
    remote_pose = np.array(outbound[-1], copy=True)
    if np.linalg.norm(qe_pose[:3, 3] - q0_pose[:3, 3]) <= 0.05:
        raise TrajectoryFeasibilityError(
            "registered qe route control must translate more than 0.05m from q0"
        )

    explicit_return_route = return_route_control_poses_world_camera is not None
    if return_route_control_poses_world_camera is None:
        return_route = np.array(outbound[::-1], copy=True)
        return_qc_local = len(return_route) - 1 - qc_local
        return_ids = (
            None if outbound_ids is None else list(reversed(outbound_ids))
        )
    else:
        return_route = _coerce_route_control_poses(
            return_route_control_poses_world_camera,
            name="return_route_control_poses_world_camera",
        )
        if return_route_node_ids is not None and len(return_route_node_ids) != len(
            return_route
        ):
            raise ValueError(
                "return_route_node_ids must align with "
                "return_route_control_poses_world_camera"
            )
        return_ids = (
            None
            if return_route_node_ids is None
            else [str(node_id) for node_id in return_route_node_ids]
        )
        _require_route_control_match(
            return_route[0],
            remote_pose,
            name="return route start",
        )
        _require_route_control_match(
            return_route[-1],
            q0_pose,
            name="return route end",
        )
        if return_qc_route_index is None:
            matching = [
                index
                for index, pose in enumerate(return_route)
                if _pose_matches_route_control(pose, qc_pose)[2]
            ]
            if not matching:
                raise TrajectoryFeasibilityError(
                    "explicit return route must contain the registered qc control pose"
                )
            return_qc_local = int(matching[0])
        else:
            return_qc_local = _coerce_route_index(
                return_qc_route_index,
                name="return_qc_route_index",
                maximum_exclusive=len(return_route),
            )
        if not 0 < return_qc_local < len(return_route) - 1:
            raise ValueError(
                "return_qc_route_index must be strictly between remote and q0"
            )
        _require_route_control_match(
            return_route[return_qc_local],
            qc_pose,
            name="return qc route control",
        )

    # The remote control is shared by the two polylines.  Keep one copy in the
    # full sampled route so no zero-length remote segment is invented.
    full_route = np.concatenate([outbound, return_route[1:]], axis=0)
    remote_local = len(outbound) - 1
    qc_revisit_local = remote_local + return_qc_local
    if not (remote_local < qc_revisit_local < len(full_route) - 1):
        raise AssertionError("return route control indexing is inconsistent")
    event_control_indices = {
        "q0": 0,
        "w1": int(w1_local),
        "qe1": int(qe_local),
        "qc_first": int(qc_local),
        "w2": int(remote_local),
        "qc_revisit": int(qc_revisit_local),
        "q0_return": len(full_route) - 1,
    }
    if list(event_control_indices.values()) != sorted(
        event_control_indices.values()
    ) or len(set(event_control_indices.values())) != len(event_control_indices):
        raise TrajectoryFeasibilityError(
            "capture-route controls cannot realize the seven canonical event roles"
        )

    segment_count = len(full_route) - 1
    start_holds = np.zeros(segment_count, dtype=np.int64)
    end_holds = np.zeros(segment_count, dtype=np.int64)
    radius = COMPLETION_REVISIT_HOLD_RADIUS_FRAMES
    for completion_local in (
        event_control_indices["qc_first"],
        event_control_indices["qc_revisit"],
    ):
        end_holds[completion_local - 1] = radius
        start_holds[completion_local] = radius
    limits = _coerce_dynamics_limits(dynamics_limits)
    control_frame_indices = _allocate_route_polyline_frame_indices(
        full_route,
        frame_count=CANONICAL_METRIC_TRAJECTORY_FRAME_COUNT,
        fps=CANONICAL_METRIC_TRAJECTORY_FPS,
        limits=limits,
        start_holds=start_holds,
        end_holds=end_holds,
        template_name=template_name,
    )
    key_indices = {
        role: int(control_frame_indices[control_index])
        for role, control_index in event_control_indices.items()
    }
    poses = _interpolate_route_polyline_poses(
        full_route,
        control_frame_indices,
        start_holds=start_holds,
        end_holds=end_holds,
    )
    # Make all registered control bindings bit-exact, including both completion
    # centres.  Holds are written afterwards to preserve bit-exact windows.
    poses[key_indices["q0"]] = q0_pose
    poses[key_indices["qe1"]] = qe_pose
    poses[key_indices["qc_first"]] = qc_pose
    poses[key_indices["w2"]] = remote_pose
    poses[key_indices["qc_revisit"]] = qc_pose
    _enforce_completion_hold_windows(poses, key_indices)
    poses[key_indices["q0_return"]] = q0_pose

    if outbound_ids is not None and return_ids is not None:
        full_route_ids = outbound_ids + return_ids[1:]
    else:
        full_route_ids = None
    trajectory = Trajectory(
        template_name=template_name,
        difficulty=(
            "hard"
            if template_name == "arc_return"
            else "medium"
            if template_name == "lateral_out_and_back"
            else "easy"
        ),
        duration_s=(
            CANONICAL_METRIC_TRAJECTORY_FRAME_COUNT
            / CANONICAL_METRIC_TRAJECTORY_FPS
        ),
        fps=CANONICAL_METRIC_TRAJECTORY_FPS,
        poses_world_camera=poses,
        key_indices=key_indices,
        parameters={
            "canonical_sampling": True,
            "completion_hold_radius_frames": COMPLETION_REVISIT_HOLD_RADIUS_FRAMES,
            "dynamics_limits": asdict(limits),
            "evidence_frame_id": evidence_frame_id,
            "evidence_pose_world_camera": qe_pose.tolist(),
            "minimum_clearance_m": minimum_clearance_m,
            "route_conforming": True,
            "route_control_indices": {
                "q0_source": q0_source_index,
                "qe_source": qe_source_index,
                "qc_source": qc_source_index,
                "remote_source": remote_source_index,
                "return_qc": int(return_qc_local),
            },
            "route_control_poses_world_camera": {
                "q0": q0_pose.tolist(),
                "qe1": qe_pose.tolist(),
                "qc": qc_pose.tolist(),
                "remote": remote_pose.tolist(),
            },
            "route_corridor_radius_m": route_corridor_radius_m,
            "route_node_ids": full_route_ids,
            "route_polyline_world_m": full_route[:, :3, 3].tolist(),
            "route_return_is_explicit": bool(explicit_return_route),
            "route_topology": "registered_capture_polyline",
        },
    )
    _raise_if_generated_dynamics_invalid(trajectory)
    return trajectory


# Explicit aliases make the capture-route intent easy to discover without
# forcing callers to remember the generic generator name.
generate_capture_route_trajectory = generate_route_conforming_trajectory


def validate_route_conforming_trajectory(
    trajectory: Trajectory,
    raycaster: MeshRaycaster,
    *,
    expected_evidence_frame_id: str,
    dynamics_limits: TrajectoryDynamicsLimits | Mapping[str, object] | None = None,
    corridor_radius_m: float | None = None,
) -> TrajectoryValidation:
    """Validate route binding, corridor conformance, collision, and topology."""

    if trajectory.template_name not in ROUTE_CONFORMING_TEMPLATE_NAMES:
        raise ValueError(
            "route-conforming validator supports forward_backward, "
            "lateral_out_and_back, and arc_return"
        )
    _validate_canonical_metric_template_contract(
        trajectory,
        template_name=trajectory.template_name,
    )
    if trajectory.parameters.get("route_conforming") is not True:
        raise ValueError(
            "route-conforming validator requires route_conforming=True provenance"
        )
    poses = trajectory.poses_world_camera
    exact_return = bool(np.array_equal(poses[-1], poses[0]))
    completion = _completion_revisit_geometry_metrics(trajectory)
    qe_translation_error, qe_rotation_error, qe_pose_bound, qe_id_bound = (
        _requested_qe_binding(
            trajectory,
            trajectory.key_indices["qe1"],
            expected_evidence_frame_id,
        )
    )
    route_bindings = _route_control_binding_metrics(trajectory)
    corridor = trajectory_route_corridor_metrics(
        trajectory,
        corridor_radius_m=corridor_radius_m,
    )
    family = trajectory_route_family_metrics(trajectory)
    dynamics = trajectory_dynamics_metrics(trajectory, dynamics_limits)
    collision = _validate_trajectory_mesh_clearance(trajectory, raycaster)
    hard_clearance = float(trajectory.parameters["minimum_clearance_m"])
    nominal_clearance = float(
        trajectory.parameters.get("nominal_minimum_clearance_m", hard_clearance)
    )
    if (
        not np.isfinite(nominal_clearance)
        or nominal_clearance < hard_clearance
    ):
        raise ValueError(
            "route-conforming nominal clearance must be finite and no smaller "
            "than the hard clearance"
        )
    mesh_crossing_free = not bool(collision.intersected_segment_indices)
    nominal_clearance_pass = bool(
        collision.minimum_clearance_m >= nominal_clearance
    )
    hidden_near_surface_diagnostic = bool(
        mesh_crossing_free and not nominal_clearance_pass
    )
    passed = bool(
        exact_return
        and collision.valid
        and bool(route_bindings["route_control_binding_valid"])
        and qe_pose_bound
        and qe_id_bound
        and bool(corridor["route_corridor_valid"])
        and bool(family["route_family_valid"])
        and bool(completion["completion_center_exact_revisit"])
        and bool(completion["completion_exact_hold_windows"])
        and bool(completion["completion_window_separation_valid"])
        and bool(dynamics["dynamics_valid"])
    )
    metrics = {
        "qe1_rotation_error_deg": qe_rotation_error,
        "qe1_translation_error_m": qe_translation_error,
        "qe1_registered_binding_valid": qe_pose_bound and qe_id_bound,
        "qe1_registered_frame_id_bound": qe_id_bound,
        "qe1_registered_pose_bound": qe_pose_bound,
        "qe1_requested_rotation_tolerance_deg": REQUESTED_QE_ROTATION_TOLERANCE_DEG,
        "qe1_requested_translation_tolerance_m": REQUESTED_QE_TRANSLATION_TOLERANCE_M,
        "route_collision_hard_clearance_m": hard_clearance,
        "route_collision_nominal_clearance_m": nominal_clearance,
        "route_collision_mesh_crossing_free": mesh_crossing_free,
        "route_collision_nominal_clearance_pass": nominal_clearance_pass,
        "route_collision_hidden_near_surface_diagnostic": (
            hidden_near_surface_diagnostic
        ),
        **trajectory_geometry_metrics(trajectory),
        **route_bindings,
        **corridor,
        **family,
        **completion,
        **dynamics,
    }
    return TrajectoryValidation(
        passed,
        trajectory.template_name,
        exact_return,
        collision,
        metrics,
    )


validate_capture_route_trajectory = validate_route_conforming_trajectory


def generate_lateral_out_and_back(
    q0_pose_world_camera: np.ndarray,
    evidence_pose_world_camera: np.ndarray,
    *,
    evidence_frame_id: str,
    distance_m: float = 8.0,
    lateral_offset_m: float = 0.75,
    direction_sign: int = 1,
    turn_yaw_deg: float = 20.0,
    minimum_clearance_m: float = 0.30,
    dynamics_limits: TrajectoryDynamicsLimits | Mapping[str, object] | None = None,
    canonical_sampling: bool = True,
) -> Trajectory:
    """Generate a lateral excursion with a SpatialVID-style pose revisit."""

    q0 = np.asarray(q0_pose_world_camera, dtype=np.float64)
    evidence = np.asarray(evidence_pose_world_camera, dtype=np.float64)
    evidence_frame_id = str(evidence_frame_id).strip()
    if not evidence_frame_id:
        raise ValueError("evidence_frame_id must be non-empty")
    if not isinstance(canonical_sampling, (bool, np.bool_)):
        raise TypeError("canonical_sampling must be a boolean")
    if not bool(canonical_sampling):
        raise ValueError(
            "lateral_out_and_back is defined only on the fixed 357/16 canonical clock"
        )
    forward, lateral, evidence_distance = _metric_template_axes(
        q0, evidence, direction_sign=direction_sign
    )
    distance_m = float(distance_m)
    lateral_offset_m = float(lateral_offset_m)
    minimum_clearance_m = float(minimum_clearance_m)
    turn_yaw_deg = float(turn_yaw_deg)
    if not 1.5 <= distance_m <= 12.0:
        raise ValueError("lateral out-and-back distance must remain in [1.5, 12.0] m")
    if distance_m - evidence_distance < 0.75:
        raise TrajectoryFeasibilityError(
            "turnaround must remain at least 0.75m beyond registered qe1"
        )
    if not 0.25 <= lateral_offset_m <= 3.0:
        raise ValueError("lateral offset must remain in [0.25, 3.0] m")
    if not np.isfinite(turn_yaw_deg) or abs(turn_yaw_deg) > 60.0:
        raise ValueError("turn_yaw_deg must be finite and within +/-60 degrees")
    if not np.isfinite(minimum_clearance_m) or minimum_clearance_m <= 0.0:
        raise ValueError("minimum_clearance_m must be finite and positive")

    first_completion_progress = evidence_distance + 0.55 * (
        distance_m - evidence_distance
    )
    revisit_completion_progress = first_completion_progress
    origin = q0[:3, 3]
    first_completion_position = (
        origin
        + first_completion_progress * forward
        + lateral_offset_m * lateral
    )
    revisit_completion_position = first_completion_position.copy()
    positions = np.stack(
        [
            origin,
            origin + 0.5 * evidence_distance * forward,
            evidence[:3, 3],
            first_completion_position,
            origin + distance_m * forward + lateral_offset_m * lateral,
            revisit_completion_position,
            origin,
        ]
    )
    turn_rotation = _turned_world_rotation(
        evidence[:3, :3],
        int(direction_sign) * turn_yaw_deg,
    )
    rotations = np.stack(
        [
            q0[:3, :3],
            q0[:3, :3],
            evidence[:3, :3],
            evidence[:3, :3],
            turn_rotation,
            evidence[:3, :3],
            q0[:3, :3],
        ]
    )
    limits = _coerce_dynamics_limits(dynamics_limits)
    start_holds, end_holds = _moving_completion_segment_holds()
    key_indices = _adaptive_metric_key_indices(
        positions,
        rotations,
        frame_count=CANONICAL_METRIC_TRAJECTORY_FRAME_COUNT,
        fps=CANONICAL_METRIC_TRAJECTORY_FPS,
        limits=limits,
        template_name="lateral_out_and_back",
        segment_start_holds=start_holds,
        segment_end_holds=end_holds,
        segment_minimum_intervals=_natural_completion_segment_minimum_intervals(),
    )
    poses = _interpolate_metric_key_poses(
        CANONICAL_METRIC_TRAJECTORY_FRAME_COUNT,
        key_indices,
        positions,
        rotations,
        segment_start_holds=start_holds,
        segment_end_holds=end_holds,
    )
    poses = _apply_continuous_canonical_parameterization(
        poses,
        key_indices,
        rotations,
        stop_roles=("q0", "w2", "q0_return"),
        orientation_roles=(
            "q0", "qe1", "qc_first", "w2", "qc_revisit", "q0_return"
        ),
    )
    _bind_completion_revisit(poses, key_indices)
    poses[0] = q0
    poses[key_indices["qe1"]] = evidence
    poses[-1] = q0
    trajectory = Trajectory(
        template_name="lateral_out_and_back",
        difficulty="medium",
        duration_s=(
            CANONICAL_METRIC_TRAJECTORY_FRAME_COUNT
            / CANONICAL_METRIC_TRAJECTORY_FPS
        ),
        fps=CANONICAL_METRIC_TRAJECTORY_FPS,
        poses_world_camera=poses,
        key_indices=key_indices,
        parameters={
            "canonical_sampling": True,
            **_natural_completion_parameters(),
            "completion_first_progress_m": first_completion_progress,
            "completion_revisit_progress_m": revisit_completion_progress,
            "direction_sign": int(direction_sign),
            "direction_world": forward.tolist(),
            "distance_m": distance_m,
            "dynamics_limits": asdict(limits),
            "evidence_frame_id": evidence_frame_id,
            "evidence_pose_world_camera": evidence.tolist(),
            "lateral_direction_world": lateral.tolist(),
            "lateral_offset_m": lateral_offset_m,
            "minimum_clearance_m": minimum_clearance_m,
            "time_parameterization": _continuous_time_parameters(
                stop_roles=("q0", "w2", "q0_return"),
                orientation_roles=(
                    "q0", "qe1", "qc_first", "w2", "qc_revisit", "q0_return"
                ),
            ),
            "turn_yaw_deg": turn_yaw_deg,
        },
    )
    _raise_if_generated_dynamics_invalid(trajectory)
    return trajectory


def _integrate_planar_constant_curvature(
    start_xy: np.ndarray,
    start_heading_rad: float,
    curvature: float,
    arc_length: float,
    fractions: np.ndarray,
) -> np.ndarray:
    """Evaluate a planar constant-curvature arc by normalized arc length."""

    values = np.asarray(fractions, dtype=np.float64)
    distance = float(arc_length) * values
    start = np.asarray(start_xy, dtype=np.float64)
    if abs(float(curvature)) < 1e-8:
        direction = np.asarray(
            [math.cos(start_heading_rad), math.sin(start_heading_rad)]
        )
        return start[None] + distance[:, None] * direction[None]
    theta = start_heading_rad + float(curvature) * distance
    x = start[0] + (
        np.sin(theta) - math.sin(start_heading_rad)
    ) / float(curvature)
    y = start[1] + (
        -np.cos(theta) + math.cos(start_heading_rad)
    ) / float(curvature)
    return np.stack([x, y], axis=1)


def _cubic_bezier_xy(
    start: np.ndarray,
    control1: np.ndarray,
    control2: np.ndarray,
    end: np.ndarray,
    fractions: np.ndarray,
) -> np.ndarray:
    values = np.asarray(fractions, dtype=np.float64)[:, None]
    inverse = 1.0 - values
    return (
        inverse**3 * np.asarray(start, dtype=np.float64)[None]
        + 3.0 * inverse**2 * values * np.asarray(control1, dtype=np.float64)[None]
        + 3.0 * inverse * values**2 * np.asarray(control2, dtype=np.float64)[None]
        + values**3 * np.asarray(end, dtype=np.float64)[None]
    )


def _pose_conditioned_piecewise_arc_geometry(
    *,
    origin: np.ndarray,
    forward: np.ndarray,
    lateral: np.ndarray,
    q0_rotation: np.ndarray,
    evidence_rotation: np.ndarray,
    evidence_distance: float,
    requested_extension_m: float,
    return_arc_offset_m: float,
    curvature_change_ratio: float,
    return_direction_sign: int,
) -> dict[str, object]:
    """Build Q0→QE arc1, tangent-continuous QE→W0 arc2, and return arc."""

    relative_yaw = float(
        _world_relative_yaw_deg(
            np.asarray(evidence_rotation, dtype=np.float64)[None],
            np.asarray(q0_rotation, dtype=np.float64),
        )[0]
    )
    bend_sign = 1.0 if relative_yaw > 1e-3 else -1.0 if relative_yaw < -1e-3 else float(return_direction_sign)
    central_angle_deg = bend_sign * float(
        np.clip(abs(relative_yaw), 8.0, 45.0)
    )
    central_angle = math.radians(central_angle_deg)
    chord = float(evidence_distance)
    arc1_length = chord * abs(central_angle) / (
        2.0 * math.sin(abs(central_angle) * 0.5)
    )
    arc1_curvature = central_angle / arc1_length
    arc2_curvature = arc1_curvature * float(curvature_change_ratio)
    arc2_length = float(requested_extension_m)
    start_heading = -0.5 * central_angle
    qe_heading = 0.5 * central_angle
    arc1 = _integrate_planar_constant_curvature(
        np.asarray([0.0, 0.0]),
        start_heading,
        arc1_curvature,
        arc1_length,
        np.asarray([0.0, 0.5, 1.0]),
    )
    # Remove only floating-point endpoint residue; QE remains exactly bound to
    # the registered pose while its inbound/outbound tangent stays continuous.
    arc1[-1] = [chord, 0.0]
    arc2 = _integrate_planar_constant_curvature(
        np.asarray([chord, 0.0]),
        qe_heading,
        arc2_curvature,
        arc2_length,
        np.asarray([0.0, 0.88, 1.0]),
    )
    w0 = arc2[-1]
    outbound_mid_sign = float(np.sign(arc1[1, 1]))
    if outbound_mid_sign == 0.0:
        outbound_mid_sign = -bend_sign
    # The return route must occupy the other side of the outbound arcs.  The
    # direction candidate already determines the fallback bend direction when
    # Q0/QE have negligible relative yaw; multiplying it again here made one
    # branch accidentally return on the same side.
    return_side = -outbound_mid_sign
    return_bow = max(float(return_arc_offset_m), 0.18 * max(chord, w0[0]))
    control1 = np.asarray([0.70 * w0[0], return_side * return_bow])
    control2 = np.asarray([0.32 * w0[0], return_side * return_bow])
    return_controls = _cubic_bezier_xy(
        w0,
        control1,
        control2,
        np.asarray([0.0, 0.0]),
        np.asarray([0.0, 0.12, 1.0]),
    )
    controls_xy = np.stack(
        [arc1[0], arc1[1], arc1[2], arc2[1], arc2[2], return_controls[1], return_controls[2]]
    )
    controls_world = (
        np.asarray(origin, dtype=np.float64)[None]
        + controls_xy[:, :1] * np.asarray(forward, dtype=np.float64)[None]
        + controls_xy[:, 1:] * np.asarray(lateral, dtype=np.float64)[None]
    )
    return {
        "origin": np.asarray(origin, dtype=np.float64),
        "forward": np.asarray(forward, dtype=np.float64),
        "lateral": np.asarray(lateral, dtype=np.float64),
        "arc1_curvature": arc1_curvature,
        "arc2_curvature": arc2_curvature,
        "arc1_central_angle_deg": central_angle_deg,
        "arc1_length": arc1_length,
        "arc2_length": arc2_length,
        "arc1_start_heading": start_heading,
        "qe_heading": qe_heading,
        "evidence_distance": chord,
        "w0_xy": w0,
        "completion_xy": arc2[1],
        "return_control1_xy": control1,
        "return_control2_xy": control2,
        "control_positions_world": controls_world,
    }


def _pose_conditioned_piecewise_arc_translations(
    key_indices: Mapping[str, int],
    *,
    geometry: Mapping[str, object],
    natural_revisit: bool = False,
) -> tuple[np.ndarray, dict[str, int | float]]:
    key = {name: int(value) for name, value in key_indices.items()}
    samples = np.arange(CANONICAL_METRIC_TRAJECTORY_FRAME_COUNT, dtype=np.float64)
    xy = np.zeros((len(samples), 2), dtype=np.float64)

    qe_index = key["qe1"]
    w2_index = key["w2"]
    outbound_fraction = np.interp(
        samples[: qe_index + 1],
        [key["q0"], key["w1"], qe_index],
        [0.0, 0.5, 1.0],
    )
    xy[: qe_index + 1] = _integrate_planar_constant_curvature(
        np.asarray([0.0, 0.0]),
        float(geometry["arc1_start_heading"]),
        float(geometry["arc1_curvature"]),
        float(geometry["arc1_length"]),
        outbound_fraction,
    )
    xy[qe_index] = [float(geometry["evidence_distance"]), 0.0]

    extension_fraction = np.interp(
        samples[qe_index : w2_index + 1],
        [qe_index, key["qc_first"], w2_index],
        [0.0, 0.88, 1.0],
    )
    xy[qe_index : w2_index + 1] = _integrate_planar_constant_curvature(
        np.asarray([float(geometry["evidence_distance"]), 0.0]),
        float(geometry["qe_heading"]),
        float(geometry["arc2_curvature"]),
        float(geometry["arc2_length"]),
        extension_fraction,
    )
    w0 = np.asarray(geometry["w0_xy"], dtype=np.float64)
    xy[w2_index] = w0

    # The natural-revisit profile keeps the return-side control and lets
    # window matching find the nearest revisit; exact-binding profiles pass C2
    # through the outbound C1 control.
    return_times = np.asarray(
        [w2_index, key["qc_revisit"], key["q0_return"]], dtype=np.float64
    )
    return_controls = np.stack(
        [
            w0,
            np.asarray(
                geometry[
                    "return_control1_xy" if natural_revisit else "completion_xy"
                ],
                dtype=np.float64,
            ),
            np.asarray([0.0, 0.0]),
        ]
    )
    for axis in range(2):
        xy[w2_index:, axis] = PchipInterpolator(
            return_times, return_controls[:, axis]
        )(samples[w2_index:])
    translations = (
        np.asarray(geometry["origin"], dtype=np.float64)[None]
        + xy[:, :1] * np.asarray(geometry["forward"], dtype=np.float64)[None]
        + xy[:, 1:] * np.asarray(geometry["lateral"], dtype=np.float64)[None]
    )
    return translations, {
        "outbound_arc1_mid_frame_index": key["w1"],
        "outbound_arc2_near_w0_frame_index": key["qc_first"],
        "turnaround_frame_index": key["w2"],
        "return_near_w0_frame_index": key["qc_revisit"],
    }


def _global_arc_translations(
    key_indices: Mapping[str, int],
    *,
    origin: np.ndarray,
    forward: np.ndarray,
    lateral: np.ndarray,
    evidence_distance: float,
    completion_progress: float,
    distance_m: float,
    arc_offset_m: float,
) -> tuple[np.ndarray, dict[str, int | float]]:
    """Author a full-route, two-branch natural-C2 lens."""

    key = {name: int(value) for name, value in key_indices.items()}
    if list(key) != list(METRIC_TRAJECTORY_EVENT_ROLES):
        raise ValueError("arc controls must use the canonical seven event roles")
    first_progress = float(completion_progress)
    revisit_progress = first_progress
    if not evidence_distance < first_progress < distance_m:
        raise TrajectoryFeasibilityError(
            "arc completion event must lie between qe1 and w2"
        )
    # Bend before qe while preserving its registered pose at zero lateral offset.
    control_progress = np.asarray(
        [
            0.0,
            0.5 * evidence_distance,
            evidence_distance,
            first_progress,
            distance_m,
            revisit_progress,
            0.0,
        ],
        dtype=np.float64,
    )
    control_lateral = np.asarray(
        [
            0.0,
            0.32 * arc_offset_m,
            0.0,
            arc_offset_m,
            0.0,
            arc_offset_m,
            0.0,
        ],
        dtype=np.float64,
    )
    times = np.asarray(
        [key[name] for name in METRIC_TRAJECTORY_EVENT_ROLES],
        dtype=np.float64,
    )
    samples = np.arange(CANONICAL_METRIC_TRAJECTORY_FRAME_COUNT, dtype=np.float64)
    # PCHIP protects branch progress monotonicity.  The lateral coordinate is
    # one natural C2 spline over the whole route, rather than one sine bow per
    # event segment.
    progress_samples = PchipInterpolator(times, control_progress)(samples)
    lateral_samples = CubicSpline(
        times,
        control_lateral,
        bc_type="natural",
    )(samples)
    translations = (
        np.asarray(origin, dtype=np.float64)[None]
        + progress_samples[:, None] * np.asarray(forward, dtype=np.float64)[None]
        + lateral_samples[:, None] * np.asarray(lateral, dtype=np.float64)[None]
    )
    # Write event controls bit-exactly: qe binding and q0 closure are
    # construction contracts, not approximate spline properties.
    for frame_index, forward_progress, side in zip(
        times.astype(int),
        control_progress,
        control_lateral,
    ):
        translations[int(frame_index)] = (
            np.asarray(origin, dtype=np.float64)
            + float(forward_progress) * np.asarray(forward, dtype=np.float64)
            + float(side) * np.asarray(lateral, dtype=np.float64)
        )
    return translations, {
        "outbound_lens_entry_frame_index": int(key["w1"]),
        "outbound_lens_peak_frame_index": int(key["qc_first"]),
        "return_lens_peak_frame_index": int(key["qc_revisit"]),
        "first_completion_progress_m": first_progress,
        "revisit_completion_progress_m": revisit_progress,
    }


def _regular_planar_dual_arc_translations(
    key_indices: Mapping[str, int],
    *,
    origin: np.ndarray,
    forward: np.ndarray,
    lateral: np.ndarray,
    evidence_distance: float,
    completion_progress: float,
    distance_m: float,
    arc_offset_m: float,
) -> tuple[np.ndarray, dict[str, int | float | list[float]]]:
    """Author a predictable planar outbound/return pair for paired-tar arc v3."""

    key = {name: int(value) for name, value in key_indices.items()}
    samples = np.arange(CANONICAL_METRIC_TRAJECTORY_FRAME_COUNT, dtype=np.float64)
    xy = np.zeros((len(samples), 2), dtype=np.float64)
    qe_index = key["qe1"]
    completion_index = key["qc_first"]
    turnaround_index = key["w2"]
    revisit_index = key["qc_revisit"]
    return_index = key["q0_return"]

    # Q0 -> QE: a single cubic bow with zero endpoint lateral displacement.
    outbound_fraction = np.interp(
        samples[: qe_index + 1],
        [key["q0"], key["w1"], qe_index],
        [0.0, 0.5, 1.0],
    )
    xy[: qe_index + 1] = _cubic_bezier_xy(
        np.asarray([0.0, 0.0]),
        np.asarray([0.30 * evidence_distance, 1.34 * arc_offset_m]),
        np.asarray([0.70 * evidence_distance, 1.34 * arc_offset_m]),
        np.asarray([evidence_distance, 0.0]),
        outbound_fraction,
    )
    xy[qe_index] = [evidence_distance, 0.0]

    # QE -> W2: a small shared tip.  The completion pose lies on the return
    # side and is revisited exactly; it is 20% of the main bow so that the
    # turnaround does not form a loop.
    completion_xy = np.asarray(
        [completion_progress, -0.20 * arc_offset_m], dtype=np.float64
    )
    outward_times = np.asarray(
        [qe_index, completion_index, turnaround_index], dtype=np.float64
    )
    outward_controls = np.stack(
        [
            np.asarray([evidence_distance, 0.0]),
            completion_xy,
            np.asarray([distance_m, 0.0]),
        ]
    )
    for axis in range(2):
        xy[qe_index : turnaround_index + 1, axis] = PchipInterpolator(
            outward_times, outward_controls[:, axis]
        )(samples[qe_index : turnaround_index + 1])

    # W2 -> revisit follows the short shared tip in reverse.
    shared_fraction = np.linspace(
        0.0, 1.0, revisit_index - turnaround_index + 1, dtype=np.float64
    )[:, None]
    w2_xy = np.asarray([distance_m, 0.0], dtype=np.float64)
    xy[turnaround_index : revisit_index + 1] = (
        (1.0 - shared_fraction) * w2_xy[None]
        + shared_fraction * completion_xy[None]
    )

    # Revisit -> Q0: one regular cubic arc.  C1 continues away from W2 and C2
    # supplies the broad return bow; both forward controls decrease, so the
    # return never doubles back in BEV.
    return_control1 = completion_xy + 0.35 * (completion_xy - w2_xy)
    return_control2 = np.asarray(
        [0.30 * completion_progress, -2.25 * arc_offset_m], dtype=np.float64
    )
    return_fraction = np.linspace(
        0.0, 1.0, return_index - revisit_index + 1, dtype=np.float64
    )
    xy[revisit_index : return_index + 1] = _cubic_bezier_xy(
        completion_xy,
        return_control1,
        return_control2,
        np.asarray([0.0, 0.0]),
        return_fraction,
    )

    translations = (
        np.asarray(origin, dtype=np.float64)[None]
        + xy[:, :1] * np.asarray(forward, dtype=np.float64)[None]
        + xy[:, 1:] * np.asarray(lateral, dtype=np.float64)[None]
    )
    return translations, {
        "outbound_arc_peak_frame_index": key["w1"],
        "shared_tip_first_frame_index": completion_index,
        "turnaround_frame_index": turnaround_index,
        "shared_tip_revisit_frame_index": revisit_index,
        "completion_lateral_m": float(completion_xy[1]),
        "return_control1_xy": return_control1.tolist(),
        "return_control2_xy": return_control2.tolist(),
    }


def generate_arc_return(
    q0_pose_world_camera: np.ndarray,
    evidence_pose_world_camera: np.ndarray,
    *,
    evidence_frame_id: str,
    distance_m: float = 8.0,
    arc_offset_m: float = 0.35,
    direction_sign: int = 1,
    turn_yaw_deg: float = 0.0,
    curve_profile: str = "full_route_natural_c2_lens",
    lateral_axis_policy: str = "gravity_aligned_global_z_cross_forward",
    curvature_change_ratio: float = 1.0,
    bound_global_z_to_endpoint_range: bool = False,
    minimum_clearance_m: float = 0.30,
    dynamics_limits: TrajectoryDynamicsLimits | Mapping[str, object] | None = None,
    canonical_sampling: bool = True,
) -> Trajectory:
    """Generate a full-route arc with a SpatialVID-style pose revisit."""

    q0 = np.asarray(q0_pose_world_camera, dtype=np.float64)
    evidence = np.asarray(evidence_pose_world_camera, dtype=np.float64)
    evidence_frame_id = str(evidence_frame_id).strip()
    if not evidence_frame_id:
        raise ValueError("evidence_frame_id must be non-empty")
    if not isinstance(canonical_sampling, (bool, np.bool_)):
        raise TypeError("canonical_sampling must be a boolean")
    if not isinstance(bound_global_z_to_endpoint_range, (bool, np.bool_)):
        raise TypeError("bound_global_z_to_endpoint_range must be a boolean")
    if not bool(canonical_sampling):
        raise ValueError(
            "arc_return is defined only on the fixed 357/16 canonical clock"
        )
    forward, lateral, evidence_distance = _metric_template_axes(
        q0, evidence, direction_sign=direction_sign
    )
    distance_m = float(distance_m)
    arc_offset_m = float(arc_offset_m)
    minimum_clearance_m = float(minimum_clearance_m)
    turn_yaw_deg = float(turn_yaw_deg)
    curve_profile = str(curve_profile).strip()
    lateral_axis_policy = str(lateral_axis_policy).strip()
    curvature_change_ratio = float(curvature_change_ratio)
    if not 1.5 <= distance_m <= 12.0:
        raise ValueError("arc-return distance must remain in [1.5, 12.0] m")
    if distance_m - evidence_distance < 0.75:
        raise TrajectoryFeasibilityError(
            "turnaround must remain at least 0.75m beyond registered qe1"
        )
    if not 0.20 <= arc_offset_m <= 2.0:
        raise ValueError("arc offset must remain in [0.20, 2.0] m")
    if not np.isfinite(turn_yaw_deg) or abs(turn_yaw_deg) > 60.0:
        raise ValueError("turn_yaw_deg must be finite and within +/-60 degrees")
    if curve_profile not in {
        "full_route_natural_c2_lens",
        "spatialvid_pose_conditioned_piecewise_arcs",
        "spatialvid_pose_conditioned_piecewise_arcs_natural_revisit",
        "planar_regular_dual_arc_v3",
    }:
        raise ValueError("unsupported arc-return curve_profile")
    if lateral_axis_policy not in {
        "gravity_aligned_global_z_cross_forward",
        "q0_camera_right_exact",
    }:
        raise ValueError("unsupported arc-return lateral_axis_policy")
    if not 0.70 <= curvature_change_ratio <= 1.30:
        raise ValueError("curvature_change_ratio must be in [0.70, 1.30]")
    if not np.isfinite(minimum_clearance_m) or minimum_clearance_m <= 0.0:
        raise ValueError("minimum_clearance_m must be finite and positive")

    if curve_profile == "planar_regular_dual_arc_v3" and (
        lateral_axis_policy != "q0_camera_right_exact"
    ):
        raise ValueError(
            "planar_regular_dual_arc_v3 requires q0_camera_right_exact"
        )
    if lateral_axis_policy == "q0_camera_right_exact":
        lateral = np.asarray(q0[:3, 0], dtype=np.float64)
        lateral /= np.linalg.norm(lateral)
        lateral *= int(direction_sign)
    elif curve_profile in {
        "spatialvid_pose_conditioned_piecewise_arcs",
        "spatialvid_pose_conditioned_piecewise_arcs_natural_revisit",
    }:
        lateral = _camera_right_lateral_axis(
            q0,
            forward,
            direction_sign=direction_sign,
        )
        lateral_axis_policy = "q0_camera_right_orthogonalized"

    first_completion_progress = evidence_distance + 0.55 * (
        distance_m - evidence_distance
    )
    revisit_completion_progress = first_completion_progress
    origin = q0[:3, 3]
    piecewise_geometry = None
    if curve_profile in {
        "spatialvid_pose_conditioned_piecewise_arcs",
        "spatialvid_pose_conditioned_piecewise_arcs_natural_revisit",
    }:
        piecewise_geometry = _pose_conditioned_piecewise_arc_geometry(
            origin=origin,
            forward=forward,
            lateral=lateral,
            q0_rotation=q0[:3, :3],
            evidence_rotation=evidence[:3, :3],
            evidence_distance=evidence_distance,
            requested_extension_m=distance_m - evidence_distance,
            return_arc_offset_m=arc_offset_m,
            curvature_change_ratio=curvature_change_ratio,
            return_direction_sign=direction_sign,
        )
        positions = piecewise_geometry["control_positions_world"].copy()
        positions[2] = evidence[:3, 3]
        if curve_profile == "spatialvid_pose_conditioned_piecewise_arcs":
            positions[5] = positions[3]
    elif curve_profile == "planar_regular_dual_arc_v3":
        completion_lateral = -0.20 * arc_offset_m
        progress = np.asarray(
            [
                0.0,
                0.5 * evidence_distance,
                evidence_distance,
                first_completion_progress,
                distance_m,
                first_completion_progress,
                0.0,
            ],
            dtype=np.float64,
        )
        sides = np.asarray(
            [
                0.0,
                arc_offset_m,
                0.0,
                completion_lateral,
                0.0,
                completion_lateral,
                0.0,
            ],
            dtype=np.float64,
        )
        positions = (
            origin
            + progress[:, None] * forward[None]
            + sides[:, None] * lateral[None]
        )
        positions[2] = evidence[:3, 3]
    else:
        progress = np.asarray(
            [
                0.0,
                0.5 * evidence_distance,
                evidence_distance,
                first_completion_progress,
                distance_m,
                revisit_completion_progress,
                0.0,
            ],
            dtype=np.float64,
        )
        positions = origin + progress[:, None] * forward[None]
    turn_rotation = _turned_world_rotation(
        evidence[:3, :3],
        int(direction_sign) * turn_yaw_deg,
    )
    rotations = np.stack(
        [
            q0[:3, :3],
            q0[:3, :3],
            evidence[:3, :3],
            evidence[:3, :3],
            turn_rotation,
            evidence[:3, :3],
            q0[:3, :3],
        ]
    )
    limits = _coerce_dynamics_limits(dynamics_limits)
    start_holds, end_holds = _moving_completion_segment_holds()
    key_indices = _adaptive_metric_key_indices(
        positions,
        rotations,
        frame_count=CANONICAL_METRIC_TRAJECTORY_FRAME_COUNT,
        fps=CANONICAL_METRIC_TRAJECTORY_FPS,
        limits=limits,
        template_name="arc_return",
        segment_start_holds=start_holds,
        segment_end_holds=end_holds,
        segment_minimum_intervals=_natural_completion_segment_minimum_intervals(),
    )
    poses = _interpolate_metric_key_poses(
        CANONICAL_METRIC_TRAJECTORY_FRAME_COUNT,
        key_indices,
        positions,
        rotations,
        segment_start_holds=start_holds,
        segment_end_holds=end_holds,
    )
    if piecewise_geometry is not None:
        translations, arc_knot_indices = (
            _pose_conditioned_piecewise_arc_translations(
                key_indices,
                geometry=piecewise_geometry,
                natural_revisit=(
                    curve_profile
                    == "spatialvid_pose_conditioned_piecewise_arcs_natural_revisit"
                ),
            )
        )
    elif curve_profile == "planar_regular_dual_arc_v3":
        translations, arc_knot_indices = _regular_planar_dual_arc_translations(
            key_indices,
            origin=origin,
            forward=forward,
            lateral=lateral,
            evidence_distance=evidence_distance,
            completion_progress=first_completion_progress,
            distance_m=distance_m,
            arc_offset_m=arc_offset_m,
        )
    else:
        translations, arc_knot_indices = _global_arc_translations(
            key_indices,
            origin=origin,
            forward=forward,
            lateral=lateral,
            evidence_distance=evidence_distance,
            completion_progress=first_completion_progress,
            distance_m=distance_m,
            arc_offset_m=arc_offset_m,
        )
    poses[:, :3, 3] = translations
    poses = _apply_continuous_canonical_parameterization(
        poses,
        key_indices,
        rotations,
        stop_roles=("q0", "w2", "q0_return"),
        orientation_roles=(
            "q0", "qe1", "qc_first", "w2", "qc_revisit", "q0_return"
        ),
    )
    if bool(bound_global_z_to_endpoint_range):
        # ScanNet++ capture poses may contain a small Q0->QE height change.
        # Extending the full 3-D displacement direction to W2 amplifies that
        # change. Author a bounded, smooth height clock instead: reach QE's
        # registered height exactly at QE, hold it over the remote arc, then
        # return smoothly to Q0 after the authored revisit control.
        qe_index = int(key_indices["qe1"])
        revisit_index = int(key_indices["qc_revisit"])
        outbound_tau = np.linspace(0.0, 1.0, qe_index + 1, dtype=np.float64)
        return_tau = np.linspace(
            0.0,
            1.0,
            len(poses) - revisit_index,
            dtype=np.float64,
        )
        poses[: qe_index + 1, 2, 3] = (
            origin[2]
            + (evidence[2, 3] - origin[2])
            * _quintic_smoothstep(outbound_tau)
        )
        poses[qe_index : revisit_index + 1, 2, 3] = evidence[2, 3]
        poses[revisit_index:, 2, 3] = (
            evidence[2, 3]
            + (origin[2] - evidence[2, 3])
            * _quintic_smoothstep(return_tau)
        )
    if curve_profile != "spatialvid_pose_conditioned_piecewise_arcs_natural_revisit":
        _bind_completion_revisit(poses, key_indices)
    poses[0] = q0
    poses[key_indices["qe1"]] = evidence
    poses[-1] = q0
    trajectory = Trajectory(
        template_name="arc_return",
        difficulty="hard",
        duration_s=(
            CANONICAL_METRIC_TRAJECTORY_FRAME_COUNT
            / CANONICAL_METRIC_TRAJECTORY_FPS
        ),
        fps=CANONICAL_METRIC_TRAJECTORY_FPS,
        poses_world_camera=poses,
        key_indices=key_indices,
        parameters={
            "arc_offset_m": arc_offset_m,
            "bound_global_z_to_endpoint_range": bool(
                bound_global_z_to_endpoint_range
            ),
            "canonical_sampling": True,
            **_natural_completion_parameters(),
            "completion_first_progress_m": first_completion_progress,
            "completion_revisit_progress_m": revisit_completion_progress,
            "direction_sign": int(direction_sign),
            "direction_world": forward.tolist(),
            "distance_m": distance_m,
            "dynamics_limits": asdict(limits),
            "evidence_frame_id": evidence_frame_id,
            "evidence_pose_world_camera": evidence.tolist(),
            "lateral_direction_world": lateral.tolist(),
            "lateral_axis_policy": lateral_axis_policy,
            "minimum_clearance_m": minimum_clearance_m,
            "global_arc_curve": curve_profile,
            "completion_revisit_binding": (
                "natural_window_matched"
                if curve_profile
                == "spatialvid_pose_conditioned_piecewise_arcs_natural_revisit"
                else "exact_key_pose"
            ),
            "global_arc_knot_indices": arc_knot_indices,
            "curvature_change_ratio": curvature_change_ratio,
            **(
                {
                    "arc1_curvature_per_m": float(piecewise_geometry["arc1_curvature"]),
                    "arc2_curvature_per_m": float(piecewise_geometry["arc2_curvature"]),
                    "arc1_central_angle_deg": float(piecewise_geometry["arc1_central_angle_deg"]),
                    "arc1_length_m": float(piecewise_geometry["arc1_length"]),
                    "arc2_length_m": float(piecewise_geometry["arc2_length"]),
                    "qe_tangent_discontinuity_deg": 0.0,
                    "w0_forward_progress_m": float(piecewise_geometry["w0_xy"][0]),
                }
                if piecewise_geometry is not None
                else {}
            ),
            "time_parameterization": _continuous_time_parameters(
                stop_roles=("q0", "w2", "q0_return"),
                orientation_roles=(
                    "q0", "qe1", "qc_first", "w2", "qc_revisit", "q0_return"
                ),
            ),
            "turn_yaw_deg": turn_yaw_deg,
        },
    )
    _raise_if_generated_dynamics_invalid(trajectory)
    return trajectory


def generate_forward_backward(
    q0_pose_world_camera: np.ndarray,
    evidence_pose_world_camera: np.ndarray,
    *,
    evidence_frame_id: str,
    distance_m: float = 2.0,
    minimum_clearance_m: float = 0.20,
    dynamics_limits: TrajectoryDynamicsLimits | Mapping[str, object] | None = None,
    canonical_sampling: bool = False,
) -> Trajectory:
    """Generate an out-and-back path whose ``qe1`` is a registered frame."""

    q0 = np.asarray(q0_pose_world_camera, dtype=np.float64)
    evidence = np.asarray(evidence_pose_world_camera, dtype=np.float64)
    evidence_frame_id = str(evidence_frame_id).strip()
    if not evidence_frame_id:
        raise ValueError("evidence_frame_id must be non-empty")
    resolved_dynamics_limits = _coerce_dynamics_limits(dynamics_limits)
    offset = evidence[:3, 3] - q0[:3, 3]
    evidence_distance = float(np.linalg.norm(offset))
    if evidence_distance <= 0.05:
        raise TrajectoryFeasibilityError(
            "registered evidence translation must exceed 0.05m"
        )
    direction = offset / evidence_distance
    distance_m = float(distance_m)
    if not 1.5 <= distance_m <= 12.0:
        raise ValueError("forward-backward distance must remain in [1.5, 12.0] m")
    if distance_m <= evidence_distance + 0.05:
        raise TrajectoryFeasibilityError(
            "turnaround must be at least 0.05m beyond registered qe1"
        )
    first_completion_progress = evidence_distance + 0.55 * (
        distance_m - evidence_distance
    )
    revisit_completion_progress = first_completion_progress
    progress = np.asarray(
        [
            0.0,
            0.5 * evidence_distance,
            evidence_distance,
            first_completion_progress,
            distance_m,
            revisit_completion_progress,
            0.0,
        ]
    )
    positions = q0[:3, 3] + progress[:, None] * direction
    positions[2] = evidence[:3, 3]
    rotations = np.stack(
        [
            q0[:3, :3],
            q0[:3, :3],
            evidence[:3, :3],
            evidence[:3, :3],
            evidence[:3, :3],
            evidence[:3, :3],
            q0[:3, :3],
        ]
    )
    if canonical_sampling:
        frame_count = CANONICAL_FORWARD_BACKWARD_FRAME_COUNT
        fps = CANONICAL_FORWARD_BACKWARD_FPS
        start_holds, end_holds = _moving_completion_segment_holds()
        key_indices = _adaptive_metric_key_indices(
            positions,
            rotations,
            frame_count=frame_count,
            fps=fps,
            limits=resolved_dynamics_limits,
            template_name="forward_backward",
            segment_start_holds=start_holds,
            segment_end_holds=end_holds,
            segment_minimum_intervals=_natural_completion_segment_minimum_intervals(),
        )
        poses = _interpolate_metric_key_poses(
            frame_count,
            key_indices,
            positions,
            rotations,
            segment_start_holds=start_holds,
            segment_end_holds=end_holds,
        )
        poses = _apply_continuous_canonical_parameterization(
            poses,
            key_indices,
            rotations,
            stop_roles=("q0", "w2", "q0_return"),
            orientation_roles=(
                "q0", "qe1", "qc_first", "w2", "qc_revisit", "q0_return"
            ),
        )
    else:
        frame_count = 216
        fps = 12
        key_indices = dict(FORWARD_BACKWARD_KEY_INDICES)
        poses = _interpolate_key_poses(
            frame_count, list(key_indices.values()), positions, rotations
        )
    _bind_completion_revisit(poses, key_indices)
    poses[0] = q0
    poses[key_indices["qe1"]] = evidence
    poses[-1] = q0
    trajectory = Trajectory(
        template_name="forward_backward",
        difficulty="easy",
        duration_s=frame_count / fps,
        fps=fps,
        poses_world_camera=poses,
        key_indices=key_indices,
        parameters={
            "canonical_sampling": bool(canonical_sampling),
            **(
                {
                    **_natural_completion_parameters(),
                    "completion_first_progress_m": first_completion_progress,
                    "completion_revisit_progress_m": revisit_completion_progress,
                    "time_parameterization": _continuous_time_parameters(
                        stop_roles=("q0", "w2", "q0_return"),
                        orientation_roles=(
                            "q0", "qe1", "qc_first", "w2", "qc_revisit", "q0_return"
                        ),
                    ),
                }
                if canonical_sampling
                else {}
            ),
            "distance_m": distance_m,
            "direction_world": direction.tolist(),
            "dynamics_limits": asdict(resolved_dynamics_limits),
            "evidence_frame_id": evidence_frame_id,
            "evidence_pose_world_camera": evidence.tolist(),
            "minimum_clearance_m": float(minimum_clearance_m),
        },
    )
    if canonical_sampling:
        _raise_if_generated_dynamics_invalid(trajectory)
    return trajectory


def _world_relative_yaw_deg(
    rotations_world_camera: np.ndarray,
    reference_rotation_world_camera: np.ndarray,
) -> np.ndarray:
    """Return unwrapped world-Z yaw relative to a reference camera rotation."""

    rotations = np.asarray(rotations_world_camera, dtype=np.float64)
    reference = np.asarray(reference_rotation_world_camera, dtype=np.float64)
    relative = rotations @ reference.T
    yaw = np.arctan2(relative[:, 1, 0], relative[:, 0, 0])
    return np.rad2deg(np.unwrap(yaw))


def generate_yaw_return(
    q0_pose_world_camera: np.ndarray,
    evidence_pose_world_camera: np.ndarray,
    *,
    evidence_frame_id: str,
    yaw_excursion_deg: float = 110.0,
    yaw_axis_policy: str = "gravity_aligned_world_z",
    direction_sign: int | None = None,
    maximum_center_translation_m: float = 0.15,
    minimum_clearance_m: float = 0.20,
    dynamics_limits: TrajectoryDynamicsLimits | Mapping[str, object] | None = None,
    canonical_sampling: bool = True,
) -> Trajectory:
    """Generate a near-centre yaw sweep and return through real ``qe1``."""

    q0 = np.asarray(q0_pose_world_camera, dtype=np.float64)
    evidence = np.asarray(evidence_pose_world_camera, dtype=np.float64)
    evidence_frame_id = str(evidence_frame_id).strip()
    if not evidence_frame_id:
        raise ValueError("evidence_frame_id must be non-empty")
    if not isinstance(canonical_sampling, (bool, np.bool_)) or not canonical_sampling:
        raise ValueError(
            "yaw_return is defined only on the fixed 357/16 canonical clock"
        )
    yaw_excursion_deg = float(yaw_excursion_deg)
    maximum_center_translation_m = float(maximum_center_translation_m)
    minimum_clearance_m = float(minimum_clearance_m)
    yaw_axis_policy = str(yaw_axis_policy).strip()
    if yaw_axis_policy not in {"gravity_aligned_world_z", "q0_camera_y"}:
        raise ValueError(
            "yaw_axis_policy must be gravity_aligned_world_z or q0_camera_y"
        )
    if not 90.0 <= yaw_excursion_deg <= 180.0:
        raise ValueError("yaw excursion must be in [90, 180] degrees")
    if not np.isfinite(maximum_center_translation_m) or not (
        0.0 < maximum_center_translation_m <= 0.30
    ):
        raise ValueError(
            "maximum_center_translation_m must be finite and in (0, 0.30]"
        )
    if not np.isfinite(minimum_clearance_m) or minimum_clearance_m <= 0.0:
        raise ValueError("minimum_clearance_m must be finite and positive")

    evidence_translation = float(
        np.linalg.norm(evidence[:3, 3] - q0[:3, 3])
    )
    if evidence_translation > maximum_center_translation_m + 1e-9:
        raise TrajectoryFeasibilityError(
            "registered yaw evidence centre exceeds q0-near translation limit"
        )
    yaw_measure = (
        _camera_relative_yaw_deg
        if yaw_axis_policy == "q0_camera_y"
        else _world_relative_yaw_deg
    )
    evidence_yaw = float(
        yaw_measure(
            evidence[None, :3, :3],
            q0[:3, :3],
        )[0]
    )
    if abs(evidence_yaw) < 1e-3:
        raise TrajectoryFeasibilityError(
            "registered yaw evidence must have a meaningful orientation change"
        )
    inferred_sign = 1 if evidence_yaw > 0.0 else -1
    if direction_sign is None:
        resolved_sign = inferred_sign
    else:
        resolved_sign = int(direction_sign)
        if resolved_sign not in {-1, 1}:
            raise ValueError("direction_sign must be -1, +1, or None")
        if resolved_sign != inferred_sign:
            raise TrajectoryFeasibilityError(
                "registered qe yaw direction conflicts with requested yaw sweep"
            )
    if abs(evidence_yaw) >= yaw_excursion_deg - 1e-6:
        raise TrajectoryFeasibilityError(
            "registered qe yaw must occur before the yaw turnaround"
        )

    evidence_fraction = abs(evidence_yaw) / yaw_excursion_deg
    q0_position = q0[:3, 3]
    evidence_position = evidence[:3, 3]
    centre_delta = evidence_position - q0_position
    positions = np.stack(
        [
            q0_position,
            q0_position + 0.50 * centre_delta,
            evidence_position,
            q0_position + 0.80 * centre_delta,
            q0_position + 0.30 * centre_delta,
            q0_position + 0.55 * centre_delta,
            q0_position,
        ]
    )
    positions[5] = positions[3]
    yaw_controls = np.asarray(
        [
            0.0,
            0.50 * evidence_yaw,
            evidence_yaw,
            resolved_sign
            * (abs(evidence_yaw) + 0.58 * (yaw_excursion_deg - abs(evidence_yaw))),
            resolved_sign * yaw_excursion_deg,
            resolved_sign
            * (abs(evidence_yaw) + 0.30 * (yaw_excursion_deg - abs(evidence_yaw))),
            0.0,
        ],
        dtype=np.float64,
    )
    yaw_controls[5] = yaw_controls[3]
    turn_rotation = (
        _turned_camera_y_rotation
        if yaw_axis_policy == "q0_camera_y"
        else _turned_world_rotation
    )
    rotations = np.stack([turn_rotation(q0[:3, :3], yaw) for yaw in yaw_controls])
    rotations[2] = evidence[:3, :3]
    limits = _coerce_dynamics_limits(dynamics_limits)
    start_holds, end_holds = _moving_completion_segment_holds()
    key_indices = _adaptive_metric_key_indices(
        positions,
        rotations,
        frame_count=CANONICAL_METRIC_TRAJECTORY_FRAME_COUNT,
        fps=CANONICAL_METRIC_TRAJECTORY_FPS,
        limits=limits,
        template_name="yaw_return",
        segment_start_holds=start_holds,
        segment_end_holds=end_holds,
        segment_minimum_intervals=_natural_completion_segment_minimum_intervals(),
    )
    poses = _interpolate_metric_key_poses(
        CANONICAL_METRIC_TRAJECTORY_FRAME_COUNT,
        key_indices,
        positions,
        rotations,
        segment_start_holds=start_holds,
        segment_end_holds=end_holds,
    )
    yaw_orientation_roles = tuple(METRIC_TRAJECTORY_EVENT_ROLES)
    poses = _apply_continuous_canonical_parameterization(
        poses,
        key_indices,
        rotations,
        stop_roles=("q0", "w2", "q0_return"),
        orientation_roles=yaw_orientation_roles,
    )
    if yaw_axis_policy == "q0_camera_y":
        poses[:, :3, :3] = _enforce_monotone_camera_yaw(
            poses[:, :3, :3], key_indices, rotations
        )
    else:
        poses[:, :3, :3] = _enforce_monotone_world_yaw(
            poses[:, :3, :3], key_indices, rotations
        )
    _bind_completion_revisit(poses, key_indices)
    poses[0] = q0
    poses[key_indices["qe1"]] = evidence
    poses[-1] = q0
    trajectory = Trajectory(
        template_name="yaw_return",
        difficulty="medium",
        duration_s=(
            CANONICAL_METRIC_TRAJECTORY_FRAME_COUNT
            / CANONICAL_METRIC_TRAJECTORY_FPS
        ),
        fps=CANONICAL_METRIC_TRAJECTORY_FPS,
        poses_world_camera=poses,
        key_indices=key_indices,
        parameters={
            "canonical_sampling": True,
            **_natural_completion_parameters(),
            "direction_sign": resolved_sign,
            "dynamics_limits": asdict(limits),
            "evidence_frame_id": evidence_frame_id,
            "evidence_pose_world_camera": evidence.tolist(),
            "evidence_yaw_deg": evidence_yaw,
            "maximum_center_translation_m": maximum_center_translation_m,
            "minimum_clearance_m": minimum_clearance_m,
            "time_parameterization": _continuous_time_parameters(
                stop_roles=("q0", "w2", "q0_return"),
                orientation_roles=yaw_orientation_roles,
                rotation_clock=(
                    "global_so3_rotation_spline_with_monotone_world_yaw"
                ),
            ),
            "yaw_excursion_deg": yaw_excursion_deg,
            "yaw_axis_policy": yaw_axis_policy,
            "qe_outbound_yaw_fraction": evidence_fraction,
        },
    )
    _raise_if_generated_dynamics_invalid(trajectory)
    return trajectory


def generate_peek_occlude_retract(
    q0_pose_world_camera: np.ndarray,
    evidence_pose_world_camera: np.ndarray,
    *,
    evidence_frame_id: str,
    peek_distance_m: float | None = None,
    lateral_offset_m: float = 0.35,
    w2_q0_distance_m: float = 0.15,
    direction_sign: int = 1,
    lateral_axis_policy: str = "gravity_aligned_global_z_cross_forward",
    minimum_clearance_m: float = 0.30,
    dynamics_limits: TrajectoryDynamicsLimits | Mapping[str, object] | None = None,
    canonical_sampling: bool = True,
) -> Trajectory:
    """Generate the compact q0→peek→q0-near→revisit shuttle topology."""

    q0 = np.asarray(q0_pose_world_camera, dtype=np.float64)
    evidence = np.asarray(evidence_pose_world_camera, dtype=np.float64)
    evidence_frame_id = str(evidence_frame_id).strip()
    if not evidence_frame_id:
        raise ValueError("evidence_frame_id must be non-empty")
    if not isinstance(canonical_sampling, (bool, np.bool_)) or not canonical_sampling:
        raise ValueError(
            "peek_occlude_retract is defined only on the fixed 357/16 canonical clock"
        )
    forward, lateral, evidence_distance = _metric_template_axes(
        q0,
        evidence,
        direction_sign=direction_sign,
    )
    lateral_axis_policy = str(lateral_axis_policy).strip()
    if lateral_axis_policy not in {
        "gravity_aligned_global_z_cross_forward",
        "q0_camera_right",
    }:
        raise ValueError(
            "lateral_axis_policy must be "
            "gravity_aligned_global_z_cross_forward or q0_camera_right"
        )
    if lateral_axis_policy == "q0_camera_right":
        lateral = _camera_right_lateral_axis(
            q0,
            forward,
            direction_sign=direction_sign,
        )
    if peek_distance_m is None:
        resolved_peek_distance = max(1.50, evidence_distance + 0.75)
    else:
        resolved_peek_distance = float(peek_distance_m)
    lateral_offset_m = float(lateral_offset_m)
    w2_q0_distance_m = float(w2_q0_distance_m)
    minimum_clearance_m = float(minimum_clearance_m)
    if not evidence_distance + 0.45 <= resolved_peek_distance <= 5.0:
        raise ValueError(
            "peek_distance_m must be at least 0.45m beyond qe and at most 5m"
        )
    if not 0.15 <= lateral_offset_m <= 1.5:
        raise ValueError("lateral_offset_m must be in [0.15, 1.5]m")
    if not 0.0 < w2_q0_distance_m <= 0.25:
        raise ValueError("w2_q0_distance_m must be in (0, 0.25]m")
    if not np.isfinite(minimum_clearance_m) or minimum_clearance_m <= 0.0:
        raise ValueError("minimum_clearance_m must be finite and positive")

    origin = q0[:3, 3]
    first_peek_progress = 0.82 * resolved_peek_distance
    # The revisit returns to the exact first-visit pose after W2.
    revisit_peek_progress = first_peek_progress
    positions = np.stack(
        [
            origin,
            origin + 0.5 * evidence_distance * forward,
            evidence[:3, 3],
            (
                origin
                + first_peek_progress * forward
                + lateral_offset_m * lateral
            ),
            origin + w2_q0_distance_m * forward,
            (
                origin
                + revisit_peek_progress * forward
                + lateral_offset_m * lateral
            ),
            origin,
        ]
    )
    rotations = np.stack(
        [
            q0[:3, :3],
            q0[:3, :3],
            evidence[:3, :3],
            evidence[:3, :3],
            q0[:3, :3],
            evidence[:3, :3],
            q0[:3, :3],
        ]
    )
    limits = _coerce_dynamics_limits(dynamics_limits)
    start_holds, end_holds = _moving_completion_segment_holds()
    key_indices = _adaptive_metric_key_indices(
        positions,
        rotations,
        frame_count=CANONICAL_METRIC_TRAJECTORY_FRAME_COUNT,
        fps=CANONICAL_METRIC_TRAJECTORY_FPS,
        limits=limits,
        template_name="peek_occlude_retract",
        segment_start_holds=start_holds,
        segment_end_holds=end_holds,
        segment_minimum_intervals=_natural_completion_segment_minimum_intervals(),
    )
    poses = _interpolate_metric_key_poses(
        CANONICAL_METRIC_TRAJECTORY_FRAME_COUNT,
        key_indices,
        positions,
        rotations,
        segment_start_holds=start_holds,
        segment_end_holds=end_holds,
    )
    peek_orientation_roles = (
        "q0",
        "qe1",
        "qc_first",
        "w2",
        "qc_revisit",
        "q0_return",
    )
    poses = _apply_continuous_canonical_parameterization(
        poses,
        key_indices,
        rotations,
        stop_roles=("q0", "w2", "q0_return"),
        orientation_roles=peek_orientation_roles,
        translation_stop_roles=(
            "q0",
            "qc_first",
            "w2",
            "qc_revisit",
            "q0_return",
        ),
    )
    _bind_completion_revisit(poses, key_indices)
    poses[0] = q0
    poses[key_indices["qe1"]] = evidence
    poses[-1] = q0
    trajectory = Trajectory(
        template_name="peek_occlude_retract",
        difficulty="hard",
        duration_s=(
            CANONICAL_METRIC_TRAJECTORY_FRAME_COUNT
            / CANONICAL_METRIC_TRAJECTORY_FPS
        ),
        fps=CANONICAL_METRIC_TRAJECTORY_FPS,
        poses_world_camera=poses,
        key_indices=key_indices,
        parameters={
            "canonical_sampling": True,
            **_natural_completion_parameters(),
            "direction_sign": int(direction_sign),
            "dynamics_limits": asdict(limits),
            "evidence_frame_id": evidence_frame_id,
            "evidence_pose_world_camera": evidence.tolist(),
            "lateral_direction_world": lateral.tolist(),
            "lateral_axis_policy": lateral_axis_policy,
            "lateral_offset_m": lateral_offset_m,
            "minimum_clearance_m": minimum_clearance_m,
            "peek_distance_m": resolved_peek_distance,
            "time_parameterization": _continuous_time_parameters(
                stop_roles=("q0", "w2", "q0_return"),
                orientation_roles=peek_orientation_roles,
                translation_stop_roles=(
                    "q0",
                    "qc_first",
                    "w2",
                    "qc_revisit",
                    "q0_return",
                ),
            ),
            "w2_q0_distance_m": w2_q0_distance_m,
        },
    )
    _raise_if_generated_dynamics_invalid(trajectory)
    return trajectory


def generate_orbit_around_anchor(
    q0_pose_world_camera: np.ndarray,
    evidence_pose_world_camera: np.ndarray,
    focus_world_m: np.ndarray,
    *,
    evidence_frame_id: str,
    angular_span_deg: float = 120.0,
    direction_sign: int = 1,
    minimum_radius_m: float = 1.0,
    maximum_radius_m: float = 2.0,
    minimum_evidence_radius_m: float | None = None,
    maximum_evidence_radius_m: float | None = None,
    maximum_evidence_radius_delta_m: float | None = None,
    maximum_evidence_radius_delta_fraction: float | None = None,
    minimum_clearance_m: float = 0.30,
    dynamics_limits: TrajectoryDynamicsLimits | Mapping[str, object] | None = None,
    canonical_sampling: bool = True,
) -> Trajectory:
    """Generate a canonical orbit whose ``qe1`` is the selected RGB-D pose."""

    q0 = np.asarray(q0_pose_world_camera, dtype=np.float64)
    evidence = np.asarray(evidence_pose_world_camera, dtype=np.float64)
    focus = np.asarray(focus_world_m, dtype=np.float64)
    evidence_frame_id = str(evidence_frame_id).strip()
    if not evidence_frame_id:
        raise ValueError("evidence_frame_id must be non-empty")
    if focus.shape != (3,) or not np.isfinite(focus).all():
        raise ValueError("focus_world_m must be a finite XYZ point")
    if not isinstance(canonical_sampling, (bool, np.bool_)):
        raise TypeError("canonical_sampling must be a boolean")
    if not bool(canonical_sampling):
        raise ValueError(
            "orbit_around_anchor is defined only on the fixed 357/16 canonical clock"
        )
    resolved_dynamics_limits = _coerce_dynamics_limits(dynamics_limits)
    minimum_radius_m = float(minimum_radius_m)
    maximum_radius_m = float(maximum_radius_m)
    if (
        not np.isfinite(minimum_radius_m)
        or not np.isfinite(maximum_radius_m)
        or minimum_radius_m <= 0.0
        or maximum_radius_m <= minimum_radius_m
    ):
        raise ValueError("orbit radius limits must be finite, positive, and ordered")
    resolved_minimum_evidence_radius_m = (
        minimum_radius_m
        if minimum_evidence_radius_m is None
        else float(minimum_evidence_radius_m)
    )
    resolved_maximum_evidence_radius_m = (
        maximum_radius_m
        if maximum_evidence_radius_m is None
        else float(maximum_evidence_radius_m)
    )
    if (
        not np.isfinite(resolved_minimum_evidence_radius_m)
        or not np.isfinite(resolved_maximum_evidence_radius_m)
        or resolved_minimum_evidence_radius_m <= 0.0
        or resolved_maximum_evidence_radius_m
        <= resolved_minimum_evidence_radius_m
    ):
        raise ValueError(
            "orbit evidence radius limits must be finite, positive, and ordered"
        )
    offset_xy = q0[:2, 3] - focus[:2]
    radius = float(np.linalg.norm(offset_xy))
    if not minimum_radius_m <= radius <= maximum_radius_m:
        raise TrajectoryFeasibilityError(
            "q0 horizontal orbit radius must be in "
            f"[{minimum_radius_m}, {maximum_radius_m}] m"
        )
    evidence_offset_xy = evidence[:2, 3] - focus[:2]
    evidence_radius = float(np.linalg.norm(evidence_offset_xy))
    if not (
        resolved_minimum_evidence_radius_m
        <= evidence_radius
        <= resolved_maximum_evidence_radius_m
    ):
        raise TrajectoryFeasibilityError(
            "registered qe1 horizontal orbit radius must be in "
            f"[{resolved_minimum_evidence_radius_m}, "
            f"{resolved_maximum_evidence_radius_m}] m"
        )
    radius_delta = abs(evidence_radius - radius)
    radius_delta_limits = []
    if maximum_evidence_radius_delta_m is not None:
        value = float(maximum_evidence_radius_delta_m)
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError("maximum evidence radius delta must be positive")
        radius_delta_limits.append(value)
    if maximum_evidence_radius_delta_fraction is not None:
        value = float(maximum_evidence_radius_delta_fraction)
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError("maximum evidence radius delta fraction must be positive")
        radius_delta_limits.append(value * radius)
    resolved_maximum_radius_delta_m = (
        min(radius_delta_limits) if radius_delta_limits else None
    )
    if (
        resolved_maximum_radius_delta_m is not None
        and radius_delta > resolved_maximum_radius_delta_m + 1e-9
    ):
        raise TrajectoryFeasibilityError(
            "registered qe1 orbit radius differs from q0 by "
            f"{radius_delta:.3f}m; maximum is "
            f"{resolved_maximum_radius_delta_m:.3f}m"
        )
    angular_span_deg = float(angular_span_deg)
    if angular_span_deg < 120.0:
        raise ValueError("orbit angular span must be at least 120 degrees")
    if direction_sign not in {-1, 1}:
        raise ValueError("direction_sign must be -1 or +1")
    initial_angle = float(np.arctan2(offset_xy[1], offset_xy[0]))
    evidence_angle = float(
        np.arctan2(evidence_offset_xy[1], evidence_offset_xy[0])
    )
    if direction_sign == 1:
        evidence_delta = (evidence_angle - initial_angle) % (2.0 * np.pi)
    else:
        evidence_delta = (initial_angle - evidence_angle) % (2.0 * np.pi)
    if evidence_delta > np.deg2rad(180.0):
        raise TrajectoryFeasibilityError(
            "registered qe1 lies more than 180deg along orbit direction"
        )
    span = max(np.deg2rad(angular_span_deg), evidence_delta + np.deg2rad(20.0))
    relative = np.asarray(
        [
            0.0,
            0.5 * evidence_delta,
            evidence_delta,
            evidence_delta + 0.55 * (span - evidence_delta),
            span,
            # A distinct but nearby return-side observation retains shared
            # q0-unseen support across the long natural window.
            evidence_delta + 0.50 * (span - evidence_delta),
            0.0,
        ]
    )
    relative[5] = relative[3]
    angles = initial_angle + direction_sign * relative
    radii = np.asarray(
        [
            radius,
            0.5 * (radius + evidence_radius),
            evidence_radius,
            0.5 * (radius + evidence_radius),
            radius,
            0.5 * (radius + evidence_radius),
            radius,
        ]
    )
    radii[5] = radii[3]
    heights = np.asarray(
        [
            q0[2, 3],
            0.5 * (q0[2, 3] + evidence[2, 3]),
            evidence[2, 3],
            0.5 * (q0[2, 3] + evidence[2, 3]),
            q0[2, 3],
            0.5 * (q0[2, 3] + evidence[2, 3]),
            q0[2, 3],
        ]
    )
    heights[5] = heights[3]
    positions = np.zeros((len(METRIC_TRAJECTORY_EVENT_ROLES), 3), dtype=np.float64)
    positions[:, 0] = focus[0] + radii * np.cos(angles)
    positions[:, 1] = focus[1] + radii * np.sin(angles)
    positions[:, 2] = heights
    positions[2] = evidence[:3, 3]
    rotations = np.stack(
        [_look_at_opencv(position, focus) for position in positions]
    )
    rotations[0] = q0[:3, :3]
    rotations[2] = evidence[:3, :3]
    rotations[-1] = q0[:3, :3]
    derivative_maxima = np.asarray(
        [
            _orbit_normalized_translation_derivative_maxima(
                angles[segment],
                angles[segment + 1],
                radii[segment],
                radii[segment + 1],
                heights[segment],
                heights[segment + 1],
            )
            for segment in range(len(angles) - 1)
        ],
        dtype=np.float64,
    )
    start_holds, end_holds = _moving_completion_segment_holds()
    key_indices = _adaptive_metric_key_indices(
        positions,
        rotations,
        frame_count=CANONICAL_METRIC_TRAJECTORY_FRAME_COUNT,
        fps=CANONICAL_METRIC_TRAJECTORY_FPS,
        limits=resolved_dynamics_limits,
        template_name="orbit_around_anchor",
        segment_start_holds=start_holds,
        segment_end_holds=end_holds,
        segment_translation_derivative_maxima=derivative_maxima,
        segment_minimum_intervals=_natural_completion_segment_minimum_intervals(),
    )
    poses = _interpolate_orbit_key_poses(
        CANONICAL_METRIC_TRAJECTORY_FRAME_COUNT,
        key_indices,
        positions,
        rotations,
        angles,
        radii,
        heights,
        focus[:2],
        segment_start_holds=start_holds,
        segment_end_holds=end_holds,
    )
    orbit_orientation_roles = tuple(METRIC_TRAJECTORY_EVENT_ROLES)
    poses = _apply_continuous_canonical_parameterization(
        poses,
        key_indices,
        rotations,
        stop_roles=("q0", "w2", "q0_return"),
        orientation_roles=orbit_orientation_roles,
        polar_focus_xy=focus[:2],
    )
    _bind_completion_revisit(poses, key_indices)
    poses[0] = q0
    poses[key_indices["qe1"]] = evidence
    poses[-1] = q0
    trajectory = Trajectory(
        template_name="orbit_around_anchor",
        difficulty="medium",
        duration_s=(
            CANONICAL_METRIC_TRAJECTORY_FRAME_COUNT
            / CANONICAL_METRIC_TRAJECTORY_FPS
        ),
        fps=CANONICAL_METRIC_TRAJECTORY_FPS,
        poses_world_camera=poses,
        key_indices=key_indices,
        parameters={
            "angular_span_deg": float(angular_span_deg),
            "canonical_sampling": True,
            **_natural_completion_parameters(),
            "direction_sign": int(direction_sign),
            "dynamics_limits": asdict(resolved_dynamics_limits),
            "evidence_frame_id": evidence_frame_id,
            "evidence_pose_world_camera": evidence.tolist(),
            "focus_world_m": focus.tolist(),
            "initial_orbit_angle_rad": initial_angle,
            "minimum_clearance_m": float(minimum_clearance_m),
            "minimum_orbit_radius_m": minimum_radius_m,
            "maximum_orbit_radius_m": maximum_radius_m,
            "minimum_evidence_orbit_radius_m": resolved_minimum_evidence_radius_m,
            "maximum_evidence_orbit_radius_m": resolved_maximum_evidence_radius_m,
            "maximum_evidence_radius_delta_m": resolved_maximum_radius_delta_m,
            "q0_radius_m": radius,
            "qe1_radius_m": evidence_radius,
            "qe1_q0_radius_delta_m": radius_delta,
            "realized_angular_span_deg": float(np.rad2deg(span)),
            "time_parameterization": _continuous_time_parameters(
                stop_roles=("q0", "w2", "q0_return"),
                orientation_roles=orbit_orientation_roles,
                translation_clock=(
                    "monotone_polar_arc_length_cubic_hermite"
                ),
            ),
        },
    )
    _raise_if_generated_dynamics_invalid(trajectory)
    return trajectory


