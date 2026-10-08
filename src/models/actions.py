"""Convert requested camera trajectories into discrete movement/look actions."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

MOVE_KEYS = ('W', 'A', 'S', 'D')
LOOK_KEYS = ('I', 'J', 'K', 'L')
KEYS = MOVE_KEYS + LOOK_KEYS
_LOOK_PAIRS = (('L', 'J'), ('I', 'K'))  # (positive, negative) for yaw-right, pitch-up


def smooth_waypoints(poses_c2w: np.ndarray, waypoints: Iterable[int] = (), *,
                     fps: float = 16.0, cutoff_hz: float = 0.2) -> np.ndarray:
    """Waypoint-constrained smoothing of positions and quaternions."""
    from scipy.sparse import diags, identity
    from scipy.sparse.linalg import spsolve
    from scipy.spatial.transform import Rotation

    poses = np.asarray(poses_c2w, dtype=np.float64)
    n = len(poses)
    fixed = np.array(sorted({0, n - 1, *[int(i) for i in waypoints if 0 <= int(i) < n]}))
    if n < 4 or len(fixed) == n:
        return poses.copy()
    q = Rotation.from_matrix(poses[:, :3, :3]).as_quat()
    for i in range(1, n):
        if q[i] @ q[i - 1] < 0:
            q[i] *= -1
    x = np.concatenate([poses[:, :3, 3], q], axis=1)
    strength = 1.0 / (2.0 - 2.0 * np.cos(2.0 * np.pi * cutoff_hz / fps)) ** 2
    d2 = diags([np.ones(n - 2), -2 * np.ones(n - 2), np.ones(n - 2)], [0, 1, 2], shape=(n - 2, n), format='csc')
    h = (identity(n, format='csc') + strength * (d2.T @ d2)).tocsc()
    free = np.setdiff1d(np.arange(n), fixed)
    y = x.copy()
    y[free] = spsolve(h[free][:, free], x[free] - h[free][:, fixed] @ x[fixed])
    norm = np.linalg.norm(y[:, 3:], axis=1)
    if norm.min() < 0.25:
        raise ValueError('quaternion smoothing degenerated; inspect the trajectory')
    result = poses.copy()
    result[:, :3, 3] = y[:, :3]
    result[:, :3, :3] = Rotation.from_quat(y[:, 3:] / norm[:, None]).as_matrix()
    result[fixed] = poses[fixed]
    return result


def local_motion(poses_c2w: np.ndarray, *, static_fraction: float = 1e-3) -> tuple[np.ndarray, np.ndarray]:
    """Camera-frame translation steps ``[N-1, 3]`` and yaw/pitch steps in degrees ``[N-1, 2]``."""
    from scipy.spatial.transform import Rotation

    poses = np.asarray(poses_c2w, dtype=np.float64)
    delta = np.linalg.inv(poses[:-1]) @ poses[1:]
    translation = delta[:, :3, 3].copy()
    rotvec = np.degrees(Rotation.from_matrix(delta[:, :3, :3]).as_rotvec())
    # Rotation about +y turns the optical axis toward +x (right); about +x toward -y (up).
    angular = np.column_stack([rotvec[:, 1], rotvec[:, 0]])
    speed = np.linalg.norm(np.diff(poses[:, :3, 3], axis=0), axis=1)
    translation[speed <= max(float(speed.max()) * static_fraction, 1e-12)] = 0
    return translation, angular


def vote_move_keys(translation: np.ndarray, threshold: float = 0.5) -> list[str]:
    """Majority vote of unit camera-frame directions over one action window."""
    if len(translation) == 0:
        return []
    unit = translation / np.maximum(np.linalg.norm(translation, axis=1, keepdims=True), 1e-12)
    keys = []
    for axis, positive, negative in ((2, 'W', 'S'), (0, 'D', 'A')):
        votes = int((unit[:, axis] > threshold).sum() - (unit[:, axis] < -threshold).sum())
        if votes > len(unit) / 2:
            keys.append(positive)
        elif votes < -len(unit) / 2:
            keys.append(negative)
    return keys


def _prepare(poses_c2w, events, template, smoothing_hz, fps):
    poses = np.asarray(poses_c2w, dtype=np.float64)
    if smoothing_hz:
        poses = smooth_waypoints(poses, events, fps=fps, cutoff_hz=smoothing_hz)
    translation, angular = local_motion(poses)
    if template == 'yaw_return':
        translation[:] = 0
    return poses, translation, angular


@dataclass(frozen=True)
class BlockActionSpec:
    """Generation-block layout and turning response of a streaming action model."""

    first_block_frames: int = 9
    block_frames: int = 12
    yaw_step_deg: float = 20.0
    pitch_step_deg: float = 17.0
    smoothing_hz: float | None = 0.2
    fps: float = 16.0


def block_actions(poses_c2w: np.ndarray, *, events: Sequence[int] = (), template: str = '',
                  spec: BlockActionSpec = BlockActionSpec()) -> dict:
    """One 8-dimensional ``W A S D I J K L`` action per generation block."""
    _, translation, angular = _prepare(poses_c2w, events, template, spec.smoothing_hz, spec.fps)
    n = len(translation) + 1
    lead = spec.block_frames - spec.first_block_frames
    count = (n + lead + spec.block_frames - 1) // spec.block_frames
    target = np.vstack([np.zeros(2), np.cumsum(angular, axis=0)])
    emitted = np.zeros(2)
    steps = (spec.yaw_step_deg, spec.pitch_step_deg)
    vectors, blocks = [], []
    for k in range(count):
        start = 0 if k == 0 else spec.block_frames * k - lead
        stop = min(spec.first_block_frames if k == 0 else spec.block_frames * k + spec.first_block_frames, n)
        a, b = max(0, start - 1), max(stop - 1, 0)
        keys = vote_move_keys(translation[a:b])
        increment = target[b] - target[a]
        for axis, (positive, negative) in enumerate(_LOOK_PAIRS):
            error = target[b, axis] - emitted[axis]
            if increment[axis] > 0.05 and error >= steps[axis] / 2:
                keys.append(positive)
                emitted[axis] += steps[axis]
            elif increment[axis] < -0.05 and error <= -steps[axis] / 2:
                keys.append(negative)
                emitted[axis] -= steps[axis]
        vectors.append([int(key in keys) for key in KEYS])
        blocks.append({'block': k, 'frames': [start, stop - 1], 'keys': keys,
                       'target_yaw_pitch_deg': target[b].tolist(), 'emitted_yaw_pitch_deg': emitted.tolist()})
    generated = spec.first_block_frames + spec.block_frames * (count - 1)
    return {'keys': list(KEYS), 'actions': vectors, 'blocks': blocks,
            'target_frames': n, 'generated_frames': generated, 'spec': spec.__dict__}


@dataclass(frozen=True)
class TimedActionSpec:
    """Held-key schedule for interactive models driven in real time."""

    yaw_deg_per_s: float
    pitch_deg_per_s: float
    control_seconds: dict = field(default_factory=lambda: {'short': 20.0, 'long': 54.0})
    window_frames: int = 12
    min_pulse_s: float = 0.1
    tail_s: float = 2.0
    smoothing_hz: float | None = 0.2
    fps: float = 16.0


def timed_actions(poses_c2w: np.ndarray, *, events: Sequence[int] = (), template: str = '',
                  spec: TimedActionSpec, horizon: str | None = None) -> dict:
    """Return ``[{start_s, end_s, keys}]`` intervals with constant held keys."""
    _, translation, angular = _prepare(poses_c2w, events, template, spec.smoothing_hz, spec.fps)
    n = len(translation) + 1
    horizon = horizon or ('long' if n >= 961 else 'short')
    duration = float(spec.control_seconds[horizon])
    seconds_per_frame = duration / (n - 1)
    rates = np.array([spec.yaw_deg_per_s, spec.pitch_deg_per_s], dtype=np.float64)
    debt, last = np.zeros(2), np.zeros(2, dtype=int)
    intervals, saturated = [], []
    for lo in range(0, n - 1, spec.window_frames):
        hi = min(lo + spec.window_frames, n - 1)
        t0, t1 = lo * seconds_per_frame, hi * seconds_per_frame
        span = t1 - t0
        move = vote_move_keys(translation[lo:hi])
        increment = angular[lo:hi].sum(axis=0)
        signs, holds = [], []
        for axis in range(2):
            sign = 1 if increment[axis] > 0.05 else -1 if increment[axis] < -0.05 else 0
            if sign and last[axis] and sign != last[axis]:
                debt[axis] = 0.0
            error = debt[axis] + increment[axis]
            hold = min(abs(error) / rates[axis], span) if sign else 0.0
            if hold < spec.min_pulse_s:
                hold = 0.0
            if sign and abs(error) > rates[axis] * span + 0.01:
                saturated.append({'start_s': t0, 'axis': ('yaw', 'pitch')[axis], 'requested_deg': float(error)})
            debt[axis] = error - sign * hold * rates[axis]
            if sign:
                last[axis] = sign
            signs.append(sign)
            holds.append(hold)
        cuts = sorted({0.0, span, *[h for h in holds if 0 < h < span]})
        for x, y in zip(cuts[:-1], cuts[1:]):
            keys = list(move)
            for axis, (positive, negative) in enumerate(_LOOK_PAIRS):
                if holds[axis] > x + 1e-9:
                    keys.append(positive if signs[axis] > 0 else negative)
            if intervals and intervals[-1]['keys'] == keys:
                intervals[-1]['end_s'] = t0 + y
            else:
                intervals.append({'start_s': t0 + x, 'end_s': t0 + y, 'keys': keys})
    if spec.tail_s > 0:
        intervals.append({'start_s': duration, 'end_s': duration + spec.tail_s, 'keys': []})
    return {'intervals': intervals, 'control_duration_s': duration, 'tail_s': spec.tail_s,
            'target_frames': n, 'saturated_windows': saturated,
            'final_angle_debt_deg': debt.tolist(), 'spec': spec.__dict__}
