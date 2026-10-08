"""Describe a requested camera trajectory as a timed motion script for TI2V models."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

TEMPLATES = ('yaw_return', 'lateral_out_and_back', 'forward_backward', 'arc_return',
             'orbit_around_anchor', 'peek_occlude_retract')
POLICY = dict(smoothing_sigma_seconds=0.6, minimum_segment_seconds=1.5,
              minimum_segment_duration_fraction=0.04, stop_fraction_of_peak=0.05,
              minimum_yaw_rate_deg_s=0.5, displayed_time_quantum_seconds=0.5)
OVERVIEW = {
    'yaw_return': 'The camera makes a smooth look-around movement, then turns back toward the opening view.',
    'lateral_out_and_back': 'The camera follows a smooth lateral out-and-back path.',
    'forward_backward': 'The camera moves out along a forward/backward path, then retraces it.',
    'arc_return': 'The camera follows a smooth curved exploration path and then returns.',
    'orbit_around_anchor': 'The camera moves around the scene along a curved path while turning its view.',
    'peek_occlude_retract': 'The camera moves out to peek into the scene, then withdraws along the same path.',
}


def _runs(labels):
    bounds = np.r_[0, np.flatnonzero(labels[1:] != labels[:-1]) + 1, len(labels)]
    return [(int(a), int(b), int(labels[a])) for a, b in zip(bounds[:-1], bounds[1:])]


def _path_samples(points, count=64):
    distance = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    distance, indices = np.unique(distance, return_index=True)
    if len(distance) < 2:
        return np.repeat(points[:1], count, axis=0)
    return np.stack([np.interp(np.linspace(0, distance[-1], count), distance, points[indices, k])
                     for k in range(3)], axis=1)


def _curve_description(position, groups):
    extent = max(float(np.linalg.norm(position - position[0], axis=1).max()), 1e-12)
    peak = int(np.argmax(np.linalg.norm(position - position[0], axis=1)))
    error = None
    if 0 < peak < len(position) - 1:
        outward = _path_samples(position[:peak + 1])
        inward = _path_samples(position[peak:][::-1])
        error = float(np.linalg.norm(outward - inward, axis=1).max() / extent)
    descriptions = []
    for a, b, _ in groups:
        samples = _path_samples(position[a:b + 1])
        delta = samples[-1] - samples[0]
        words = [pos if delta[axis] > 0 else neg
                 for axis, pos, neg in ((2, 'forward', 'backward'), (0, 'right', 'left'), (1, 'downward', 'upward'))
                 if abs(delta[axis]) > max(np.linalg.norm(delta) * .25, extent * .02)]
        start, end = samples[12] - samples[0], samples[-1] - samples[-13]
        bend = float(np.degrees(np.arctan2(start[2] * end[0] - start[0] * end[2], start[[0, 2]] @ end[[0, 2]])))
        travel = float(np.linalg.norm(np.diff(position[a:b + 1], axis=0), axis=1).sum() / extent)
        descriptions.append(dict(direction=' and '.join(words), bend_deg=bend, travel=travel))
    return descriptions, error


def _segments(rate, times, angular=False):
    cutoff = max(float(np.percentile(np.abs(rate), 95)) * POLICY['stop_fraction_of_peak'],
                 POLICY['minimum_yaw_rate_deg_s'] if angular else 0.)
    labels = np.where(np.abs(rate) > cutoff, np.sign(rate), 0).astype(int)
    minimum = max(POLICY['minimum_segment_seconds'],
                  (times[-1] - times[0]) * POLICY['minimum_segment_duration_fraction'])
    while True:
        groups = _runs(labels)
        short = [(times[b] - times[a], i) for i, (a, b, _) in enumerate(groups) if times[b] - times[a] < minimum]
        if len(groups) < 2 or not short:
            return groups
        _, i = min(short)
        a, b, _ = groups[i]
        j = max((j for j in (i - 1, i + 1) if 0 <= j < len(groups)),
                key=lambda j: times[groups[j][1]] - times[groups[j][0]])
        labels[a:b] = groups[j][2]


@dataclass
class Motion:
    template: str
    groups: list
    angular: bool
    curved: bool
    axis: int
    closed: bool
    curves: list
    retraces_path: bool
    yaw_extent_deg: float
    near_start: list
    segment_yaw_deg: list
    anchor_label: str | None
    warnings: list


def analyse(poses_c2w, timestamps_s, template, anchor_label=None, *, expect_return=True) -> Motion:
    """Group a trajectory into sustained motion phases."""
    from scipy.ndimage import gaussian_filter1d
    from scipy.spatial.transform import Rotation

    if template not in TEMPLATES:
        raise ValueError(f'unknown trajectory template: {template!r}')
    poses = np.asarray(poses_c2w, dtype=np.float64)
    times = np.asarray(timestamps_s, dtype=np.float64)
    dt = np.diff(times)
    sigma = POLICY['smoothing_sigma_seconds'] / np.median(dt)
    position = (poses[:, :3, 3] - poses[0, :3, 3]) @ poses[0, :3, :3]
    smooth_position = gaussian_filter1d(position, sigma, axis=0, mode='nearest')
    velocity = np.diff(smooth_position, axis=0) / dt[:, None]
    relative = np.swapaxes(poses[:-1, :3, :3], 1, 2) @ poses[1:, :3, :3]
    local_omega = Rotation.from_matrix(relative).as_rotvec(degrees=True) / dt[:, None]
    world_omega = np.einsum('nij,nj->ni', poses[:-1, :3, :3], local_omega)
    yaw_rate = gaussian_filter1d(world_omega @ poses[0, :3, 1], sigma, mode='nearest')
    curved = template in ('arc_return', 'orbit_around_anchor')
    angular = template == 'yaw_return'
    axis = 0 if template == 'lateral_out_and_back' else 2
    if template == 'peek_occlude_retract':
        axis = int(np.argmax(np.ptp(position, axis=0)))
    warnings = []
    if angular and np.max(np.abs(yaw_rate)) < POLICY['minimum_yaw_rate_deg_s']:
        warnings.append('negligible yaw for a turning template')
        angular, axis = False, int(np.argmax(np.ptp(position, axis=0)))
    distance = np.linalg.norm(smooth_position - smooth_position[0], axis=1)
    rate = np.diff(distance) / dt if curved else yaw_rate if angular else velocity[:, axis]
    groups = _segments(rate, times, angular)
    path = float(np.linalg.norm(np.diff(position, axis=0), axis=1).sum())
    return_ratio = float(np.linalg.norm(position[-1])) / max(path, 1e-12)
    return_angle = float(Rotation.from_matrix(poses[0, :3, :3].T @ poses[-1, :3, :3]).magnitude() * 180 / np.pi)
    closed = expect_return and return_ratio <= 1e-3 and return_angle <= 1.
    signs = [s for _, _, s in groups if s]
    if template != 'orbit_around_anchor' and len(set(signs)) < 2:
        warnings.append('no sustained reversal')
    forward = poses[0, :3, :3].T @ poses[:, :3, 2, None]
    yaw = np.degrees(np.unwrap(np.arctan2(forward[:, 0, 0], forward[:, 2, 0])))
    yaw -= yaw[0]
    displacement = np.linalg.norm(position, axis=1)
    angles = Rotation.from_matrix(poses[0, :3, :3].T @ poses[:, :3, :3]).magnitude() * 180 / np.pi
    near_start = [bool(displacement[b] <= max(float(displacement.max()) * .1, 1e-12) and angles[b] <= 5.)
                  for _, b, _ in groups]
    curves, retrace = _curve_description(position, groups) if curved else ([], None)
    return Motion(template, groups, angular, curved, axis, closed, curves,
                  retrace is not None and retrace <= .05, float(np.max(np.abs(yaw))), near_start,
                  [float(yaw[b] - yaw[a]) for a, b, _ in groups], anchor_label, warnings)


def _phase_sentence(motion: Motion, i: int, sign: int, first_sign: int, reversed_once: bool):
    if motion.curved:
        curve = motion.curves[i]
        if curve['travel'] < .02:
            action = 'pause the camera movement'
        else:
            direction = (' ' + curve['direction']) if curve['direction'] else ''
            if sign < 0 and motion.retraces_path:
                action = 'retrace the same outward arc' + direction + ' toward the starting position'
            elif sign < 0:
                action = 'follow a separate returning arc' + direction + ' toward the starting position'
            else:
                action = 'move' + direction + ' along the outward arc' if sign > 0 else 'continue along the arc'
            if abs(curve['bend_deg']) >= 15:
                action += ', with the path bending ' + ('right' if curve['bend_deg'] > 0 else 'left') \
                          + ' relative to the direction of travel'
        angle = 5 * round(motion.segment_yaw_deg[i] / 5)
        if motion.template != 'orbit_around_anchor':
            action += (f'; turn the view {"right" if angle > 0 else "left"} by approximately {abs(angle):g} degrees'
                       if abs(angle) >= 5 else '; keep the viewing direction approximately unchanged')
        return action, reversed_once
    if sign == 0:
        return 'hold the camera still', reversed_once
    if motion.angular:
        action = f'turn the view smoothly to the {"right" if sign > 0 else "left"}'
        if sign != first_sign and not reversed_once:
            action, reversed_once = 'reverse the turning direction and ' + action, True
        angle = 5 * round(abs(motion.segment_yaw_deg[i]) / 5)
        if angle >= 5:
            action += f' by approximately {angle:g} degrees'
        return action, reversed_once
    direction = [('right', 'left'), ('downward', 'upward'), ('forward', 'backward')][motion.axis][0 if sign > 0 else 1]
    direction = 'to the ' + direction if motion.axis == 0 else direction
    action = f'move steadily {direction}'
    if sign != first_sign and not reversed_once:
        action, reversed_once = 'reverse direction and ' + action + ' along the return leg', True
    return action, reversed_once


def render(motion: Motion, frame_count: int, clock, *, continuation=False) -> dict:
    """Render motion phases as text; ``clock(i)`` converts a frame index to seconds."""
    quantum = POLICY['displayed_time_quantum_seconds']
    end = clock(frame_count - 1)
    boundaries = [0.] + [round(clock(b) / quantum) * quantum for _, b, _ in motion.groups[:-1]] + [end]
    if continuation:
        lines = ['Continue the same continuous shot from the provided frame.']
    else:
        overview = OVERVIEW[motion.template]
        if 'no sustained reversal' in motion.warnings:
            overview = 'The camera follows one smooth, continuous exploration path.'
        if motion.template == 'orbit_around_anchor':
            overview = f"Follow a curved path around the {motion.anchor_label or 'subject shown in the opening view'}."
        lines = [overview, 'Use one continuous shot with a fixed focal length.']
    if motion.angular:
        angle = 5 * round(motion.yaw_extent_deg / 5)
        if angle >= 5 and not continuation:
            lines.append(f'Keep the maximum horizontal turn to approximately {angle:g} degrees from the opening viewing direction.')
    elif not continuation:
        lines.append('Use a slow, cautious walking pace for the small outward and retreating movements.'
                     if motion.template == 'peek_occlude_retract' else
                     'Move at a relaxed, steady walking pace, easing smoothly into stops and reversals.')
    if motion.template == 'orbit_around_anchor':
        lines.append('Keep the camera aimed at the same subject throughout the outward and return movements.')
    if motion.curved:
        lines.append('Movement directions below are relative to the '
                     + ('current' if continuation else 'opening') + ' camera orientation, not the changing viewing direction.')
    first_sign = next((s for _, _, s in motion.groups if s), 0)
    reversed_once, segments = False, []
    for i, (a, b, sign) in enumerate(motion.groups):
        action, reversed_once = _phase_sentence(motion, i, sign, first_sign, reversed_once)
        text = f'From {boundaries[i]:g} to {boundaries[i + 1]:g} seconds, {action}.'
        if motion.template == 'peek_occlude_retract' and motion.near_start[i] and i < len(motion.groups) - 1:
            text += ' End this leg near the starting viewpoint before the next peek.'
        lines.append(text)
        segments.append(dict(start_frame=a, end_frame=b, start_s=boundaries[i], end_s=boundaries[i + 1], text=text))
    if motion.closed:
        lines.append('Finish at the original camera position and orientation, matching the opening view.')
    lines.append('Preserve the scene layout and objects throughout, including when revisiting. Do not cut or zoom.')
    return dict(duration_s=end, camera_prompt='\n'.join(lines), segments=segments, warnings=motion.warnings)


def camera_prompt(case, native_fps: float | None = None) -> dict:
    """Motion script for the full trajectory of a :class:`~src.models.case.ModelCase`."""
    motion = analyse(case.poses_c2w, case.timestamps_s, case.template, case.anchor_label)
    clock = (lambda i: i / native_fps) if native_fps else (lambda i: float(case.timestamps_s[i] - case.timestamps_s[0]))
    result = render(motion, case.num_frames, clock)
    result['prompt'] = case.prompt + '\n\n' + result['camera_prompt']
    return result


def window_prompts(case, spans, native_fps: float | None = None) -> list[dict]:
    """One prompt per generation window ``(start, end)`` (end exclusive, may exceed the trajectory)."""
    results = []
    n = case.num_frames
    for k, (start, end) in enumerate(spans):
        stop = min(end, n)
        if stop - start < 2:
            results.append(dict(window=(start, end), prompt=case.prompt, camera_prompt='', segments=[]))
            continue
        poses = np.linalg.inv(case.poses_c2w[start]) @ case.poses_c2w[start:stop]
        times = case.timestamps_s[start:stop] - case.timestamps_s[start]
        final = stop == n
        motion = analyse(poses, times, case.template, case.anchor_label, expect_return=False)
        if final:
            motion.closed = bool(np.allclose(case.poses_c2w[-1], case.poses_c2w[0], atol=1e-5))
        clock = (lambda i: i / native_fps) if native_fps else (lambda i, t=times: float(t[i]))
        rendered = render(motion, stop - start, clock, continuation=k > 0)
        rendered['window'] = (start, end)
        rendered['prompt'] = case.prompt + '\n\n' + rendered['camera_prompt']
        results.append(rendered)
    return results
