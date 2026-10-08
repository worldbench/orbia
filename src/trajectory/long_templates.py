"""Extend an admitted short ORBIA route to the six long templates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from src.trajectory.long_horizon import LONG_HORIZON, RevisitPair, validate_revisit_excursion
from src.trajectory.templates import Trajectory


LONG_TEMPLATE_NAMES = (
    "forward_backward",
    "lateral_out_and_back",
    "yaw_return",
    "arc_return",
    "orbit_around_anchor",
    "peek_occlude_retract",
)

LONG_REVISIT_PAIRS = (
    RevisitPair("C", "qc_first", "qc_revisit"),
    RevisitPair("A", "qa_first", "qa_revisit"),
    RevisitPair("B", "qb_first", "qb_revisit"),
)

_EVENTS_AFTER_QE = (
    "qc_first",
    "qa_first",
    "qb_first",
    "w",
    "qb_revisit",
    "qa_revisit",
    "qc_revisit",
    "q0_return",
)
_ALIASES = {"peekaboo": "peek_occlude_retract"}


@dataclass(frozen=True)
class InheritedMotion:
    linear_speed: float
    angular_speed_deg_s: float
    moving_linear_sample_count: int
    moving_angular_sample_count: int


def _positive_median(values: np.ndarray, *, threshold: float) -> tuple[float, int]:
    moving = np.asarray(values, dtype=np.float64)
    moving = moving[np.isfinite(moving) & (moving > threshold)]
    if not len(moving):
        return 0.0, 0
    return float(np.median(moving)), int(len(moving))


def infer_short_motion(trajectory: Trajectory) -> InheritedMotion:
    """Measure representative local speeds without counting stationary holds."""

    poses = trajectory.poses_world_camera
    dt = 1.0 / float(trajectory.fps)
    linear = np.linalg.norm(np.diff(poses[:, :3, 3], axis=0), axis=1) / dt
    relative = np.einsum(
        "nij,njk->nik",
        np.swapaxes(poses[:-1, :3, :3], 1, 2),
        poses[1:, :3, :3],
    )
    angular = np.rad2deg(Rotation.from_matrix(relative).magnitude()) / dt
    linear_speed, linear_count = _positive_median(linear, threshold=1e-6)
    angular_speed, angular_count = _positive_median(angular, threshold=1e-5)
    return InheritedMotion(linear_speed, angular_speed, linear_count, angular_count)


def _qe_event(key_indices: Mapping[str, int]) -> str:
    for name in ("qe1", "qe"):
        if name in key_indices:
            return name
    raise ValueError("short trajectory must register qe1 or qe")


def _unit(value: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(value))
    if norm > 1e-9:
        return value / norm
    fallback_norm = float(np.linalg.norm(fallback))
    if fallback_norm <= 1e-9:
        raise ValueError("cannot construct a local route basis")
    return fallback / fallback_norm


def _route_basis(q0: np.ndarray, qe: np.ndarray, template: str) -> tuple[np.ndarray, np.ndarray]:
    displacement = qe[:3, 3] - q0[:3, 3]
    camera_right = q0[:3, 0]
    camera_forward = q0[:3, 2]
    if template == "lateral_out_and_back":
        primary = _unit(displacement, camera_right)
    else:
        primary = _unit(displacement, camera_forward)
    lateral_raw = camera_right - primary * float(np.dot(camera_right, primary))
    if np.linalg.norm(lateral_raw) <= 1e-9:
        lateral_raw = camera_forward - primary * float(np.dot(camera_forward, primary))
    if np.linalg.norm(lateral_raw) <= 1e-9:
        lateral_raw = np.cross(primary, q0[:3, 1])
    return primary, _unit(lateral_raw, camera_right)


def _solve_amplitude(
    target_length: float,
    length_at,
    *,
    reference_distance: float,
) -> float:
    if target_length <= length_at(0.0) + 1e-9:
        return max(reference_distance * 0.25, 1e-3)
    high = max(reference_distance, 0.25)
    while length_at(high) < target_length and high < 1e6:
        high *= 2.0
    low = 0.0
    for _ in range(64):
        middle = 0.5 * (low + high)
        if length_at(middle) < target_length:
            low = middle
        else:
            high = middle
    return float(0.5 * (low + high))


def _allocate_segment_frames(weights: np.ndarray, total_frames: int, fps: int) -> np.ndarray:
    """Allocate exact event frames; B-W and W-B each get at least five seconds."""

    if total_frames < 8:
        raise ValueError("long profile does not leave enough samples after QE")
    minimum = np.ones(8, dtype=np.int64)
    minimum[3:5] = int(5 * fps)
    if int(minimum.sum()) > total_frames:
        raise ValueError("post-QE interval is too short for the revisit schedule")
    safe = np.maximum(np.asarray(weights, dtype=np.float64), 1e-9)
    shares = total_frames * safe / safe.sum()
    fixed = np.zeros(8, dtype=bool)
    while np.any((shares < minimum) & ~fixed):
        fixed |= shares < minimum
        shares[fixed] = minimum[fixed]
        shares[~fixed] = (total_frames - shares[fixed].sum()) * safe[~fixed] / safe[~fixed].sum()
    counts = np.floor(shares).astype(np.int64)
    remainder = total_frames - int(counts.sum())
    order = np.argsort(-(shares - counts), kind="stable")
    counts[order[:remainder]] += 1
    return counts



def _segment_distance(poses: np.ndarray, *, angular: bool = False) -> np.ndarray:
    if angular:
        relative = np.swapaxes(poses[:-1, :3, :3], 1, 2) @ poses[1:, :3, :3]
        steps = np.rad2deg(Rotation.from_matrix(relative).magnitude())
    else:
        steps = np.linalg.norm(np.diff(poses[:, :3, 3], axis=0), axis=1)
    return np.r_[0.0, np.cumsum(steps)]


def _long_segments(short: Trajectory, qe_index: int, template: str,
                   target_length: float, arc_parameters: Mapping | None = None) -> tuple[list[np.ndarray], float, dict]:
    q0, qe = short.poses_world_camera[[0, qe_index]]
    primary, lateral = _route_basis(q0, qe, template)
    direction_deg=float(short.parameters.get('long_direction_deg',0.))
    direction_limit = 20 if template in {'forward_backward','lateral_out_and_back'} else 30
    if not np.isfinite(direction_deg) or abs(direction_deg)>direction_limit:
        raise ValueError(f'long direction change must be within {direction_limit} degrees')
    details = {'long_direction_deg':direction_deg,
               'long_heading_deg':float(short.parameters.get('long_heading_deg',0.)),
               'long_speed_ratio':float(short.parameters.get('long_speed_ratio',1.))}
    arc = dict(arc_parameters or {})
    if arc:
        if template != "arc_return":
            raise ValueError("arc_parameters only apply to arc_return")
        if set(arc) - {"width_ratio", "direction_deg", "circle_ratios", "length_ratio"}:
            raise ValueError("unsupported arc parameter")
    width = float(arc.get("width_ratio", 1.0))
    direction = float(arc.get("direction_deg", 0.0))
    circles = np.asarray(arc.get("circle_ratios", [1.0,.85,.7]), dtype=float)
    if not np.isfinite([width,direction]).all() or not .3 <= width <= 1.5 or abs(direction)>60:
        raise ValueError("invalid arc width or direction")
    if circles.shape != (3,) or not np.isfinite(circles).all() or np.any(circles<.2) or np.any(circles>1.5):
        raise ValueError("invalid arc circle ratios")
    if template == "arc_return":
        theta=np.deg2rad(direction)
        primary,lateral = (np.cos(theta)*primary+np.sin(theta)*lateral,
                           -np.sin(theta)*primary+np.cos(theta)*lateral)
        details["arc_parameters"] = {"width_ratio":width,"direction_deg":direction,
                                      "circle_ratios":circles.tolist(),"length_ratio":float(arc.get("length_ratio",1.0))}
    if template == "yaw_return":
        prefix = q0[:3, :3].T @ short.poses_world_camera[:qe_index + 1, :3, :3]
        heading = np.rad2deg(np.unwrap(np.arctan2(prefix[:, 0, 2], prefix[:, 2, 2])))
        if np.max(np.abs(heading)) > 300 + 1e-6:
            raise ValueError("source prefix exceeds Q0-relative 300 degree limit")
        start = float(heading[-1])
        direction = 1.0 if start >= 0 else -1.0
        seconds = (960 - qe_index) / short.fps
        budget = infer_short_motion(short).angular_speed_deg_s * seconds
        peak = direction * np.clip((budget + abs(start)) / 2.0, 180.0, 300.0)
        # A source already at the bound needs selection/re-authoring, not a zero excursion.
        if abs(peak - start) < 1.0:
            raise ValueError("source QE leaves no room for bounded yaw exploration")
        if abs(peak) <= abs(start) + 1.0:
            raise ValueError("source speed budget leaves no yaw exploration beyond QE")
        c, a, b = start + (peak - start) * np.asarray([.25, .5, .75])
        seconds = (960 - qe_index) / short.fps
        budget = infer_short_motion(short).angular_speed_deg_s * seconds
        base_travel = abs(peak - start) + abs(peak)
        loops = max(0, int(np.floor((budget - base_travel) / 120.0 + .5)))
        turns = [float(b), peak]
        for _ in range(loops):
            turns.extend([direction * 240.0, peak])
        angle_segments = [[start, c], [c, a], [a, b], turns,
                          [peak, b], [b, a], [a, c], [c, 0.0]]
        residual = Rotation.from_euler("y", -start, degrees=True).as_matrix() @ prefix[-1]
        result = []
        for number, nodes in enumerate(angle_segments):
            angles = np.concatenate([np.linspace(x, y, max(2, int(abs(y-x)*4)+1))[:-1]
                                     for x, y in zip(nodes, nodes[1:])] + [np.array(nodes[-1:])])
            part = np.repeat(qe[None], len(angles), axis=0)
            tilt = np.repeat(residual[None], len(angles), axis=0)
            if number == 7:
                alpha = np.linspace(0, 1, len(angles))
                tilt = Slerp([0, 1], Rotation.from_matrix([residual, np.eye(3)]))(alpha).as_matrix()
                part[:, :3, 3] = (1-alpha[:, None])*qe[:3, 3] + alpha[:, None]*q0[:3, 3]
            part[:, :3, :3] = q0[:3, :3] @ Rotation.from_euler("y", angles[:, None], degrees=True).as_matrix() @ tilt
            result.append(part)
        result[0][0], result[-1][-1] = qe, q0
        details.update(yaw_minimum_q0_deg=180.0, yaw_limit_q0_deg=300.0, yaw_peak_deg=float(peak), yaw_local_return_count=loops,
                       yaw_local_return_range_deg=[240.0, 300.0])
        return result, 0.0, details

    orbit_focus = None
    if template == "orbit_around_anchor":
        orbit_focus = np.asarray(short.parameters.get("orbit_focus_q0"),dtype=float)
        if orbit_focus.shape != (3,) or not np.isfinite(orbit_focus).all():
            raise ValueError("Orbit requires inherited orbit_focus_q0; no arbitrary center")
        orbit_radius = float(np.linalg.norm((qe[:3,3]-orbit_focus)[[0,2]]))
        if orbit_radius < 1e-6:
            raise ValueError("QE coincides with Orbit center")
        orbit_start = float(np.arctan2(qe[2,3]-orbit_focus[2],qe[0,3]-orbit_focus[0]))
        recent = qe[:3,3]-short.poses_world_camera[max(0,qe_index-4),:3,3]
        radial = (qe[:3,3]-orbit_focus)[[0,2]]/orbit_radius
        tangent = np.array([-radial[1],radial[0]])
        tangential = float(recent[[0,2]]@tangent)
        if abs(tangential)<1e-10:
            raise ValueError("QE motion has no identifiable orbit direction about bound anchor")
        orbit_sign = 1 if tangential>0 else -1
        radius_ratio = float(short.parameters.get('orbit_radius_ratio',1.))
        if not np.isfinite(radius_ratio) or not .8<=radius_ratio<=1.2:
            raise ValueError('Orbit radius ratio must be within [0.8,1.2]')
        radial_slope = orbit_radius*float(recent[[0,2]]@radial)/abs(tangential)
        def orbit_radii(theta, span):
            # A single monotone radius transition: no inward/outward bump.
            delta=orbit_radius*(radius_ratio-1)
            width=.5*span
            slope=0.
            if delta*radial_slope>0:
                width=min(width,2*abs(delta/radial_slope))
                slope=radial_slope*width/delta
            u=np.clip(theta/max(width,1e-12),0,1)
            blend=(-2*u**3+3*u**2)+slope*(u**3-2*u**2+u)
            return orbit_radius+delta*blend
        budget_span = np.rad2deg(max(0., target_length-_segment_distance(short.poses_world_camera[:qe_index+1])[-1])/(2*orbit_radius))
        requested_span = short.parameters.get("orbit_long_span_deg")
        orbit_span = float(np.clip(budget_span, 20., 300.) if requested_span is None else requested_span)
        if orbit_sign not in {-1,1} or not np.isfinite(orbit_span) or not 20 <= orbit_span <= 300:
            raise ValueError("invalid inherited Orbit direction or long span")
        def look_at(position):
            forward=_unit(orbit_focus-position,np.array([0.,0.,1.]))
            right=_unit(np.cross(np.array([0.,1.,0.]),forward),np.array([1.,0.,0.]))
            return np.stack([right,np.cross(forward,right),forward],axis=1)
        orbit_orientation_offset=look_at(qe[:3,3]).T @ qe[:3,:3]

    prefix_back = short.poses_world_camera[:qe_index + 1][::-1]
    prefix_length = _segment_distance(prefix_back)[-1]

    if template not in {'orbit_around_anchor','yaw_return','arc_return'} and direction_deg:
        primary,lateral = (np.cos(np.deg2rad(direction_deg))*primary+np.sin(np.deg2rad(direction_deg))*lateral,
                           -np.sin(np.deg2rad(direction_deg))*primary+np.cos(np.deg2rad(direction_deg))*lateral)
    peek_offset = np.zeros(3)
    peek_source_progress = 1.0
    if template == "peek_occlude_retract":
        xyz = short.poses_world_camera[:, :3, 3] - q0[:3, 3]
        transverse = xyz - (xyz @ primary)[:, None] * primary
        stop = short.key_indices.get("w2", len(xyz)//2)
        stop = max(qe_index + 1, stop)
        index = qe_index + int(np.argmax(np.linalg.norm(transverse[qe_index:stop], axis=1)))
        measured = float(np.linalg.norm(transverse[index]))
        reference = float(np.linalg.norm(qe[:3,3]-q0[:3,3]))
        minimum = .15 * reference
        direction = transverse[index]/measured if measured > 1e-6 else lateral
        peek_offset = direction * max(measured, minimum, 1e-3)
        peek_source_progress = max(float(np.max(xyz[qe_index:stop] @ primary))-reference, reference*.5, 1e-3)
        details.update(peek_source_lateral_units=measured,
                       peek_minimum_lateral_units=minimum,
                       peek_lateral_floor_applied=bool(measured < minimum),
                       peek_lateral_direction=direction.tolist())

    def make(scale):
        def route(t):
            t = np.asarray(t)
            part = np.repeat(qe[None], len(t), axis=0)
            if template == "orbit_around_anchor":
                angle = orbit_start + orbit_sign*np.deg2rad(orbit_span)*t
                radius=orbit_radii(np.deg2rad(orbit_span)*t,np.deg2rad(orbit_span))
                part[:,0,3]=orbit_focus[0]+radius*np.cos(angle)
                part[:,2,3]=orbit_focus[2]+radius*np.sin(angle)
                forward=orbit_focus-part[:,:3,3]
                forward/=np.linalg.norm(forward,axis=1)[:,None]
                right=np.cross(np.array([0.,1.,0.]),forward)
                right/=np.linalg.norm(right,axis=1)[:,None]
                offsets=orbit_orientation_offset
                if short.parameters.get('orbit_center_view_transition', False):
                    # Preserve QE exactly, then remove its off-axis viewing offset by C1.
                    blend=np.clip(t/.25,0,1);blend=blend*blend*(3-2*blend)
                    offsets=Slerp([0,1],Rotation.from_matrix([orbit_orientation_offset,np.eye(3)]))(blend).as_matrix()
                part[:,:3,:3]=np.stack([right,np.cross(forward,right),forward],axis=2)@offsets
            else:
                part[:, :3, 3] += scale*t[:, None]*primary
                if template == "peek_occlude_retract":
                    # Complete the sideways reveal early, then explore beyond it.
                    u = np.clip(t/.3, 0, 1)
                    ramp = u*u*(3-2*u)
                    offset = peek_offset * max(1.0, scale/peek_source_progress)
                    part[:, :3, 3] += ramp[:, None]*offset
            return part
        if template == "arc_return":
            # C/A/B are the contact points of consecutive circles. Outbound
            # and inbound use opposite semicircles; no endpoint detours.
            c = qe[:3,3] + .3*scale*primary
            radii = circles*scale
            bases = [c, c+2*radii[0]*primary, c+2*(radii[0]+radii[1])*primary]
            def half(number, returning=False):
                angle = np.linspace(np.pi if returning else 0,
                                    2*np.pi if returning else np.pi, 401)
                radius = radii[number]
                part = np.repeat(qe[None], len(angle), axis=0)
                part[:,:3,3] = bases[number] + radius*((1-np.cos(angle))[:,None]*primary
                                                      + width*np.sin(angle)[:,None]*lateral)
                # Set shared endpoints analytically to avoid trigonometric drift.
                lower, upper = bases[number], bases[number]+2*radius*primary
                part[0,:3,3],part[-1,:3,3] = (upper,lower) if returning else (lower,upper)
                return part
            stem = np.repeat(qe[None],201,axis=0)
            stem[:,:3,3] = qe[:3,3]+np.linspace(0,1,201)[:,None]*(c-qe[:3,3])
            return [stem,half(0),half(1),half(2),half(2,True),half(1,True),half(0,True),
                    np.concatenate([stem[::-1],prefix_back[1:]])]
        if template == "peek_occlude_retract":
            far = route([1.0])[0]
            near = np.array(q0, copy=True)
            near[:3, 3] = q0[:3, 3] + .1*(qe[:3, 3]-q0[:3, 3])
            out = [route(np.linspace(x, y, 201)) for x,y in [(0,.25),(.25,.5),(.5,1)]]
            alpha = np.linspace(0,1,401)
            retract = np.repeat(far[None], len(alpha), axis=0)
            retract[:, :3, 3] = (1-alpha[:, None])*far[:3,3]+alpha[:, None]*near[:3,3]
            retract[:, :3, :3] = Slerp([0,1],Rotation.from_matrix([far[:3,:3],near[:3,:3]]))(alpha).as_matrix()
            return out+[retract, retract[::-1].copy(), out[2][::-1].copy(), out[1][::-1].copy(),
                        np.concatenate([out[0][::-1], prefix_back[1:]])]
        out = [route(np.linspace(x,y,401)) for x,y in [(0,.25),(.25,.5),(.5,.8),(.8,1)]]
        return out+[out[3][::-1].copy(),out[2][::-1].copy(),out[1][::-1].copy(),
                    np.concatenate([out[0][::-1],prefix_back[1:]])]

    if template == "orbit_around_anchor" and requested_span is None:
        # Account for the bounded radius adjustment when budgeting angular travel.
        lower,upper=20.,300.
        for _ in range(24):
            orbit_span=(lower+upper)/2
            theta=np.linspace(0,np.deg2rad(orbit_span),2401)
            radii=orbit_radii(theta,theta[-1])
            points=np.stack([radii*np.cos(theta),radii*np.sin(theta)],axis=1)
            length=2*np.linalg.norm(np.diff(points,axis=0),axis=1).sum()+prefix_length
            if length>target_length:upper=orbit_span
            else:lower=orbit_span
        orbit_span=lower
    if template == "orbit_around_anchor" and sum(_segment_distance(part)[-1] for part in make(0.)) > target_length + 1e-6:
        raise ValueError("Orbit span exceeds inherited speed budget; reduce span rather than accelerate")
    amplitude = 0.0 if template == "orbit_around_anchor" else _solve_amplitude(target_length,
        lambda scale: sum(_segment_distance(part)[-1] for part in make(scale)),
        reference_distance=prefix_length)
    if template == "arc_return":
        details["arc_shape"] = "three_tangent_circles_complementary_return"
    if template == "orbit_around_anchor":
        details["orbit_radius_policy"] = "inherited_direction_bounded_0.8_1.2"
        details["orbit_peak_deg"] = orbit_span
        details["orbit_b_to_w_deg"] = .2*orbit_span
        details["orbit_focus_q0"] = orbit_focus.tolist()
        details["orbit_focus_provenance"] = short.parameters.get("orbit_focus_provenance",{})
        details["orbit_qe_radius"] = orbit_radius
        details["orbit_outer_radius"] = orbit_radius*radius_ratio
        details["orbit_radius_ratio"] = radius_ratio
        details["orbit_direction_sign"] = orbit_sign
    if template == "peek_occlude_retract":
        details["peek_return_q0_qe_fraction"] = .1
        details["peek_applied_lateral_units"] = float(np.linalg.norm(peek_offset)*max(1.0,amplitude/peek_source_progress))
    return make(amplitude), amplitude, details


def extend_short_trajectory_to_long(
    short: Trajectory,
    *,
    template_name: str | None = None,
    minimum_revisit_gap_s: float = 5.0,
    arc_parameters: Mapping | None = None,
) -> Trajectory:
    """Create one 961-pose route while preserving the admitted Q0-QE prefix."""

    template = str(template_name or short.template_name).strip()
    template = _ALIASES.get(template, template)
    if template not in LONG_TEMPLATE_NAMES:
        raise ValueError(f"unsupported long template: {template!r}")
    if short.fps != LONG_HORIZON.fps:
        raise ValueError("short and long profiles must use the same frame rate")
    qe_name = _qe_event(short.key_indices)
    qe_index = int(short.key_indices[qe_name])
    if qe_index <= 0 or qe_index >= len(short.poses_world_camera):
        raise ValueError("registered QE must follow Q0 inside the short trajectory")
    if qe_index >= LONG_HORIZON.frame_count - 1:
        raise ValueError("registered QE leaves no long exploration interval")
    if not np.isfinite(minimum_revisit_gap_s) or minimum_revisit_gap_s < 5.0:
        raise ValueError("long revisit gap must be finite and at least five seconds")

    if template == "peek_occlude_retract" and short.parameters.get("peek_midpoint_qe", False):
        from src.trajectory.long_peek import extend_midpoint_peek
        return extend_midpoint_peek(short)

    q0 = np.array(short.poses_world_camera[0], copy=True)
    qe = np.array(short.poses_world_camera[qe_index], copy=True)
    motion = infer_short_motion(short)
    remaining_s = (LONG_HORIZON.frame_count - 1 - qe_index) / float(short.fps)
    reference_distance = float(np.linalg.norm(qe[:3, 3] - q0[:3, 3]))
    # Static/yaw-only source routes have no meaningful translational median.
    # Preserve their scale through Q0-QE displacement, with a small coordinate-
    # system-local fallback so the generated revisit pairs still have excursions.
    inherited_linear_speed = motion.linear_speed
    if inherited_linear_speed <= 0.0:
        inherited_linear_speed = reference_distance / max(qe_index / short.fps, 1.0)
    if inherited_linear_speed <= 0.0:
        inherited_linear_speed = 0.25
    target_length = reference_distance if template == "yaw_return" else inherited_linear_speed * remaining_s

    length_ratio = float((arc_parameters or {}).get("length_ratio",1.0))
    if not np.isfinite(length_ratio) or not .75 <= length_ratio <= 1.0:
        raise ValueError("arc length_ratio must be in [.75,1]")
    speed_ratio=float(short.parameters.get("long_speed_ratio",1.))
    if not np.isfinite(speed_ratio) or not .75 <= speed_ratio <= 1.25:
        raise ValueError("long speed ratio must be in [.75,1.25]")
    target_length *= length_ratio*speed_ratio
    segments, amplitude, details = _long_segments(short, qe_index, template, target_length, arc_parameters)
    weights = np.asarray([_segment_distance(part, angular=template == "yaw_return")[-1]
                          for part in segments])
    total_frames = LONG_HORIZON.frame_count - 1 - qe_index
    segment_frames = _allocate_segment_frames(weights, total_frames, short.fps)
    indices = qe_index + np.cumsum(segment_frames)
    key_indices = {"q0": 0, qe_name: qe_index}
    key_indices.update({name: int(index) for name, index in zip(_EVENTS_AFTER_QE, indices)})
    poses = np.repeat(np.eye(4)[None], LONG_HORIZON.frame_count, axis=0)
    poses[:qe_index + 1] = short.poses_world_camera[:qe_index + 1]
    left = qe_index
    for part, count in zip(segments, segment_frames):
        distance = _segment_distance(part, angular=template == "yaw_return")
        # Remove stationary source samples before Slerp; keep exact endpoints below.
        keep = np.r_[True, np.diff(distance) > 1e-12]
        if keep.sum() < 2:
            raise ValueError("long segment has no motion")
        knots, samples = distance[keep], part[keep]
        t = np.clip(np.linspace(0, distance[-1], int(count) + 1), knots[0], knots[-1])
        for axis in range(3):
            poses[left:left + count + 1, axis, 3] = np.interp(t, knots, samples[:, axis, 3])
        poses[left:left + count + 1, :3, :3] = Slerp(
            knots, Rotation.from_matrix(samples[:, :3, :3]))(t).as_matrix()
        poses[left], poses[left + count] = part[0], part[-1]
        left += count
    poses[:qe_index + 1] = short.poses_world_camera[:qe_index + 1]
    if template == 'orbit_around_anchor':
        # Last segment joins the reverse arc to the preserved source prefix.
        # Record the last sample still on the arc using its travelled distance.
        final_count=int(segment_frames[-1])
        arc_distance = _segment_distance(segments[-1][:401])[-1]
        final_distance=_segment_distance(segments[-1])[-1]
        details['orbit_return_qe_index']=int(indices[-2]+np.floor(final_count*arc_distance/final_distance))
    if template == 'orbit_around_anchor' and short.parameters.get('orbit_center_view_transition', False):
        # Spread the C2-to-Q0 return rotation over the available return time.
        c2=key_indices['qc_revisit']
        u=np.linspace(0,1,len(poses)-c2);blend=u*u*(3-2*u)
        poses[c2:,:3,:3]=Slerp([0,1],Rotation.from_matrix([poses[c2,:3,:3],q0[:3,:3]]))(blend).as_matrix()
    heading=float(short.parameters.get('long_heading_deg',0.))
    if template in {'arc_return','peek_occlude_retract'} and heading != 0:
        raise ValueError('Arc/Peek preserve source heading without extra heading adjustment')
    heading_limit = 20 if template in {'forward_backward','lateral_out_and_back'} else 60
    if not np.isfinite(heading) or abs(heading)>heading_limit:
        raise ValueError(f'long heading adjustment must be within {heading_limit} degrees')
    if heading:
        if template in {'yaw_return','orbit_around_anchor'}:
            raise ValueError('heading search is not allowed for Yaw/Orbit')
        ramp=np.zeros(len(poses));c1=key_indices['qc_first'];c2=key_indices['qc_revisit']
        u=np.linspace(0,1,c1-qe_index+1);ramp[qe_index:c1+1]=u*u*(3-2*u)
        ramp[c1:c2+1]=1
        end=max(c2+1,len(poses)-1-qe_index)
        u=np.linspace(0,1,end-c2+1);ramp[c2:end+1]=1-u*u*(3-2*u)
        poses[:,:3,:3]=poses[:,:3,:3]@Rotation.from_euler('y',heading*ramp,degrees=True).as_matrix()
    for first, second in (("qc_first", "qc_revisit"), ("qa_first", "qa_revisit"),
                          ("qb_first", "qb_revisit")):
        poses[key_indices[second]] = poses[key_indices[first]]
    poses[-1] = q0
    if template == "yaw_return":
        relative = np.einsum("ij,njk->nik", q0[:3, :3].T, poses[:, :3, :3])
        yaw = np.rad2deg(np.unwrap(np.arctan2(relative[:, 0, 2], relative[:, 2, 2])))
        if np.max(np.abs(yaw - yaw[0])) > 300.0 + 1e-6:
            raise ValueError("source prefix or long yaw exceeds Q0-relative 300 degree limit")
    for pair in LONG_REVISIT_PAIRS:
        first = key_indices[pair.first_event]
        revisit = key_indices[pair.revisit_event]
        if (revisit - first) / short.fps < minimum_revisit_gap_s:
            raise ValueError(f"revisit pair {pair.name} is closer than the requested gap")
        validate_revisit_excursion(poses, first, revisit)

    parameters = dict(short.parameters)
    parameters.update({
        "horizon_profile": "long",
        "long_extension_contract": "orbia.long_six_template.v1",
        "trajectory_revision": ("gourd_lateral_v3" if template in {"arc_return", "peek_occlude_retract"}
                                else "evaluation_orbit_focus_v5" if template == "orbit_around_anchor" else "bounded_double_arc_v2"),
        **details,
        "source_short_template": short.template_name,
        "source_qe_event": qe_name,
        "source_qe_frame_index": qe_index,
        "preserved_prefix_frame_count": qe_index + 1,
        "inherited_linear_speed_units_per_s": float(motion.linear_speed),
        "inherited_angular_speed_deg_s": float(motion.angular_speed_deg_s),
        "target_post_qe_translation_length_units": float(target_length),
        "exploration_amplitude_units": float(amplitude),

        "minimum_revisit_gap_s": float(minimum_revisit_gap_s),
        "revisit_pairs": [
            {"name": pair.name, "first_event": pair.first_event, "revisit_event": pair.revisit_event}
            for pair in LONG_REVISIT_PAIRS
        ],
        "collision_validation": "required_downstream",
    })
    return Trajectory(
        template_name=template,
        difficulty=short.difficulty,
        duration_s=LONG_HORIZON.duration_s,
        fps=LONG_HORIZON.fps,
        poses_world_camera=poses,
        key_indices=key_indices,
        parameters=parameters,
    )

