"""Six trajectory templates and their one-minute extensions."""
from dataclasses import asdict
import inspect
import numpy as np
from src.trajectory import templates as core
from src.trajectory.long_templates import extend_short_trajectory_to_long

TEMPLATES = ("forward_backward", "lateral_out_and_back", "yaw_return", "arc_return",
             "orbit_around_anchor", "peek_occlude_retract")


def generate(template, q0, qe, *, parameters=None, focus=None, scale_free=False):
    """Return a 357-frame, 16-FPS trajectory bound to the supplied endpoint poses."""
    if template not in TEMPLATES:
        raise ValueError(f"unknown template {template!r}; choose from {TEMPLATES}")
    q0, qe = np.asarray(q0, dtype=float), np.asarray(qe, dtype=float)
    kwargs = dict(parameters or {})
    extension_keys = {"long_direction_deg", "long_heading_deg", "long_speed_ratio",
                      "orbit_radius_ratio", "orbit_long_span_deg", "orbit_center_view_transition"}
    extension_parameters = {k: kwargs.pop(k) for k in extension_keys if k in kwargs}
    kwargs.update(evidence_frame_id="qe", canonical_sampling=True)
    if scale_free:
        kwargs.setdefault("dynamics_limits", asdict(core.TrajectoryDynamicsLimits(
            maximum_linear_speed_m_s=1e9, maximum_angular_speed_deg_s=120,
            maximum_linear_acceleration_m_s2=1e9,
            maximum_angular_acceleration_deg_s2=720,
            maximum_linear_jerk_m_s3=1e9, maximum_angular_jerk_deg_s3=10000)))
        if template == "yaw_return":
            kwargs.setdefault("yaw_axis_policy", "q0_camera_y")
        if template in {"arc_return", "peek_occlude_retract"}:
            kwargs.setdefault("lateral_axis_policy", "q0_camera_right_exact" if template == "arc_return" else "q0_camera_right")
        if template == "arc_return":
            kwargs.setdefault("curve_profile", "spatialvid_pose_conditioned_piecewise_arcs_natural_revisit")
    generator = getattr(core, "generate_" + template)
    unknown = set(kwargs) - set(inspect.signature(generator).parameters)
    if unknown:
        raise ValueError(f"unsupported {template} parameters: {sorted(unknown)}")
    if template == "orbit_around_anchor":
        if focus is None:
            raise ValueError("orbit requires a focus point from an anchor's valid depth")
        focus = np.asarray(focus, dtype=float)
        if scale_free:
            origin = q0[:3, 3]
            basis = np.stack((q0[:3, 0], q0[:3, 2], -q0[:3, 1]), axis=1)
            gauge = np.eye(4); gauge[:3, :3] = basis; gauge[:3, 3] = origin
            radii = [np.linalg.norm((basis.T @ (pose[:3, 3]-focus))[:2]) for pose in (q0, qe)]
            kwargs.setdefault("minimum_radius_m", max(1e-4, .5*min(radii)))
            kwargs.setdefault("maximum_radius_m", max(2e-4, 1.5*max(radii)))
            route = generator(np.linalg.inv(gauge) @ q0,
                              np.linalg.inv(gauge) @ qe,
                              basis.T @ (focus-origin), **kwargs)
            route = core.Trajectory(route.template_name, route.difficulty, route.duration_s,
                route.fps, gauge @ route.poses_world_camera, route.key_indices,
                {**route.parameters, "focus_world_m": focus.tolist(),
                 "orbit_basis_world": basis.tolist(),
                 "orbit_origin_world": origin.tolist(),
                 "evidence_pose_world_camera": qe.tolist(),
                 "orbit_frame_policy": "q0_camera_right_forward_up"})
        else:
            route = generator(q0, qe, focus, **kwargs)
    else:
        route = generator(q0, qe, **kwargs)
    if template == "orbit_around_anchor":
        focus_q0 = np.linalg.inv(q0) @ np.r_[focus, 1.]
        route = core.Trajectory(route.template_name, route.difficulty, route.duration_s,
            route.fps, route.poses_world_camera, route.key_indices,
            {**route.parameters, "focus_world_m": np.asarray(focus).tolist(),
             "orbit_focus_q0": focus_q0[:3].tolist(),
             "orbit_focus_provenance": {"binding_basis": "selected_anchor_valid_depth_median"}})
    if template == "peek_occlude_retract":
        qi = route.key_indices["qe1"]
        distance = np.linalg.norm(route.poses_world_camera[:qi+1, :3, 3]-q0[:3, 3], axis=1)
        route = core.Trajectory(route.template_name, route.difficulty, route.duration_s,
            route.fps, route.poses_world_camera, route.key_indices,
            {**route.parameters, "peek_midpoint_qe": bool(distance.max() > 2*max(distance[-1], 1e-6))})
    if extension_parameters:
        route = core.Trajectory(route.template_name, route.difficulty, route.duration_s,
            route.fps, route.poses_world_camera, route.key_indices,
            {**route.parameters, **extension_parameters})
    if len(route.poses_world_camera) != 357 or route.fps != 16:
        raise ValueError("short trajectories must use the 357/16 clock")
    if not np.allclose(route.poses_world_camera[route.key_indices["qe1"]], qe, atol=1e-8):
        raise ValueError("route changed the registered query pose")
    return route


def extend(short, *, minimum_revisit_gap_s=5, arc_parameters=None):
    """Author the 961-frame, 16-FPS extension in Q0 coordinates, then restore world poses."""
    q0 = short.poses_world_camera[0]
    local = core.Trajectory(short.template_name, short.difficulty, short.duration_s,
        short.fps, np.linalg.inv(q0) @ short.poses_world_camera,
        short.key_indices, short.parameters)
    route = extend_short_trajectory_to_long(local,
        minimum_revisit_gap_s=minimum_revisit_gap_s, arc_parameters=arc_parameters)
    return core.Trajectory(route.template_name, route.difficulty, route.duration_s,
        route.fps, q0 @ route.poses_world_camera, route.key_indices, route.parameters)
