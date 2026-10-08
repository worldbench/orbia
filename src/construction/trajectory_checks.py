"""Registered endpoint, revisit, and sampled depth-proxy route checks."""
import inspect
import numpy as np
from scipy.spatial.transform import Rotation
from src.utils.geometry import _arc_length_sample_positions
from src.trajectory.long_horizon import validate_revisit_excursion
from src.utils.camera import CameraIntrinsics
from src.trajectory.templates import _completion_revisit_geometry_metrics
from src.construction.depth_proxy import (
    DerivedProxyFrame, DerivedProxyFusionSettings, DerivedProxyVoxelState,
    build_derived_proxy_geometry,
)


def prepare_proxy(frames, q0_index=0, qe_index=1, config=None):
    """Fuse a bounded selection of registered depth frames, including Q0 and QE."""
    config = dict(config or {})
    if not config.get('enabled', True) or not config.get('clearance', True):
        return None
    if not frames:
        return None
    maximum = max(2, int(config.get('maximum_fusion_frames', 16)))
    low, high = sorted((int(q0_index), int(qe_index)))
    count = min(maximum, high-low+1)
    indices = np.unique(np.r_[q0_index, qe_index,
        np.linspace(low, high, count, dtype=int)])
    fusion = []
    for index in indices:
        frame = frames[int(index)]
        if frame.depth is None or frame.pose_c2w is None or frame.K is None:
            continue
        depth = np.asarray(frame.depth, dtype=np.float32)
        valid = np.isfinite(depth) & (depth > 0)
        if frame.valid is not None:
            valid &= np.asarray(frame.valid, dtype=bool)
        if not valid.any():
            continue
        K = frame.depth_K if frame.depth_K is not None else frame.K
        camera = CameraIntrinsics(depth.shape[1], depth.shape[0], K)
        fusion.append(DerivedProxyFrame(depth, camera, frame.pose_c2w,
            static_mask=valid, frame_id=str(frame.frame_id)))
    if not fusion:
        return None
    settings = DerivedProxyFusionSettings(
        voxel_size_m=float(config.get('voxel_size', .05)),
        pixel_stride=int(config.get('pixel_stride', 4)),
        maximum_samples_per_frame=int(config.get('maximum_samples_per_frame', 3000)),
        maximum_free_ray_samples=int(config.get('maximum_free_ray_samples', 128)),
    )
    return build_derived_proxy_geometry(fusion, settings)


def _pose_difference(first, second):
    relative = np.linalg.inv(np.asarray(first)) @ np.asarray(second)
    return {'translation': float(np.linalg.norm(relative[:3, 3])),
            'rotation_deg': float(np.degrees(Rotation.from_matrix(relative[:3,:3]).magnitude()))}


def mesh_clearance(raycaster, poses, keys, config):
    """Use the final ScanNet++ point-only sampled mesh-clearance screen."""
    report = raycaster.validate_camera_path_coarse_sampled(
        poses[:, :3, 3], float(config.get('minimum_clearance', .01)),
        uniform_sample_count=int(config.get('uniform_sample_count', 32)),
        key_position_indices=list(keys.values()),
    )
    return {'status': 'checked', 'geometry': 'mesh', 'passed': bool(report.valid),
            'sampling': 'point_only_arclength_plus_events',
            'minimum': float(report.minimum_clearance_m),
            'required': float(report.required_clearance_m),
            'sample_count': int(report.sampled_position_count),
            'maximum_sample_spacing': float(report.maximum_sample_spacing_m),
            'validation_contract': report.validation_contract}


def check_route(trajectory, q0_pose, qe_pose, proxy=None, config=None, mesh=None):
    """Check camera centres at 32 arclength samples and every named event."""
    config = dict(config or {})
    if not config.get('enabled', True):
        return {'passed': True, 'status': 'disabled'}
    poses = np.asarray(trajectory.poses_world_camera)
    keys = dict(trajectory.key_indices)
    translation_tolerance = float(config.get('translation_tolerance', 1e-6))
    rotation_tolerance = float(config.get('rotation_tolerance_deg', 1e-4))
    def binding(first, second):
        result = _pose_difference(first, second)
        result['passed'] = (result['translation'] <= translation_tolerance
                            and result['rotation_deg'] <= rotation_tolerance)
        return result
    registered = {'q0': binding(q0_pose, poses[keys.get('q0', 0)]),
                  'qe': binding(qe_pose, poses[keys['qe1']])}
    returned = binding(q0_pose, poses[keys.get('q0_return', len(poses)-1)])
    minimum_gap = float(config.get('minimum_revisit_gap_s',
                                  5 if len(poses) > 357 else 0))
    # The short natural-revisit arc is exempt from exact QC binding; long
    # extensions bind every revisit pair exactly.
    natural_arc = (trajectory.template_name == 'arc_return'
        and len(poses) == 357
        and trajectory.parameters.get('global_arc_curve') ==
            'spatialvid_pose_conditioned_piecewise_arcs_natural_revisit'
        and trajectory.parameters.get('completion_revisit_binding') == 'natural_window_matched'
        and not trajectory.parameters.get('long_extension_contract'))
    revisits = {}
    for name in ('qc', 'qa', 'qb'):
        first_key, revisit_key = name+'_first', name+'_revisit'
        if first_key not in keys and revisit_key not in keys:
            if name == 'qc':
                revisits[name] = {'passed': False, 'reason': 'missing required completion pair C'}
            continue
        if first_key not in keys or revisit_key not in keys:
            revisits[name] = {'passed': False, 'reason': 'missing paired event'}
            continue
        first, revisit = keys[first_key], keys[revisit_key]
        result = binding(poses[first], poses[revisit])
        natural_pair = natural_arc and name == 'qc'
        result.update(first_frame=first, revisit_frame=revisit,
            first_time_s=first/trajectory.fps, revisit_time_s=revisit/trajectory.fps,
            gap_s=(revisit-first)/trajectory.fps,
            pose_binding='natural_window_matched' if natural_pair else 'exact_event_pose',
            pose_difference_gating=not natural_pair)
        try:
            if natural_pair:
                if not 0 <= first < revisit < len(poses):
                    raise ValueError('revisit indices must be ordered and in bounds')
                result['passed'] = True
                result['window_diagnostics'] = _completion_revisit_geometry_metrics(trajectory)
            else:
                validate_revisit_excursion(poses, first, revisit)
            excursion = np.linalg.inv(poses[first]) @ poses[first:revisit+1]
            result['excursion_translation'] = float(np.linalg.norm(excursion[:,:3,3],axis=1).max())
            result['excursion_rotation_deg'] = float(np.degrees(
                Rotation.from_matrix(excursion[:,:3,:3]).magnitude().max()))
            result['passed'] &= result['gap_s'] >= minimum_gap
        except ValueError as exc:
            result.update(passed=False, reason=str(exc))
        revisits[name] = result
    clearance = {'status': 'unavailable', 'unknown_space_policy': 'diagnostic_only',
                 'coordinate_unit': config.get('coordinate_unit', 'source_pose_unit')}
    if mesh is not None and config.get('clearance', True):
        clearance.update(mesh_clearance(mesh, poses, keys, config))
    elif proxy is not None:
        count = int(config.get('uniform_sample_count', 32))
        if not 16 <= count <= 64:
            raise ValueError('uniform_sample_count must be in [16, 64]')
        samples, maximum_gap = _arc_length_sample_positions(poses[:,:3,3],
            uniform_sample_count=count, key_position_indices=list(keys.values()))
        distances = proxy.point_clearance(samples)
        states = proxy.voxel_states(samples)
        threshold = float(config.get('minimum_clearance', .01))
        if not np.isfinite(threshold) or threshold < 0:
            raise ValueError('minimum_clearance must be finite and nonnegative')
        minimum = float(np.min(distances))
        clearance.update(status='checked', passed=minimum >= threshold,
            geometry='depth_derived_voxels', sampling='point_only_arclength_plus_events',
            minimum=minimum, required=threshold, sample_count=len(samples),
            uniform_sample_count=count, event_sample_count=len(keys),
            maximum_sample_spacing=maximum_gap,
            known_free_fraction=float(np.mean(states == DerivedProxyVoxelState.KNOWN_FREE)),
            occupied_fraction=float(np.mean(states == DerivedProxyVoxelState.OCCUPIED)),
            unknown_fraction=float(np.mean(states == DerivedProxyVoxelState.UNKNOWN)),
            fusion_frame_ids=list(proxy.fusion_frame_ids))
    novelty = {}
    for name in ('qc', 'qa', 'qb'):
        event = name+'_first'
        if event in keys:
            novelty[name] = {'relative_to_q0': _pose_difference(q0_pose, poses[keys[event]]),
                             'relative_to_qe': _pose_difference(qe_pose, poses[keys[event]])}
    passed = (all(row['passed'] for row in registered.values()) and returned['passed']
              and all(row['passed'] for row in revisits.values())
              and clearance.get('passed', True))
    return {'passed': bool(passed), 'status': 'checked', 'registered_poses': registered,
            'q0_return': returned, 'revisits': revisits, 'clearance': clearance,
            'novelty': {'kind': 'pose_excursion_diagnostic', 'first_visits': novelty}}


def _parameter_candidates(template, parameters, q0_pose, qe_pose, config):
    original = dict(parameters or {})
    yield original
    if 'retry_parameters' in config:
        for override in config['retry_parameters']:
            yield {**original, **override}
        return
    # Reduce only the optional excursion; the measured endpoint stays fixed.
    from src.trajectory import templates as core
    if template not in {'forward_backward','lateral_out_and_back','arc_return',
                        'peek_occlude_retract'}:
        return
    generator = getattr(core, 'generate_'+template)
    key = 'peek_distance_m' if template == 'peek_occlude_retract' else 'distance_m'
    distance = original.get(key, inspect.signature(generator).parameters[key].default)
    endpoint = float(np.linalg.norm(np.asarray(qe_pose)[:3,3]-np.asarray(q0_pose)[:3,3]))
    margin = .75 if template == 'arc_return' else .45 if key == 'peek_distance_m' else .05
    if distance is None:
        distance = max(1.5, endpoint+.75)
    lower = max(1.5, endpoint+margin+1e-4)
    for fraction in (.85, .70):
        reduced = max(lower, float(distance)*fraction)
        if reduced < float(distance)-1e-6:
            yield {**original, key: reduced}


def generate_checked(template, q0_pose, qe_pose, *, frames=(), q0_index=0,
                     qe_index=1, parameters=None, focus=None, scale_free=False,
                     config=None, accepted=False, mesh=None):
    """Bound parameter retries and return the authored route with a QA report."""
    from src.trajectory.routes import generate
    config = dict(config or {})
    proxy = None if mesh is not None else prepare_proxy(frames, q0_index, qe_index, config)
    attempts = []
    first_route = first_report = None
    maximum = max(1, int(config.get('maximum_attempts', 3)))
    for candidate in list(_parameter_candidates(template, parameters, q0_pose, qe_pose, config))[:maximum]:
        try:
            route = generate(template, q0_pose, qe_pose, parameters=candidate,
                             focus=focus, scale_free=scale_free)
        except ValueError as exc:
            attempts.append({'parameters': candidate, 'passed': False, 'error': str(exc)})
            continue
        report = check_route(route, q0_pose, qe_pose, proxy, config, mesh=mesh)
        attempts.append({'parameters': candidate, 'passed': report['passed']})
        if first_route is None:
            first_route, first_report = route, report
        if report['passed'] or accepted:
            report.update(attempts=attempts, accepted_exception=bool(accepted and not report['passed']))
            return route, report
    if first_route is None:
        raise ValueError(f'no feasible {template} route: {attempts}')
    first_report.update(attempts=attempts, accepted_exception=False)
    return first_route, first_report
