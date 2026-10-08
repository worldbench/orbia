"""Resolve case-specific reference visibility; optionally render registered meshes."""
from pathlib import Path
import json
import numpy as np
from src.metrics.stages import atomic_json, file_identity


def prepare_reference_visibility(case, info, options, job, cache_root):
    from src.metrics.anchor_tracking import tracking_windows
    from src.metrics.runner import _interpolate_poses
    windows = tracking_windows(case, info, options.anchor_tracking_frame_stride)
    indices = sorted(set(sum(windows.values(), [])))
    roots = []
    if job.get('reference_visibility'):
        roots.append(Path(job['reference_visibility']))
    if options.anchor_tracking_visibility:
        roots.extend([options.anchor_tracking_visibility/case.case_id, options.anchor_tracking_visibility])
    roots.append(Path(job['case_root'])/'evaluator_hidden'/'reference_visibility')
    K = np.diag([info.width/case.canonical_camera.width, info.height/case.canonical_camera.height, 1.]) @ case.canonical_camera.K
    poses = _interpolate_poses(case.timestamps_s, case.requested_poses, np.asarray(indices)/info.fps)
    def valid(root):
        try:
            meta = json.loads((root/'reference_visibility.json').read_text())
            if meta.get('case_id') != case.case_id:
                return False
            for i, pose in zip(indices, poses):
                with np.load(root/f'frame_{i:06d}.npz', allow_pickle=False) as a:
                    if (a['depth_m'].shape != (info.height, info.width)
                        or not np.allclose(a['K'], K) or not np.allclose(a['pose_c2w'], pose)):
                        return False
            return True
        except (OSError, ValueError, KeyError):
            return False
    # A mesh must be explicitly bound to the hidden Q0 world coordinates.
    spec = job.get('reference_geometry')
    registration = Path(job['case_root'])/'evaluator_hidden'/'reference_geometry.json'
    if spec is None and registration.is_file():
        spec = json.loads(registration.read_text())
        if spec.get('mesh') and not Path(spec['mesh']).is_absolute():
            spec = dict(spec, mesh=str(registration.parent/spec['mesh']))
    identity = None
    if spec:
        if spec.get('coordinate_binding') != 'hidden q0 T_world_camera':
            raise ValueError('reference mesh requires hidden q0 T_world_camera coordinate binding')
        identity = {'mesh': file_identity(spec['mesh']),
                    'reference_pose': case.q0.camera_document['T_world_camera'],
                    'case_id': case.case_id, 'frame_indices': indices,
                    'K': K.tolist(), 'poses': poses.tolist(), 'format': 'reference-mesh-v2'}
    destination = Path(cache_root)/case.case_id
    roots.append(destination)
    for root in roots:
        if not valid(root):
            continue
        if identity is not None:
            meta = json.loads((root/'reference_visibility.json').read_text())
            if meta.get('identity') != identity:
                continue
        return root, {'status': 'ready', 'reused': True, 'root': str(root), 'frame_count': len(indices)}
    if identity is None:
        return None, {'status': 'not_available', 'reason': 'no matching reference visibility or registered reference mesh; unknown intervals remain unscored'}
    from src.utils.geometry import MeshRaycaster
    from src.utils.camera import CameraIntrinsics
    destination.mkdir(parents=True, exist_ok=True)
    # Serialize competing models sharing the same reference cache.
    import fcntl
    with (destination/'.render.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if valid(destination) and json.loads((destination/'reference_visibility.json').read_text()).get('identity') == identity:
            return destination, {'status': 'ready', 'reused': True, 'root': str(destination), 'frame_count': len(indices)}
        (destination/'reference_visibility.json').unlink(missing_ok=True)
        raycaster = MeshRaycaster(Path(spec['mesh']))
        camera = CameraIntrinsics(width=info.width, height=info.height, K=K)
        transform = np.asarray(case.q0.camera_document['T_world_camera']) @ np.linalg.inv(case.requested_poses[case.events['q0']])
        for i, pose in zip(indices, poses):
            render = raycaster.render(camera, transform @ pose)
            target = destination/f'frame_{i:06d}.npz'
            temp = target.with_suffix('.tmp.npz')
            np.savez_compressed(temp, depth_m=np.where(render.valid, render.depth_m, np.nan).astype(np.float32), pose_c2w=pose, K=K)
            temp.replace(target)
        atomic_json(destination/'reference_visibility.json', {'case_id': case.case_id, 'identity': identity, 'windows': windows, 'generated_geometry_used': False})
    return destination, {'status': 'ready', 'reused': False, 'root': str(destination), 'frame_count': len(indices)}
