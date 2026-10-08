"""Plan Unreal captures and export rendered ground truth into the ORBIA format."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

# OpenCV right/down/forward -> Unreal forward/right/up.
CV_TO_UE = np.array([[0., 0., 1.], [1., 0., 0.], [0., -1., 0.]])


def quaternion_matrix(value):
    q = np.asarray(value, dtype=np.float64)
    if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) == 0:
        raise ValueError('quaternion must be finite nonzero XYZW')
    x, y, z, w = q / np.linalg.norm(q)
    return np.array([[1 - 2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1 - 2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1 - 2*(x*x+y*y)]])


def matrix_quaternion(rotation):
    m = np.asarray(rotation, dtype=np.float64)
    # The symmetric eigenproblem remains well-conditioned at a 180-degree turn.
    k = np.array([[m[0,0]-m[1,1]-m[2,2], m[0,1]+m[1,0], m[0,2]+m[2,0], m[2,1]-m[1,2]],
                  [m[0,1]+m[1,0], m[1,1]-m[0,0]-m[2,2], m[1,2]+m[2,1], m[0,2]-m[2,0]],
                  [m[0,2]+m[2,0], m[1,2]+m[2,1], m[2,2]-m[0,0]-m[1,1], m[1,0]-m[0,1]],
                  [m[2,1]-m[1,2], m[0,2]-m[2,0], m[1,0]-m[0,1], np.trace(m)]]) / 3.
    _, vectors = np.linalg.eigh(k)
    q = vectors[:, -1]
    return q if q[3] >= 0 else -q


def ue_pose(frame):
    pose = np.eye(4)
    pose[:3, :3] = quaternion_matrix(frame['quaternion_xyzw'])
    pose[:3, 3] = np.asarray(frame['location_cm'], dtype=np.float64)
    return pose


def opencv_to_ue(poses, origin):
    """Convert Q0-relative metric C2W poses to UE actor poses in centimetres."""
    axis = np.eye(4)
    axis[:3, :3] = CV_TO_UE
    poses = np.asarray(poses, dtype=np.float64).copy()
    poses[:, :3, 3] *= 100.
    return np.asarray(origin) @ axis @ poses @ np.linalg.inv(axis)


def ue_to_opencv(poses):
    """Express measured UE actor poses in the first camera's OpenCV frame."""
    poses = np.asarray(poses, dtype=np.float64)
    axis = np.eye(4)
    axis[:3, :3] = CV_TO_UE
    result = np.linalg.inv(axis) @ np.linalg.inv(poses[0]) @ poses @ axis
    result[:, :3, 3] /= 100.
    return result


def camera_intrinsics(frame):
    width, height = map(int, frame['resolution_px'])
    focal = float(frame['focal_length_mm'])
    return np.array([[focal / float(frame['sensor_width_mm']) * width, 0., width/2.],
                     [0., focal / float(frame['sensor_height_mm']) * height, height/2.],
                     [0., 0., 1.]])


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')


def plan(config):
    """Write short/long UE trajectory specs with the shared template generators."""
    from src.trajectory.routes import generate, extend

    from src.construction.pipeline import local_work
    output = local_work(config['work_dir'])
    camera = config['camera']
    origin = ue_pose(config['origin_ue'])
    # A metric world with X=UE right, Y=UE forward, Z=UE up has proper
    # OpenCV camera rotations and retains the level's true vertical axis.
    metric_to_ue = np.array([[0., 1., 0.], [1., 0., 0.], [0., 0., 1.]])
    q0 = np.eye(4)
    q0[:3, :3] = metric_to_ue.T @ origin[:3, :3] @ CV_TO_UE
    q0[:3, 3] = metric_to_ue.T @ origin[:3, 3] / 100.
    entries = []
    for case in config['cases']:
        qe = q0 @ np.asarray(case['qe_c2w'], dtype=np.float64)
        parameters = dict(case.get('parameters', {}))
        if case['template'] == 'arc_return':
            parameters.setdefault('curve_profile', 'spatialvid_pose_conditioned_piecewise_arcs_natural_revisit')
            parameters.setdefault('lateral_axis_policy', 'q0_camera_right_exact')
        focus = case.get('focus_q0_m') if case['template'] == 'orbit_around_anchor' else None
        focus = None if focus is None else q0[:3, :3] @ np.asarray(focus) + q0[:3, 3]
        short = generate(case['template'], q0, qe, parameters=parameters, focus=focus)
        variants = [(case['case_id'], short, None)]
        if case.get('long_case_id'):
            variants.append((case['long_case_id'], extend(short), case['case_id']))
        for case_id, trajectory, parent in variants:
            poses = np.linalg.inv(q0) @ trajectory.poses_world_camera
            world_poses = opencv_to_ue(poses, origin)
            frames = [{'frame_index': i, 'timestamp_s': i / trajectory.fps,
                       'pose_c2w_opencv_q0_m': pose.tolist(),
                       'ue_world': {'location_cm': world[:3, 3].tolist(),
                                    'quaternion_xyzw': matrix_quaternion(world[:3, :3]).tolist()}}
                      for i, (pose, world) in enumerate(zip(poses, world_poses))]
            events = {name: {'frame_index': int(i), 'timestamp_s': int(i)/trajectory.fps}
                      for name, i in trajectory.key_indices.items()}
            payload = {'case_id': case_id, 'parent_case_id': parent, 'template': case['template'],
                       'frame_count': len(poses), 'fps': trajectory.fps, 'camera': camera,
                       'scene_prompt': case['caption'], 'events': events,
                       'parameters': dict(trajectory.parameters), 'frames': frames}
            path = output / case_id / 'trajectory_spec.json'
            _write_json(path, payload)
            entries.append({'case_id': case_id, 'template': case['template'],
                            'scene': config.get('map', ''), 'trajectory': str(path),
                            'capture': str(output / case_id),
                            'anchor_actor_labels': [a.get('actor_label', '') for a in case['anchors']],
                            'anchors': case['anchors']})
    index = output / 'plan_index.json'
    _write_json(index, {'cases': entries})
    return index


def _curate(capture, spec, events, poses, source, measured, anchors, curation, parent_capture):
    """Choose the query and revisit events on a rendered path (see :mod:`src.construction.revisits`)."""
    from src.construction.revisits import CaptureObserver, evaluate_revisit_pairs, select_query, select_revisit_pair

    def depth(i):
        return np.load(capture / 'depth' / f'{source[i]:06d}.npy', allow_pickle=False)

    def intrinsics(i):
        return camera_intrinsics(measured[source[i]])

    primary = list(map(int, anchors[0]['instance_ids']))

    def anchor_mask(i):
        return np.isin(np.asarray(Image.open(capture / 'instance_mask' / f'{source[i]:06d}.png')), primary)

    record = {'source_frame_count': len(measured), 'planned_events': dict(events)}
    if parent_capture is not None and (Path(parent_capture) / 'curation.json').is_file():
        parent = json.loads((Path(parent_capture) / 'curation.json').read_text())
        events['qe1'] = int(parent['events']['qe1'])
        record['query'] = {'inherited_from': str(parent_capture), 'frame_index': events['qe1']}
    elif curation.get('query', 'planned') == 'search':
        result = select_query(len(poses), anchor_mask, depth, intrinsics, poses,
                              search_end=int(curation.get('query_search_end', 170)))
        if result['selected'] is not None:
            events['qe1'] = int(result['selected']['frame_index'])
        record['query'] = result
    observer = CaptureObserver(depth, intrinsics, poses, voxel=float(curation.get('voxel_size', 0.10)))
    if curation.get('revisit', 'planned') == 'search' and parent_capture is None:
        result = select_revisit_pair(poses, events, spec['template'], observer, fps=float(spec['fps']))
        if result['selected'] is not None:
            events.update(result['selected'])
        record['revisit'] = result
    else:
        record['revisit'] = evaluate_revisit_pairs(poses, events, observer)
    # Trajectories require unique, increasing event indices; auxiliary waypoints yield on collision.
    essential = {'q0', 'qe1', 'q0_return'} | {k for k in events if k.endswith(('_first', '_revisit'))}
    taken = {events[k] for k in essential}
    ordered = sorted((i, k) for k, i in events.items() if k in essential or events[k] not in taken)
    events.clear()
    events.update({k: int(i) for i, k in ordered})
    record['events'] = dict(events)
    _write_json(capture / 'curation.json', record)
    return record


def convert(capture, anchors, output, *, case_id=None, parent_case_id=None, curation=None,
            parent_capture=None):
    """Export a synchronized capture with active instance masks and measured cameras."""
    from src.construction.sources import RegisteredFrame
    from src.construction.pipeline import export_case, local_work
    from src.trajectory.templates import Trajectory
    from src.construction.revisits import canonical_to_source

    capture, output = Path(capture).resolve(), local_work(output)
    spec = json.loads((capture / 'trajectory_spec.json').read_text())
    payload = json.loads((capture / 'camera_frames.json').read_text())
    measured = payload['frames'] if isinstance(payload, dict) else payload
    count, fps = int(spec['frame_count']), int(spec['fps'])
    if [f['frame_index'] for f in measured] != list(range(len(measured))):
        raise ValueError('camera frame indices must be consecutive')
    if len(measured) != count and not curation:
        raise ValueError('capture and trajectory clocks disagree; pass curation to resample the capture')
    source = [canonical_to_source(i, len(measured), count) for i in range(count)]
    poses = ue_to_opencv(np.stack([ue_pose(measured[i]) for i in source]))
    events = {name: int(value['frame_index']) for name, value in spec['events'].items()}
    for anchor in anchors:
        if not list(anchor.get('instance_ids', [])):
            raise ValueError(f"{anchor.get('anchor_id', anchor.get('label'))}: instance_ids is empty")
    curation_record = (_curate(capture, spec, events, poses, source, measured, anchors, curation, parent_capture)
                       if curation else None)
    trajectory = Trajectory(spec['template'], 'rendered', count/fps, fps, poses, events,
                            dict(spec.get('parameters', {})))
    q0, qe = events['q0'], events['qe1']
    frames = []
    for index in (q0, qe):
        frame = measured[source[index]]
        rgb = capture / 'rgb' / f'{source[index]:06d}.png'
        depth = capture / 'depth' / f'{source[index]:06d}.npy'
        if not rgb.is_file() or not depth.is_file():
            raise FileNotFoundError(f'missing synchronized frame {source[index]}: {capture}')
        depth_array = np.load(depth, allow_pickle=False)
        rgb_array = np.asarray(Image.open(rgb).convert('RGB'))
        frames.append(RegisteredFrame(frame_id=str(index), rgb=rgb_array, depth=depth_array,
                      valid=np.isfinite(depth_array) & (depth_array > 0), pose_c2w=poses[index],
                      K=camera_intrinsics(frame), depth_K=camera_intrinsics(frame),
                      depth_unit='metre', timestamp_s=index/fps))
    work = capture / 'anchor_masks'
    work.mkdir(exist_ok=True)
    endpoint_instances = {index: np.asarray(Image.open(capture / 'instance_mask' / f'{source[index]:06d}.png'))
                          for index in (q0, qe)}
    selected = []
    for number, anchor in enumerate(anchors, 1):
        anchor_id = str(anchor.get('anchor_id', f'A{number}'))
        ids = list(map(int, anchor['instance_ids']))
        if not ids:
            raise ValueError(f'{anchor_id}: instance_ids is empty')
        item = {'anchor_id': anchor_id, 'label': anchor['label']}
        masks = {name: np.isin(endpoint_instances[index], ids) for name, index in (('q0', q0), ('qe', qe))}
        missing = [name for name, mask in masks.items() if not mask.any()]
        if missing and curation and number > 1:
            curation_record.setdefault('dropped_anchors', []).append({'anchor_id': anchor_id, 'absent_at': missing})
            continue
        if missing:
            raise ValueError(f'{anchor_id}: selected instance is absent at {missing[0]}')
        for name, mask in masks.items():
            Image.fromarray(mask.astype(np.uint8)*255).save(work / f'{anchor_id}_{name}.png')
            item[name] = mask
        selected.append(item)
    record = export_case(output, case_id or spec['case_id'], 'Unreal Engine', frames, 0, 1,
                         selected, spec['scene_prompt'], trajectory,
                         parent_case_id=parent_case_id or spec.get('parent_case_id'))
    if curation_record is not None:
        _write_json(capture / 'curation.json', curation_record)
    manifest_path = output / 'manifest.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {'cases': []}
    manifest['cases'].append(record)
    _write_json(manifest_path, manifest)
    return record


def _configured_capture(config, case_id):
    cases = config['cases']
    if case_id is None:
        if len(cases) != 1:
            raise ValueError('--case-id is required when the configuration contains several cases')
        case_id = cases[0]['case_id']
    for case in cases:
        if case_id == case['case_id'] or case_id == case.get('long_case_id'):
            return Path(config['work_dir']).expanduser().resolve()/case_id, case, case_id
    raise ValueError(f'case {case_id!r} is not listed in the configuration')


def _configured_anchors(capture, anchors):
    """Resolve actor names separately for each capture's Cryptomatte ID map."""
    instances = None
    result = []
    for anchor in anchors:
        item = dict(anchor)
        if not item.get('instance_ids'):
            if instances is None:
                instances = json.loads((Path(capture)/'instances.json').read_text())['instances']
            paths = set(item.get('actor_paths', []))
            if item.get('actor_path'):
                paths.add(item['actor_path'])
            labels = set(item.get('actor_labels', []))
            if item.get('actor_label'):
                labels.add(item['actor_label'])
            ids = [int(i) for i, value in instances.items()
                   if value.get('actor_path') in paths or value.get('actor_label') in labels]
            if not ids:
                raise ValueError(f"{item.get('anchor_id', item['label'])}: no selected actor in {capture}/instances.json")
            item['instance_ids'] = ids
        result.append(item)
    return result


def _load_config(path):
    path = Path(path).expanduser().resolve()
    config = json.loads(path.read_text())
    for key in ('work_dir', 'output_dir'):
        if key in config:
            value = Path(config[key]).expanduser()
            config[key] = str(value if value.is_absolute() else path.parent/value)
    config['_config_dir'] = str(path.parent)
    return config


def extract_config(config, *, case_id=None):
    from src.ue.extract_ground_truth import main as extract
    from src.construction.pipeline import local_work
    capture, _, _ = _configured_capture(config, case_id)
    local_work(capture)
    settings = config['extraction']
    arguments = ['--input', str(capture/'multilayer_exr'), '--capture', str(capture),
                 '--camera-frames', str(capture/'camera_frames.json'),
                 '--trajectory', str(capture/'trajectory_spec.json'),
                 '--world-depth-definition', settings['world_depth_definition']]
    for key in ('world_unit_to_m', 'max_depth_m', 'rgb_transfer'):
        if key in settings:
            arguments += ['--'+key.replace('_', '-'), str(settings[key])]
    return extract(arguments)


def convert_config(config, *, case_id=None):
    capture, case, case_id = _configured_capture(config, case_id)
    is_long = case_id == case.get('long_case_id')
    anchor_file = case.get('long_anchor_file') if is_long else None
    anchor_file = anchor_file or case.get('anchor_file')
    if anchor_file:
        path = Path(anchor_file).expanduser()
        if not path.is_absolute():
            path = Path(config.get('_config_dir', '.'))/path
        anchors = json.loads(path.read_text())
    else:
        anchors = case.get('long_anchors', case['anchors']) if is_long else case['anchors']
    anchors = _configured_anchors(capture, anchors)
    curation = dict(config.get('curation', {}), **case.get('curation', {}))
    return convert(capture, anchors, config['output_dir'], case_id=case_id,
                   parent_case_id=case['case_id'] if is_long else None, curation=curation or None,
                   parent_capture=capture.parent / case['case_id'] if is_long else None)


def main(argv=None):
    import sys
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == 'extract' and not any(v == '--config' or v.startswith('--config=') for v in argv):
        from src.ue.extract_ground_truth import main as extract
        return extract(argv[1:])
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='stage', required=True)
    planner = sub.add_parser('plan')
    planner.add_argument('--config', type=Path, required=True)
    extractor = sub.add_parser('extract', help='extract a configured synchronized MRQ capture')
    extractor.add_argument('--config', type=Path, required=True)
    extractor.add_argument('--case-id', help='short or long case ID from the configuration')
    converter = sub.add_parser('convert')
    converter.add_argument('--config', type=Path)
    converter.add_argument('--capture', type=Path)
    converter.add_argument('--anchors', type=Path, help='JSON list of {anchor_id,label,instance_ids}')
    converter.add_argument('--output', type=Path)
    converter.add_argument('--case-id')
    converter.add_argument('--parent-case-id')
    args = parser.parse_args(argv)
    if args.stage == 'plan':
        print(plan(_load_config(args.config)))
    elif args.stage == 'extract':
        return extract_config(_load_config(args.config), case_id=args.case_id)
    elif args.config:
        print(convert_config(_load_config(args.config), case_id=args.case_id))
    else:
        if not all((args.capture, args.anchors, args.output)):
            parser.error('convert requires --config, or --capture together with --anchors and --output')
        print(convert(args.capture, json.loads(args.anchors.read_text()), args.output,
                      case_id=args.case_id, parent_case_id=args.parent_case_id))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
