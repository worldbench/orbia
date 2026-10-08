"""Construct registered exploration-and-revisit cases on a local work disk."""
from dataclasses import replace
from functools import lru_cache
from pathlib import Path
import argparse
import html
import json
import numpy as np
from PIL import Image

from src.construction.sources import path, read_source
from src.trajectory.routes import extend


def write_json(target, value):
    target = Path(target); target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, indent=2, ensure_ascii=False,
        default=lambda v: v.tolist() if isinstance(v, np.ndarray) else v.item()) + '\n')


def local_work(pathname):
    p = Path(pathname).expanduser().resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p


def validate_frames(frames, *, allow_missing_cameras=False):
    if len(frames) < 2 or len({f.frame_id for f in frames}) != len(frames):
        raise ValueError('a source needs at least two distinct registered observations')
    for f in frames:
        if f.rgb.ndim != 3 or f.rgb.shape[2] != 3 or f.rgb.dtype != np.uint8:
            raise ValueError(f'{f.frame_id}: expected uint8 RGB')
        if allow_missing_cameras and (f.K is None or f.pose_c2w is None):
            continue
        if f.K is None or f.K.shape != (3, 3) or not np.isfinite(f.K).all() or min(f.K[0, 0], f.K[1, 1]) <= 0:
            raise ValueError(f'{f.frame_id}: invalid RGB intrinsics')
        if f.pose_c2w is None or f.pose_c2w.shape != (4, 4) or not np.isfinite(f.pose_c2w).all():
            raise ValueError(f'{f.frame_id}: invalid c2w pose')
        R = f.pose_c2w[:3, :3]
        if not np.allclose(R.T @ R, np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(R), 1, atol=1e-5):
            raise ValueError(f'{f.frame_id}: pose rotation is not rigid')
        if f.depth is not None and (f.depth.ndim != 2 or f.valid.shape != f.depth.shape):
            raise ValueError(f'{f.frame_id}: depth and valid grid mismatch')


def vipe_geometry(frames, config, work):
    """Run ViPE on the source video (or reuse its output) and read the selected frames."""
    import subprocess
    from src.metrics.reconstruction import load_vipe_depths
    output = Path(config['output']) if config.get('output') else work/'vipe'
    if not (output/'pose').is_dir():
        subprocess.run([config.get('command', 'vipe'), 'infer', str(config['video']),
                        '--output', str(output)], check=True)
    def archive(name):
        files = sorted((output/name).glob('*.npz'))
        if len(files) != 1:
            raise ValueError(f'expected one ViPE {name}/*.npz in {output}')
        data = np.load(files[0], allow_pickle=False)
        inds = data['inds'] if 'inds' in data else np.arange(len(data['data']))
        return dict(zip((int(i) for i in inds), data['data']))
    poses, intrinsics = archive('pose'), archive('intrinsics')
    depths = load_vipe_depths(output/'depth')
    result = []
    for f in frames:
        i = int(f.frame_id)
        if i not in poses or i not in depths:
            raise ValueError(f'ViPE output has no frame {i}')
        fx, fy, cx, cy = intrinsics[i]
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.]])
        d = np.asarray(depths[i], dtype=np.float32)
        h, w = f.rgb.shape[:2]
        result.append(replace(f, depth=d, valid=np.isfinite(d) & (d > 0), pose_c2w=np.asarray(poses[i], float),
            K=K, depth_K=np.diag([d.shape[1]/w, d.shape[0]/h, 1]) @ K, depth_unit='source_pose_unit'))
    validate_frames(result)
    return result


def geometry(frames, config, work):
    """Reuse provided geometry, or estimate cameras and depth with DA3 or ViPE."""
    mode = config.get('mode', 'provided')
    if mode == 'provided':
        validate_frames(frames)
        if any(f.depth is None for f in frames):
            raise ValueError('provided geometry requires depth for every registered frame')
        return frames
    if mode == 'vipe':
        return vipe_geometry(frames, config, work)
    if mode != 'da3':
        raise ValueError('geometry mode must be provided, da3 or vipe')
    identity = {'frame_ids': [f.frame_id for f in frames], 'checkpoint': config['checkpoint'],
                'process_res': config.get('process_res')}
    cache = work/'geometry'; cache.mkdir(exist_ok=True)
    metadata = cache/'frames.json'
    if metadata.exists() and not config.get('overwrite', False):
        previous = json.loads(metadata.read_text())
        if previous.get('identity') != identity:
            raise ValueError('geometry cache belongs to different frames or checkpoint; use overwrite')
        result = []
        for i, f in enumerate(frames):
            depth = np.load(cache/f'{i:06d}.npy', allow_pickle=False)
            result.append(replace(f, depth=depth, valid=np.isfinite(depth) & (depth > 0),
                pose_c2w=np.asarray(previous['poses_c2w'][i]), K=np.asarray(previous['intrinsics'][i]),
                depth_K=np.asarray(previous['intrinsics'][i]), depth_unit=previous['depth_unit']))
        validate_frames(result)
        return result
    from depth_anything_3.api import DepthAnything3
    import torch
    if len({f.rgb.shape for f in frames}) != 1:
        raise ValueError('joint DA3 inference requires one RGB size')
    paths = []
    for i, f in enumerate(frames):
        target = cache/f'{i:06d}.png'; Image.fromarray(f.rgb).save(target); paths.append(str(target))
    model = DepthAnything3.from_pretrained(config['checkpoint']).to(config.get('device', 'cuda')).eval()
    h, w = frames[0].rgb.shape[:2]
    resolution = int(config.get('process_res', np.ceil(max(h, w)/14)*14))
    arguments = dict(image=paths, process_res=resolution, process_res_method='upper_bound_resize', infer_gs=False)
    with torch.inference_mode():
        prediction = model.inference(**arguments)
    depth = np.asarray(prediction.depth, dtype=np.float32)
    predicted_extr = np.asarray(prediction.extrinsics)
    if depth.ndim != 3 or len(depth) != len(frames) or len(predicted_extr) != len(frames):
        raise ValueError('DA3 depth and camera outputs do not match the registered frames')
    w2c = np.repeat(np.eye(4)[None], len(frames), axis=0)
    w2c[:, :3, :] = predicted_extr[:, :3, :]
    poses = np.linalg.inv(w2c)
    intrinsics = np.asarray(prediction.intrinsics).copy()
    intrinsics[:, 0] *= w/depth.shape[2]
    intrinsics[:, 1] *= h/depth.shape[1]
    depth_unit = 'DA3_estimated_unit'
    import cv2
    result = []
    for i, f in enumerate(frames):
        d = cv2.resize(depth[i], (w, h), interpolation=cv2.INTER_NEAREST)
        np.save(cache/f'{i:06d}.npy', d)
        result.append(replace(f, depth=d, valid=np.isfinite(d) & (d > 0), pose_c2w=poses[i],
                              K=intrinsics[i], depth_K=intrinsics[i], depth_unit=depth_unit))
    validate_frames(result)
    write_json(metadata, {'identity': identity, 'poses_c2w': poses.tolist(), 'intrinsics': intrinsics.tolist(),
        'depth_unit': depth_unit, 'scale_policy': 'one joint camera/depth prediction in DA3 estimated units'})
    return result


def _grid(h, w):
    y, x = np.mgrid[:h, :w]
    return np.stack((x.ravel()+.5, y.ravel()+.5, np.ones(h*w)))


def _sample(array, coords):
    """Sample integer sensor measurements at pixel-centre calibrated coordinates."""
    x = np.floor(np.where(np.isfinite(coords[0]), coords[0], 0)).astype(np.int64)
    y = np.floor(np.where(np.isfinite(coords[1]), coords[1], 0)).astype(np.int64)
    good = np.isfinite(coords).all(axis=0) & (x >= 0) & (y >= 0) & (x < array.shape[1]) & (y < array.shape[0])
    result = np.zeros((coords.shape[1],) + array.shape[2:], dtype=array.dtype)
    result[good] = array[y[good], x[good]]
    return result, good


def dense_projection(q0, qe, *, rgb_grid=True, relative_tolerance=.10, absolute_tolerance=.02):
    """Find QE pixels whose surfaces are visible from Q0, with dense residuals."""
    shape = qe.rgb.shape[:2] if rgb_grid else qe.depth.shape
    K = qe.K if rgb_grid else qe.depth_K
    pixels = _grid(*shape)
    depth_coords = qe.depth_K @ np.linalg.inv(K) @ pixels
    depth_coords /= depth_coords[2:3]
    depths, sample_valid = _sample(qe.depth, depth_coords)
    valid, _ = _sample(qe.valid, depth_coords)
    points = (np.linalg.inv(K) @ pixels) * depths
    transform = np.linalg.inv(q0.pose_c2w) @ qe.pose_c2w
    points0 = transform[:3, :3] @ points + transform[:3, 3:4]
    projected = q0.depth_K @ points0
    projected /= np.where(projected[2:3] > 0, projected[2:3], np.nan)
    source_depth, source_inside = _sample(q0.depth, projected)
    source_valid, _ = _sample(q0.valid, projected)
    candidate = sample_valid & valid & source_inside & source_valid & (points0[2] > 0)
    residual = np.full(len(depths), np.nan)
    residual[candidate] = np.abs(points0[2, candidate]-source_depth[candidate]) / source_depth[candidate]
    accepted = candidate & (np.abs(points0[2]-source_depth) <=
        np.maximum(absolute_tolerance, relative_tolerance*source_depth))
    source_rgb_coords = q0.K @ points0
    source_rgb_coords /= np.where(source_rgb_coords[2:3] > 0, source_rgb_coords[2:3], np.nan)
    projected_rgb, rgb_inside = _sample(q0.rgb, source_rgb_coords)
    target_rgb, _ = _sample(qe.rgb, qe.K @ np.linalg.inv(K) @ pixels)
    colour = np.full(len(depths), np.nan)
    colour[candidate & rgb_inside] = np.mean(np.abs(projected_rgb[candidate & rgb_inside].astype(float)-
        target_rgb[candidate & rgb_inside].astype(float)), axis=1) / 255
    return {'support': accepted.reshape(shape), 'candidate': candidate.reshape(shape),
            'depth_absrel': residual.reshape(shape), 'rgb_mae': colour.reshape(shape),
            'projected_rgb': projected_rgb.reshape(shape+(3,)),
            'q0_coords': source_rgb_coords.reshape(3, *shape)}


def anchor_diagnostics(q0, qe, anchors, options=None):
    opts = options or {}
    projection = dense_projection(q0, qe,
        relative_tolerance=opts.get('relative_tolerance', .10),
        absolute_tolerance=opts.get('absolute_tolerance', .02))
    reverse = dense_projection(qe, q0,
        relative_tolerance=opts.get('relative_tolerance', .10),
        absolute_tolerance=opts.get('absolute_tolerance', .02))
    projection['reverse'] = reverse
    rows = []
    for anchor in anchors:
        q0_mask, qe_mask = anchor['q0'], anchor['qe']
        coords = projection['q0_coords'].reshape(3, -1)
        q0_membership, _ = _sample(q0_mask, coords)
        valid = qe_mask & projection['support'] & q0_membership.reshape(qe_mask.shape)
        comparable = qe_mask & projection['candidate'] & q0_membership.reshape(qe_mask.shape)
        d = projection['depth_absrel'][comparable]
        rgb = projection['rgb_mae'][valid]
        coverage = float(valid.sum()/max(1, qe_mask.sum()))
        depth_error = float(np.nanmedian(d)) if np.any(np.isfinite(d)) else None
        rgb_error = float(np.nanmedian(rgb)) if np.any(np.isfinite(rgb)) else None
        qe_membership, _ = _sample(qe_mask, reverse['q0_coords'].reshape(3, -1))
        reverse_member = qe_membership.reshape(q0_mask.shape)
        reverse_valid = q0_mask & reverse['support'] & reverse_member
        reverse_comparable = q0_mask & reverse['candidate'] & reverse_member
        rd, rr = reverse['depth_absrel'][reverse_comparable], reverse['rgb_mae'][reverse_valid]
        row = {'anchor_id': anchor['anchor_id'], 'label': anchor.get('label', ''),
            'q0_supported_fraction': float(reverse_valid.sum()/max(1, q0_mask.sum())),
            'reverse_dense_depth_absrel': float(np.nanmedian(rd)) if np.any(np.isfinite(rd)) else None,
            'reverse_dense_rgb_mae': float(np.nanmedian(rr)) if np.any(np.isfinite(rr)) else None,
            'q0_image_fraction': float(q0_mask.mean()), 'qe_anchor_pixels': int(qe_mask.sum()),
            'evaluation_pixels': int(valid.sum()), 'qe_supported_fraction': coverage,
            'dense_depth_absrel': depth_error, 'dense_rgb_mae': rgb_error,
            'passed': coverage >= opts.get('minimum_coverage', .10)
                and depth_error is not None and depth_error <= opts.get('maximum_depth_absrel', .10)
                and rgb_error is not None and rgb_error <= opts.get('maximum_rgb_mae', .20),
            'evaluation_mask': valid}
        rows.append(row)
    return projection, rows


def select_pairs(frames, selection):
    """Rank endpoint pairs by registered viewpoint change and dense visible support."""
    from scipy.spatial.transform import Rotation
    from src.construction.pose_selection import evaluate_pose_pair
    template = selection.get('template', 'lateral_out_and_back')
    stride = int(selection.get('stride', 1))
    gates = selection.get('pose_gates', {})
    gate = gates.get(template)
    prefix = selection.get('case_id_prefix', 'candidate')
    candidates = []
    for i in range(0, len(frames), stride):
        if i > selection.get('maximum_q0_index', len(frames)):
            break
        for j in range(i+1, len(frames), stride):
            if j-i < selection.get('minimum_frame_gap', 1):
                continue
            q0, qe = frames[i], frames[j]
            gap = abs(qe.timestamp_s-q0.timestamp_s)
            if gap < selection.get('minimum_source_gap_s', 0) or gap > selection.get('maximum_source_gap_s', float('inf')):
                continue
            pose_quality = evaluate_pose_pair(q0.pose_c2w, qe.pose_c2w, template, gate) if gate else None
            if pose_quality and not pose_quality['passed']:
                continue
            relative = np.linalg.inv(q0.pose_c2w) @ qe.pose_c2w
            distance = float(np.linalg.norm(relative[:3, 3]))
            angle = float(np.degrees(Rotation.from_matrix(relative[:3, :3]).magnitude()))
            if not selection.get('minimum_translation', 0) <= distance <= selection.get('maximum_translation', float('inf')):
                continue
            if not selection.get('minimum_rotation_deg', 0) <= angle <= selection.get('maximum_rotation_deg', 180):
                continue
            p = dense_projection(q0, qe, rgb_grid=False)
            support = float(p['support'].mean())
            if support < selection.get('minimum_support', .05):
                continue
            candidates.append({'case_id': f'{prefix}_{i:04d}_{j:04d}', 'q0': q0.frame_id,
                'qe': qe.frame_id, 'template': template, 'parameters': selection.get('parameters', {}),
                'viewpoint_translation': distance, 'viewpoint_rotation_deg': angle,
                'dense_support_fraction': support, 'pose_gate': pose_quality, 'accepted': False})
    # Prefer greater viewpoint change while retaining a substantial input-visible surface.
    candidates.sort(key=lambda r: r['dense_support_fraction'] * (r['viewpoint_translation'] +
        selection.get('rotation_weight', .02)*r['viewpoint_rotation_deg']), reverse=True)
    candidates = candidates[:selection.get('maximum_candidates', 20)]
    if selection.get('include_long', False):
        for case in candidates:
            case['long_case_id'] = case['case_id']+'_long'
    return candidates


def freeze_rank_slate(work, cases, selection, source_name):
    """Keep ranked pair identities fixed while human decisions remain editable."""
    records = []
    rank_by_template = {}
    for case in cases:
        template = case['template']
        rank_by_template[template] = rank_by_template.get(template, 0)+1
        case.setdefault('source_rank', rank_by_template[template])
        records.append({'case_id': case['case_id'], 'q0': str(case['q0']), 'qe': str(case['qe']),
                        'template': template, 'source_rank': case['source_rank']})
    slate = {'schema': 'orbia.construction.rank_slate.v1', 'source': source_name,
             'selection': selection, 'ranked_pairs': records}
    target = work/'rank_slate.json'
    if target.exists():
        if json.loads(target.read_text()) != slate:
            raise ValueError('ranked endpoint slate is immutable; use another work_dir for a new selection')
    else:
        write_json(target, slate)


class CaseRejected(ValueError):
    """An annotated candidate failed a recorded construction gate."""


@lru_cache(maxsize=1)
def _vlm_provider(model_path):
    from src.construction.vlm import LocalQwenProvider
    return LocalQwenProvider(Path(model_path))


@lru_cache(maxsize=1)
def _sam_predictor(checkpoint, repository, threshold):
    if repository:
        import sys
        if repository not in sys.path:
            sys.path.insert(0, repository)
    from sam3.model_builder import build_sam3_multiplex_video_predictor
    return build_sam3_multiplex_video_predictor(checkpoint_path=checkpoint,
        max_num_objects=16, use_fa3=False, compile=False, warm_up=False,
        default_output_prob_thresh=threshold, async_loading_frames=False)


def _vlm_proposals(images, frame_ids, config, cache):
    from dataclasses import asdict
    from src.construction.anchor_prompt import (
        VLM_SHORT_ANCHOR_PROMPT, parse_sam31_prompt_review_response, SCHEMA_VERSION,
    )
    target = cache/'scene_proposals.json'
    identity = {'model': config['model'], 'frame_ids': frame_ids, 'schema': SCHEMA_VERSION}
    if target.exists():
        record = json.loads(target.read_text())
        if record['identity'] != identity:
            raise ValueError('scene proposal cache belongs to another model or endpoint interval')
        response = record['raw_response']
    else:
        response = _vlm_provider(config['model']).generate(images,
            ['Q0', 'INTERMEDIATE_1', 'INTERMEDIATE_2', 'INTERMEDIATE_3', 'QE'],
            max_output_tokens=config.get('max_tokens', 4096), instruction=VLM_SHORT_ANCHOR_PROMPT).text
        record = {'identity': identity, 'raw_response': response}
    review = parse_sam31_prompt_review_response(response)
    write_json(target, {**record, 'review': asdict(review)})
    if not review.pair_usable:
        raise CaseRejected('scene gate: '+', '.join(review.pair_reject_reasons))
    return review.candidates


def scene_review(case, frames, config, base, work):
    """Five-frame VLM scene review (shot changes, people, weather, overlays)."""
    from dataclasses import asdict
    from src.construction.scene_review import SCENE_REVIEW_PROMPT, parse_scene_review
    ids = [f.frame_id for f in frames]
    q0i, qei = ids.index(str(case['q0'])), ids.index(str(case['qe']))
    indices = np.rint(np.linspace(q0i, qei, 5)).astype(int)
    model = str(path(config['model'], base))
    target = work/'annotations'/case['case_id']/'scene_review.json'
    identity = {'model': model, 'frame_ids': [ids[i] for i in indices]}
    record = json.loads(target.read_text()) if target.exists() else None
    if record is None or record['identity'] != identity:
        response = _vlm_provider(model).generate([Image.fromarray(frames[i].rgb) for i in indices],
            ['Q0', '25%', 'MIDDLE', '75%', 'QE'], max_output_tokens=config.get('max_tokens', 4096),
            instruction=SCENE_REVIEW_PROMPT).text
        record = {'identity': identity, 'raw_response': response}
    review = parse_scene_review(record['raw_response'])
    write_json(target, {**record, 'review': asdict(review)})
    return review


def _endpoint_pairer(q0, qe, visibility, minimum_support=.10):
    """Bind independently segmented capture objects using both dense projections."""
    opts = visibility or {}
    arguments = dict(relative_tolerance=opts.get('relative_tolerance', .10),
                     absolute_tolerance=opts.get('absolute_tolerance', .02))
    forwards = dense_projection(q0, qe, **arguments)
    backwards = dense_projection(qe, q0, **arguments)
    def match(q0_masks, qe_masks, seed_index):
        rows = []
        for j, target in enumerate(qe_masks):
            member, _ = _sample(q0_masks[seed_index], forwards['q0_coords'].reshape(3, -1))
            qe_fraction = float((target & forwards['support'] & member.reshape(target.shape)).sum()/max(1, target.sum()))
            member, _ = _sample(target, backwards['q0_coords'].reshape(3, -1))
            q0_fraction = float((q0_masks[seed_index] & backwards['support'] &
                member.reshape(q0_masks[seed_index].shape)).sum()/max(1, q0_masks[seed_index].sum()))
            if min(q0_fraction, qe_fraction) >= minimum_support:
                rows.append({'q0_independent_index': seed_index, 'qe_independent_index': j,
                    'q0_dense_identity_fraction': q0_fraction, 'qe_dense_identity_fraction': qe_fraction})
        rows.sort(key=lambda r: min(r['q0_dense_identity_fraction'], r['qe_dense_identity_fraction']), reverse=True)
        return rows[:1]
    return match


@lru_cache(maxsize=2)
def _semantic_mesh(scene_root):
    from src.construction.scannetpp import SemanticMesh
    return SemanticMesh(scene_root)


@lru_cache(maxsize=2)
def _raycaster(mesh_path):
    from src.utils.geometry import MeshRaycaster
    return MeshRaycaster(mesh_path)


def annotate(case, frames, model_config, base, work, annotation_options=None, visibility=None,
             semantic_mesh=None):
    by_id = {f.frame_id: f for f in frames}
    q0, qe = by_id[str(case['q0'])], by_id[str(case['qe'])]
    q0i, qei = next(i for i,f in enumerate(frames) if f is q0), next(i for i,f in enumerate(frames) if f is qe)
    indices = np.rint(np.linspace(q0i, qei, 5)).astype(int)
    images = [Image.fromarray(frames[i].rgb) for i in indices]
    cache = work/'annotations'/case['case_id']; cache.mkdir(parents=True, exist_ok=True)
    masks = []
    if case.get('anchors') and all('q0_mask' in a and 'qe_mask' in a for a in case['anchors']):
        for a in case['anchors']:
            masks.append({**a, 'q0': np.asarray(Image.open(path(a['q0_mask'], base)).convert('L')) > 0,
                          'qe': np.asarray(Image.open(path(a['qe_mask'], base)).convert('L')) > 0})
    elif semantic_mesh is not None:
        from src.construction.scannetpp import mesh_anchor_candidates
        options = dict((annotation_options or {}).get('mesh', {}))
        masks, rejected = mesh_anchor_candidates(semantic_mesh(), q0, qe, **options)
        write_json(cache/'anchor_rejections.json', rejected)
        if not masks:
            raise CaseRejected('no mesh instance passed the configured anchor checks')
    else:
        from src.construction.anchor_prompt import candidate_from_mapping
        from src.construction.annotation import ground_and_track
        if case.get('anchors'):
            proposals = [candidate_from_mapping(a) for a in case['anchors']]
        else:
            vlm = dict(model_config['vlm']); vlm['model'] = str(path(vlm['model'], base))
            proposals = _vlm_proposals(images, [frames[i].frame_id for i in indices], vlm, cache)
        sam = dict(model_config['sam3'])
        predictor = _sam_predictor(str(path(sam['checkpoint'], base)),
            str(path(sam['repository'], base)) if sam.get('repository') else '', float(sam.get('threshold', .35)))
        step = 1 if qei >= q0i else -1
        tracking_images = [Image.fromarray(frames[i].rgb) for i in range(q0i, qei+step, step)]
        opts = {**(annotation_options or {}), 'threshold': sam.get('threshold', .35)}
        masks, rejected = ground_and_track(predictor, tracking_images, proposals, opts,
            _endpoint_pairer(q0, qe, visibility, opts.get('minimum_endpoint_identity_support', .10))
                if not opts.get('tracking', True) else None)
        write_json(cache/'anchor_rejections.json', rejected)
        if not masks:
            raise CaseRejected('no physical anchor passed independent endpoint and identity checks')
    if len({a['anchor_id'] for a in masks}) != len(masks):
        raise ValueError('anchor IDs must be unique')
    for a in masks:
        if a['q0'].shape != q0.rgb.shape[:2] or a['qe'].shape != qe.rgb.shape[:2]:
            raise ValueError('anchor mask size differs from its RGB image')
        if not a['q0'].any() or not a['qe'].any():
            raise ValueError('an anchor must be visible at both endpoints')
    caption = case.get('caption')
    if not caption:
        from src.construction.caption import INSTRUCTION, FRAME_LABELS, _caption
        model_path = str(path(model_config['vlm']['model'], base))
        result = _vlm_provider(model_path).generate(images, FRAME_LABELS,
            max_output_tokens=1024, instruction=INSTRUCTION)
        caption = _caption(result.text)
    return masks, str(caption)


def _intrinsics(K, width, height):
    return {'K': np.asarray(K).tolist(), 'width': int(width), 'height': int(height)}


def export_case(root, case_id, source_name, frames, q0_index, qe_index, anchors,
                caption, trajectory, parent_case_id=None, visibility=None):
    """Export public inputs and evaluator references in the dataset format."""
    if Path(case_id).name != case_id or case_id in {'.', '..'}:
        raise ValueError('case ID must be one directory name')
    root = Path(root); public = root/'public'/case_id; ref = root/'references'/case_id
    if public.exists() or ref.exists():
        raise FileExistsError(f'case already exists: {case_id}')
    public.mkdir(parents=True); ref.mkdir(parents=True)
    q0, qe = frames[q0_index], frames[qe_index]
    Image.fromarray(q0.rgb).save(public/'q0.png')
    (public/'prompt.txt').write_text(caption+'\n')
    poses = np.linalg.inv(q0.pose_c2w) @ trajectory.poses_world_camera
    times = np.arange(len(poses), dtype=float)/trajectory.fps
    np.savez_compressed(public/'camera.npz', poses_c2w=poses, timestamps_s=times)
    keys = dict(trajectory.key_indices)
    events = {name: {'frame_index': int(i), 'timestamp_s': float(times[i])} for name, i in keys.items()}
    pairs = {}
    for name, prefix in [('C', 'qc'), ('A', 'qa'), ('B', 'qb')]:
        first, revisit = prefix+'_first', prefix+'_revisit'
        if first in keys and revisit in keys:
            pairs[name] = {'first_event': first, 'revisit_event': revisit,
                'first_frame_index': keys[first], 'revisit_frame_index': keys[revisit]}
    if 'C' not in pairs:
        raise ValueError('trajectory lacks its generated-content revisit pair C')
    h, w = q0.rgb.shape[:2]
    write_json(public/'case.json', {'case_id': case_id, 'parent_case_id': parent_case_id, 'source': source_name,
        'template': trajectory.template_name, 'frame_count': len(poses), 'fps': trajectory.fps,
        'pose_unit': q0.depth_unit, 'intrinsics': _intrinsics(q0.K, w, h),
        'events': events, 'revisit_pairs': pairs})
    reference = {'case_id': case_id, 'source': source_name, 'native_to_input': np.eye(3).tolist(),
                 'exact_revisit': True, 'frames': {}, 'anchors': [],
                 'qe_scene_region': 'evaluation_regions/scene.png', 'observability': {}}
    for role, f in [('q0', q0), ('qe', qe)]:
        (ref/role).mkdir()
        Image.fromarray(f.rgb).save(ref/role/'rgb.png'); np.save(ref/role/'depth.npy', f.depth.astype(np.float32))
        Image.fromarray(f.valid.astype(np.uint8)*255).save(ref/role/'valid.png')
        reference['frames'][role] = {'rgb': f'{role}/rgb.png', 'depth': f'{role}/depth.npy',
            'valid': f'{role}/valid.png', 'depth_unit': f.depth_unit, 'T_world_camera': f.pose_c2w.tolist(),
            'rgb_camera': _intrinsics(f.K, f.rgb.shape[1], f.rgb.shape[0]),
            'depth_camera': _intrinsics(f.depth_K, f.depth.shape[1], f.depth.shape[0])}
    projection, diagnostics = anchor_diagnostics(q0, qe, anchors, visibility)
    depth_projection = dense_projection(q0, qe, rgb_grid=False,
        relative_tolerance=(visibility or {}).get('relative_tolerance', .10),
        absolute_tolerance=(visibility or {}).get('absolute_tolerance', .02))
    (ref/'observability').mkdir(); (ref/'evaluation_regions').mkdir()
    for key, mask in [('input_supported', depth_projection['support']), ('invalid', ~qe.valid)]:
        Image.fromarray(mask.astype(np.uint8)*255).save(ref/'observability'/f'{key}.png')
        reference['observability'][key] = f'observability/{key}.png'
    Image.fromarray(projection['support'].astype(np.uint8)*255).save(ref/'evaluation_regions'/'scene.png')
    for i, (a, row) in enumerate(zip(anchors, diagnostics), 1):
        aid = f'A{i}'
        folder = ref/'anchors'/aid; folder.mkdir(parents=True)
        for role in ('q0', 'qe'):
            Image.fromarray(a[role].astype(np.uint8)*255).save(folder/f'{role}.png')
        Image.fromarray(row['evaluation_mask'].astype(np.uint8)*255).save(ref/'evaluation_regions'/f'{aid}.png')
        entry = {'anchor_id': aid}
        if a.get('label'):
            entry['label'] = a['label']
        entry.update(q0_mask=f'anchors/{aid}/q0.png', qe_mask=f'anchors/{aid}/qe.png',
                     qe_region=f'evaluation_regions/{aid}.png')
        reference['anchors'].append(entry)
    write_json(ref/'reference.json', reference)
    return {'case_id': case_id, 'parent_case_id': parent_case_id, 'source': source_name,
            'template': trajectory.template_name, 'frame_count': len(poses), 'anchors': len(anchors)}


def _focus(q0, anchor):
    h, w = q0.depth.shape
    coords = q0.K @ np.linalg.inv(q0.depth_K) @ _grid(h, w)
    coords /= coords[2:3]
    member, _ = _sample(anchor['q0'], coords)
    good = member & q0.valid.ravel()
    if not good.any():
        raise ValueError('orbit anchor has no valid depth samples')
    xyz = (np.linalg.inv(q0.depth_K) @ _grid(h, w))*q0.depth.ravel()
    return q0.pose_c2w[:3, :3] @ np.median(xyz[:, good], axis=1) + q0.pose_c2w[:3, 3]


def scene_mesh_path(source, base):
    """Mesh used for clearance and revisit visibility, when the source provides one."""
    if source.get('mesh'):
        return path(source['mesh'], base)
    if source.get('kind') == 'scannetpp' and source.get('scene_root'):
        return path(source['scene_root'], base)/'scans'/'mesh_aligned_0.05.ply'
    return None


def revisit_evidence(trajectory, frames, config, options, base, *, q0_index):
    """Final event-pixel evidence using the selected case's Q0 calibration."""
    from src.construction.revisits import MeshObserver, ProxyObserver, evaluate_revisit_pairs
    from src.construction.completion import SpatialVIDEventRevisitThresholds
    poses = trajectory.poses_world_camera
    reference = frames[q0_index]
    source = config['source']
    sekai_nonarc = source.get('kind') in {'sekai', 'mira'} and trajectory.template_name != 'arc_return'
    defaults = SpatialVIDEventRevisitThresholds(
        maximum_pose_translation_error=1.0 if sekai_nonarc else 0.15,
        maximum_pose_rotation_error_deg=15.0 if sekai_nonarc else 8.0)
    common = {'voxel': options.get('voxel_size', defaults.surface_voxel_size),
              'q0_exclusion_radius': options.get('q0_exclusion_radius', defaults.q0_exclusion_radius)}
    mesh = scene_mesh_path(source, base)
    if mesh is not None:
        observer = MeshObserver(_raycaster(str(mesh)), reference.K, reference.rgb.shape[1],
                                reference.rgb.shape[0], poses, **common)
    else:
        observer = ProxyObserver([f for f in frames if f.depth is not None], poses,
            reference=reference, **common,
            minimum_depth=options.get('minimum_event_depth', defaults.minimum_event_depth),
            maximum_depth=options.get('maximum_event_depth', defaults.maximum_event_depth))
    result = evaluate_revisit_pairs(poses, trajectory.key_indices, observer,
        max_translation=options.get('maximum_pose_translation_error', defaults.maximum_pose_translation_error),
        max_rotation_deg=options.get('maximum_pose_rotation_error_deg', defaults.maximum_pose_rotation_error_deg))
    result['geometry_grade'] = observer.geometry_grade
    result['crop'] = observer.crop_provenance
    result['coordinate_unit'] = reference.depth_unit
    return result


def quality_record(cache, case, q0, qe, anchors, short, trajectory_quality):
    """Assemble the cached annotations of one case for the quality ranking."""
    from src.utils.camera import CameraIntrinsics
    from src.construction.anchor_depth import anchor_bidirectional_depth_consistency
    for role, frame in [('q0', q0), ('qe', qe)]:
        Image.fromarray(frame.rgb).save(cache/f'{role}.png')
        np.savez_compressed(cache/f'{role}_masks.npz', out_binary_masks=np.stack([a[role] for a in anchors]))
    np.savez_compressed(cache/'trajectory.npz', timestamps_s=np.arange(len(short.poses_world_camera))/short.fps,
        poses_c2w=np.linalg.inv(q0.pose_c2w) @ short.poses_world_camera)
    cameras = [CameraIntrinsics(f.depth.shape[1], f.depth.shape[0], f.depth_K) for f in (q0, qe)]
    ranked_anchors = []
    for i, anchor in enumerate(anchors):
        identity = anchor.get('identity', {})
        q0_depth = np.where(q0.valid, q0.depth, np.nan)
        qe_depth = np.where(qe.valid, qe.depth, np.nan)
        depth = anchor_bidirectional_depth_consistency(q0_depth, qe_depth, *cameras,
            q0.pose_c2w, qe.pose_c2w, anchor['q0'], anchor['qe'])
        ranked_anchors.append({'label': anchor.get('label', ''),
            'identity_confirmed': 'q0_iou' in identity and 'qe_iou' in identity,
            'tracking_iou': {'q0': identity.get('q0_iou'), 'qe': identity.get('qe_iou')},
            'area_percent': {'q0':100*anchor['q0'].mean(), 'qe':100*anchor['qe'].mean()},
            'integrity': {role:anchor.get(role+'_integrity', {}) for role in ('q0','qe')},
            'mask_sources': {role:{'npz':str(cache/f'{role}_masks.npz'),'index':i} for role in ('q0','qe')},
            'depth_consistency': depth})
    completion = case.get('completion_quality')
    return {'case_id':case['case_id'], 'video_id':case.get('video_id', case['case_id']),
        'source_group':case.get('source_group', 'registered'), 'template':case['template'],
        'assets':{'raw_endpoints':{role:str(cache/f'{role}.png') for role in ('q0','qe')},
                  'trajectory_npz':str(cache/'trajectory.npz')},
        'anchors':ranked_anchors,
        'diagnostics':{'trajectory':{'passed':trajectory_quality['short']['passed'],
                                   'template_name':case['template']},
                       'completion':completion or {}},
        'completion_measured': completion is not None and completion.get('measured', True),
        'tracking_measured': all(a['identity_confirmed'] for a in ranked_anchors)}


def rank_review(records):
    """Rank candidates; unavailable evidence stays null."""
    from src.construction.quality import fit_motion_calibration, score_case
    if not records:
        return []
    calibration = fit_motion_calibration(records)
    results = []
    for record in records:
        row = score_case(record, motion_calibration=calibration)
        row['measured_subtotal'] = sum(row['scores'][name] for name in
            ('visual_quality', 'anchor_quality', 'trajectory_quality'))
        row['maximum_measured_subtotal'] = 60.0
        if not record['completion_measured']:
            row['scores'].update(completion_value=None, total=None)
            row['completion_value'] = {'score':None,'maximum':10.0,'status':'unavailable',
                'reason':'shared unseen surface evidence was not provided'}
        if not record['tracking_measured']:
            row['scores']['anchor_quality'] = None
            row['anchor_quality']['status'] = 'independent_endpoint_route_no_tracking_score'
            row['anchor_quality']['score'] = None
            row['maximum_measured_subtotal'] = 25.0
            row['measured_subtotal'] = row['scores']['visual_quality']+row['scores']['trajectory_quality']
            row['scores']['total'] = None
        results.append(row)
    results.sort(key=lambda r:(r['source_group'], r['template'], -r['measured_subtotal']))
    return results


def _review(work, case, q0, qe, anchors, projection, diagnostics):
    folder = work/'review'/case['case_id']; folder.mkdir(parents=True, exist_ok=True)
    for role, f in [('q0', q0), ('qe', qe)]:
        Image.fromarray(f.rgb).save(folder/f'{role}.png')
    cards = []
    for a, r in zip(anchors, diagnostics):
        aid = r['anchor_id']
        Image.fromarray(r['evaluation_mask'].astype(np.uint8)*255).save(folder/f'{aid}_evaluation.png')
        for role, f in [('q0', q0), ('qe', qe)]:
            overlay = f.rgb.copy()
            overlay[a[role]] = (.65*overlay[a[role]] + .35*np.array([245,80,70])).astype(np.uint8)
            if role == 'qe':
                mask = r['evaluation_mask']
                overlay[mask] = (.6*overlay[mask] + .4*np.array([20,225,230])).astype(np.uint8)
            Image.fromarray(overlay).save(folder/f'{aid}_{role}.png')
        warped = projection['projected_rgb'].copy(); warped[~(projection['candidate'] & a['qe'])] = 0
        Image.fromarray(warped).save(folder/f'{aid}_projected.png')
        reverse = projection['reverse']
        warped_back = reverse['projected_rgb'].copy()
        warped_back[~(reverse['candidate'] & a['q0'])] = 0
        Image.fromarray(warped_back).save(folder/f'{aid}_reverse.png')
        cards.append('<h2>'+html.escape(aid+' '+a.get('label',''))+'</h2><p>'+
            html.escape(json.dumps({k:v for k,v in r.items() if k!='evaluation_mask'}))+
            f'</p><figure><a href="{aid}_q0.png"><img src="{aid}_q0.png"></a><a href="{aid}_qe.png"><img src="{aid}_qe.png"></a></figure>'+
            f'<p>Red: complete anchor. Cyan: QE evaluation region. <a href="{aid}_projected.png">Dense Q0 RGB projected into QE</a>; <a href="{aid}_evaluation.png">evaluation mask</a>; <a href="{aid}_reverse.png">dense QE RGB projected into Q0</a>.</p>')
    (folder/'index.html').write_text('<meta charset="utf-8"><style>body{font:16px sans-serif;max-width:1200px;margin:auto}img{width:48%}</style>'+ 
        '<h1>'+html.escape(case['case_id'])+'</h1><img src="q0.png"><img src="qe.png">'+
        '<h2>Trajectory QA</h2><pre>'+html.escape(json.dumps(case.get('trajectory_quality', {}), indent=2))+'</pre>'+
        '<h2>Revisit evidence (shared surfaces unseen at Q0)</h2><pre>'+
        html.escape(json.dumps({k: case.get(k) for k in ('completion_quality', 'long_completion_quality')
                                if case.get(k) is not None}, indent=2))+'</pre>'+''.join(cards)+
        '<p>Set accepted=true and list selected_anchor_ids in cases.json after reviewing.</p>')


def run(config, stage='all'):
    """Stages: geometry, select, annotate, review, export, or all."""
    if isinstance(config, (str, Path)):
        config_path = Path(config).resolve(); config = json.loads(config_path.read_text()); base = config_path.parent
    else:
        config = dict(config); base = Path(config.get('base_dir', '.')).resolve()
    stage = {'candidates': 'select'}.get(stage, stage)
    if stage not in {'all', 'geometry', 'select', 'annotate', 'caption', 'review', 'export'}:
        raise ValueError(f'unknown construction stage {stage}')
    work = local_work(path(config['work_dir'], base))
    output = local_work(path(config['output_dir'], base))
    frames = read_source(config['source'], base)
    geometry_options = dict(config.get('geometry', {}))
    validate_frames(frames, allow_missing_cameras=geometry_options.get('mode') in {'da3', 'vipe'})
    if geometry_options.get('mode') == 'da3':
        geometry_options['checkpoint'] = str(path(geometry_options['checkpoint'], base))
    if geometry_options.get('mode') == 'vipe':
        geometry_options['video'] = str(path(config['source']['video'], base))
        if geometry_options.get('output'):
            geometry_options['output'] = str(path(geometry_options['output'], base))
    frames = geometry(frames, geometry_options, work)
    if stage == 'geometry':
        return {'registered_frames': len(frames), 'work_dir': str(work)}
    selection = dict(config.get('selection', {}))
    if isinstance(selection.get('pose_gates'), str):
        selection['pose_gates'] = json.loads(path(selection['pose_gates'], base).read_text())
    import re
    source = config['source']
    prefix = source.get('case_prefix') or '_'.join([str(source.get('name', '')),
        Path(source['video']).stem if source.get('video') else str(source.get('video_id', ''))])
    selection.setdefault('case_id_prefix', re.sub(r'[^a-zA-Z0-9_-]+', '_', prefix).strip('_'))
    annotation_options = dict(config.get('annotation', {}))
    annotation_options.setdefault('tracking', config['source'].get('kind') != 'self_captured'
        and config['source'].get('name', '').lower() not in {'self-captured', 'self captured', 'self_capture'})
    cases_path = work/'cases.json'
    cases = (json.loads(cases_path.read_text()) if cases_path.exists() and stage in {'export', 'review', 'caption'}
             else config.get('cases'))
    if cases is None:
        cases = json.loads(cases_path.read_text()) if cases_path.exists() else select_pairs(frames, selection)
    if not cases:
        raise ValueError('no candidate endpoint pair passed selection')
    freeze_rank_slate(work, cases, selection, config['source']['name'])
    if stage == 'select':
        write_json(cases_path, cases)
        return {'candidates': len(cases), 'review_file': str(cases_path)}
    ids = {f.frame_id: i for i, f in enumerate(frames)}
    mesh_path = scene_mesh_path(config['source'], base)
    scene_root = config['source'].get('scene_root')
    semantic_mesh = None
    if config['source'].get('kind') == 'scannetpp' and scene_root and annotation_options.get('mesh_anchors', True):
        semantic_mesh = lambda: _semantic_mesh(str(path(scene_root, base)))
    clearance_mesh = None
    if mesh_path is not None and config.get('trajectory_qa', {}).get('mesh_clearance', True):
        clearance_mesh = _raycaster(str(mesh_path))
    results = []; review_rows = []; quality_records = []
    for case in cases:
        q0i, qei = ids[str(case['q0'])], ids[str(case['qe'])]
        q0, qe = frames[q0i], frames[qei]
        try:
            anchors, caption = annotate(case, frames, config.get('models', {}), base, work,
                annotation_options, config.get('visibility'), semantic_mesh)
        except CaseRejected as error:
            review_rows.append({**case, 'accepted': False, 'rejection': str(error)})
            continue
        selected = case.get('selected_anchor_ids')
        if selected is not None:
            anchors = [a for a in anchors if a['anchor_id'] in selected]
        if not anchors:
            review_rows.append({**case, 'accepted': False, 'rejection': 'no selected anchor'})
            continue
        projection, diagnostics = anchor_diagnostics(q0, qe, anchors, config.get('visibility'))
        cached_anchors = []
        cache = work/'annotations'/case['case_id']; cache.mkdir(parents=True, exist_ok=True)
        for a in anchors:
            record = {k:v for k,v in a.items() if k not in {'q0', 'qe'}}
            for role in ('q0', 'qe'):
                target = cache/f'{a["anchor_id"]}_{role}.png'
                Image.fromarray(a[role].astype(np.uint8)*255).save(target)
                record[role+'_mask'] = str(target)
            cached_anchors.append(record)
        row = {**case, 'caption': caption, 'anchors': cached_anchors,
               'anchor_quality': [{k:v for k,v in r.items() if k!='evaluation_mask'} for r in diagnostics]}
        from src.construction.trajectory_checks import generate_checked, prepare_proxy, check_route
        qa_options = {**config.get('trajectory_qa', {}), 'coordinate_unit': q0.depth_unit}
        focus = case.get('focus')
        if case['template'] == 'orbit_around_anchor' and focus is None:
            focus = _focus(q0, anchors[0])
        try:
            short, short_quality = generate_checked(case['template'], q0.pose_c2w, qe.pose_c2w,
                frames=frames, q0_index=q0i, qe_index=qei,
                parameters=case.get('parameters'), focus=focus, scale_free=q0.depth_unit != 'metre',
                config=qa_options, accepted=case.get('accepted', False), mesh=clearance_mesh)
            trajectory_quality = {'short': short_quality}
            long = None
            if case.get('long_case_id'):
                long = extend(short, minimum_revisit_gap_s=config.get('long', {}).get('minimum_revisit_gap_s', 5),
                    arc_parameters=case.get('long', {}).get('arc_parameters', config.get('long', {}).get('arc_parameters'))
                        if case['template'] == 'arc_return' else None)
                trajectory_quality['long'] = check_route(long, q0.pose_c2w, qe.pose_c2w,
                    prepare_proxy(frames, q0i, qei, qa_options),
                    {**qa_options, 'minimum_revisit_gap_s': config.get('long', {}).get('minimum_revisit_gap_s', 5)},
                    mesh=clearance_mesh)
        except ValueError as error:
            row.update(accepted=False, rejection=str(error), trajectory_quality={'passed': False, 'error': str(error)})
            review_rows.append(row)
            _review(work, row, q0, qe, anchors, projection, diagnostics)
            continue
        row['trajectory_quality'] = trajectory_quality
        if annotation_options.get('scene_review', False) and semantic_mesh is None and stage != 'annotate':
            review = scene_review(case, frames, config['models']['vlm'], base, work)
            row['scene_review'] = {'environment': review.environment, 'hazards': review.flagged}
            if not review.passed:
                row.update(accepted=False, rejection='scene review: '+', '.join(review.flagged))
                review_rows.append(row)
                continue
        revisit_options = dict(config.get('revisits', {}))
        if case.get('completion_quality') is None and revisit_options.get('enabled', True):
            row['completion_quality'] = revisit_evidence(short, frames, config, revisit_options, base, q0_index=q0i)
            if long is not None:
                row['long_completion_quality'] = revisit_evidence(long, frames, config, revisit_options, base, q0_index=q0i)
        actual = short_quality.get('attempts', [])
        if actual:
            chosen = actual[-1] if short_quality['passed'] or short_quality.get('accepted_exception') else actual[0]
            row['parameters'] = chosen['parameters']
        quality_records.append(quality_record(cache, row, q0, qe, anchors, short, trajectory_quality))
        review_rows.append(row)
        _review(work, row, q0, qe, anchors, projection, diagnostics)
        if stage in {'annotate', 'caption', 'review'} or not case.get('accepted', False):
            continue
        results.append(export_case(output, case['case_id'], config['source']['name'], frames,
            q0i, qei, anchors, caption, short, visibility=config.get('visibility')))
        if long is not None:
            results.append(export_case(output, case['long_case_id'], config['source']['name'], frames,
                q0i, qei, anchors, caption, long, parent_case_id=case['case_id'], visibility=config.get('visibility')))
    quality_rows = rank_review(quality_records)
    quality_by_id = {r['case_id']:r for r in quality_rows}
    for row in review_rows:
        if row['case_id'] in quality_by_id:
            row['construction_quality'] = quality_by_id[row['case_id']]
    write_json(work/'ranked_review.json', quality_rows)
    write_json(work/'review.json', review_rows)
    write_json(cases_path, review_rows)
    if results:
        manifest_path = output/'manifest.json'
        previous = json.loads(manifest_path.read_text()).get('cases', []) if manifest_path.exists() else []
        write_json(manifest_path, {'cases': previous+results})
    return {'exported_cases': len(results), 'cases': results, 'review_file': str(work/'review.json')}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('config_path', nargs='?'); p.add_argument('--config')
    p.add_argument('--stage', default='all', choices=['all','geometry','candidates','select','annotate','caption','review','export'])
    args = p.parse_args(argv)
    config = args.config or args.config_path
    if not config: p.error('provide --config PATH')
    print(json.dumps(run(config, args.stage), indent=2))


if __name__ == '__main__':
    main()
