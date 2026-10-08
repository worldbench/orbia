"""Exact-event RGB/depth persistence, grouped by requested (not recovered) pose."""
import numpy as np
from src.metrics.pose import rotation_error_deg


def direct_depth(first, revisit):
    if first is None or revisit is None:
        return {'status': 'not_available', 'abs_rel': None, 'coverage': 0.0}
    a, b = np.asarray(first).squeeze(), np.asarray(revisit).squeeze()
    if a.ndim != 2 or a.shape != b.shape:
        raise ValueError('same-pose depth maps must share a 2D grid')
    valid = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)
    return {'status': 'ok' if valid.any() else 'not_covered',
            'abs_rel': float(np.mean(np.abs(b[valid]-a[valid])/a[valid])) if valid.any() else None,
            'coverage': float(valid.mean()), 'valid_pixel_count': int(valid.sum()),
            'expected_pixel_count': int(a.size), 'scale_policy': 'shared_estimator_scale_no_pair_refit'}


def direct_revisit(case, frame_pairs, frames, depths, comparator):
    records = {}
    for name, pair in case.revisit_pairs.items():
        i, j = pair['first_window_indices'][0], pair['revisit_window_indices'][0]
        left, right = case.requested_poses[i], case.requested_poses[j]
        translation = float(np.linalg.norm(left[:3, 3]-right[:3, 3]))
        rotation = rotation_error_deg(left, right)
        same = translation <= 1e-6 and rotation <= 1e-4
        video_pairs = frame_pairs.get(name, [])
        row = {'pose_class': 'same_pose' if same else 'different_pose',
               'requested_translation_difference': translation, 'requested_rotation_difference_deg': rotation,
               'first_canonical_index': i, 'revisit_canonical_index': j,
               'status': 'not_covered', 'appearance': {},
               'depth': {'status': 'not_available' if same else 'not_applicable'}}
        if video_pairs:
            if len(video_pairs) != 1:
                raise ValueError('direct revisit requires one registered event pair')
            u, v = video_pairs[0]
            row.update(first_video_index=u, revisit_video_index=v)
            if u in frames and v in frames:
                if same:
                    values = comparator.compare(frames[u], frames[v])
                    row['appearance'] = {'rgb_l1': values['perceptual_l1'], 'psnr': values['psnr'],
                        'lpips': values['lpips'], 'dino_patch_similarity': values['dino_similarity'],
                        'dino_global_similarity': values['global_feature_similarity']}
                    row['depth'] = direct_depth(depths.get(u), depths.get(v))
                else:
                    a = comparator._descriptor(frames[u], None, cache_key=('frame', u))[0]
                    b = comparator._descriptor(frames[v], None, cache_key=('frame', v))[0]
                    denom = float(np.linalg.norm(a)*np.linalg.norm(b))
                    row['appearance'] = {'dino_global_similarity': float(np.dot(a,b)/denom) if denom > 0 else None}
                row['status'] = 'ok' if all(x is not None for k,x in row['appearance'].items() if k.startswith('dino')) else 'not_covered'
        records[name] = row
    groups = {}
    for kind in ('same_pose', 'different_pose'):
        items = [r for r in records.values() if r['pose_class'] == kind]
        fields = ['dino_global_similarity']
        if kind == 'same_pose': fields += ['dino_patch_similarity', 'rgb_l1', 'psnr', 'lpips', 'depth_abs_rel', 'depth_coverage']
        group = {'expected_pair_count': len(items), 'evaluated_pair_count': sum(r['status']=='ok' for r in items)}
        for field in fields:
            values = [r['depth'].get(field[6:]) if field.startswith('depth_') else r['appearance'].get(field) for r in items]
            present = [float(x) for x in values if x is not None and np.isfinite(x)]
            group[field] = {'mean': float(np.mean(present)) if items and len(present)==len(items) else None,
                            'conditional_mean': float(np.mean(present)) if present else None,
                            'evaluated_pair_count': len(present), 'expected_pair_count': len(items)}
        groups[kind] = group
    return {'pairs': records, **groups, 'protocol': 'fixed requested-pose classification; full-image comparisons; generated first visit reference',
            'pose_tolerance': {'translation': 1e-6, 'rotation_deg': 1e-4},
            'different_pose_interpretation': 'global semantic similarity, not complete expected co-visibility'}
