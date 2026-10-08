"""Action control, visual quality and temporal profiles for a generated video."""
from __future__ import annotations

import sys
import tempfile
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from src.metrics.scoring import mean, normalize, error_score, CONFIG


def window_bounds(duration, size=5.):
    starts = list(np.arange(0., max(duration-size,0.)+1e-9,size))
    tail = max(duration-size,0.)
    if not starts or not np.isclose(starts[-1],tail):
        starts.append(tail)
    return [(float(s),float(min(s+size,duration))) for s in starts]


def window_means(values, times, bounds):
    values, times = np.asarray(values,float), np.asarray(times,float)
    return [mean(values[(times>=a)&((times<=b+1e-8) if i == len(bounds)-1 else times<b)])
            for i,(a,b) in enumerate(bounds)]


def action_series(indices, predicted, requested, fps, *, movement=True):
    step = max(1, round(fps*(2 if movement else 4)/16))
    selected = np.asarray(indices)%step == 0
    idx, p, q = np.asarray(indices)[selected], np.asarray(predicted)[selected], np.asarray(requested)[selected]
    valid = np.diff(idx) == step
    def labels(poses):
        if movement:
            delta = np.diff(poses[:,:3,3],axis=0)
            speed = np.linalg.norm(delta,axis=1)
            local = np.einsum('nij,nj->ni',poses[:-1,:3,:3].transpose(0,2,1),delta)
            direction = local/np.maximum(speed[:,None],1e-12)
            active = speed > .01*float(speed.mean())
            return np.stack([direction[:,2]>.5,direction[:,0]<-.5,direction[:,2]<-.5,direction[:,0]>.5],axis=1)&active[:,None]
        angle = np.degrees(Rotation.from_matrix(poses[:-1,:3,:3].transpose(0,2,1)@poses[1:,:3,:3]).as_rotvec())
        return np.stack([angle[:,0]>.4,angle[:,1]<-.4,angle[:,0]<-.4,angle[:,1]>.4],axis=1)
    pl, ql = labels(p), labels(q)
    active = valid & ql.any(axis=1)
    return ((idx[:-1]+idx[1:])/2/fps)[active], (pl==ql).all(axis=1)[active].astype(float)*100


@lru_cache(maxsize=2)
def _hps(repo, config, checkpoint, device):
    sys.path.insert(0,repo)
    from hpsv3.inference import HPSv3RewardInferencer
    return HPSv3RewardInferencer(config_path=config,checkpoint_path=checkpoint,device=device)


def hps_scores(frames, config, output):
    keys = ('hps_repo','hps_config','hps_checkpoint')
    if not all(config.get(k) for k in keys):
        raise ValueError('configure hps_repo, hps_config and hps_checkpoint for full T2/T6')
    from PIL import Image
    scorer = _hps(*(str(config[k]) for k in keys),str(config.get('device','cuda')))
    with tempfile.TemporaryDirectory(dir=output,prefix='hps-') as temp:
        paths = []
        for i,frame in enumerate(frames):
            path = Path(temp)/f'{i:06d}.png'
            Image.fromarray(frame).save(path)
            paths.append(str(path))
        batch_size = int(config.get('hps_batch_size', 8))
        if batch_size < 1:
            raise ValueError('hps_batch_size must be positive')
        scores = []
        for start in range(0, len(paths), batch_size):
            batch = paths[start:start + batch_size]
            rewards = scorer.reward([''] * len(batch), batch)
            scores.extend(float(r[0].item() if hasattr(r[0], 'item') else r[0]) for r in rewards)
        return scores


def calculate(case, info, frames, report, pose_cache, options, config, output, *, control_profile='pose_and_action', group='world', tiers=None):
    """Calculate the requested T1/T2/T6 records; omission keeps the full default."""
    if tiers is None:
        selected = {'T1', 'T2', 'T6'}
    else:
        values = tiers.replace(',', ' ').split() if isinstance(tiers, str) else tiers
        selected = {str(value).upper() for value in values}
        unknown = selected - {'T1', 'T2', 'T3', 'T4', 'T5', 'T6'}
        if unknown:
            raise ValueError('unknown tiers: '+', '.join(sorted(unknown)))
    result = {'horizon': 'long' if info.frame_count>500 else 'short'}
    if not selected.intersection({'T1', 'T2', 'T6'}):
        return result
    bounds = window_bounds((info.frame_count-1)/info.fps)
    curves = {}
    if selected.intersection({'T1', 'T6'}):
        from src.metrics.pose import load_pose_cache
        from src.metrics.runner import _align_track
        raw = report['metrics'].get('control',{}).get('raw',{})
        pose_errors = raw.get('per_pose_errors',[])
        pose_errors = [r for r in pose_errors if r['frame_index']%options.control_frame_stride == 0]
        times = [r['frame_index']/info.fps for r in pose_errors]
        control = {}
        for key,field,bound in [('translation','translation_error_normalized',.215320517416),
                                ('rotation','rotation_error_deg',32.764762231129)]:
            values = [r[field] for r in pose_errors]
            control[key] = error_score(mean(values),bound)
            curves[key] = [error_score(v,bound) for v in window_means(values,times,bounds)]
        cache = load_pose_cache(pose_cache,video_frame_count=info.frame_count)
        track = _align_track(case,info,cache,options.control_frame_stride)
        template = case.template
        if np.linalg.norm(np.diff(case.requested_poses[:,:3,3],axis=0),axis=1).sum() < 1e-6:
            template = 'yaw_return'
        action_applicable = group != 'ti2v' or template not in ('arc_return','orbit_around_anchor')
        for key in ('move','turn'):
            if track is not None and action_applicable and not (key == 'move' and template == 'yaw_return'):
                at, scores = action_series(track.cache.frame_indices,track.executed_relative_unscaled,
                                           track.poses,info.fps,movement=key == 'move')
            else:
                at, scores = [], []
            control[key] = mean(scores)
            curves[key] = window_means(scores,at,bounds)
        if template == 'yaw_return':
            control['translation'] = None
            curves['translation'] = [None]*len(bounds)
        if 'T1' in selected:
            result['control'] = control
    if selected.intersection({'T2', 'T6'}):
        quality = {}
        for key,name,field in [('aesthetic','aesthetic_quality','laion_aesthetic'),('imaging','imaging_quality','musiq')]:
            metric = report['metrics'].get(name,{}).get('raw',{})
            quality[key] = 100*metric[field]['mean'] if metric.get(field,{}).get('mean') is not None else None
            idx = list(range(0,info.frame_count,options.tier2_quality_frame_stride))
            per_frame = metric.get('per_frame',[])
            if len(per_frame) != len(idx):
                raise ValueError(name+' has missing per-frame outputs')
            curves[key] = window_means(np.asarray(per_frame)*100,np.asarray(idx)/info.fps,bounds)
        if 'T2' in selected:
            fidx = list(range(0,info.frame_count,2))
            flicker = [float(np.abs(frames[b].astype(float)-frames[a]).mean()) for a,b in zip(fidx[:-1],fidx[1:])]
            quality['flicker'] = 100*(1-mean(flicker)/255) if flicker else None
            smooth = report['metrics'].get('motion_smoothness',{}).get('raw',{})
            quality['smoothness'] = 100*smooth['amt_interpolation_score'] if smooth.get('amt_interpolation_score') is not None else None
        hi = np.linspace(0,info.frame_count-1,54 if info.frame_count>500 else 20,dtype=int)
        hraw = hps_scores([frames[i] for i in hi],config,output)
        hmap = {'mode':'linear','lo':-5.21,'hi':8.61,'direction':1}
        quality['hps'] = normalize(mean(hraw),hmap)
        curves['hps'] = [normalize(v,hmap) for v in window_means(hraw,hi/info.fps,bounds)]
        if 'T2' in selected:
            result['quality'] = quality
    if 'T6' in selected:
        for name,field,branches in [('depth_temporal_consistency','abs_rel',[('short','depth_short'),('medium','depth_medium')]),
                                    ('covisible_reprojection','feature_reprojection_similarity',[('short','reproj_short'),('medium','reproj_medium')])]:
            for branch,key in branches:
                pairs = report['metrics'].get(name,{}).get('raw',{}).get(branch,{}).get('pairs',[])
                by = {}
                for pair in pairs:
                    if pair.get(field) is not None:
                        by.setdefault(pair['target_frame_index'],[]).append(pair[field])
                idx = sorted(by)
                curves[key] = [normalize(v,CONFIG['maps'][key]) for v in window_means(
                    [mean(by[i]) for i in idx],np.asarray(idx)/info.fps,bounds)]
        regional = report['metrics'].get('regional_scale_inconsistency',{}).get('raw',{}).get('pairs',[])
        if not regional:
            regional = report['metrics'].get('regional_scale_inconsistency',{}).get('raw',{}).get('per_pair',[])
        curves['scale'] = [normalize(v,CONFIG['maps']['scale']) for v in window_means(
            [r.get('regional_scale_inconsistency',r.get('value')) for r in regional],
            [r.get('target_frame_index',0)/info.fps for r in regional],bounds)]
        def combine(names):
            return [mean(curves[k][i] for k in names) for i in range(len(bounds))]
        curves['depth'] = combine(['depth_short','depth_medium'])
        curves['feature'] = combine(['reproj_short','reproj_medium'])
        control_keys = ['move','turn'] if control_profile == 'action_only' else ['translation','rotation','move','turn']
        dimensions = {'quality':combine(['hps','imaging','aesthetic']),
                      'control':combine(control_keys),'geometry':combine(['depth','feature','scale'])}
        from src.metrics.scoring import retention
        indicator_parts = {name:retention(values) for name,values in curves.items()}
        def combine_parts(names):
            valid = [indicator_parts[n] for n in names if indicator_parts[n] is not None]
            return {k:mean(v[k] for v in valid) for k in ('level','retention')} if valid else None
        indicator_parts['depth'] = combine_parts(['depth_short','depth_medium'])
        indicator_parts['feature'] = combine_parts(['reproj_short','reproj_medium'])
        parts = {'quality':combine_parts(['hps','imaging','aesthetic']),
                 'control':combine_parts(control_keys),'geometry':combine_parts(['depth','feature','scale'])}
        score = mean(.3*v['level']+.7*v['retention'] for v in parts.values()) if all(parts.values()) else None
        result['temporal'] = dict(score=score,dimensions=dimensions,parts=parts,bounds=bounds,curves=curves)
    return result
