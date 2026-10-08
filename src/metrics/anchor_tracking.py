"""Two-window anchor matching, independent of the requested image coordinates."""
from pathlib import Path
import numpy as np
from src.metrics.contract import MetricResult


def tracking_windows(case, info, stride):
    if type(stride) is not int or stride < 1:
        raise ValueError('anchor_tracking_frame_stride must be a positive integer')
    def index(name):
        return info.timestamp_to_video_index(float(case.timestamps_s[case.events[name]]))
    first = str(case.revisit_pairs['C']['first_event'])
    revisit = str(case.revisit_pairs['C']['revisit_event'])
    bounds = {'departure': (index('q0'), index(first)),
              'return': (index(revisit), index('q0_return'))}
    windows = {}
    for name, (a,b) in bounds.items():
        if a is None or b is None or b < a:
            windows[name] = []
        else:
            windows[name] = sorted(set(range(a,b+1,stride)) | {a,b})
    return windows


def _project(points, pose, K):
    cam = (np.linalg.inv(pose) @ np.c_[points, np.ones(len(points))].T).T[:,:3]
    pixels = (K @ cam.T).T
    return pixels[:,:2]/np.maximum(pixels[:,2:3], 1e-12), cam[:,2]


def anchor_trajectory_metric(*, reference, reference_depth, reference_valid, masks,
                             frames, requested_poses, reference_pose, K, windows,
                             options, executed_poses=None):
    import cv2
    h,w = reference.shape[:2]
    detector = cv2.SIFT_create(nfeatures=4096)
    gray = lambda rgb: cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    reference_gray = gray(reference)
    points, labels = [], []
    # Fix surface samples inside every reference anchor, including small masks
    # that contain no globally detected SIFT extrema.
    for name, mask in masks.items():
        valid = np.asarray(mask,bool) & reference_valid & np.isfinite(reference_depth) & (reference_depth>0)
        y,x = np.nonzero(valid)
        if not len(x):
            continue
        corners = cv2.goodFeaturesToTrack(reference_gray, maxCorners=options.anchor_tracking_max_points,
            qualityLevel=0.01, minDistance=3, mask=valid.astype(np.uint8)*255)
        chosen = [] if corners is None else [tuple(v) for v in corners.reshape(-1,2)]
        # Grid points keep featureless anchors in expected support. They do not
        # become successful matches unless the independent matcher accepts them.
        if len(chosen)<options.anchor_tracking_min_matches:
            for j in np.linspace(0,len(x)-1,min(options.anchor_tracking_max_points,len(x))).astype(int):
                pt=(float(x[j]),float(y[j]))
                if pt not in chosen:chosen.append(pt)
        chosen=chosen[:options.anchor_tracking_max_points]
        points.extend(chosen);labels.extend([str(name)]*len(chosen))
    if not points:
        return MetricResult(name='anchor_trajectory', status='not_covered', applicable=True,
                            failure='no reference anchor depth support', coverage={'reference_points':0})
    xy=np.asarray(points,dtype=np.float32)
    keypoints=[cv2.KeyPoint(float(x),float(y),16) for x,y in xy]
    _,desc=detector.compute(reference_gray,keypoints)
    z = reference_depth[np.rint(xy[:,1]).astype(int),np.rint(xy[:,0]).astype(int)]
    xyz = (np.linalg.inv(K) @ np.c_[xy,np.ones(len(xy))].T).T*z[:,None]
    world = (reference_pose @ np.c_[xyz,np.ones(len(xyz))].T).T[:,:3]
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    rows = {}
    visibility_root = Path(options.anchor_tracking_visibility) if options.anchor_tracking_visibility else None
    for i in sorted({i for window in windows.values() for i in window}):
        if i not in frames or i not in requested_poses:
            continue
        expected, depth = _project(world, requested_poses[i], K)
        inside = ((depth > 0) & (expected[:,0] >= 0) & (expected[:,0] < w)
                  & (expected[:,1] >= 0) & (expected[:,1] < h))
        known = ~inside
        visible = np.zeros(len(xy),bool)
        ref_depth = None
        if visibility_root and (visibility_root/f'frame_{i:06d}.npz').is_file():
            with np.load(visibility_root/f'frame_{i:06d}.npz', allow_pickle=False) as archive:
                ref_depth = np.asarray(archive['depth_m'])
                if not np.allclose(archive['pose_c2w'], requested_poses[i], atol=1e-5) or not np.allclose(archive['K'],K):
                    raise ValueError('reference visibility camera does not match requested camera')
        elif np.allclose(requested_poses[i],reference_pose,atol=1e-6):
            ref_depth = np.where(reference_valid,reference_depth,np.nan)
        if ref_depth is not None:
            if ref_depth.shape != (h,w):
                raise ValueError('reference visibility depth must use output resolution')
            ids = np.flatnonzero(inside)
            uv = np.clip(np.rint(expected[ids]).astype(int),[0,0],[w-1,h-1])
            observed_z = ref_depth[uv[:,1],uv[:,0]]
            valid = np.isfinite(observed_z) & (observed_z>0)
            known[ids[valid]] = True
            visible[ids[valid]] = np.abs(depth[ids[valid]]-observed_z[valid]) <= np.maximum(0.02,0.02*observed_z[valid])
        target_gray = gray(frames[i])
        kp, ds = detector.detectAndCompute(target_gray,None)
        matches = {}
        # Bidirectional LK is image-only; requested projections are never used
        # as initial flow, matching gates, or replacements for failed tracks.
        target, forward_status, _ = cv2.calcOpticalFlowPyrLK(reference_gray,target_gray,xy.reshape(-1,1,2),None,
            winSize=(21,21),maxLevel=4,criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,0.01))
        if target is not None:
            back, reverse_status, _ = cv2.calcOpticalFlowPyrLK(target_gray,reference_gray,target,None,
                winSize=(21,21),maxLevel=4,criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,0.01))
            if back is not None:
                for j,uv in enumerate(target.reshape(-1,2)):
                    if (forward_status[j,0] and reverse_status[j,0]
                            and np.linalg.norm(back[j,0]-xy[j])<=1.0
                            and 0<=uv[0]<w and 0<=uv[1]<h):
                        matches[j]=uv
        if desc is not None and ds is not None and len(ds)>1 and len(desc)>1:
            forward = matcher.knnMatch(desc,ds,k=2)
            reverse = matcher.knnMatch(ds,desc,k=2)
            ratio = options.anchor_tracking_ratio_threshold
            reverse_ok = {pair[0].queryIdx:pair[0].trainIdx for pair in reverse
                          if len(pair)==2 and pair[0].distance < ratio*pair[1].distance}
            for pair in forward:
                if len(pair)==2:
                    m,n = pair
                    if m.distance < ratio*n.distance and reverse_ok.get(m.trainIdx)==m.queryIdx:
                        matches[m.queryIdx] = np.asarray(kp[m.trainIdx].pt)
        exec_xy = _project(world,executed_poses[i],K)[0] if executed_poses and i in executed_poses else None
        anchors = {}
        for name in masks:
            ids = [j for j,label in enumerate(labels) if label==str(name)]
            expected_ids = [j for j in ids if visible[j]]
            matched = [j for j in expected_ids if j in matches]
            errors = [float(np.linalg.norm(matches[j]-expected[j])/np.hypot(h,w)) for j in matched]
            ex = [float(np.linalg.norm(matches[j]-exec_xy[j])/np.hypot(h,w)) for j in matched] if exec_xy is not None else []
            anchors[str(name)] = {'reference_points':len(ids), 'known_visibility_points':int(known[ids].sum()),
                                  'expected_points':len(expected_ids),'matched_points':len(matched),
                                  'displacement_diagonal':float(np.mean(errors)) if errors else None,
                                  'executed_displacement_diagonal':float(np.mean(ex)) if ex else None,
                                  'recovered':len(matched)>=options.anchor_tracking_min_matches if expected_ids else None}
        rows[str(i)] = {'frame_index':i,'anchors':anchors}
    groups = {}
    for name,indices in windows.items():
        records = [a for i in indices if str(i) in rows for a in rows[str(i)]['anchors'].values()]
        errors = [a['displacement_diagonal'] for a in records if a['displacement_diagonal'] is not None]
        expected_count = sum(a['expected_points']>0 for a in records)
        recovered_count = sum(a['recovered'] is True for a in records)
        unknown = sum(a['reference_points']-a['known_visibility_points'] for a in records)
        groups[name] = {'frame_indices':indices, 'displacement_diagonal':float(np.mean(errors)) if errors else None,
                        'expected_anchor_observations':expected_count,'matched_anchor_observations':len(errors),
                        'recovered_anchor_observations':recovered_count,
                        'recovery_rate':recovered_count/expected_count if expected_count else None,
                        'unknown_visibility_points':unknown,
                        'status':'ok' if expected_count and not unknown and len(rows)>=len(indices) else 'not_covered'}
    return MetricResult(name='anchor_trajectory',status='ok' if all(v['status']=='ok' for v in groups.values()) else 'not_covered',
                        applicable=True,version='v0.1',raw={'windows':groups,'per_frame':rows},
                        coverage={'reference_points':len(points),'reference_anchor_count':len(masks),
                                  'evaluated_frame_count':len(rows),'expected_frame_count':len(set(sum(windows.values(),[])))},
                        provenance={'matcher':'fixed reference points; bidirectional LK (1px) + mutual-ratio SIFT; no pose-guided search',
                                    'visibility':'reference depth sidecar or exact Q0 pose; others unknown',
                                    'normalization':'image diagonal','reference_scale':'inherited metric depth'})
