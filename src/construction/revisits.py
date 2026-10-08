"""Query search and final shared-unseen event-pixel evidence."""
from __future__ import annotations
from typing import Callable, Mapping
import numpy as np
from src.utils.camera import CameraIntrinsics
from src.data.canonical import canonicalize_camera
from src.construction.completion import (
    CompletionProxyRaycaster,
    completion_strength_grade, _quantize_surface_points, _sorted_unique_voxels,
    _surface_observation, _event_surface_pixel_voxels, _shared_event_pixel_metrics,
    _exclude_surface_near_q0, _voxel_keys,
)


def backproject(depth: np.ndarray, K: np.ndarray, pose_c2w: np.ndarray, valid: np.ndarray | None = None,
                stride: int = 4) -> np.ndarray:
    """World points of valid depth pixels sampled every ``stride`` pixels."""
    depth = np.asarray(depth, dtype=np.float64)[::stride, ::stride]
    ok = np.isfinite(depth) & (depth > 0)
    if valid is not None:
        ok &= np.asarray(valid, dtype=bool)[::stride, ::stride]
    ys, xs = np.nonzero(ok)
    z = depth[ys, xs]
    K = np.asarray(K, dtype=np.float64)
    u, v = xs * stride + 0.5, ys * stride + 0.5
    local = np.stack([(u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z, z], axis=1)
    pose = np.asarray(pose_c2w, dtype=np.float64)
    return local @ pose[:3, :3].T + pose[:3, 3]


class Observer:
    """Calibrated event surface pixels and native input-visible surfels."""

    voxel: float
    q0_exclusion_radius: float = 0.075
    minimum_valid_hit_pixels: int = 64
    geometry_grade: str = 'captured_depth'

    def points(self, index: int) -> np.ndarray:
        raise NotImplementedError

    def event_pixels(self, index: int) -> np.ndarray:
        cache = self.__dict__.setdefault('_event_cache', {})
        if index not in cache:
            cache[index] = _quantize_surface_points(self.points(index), self.voxel)
        return cache[index]

    def voxels(self, index: int) -> np.ndarray:
        return _sorted_unique_voxels(self.event_pixels(index))

    def q0_voxels(self, index: int) -> np.ndarray:
        return self.voxels(index)


class CaptureObserver(Observer):
    """Captured depth pixels on an already rendered route (UE)."""

    def __init__(self, depth_for_index: Callable[[int], np.ndarray], K_for_index: Callable[[int], np.ndarray],
                 poses_c2w: np.ndarray, voxel: float = 0.10, stride: int = 1,
                 q0_exclusion_radius: float = 0.075):
        self.depth_for_index, self.K_for_index = depth_for_index, K_for_index
        self.poses, self.voxel, self.stride = np.asarray(poses_c2w), float(voxel), int(stride)
        self.q0_exclusion_radius = float(q0_exclusion_radius)

    def points(self, index):
        return backproject(self.depth_for_index(index), self.K_for_index(index), self.poses[index], stride=self.stride)


class _RaycastObserver(Observer):
    def points(self, index):
        result = self.raycaster.render(self.event_camera, self.poses[index])
        valid = result.valid.copy()
        if self.minimum_depth is not None:
            valid &= result.depth_m >= self.minimum_depth
        if self.maximum_depth is not None:
            valid &= result.depth_m < self.maximum_depth
        return self.raycaster.hit_points_world(self.event_camera, self.poses[index], result)[valid]

    def event_pixels(self, index):
        cache = self.__dict__.setdefault('_event_cache', {})
        if index not in cache:
            cache[index] = _event_surface_pixel_voxels(self.event_camera, self.poses[index],
                self.raycaster, self.voxel, minimum_depth_m=self.minimum_depth,
                maximum_depth_m=self.maximum_depth)
        return cache[index]

    def q0_voxels(self, index):
        cache = self.__dict__.setdefault('_q0_cache', {})
        if index not in cache:
            _, cache[index] = _surface_observation(self.camera, self.poses[index], self.raycaster, self.voxel)
        return cache[index]


class ProxyObserver(_RaycastObserver):
    """Final visibility-only depth fusion and nearest-sample z-buffer."""

    geometry_grade = 'derived_proxy'

    def __init__(self, frames, poses_c2w: np.ndarray, *, reference=None, voxel: float = 0.05,
                 render_size: tuple[int, int] | None = None, stride: int = 2,
                 q0_exclusion_radius: float = 0.075, minimum_depth: float = 0.5,
                 maximum_depth: float = 8.0, maximum_samples_per_frame: int = 50000):
        from src.construction.depth_proxy import DerivedProxyFrame, DerivedProxyFusionSettings, _static_world_points
        frames = list(frames)
        reference = reference if reference is not None else next(f for f in frames if f.depth is not None)
        settings = DerivedProxyFusionSettings(voxel_size_m=voxel, pixel_stride=stride,
                                               maximum_samples_per_frame=maximum_samples_per_frame)
        clouds = []
        for f in frames:
            if f.depth is None or f.pose_c2w is None:
                continue
            h, w = f.depth.shape
            camera = CameraIntrinsics(w, h, f.depth_K if f.depth_K is not None else f.K)
            samples = DerivedProxyFrame(f.depth, camera, f.pose_c2w, f.valid, str(f.frame_id))
            points, _ = _static_world_points(samples, settings)
            clouds.append(points.astype(np.float32))
        cloud = np.concatenate(clouds) if clouds else np.empty((0, 3), np.float32)
        self.raycaster = CompletionProxyRaycaster(cloud)
        self.poses, self.voxel = np.asarray(poses_c2w), float(voxel)
        self.q0_exclusion_radius = float(q0_exclusion_radius)
        self.minimum_depth, self.maximum_depth = minimum_depth, maximum_depth
        h, w = reference.rgb.shape[:2]
        self.camera = CameraIntrinsics(w, h, np.asarray(reference.K, dtype=np.float64))
        canonical = canonicalize_camera(self.camera)
        self.event_camera = canonical.camera
        self.crop_provenance = dict(canonical.provenance)
        if render_size is not None:
            tw, th = render_size
            self.event_camera = CameraIntrinsics(tw, th, np.diag([tw/1280, th/720, 1.0]) @ self.event_camera.K)


class MeshObserver(_RaycastObserver):
    """Native Q0 mesh visibility and exact canonical-crop event raycasts."""

    geometry_grade = 'mesh'

    def __init__(self, raycaster, K: np.ndarray, width: int, height: int, poses_c2w: np.ndarray,
                 voxel: float = 0.05, q0_exclusion_radius: float = 0.075):
        self.camera = CameraIntrinsics(width, height, np.asarray(K, dtype=np.float64))
        canonical = canonicalize_camera(self.camera)
        self.event_camera, self.crop_provenance = canonical.camera, dict(canonical.provenance)
        self.raycaster, self.poses, self.voxel = raycaster, np.asarray(poses_c2w), float(voxel)
        self.q0_exclusion_radius = float(q0_exclusion_radius)
        self.minimum_depth = self.maximum_depth = None


def grade(first_fraction: float, revisit_fraction: float, valid: bool = True) -> str:
    return completion_strength_grade(first_fraction, revisit_fraction, contract_valid=valid)


def rotation_error_deg(a: np.ndarray, b: np.ndarray) -> float:
    relative = np.asarray(a)[:3, :3].T @ np.asarray(b)[:3, :3]
    return float(np.degrees(np.arccos(np.clip((np.trace(relative) - 1) / 2, -1, 1))))


def pair_evidence(observer: Observer, first: int, revisit: int, q0: int = 0) -> dict:
    """Exact shared-unseen pixels over all valid hit pixels at the event views."""
    seen = observer.q0_voxels(q0)
    a, b = observer.event_pixels(first), observer.event_pixels(revisit)
    shared, n_first, valid_first, first_fraction, n_revisit, valid_revisit, revisit_fraction = (
        _shared_event_pixel_metrics(a, b, seen, observer.voxel, observer.q0_exclusion_radius))
    a_unique, b_unique = _sorted_unique_voxels(a), _sorted_unique_voxels(b)
    a_unseen = _exclude_surface_near_q0(a_unique, seen, observer.voxel, observer.q0_exclusion_radius)
    b_unseen = _exclude_surface_near_q0(b_unique, seen, observer.voxel, observer.q0_exclusion_radius)
    novelty_a = np.count_nonzero(np.isin(_voxel_keys(a), _voxel_keys(a_unseen)))
    novelty_b = np.count_nonzero(np.isin(_voxel_keys(b), _voxel_keys(b_unseen)))
    observed = valid_first > 0 and valid_revisit > 0
    return {'first_visible_voxels': len(a_unique), 'revisit_visible_voxels': len(b_unique),
        'first_valid_hit_pixels': valid_first, 'revisit_valid_hit_pixels': valid_revisit,
        'first_shared_unseen_pixels': n_first, 'revisit_shared_unseen_pixels': n_revisit,
        'first_novelty': float(novelty_a/valid_first) if valid_first else 0.0,
        'revisit_novelty': float(novelty_b/valid_revisit) if valid_revisit else 0.0,
        'shared_unseen_voxels': len(shared), 'first_shared_unseen_fraction': first_fraction,
        'revisit_shared_unseen_fraction': revisit_fraction,
        'strength_grade': grade(first_fraction, revisit_fraction), 'observed': observed,
        'diagnostic_checks': {'minimum_valid_hit_pixels':
            min(valid_first, valid_revisit) >= observer.minimum_valid_hit_pixels},
        'voxel_size': observer.voxel, 'q0_exclusion_radius': observer.q0_exclusion_radius,
        'pixel_denominator': 'all_valid_event_hit_pixels', 'geometry_grade': observer.geometry_grade}


def select_revisit_pair(poses_c2w: np.ndarray, events: Mapping[str, int], template: str, observer: Observer, *,
                        fps: float = 16.0, min_gap_s: float = 8.0, max_translation: float | None = 1.0,
                        max_rotation_deg: float = 15.0, window_radius: int = 24, coarse_stride: int = 4) -> dict:
    """Select ``qc_first``/``qc_revisit`` on a captured path."""
    poses = np.asarray(poses_c2w, dtype=np.float64)
    positions = poses[:, :3, 3]
    min_gap = int(np.floor(min_gap_s * fps)) + 1
    w2 = int(events.get('w2', events.get('w', len(poses) // 2)))
    end = int(events.get('q0_return', len(poses) - 1))

    def admissible(first, revisit):
        if revisit - first < min_gap:
            return None
        t = float(np.linalg.norm(positions[revisit] - positions[first]))
        r = rotation_error_deg(poses[first], poses[revisit])
        if (max_translation is None or t <= max_translation) and r <= max_rotation_deg:
            return t, r
        return None

    candidates = []
    if template == 'peek_occlude_retract':
        first_center, revisit_center = int(events['qc_first']), int(events['qc_revisit'])
        for first in range(max(1, first_center - window_radius), min(w2 - 1, first_center + window_radius) + 1):
            local = []
            for revisit in range(max(w2 + 1, revisit_center - window_radius),
                                 min(end - 1, revisit_center + window_radius) + 1):
                errors = admissible(first, revisit)
                if errors:
                    local.append((*errors, revisit))
            if local:
                t, r, revisit = min(local)
                candidates.append((first, revisit, t, r))
        coarse = candidates
        policy = 'peek_repeated_far_visit'
    else:
        seen = set()
        for first in range(w2 - 1, 0, -1):
            revisit = w2 + int(round((w2 - first) / float(w2) * (end - w2)))
            if (first, revisit) in seen or not w2 < revisit < end:
                continue
            seen.add((first, revisit))
            errors = admissible(first, revisit)
            if errors:
                candidates.append((first, revisit, *errors))
        coarse = candidates[::coarse_stride]
        policy = 'mirrored_around_w2'
    search = {'policy': policy, 'candidates': len(candidates), 'minimum_gap_frames': min_gap,
              'maximum_translation': max_translation, 'maximum_rotation_deg': max_rotation_deg}
    if not candidates:
        return {'selected': None, 'strength_grade': 'invalid', 'reason': 'no pose-aligned pair with the required gap',
                'search': search}

    def score(first, revisit, t, r):
        evidence = pair_evidence(observer, first, revisit, int(events.get('q0', 0)))
        key = (min(evidence['first_novelty'], evidence['revisit_novelty']), evidence['shared_unseen_voxels'],
               revisit - first, -t, -r)
        return key, first, revisit, {**evidence, 'time_gap_frames': revisit - first,
                                     'event_pose_translation_error': t, 'event_pose_rotation_error_deg': r}

    ranked = [score(*c) for c in coarse]
    best = max(ranked)
    ranked += [score(*c) for c in candidates if abs(c[0] - best[1]) <= coarse_stride - 1 and c not in coarse]
    _, first, revisit, evidence = max(ranked)
    return {'selected': {'qc_first': first, 'qc_revisit': revisit}, **evidence,
            'passed': evidence['strength_grade'] != 'invalid', 'search': search}


def evaluate_revisit_pairs(poses_c2w: np.ndarray, key_indices: Mapping[str, int], observer: Observer, *,
                          max_translation: float = 0.15, max_rotation_deg: float = 8.0) -> dict:
    """Apply source pose gates and final event-pixel grading to QC/QA/QB."""
    poses = np.asarray(poses_c2w, dtype=np.float64)
    pairs = {}
    for prefix in ('qc', 'qa', 'qb'):
        first, revisit = key_indices.get(prefix + '_first'), key_indices.get(prefix + '_revisit')
        if first is None or revisit is None:
            continue
        evidence = pair_evidence(observer, int(first), int(revisit), int(key_indices.get('q0', 0)))
        translation = float(np.linalg.norm(poses[revisit, :3, 3] - poses[first, :3, 3]))
        rotation = rotation_error_deg(poses[first], poses[revisit])
        valid = translation <= max_translation and rotation <= max_rotation_deg
        evidence.update(event_pose_translation_error=translation, event_pose_rotation_error_deg=rotation,
                        passed=bool(valid), strength_grade=grade(evidence['first_shared_unseen_fraction'],
                        evidence['revisit_shared_unseen_fraction'], valid),
                        checks={'event_pose_translation': translation <= max_translation,
                                'event_pose_rotation': rotation <= max_rotation_deg})
        pairs[prefix] = evidence
    if not pairs:
        return {'passed': False, 'strength_grade': 'invalid', 'measured': False, 'pairs': {}}
    order = {'invalid': 0, 'weak': 1, 'moderate': 2, 'strong': 3}
    weakest = min(pairs.values(), key=lambda e: (order[e['strength_grade']],
                    min(e['first_shared_unseen_fraction'], e['revisit_shared_unseen_fraction'])))
    return {'passed': all(e['passed'] for e in pairs.values()), 'strength_grade': weakest['strength_grade'],
        'measured': all(e['observed'] for e in pairs.values()),
        'first_shared_unseen_fraction': weakest['first_shared_unseen_fraction'],
        'revisit_shared_unseen_fraction': weakest['revisit_shared_unseen_fraction'],
        'event_pose_translation_error': max(e['event_pose_translation_error'] for e in pairs.values()),
        'event_pose_rotation_error_deg': max(e['event_pose_rotation_error_deg'] for e in pairs.values()),
        'pose_limits': {'translation': max_translation, 'rotation_deg': max_rotation_deg},
        'pixel_denominator': 'all_valid_event_hit_pixels', 'pairs': pairs}


# ----------------------------------------------------------------------------
# Query search on rendered paths
# ----------------------------------------------------------------------------

def projected_anchor_support(q0_mask, q0_depth, K0, pose0, target_mask, target_depth, Kt, pose_t, *,
                             relative_tolerance: float = 0.02, absolute_tolerance: float = 0.03) -> dict:
    """Fraction of the target anchor explained by visible input-anchor surfaces."""
    q0_mask, target_mask = np.asarray(q0_mask, bool), np.asarray(target_mask, bool)
    q0_depth, target_depth = np.asarray(q0_depth, np.float64), np.asarray(target_depth, np.float64)
    source = q0_mask & np.isfinite(q0_depth) & (q0_depth > 0)
    target_pixels = int(target_mask.sum())
    empty = {'query_anchor_support': 0.0, 'input_anchor_survival': 0.0}
    if not source.any() or not target_pixels:
        return empty
    rows, cols = np.nonzero(source)
    z0 = q0_depth[rows, cols]
    K0, Kt = np.asarray(K0, np.float64), np.asarray(Kt, np.float64)
    points = np.stack([(cols + .5 - K0[0, 2]) / K0[0, 0] * z0, (rows + .5 - K0[1, 2]) / K0[1, 1] * z0, z0,
                       np.ones_like(z0)], axis=1)
    local = (np.linalg.inv(pose_t) @ np.asarray(pose0) @ points.T).T
    z = local[:, 2]
    with np.errstate(divide='ignore', invalid='ignore'):
        u = np.floor(Kt[0, 0] * local[:, 0] / z + Kt[0, 2]).astype(np.int64)
        v = np.floor(Kt[1, 1] * local[:, 1] / z + Kt[1, 2]).astype(np.int64)
    h, w = target_mask.shape
    inside = (z > 1e-8) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    rows, cols, z, u, v = rows[inside], cols[inside], z[inside], u[inside], v[inside]
    if not len(z):
        return empty
    linear = v * w + u
    order = np.lexsort((z, linear))
    first = order[np.r_[True, linear[order][1:] != linear[order][:-1]]]
    rows, cols, z, u, v = rows[first], cols[first], z[first], u[first], v[first]
    observed = target_depth[v, u]
    tolerance = np.maximum(absolute_tolerance, relative_tolerance * np.maximum(np.minimum(z, observed), 1e-4))
    visible = np.isfinite(observed) & (observed > 0) & (np.abs(z - observed) <= tolerance) & target_mask[v, u]
    support = np.zeros_like(target_mask)
    support[v[visible], u[visible]] = True
    from scipy.ndimage import binary_dilation
    support = binary_dilation(support, np.ones((3, 3), bool)) & target_mask
    return {'query_anchor_support': float(support.sum() / target_pixels),
            'input_anchor_survival': float(len(set(zip(rows[visible], cols[visible]))) / int(source.sum()))}


def select_query(num_frames: int, anchor_mask: Callable[[int], np.ndarray], depth: Callable[[int], np.ndarray],
                 K: Callable[[int], np.ndarray], poses_c2w: np.ndarray, *, search_end: int = 170,
                 minimum_anchor_fraction: float = 0.005, minimum_retained_fraction: float = 0.5,
                 minimum_projected_support: float = 0.40, coarse_stride: int = 4, spacing: int = 12,
                 alternatives: int = 5) -> dict:
    """Latest frame before ``search_end`` that satisfies the query criteria."""
    poses = np.asarray(poses_c2w, dtype=np.float64)
    m0, d0 = anchor_mask(0), depth(0)
    base = float(m0.mean())
    upper = min(int(search_end), num_frames - 1)
    cache: dict[int, dict] = {}

    def evaluate(i):
        if i in cache:
            return cache[i]
        mask = anchor_mask(i)
        fraction = float(mask.mean())
        row = {'frame_index': i, 'anchor_fraction': fraction,
               'retained_fraction': fraction / base if base else 0.0,
               'clipped': bool(mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any())}
        row['mask_eligible'] = fraction >= minimum_anchor_fraction and row['retained_fraction'] >= minimum_retained_fraction
        if row['mask_eligible']:
            d = depth(i)
            row['valid_anchor_depth'] = bool((mask & np.isfinite(d) & (d > 0)).any())
            row.update(projected_anchor_support(m0, d0, K(0), poses[0], mask, d, K(i), poses[i]))
            row['passed'] = row['valid_anchor_depth'] and row['query_anchor_support'] >= minimum_projected_support
        else:
            row['passed'] = False
        cache[i] = row
        return row

    selected = None
    coarse = next((i for i in range(upper, 0, -coarse_stride) if evaluate(i)['passed']), None)
    if coarse is not None:
        selected = next((evaluate(i) for i in range(min(upper, coarse + coarse_stride - 1), coarse - 1, -1)
                         if evaluate(i)['passed']), None)
    candidates, cursor = [], selected['frame_index'] if selected else upper
    while cursor > 0 and len(candidates) < alternatives:
        row = evaluate(cursor)
        if row['mask_eligible'] and row.get('valid_anchor_depth'):
            candidates.append(row)
            cursor -= spacing
        else:
            cursor -= 1
    return {'selected': selected, 'candidates': candidates,
            'criteria': {'search_end': upper, 'minimum_anchor_fraction': minimum_anchor_fraction,
                         'minimum_retained_fraction': minimum_retained_fraction,
                         'minimum_projected_support': minimum_projected_support},
            'input_anchor_fraction': base}


def canonical_to_source(index: int, source_count: int, canonical_count: int) -> int:
    """Map a benchmark frame index to the nearest frame of a capture with another length."""
    if source_count == canonical_count:
        return int(index)
    return int(round(index * (source_count - 1) / (canonical_count - 1)))
