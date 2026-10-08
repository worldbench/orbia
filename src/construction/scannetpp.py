"""ScanNet++ iPhone RGB-D frames and mesh-guided anchor annotation."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image

from src.construction.sources import RegisteredFrame

EXCLUDED_LABEL_WORDS = ('wall', 'floor', 'ceiling', 'window', 'glass', 'mirror', 'transparent', 'remove')


# ----------------------------------------------------------------------------
# iPhone RGB-D
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class ColmapCamera:
    width: int
    height: int
    K: np.ndarray
    model: str
    distortion: tuple


def _quaternion_to_matrix(qw, qx, qy, qz):
    q = np.array([qw, qx, qy, qz], dtype=np.float64)
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def read_colmap(colmap_dir: str | Path) -> tuple[dict[int, ColmapCamera], dict[str, tuple[int, np.ndarray]]]:
    """Read COLMAP text cameras and images; returns camera models and ``name -> (camera_id, c2w)``."""
    colmap_dir = Path(colmap_dir)
    cameras = {}
    for line in (colmap_dir / 'cameras.txt').read_text().splitlines():
        if not line.strip() or line.startswith('#'):
            continue
        parts = line.split()
        cid, model, w, h = int(parts[0]), parts[1].upper(), int(parts[2]), int(parts[3])
        p = [float(v) for v in parts[4:]]
        if model == 'SIMPLE_PINHOLE':
            fx = fy = p[0]; cx, cy = p[1:3]; dist = ()
        elif model == 'PINHOLE':
            fx, fy, cx, cy = p[:4]; dist = ()
        elif model == 'OPENCV':
            fx, fy, cx, cy = p[:4]; dist = tuple(p[4:8])
        else:
            raise ValueError(f'unsupported COLMAP camera model {model}')
        if not any(dist):
            model, dist = 'PINHOLE', ()
        cameras[cid] = ColmapCamera(w, h, np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.]]), model, dist)
    images = {}
    for line in (colmap_dir / 'images.txt').read_text().splitlines():
        parts = line.split()
        # Image rows have exactly 10 fields; POINTS2D rows hold (x, y, id) triplets.
        if line.startswith('#') or len(parts) != 10:
            continue
        qw, qx, qy, qz, tx, ty, tz = map(float, parts[1:8])
        w2c = np.eye(4)
        rotation = _quaternion_to_matrix(qw, qx, qy, qz)
        u, _, vt = np.linalg.svd(rotation)  # remove decimal rounding drift
        w2c[:3, :3] = u @ vt
        w2c[:3, 3] = [tx, ty, tz]
        images[parts[9]] = (int(parts[8]), np.linalg.inv(w2c))
    return cameras, images


def sharpness(rgb: np.ndarray) -> float:
    """Variance of the Laplacian of the grey image."""
    grey = np.asarray(rgb, dtype=np.float64).mean(axis=-1)
    lap = grey[1:-1, :-2] + grey[1:-1, 2:] + grey[:-2, 1:-1] + grey[2:, 1:-1] - 4 * grey[1:-1, 1:-1]
    return float(lap.var())


def read_frames(scene_root: str | Path, names: Sequence[str] | None = None, *, stride: int = 1,
                depth_scale: float = 0.001, minimum_valid_depth: float = 0.0) -> list[RegisteredFrame]:
    """Registered iPhone frames with native sensor depth and depth-grid intrinsics."""
    root = Path(scene_root)
    cameras, images = read_colmap(root / 'iphone' / 'colmap')
    selected = list(names) if names is not None else sorted(images)[::stride]
    frames = []
    for name in selected:
        cid, c2w = images[name]
        camera = cameras[cid]
        rgb = np.asarray(Image.open(root / 'iphone' / 'rgb' / name).convert('RGB'))
        depth = np.asarray(Image.open(root / 'iphone' / 'depth' / (Path(name).stem + '.png')), dtype=np.float32)
        depth *= float(depth_scale)
        valid = np.isfinite(depth) & (depth > 0)
        if names is None and valid.mean() < minimum_valid_depth:
            continue
        h, w = rgb.shape[:2]
        K = np.diag([w / camera.width, h / camera.height, 1.0]) @ camera.K
        depth_K = np.diag([depth.shape[1] / w, depth.shape[0] / h, 1.0]) @ K
        index = int(''.join(c for c in Path(name).stem if c.isdigit()) or 0)
        frames.append(RegisteredFrame(Path(name).stem, rgb, depth, valid, c2w, K, depth_K, 'metre', index / 60.0))
    return frames


# ----------------------------------------------------------------------------
# Semantic mesh
# ----------------------------------------------------------------------------

class SemanticMesh:
    """Laser-scan mesh with per-triangle object IDs and labels."""

    def __init__(self, scene_root: str | Path | None = None, *, mesh: str | Path | None = None,
                 segments: str | Path | None = None, annotation: str | Path | None = None, raycaster=None,
                 vertex_objects: np.ndarray | None = None, labels: dict | None = None):
        from src.utils.geometry import MeshRaycaster
        scans = Path(scene_root) / 'scans' if scene_root is not None else None
        self.raycaster = raycaster or MeshRaycaster(mesh or scans / 'mesh_aligned_0.05.ply')
        if vertex_objects is None:
            seg = np.asarray(json.loads(Path(segments or scans / 'segments.json').read_text())['segIndices'])
            groups = json.loads(Path(annotation or scans / 'segments_anno.json').read_text())['segGroups']
            lookup = {}
            labels = {}
            for group in groups:
                oid = int(group['objectId']) + 1  # 0 is reserved for background
                labels[oid] = str(group.get('label', ''))
                for s in group['segments']:
                    lookup[int(s)] = oid
            vertex_objects = np.array([lookup.get(int(s), 0) for s in seg], dtype=np.int32)
        self.labels = dict(labels or {})
        tri = self.raycaster.triangles
        votes = vertex_objects[tri]
        majority = np.where(votes[:, 1] == votes[:, 2], votes[:, 1], votes[:, 0])
        self.triangle_objects = np.where(votes[:, 0] == votes[:, 1], votes[:, 0], majority).astype(np.int32)

    def render(self, K: np.ndarray, width: int, height: int, pose_c2w: np.ndarray,
               model: str = 'PINHOLE', distortion: tuple = ()) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Object IDs, metric z-depth and triangle IDs of the first visible surface per pixel."""
        from src.utils.camera import CameraIntrinsics
        result = self.raycaster.render(CameraIntrinsics(width, height, K, model, distortion), pose_c2w)
        triangles = np.where(result.valid, result.primitive_ids, 0).astype(np.int64)
        objects = np.where(result.valid, self.triangle_objects[triangles], 0)
        return objects, np.where(result.valid, result.depth_m, np.nan), np.where(result.valid, triangles, -1)

    def excluded(self, object_id: int) -> bool:
        label = self.labels.get(int(object_id), '').lower()
        return not label or any(word in label for word in EXCLUDED_LABEL_WORDS)


def _depth_agreement(mask, mesh_depth, frame, tolerance):
    """Fraction of valid sensor-depth samples in the mask that agree with the mesh."""
    dh, dw = frame.depth.shape
    h, w = mask.shape
    ys, xs = np.nonzero(frame.valid)
    u = np.clip(((xs + .5) * w / dw).astype(int), 0, w - 1)
    v = np.clip(((ys + .5) * h / dh).astype(int), 0, h - 1)
    inside = mask[v, u]
    if not inside.any():
        return 0.0
    sensor = frame.depth[ys[inside], xs[inside]]
    mesh = mesh_depth[v[inside], u[inside]]
    ok = np.isfinite(mesh) & (np.abs(mesh - sensor) <= tolerance * sensor)
    return float(ok.mean())


def mesh_anchor_candidates(mesh: SemanticMesh, q0: RegisteredFrame, qe: RegisteredFrame, *,
                           minimum_area_percent: float = 0.3, maximum_area_percent: float = 50.0,
                           minimum_depth_agreement: float | None = None, depth_tolerance: float = 0.05,
                           minimum_shared_surface: float = 0.3, maximum_anchors: int = 3) -> tuple[list, list]:
    """Rendered anchor masks for objects visible at both endpoints."""
    renders = {}
    for role, frame in (('q0', q0), ('qe', qe)):
        h, w = frame.rgb.shape[:2]
        renders[role] = mesh.render(frame.K, w, h, frame.pose_c2w)
    ids = sorted(set(np.unique(renders['q0'][0])) & set(np.unique(renders['qe'][0])) - {0})
    anchors, rejected = [], []
    for oid in ids:
        label = mesh.labels.get(int(oid), '')
        row = {'object_id': int(oid), 'label': label}
        if mesh.excluded(oid):
            rejected.append({**row, 'reason': 'excluded structure or transparent object'})
            continue
        masks = {role: renders[role][0] == oid for role in renders}
        area = {role: 100.0 * float(m.mean()) for role, m in masks.items()}
        row['area_percent'] = area
        if not all(minimum_area_percent <= a <= maximum_area_percent for a in area.values()):
            rejected.append({**row, 'reason': 'endpoint area outside range'})
            continue
        agreement = {role: _depth_agreement(masks[role], renders[role][1], frame, depth_tolerance)
                     for role, frame in (('q0', q0), ('qe', qe))}
        row['depth_agreement'] = agreement
        if minimum_depth_agreement is not None and min(agreement.values()) < minimum_depth_agreement:
            rejected.append({**row, 'reason': 'mesh and sensor depth disagree'})
            continue
        tri_q0 = set(np.unique(renders['q0'][2][masks['q0']]).tolist())
        tri_qe = np.unique(renders['qe'][2][masks['qe']])
        shared = float(np.isin(tri_qe, list(tri_q0)).mean()) if len(tri_qe) else 0.0
        row['shared_surface'] = shared
        if shared < minimum_shared_surface:
            rejected.append({**row, 'reason': 'too little query surface visible at the input'})
            continue
        anchors.append({**row, 'q0': masks['q0'], 'qe': masks['qe'],
                        'score': min(area.values()) * shared, 'mask_source': 'scannetpp_semantic_mesh'})
    anchors.sort(key=lambda a: a['score'], reverse=True)
    for rank, anchor in enumerate(anchors[:maximum_anchors], 1):
        anchor['anchor_id'] = f'A{rank}'
    return anchors[:maximum_anchors], rejected
