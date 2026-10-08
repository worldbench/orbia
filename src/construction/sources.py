"""Registered observations from images or source videos."""
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
from PIL import Image


@dataclass(frozen=True)
class RegisteredFrame:
    frame_id: str
    rgb: np.ndarray
    depth: np.ndarray | None
    valid: np.ndarray | None
    pose_c2w: np.ndarray | None
    K: np.ndarray | None
    depth_K: np.ndarray | None = None
    depth_unit: str = "metre"
    timestamp_s: float = 0.0


def path(value, base):
    p = Path(value).expanduser()
    return p if p.is_absolute() else Path(base) / p


def read_array(value, base):
    if isinstance(value, (list, tuple, np.ndarray)):
        return np.asarray(value, dtype=float)
    p = path(value, base)
    if p.suffix == '.json':
        return np.asarray(json.loads(p.read_text()), dtype=float)
    return np.load(p, allow_pickle=False)


def read_depth(value, base, scale=1):
    p = path(value, base)
    if p.suffix == '.npy':
        depth = np.load(p, allow_pickle=False)
    elif p.suffix.lower() == '.exr':
        import OpenImageIO as oiio
        image = oiio.ImageInput.open(str(p))
        if image is None:
            raise ValueError(f'cannot open depth EXR: {p}')
        try:
            depth = image.read_image()
            if depth.ndim == 3:
                depth = depth[..., 0]
        finally:
            image.close()
    else:
        depth = np.asarray(Image.open(p))
    return np.asarray(depth, dtype=np.float32) * float(scale)


def decode_video(video, indices):
    """Decode selected source frame numbers in order; never substitute timestamps."""
    import cv2
    wanted = set(int(i) for i in indices)
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ValueError(f'cannot decode {video}')
    result = {}
    try:
        for i in range(max(wanted) + 1):
            ok, frame = capture.read()
            if not ok:
                raise ValueError(f'missing source video frame {i}')
            if i in wanted:
                result[i] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    finally:
        capture.release()
    return result


def read_source(config, base):
    """Read registered frames, or decode ``frame_indices`` from a source video."""
    kind = config.get('kind', 'registered')
    if kind == 'scannetpp' and config.get('scene_root'):
        from src.construction.scannetpp import read_frames
        return read_frames(path(config['scene_root'], base), config.get('frame_names'),
                           stride=int(config.get('stride', 1)), depth_scale=float(config.get('depth_scale', 0.001)),
                           minimum_valid_depth=float(config.get('minimum_valid_depth', 0.0)))
    if kind == 'spatialvid' and config.get('root'):
        from src.construction.spatialvid import read_frames
        return read_frames(path(config['root'], base), config['video_id'], config['group_id'],
                           config.get('frame_ids'), config.get('depth_unit', 'source_pose_unit'), config.get('fps'))
    if kind not in {'registered', 'scannetpp', 'self_captured', 'spatialvid', 'sekai', 'mira', 'ue', 'video'}:
        raise ValueError(f'unknown source adapter: {kind}')
    video = config.get('video')
    rows = config.get('frames')
    if rows is None:
        if not video or not config.get('frame_indices'):
            raise ValueError('provide registered frames or video with frame_indices')
        rows = [{'id': str(i), 'source_frame': int(i),
                 'timestamp_s': int(i)/float(config.get('fps', 30))}
                for i in config['frame_indices']]
    decoded = decode_video(path(video, base), [r['source_frame'] for r in rows]) if video else {}
    result = []
    for row in rows:
        rgb = decoded[row['source_frame']] if video else np.asarray(Image.open(path(row['rgb'], base)).convert('RGB'))
        depth = read_depth(row['depth'], base, row.get('depth_scale', config.get('depth_scale', 1))) if row.get('depth') else None
        valid = np.asarray(Image.open(path(row['valid'], base)).convert('L')) > 0 if row.get('valid') else None
        if depth is not None:
            valid = (np.isfinite(depth) & (depth > 0)) if valid is None else valid & np.isfinite(depth) & (depth > 0)
        K = read_array(row['intrinsics'], base) if row.get('intrinsics') is not None else None
        dK = read_array(row['depth_intrinsics'], base) if row.get('depth_intrinsics') is not None else None
        if depth is not None and dK is None and K is not None:
            dK = np.diag([depth.shape[1]/rgb.shape[1], depth.shape[0]/rgb.shape[0], 1]) @ K
        result.append(RegisteredFrame(str(row['id']), rgb, depth, valid,
            read_array(row['pose_c2w'], base) if row.get('pose_c2w') is not None else None, K, dK,
            row.get('depth_unit', config.get('depth_unit', 'metre')),
            float(row.get('timestamp_s', 0))))
    return result
