"""Read registered frames from the official SpatialVID release."""
from pathlib import Path
import tempfile
import zipfile

import numpy as np


def _group(group_id):
    value = str(group_id)
    value = value[len('group_'):] if value.startswith('group_') else value
    return f'group_{int(value):04d}'


def _c2w(pose):
    from scipy.spatial.transform import Rotation
    t, q = np.asarray(pose[:3], float), np.asarray(pose[3:], float)
    R = Rotation.from_quat(q/np.linalg.norm(q)).as_matrix()
    result = np.eye(4)
    result[:3, :3] = R.T
    result[:3, 3] = -R.T @ t
    return result


def _K(normalized, width, height):
    fx, fy, cx, cy = normalized
    return np.array([[fx*width, 0, cx*width], [0, fy*height, cy*height], [0, 0, 1.]])


def read_indexes(path):
    rows = [tuple(int(v) for v in line.split()) for line in Path(path).read_text().splitlines()
            if line.strip() and not line.startswith('#')]
    if any(len(r) != 2 for r in rows):
        raise ValueError(f'{path}: expected "pose_row source_frame" per line')
    return rows


def read_frames(root, video_id, group_id, frame_ids=None, depth_unit='source_pose_unit', fps=None):
    from src.construction.sources import RegisteredFrame, decode_video, read_depth
    import cv2
    root, group = Path(root), _group(group_id)
    annotations = root/'annotations'/group/video_id
    video = root/'videos'/group/f'{video_id}.mp4'
    indexes = read_indexes(annotations/'indexes.txt')
    poses = np.load(annotations/'poses.npy', allow_pickle=False)
    intrinsics = np.load(annotations/'intrinsics.npy', allow_pickle=False)
    if not len(indexes) == len(poses) == len(intrinsics):
        raise ValueError('indexes.txt, poses.npy and intrinsics.npy have different row counts')
    if frame_ids is not None:
        wanted = {int(str(i).removeprefix('pose_')) for i in frame_ids}
        indexes = [r for r in indexes if r[0] in wanted]
        if len(indexes) != len(wanted):
            raise ValueError('some frame_ids are not SpatialVID pose rows')
    if fps is None:
        capture = cv2.VideoCapture(str(video))
        fps = capture.get(cv2.CAP_PROP_FPS); capture.release()
    rgb = decode_video(video, [source for _, source in indexes])
    result = []
    with zipfile.ZipFile(root/'depths'/group/f'{video_id}.zip') as archive, \
            tempfile.TemporaryDirectory() as scratch:
        for row, source in indexes:
            target = Path(scratch)/f'{row:05d}.exr'
            target.write_bytes(archive.read(f'{row:05d}.exr'))
            inverse = read_depth(target, scratch)
            depth = np.zeros_like(inverse)
            valid = np.isfinite(inverse) & (inverse > 0)
            depth[valid] = 1/inverse[valid]
            image = rgb[source]
            h, w = image.shape[:2]; dh, dw = depth.shape
            result.append(RegisteredFrame(f'pose_{row:05d}', image, depth, valid, _c2w(poses[row]),
                _K(intrinsics[row], w, h), _K(intrinsics[row], dw, dh), depth_unit, source/fps))
    return result
