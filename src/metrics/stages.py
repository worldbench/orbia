"""Cached evaluation stages keyed by their inputs, configuration and artifacts."""
from __future__ import annotations
import json
import os
from pathlib import Path
import uuid


def file_identity(path):
    p = Path(path).resolve()
    st = p.stat()
    return {'path': str(p), 'size': st.st_size, 'mtime_ns': st.st_mtime_ns}


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        temp.write_text(json.dumps(value, indent=2, allow_nan=False, default=str)+'\n')
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def case_identity(root):
    root = Path(root)
    if (root/'case.json').is_file():
        paths = [p for p in root.rglob('*') if p.is_file()]
        ref = root.parent.parent/'references'/root.name
        paths.extend(p for p in ref.rglob('*') if p.is_file())
        return [file_identity(p) for p in sorted(paths)]
    return [file_identity(p) for d in ('public', 'evaluator_hidden')
            for p in sorted((root/d).rglob('*')) if p.is_file()]


class StageCache:
    def __init__(self, root):
        self.root = Path(root)

    def path(self, stage):
        return self.root / (stage+'.json')

    def ready(self, stage, identity, validator):
        try:
            receipt = json.loads(self.path(stage).read_text())
            return (receipt['identity'] == identity and receipt['status'] == 'complete'
                    and all(file_identity(p['path']) == p for p in receipt['artifacts'])
                    and bool(validator()))
        except (OSError, ValueError, KeyError, TypeError):
            return False

    def invalidate(self, stage):
        self.path(stage).unlink(missing_ok=True)

    def commit(self, stage, identity, artifacts):
        atomic_json(self.path(stage), {'status':'complete', 'identity':identity,
                    'artifacts':[file_identity(p) for p in artifacts]})


def validate_geometry(root, expected, frame_count):
    from src.metrics.pose import load_pose_cache
    from src.metrics.reconstruction import list_vipe_depth_indices
    root = Path(root)
    cache = load_pose_cache(root/'evaluator_pose_cache', video_frame_count=frame_count)
    return (cache is not None and list(cache.frame_indices) == list(expected)
            and cache.intrinsics is not None
            and set(list_vipe_depth_indices(root/'depth')) == set(expected))
