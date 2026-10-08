"""Generate benchmark videos with a :class:`~src.models.base.WorldModel`."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Iterable

import numpy as np

from src.models.base import WorldModel
from src.models.case import BENCHMARK_FPS, list_case_ids, load_model_case


def write_video(frames, path: str | Path, fps: int = BENCHMARK_FPS) -> Path:
    """Write RGB frames to an MP4 file (H.264 through imageio-ffmpeg if available, else OpenCV)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = [np.asarray(f, dtype=np.uint8) for f in frames]
    height, width = frames[0].shape[:2]
    if any(f.shape[:2] != (height, width) for f in frames):
        raise ValueError('all frames must share one resolution')
    tmp = path.with_name(path.stem + '.partial' + path.suffix)
    try:
        import imageio.v2 as imageio
        with imageio.get_writer(tmp, fps=fps, codec='libx264', quality=9,
                                macro_block_size=1, pixelformat='yuv420p') as writer:
            for frame in frames:
                writer.append_data(frame)
    except ImportError:
        import cv2
        writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        for frame in frames:
            writer.write(frame[:, :, ::-1])
        writer.release()
    tmp.replace(path)
    return path


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def _update_manifest(manifest: Path, rows: list[dict]) -> None:
    existing = {}
    if manifest.is_file():
        for line in manifest.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                existing[(row['model'], row['case_id'])] = row
    for row in rows:
        existing[(row['model'], row['case_id'])] = row
    manifest.write_text(''.join(json.dumps(r) + '\n' for _, r in sorted(existing.items())))


def generate(model: WorldModel, dataset: str | Path, output: str | Path, *,
             case_ids: Iterable[str] | None = None, horizon: str | None = None,
             shard: int = 0, num_shards: int = 1, overwrite: bool = False,
             model_name: str | None = None) -> dict:
    """Generate every selected case and write the evaluation manifest."""
    dataset, output = Path(dataset).resolve(), Path(output).resolve()
    name = model_name or getattr(model, 'name', 'world-model')
    ids = list(case_ids) if case_ids else list_case_ids(dataset, horizon)
    ids = ids[shard::num_shards]
    model.setup()
    rows, failures = [], []
    for case_id in ids:
        video = output / 'videos' / f'{case_id}.mp4'
        row = {'model': name, 'case_id': case_id, 'video': f'videos/{case_id}.mp4',
               'group': model.group, 'control_profile': model.control_profile}
        if video.is_file() and not overwrite:
            rows.append(row)
            continue
        case = load_model_case(dataset, case_id)
        started = time.time()
        try:
            frames, condition = model.generate(case)
        except Exception as error:  # keep going; failures are reported at the end
            failures.append({'case_id': case_id, 'error': repr(error)})
            print(f'[{name}] {case_id} failed: {error!r}', flush=True)
            continue
        if len(frames) != case.num_frames:
            raise RuntimeError(f'{case_id}: {len(frames)} frames, expected {case.num_frames}')
        write_video(frames, video)
        record = {'model': name, 'case_id': case_id, 'num_frames': case.num_frames, 'fps': BENCHMARK_FPS,
                  'resolution': list(frames[0].shape[1::-1]), 'seconds': round(time.time() - started, 2),
                  'condition': condition}
        (output / 'conditions').mkdir(parents=True, exist_ok=True)
        (output / 'conditions' / f'{case_id}.json').write_text(
            json.dumps(record, indent=2, default=_json_default) + '\n')
        rows.append(row)
        print(f'[{name}] {case_id} done ({record["seconds"]} s)', flush=True)
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / ('videos.jsonl' if num_shards == 1 else f'videos.shard{shard}.jsonl')
    _update_manifest(manifest, rows)
    return {'model': name, 'cases': len(ids), 'written': len(rows), 'failed': failures,
            'manifest': str(manifest)}


def merge_manifests(output: str | Path) -> Path:
    """Merge ``videos.shard*.jsonl`` into ``videos.jsonl``."""
    output = Path(output)
    rows = []
    for shard in sorted(output.glob('videos.shard*.jsonl')):
        rows += [json.loads(line) for line in shard.read_text().splitlines() if line.strip()]
    target = output / 'videos.jsonl'
    _update_manifest(target, rows)
    return target
