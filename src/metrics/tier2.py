"""Learned Tier-2 visual-quality metrics: MUSIQ, LAION aesthetic and AMT smoothness."""

from __future__ import annotations

import importlib
from functools import lru_cache
import sys
from time import perf_counter
from typing import Iterable, Mapping

import numpy as np

from src.metrics.contract import EvaluationOptions, MetricResult
from src.metrics.registry import TIER2_METRICS


OFFICIAL_TIER2_METRICS = TIER2_METRICS

__all__ = [
    "OFFICIAL_TIER2_METRICS",
    "imaging_quality_metric",
    "aesthetic_quality_metric",
    "motion_smoothness_metric",
]


def _summary(values: Iterable[float]) -> dict[str, float | None]:
    values = np.asarray(list(values), dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"mean": None, "median": None, "p05": None, "p95": None}
    return {"mean": float(values.mean()), "median": float(np.median(values)), "p05": float(np.percentile(values, 5)), "p95": float(np.percentile(values, 95))}


def _unavailable(name: str, reason: str) -> MetricResult:
    return MetricResult(name=name, status="not_available", applicable=True, failure=reason)


def _not_covered(name: str, reason: str) -> MetricResult:
    return MetricResult(name=name, status="not_covered", applicable=True, failure=reason)


@lru_cache(maxsize=4)
def _musiq_model(checkpoint: str, device: str):
    from pyiqa.archs.musiq_arch import MUSIQ

    return MUSIQ(pretrained_model_path=checkpoint).to(device).eval()


def _musiq_scores(
    frames: Mapping[int, np.ndarray], options: EvaluationOptions,
) -> tuple[list[float], dict[str, float]]:
    import torch
    import torch.nn.functional as F

    resolve_started = perf_counter()
    model = _musiq_model(str(options.musiq_checkpoint), options.device)
    resolve_s = perf_counter() - resolve_started
    inference_started = perf_counter()
    with torch.no_grad():
        tensors = []
        for frame in frames.values():
            tensor = torch.from_numpy(frame.copy()).permute(2, 0, 1).unsqueeze(0).float().to(options.device) / 255.0
            height, width = tensor.shape[-2:]
            if max(height, width) > 512:  # VBench's default `longer` preprocessing mode.
                scale = 512.0 / max(height, width)
                tensor = F.interpolate(
                    tensor,
                    size=(round(height * scale), round(width * scale)),
                    mode="bilinear",
                    align_corners=False,
                )
            tensors.append(tensor)
        scores: list[float] = []
        batch_size = max(int(options.tier2_musiq_batch_size), 1)
        for start in range(0, len(tensors), batch_size):
            batch = torch.cat(tensors[start:start + batch_size], dim=0)
            values = np.asarray(
                model(batch).detach().float().cpu(), dtype=np.float64
            ).reshape(-1)
            if len(values) != len(batch):
                raise ValueError("MUSIQ returned an unexpected batch shape")
            scores.extend(float(value) / 100.0 for value in values)
    return scores, {
        "model_resolve": resolve_s,
        "inference": perf_counter() - inference_started,
    }


def imaging_quality_metric(frames: Mapping[int, np.ndarray], options: EvaluationOptions) -> MetricResult:
    """VBench MUSIQ frame-quality diagnostic, with no implicit download."""
    if not frames:
        return _not_covered("imaging_quality", "no quality frames were decoded")
    if options.musiq_checkpoint is None or not options.musiq_checkpoint.is_file():
        return _unavailable("imaging_quality", "configure an existing musiq_checkpoint; evaluator never downloads weights")
    try:
        payload = _musiq_scores(frames, options)
        scores, timing = payload if isinstance(payload, tuple) else (payload, {})
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        return _unavailable("imaging_quality", f"MUSIQ unavailable: {type(exc).__name__}: {exc}")
    return MetricResult(name="imaging_quality", status="ok", applicable=True, coverage={"evaluated_frame_count": len(scores)}, raw={"musiq": _summary(scores), "per_frame": scores}, provenance={"backend": "VBench MUSIQ", "checkpoint": str(options.musiq_checkpoint), "device": options.device, "batch_size": max(int(options.tier2_musiq_batch_size), 1), "timing_s": timing})


@lru_cache(maxsize=4)
def _aesthetic_models(clip_checkpoint: str, head_checkpoint: str, device: str):
    import clip
    import torch
    import torch.nn as nn

    clip_model, preprocess = clip.load(clip_checkpoint, device=device)
    head = nn.Linear(768, 1)
    state = torch.load(head_checkpoint, map_location="cpu", weights_only=True)
    head.load_state_dict(state if "weight" in state else state["state_dict"])
    head.to(device).eval()
    return clip_model, preprocess, head


def _aesthetic_scores(
    frames: Mapping[int, np.ndarray], options: EvaluationOptions,
) -> tuple[list[float], dict[str, float]]:
    import torch
    import torch.nn.functional as F
    from PIL import Image

    resolve_started = perf_counter()
    clip_model, preprocess, head = _aesthetic_models(
        str(options.aesthetic_clip_checkpoint),
        str(options.aesthetic_checkpoint),
        options.device,
    )
    resolve_s = perf_counter() - resolve_started
    inference_started = perf_counter()
    with torch.no_grad():
        tensors = [preprocess(Image.fromarray(frame)) for frame in frames.values()]
        scores: list[float] = []
        batch_size = max(int(options.tier2_aesthetic_batch_size), 1)
        for start in range(0, len(tensors), batch_size):
            batch = torch.stack(tensors[start:start + batch_size]).to(options.device)
            features = F.normalize(
                clip_model.encode_image(batch).float(), dim=-1,
            )
            values = np.asarray(
                head(features).detach().float().cpu(), dtype=np.float64
            ).reshape(-1)
            scores.extend(float(value) / 10.0 for value in values)
    return scores, {
        "model_resolve": resolve_s,
        "inference": perf_counter() - inference_started,
    }


def aesthetic_quality_metric(frames: Mapping[int, np.ndarray], options: EvaluationOptions) -> MetricResult:
    """VBench CLIP ViT-L/14 plus LAION linear-aesthetic-head diagnostic."""
    if not frames:
        return _not_covered("aesthetic_quality", "no quality frames were decoded")
    if any(item is None for item in (options.aesthetic_clip_checkpoint, options.aesthetic_checkpoint)):
        return _unavailable("aesthetic_quality", "configure aesthetic_clip_checkpoint and aesthetic_checkpoint")
    assert options.aesthetic_clip_checkpoint is not None and options.aesthetic_checkpoint is not None
    if not options.aesthetic_clip_checkpoint.is_file() or not options.aesthetic_checkpoint.is_file():
        return _unavailable("aesthetic_quality", "configured CLIP or LAION-aesthetic checkpoint is missing")
    try:
        payload = _aesthetic_scores(frames, options)
        scores, timing = payload if isinstance(payload, tuple) else (payload, {})
    except (ImportError, OSError, RuntimeError, ValueError, KeyError) as exc:
        return _unavailable("aesthetic_quality", f"aesthetic backend unavailable: {type(exc).__name__}: {exc}")
    return MetricResult(name="aesthetic_quality", status="ok", applicable=True, coverage={"evaluated_frame_count": len(scores)}, raw={"laion_aesthetic": _summary(scores), "per_frame": scores}, provenance={"backend": "VBench CLIP ViT-L/14 + LAION aesthetic head", "clip_checkpoint": str(options.aesthetic_clip_checkpoint), "aesthetic_checkpoint": str(options.aesthetic_checkpoint), "device": options.device, "batch_size": max(int(options.tier2_aesthetic_batch_size), 1), "timing_s": timing})


@lru_cache(maxsize=4)
def _amt_model(vbench_root: str, config: str, checkpoint: str, device: str):
    sys.path.insert(0, vbench_root)
    try:
        module = importlib.import_module("vbench.motion_smoothness")
        return module.MotionSmoothness(config, checkpoint, device)
    finally:
        if vbench_root in sys.path:
            sys.path.remove(vbench_root)


def _amt_triplets(frame_count: int, stride: int = 2) -> list[tuple[int, int, int]]:
    """Sample triplet starts; each interpolation still spans two source frames."""
    if isinstance(stride, bool) or not isinstance(stride, int) or stride < 1:
        raise ValueError("AMT triplet stride must be a positive integer")
    return [(start, start + 1, start + 2) for start in range(0, frame_count - 2, stride)]


def _amt_score(
    video_path: str, options: EvaluationOptions,
) -> tuple[float, dict[str, object]]:
    import torch

    assert options.vbench_root is not None
    assert options.amt_config is not None
    assert options.amt_checkpoint is not None
    resolve_started = perf_counter()
    motion = _amt_model(
        str(options.vbench_root), str(options.amt_config),
        str(options.amt_checkpoint), options.device,
    )
    from vbench.third_party.amt.utils.utils import (
        InputPadder,
        check_dim_and_resize,
        img2tensor,
    )
    resolve_s = perf_counter() - resolve_started
    inference_started = perf_counter()
    from src.metrics.video import cached_video_frames
    frames = cached_video_frames(video_path)
    if frames is None:
        frames = motion.fp.get_frames(video_path)
    triplets = _amt_triplets(len(frames), options.tier2_amt_triplet_stride)
    if not triplets:
        raise ValueError("AMT requires at least three source frames")
    endpoint_indices = sorted({i for first, _, last in triplets for i in (first, last)})
    inputs = check_dim_and_resize([img2tensor(frames[i]) for i in endpoint_indices])
    endpoint_inputs = dict(zip(endpoint_indices, inputs))
    height, width = inputs[0].shape[-2:]
    scale = motion.anchor_resolution / (height * width) * np.sqrt(
        (motion.vram_avail - motion.anchor_memory_bias) / motion.anchor_memory
    )
    scale = min(float(scale), 1.0)
    scale = 1 / np.floor(1 / np.sqrt(scale) * 16) * 16
    padder = InputPadder(inputs[0].shape, int(16 / scale))
    batch_size = max(int(options.tier2_amt_batch_size), 1)
    differences: list[float] = []
    with torch.no_grad():
        for start in range(0, len(triplets), batch_size):
            batch = triplets[start:start + batch_size]
            first = torch.cat([endpoint_inputs[a] for a, _, _ in batch], dim=0).to(options.device)
            second = torch.cat([endpoint_inputs[c] for _, _, c in batch], dim=0).to(options.device)
            first, second = padder.pad(first, second)
            embt = motion.embt.expand(len(first), -1, -1, -1)
            predicted = motion.model(
                first, second, embt, scale_factor=scale, eval=True,
            )["imgt_pred"]
            predicted = padder.unpad(predicted)
            predicted_u8 = (
                predicted.mul(255.0).permute(0, 2, 3, 1).detach().cpu().numpy()
                .clip(0, 255).astype(np.uint8)
            )
            target_u8 = np.stack(
                [frames[b] for _, b, _ in batch]
            )
            per_pair = np.abs(
                target_u8.astype(np.int16) - predicted_u8.astype(np.int16)
            ).mean(axis=(1, 2, 3))
            differences.extend(float(value) for value in per_pair)
    score = (255.0 - float(np.mean(differences))) / 255.0
    return score, {
        "model_resolve": resolve_s,
        "inference": perf_counter() - inference_started,
        "evaluated_pair_count": len(differences),
        "triplet_sampling": {
            "source_frame_count": len(frames),
            "triplet_start_stride": options.tier2_amt_triplet_stride,
            "endpoint_interval_frames": 2,
            "tail_policy": "regular_stride_grid_no_extra_tail",
            "evaluated_triplet_count": len(differences),
            "triplet_frame_indices": [list(t) for t in triplets],
        },
    }


def motion_smoothness_metric(video_path: str, options: EvaluationOptions) -> MetricResult:
    """Reuse VBench AMT interpolation residual without copying its runtime."""
    required = (options.vbench_root, options.amt_config, options.amt_checkpoint)
    if any(item is None for item in required):
        return _unavailable("motion_smoothness", "configure vbench_root, amt_config, and amt_checkpoint for VBench AMT")
    assert options.vbench_root is not None and options.amt_config is not None and options.amt_checkpoint is not None
    if not options.vbench_root.is_dir() or not options.amt_config.is_file() or not options.amt_checkpoint.is_file():
        return _unavailable("motion_smoothness", "configured VBench AMT source/config/checkpoint is missing")
    try:
        payload = _amt_score(video_path, options)
        score, timing = payload if isinstance(payload, tuple) else (payload, {})
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        return _unavailable("motion_smoothness", f"AMT unavailable: {type(exc).__name__}: {exc}")
    timing = dict(timing)
    sampling = timing.pop("triplet_sampling", None)
    coverage = {"evaluated_video_count": 1, "source_frame_stride": 2}
    raw = {"amt_interpolation_score": score}
    if sampling is not None:
        raw["triplet_sampling"] = sampling
        coverage.update(evaluated_triplet_count=sampling["evaluated_triplet_count"],
                        triplet_start_stride=options.tier2_amt_triplet_stride)
    return MetricResult(name="motion_smoothness", status="ok", applicable=True,
        coverage=coverage, raw=raw, provenance={
            "backend": "VBench AMT-S interpolation residual", "vbench_root": str(options.vbench_root),
            "config": str(options.amt_config), "checkpoint": str(options.amt_checkpoint),
            "device": options.device, "batch_size": max(int(options.tier2_amt_batch_size), 1),
            "triplet_start_stride": options.tier2_amt_triplet_stride, "endpoint_interval_frames": 2,
            "tail_policy": "regular_stride_grid_no_extra_tail", "timing_s": timing})
