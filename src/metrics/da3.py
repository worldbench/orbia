"""Depth Anything 3 held-out rendering adapter for Tier 4.1."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import subprocess
from typing import Any, Mapping

import numpy as np
from PIL import Image

from src.metrics.contract import MetricResult
from src.metrics.features import ImageComparator, masked_psnr
from src.metrics.io import write_json


@dataclass(frozen=True)
class DA3Adapter:
    python: Path
    bridge_script: Path
    repo: Path
    checkpoint: Path
    device: str = "cuda"
    timeout_s: int = 1800
    backend: str = "da3"


def context_heldout_indices(
    frame_count: int, *, context_count: int = 64, heldout_count: int = 16, candidate_indices=None,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Uniform context views plus a fixed, disjoint held-out set."""

    if candidate_indices is not None:
        candidates = tuple(int(i) for i in candidate_indices)
        if (not candidates or list(candidates) != sorted(set(candidates))
                or candidates[0] < 0 or candidates[-1] >= frame_count):
            raise ValueError("invalid sampled GS candidate indices")
        context, heldout = context_heldout_indices(len(candidates),
            context_count=context_count, heldout_count=heldout_count)
        return tuple(candidates[i] for i in context), tuple(candidates[i] for i in heldout)

    if context_count < 2:
        raise ValueError("context_count must be at least two")
    if heldout_count < 1:
        raise ValueError("heldout_count must be positive")
    if frame_count < context_count + heldout_count:
        raise ValueError(
            f"{frame_count} frames cannot provide {context_count} context and "
            f"{heldout_count} disjoint held-out views"
        )
    heldout = tuple(
        int(round((index + 0.5) * (frame_count - 1) / heldout_count))
        for index in range(heldout_count)
    )
    available = tuple(index for index in range(frame_count) if index not in set(heldout))
    context = tuple(
        available[int(round(index * (len(available) - 1) / (context_count - 1)))]
        for index in range(context_count)
    )
    if len(set(context)) != context_count or len(set(heldout)) != heldout_count:
        raise ValueError("uniform DA3 sampling produced duplicate frames")
    if set(context) & set(heldout):
        raise ValueError("DA3 context and held-out frames must be disjoint")
    return context, heldout


def _summary(values: list[float | None]) -> dict[str, float | int | None]:
    finite = [float(value) for value in values if value is not None and math.isfinite(value)]
    return {
        "mean": float(np.mean(finite)) if finite else None,
        "median": float(np.median(finite)) if finite else None,
        "count": len(finite),
    }


def _prepare_frame(image: np.ndarray, output: Path, resolution: int) -> tuple[int, int]:
    rgb = np.asarray(image, dtype=np.uint8)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("DA3 frames must be HxWx3 RGB arrays")
    height, width = rgb.shape[:2]
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).resize(
        (resolution, resolution), Image.Resampling.BILINEAR
    ).save(output)
    return width, height


def _scaled_intrinsics(
    value: np.ndarray, width: int, height: int, resolution: int,
) -> list[list[float]]:
    intrinsic = np.asarray(value, dtype=np.float64).copy()
    if intrinsic.shape != (3, 3) or not np.isfinite(intrinsic).all():
        raise ValueError("DA3 intrinsics must be a finite 3x3 matrix")
    intrinsic[0] *= resolution / width
    intrinsic[1] *= resolution / height
    return intrinsic.tolist()


def prepare_da3_request(
    *,
    frames: Mapping[int, np.ndarray],
    poses_c2w: Mapping[int, np.ndarray],
    intrinsics: Mapping[int, np.ndarray],
    frame_count: int,
    output_root: Path,
    context_count: int = 64,
    heldout_count: int = 16,
    resolution: int = 504,
    candidate_indices=None,
    context_images: dict[int, np.ndarray] | None = None,
) -> Path:
    """Write a context-only DA3 reconstruction request plus held-out targets."""

    if resolution < 14 or resolution % 14:
        raise ValueError("DA3 resolution must be a positive multiple of 14")
    context, heldout = context_heldout_indices(
        frame_count, context_count=context_count, heldout_count=heldout_count,
        candidate_indices=candidate_indices,
    )
    selected = (*context, *heldout)
    missing = [
        index for index in selected
        if index not in frames or index not in poses_c2w or index not in intrinsics
    ]
    if missing:
        raise ValueError(
            "DA3 inputs are missing selected frames/poses/intrinsics: " f"{missing}"
        )

    work = Path(output_root)
    inputs = work / "inputs"
    renders = work / "renders"
    records: dict[int, dict[str, Any]] = {}
    for index in selected:
        image_path = inputs / f"frame_{index:06d}.png"
        if context_images is not None and index in context:
            rgb = np.asarray(frames[index], dtype=np.uint8)
            if rgb.ndim != 3 or rgb.shape[2] != 3:
                raise ValueError("DA3 frames must be HxWx3 RGB arrays")
            height, width = rgb.shape[:2]
            context_images[index] = np.asarray(Image.fromarray(rgb).resize(
                (resolution, resolution), Image.Resampling.BILINEAR))
        else:
            width, height = _prepare_frame(frames[index], image_path, resolution)
        pose = np.asarray(poses_c2w[index], dtype=np.float64)
        if pose.shape != (4, 4) or not np.isfinite(pose).all():
            raise ValueError(f"frame {index} has an invalid c2w pose")
        records[index] = {
            "frame_index": index,
            "rgb": None if context_images is not None and index in context else str(image_path.resolve()),
            "extrinsics_w2c": np.linalg.inv(pose).tolist(),
            "intrinsics": _scaled_intrinsics(
                intrinsics[index], width, height, resolution
            ),
        }

    request_path = work / "request.json"
    write_json(
        request_path,
        {
            "format": "orbia.da3.request.v1",
            "sampling_policy": "existing_geometry_frames" if candidate_indices is not None else "full_video_uniform",
            "context": [records[index] for index in context],
            "heldout": [records[index] for index in heldout],
            "resolution": resolution,
            "process_resolution": resolution,
            "render_root": str(renders.resolve()),
            "reconstruction_inputs": "context_rgb_only",
            "align_to_input_ext_scale": True,
            "infer_gs": True,
            "no_post_optimization": True,
        },
    )
    return request_path


def _read_da3_artifact(work: Path):
    """Resolve portable archived image paths against the artifact directory."""
    request = json.loads((work / "request.json").read_text(encoding="utf-8"))
    result = json.loads((work / "bridge_result.json").read_text(encoding="utf-8"))
    for record in [*request.get("context", []), *request.get("heldout", []), *result.get("renders", [])]:
        for key in ("rgb", "depth", "alpha"):
            if record.get(key) and not Path(record[key]).is_absolute():
                record[key] = str(work / record[key])
    return request, result


def da3_artifact_complete(
    output_root: Path, *, context_count: int | None = None,
    heldout_count: int | None = None,
    expected_context=None, expected_heldout=None,
) -> bool:
    """Check the request/result files and all expected held-out renders."""

    work = Path(output_root)
    try:
        request, result = _read_da3_artifact(work)
        if (
            request.get("format") != "orbia.da3.request.v1"
            or request.get("reconstruction_inputs") != "context_rgb_only"
            or request.get("align_to_input_ext_scale") is not True
            or request.get("infer_gs") is not True
            or request.get("no_post_optimization") is not True
            or result.get("status") != "ok"
        ):
            return False
        context = request.get("context")
        heldout = request.get("heldout")
        renders = result.get("renders")
        if not all(isinstance(value, list) for value in (context, heldout, renders)):
            return False
        if len(context) < 2 or len(heldout) < 1:
            return False
        if context_count is not None and len(context) != context_count:
            return False
        if heldout_count is not None and len(heldout) != heldout_count:
            return False
        context_indices = [int(item["frame_index"]) for item in context]
        heldout_indices = [int(item["frame_index"]) for item in heldout]
        if (
            len(set(context_indices)) != len(context_indices)
            or len(set(heldout_indices)) != len(heldout_indices)
            or set(context_indices) & set(heldout_indices)
        ):
            return False
        if expected_context is not None and context_indices != list(expected_context):
            return False
        if expected_heldout is not None and heldout_indices != list(expected_heldout):
            return False
        actual = [int(item["frame_index"]) for item in renders]
        if len(set(actual)) != len(actual) or set(actual) != set(heldout_indices):
            return False
        for item in heldout:
            target = Path(str(item["rgb"]))
            if not target.is_file() or target.stat().st_size == 0:
                return False
        for item in renders:
            paths = [Path(str(item[key])) for key in ("rgb", "depth")]
            if item.get("alpha") is not None:
                paths.append(Path(str(item["alpha"])))
            if not all(path.is_file() and path.stat().st_size > 0 for path in paths):
                return False
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return False
    return True


def sanitized_da3_environment(
    base: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Isolate the DA3 interpreter from evaluator and ViPE packages."""

    environment = dict(os.environ if base is None else base)
    environment.pop("PYTHONPATH", None)
    environment["PYTHONNOUSERSITE"] = "1"
    environment["HF_HUB_OFFLINE"] = "1"
    environment["TRANSFORMERS_OFFLINE"] = "1"
    return environment


def run_da3_bridge(
    *, request_path: Path, output_root: Path, adapter: DA3Adapter,
) -> None:
    result_path = Path(output_root) / "bridge_result.json"
    completed = subprocess.run(
        [
            str(adapter.python), str(adapter.bridge_script),
            "--request", str(request_path), "--output", str(result_path),
            "--repo", str(adapter.repo), "--checkpoint", str(adapter.checkpoint),
            "--device", adapter.device, "--backend", adapter.backend,
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=adapter.timeout_s,
        check=False,
        env=sanitized_da3_environment(),
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"DA3 bridge failed ({completed.returncode}): "
            f"{completed.stderr.strip()[-2000:]}"
        )
    if not da3_artifact_complete(output_root):
        raise RuntimeError("DA3 bridge returned an incomplete render artifact")


def score_da3_artifact(
    *,
    output_root: Path,
    comparator: ImageComparator,
    adapter: DA3Adapter | None = None,
) -> MetricResult:
    """Score held-out DA3 Gaussian renders without rerunning reconstruction."""

    work = Path(output_root)
    if not da3_artifact_complete(work):
        raise RuntimeError(f"incomplete precomputed DA3 artifact: {work}")
    request, result = _read_da3_artifact(work)
    context = tuple(int(item["frame_index"]) for item in request["context"])
    heldout = tuple(int(item["frame_index"]) for item in request["heldout"])
    targets = {int(item["frame_index"]): item for item in request["heldout"]}
    renders = {int(item["frame_index"]): item for item in result["renders"]}

    loaded: list[tuple[int, dict[str, Any], np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]] = []
    for index in heldout:
        record = renders[index]
        with Image.open(targets[index]["rgb"]) as image:
            target = np.asarray(image.convert("RGB"))
        with Image.open(record["rgb"]) as image:
            rendered = np.asarray(image.convert("RGB"))
        depth = np.asarray(np.load(record["depth"], allow_pickle=False), dtype=np.float32).squeeze()
        alpha = (
            np.asarray(np.load(record["alpha"], allow_pickle=False), dtype=np.float32).squeeze()
            if record.get("alpha") is not None else None
        )
        if (
            target.shape != rendered.shape
            or depth.shape != target.shape[:2]
            or (alpha is not None and alpha.shape != target.shape[:2])
        ):
            raise ValueError(f"DA3 render {index} has an unexpected shape")
        valid = np.isfinite(depth) & (depth > 0)
        loaded.append((index, record, target, rendered, valid, alpha))

    comparisons = comparator.compare_batch(
        [item[2] for item in loaded],
        [item[3] for item in loaded],
        masks=[item[4] for item in loaded],
        batch_size=16,
    )
    per_frame: list[dict[str, Any]] = []
    for (index, record, target, rendered, valid, alpha), comparison in zip(loaded, comparisons):
        frame_result = {
            "frame_index": index,
            "psnr": masked_psnr(target, rendered, valid),
            "ssim": comparison["ssim"],
            "lpips": comparison["lpips"],
            "dino_similarity": comparison["global_feature_similarity"],
            "rendered_pixel_coverage": float(valid.mean()),
            "depth": record["depth"],
        }
        if alpha is not None:
            finite_alpha = np.isfinite(alpha)
            frame_result["mean_alpha"] = (
                float(np.mean(alpha[finite_alpha])) if np.any(finite_alpha) else None
            )
        per_frame.append(frame_result)

    names: tuple[str, ...] = (
        "psnr", "ssim", "lpips", "dino_similarity",
        "rendered_pixel_coverage",
    )
    if all("mean_alpha" in item for item in per_frame):
        names += ("mean_alpha",)
    raw = {name: _summary([item[name] for item in per_frame]) for name in names}
    raw["per_frame"] = per_frame
    covered = bool(per_frame and (raw["rendered_pixel_coverage"]["mean"] or 0) > 0)
    adapter_provenance = {} if adapter is None else {
        "backend": adapter.backend,
        "repo": str(adapter.repo),
        "checkpoint": str(adapter.checkpoint),
        "device": adapter.device,
    }
    return MetricResult(
        name="da3_reconstruction",
        status="ok" if covered else "not_covered",
        applicable=True,
        version="v0.1",
        coverage={
            "context_frame_count": len(context),
            "heldout_frame_count": len(heldout),
            "evaluated_frame_count": len(per_frame),
            "context_indices": list(context),
            "heldout_indices": list(heldout),
        },
        raw=raw,
        failure=None if covered else "DA3 returned no covered held-out pixels",
        provenance={
            **adapter_provenance,
            "resolution": int(request["resolution"]),
            "reconstruction_inputs": "context_rgb_only",
            "align_to_input_ext_scale": True,
            "post_optimization": False,
            **dict(result.get("provenance", {})),
            **comparator.provenance,
        },
    )


def run_da3_metric(
    *,
    frames: Mapping[int, np.ndarray],
    poses_c2w: Mapping[int, np.ndarray],
    intrinsics: Mapping[int, np.ndarray],
    frame_count: int,
    output_root: Path,
    adapter: DA3Adapter,
    comparator: ImageComparator,
    context_count: int = 64,
    heldout_count: int = 16,
    resolution: int = 504,
    candidate_indices=None,
) -> MetricResult:
    request_path = prepare_da3_request(
        frames=frames,
        poses_c2w=poses_c2w,
        intrinsics=intrinsics,
        frame_count=frame_count,
        output_root=output_root,
        context_count=context_count,
        heldout_count=heldout_count,
        resolution=resolution,
        candidate_indices=candidate_indices,
    )
    run_da3_bridge(
        request_path=request_path, output_root=output_root, adapter=adapter
    )
    return score_da3_artifact(
        output_root=output_root, comparator=comparator, adapter=adapter
    )
