#!/usr/bin/env python3
"""Factual five-frame scene captions with one resident Qwen process."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping

import numpy as np
from PIL import Image

from src.construction.vlm import LocalQwenProvider, VLMRequest


PROMPT_VERSION = "orbia.caption.v1"
FRAME_LABELS = ["Q0", "25%", "middle", "75%", "QE"]
WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*")
META_RE = re.compile(r"^(?:this\s+(?:image|video|frame|scene)|the\s+(?:image|video|frame)|in\s+this\s+(?:image|video|frame))\b", re.I)
IMAGE_PLANE_RE = re.compile(r"\b(?:on|to)\s+the\s+(?:left|right)\b|\b(?:foreground|background)\b", re.I)
INSTRUCTION = """Write one dense English scene caption from the five ordered frames.

Describe only persistent visible architecture, terrain, vegetation, water,
furniture, materials, lighting, weather, atmosphere, and stable spatial
relations supported across the frames. Do not mention people, animals, human
actions, moving vehicles, transient subjects, or any other temporary event. A
stationary parked vehicle may be mentioned only when it is clearly persistent.
Never mention the camera, viewpoint, framing, video, trajectory, temporal
change, future frames, generation, rendering, games, animation, media, or
these instructions. Do not use image-plane directions such as left, right,
centre, foreground, or background. Do not speculate beyond visible evidence.
Begin directly with scene language such as \"A ... scene where ...\"; do not
write \"This image/video...\". Use 80-150 English words.

Return exactly one JSON object and no markdown:
{"dense_caption":"..."}"""


def _repair_instruction(rejected_caption: str) -> str:
    """Ask once more for a caption that satisfies the same instructions."""
    return INSTRUCTION + f"""

The following rejected draft was too short or otherwise invalid:
<rejected_draft>{rejected_caption}</rejected_draft>

Rewrite it as exactly one JSON object. Keep only factual, visibly persistent
content from the five frames; add further factual persistent architectural,
material, terrain, vegetation, water, furniture, lighting, weather, atmosphere,
or stable spatial detail until the dense_caption contains 100-120 English
words. Do not invent details or add people, animals, actions, moving vehicles,
transient subjects, media terms, image-plane directions, or speculation.
Return exactly {{"dense_caption":"..."}} and no markdown."""


def _read_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, Mapping):
            raise ValueError(f"{path}:{line_number}: expected object")
        frames = value.get("source_frame_ids")
        if not isinstance(frames, Mapping) or set(frames) != {"q0", "q25", "mid", "q75", "qe"}:
            raise ValueError(f"{path}:{line_number}: expected five source_frame_ids")
        frame_paths = value.get("source_frame_paths")
        if frame_paths is not None and (
            not isinstance(frame_paths, Mapping) or set(frame_paths) != {"q0", "q25", "mid", "q75", "qe"}
        ):
            raise ValueError(f"{path}:{line_number}: expected five source_frame_paths")
        if not str(value.get("case_id", "")) or (frame_paths is None and not str(value.get("video_path", ""))):
            raise ValueError(f"{path}:{line_number}: missing case_id or frame source")
        rows.append(dict(value))
    return rows


def _decode_frames(row: Mapping[str, Any]) -> list[Image.Image]:
    frame_paths = row.get("source_frame_paths")
    if isinstance(frame_paths, Mapping):
        images: list[Image.Image] = []
        for role in ("q0", "q25", "mid", "q75", "qe"):
            with Image.open(Path(str(frame_paths[role]))) as image:
                images.append(image.convert("RGB").copy())
        return images

    import cv2

    path = Path(str(row["video_path"]))
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"cannot open source video: {path}")
    try:
        images: list[Image.Image] = []
        for role in ("q0", "q25", "mid", "q75", "qe"):
            frame_index = int(dict(row["source_frame_ids"])[role])
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, bgr = capture.read()
            if not ok or bgr is None:
                raise RuntimeError(f"{path}: cannot decode frame {frame_index}")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            images.append(Image.fromarray(np.asarray(rgb, dtype=np.uint8), mode="RGB"))
        return images
    finally:
        capture.release()


def _caption(raw: str, *, require_target_length: bool = True, reject_image_plane: bool = True) -> str:
    value = raw.strip()
    if value.startswith("```") and value.endswith("```"):
        value = value.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    payload = json.loads(value)
    if not isinstance(payload, Mapping) or set(payload) != {"dense_caption"}:
        raise ValueError("response must be exactly {'dense_caption': ...}")
    caption = " ".join(str(payload["dense_caption"]).split())
    word_count = len(WORD_RE.findall(caption))
    if require_target_length and not 80 <= word_count <= 150:
        raise ValueError(f"dense_caption must be 80-150 words, got {word_count}")
    if META_RE.match(caption):
        raise ValueError("dense_caption must not use image/video meta framing")
    if reject_image_plane and IMAGE_PLANE_RE.search(caption):
        raise ValueError("dense_caption must not use image-plane directions")
    return caption


def _strip_image_plane_phrases(caption: str) -> str:
    """Remove framing-only phrases without changing scene facts."""
    value = re.sub(r"\b(?:on|to)\s+the\s+(?:left|right)\s*,?\s*", "", caption, flags=re.I)
    value = re.sub(r"\b(?:in\s+the\s+)?(?:foreground|background)\s*,?\s*", "", value, flags=re.I)
    value = " ".join(value.split())
    if value:
        value = value[0].upper() + value[1:]
    if IMAGE_PLANE_RE.search(value):
        raise ValueError("image-plane phrase normalization was incomplete")
    return value


def _existing_ids(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    return {str(json.loads(line)["case_id"]) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def _requests(rows: Iterable[Mapping[str, Any]]) -> tuple[list[VLMRequest], list[Mapping[str, Any]]]:
    requests: list[VLMRequest] = []
    materialized: list[Mapping[str, Any]] = []
    for row in rows:
        requests.append(VLMRequest(images=_decode_frames(row), labels=FRAME_LABELS,
                                   max_output_tokens=420, instruction=INSTRUCTION))
        materialized.append(row)
    return requests, materialized


def _result(row: Mapping[str, Any], caption: str, model: str, normalization: str | None) -> dict[str, Any]:
    result = {
        "case_id": str(row["case_id"]), "dataset": str(row["dataset"]),
        "template": str(row["template"]), "source_frame_ids": dict(row["source_frame_ids"]),
        "dense_caption": caption, "model": model, "prompt_version": PROMPT_VERSION,
    }
    if normalization is not None:
        result["normalization"] = normalization
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--micro-batch-size", type=int, default=4)
    parser.add_argument("--attempts", type=int, default=4)
    parser.add_argument("--resume-shard-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.shard_count or args.micro_batch_size < 1 or args.attempts < 1:
        parser.error("invalid shard/batch/attempt values")
    rows = _read_rows(args.input_jsonl)
    selected = [row for index, row in enumerate(rows) if index % args.shard_count == args.shard_index]
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if args.output_jsonl.exists() and not args.resume:
        raise FileExistsError(f"refusing to overwrite {args.output_jsonl}; pass --resume")
    completed = _existing_ids(args.output_jsonl) if args.resume else set()
    if args.resume and args.resume_shard_dir:
        for path in args.resume_shard_dir.glob("*.jsonl"):
            completed.update(_existing_ids(path))
    pending = [row for row in selected if str(row["case_id"]) not in completed]
    if not pending:
        return 0
    provider = LocalQwenProvider(args.model)
    with args.output_jsonl.open("a", encoding="utf-8") as stream:
        for offset in range(0, len(pending), args.micro_batch_size):
            batch = pending[offset:offset + args.micro_batch_size]
            requests, source_rows = _requests(batch)
            generated = provider.generate_batch(requests)
            for row, request, generation in zip(source_rows, requests, generated, strict=True):
                caption: str | None = None
                normalization: str | None = None
                error: Exception | None = None
                text = generation.text
                for attempt in range(args.attempts):
                    if attempt:
                        text = provider.generate(
                            request.images, request.labels, request.max_output_tokens, _repair_instruction(text)
                        ).text
                    try:
                        caption = _caption(text)
                        break
                    except (ValueError, json.JSONDecodeError) as exc:
                        error = exc
                if caption is None:
                    # Length is a target, not a hard limit: keep the last
                    # well-formed caption; malformed replies still fail.
                    try:
                        caption = _caption(text, require_target_length=False)
                    except ValueError as exc:
                        if "image-plane directions" not in str(exc):
                            raise
                        caption = _strip_image_plane_phrases(
                            _caption(text, require_target_length=False, reject_image_plane=False)
                        )
                        normalization = "strip-image-plane-phrases-v1"
                    print(f"accepted short {row['case_id']}: {error}", flush=True)
                stream.write(json.dumps(_result(row, caption, args.model.name, normalization), ensure_ascii=False, separators=(",", ":")) + "\n")
                stream.flush()
                print(f"completed {row['case_id']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
