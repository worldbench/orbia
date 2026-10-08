"""Local Qwen generation for landmark proposals and scene captions."""


from __future__ import annotations


from dataclasses import dataclass






import json


from pathlib import Path


from typing import Any, Mapping, Sequence


from PIL import Image


LOCAL_QWEN_PROVIDER = "local_qwen"


@dataclass(frozen=True)
class VLMGeneration:
    """Raw provider output plus non-secret request provenance."""

    text: str
    provider: str
    model_id: str
    attempts: int = 1
    request_id: str | None = None
    usage: Mapping[str, Any] | None = None
    cached: bool = False


@dataclass(frozen=True)
class VLMRequest:
    """One independently cacheable multimodal request."""

    images: Sequence[Image.Image]
    labels: Sequence[str]
    max_output_tokens: int
    instruction: str


def temporal_role_text(labels: Sequence[str], instruction: str) -> str:
    """Build the one shared text block used by local and API providers."""

    role_map = "; ".join(
        f"Image {index} = {label.upper()}"
        for index, label in enumerate(labels, start=1)
    )
    return f"Images are in temporal order: {role_map}.\n\n{instruction}"


def _model_loader_family(path: Path) -> str:
    """Choose the Transformers auto class without importing model code."""

    config_path = path / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"model config does not exist: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    return "multimodal_lm" if config.get("model_type") == "qwen3_5" else "image_text_to_text"


def _chat_template_kwargs(model: Any) -> dict[str, Any]:
    """Keep JSON classification deterministic on default-thinking Qwen3.5/3.6."""

    config = getattr(model, "config", None)
    return {"enable_thinking": False} if getattr(config, "model_type", None) == "qwen3_5" else {}


def _load_model(path: Path):
    """Load a local Qwen checkpoint and its processor."""

    import torch
    from transformers import AutoProcessor

    family = _model_loader_family(path)
    if family == "multimodal_lm":
        try:
            from transformers import AutoModelForMultimodalLM as model_class
        except ImportError as exc:
            raise RuntimeError(
                "This qwen3_5 checkpoint requires a current Transformers release "
                "with AutoModelForMultimodalLM; use a separate Qwen3.5/3.6 environment."
            ) from exc
    else:
        from transformers import AutoModelForImageTextToText as model_class

    processor = AutoProcessor.from_pretrained(path, local_files_only=True)
    model = model_class.from_pretrained(
        path, torch_dtype="auto", device_map="auto", local_files_only=True
    )
    return torch, processor, model


def _generate(
    torch,
    processor,
    model,
    images: list[Image.Image],
    labels: list[str],
    max_new_tokens: int,
    instruction: str,
) -> str:
    """Run one Qwen generation on a list of images."""

    content: list[dict[str, Any]] = [
        {"type": "image", "image": image} for image in images
    ]
    content.append({"type": "text", "text": temporal_role_text(labels, instruction)})
    messages = [{"role": "user", "content": content}]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
        **_chat_template_kwargs(model),
    )
    inputs = {
        key: value.to(model.device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }
    with torch.inference_mode():
        generated = model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens)
    return processor.batch_decode(
        generated[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()


def _generate_batch(
    torch,
    processor,
    model,
    requests: Sequence[VLMRequest],
) -> list[str]:
    """Generate a homogeneous local-Qwen microbatch."""

    if not requests:
        return []
    image_counts = {len(item.images) for item in requests}
    token_budgets = {int(item.max_output_tokens) for item in requests}
    if len(image_counts) != 1:
        raise ValueError("local Qwen microbatch requires the same image count per request")
    if len(token_budgets) != 1:
        raise ValueError("local Qwen microbatch requires one max_output_tokens value")
    for item in requests:
        if len(item.images) != len(item.labels):
            raise ValueError("images and labels must have the same length")

    conversations: list[list[dict[str, Any]]] = []
    for item in requests:
        content: list[dict[str, Any]] = [
            {"type": "image", "image": image} for image in item.images
        ]
        content.append({
            "type": "text",
            "text": temporal_role_text(item.labels, item.instruction),
        })
        conversations.append([{"role": "user", "content": content}])
    # Left padding keeps the generation start aligned in decoder-only batches.
    tokenizer = getattr(processor, "tokenizer", None)
    previous_side = getattr(tokenizer, "padding_side", None) if tokenizer is not None else None
    if tokenizer is not None:
        tokenizer.padding_side = "left"
    try:
        inputs = processor.apply_chat_template(
            conversations,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
            **_chat_template_kwargs(model),
        )
    finally:
        # The processor is reused across calls, so never leak the override.
        if tokenizer is not None and previous_side is not None:
            tokenizer.padding_side = previous_side
    inputs = {
        key: value.to(model.device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=next(iter(token_budgets)),
        )
    decoded = processor.batch_decode(
        generated[:, inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    if len(decoded) != len(requests):
        raise RuntimeError(
            f"local Qwen microbatch decoded {len(decoded)} responses for {len(requests)} requests"
        )
    return [str(value).strip() for value in decoded]


class LocalQwenProvider:
    name = LOCAL_QWEN_PROVIDER

    def __init__(self, model_path: Path) -> None:
        self.model_path = model_path
        self.model_id = str(model_path)
        self._bundle = _load_model(model_path)

    def generate(
        self,
        images: Sequence[Image.Image],
        labels: Sequence[str],
        max_output_tokens: int,
        instruction: str,
    ) -> VLMGeneration:
        text = _generate(
            *self._bundle, list(images), list(labels), max_output_tokens, instruction
        )
        return VLMGeneration(text=text, provider=self.name, model_id=self.model_id)

    def generate_batch(self, requests: Sequence[VLMRequest]) -> list[VLMGeneration]:
        """Generate a local homogeneous microbatch without changing identity."""

        texts = _generate_batch(*self._bundle, list(requests))
        return [
            VLMGeneration(text=text, provider=self.name, model_id=self.model_id)
            for text in texts
        ]
