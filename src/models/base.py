"""Base classes for plugging a world model into ORBIA."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Sequence

import numpy as np

from src.models import actions as action_rules
from src.models import camera as cam
from src.models.camera_text import camera_prompt, window_prompts
from src.models.case import ModelCase

_REGISTRY: dict[str, type] = {}


def register_model(name: str):
    """Class decorator that makes a model available as ``--model name``."""
    def decorate(cls):
        if name in _REGISTRY and _REGISTRY[name] is not cls:
            raise ValueError(f'model {name!r} is already registered')
        _REGISTRY[name] = cls
        cls.name = name
        return cls
    return decorate


def get_model(name: str, **kwargs) -> 'WorldModel':
    """Instantiate a registered model or ``package.module:ClassName``."""
    if ':' in name:
        import importlib
        module, attribute = name.split(':', 1)
        return getattr(importlib.import_module(module), attribute)(**kwargs)
    if name not in _REGISTRY:
        raise KeyError(f'unknown model {name!r}; registered: {sorted(_REGISTRY)}')
    return _REGISTRY[name](**kwargs)


def registered_models() -> list[str]:
    return sorted(_REGISTRY)


def conform(frames: Sequence[np.ndarray], num_frames: int) -> list[np.ndarray]:
    """Trim extra terminal frames; report incomplete generation."""
    frames = [np.asarray(f, dtype=np.uint8) for f in frames]
    if not frames:
        raise ValueError('model returned no frames')
    if len(frames) < num_frames:
        raise ValueError(f'model returned {len(frames)} frames, expected at least {num_frames}')
    return frames[:num_frames]


class WorldModel(ABC):
    """Common interface. ``group`` and ``control_profile`` go into the evaluation manifest."""

    name: str = 'world-model'
    group: str = 'world'
    control_profile: str = 'pose_and_action'

    @abstractmethod
    def generate(self, case: ModelCase) -> tuple[list[np.ndarray], dict[str, Any]]:
        """Return the generated frames and a JSON-serializable record of the condition."""

    def setup(self) -> None:
        """Load weights once before the first case (optional)."""


class CameraModel(WorldModel):
    """Models conditioned on numerical camera poses."""

    window_frames: int | None = None
    overlap_frames: int = 1

    @abstractmethod
    def generate_with_poses(self, image: np.ndarray, prompt: str, poses_c2w: np.ndarray,
                            K: np.ndarray, *, case: ModelCase, history: list[np.ndarray] | None = None
                            ) -> list[np.ndarray]:
        """Generate ``len(poses_c2w)`` frames."""

    def generate(self, case):
        if not self.window_frames:
            frames = self.generate_with_poses(case.image, case.prompt, case.poses_c2w, case.K, case=case)
            return conform(frames, case.num_frames), {'mode': 'single_call'}
        spans = cam.windows(case.num_frames, self.window_frames, self.overlap_frames)
        poses = cam.pad_trajectory(case.poses_c2w, spans[-1][1])
        output: list[np.ndarray] = []
        for start, end in spans:
            image = case.image if not output else output[-1]
            window = cam.rebase(poses[start:end])
            frames = list(self.generate_with_poses(image, case.prompt, window, case.K, case=case,
                                                   history=output or None))
            output.extend(frames if not output else frames[self.overlap_frames:])
        return conform(output, case.num_frames), {'mode': 'windowed', 'windows': spans}


class ActionModel(WorldModel):
    """Models conditioned on discrete movement/look actions."""

    control_profile = 'action_only'
    action_mode: str = 'block'
    block_spec: action_rules.BlockActionSpec = action_rules.BlockActionSpec()
    timed_spec: action_rules.TimedActionSpec | None = None

    def plan_actions(self, case: ModelCase) -> dict:
        if self.action_mode == 'block':
            return action_rules.block_actions(case.poses_c2w, events=case.event_frames,
                                              template=case.template, spec=self.block_spec)
        if self.action_mode == 'timed':
            if self.timed_spec is None:
                raise ValueError('timed action models must define timed_spec with calibrated turning rates')
            return action_rules.timed_actions(case.poses_c2w, events=case.event_frames,
                                              template=case.template, spec=self.timed_spec)
        raise ValueError("action_mode must be 'block' or 'timed'")

    @abstractmethod
    def generate_with_actions(self, image: np.ndarray, prompt: str, plan: dict, *,
                              case: ModelCase) -> list[np.ndarray]:
        """Generate frames from ``plan['actions']`` (block) or ``plan['intervals']`` (timed)."""

    def generate(self, case):
        plan = self.plan_actions(case)
        frames = self.generate_with_actions(case.image, case.prompt, plan, case=case)
        return conform(frames, case.num_frames), {'mode': self.action_mode, 'plan': plan}


class TextModel(WorldModel):
    """Image-and-text-to-video models driven by a timed motion script."""

    group = 'ti2v'
    native_fps: float | None = 24.0
    segment_frames: Sequence[int] | int | None = None
    context_frames: int = 1
    overlap_frames: int = 1

    @abstractmethod
    def generate_segment(self, context: list[np.ndarray], prompt: str, num_frames: int, *,
                         case: ModelCase, index: int) -> list[np.ndarray]:
        """Generate ``num_frames`` frames; ``context`` holds the conditioning frame(s)."""

    def segment_spans(self, case: ModelCase) -> list[tuple[int, int]]:
        n = case.num_frames
        if self.segment_frames is None:
            return [(0, n)]
        if isinstance(self.segment_frames, int):
            return cam.windows(n, self.segment_frames, self.overlap_frames)
        spans, start = [], 0
        for k, length in enumerate(self.segment_frames):
            spans.append((start, start + length))
            start += length - self.overlap_frames
        if spans[-1][1] < n:
            raise ValueError(f'segments cover {spans[-1][1]} frames, case needs {n}')
        return spans

    def generate(self, case):
        spans = self.segment_spans(case)
        prompts = [camera_prompt(case, self.native_fps)] if len(spans) == 1 else \
            window_prompts(case, spans, self.native_fps)
        output: list[np.ndarray] = []
        for index, ((start, end), prompt) in enumerate(zip(spans, prompts)):
            context = [case.image] if not output else output[-self.context_frames:]
            frames = list(self.generate_segment(context, prompt['prompt'], end - start, case=case, index=index))
            output.extend(frames if not output else frames[self.overlap_frames:])
        record = {'mode': 'text', 'native_fps': self.native_fps,
                  'segments': [{'window': list(s), 'prompt': p['prompt']} for s, p in zip(spans, prompts)]}
        return conform(output, case.num_frames), record
