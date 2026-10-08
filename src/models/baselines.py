"""Trivial reference models."""
from __future__ import annotations

from src.models.base import ActionModel, CameraModel, TextModel, register_model


@register_model('static-camera')
class StaticCameraModel(CameraModel):
    """Ignores the poses; windowed to exercise chunked generation."""

    window_frames = 121

    def generate_with_poses(self, image, prompt, poses_c2w, K, *, case, history=None):
        return [image] * len(poses_c2w)


@register_model('static-action')
class StaticActionModel(ActionModel):
    action_mode = 'block'

    def generate_with_actions(self, image, prompt, plan, *, case):
        return [image] * plan['generated_frames']


@register_model('static-text')
class StaticTextModel(TextModel):
    native_fps = 24.0
    segment_frames = 121

    def generate_segment(self, context, prompt, num_frames, *, case, index):
        return [context[-1]] * num_frames
