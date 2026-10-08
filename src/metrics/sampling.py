"""Frames sampled for geometry recovery, as decoded-video frame indices."""
from __future__ import annotations

import numpy as np


def geometry_sampling_plan(case, info, options, *, include_gs=True):
    from src.metrics.da3 import context_heldout_indices

    stride = options.geometry_frame_stride
    if type(stride) is not int or stride < 1:
        raise ValueError('geometry_frame_stride must be a positive integer')
    n = info.frame_count
    uniform = list(range(0, n, stride))
    reasons = {i: {'uniform'} for i in uniform}

    def keep(indices, reason):
        for i in indices:
            if i is not None and 0 <= i < n:
                reasons.setdefault(int(i), set()).add(reason)

    def mapped(indices):
        return [info.timestamp_to_video_index(float(case.timestamps_s[i])) for i in indices]

    keep([0, n - 1], 'endpoint')
    for name, center in case.events.items():
        keep(mapped([center]), 'event:'+name)
    for name, pair in case.revisit_pairs.items():
        keep(mapped(pair['first_window_indices'] + pair['revisit_window_indices']), 'completion:'+name)
    if include_gs:
        context, heldout = context_heldout_indices(n, context_count=options.da3_context_views,
            heldout_count=options.da3_heldout_views, candidate_indices=sorted(reasons))
        keep(context, 'gs_context')
        keep(heldout, 'gs_heldout')
    return {'version': 'orbia.geometry_sampling.v2', 'video_frame_count': n,
            'video_fps': info.fps, 'stride': stride, 'uniform_indices': uniform,
            'frame_indices': sorted(reasons), 'frame_reasons': {str(i): sorted(v) for i,v in sorted(reasons.items())}}


def primary_pose_selection(indices, stride):
    if type(stride) is not int or stride < 1:
        raise ValueError('control_frame_stride must be a positive integer')
    return np.asarray(indices) % stride == 0
