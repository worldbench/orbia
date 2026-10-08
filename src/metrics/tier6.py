"""Raw temporal diagnostics; missing measurements never become stability."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from src.metrics.contract import MetricResult


INCOMPLETE_STATUSES = ("invalid_input", "evaluator_failed", "not_available")


def _windows(
    *, values: Sequence[float], frame_indices: Sequence[int], fps: float,
    duration_s: float, window_s: float,
) -> list[dict[str, Any]]:
    if not all(np.isfinite(x) and x > 0 for x in (fps, duration_s, window_s)):
        raise ValueError("window timing must be finite and positive")
    if len(values) != len(frame_indices):
        raise ValueError("quality values and frame indices must have equal length")
    indices = np.asarray(frame_indices)
    if len(indices) and (
        not np.issubdtype(indices.dtype, np.integer)
        or np.any(indices < 0) or np.any(np.diff(indices) <= 0)
    ):
        raise ValueError("quality indices must be unique, increasing non-negative integers")
    times = indices.astype(float) / fps
    scores = np.asarray(values, dtype=float)
    starts = list(np.arange(0.0, max(duration_s - window_s, 0.0) + 1e-9, window_s))
    tail = max(duration_s - window_s, 0.0)
    if not starts or not np.isclose(starts[-1], tail, rtol=0, atol=1e-9):
        starts.append(tail)
    result = []
    for start in starts:
        stop = min(start + window_s, duration_s)
        # Half-open windows, except the actual last requested timestamp.
        right = times <= stop if np.isclose(stop, duration_s, rtol=0, atol=1e-9) else times < stop
        selected = scores[(times >= start) & right & np.isfinite(scores)]
        result.append({
            "start_s": float(start), "end_s": float(stop),
            "frame_count": len(selected),
            "status": "ok" if len(selected) else "not_covered",
            "mean": float(selected.mean()) if len(selected) else None,
        })
    return result


def window_changes(windows, *, higher_is_better):
    """Signed absolute differences; positive means worse, missing stays missing."""
    first = windows[0]['mean'] if windows else None
    previous = None
    direction = -1.0 if higher_is_better else 1.0
    result = []
    for index, window in enumerate(windows):
        value = window['mean']
        result.append({**window, 'window_index': index,
            'higher_is_better': higher_is_better,
            'degradation_from_first': direction*(value-first) if value is not None and first is not None else None,
            'degradation_from_previous': direction*(value-previous) if value is not None and previous is not None else None})
        previous = value
    return result


def _curve(
    metric: Mapping[str, Any] | None, field: str, frame_indices: Sequence[int],
    *, fps: float, duration_s: float, window_s: float,
) -> Mapping[str, Any]:
    if not isinstance(metric, Mapping):
        return {"status": "not_available", "windows": [], "reason": "missing source metric"}
    if metric.get("status") != "ok":
        return {"status": str(metric.get("status", "not_available")),
                "windows": [], "reason": metric.get("reason")}
    raw = metric.get("raw")
    values = raw.get("per_frame") if isinstance(raw, Mapping) else None
    if not isinstance(values, list):
        return {"status": "invalid_input", "windows": [], "reason": "missing per-frame scores"}
    try:
        windows = _windows(values=values, frame_indices=frame_indices, fps=fps,
                           duration_s=duration_s, window_s=window_s)
    except (TypeError, ValueError) as exc:
        return {"status": "invalid_input", "windows": [], "reason": str(exc)}
    windows = window_changes(windows, higher_is_better=True)
    valid = [w["mean"] for w in windows if w["mean"] is not None]
    first, last = windows[0]["mean"], windows[-1]["mean"]
    return {
        "status": "ok" if len(valid) == len(windows) else "not_covered",
        "field": field, "windows": windows,
        "expected_window_count": len(windows), "evaluated_window_count": len(valid),
        "first_window": first, "last_window": last,
        "first_minus_last": first - last if first is not None and last is not None else None,
        "worst_window": min(valid) if valid else None,
    }


def long_horizon_metric(
    *, timestamps_s: np.ndarray, events: Mapping[str, int],
    revisit_pairs: Mapping[str, Mapping[str, Any]], video_fps: float,
    quality_frame_indices: Sequence[int], imaging: Mapping[str, Any] | None,
    aesthetic: Mapping[str, Any] | None, window_s: float = 5.0,
) -> MetricResult:
    """Use the requested horizon, never shorten it to the decoded video."""
    timestamps = np.asarray(timestamps_s, dtype=float)
    if (timestamps.ndim != 1 or len(timestamps) < 2
            or not np.isfinite(timestamps).all() or np.any(np.diff(timestamps) <= 0)
            or timestamps[0] != 0 or not np.isfinite(video_fps) or video_fps <= 0
            or not np.isfinite(window_s) or window_s <= 0):
        return MetricResult(name="long_horizon", status="invalid_input", applicable=True,
                            failure="invalid requested or video timing")
    duration_s = float(timestamps[-1])
    schedules = {}
    for name, pair in revisit_pairs.items():
        first, revisit = str(pair["first_event"]), str(pair["revisit_event"])
        schedules[name] = {
            "first_event": first, "revisit_event": revisit,
            "first_time_s": float(timestamps[events[first]]),
            "revisit_time_s": float(timestamps[events[revisit]]),
            "interval_s": float(timestamps[events[revisit]] - timestamps[events[first]]),
        }
    curves = {
        "musiq": _curve(imaging, "musiq", quality_frame_indices, fps=video_fps,
                        duration_s=duration_s, window_s=window_s),
        "laion_aesthetic": _curve(aesthetic, "laion_aesthetic", quality_frame_indices,
                                 fps=video_fps, duration_s=duration_s, window_s=window_s),
    }
    statuses = [c["status"] for c in curves.values()]
    status = next((s for s in INCOMPLETE_STATUSES if s in statuses),
                  "ok" if all(s == "ok" for s in statuses) else "not_covered")
    return MetricResult(
        name="long_horizon", status=status, applicable=True, version="v0.3",
        coverage={"quality_frame_count": len(quality_frame_indices),
                  "registered_revisit_pair_count": len(schedules),
                  "quality_metric_statuses": {k: v["status"] for k, v in curves.items()}},
        raw={"window_s": float(window_s), "duration_s": duration_s,
             "quality_curves": curves, "revisit_schedule": schedules},
        failure=None if status == "ok" else "quality measurements are incomplete; see per-metric status and window coverage",
        provenance={"protocol": "requested-horizon quality windows; preserve missing endpoints and source failures; no composite score"},
    )


def reused_temporal_diagnostics(metrics, *, fps, duration_s, window_s=5.0):
    """Aggregate existing pair/event traces; never re-run a neural estimator."""
    def source(name):
        obj = metrics.get(name)
        return obj.to_dict() if hasattr(obj,'to_dict') else obj or {}
    def pair_records(value):
        if isinstance(value,dict):
            if 'source_frame_index' in value and 'target_frame_index' in value:
                yield value
            else:
                for v in value.values():
                    yield from pair_records(v)
        elif isinstance(value,list):
            for v in value:
                yield from pair_records(v)
    geometry = {}
    for name,field in [('depth_temporal_consistency','abs_rel'),
                       ('covisible_reprojection','feature_reprojection_similarity'),
                       ('regional_scale_inconsistency','regional_scale_inconsistency')]:
        obj = source(name)
        records = list({(r.get('group'),r['source_frame_index'],r['target_frame_index']):r for r in pair_records(obj.get('raw',{}))}.values())
        # Aggregate both directions/pairs at the target timestamp first.
        by_index = {}
        for row in records:
            value = row.get(field)
            if isinstance(value,(float,int)) and np.isfinite(value):
                by_index.setdefault(int(row['target_frame_index']),[]).append(float(value))
        indices = sorted(by_index)
        values = [float(np.mean(by_index[i])) for i in indices]
        windows = _windows(values=values,frame_indices=indices,fps=fps,duration_s=duration_s,window_s=window_s)
        windows = window_changes(windows, higher_is_better=name == 'covisible_reprojection')
        means = [w['mean'] for w in windows]
        geometry[name] = {'status':'ok' if means and all(x is not None for x in means) else 'not_covered',
                          'source_status':obj.get('status','not_available'), 'field':field,
                          'windows':windows,'pair_count':len(records),
                          'last_minus_first':means[-1]-means[0] if means and means[0] is not None and means[-1] is not None else None}
    completion = source('completion_revisit').get('raw',{}).get('revisit_pairs',{})
    return {'geometry_curves':geometry,
            'evidence_revisits':{
                'q0_reference':{'q0_return':source('q0_return').get('raw',{}).get('canonical')},
                'qe_reference':{'qe1':source('evidence_query').get('raw',{}).get('canonical'),
                               **source('evidence_query').get('raw',{}).get('repeated_events',{})}},
            'registered_additional_evidence_event_count':len(source('evidence_query').get('raw',{}).get('repeated_events',{})),
            'anchor_tracks':source('anchor_trajectory').get('raw',{}).get('windows',{}),
            'anchor_identity':source('anchor_identity').get('raw',{}),
            'completion_revisits':completion,
            'direct_completion_revisits':source('completion_revisit').get('raw',{}).get('direct',{}),
            'completion_reference_policy':'registered first_event; no rolling reference',
            'reuse_policy':'raw existing measurements only; absent events remain unmeasured'}
