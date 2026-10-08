"""Independent endpoint segmentation and mask-seeded SAM3.1 tracking."""
from __future__ import annotations
import gc
from typing import Any, Mapping, Sequence
import numpy as np
from PIL import Image
from src.construction.anchor_policy import measure_endpoint_mask
from src.backends.sam_masks import _start_session, _add_mask_prompt

def _object_ids(output: Mapping[str, Any], count: int) -> np.ndarray:
    values = np.asarray(output.get("out_obj_ids", np.arange(count)), dtype=np.int64).reshape(-1)
    return values if len(values) == count else np.arange(count, dtype=np.int64)

def _mask_iou(first: np.ndarray, second: np.ndarray) -> float:
    if first.shape != second.shape:
        return 0.0
    union = int(np.count_nonzero(first | second))
    return float(np.count_nonzero(first & second) / union) if union else 0.0

def _masks(output: Mapping[str, Any]) -> np.ndarray:
    masks = np.asarray(output.get("out_binary_masks", []), dtype=bool)
    if masks.ndim == 2:
        masks = masks[None]
    return masks if masks.ndim == 3 else np.zeros((0, 1, 1), dtype=bool)


def _mask_integrity(mask: np.ndarray) -> dict[str, Any]:
    """Measure endpoint mask integrity with the shared anchor policy."""

    return measure_endpoint_mask(mask)

def _instance_matches(
    q0_independent: Mapping[str, Any],
    qe_independent: Mapping[str, Any],
    q0_track: Mapping[str, Any],
    qe_track: Mapping[str, Any],
    *,
    minimum_endpoint_iou: float = 0.10,
) -> list[dict[str, Any]]:
    """Match separately emitted endpoint masks through soft tracking evidence."""

    q0i, qei = _masks(q0_independent), _masks(qe_independent)
    q0t, qet = _masks(q0_track), _masks(qe_track)
    q0_ids = _object_ids(q0_track, len(q0t))
    qe_ids = _object_ids(qe_track, len(qet))
    proposals: list[dict[str, Any]] = []
    for object_id in sorted(set(map(int, q0_ids)).intersection(map(int, qe_ids))):
        q0_track_index = int(np.flatnonzero(q0_ids == object_id)[0])
        qe_track_index = int(np.flatnonzero(qe_ids == object_id)[0])
        q0_ious = [_mask_iou(mask, q0t[q0_track_index]) for mask in q0i]
        qe_ious = [_mask_iou(mask, qet[qe_track_index]) for mask in qei]
        if not q0_ious or not qe_ious:
            continue
        q0_index, qe_index = int(np.argmax(q0_ious)), int(np.argmax(qe_ious))
        if min(q0_ious[q0_index], qe_ious[qe_index]) < minimum_endpoint_iou:
            continue
        proposals.append({
            "track_object_id": object_id,
            "q0_independent_index": q0_index,
            "qe_independent_index": qe_index,
            "q0_track_index": q0_track_index,
            "qe_track_index": qe_track_index,
            "q0_iou": round(q0_ious[q0_index], 4),
            "qe_iou": round(qe_ious[qe_index], 4),
            "q0_area_percent": round(100 * np.count_nonzero(q0i[q0_index]) / q0i[q0_index].size, 3),
            "qe_area_percent": round(100 * np.count_nonzero(qei[qe_index]) / qei[qe_index].size, 3),
            "q0_integrity": _mask_integrity(q0i[q0_index]),
            "qe_integrity": _mask_integrity(qei[qe_index]),
        })
    proposals.sort(
        key=lambda item: (
            min(float(item["q0_iou"]), float(item["qe_iou"])),
            min(float(item["q0_area_percent"]), float(item["qe_area_percent"])),
        ),
        reverse=True,
    )
    matches: list[dict[str, Any]] = []
    used_q0: set[int] = set()
    used_qe: set[int] = set()
    for item in proposals:
        q0_index = int(item["q0_independent_index"])
        qe_index = int(item["qe_independent_index"])
        if q0_index in used_q0 or qe_index in used_qe:
            continue
        matches.append(item)
        used_q0.add(q0_index)
        used_qe.add(qe_index)
    return matches

def _filter_output(
    output: Mapping[str, Any],
    box_norm: list[float] | tuple[float, ...] | None,
    *,
    allowed_object_ids: set[int] | None = None,
) -> dict[str, Any]:
    """Keep text-grounded objects localized by the endpoint-specific VLM box."""

    masks = np.asarray(output.get("out_binary_masks", []), dtype=bool)
    if masks.ndim == 2:
        masks = masks[None]
    if masks.ndim != 3:
        return {"out_binary_masks": np.zeros((0, 1, 1), dtype=bool), "out_obj_ids": np.zeros(0, dtype=np.int64)}
    object_ids = _object_ids(output, len(masks))
    keep = np.ones(len(masks), dtype=bool)
    if allowed_object_ids is not None:
        keep = np.logical_and(
            keep,
            np.asarray([int(value) in allowed_object_ids for value in object_ids], dtype=bool),
        )
    if box_norm is not None and len(box_norm) == 4 and len(masks):
        height, width = masks.shape[1:]
        x0 = max(0, min(width, int(float(box_norm[0]) * width)))
        y0 = max(0, min(height, int(float(box_norm[1]) * height)))
        x1 = max(x0 + 1, min(width, int(np.ceil(float(box_norm[2]) * width))))
        y1 = max(y0 + 1, min(height, int(np.ceil(float(box_norm[3]) * height))))
        box_area = max(1, (x1 - x0) * (y1 - y0))
        localized = []
        for mask in masks:
            mask_area = int(np.count_nonzero(mask))
            intersection = int(np.count_nonzero(mask[y0:y1, x0:x1]))
            if not mask_area or not intersection:
                localized.append(False)
                continue
            ys, xs = np.nonzero(mask)
            centroid_inside = x0 <= float(xs.mean()) < x1 and y0 <= float(ys.mean()) < y1
            overlap = intersection / min(mask_area, box_area)
            localized.append(centroid_inside or overlap >= 0.15)
        keep = np.logical_and(keep, np.asarray(localized, dtype=bool))
    return {"out_binary_masks": masks[keep], "out_obj_ids": object_ids[keep]}

class _EndpointDetector:
    """Ground many text prompts against fixed endpoint frames, encoding once."""

    def __init__(self, predictor: Any, images: Sequence[Any], threshold: float) -> None:
        self._predictor = predictor
        self._images = list(images)
        self._threshold = threshold
        self._sessions: dict[int, str] = {}

    def __enter__(self) -> "_EndpointDetector":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    def __call__(self, image_index: int, prompt: str) -> Mapping[str, Any]:
        session_id = self._sessions.get(image_index)
        if session_id is None:
            session_id = _start_session(self._predictor, [self._images[image_index]])
            self._sessions[image_index] = session_id
        return self._predictor.handle_request({
            "type": "add_prompt", "session_id": session_id, "frame_index": 0,
            "text": prompt, "output_prob_thresh": self._threshold,
        })["outputs"]

    def close(self) -> None:
        """Close every session, even if one close fails."""
        error: BaseException | None = None
        while self._sessions:
            _, session_id = self._sessions.popitem()
            try:
                self._predictor.handle_request({
                    "type": "close_session", "session_id": session_id, "run_gc_collect": True,
                })
            except BaseException as exc:  # noqa: BLE001 - re-raised below
                if error is None:
                    error = exc
        gc.collect()
        if error is not None:
            raise error

def _track_mask(
    predictor: Any,
    images: list[Image.Image],
    q0_mask: np.ndarray,
    threshold: float,
):
    """Track an independently segmented Q0 object without re-grounding text."""

    import torch

    session_id = _start_session(predictor, images)
    try:
        with torch.inference_mode():
            first, sam2_state = _add_mask_prompt(
                predictor, session_id, q0_mask
            )
            session = predictor._get_session(session_id)
            inference_state = session["state"]
            model = predictor.model
        last = first
        tracker = model.tracker
        output_dict = sam2_state["output_dict"]
        consolidated = sam2_state["consolidated_frame_inds"]
        object_ids = sam2_state["obj_ids"]
        batch_size = tracker._get_obj_num(sam2_state)
        with torch.inference_mode():
            for frame_index in range(len(images)):
                if frame_index in consolidated["cond_frame_outputs"]:
                    storage_key = "cond_frame_outputs"
                    current_out = output_dict[storage_key][frame_index]
                    pred_masks = current_out["pred_masks"]
                else:
                    # The outer SAM3 loop keeps only the current frame in its
                    # shared feature cache.  Prepare and consume that frame in one
                    # step, exactly as its partial-propagation path does.
                    model._prepare_backbone_feats(
                        inference_state, frame_index, reverse=False
                    )
                    storage_key = "non_cond_frame_outputs"
                    current_out, pred_masks = tracker._run_single_frame_inference(
                        inference_state=sam2_state,
                        output_dict=output_dict,
                        frame_idx=frame_index,
                        batch_size=batch_size,
                        is_init_cond_frame=False,
                        point_inputs=None,
                        mask_inputs=None,
                        reverse=False,
                        run_mem_encoder=True,
                    )
                    output_dict[storage_key][frame_index] = current_out
                tracker._add_output_per_object(
                    sam2_state, frame_index, current_out, storage_key
                )
                sam2_state["frames_already_tracked"][frame_index] = {
                    "reverse": False
                }
                _, video_masks = tracker._get_orig_video_res_output(
                    sam2_state, pred_masks
                )
                binary_masks = (video_masks[:, 0] > 0.0).detach().cpu().numpy()
                keep = np.asarray(binary_masks).reshape(
                    len(object_ids), -1
                ).any(axis=1)
                outputs = {
                    "out_binary_masks": binary_masks[keep],
                    "out_obj_ids": np.asarray(object_ids, dtype=np.int64)[keep],
                }
                if int(frame_index) == len(images) - 1:
                    last = outputs
        first = _filter_output(first, None)
        selected_ids = {int(value) for value in np.asarray(first["out_obj_ids"]).reshape(-1)}
        last = _filter_output(last, None, allowed_object_ids=selected_ids)
        return first, last
    finally:
        predictor.handle_request({"type": "close_session", "session_id": session_id, "run_gc_collect": True})
        gc.collect()

def _deduplicate_outputs(outputs: list[Mapping[str, Any]], iou_threshold: float = 0.85) -> Mapping[str, Any]:
    unique: list[np.ndarray] = []
    for output in outputs:
        masks = np.asarray(output.get("out_binary_masks", []), dtype=bool)
        if masks.ndim == 2:
            masks = masks[None]
        if masks.ndim != 3:
            continue
        for mask in masks:
            if not np.any(mask):
                continue
            duplicate = False
            for retained in unique:
                intersection = np.count_nonzero(mask & retained)
                union = np.count_nonzero(mask | retained)
                if union and intersection / union >= iou_threshold:
                    duplicate = True
                    break
            if not duplicate:
                unique.append(mask)
    return {"out_binary_masks": np.stack(unique) if unique else np.zeros((0, 1, 1), dtype=bool)}


def ground_and_track(predictor, images, candidates, options=None, endpoint_pairer=None):
    """Select distinct physical anchors through independent masks and tracking."""
    from dataclasses import replace
    from src.construction.anchor_policy import (
        STRICT_ANCHOR_THRESHOLDS, strict_anchor_policy_failures,
    )
    options = options or {}
    thresholds = replace(STRICT_ANCHOR_THRESHOLDS, **options.get('thresholds', {}))
    track = options.get('tracking', True)
    threshold = float(options.get('threshold', .35))
    duplicate_iou = float(options.get('deduplicate_iou', .90))
    detected = []
    rejected = []
    with _EndpointDetector(predictor, [images[0], images[-1]], threshold) as detector:
        for candidate in candidates:
            endpoints = [_deduplicate_outputs([detector(role, noun) for noun in candidate.sam_prompts],
                                               duplicate_iou) for role in (0, 1)]
            for index, mask in enumerate(_masks(endpoints[0])):
                integrity = measure_endpoint_mask(mask, thresholds=thresholds)
                # This preliminary check measures the seed, not its identity.
                failures = strict_anchor_policy_failures(identity_confirmed=True,
                    q0_tracking_iou=1, qe_tracking_iou=1,
                    q0_area_percent=100*mask.mean(), qe_area_percent=100*mask.mean(),
                    q0_integrity=integrity, qe_integrity=integrity, thresholds=thresholds)
                if failures or not len(_masks(endpoints[1])):
                    rejected.append({'label': candidate.label, 'q0_instance': index,
                        'failures': failures or ['no_qe_detection']})
                    continue
                detected.append({'candidate': candidate, 'q0': endpoints[0], 'qe': endpoints[1],
                                 'seed_index': index, 'area': float(mask.mean())})
    detected.sort(key=lambda row: row['area'], reverse=True)
    shortlisted = []
    for row in detected:
        seed = _masks(row['q0'])[row['seed_index']]
        if any(_mask_iou(seed, _masks(r['q0'])[r['seed_index']]) >= duplicate_iou for r in shortlisted):
            continue
        shortlisted.append(row)
        if len(shortlisted) == int(options.get('tracking_shortlist', 5)):
            break
    retained = []
    for row in shortlisted:
        q0_masks, qe_masks = _masks(row['q0']), _masks(row['qe'])
        if track:
            first, last = _track_mask(predictor, images, q0_masks[row['seed_index']], threshold)
            matches = _instance_matches(row['q0'], row['qe'], first, last,
                                       minimum_endpoint_iou=thresholds.minimum_tracking_iou)
        else:
            if endpoint_pairer is None:
                raise ValueError('endpoint-only annotation requires calibrated dense mask matching')
            matches = endpoint_pairer(q0_masks, qe_masks, row['seed_index'])
        if not matches:
            rejected.append({'label': row['candidate'].label, 'q0_instance': row['seed_index'],
                             'failures': ['independent_endpoint_identity_not_confirmed']})
        for match in matches:
            i, j = match['q0_independent_index'], match['qe_independent_index']
            q0_mask, qe_mask = q0_masks[i], qe_masks[j]
            qi = measure_endpoint_mask(q0_mask, thresholds=thresholds)
            ei = measure_endpoint_mask(qe_mask, thresholds=thresholds)
            failures = strict_anchor_policy_failures(identity_confirmed=True,
                q0_tracking_iou=match.get('q0_iou', 0), qe_tracking_iou=match.get('qe_iou', 0),
                q0_area_percent=100*q0_mask.mean(), qe_area_percent=100*qe_mask.mean(),
                q0_integrity=qi, qe_integrity=ei,
                thresholds=thresholds if track else replace(thresholds, minimum_tracking_iou=0))
            if failures:
                rejected.append({'label': row['candidate'].label, 'failures': failures})
                continue
            retained.append({'label': row['candidate'].label,
                'sam_prompts': list(row['candidate'].sam_prompts), 'q0': q0_mask, 'qe': qe_mask,
                'identity_method': 'mask_seeded_video_tracking' if track else 'independent_endpoints_dense_projection',
                'identity': match, 'q0_integrity': qi, 'qe_integrity': ei,
                'tracking_preferred': bool(track and min(match['q0_iou'], match['qe_iou']) >= .90)})
    retained.sort(key=lambda a: (a['tracking_preferred'],
        min(a['identity'].get('q0_iou', 0), a['identity'].get('qe_iou', 0)),
        min(a['q0'].mean(), a['qe'].mean())), reverse=True)
    final = []
    for anchor in retained:
        duplicate = next((a for a in final if _mask_iou(a['q0'], anchor['q0']) >= duplicate_iou
                          and _mask_iou(a['qe'], anchor['qe']) >= duplicate_iou), None)
        if duplicate is not None:
            duplicate.setdefault('aliases', []).append(anchor['label'])
            continue
        anchor['anchor_id'] = f'A{len(final)+1}'
        final.append(anchor)
        if len(final) == int(options.get('maximum_anchors', 3)):
            break
    return final, rejected
