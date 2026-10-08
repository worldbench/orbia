from __future__ import annotations
"""SAM3 dense-mask session helpers shared by annotation and evaluation."""
from PIL import Image
from typing import Any, Mapping
import numpy as np

def _start_session(predictor: Any, images: list[Image.Image]) -> str:
    # SAM's base predictor forwards options that multiplex init_state may omit.
    from inspect import Parameter, signature

    init_state = predictor.model.init_state
    parameters = signature(init_state).parameters
    accepts_kwargs = any(p.kind == Parameter.VAR_KEYWORD for p in parameters.values())

    def initialize(**kwargs):
        return init_state(**(kwargs if accepts_kwargs else {
            key: value for key, value in kwargs.items() if key in parameters
        }))

    predictor.model.init_state = initialize
    try:
        response = predictor.handle_request({
            "type": "start_session", "resource_path": images,
            "offload_video_to_cpu": False,
        })
    finally:
        predictor.model.init_state = init_state
    return str(response["session_id"])


def _add_mask_prompt(
    predictor: Any,
    session_id: str,
    mask: np.ndarray,
    *,
    frame_index: int = 0,
    object_id: int = 1,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    """Expose SAM3's existing dense-mask tracker seed through the local wrapper."""

    import torch

    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 2 or not np.any(binary):
        raise RuntimeError("No mask is provided; please add a non-empty mask first")
    session = predictor._get_session(session_id)
    inference_state = session["state"]
    model = predictor.model
    tracker_metadata = inference_state["tracker_metadata"]
    if tracker_metadata == {}:
        tracker_metadata.update(model._initialize_metadata())
    if model.world_size != 1 or model.rank != 0:
        raise RuntimeError("mask-seeded tracking currently requires one model process per GPU")
    if model._get_gpu_id_by_obj_id(inference_state, object_id) is not None:
        raise RuntimeError(f"object id already exists: {object_id}")

    model._prepare_backbone_feats(inference_state, frame_index, reverse=False)
    object_rank = int(model._assign_new_det_to_gpus(
        new_det_num=1,
        prev_workload_per_gpu=tracker_metadata["num_obj_per_gpu"],
    )[0])
    sam2_state = model._init_new_sam2_state(inference_state)
    inference_state["sam2_inference_states"].append(sam2_state)

    tracker_metadata["obj_ids_per_gpu"][object_rank] = np.concatenate([
        tracker_metadata["obj_ids_per_gpu"][object_rank],
        np.asarray([object_id], dtype=np.int64),
    ])
    tracker_metadata["num_obj_per_gpu"][object_rank] = len(
        tracker_metadata["obj_ids_per_gpu"][object_rank]
    )
    tracker_metadata["obj_ids_all_gpu"] = np.concatenate(
        tracker_metadata["obj_ids_per_gpu"]
    )
    tracker_metadata["max_obj_id"] = max(
        int(tracker_metadata["max_obj_id"]), object_id
    )
    tracker_metadata["obj_id_to_score"][object_id] = 1.0
    tracker_metadata["obj_id_to_sam2_score_frame_wise"][frame_index][object_id] = (
        torch.tensor(1.0, dtype=torch.float32, device=model.device)
    )
    model.add_action_history(
        inference_state, "add", frame_idx=frame_index, obj_ids=[object_id]
    )
    rank0_metadata = tracker_metadata.get("rank0_metadata", {})
    rank0_metadata.get("removed_obj_ids", set()).discard(object_id)
    for suppressed in rank0_metadata.get("suppressed_obj_ids", {}).values():
        suppressed.discard(object_id)
    confirmation = rank0_metadata.get("masklet_confirmation")
    if isinstance(confirmation, Mapping):
        object_count = len(tracker_metadata["obj_ids_all_gpu"])
        status = np.asarray(confirmation.get("status", []), dtype=np.int64)
        consecutive = np.asarray(
            confirmation.get("consecutive_det_num", []), dtype=np.int64
        )
        if len(status) < object_count:
            status = np.pad(status, (0, object_count - len(status)))
        if len(consecutive) < object_count:
            consecutive = np.pad(consecutive, (0, object_count - len(consecutive)))
        object_index = int(
            np.where(tracker_metadata["obj_ids_all_gpu"] == object_id)[0][0]
        )
        status[object_index] = 1
        consecutive[object_index] = int(
            model.masklet_confirmation_consecutive_det_thresh
        )
        confirmation["status"] = status
        confirmation["consecutive_det_num"] = consecutive

    _, object_ids, _, video_masks = model.tracker.add_new_masks(
        inference_state=sam2_state,
        frame_idx=frame_index,
        obj_ids=[object_id],
        masks=torch.as_tensor(
            binary[None], dtype=torch.bool, device=model.device
        ),
    )
    # An externally supplied mask is authoritative evidence that the object is
    # present on the conditioning frame.  The multiplex decoder may otherwise
    # leave a negative no-object logit even though add_new_masks deliberately
    # returns the exact input mask for frame 0; that negative score makes the
    # object disappear on the very first propagated frame.
    condition_output = sam2_state["output_dict"]["cond_frame_outputs"].get(
        frame_index
    )
    if condition_output is not None and condition_output.get(
        "object_score_logits"
    ) is not None:
        condition_output["object_score_logits"].fill_(10.0)
    model.tracker.propagate_in_video_preflight(sam2_state, run_mem_encoder=True)
    if video_masks is None or object_id not in object_ids:
        raise RuntimeError("SAM3 add_new_masks returned no seeded object")
    seeded_mask = (video_masks[object_ids.index(object_id)] > 0.0).to(torch.bool)
    object_to_mask = {object_id: seeded_mask}
    model._cache_frame_outputs(inference_state, frame_index, object_to_mask)
    inference_state["previous_stages_out"][frame_index] = (
        "_THIS_FRAME_HAS_OUTPUTS_"
    )
    output = {
        "obj_id_to_mask": object_to_mask,
        "obj_id_to_score": tracker_metadata["obj_id_to_score"],
        "obj_id_to_sam2_score": tracker_metadata[
            "obj_id_to_sam2_score_frame_wise"
        ][frame_index],
    }
    return model._postprocess_output(inference_state, output), sam2_state

