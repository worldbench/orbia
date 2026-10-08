"""Inference-only identities. Bump protocol when inference semantics change."""
from pathlib import Path
from src.metrics.stages import file_identity
from src.metrics.da3 import context_heldout_indices

PROTOCOL = "orbia.precompute.v1"

def geometry_identity_for(job, config, plan, revision="1"):
    joint = len(plan['frame_indices']) <= int(config.get('da3_joint_max_frames', 0))
    checkpoint = Path(config['da3_streaming_checkpoint'])
    value = dict(protocol=PROTOCOL, revision=revision, video=file_identity(job['video']),
                 frame_indices=list(plan['frame_indices']), backend='joint' if joint else 'streaming',
                 checkpoint=[file_identity(checkpoint/name) for name in ('config.json','model.safetensors')],
                 resolution=int(config.get('da3_geometry_resolution', 504)))
    if not joint:
        value['streaming'] = {key:config.get(key, default) for key,default in
            [('da3_streaming_chunk_size',120),('da3_streaming_overlap',30),('da3_streaming_loop_enable',True)]}
        value['runtime'] = file_identity(Path(config['da3_streaming_repo'])/'da3_streaming/da3_streaming.py')
        value['base_config'] = file_identity(Path(config['da3_streaming_repo'])/'da3_streaming/configs/base_config.yaml')
        if value['streaming']['da3_streaming_loop_enable']:
            value['salad'] = file_identity(config['da3_streaming_salad_checkpoint'])
    return value

def gs_identity_for(geometry_identity, config, plan, frame_count):
    context, heldout = context_heldout_indices(frame_count,
        context_count=int(config.get('da3_context_views',64)),
        heldout_count=int(config.get('da3_heldout_views',16)), candidate_indices=plan['frame_indices'])
    checkpoint = Path(config['da3_checkpoint'])
    return dict(protocol=PROTOCOL, geometry=geometry_identity,
                checkpoint=[file_identity(checkpoint/name) for name in ('config.json','model.safetensors')],
                context=list(context), heldout=list(heldout),
                resolution=int(config.get('da3_input_resolution',504)),
                context_only=True, align_to_input_ext_scale=True)


def load_precompute_case(root):
    """Load public camera/sampling inputs only; inference needs no hidden references."""
    from types import SimpleNamespace
    from src.metrics.case import camera_from_document
    from src.data.case_loader import load_case
    root = Path(root)
    case = load_case(root.parent.parent, root.name, references=False)
    pairs = {name: dict(pair,
                       first_window_indices=[pair['first_frame_index']],
                       revisit_window_indices=[pair['revisit_frame_index']])
             for name, pair in case.document['revisit_pairs'].items()}
    return SimpleNamespace(case_id=case.case_id, frame_count=len(case.timestamps_s),
        timestamps_s=case.timestamps_s,
        events={k:v['frame_index'] for k,v in case.document['events'].items()},
        revisit_pairs=pairs,
        canonical_camera=camera_from_document(case.document['intrinsics'], name='input'), root=root)
