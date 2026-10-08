"""Joint DA3 inference exported in the same per-frame format as streaming."""
from pathlib import Path
import numpy as np
from src.backends.run_da3_streaming_evaluator import _materialize_contract

def run_joint_geometry(*, model, frames, info, job, plan, output, temp, resolution=504):
    indices = plan['frame_indices']
    prediction = model.inference(image=[frames[i] for i in indices], process_res=resolution, infer_gs=False)
    depth = np.asarray(prediction.depth)
    extrinsics = np.asarray(prediction.extrinsics)
    intrinsics = np.asarray(prediction.intrinsics)
    if len(depth) != len(indices) or depth.ndim != 3:
        raise ValueError('joint DA3 depth shape does not match selected frames')
    if not all(np.isfinite(x).all() for x in (depth,extrinsics,intrinsics)):
        raise ValueError('joint DA3 returned nonfinite geometry')
    w2c = np.tile(np.eye(4), (len(indices),1,1))
    w2c[:,:3,:] = extrinsics[:,:3,:]
    poses = np.linalg.inv(w2c)
    video_k = intrinsics.copy()
    video_k[:,0] *= info.width/depth.shape[2]
    video_k[:,1] *= info.height/depth.shape[1]
    sources = []
    for local in range(len(indices)):
        target = Path(temp)/f'frame_{local}.npz'
        fields = dict(depth=depth[local], intrinsics=intrinsics[local])
        if getattr(prediction,'conf',None) is not None:
            fields['conf'] = np.asarray(prediction.conf)[local] - 1.0
        np.savez_compressed(target, **fields)
        sources.append(target)
    _materialize_contract(job=job, output=output, poses_c2w=poses,
        intrinsics_video=video_k, source_depths=sources,
        provenance=dict(backend='da3_joint', sampling_plan=plan,
            process_resolution=resolution, loop_closure=False, infer_gs=False))
