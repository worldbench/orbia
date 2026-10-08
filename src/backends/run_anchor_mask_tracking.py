#!/usr/bin/env python3
"""SAM3.1 mask-seeded tracking using the existing annotation implementation."""
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np


def request_frame_indices(r):
    """Return the frame indices to track; dense requests have no explicit map."""
    start,end=r['start'],r['end']
    stride=r.get('tracking_stride',1)
    if type(stride) is not int or stride<1:raise ValueError('tracking_stride must be a positive integer')
    if type(start) is not int or type(end) is not int or start<0 or end<start:raise ValueError('invalid tracking interval')
    expected=sorted(set(range(start,end+1,stride)) | {start,end,*r['samples']})
    indices=r.get('frame_indices',expected)
    if any(type(i) is not int for i in indices) or indices!=expected:
        raise ValueError('tracking frame indices must match stride plus interval endpoints and scoring samples')
    if any(i<start or i>end for i in indices):raise ValueError('tracking frame outside Q0/QE interval')
    return indices


def run_request(r, predictors):
    sys.path.insert(0, r['sam_repo'])
    import cv2,torch
    from PIL import Image
    from sam3.model_builder import build_sam3_multiplex_video_predictor
    from src.backends.sam_masks import _start_session, _add_mask_prompt
    cv2.setNumThreads(1);torch.set_num_threads(4)
    started=time.perf_counter();root=Path(r['output']);root.mkdir(parents=True,exist_ok=True)
    with np.load(r['seed'],allow_pickle=False) as seed:
        reference=seed['reference'];masks=seed['masks'];names=seed['names'].tolist()
    frame_indices=request_frame_indices(r)
    images=[Image.fromarray(reference)]
    if r.get('frames_npy'):
        rgb_frames=np.load(r['frames_npy'],mmap_mode='r',allow_pickle=False)
        if len(rgb_frames)!=len(frame_indices):raise ValueError('incorrect shared frame count for original-index map')
        images.extend(Image.fromarray(frame) for frame in rgb_frames)
    else:
        _decode_images(r, images, cv2, Image, frame_indices)
    key=(r['sam_repo'],r['checkpoint'],r.get('sam_config'))
    model_reused=key in predictors
    if not model_reused:
        predictors.clear()
        predictors[key]=build_sam3_multiplex_video_predictor(checkpoint_path=r['checkpoint'],max_num_objects=16,use_fa3=False,
            compile=False,warm_up=False,default_output_prob_thresh=.35,async_loading_frames=False)
    predictor=predictors[key]
    load_s=time.perf_counter()-started;session_id=None;written=[]
    try:
        session_id=_start_session(predictor,images)
        model=predictor.model;tracker=model.tracker
        inference_state=predictor._get_session(session_id)['state']
        with torch.inference_mode():
            states=[]
            for j,mask in enumerate(masks):
                if mask.any():
                    _,state=_add_mask_prompt(predictor,session_id,mask,object_id=j+1)
                    states.append((j,state))
            for frame_index in range(len(images)):
                original=frame_indices[frame_index-1] if frame_index>0 else None
                selected=frame_index>0 and original in r['samples']
                saved=np.zeros_like(masks,dtype=bool) if selected else None
                if frame_index>0 and states:model._prepare_backbone_feats(inference_state,frame_index,reverse=False)
                for j,state in states:
                    outputs=state['output_dict']
                    if frame_index in state['consolidated_frame_inds']['cond_frame_outputs']:
                        key='cond_frame_outputs';current=outputs[key][frame_index];pred=current['pred_masks']
                    else:
                        key='non_cond_frame_outputs'
                        current,pred=tracker._run_single_frame_inference(inference_state=state,output_dict=outputs,
                            frame_idx=frame_index,batch_size=tracker._get_obj_num(state),is_init_cond_frame=False,
                            point_inputs=None,mask_inputs=None,reverse=False,run_mem_encoder=True)
                        outputs[key][frame_index]=current
                    tracker._add_output_per_object(state,frame_index,current,key)
                    state['frames_already_tracked'][frame_index]={'reverse':False}
                    if selected:
                        _,video_masks=tracker._get_orig_video_res_output(state,pred)
                        saved[j]=(video_masks[0,0]>0).detach().cpu().numpy()
                if selected:
                    np.savez_compressed(root/f'masks_{original:06d}.npz',masks=saved,names=np.asarray(names),frame_index=original)
                    rgb=np.asarray(images[frame_index]).copy();colors=[(255,200,0),(0,100,255),(80,255,80),(200,0,255)]
                    for j,mask in enumerate(saved):rgb[mask]=(.6*rgb[mask]+.4*np.array(colors[j%len(colors)])).astype(np.uint8)
                    Image.fromarray(rgb).save(root/f'overlay_{original:06d}.jpg');written.append(original)
    finally:
        if session_id is not None:predictor.handle_request({'type':'close_session','session_id':session_id,'run_gc_collect':True})
    if sorted(written)!=sorted(r['samples']):raise ValueError('missing sampled masks')
    result={'samples':sorted(written),'read_frames':len(images)-1,
        'tracking_stride':r.get('tracking_stride',1),'tracking_frame_indices':frame_indices,
        'tracking_frame_count':len(frame_indices),'tracking_frame_count_with_reference':len(images),
        'source_interval_frame_count':r['end']-r['start']+1,'anchor_count':len(names),'active_seed_count':len(states),
        'model_reused':model_reused,'frame_source':'shared_memory' if r.get('frames_npy') else 'video_decode','load_and_decode_s':load_s,'elapsed_s':time.perf_counter()-started,
        'protocol':'SAM3.1 annotation mask seed; shared per-frame backbone; reference Q0 then generated RGB; no geometry/text regrounding'}
    (root/'tracker_stats.json').write_text(json.dumps(result,indent=2));return result
def _decode_images(r, images, cv2, Image, frame_indices=None):
    wanted=set(request_frame_indices(r) if frame_indices is None else frame_indices)
    cap=cv2.VideoCapture(r['video'])
    try:
        for original in range(r['end']+1):
            ok,bgr=cap.read()
            if not ok:raise ValueError(f'video ends before requested frame {original}')
            if original in wanted:images.append(Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)))
    finally:cap.release()


def serve(response_fd, handler=run_request):
    # stdout/stderr remain ordinary model logs; replies have a separate pipe.
    import os,traceback
    predictors={}
    with os.fdopen(response_fd,'w',buffering=1) as response:
        for line in sys.stdin:
            try:
                request=json.loads(line)
                result=handler(request,predictors)
                reply={'ok':True,'result':result}
            except Exception:
                predictors.clear()
                reply={'ok':False,'error':traceback.format_exc()}
            # Session tensors are out of scope. Return unused GPU blocks to
            # the scoring process while retaining the predictor weights.
            if 'torch' in sys.modules:
                import gc
                gc.collect()
                sys.modules['torch'].cuda.empty_cache()
            response.write(json.dumps(reply)+'\n')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--request',type=Path)
    p.add_argument('--response-fd',type=int)
    p.add_argument('--parent-pid',type=int)
    a=p.parse_args()
    if a.response_fd is not None:
        # Ensure abnormal scoring-worker exit cannot strand GPU memory.
        import ctypes,os,signal
        ctypes.CDLL(None).prctl(1,signal.SIGTERM)
        if a.parent_pid is not None and os.getppid()!=a.parent_pid:return
        serve(a.response_fd)
    elif a.request is not None:print(json.dumps(run_request(json.loads(a.request.read_text()),{})))
    else:p.error('--request or --response-fd is required')

if __name__=='__main__':main()
