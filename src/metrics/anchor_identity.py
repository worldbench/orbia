"""Q0-to-QE object-mask feature consistency, conditional on measurable masks."""
import hashlib,json,subprocess,sys
from pathlib import Path
from time import perf_counter
import numpy as np
from src.metrics.contract import MetricResult
from src.metrics.stages import StageCache,file_identity,atomic_json


# One resident SAM subprocess per scoring process/GPU. Each request closes its
# tracking session; only weights survive across cases.
class TrackingWorker:
    def __init__(self, python, script, log_path):
        import os
        read_fd,write_fd=os.pipe()
        self.log=open(log_path,'a')
        try:
            self.process=subprocess.Popen([str(python),str(script),'--response-fd',str(write_fd),'--parent-pid',str(os.getpid())],
                stdin=subprocess.PIPE,stdout=self.log,stderr=subprocess.STDOUT,text=True,pass_fds=(write_fd,))
        except BaseException:
            os.close(read_fd);self.log.close();raise
        finally:os.close(write_fd)
        self.response=os.fdopen(read_fd,'r')
    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:self.process.kill();self.process.wait()
        try:self.process.stdin.close()
        except BrokenPipeError:pass
        self.response.close();self.log.close()
    def request(self, request, timeout):
        import select
        self.process.stdin.write(json.dumps(request)+'\n');self.process.stdin.flush()
        if not select.select([self.response],[],[],timeout)[0]:raise TimeoutError('SAM tracking request timed out')
        line=self.response.readline()
        if not line:raise RuntimeError(f'SAM worker exited: {self.process.poll()}')
        result=json.loads(line)
        if not result['ok']:raise RuntimeError(result['error'])
        return result['result']

_TRACKING_WORKERS={}
def close_tracking_workers():
    for worker in _TRACKING_WORKERS.values():worker.close()
    _TRACKING_WORKERS.clear()

import atexit
atexit.register(close_tracking_workers)

def _track(request, options, script, root, frames):
    import os,tempfile
    from src.metrics.video import cached_video_frames
    python=options.anchor_sam_python or sys.executable
    key=(str(python),str(script))
    worker=_TRACKING_WORKERS.get(key)
    if worker is None or worker.process.poll() is not None:
        if worker is not None:worker.close()
        worker=TrackingWorker(python,script,root.parent/'sam_worker.log')
        _TRACKING_WORKERS[key]=worker
    full=cached_video_frames(request['video'])
    indices=request.get('frame_indices',list(range(request['start'],request['end']+1)))
    source=full if full is not None else frames
    shared=None
    try:
        has_interval=(all(i in source for i in indices) if isinstance(source,dict)
                      else all(0<=i<len(source) for i in indices))
        if has_interval:
            # Linux memfd: no disk write and no shared_memory resource tracker.
            # Child maps this file through /proc while parent owns its lifetime.
            if hasattr(os,'memfd_create'):shared=os.fdopen(os.memfd_create('orbia-sam-rgb'),'w+b')
            else:shared=tempfile.TemporaryFile()
            shape=(len(indices),*source[request['start']].shape)
            np.lib.format.write_array_header_1_0(shared,{'descr':np.dtype('uint8').str,'fortran_order':False,'shape':shape})
            for i in indices:shared.write(np.ascontiguousarray(source[i],dtype=np.uint8).tobytes())
            shared.flush()
            proc_pid=os.readlink('/proc/self')
            request=dict(request,frames_npy=f'/proc/{proc_pid}/fd/{shared.fileno()}')
        result=worker.request(request,options.anchor_tracking_timeout_s)
        atomic_json(root/'tracker_transport.json',result)
    except BaseException:
        worker.close();_TRACKING_WORKERS.pop(key,None);raise
    finally:
        if shared is not None:shared.close()


def sample_indices(start,end):
    if start is None or end is None or end<=start:raise ValueError('Q0/QE interval is missing or empty')
    return sorted({min(end,max(start+1,int(np.floor(start+fraction*(end-start)+.5)))) for fraction in (.25,.5,.75,1.)})


def tracking_indices(start,end,stride=1):
    """Original video indices sent to SAM; scoring instants are never dropped."""
    import operator
    if isinstance(stride,bool):raise ValueError('anchor_tracking_stride must be a positive integer')
    try:stride=operator.index(stride)
    except TypeError:raise ValueError('anchor_tracking_stride must be a positive integer') from None
    if isinstance(stride,bool) or stride<1:raise ValueError('anchor_tracking_stride must be a positive integer')
    samples=sample_indices(start,end)
    return sorted(set(range(start,end+1,stride)) | {start,end,*samples})


def pooled_descriptor(patches,mask,min_pixels=16):
    import cv2
    mask=np.asarray(mask,bool)
    if int(mask.sum())<min_pixels:return None,'empty_or_small_mask'
    h,w=patches.shape[:2]
    weights=cv2.resize(mask.astype(np.float32),(w,h),interpolation=cv2.INTER_AREA)
    if weights.sum()<1.:return None,'less_than_one_feature_patch_of_support'
    vector=np.sum(patches*weights[...,None],axis=(0,1))/weights.sum()
    norm=np.linalg.norm(vector)
    if not np.isfinite(vector).all() or norm<=1e-12:return None,'invalid_descriptor'
    return vector/norm,None


def score_masks(reference_patches,reference_masks,patches_by_frame,masks_by_frame,*,min_pixels=16):
    anchors={}
    for j,name in enumerate(reference_masks):
        ref,reason=pooled_descriptor(reference_patches,reference_masks[name],min_pixels)
        samples=[]
        for index,patches in sorted(patches_by_frame.items()):
            mask=masks_by_frame[index][j]
            value,why=pooled_descriptor(patches,mask,min_pixels)
            why=reason or why
            samples.append({'frame_index':index,'dino_similarity':float(np.clip(ref@value,-1,1)) if why is None else None,
                            'mask_pixels':int(mask.sum()),'status':'ok' if why is None else 'not_covered','reason':why})
        values=[r['dino_similarity'] for r in samples if r['dino_similarity'] is not None]
        anchors[name]={'samples':samples,'mean_similarity':float(np.mean(values)) if values else None,
                       'min_similarity':float(np.min(values)) if values else None,
                       'evaluated_sample_count':len(values),'expected_sample_count':len(samples),
                       'coverage':len(values)/len(samples) if samples else 0.,
                       'status':'ok' if values and len(values)==len(samples) else 'not_covered'}
    means=[a['mean_similarity'] for a in anchors.values() if a['mean_similarity'] is not None]
    minima=[a['min_similarity'] for a in anchors.values() if a['min_similarity'] is not None]
    expected=sum(a['expected_sample_count'] for a in anchors.values());evaluated=sum(a['evaluated_sample_count'] for a in anchors.values())
    return {'anchors':anchors,'macro_mean_similarity':float(np.mean(means)) if means else None,
            'macro_min_similarity':float(np.mean(minima)) if minima else None,
            'expected_anchor_count':len(anchors),'evaluated_anchor_count':len(means),
            'expected_sample_count':expected,'evaluated_sample_count':evaluated,
            'coverage':evaluated/expected if expected else 0.,'aggregation':'equal anchor weight; conditional on valid masks; Q0 excluded from score'}


def anchor_identity_metric(*,reference,masks,frames,start,end,video,output,options,comparator,case_id):
    if comparator.backend!='dinov2':raise ValueError('anchor_identity requires official DINOv2 dense features')
    indices=sample_indices(start,end)
    stride=getattr(options,'anchor_tracking_stride',1)
    selected=tracking_indices(start,end,stride)
    root=Path(output)/'anchor_mask_cache';root.mkdir(parents=True,exist_ok=True)
    script=Path(__file__).resolve().parents[1]/'backends/run_anchor_mask_tracking.py'
    names=sorted(masks);mask_array=np.stack([masks[n] for n in names])
    # Hash small seed arrays only, never the video. DINO settings are not tracker inputs.
    digest=hashlib.sha256(reference.tobytes()+mask_array.tobytes()+json.dumps(names).encode()).hexdigest()
    identity={'version':'q0-qe-sam31-v1','video':file_identity(video),'case_id':case_id,'seed':digest,
              'shape':list(reference.shape),'start':start,'end':end,'samples':indices,
              'tracking_stride':stride,'frame_indices':selected,
              'checkpoint':file_identity(options.anchor_sam_checkpoint),'sam_config':options.anchor_sam_config,
              'script':file_identity(script),'annotation_helper':file_identity(script.with_name('sam_masks.py')),'sam_source':[file_identity(Path(options.anchor_sam_repo)/p) for p in ('sam3/model_builder.py','sam3/model/sam3_tracking_predictor.py')]}
    cache=StageCache(root)
    artifacts=[root/f'masks_{i:06d}.npz' for i in indices]
    def valid():
        for i,p in zip(indices,artifacts):
            with np.load(p,allow_pickle=False) as a:
                if a['masks'].shape!=mask_array.shape or a['names'].tolist()!=names or int(a['frame_index'])!=i:return False
        return True
    reused=cache.ready('tracking',identity,valid);started=perf_counter()
    if not reused:
        cache.invalidate('tracking');np.savez_compressed(root/'seed.npz',reference=reference,masks=mask_array,names=np.asarray(names))
        request={'video':str(video),'start':start,'end':end,'samples':indices,
                 'tracking_stride':stride,'frame_indices':selected,'seed':str(root/'seed.npz'),
                 'output':str(root),'sam_repo':str(options.anchor_sam_repo),'checkpoint':str(options.anchor_sam_checkpoint),'sam_config':options.anchor_sam_config}
        atomic_json(root/'request.json',request)
        _track(request,options,script,root,frames)
        if not valid():raise ValueError('invalid tracking output')
        cache.commit('tracking',identity,[*artifacts,root/'tracker_stats.json'])
    tracking_s=perf_counter()-started
    _,ref_patches,_=comparator._descriptor(reference,None,cache_key=('anchor_identity','q0'))
    ref_patches=comparator._patch_grid(ref_patches)
    patches={i:comparator._patch_grid(comparator._descriptor(frames[i],None,cache_key=('anchor_identity',i))[1]) for i in indices}
    targets={}
    for i,p in zip(indices,artifacts):
        with np.load(p,allow_pickle=False) as a:targets[i]=a['masks']
    raw=score_masks(ref_patches,{n:masks[n] for n in names},patches,targets,min_pixels=options.anchor_identity_min_pixels)
    raw.update(sample_frame_indices=indices,q0_frame_index=start,qe_frame_index=end,tracking_reused=reused,tracking_wall_s=tracking_s,
               tracking_stride=stride,tracking_frame_indices=selected,tracking_frame_count=len(selected),
               tracking_frame_count_with_reference=len(selected)+1)
    return MetricResult(name='anchor_identity',status='ok' if raw['coverage']==1. and raw['expected_anchor_count'] else 'not_covered',applicable=True,version='v0.1',raw=raw,
        coverage={k:raw[k] for k in ('expected_sample_count','evaluated_sample_count','coverage','expected_anchor_count','evaluated_anchor_count')},
        provenance={'tracker':'SAM3.1 annotation mask-seeded propagation','feature':'DINOv2 mask-weighted patch pooling; fixed Q0 reference',
                    'interpretation':'conditional feature consistency, not identity accuracy or visibility ground truth',
                    'tracking_stride':stride,'tracking_frame_indices':selected,
                    'tracking_receipt':str(cache.path('tracking'))})
