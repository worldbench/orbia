"""Load a case from the public/ and references/ folders."""
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json
import numpy as np
from PIL import Image


def _json(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def _asset(root, relative):
    root = Path(root).resolve()
    candidate = Path(relative)
    if candidate.is_absolute() or '..' in candidate.parts or candidate in (Path('.'), Path('')):
        raise ValueError('asset escapes its case directory')
    return root / candidate


def _rgb(path):
    with Image.open(path) as image:
        return np.asarray(image.convert('RGB'))


def _mask(path):
    with Image.open(path) as image:
        return np.asarray(image.convert('L')) > 0


@dataclass(frozen=True)
class ReferenceFrame:
    rgb: np.ndarray
    depth: np.ndarray
    valid: np.ndarray
    camera: dict[str, Any]
    depth_unit: str


@dataclass(frozen=True)
class Case:
    dataset_root: Path
    case_id: str
    document: dict[str, Any]
    q0_rgb: np.ndarray
    poses_c2w: np.ndarray
    timestamps_s: np.ndarray
    reference: dict[str, Any] | None
    frames: dict[str, ReferenceFrame]
    anchor_masks: dict[str, dict[str, np.ndarray]]
    observability: dict[str, np.ndarray]

    @property
    def prompt(self):
        return (self.dataset_root/'public'/self.case_id/'prompt.txt').read_text(encoding='utf-8').strip()

    @property
    def template(self):
        return self.document['template']

    @property
    def parent_case_id(self):
        return self.document['parent_case_id']

    def to_evaluation_case(self):
        """Convert to the in-memory case used by the evaluator."""
        if self.reference is None:
            raise ValueError('evaluation requires the reference bundle')
        from src.metrics.case import EvaluationCase, HiddenFrame, camera_from_document
        public = self.document
        def frame(role):
            item = self.frames[role]
            return HiddenFrame(role='qe1' if role == 'qe' else role, rgb=item.rgb,
                depth_m=item.depth, depth_valid=item.valid,
                rgb_camera=camera_from_document(item.camera['rgb_camera'], name=role+' RGB'),
                depth_camera=camera_from_document(item.camera['depth_camera'], name=role+' depth'),
                camera_document=item.camera)
        pairs = {name: {'first_event': p['first_event'], 'revisit_event': p['revisit_event'],
                 'first_window_indices':[p['first_frame_index']],
                 'revisit_window_indices':[p['revisit_frame_index']], 'event_offset_in_window':0}
                 for name,p in public['revisit_pairs'].items()}
        primary = pairs['C']
        completion = {**primary, 'first_event_frame_index': primary['first_window_indices'][0],
                      'revisit_event_frame_index': primary['revisit_window_indices'][0]}
        if self.reference['exact_revisit']:
            completion['pairs'] = public['revisit_pairs']
        ref_root = self.dataset_root / 'references' / self.case_id
        appearance = _mask(_asset(ref_root, self.reference['qe_scene_region']))
        appearance_anchors = {a['anchor_id']: _mask(_asset(ref_root, a['qe_region'])) for a in self.reference['anchors']}
        if any(m.shape != appearance.shape or np.any(m & ~appearance) for m in appearance_anchors.values()):
            raise ValueError('anchor regions must be within the QE scene region')
        return EvaluationCase(template=public['template'], source=self.reference['source'],
            appearance_support_rgb=appearance, appearance_anchor_masks_rgb=appearance_anchors,
            root=self.dataset_root/'public'/self.case_id, case_id=self.case_id,
            q0_rgb=self.q0_rgb, canonical_camera=camera_from_document(public['intrinsics'], name='input'),
            native_to_canonical=np.asarray(self.reference['native_to_input']),
            requested_poses=self.poses_c2w, timestamps_s=self.timestamps_s,
            events={name: int(event['frame_index']) for name,event in public['events'].items()},
            completion_pairs=completion,
            revisit_pairs=pairs, q0=frame('q0'), qe=frame('qe'),
            observability_input_supported_depth=self.observability['input_supported'],
            observability_invalid_depth=self.observability['invalid'],
            q0_anchor_masks_rgb={a:m['q0'] for a,m in self.anchor_masks.items()},
            qe_anchor_masks_rgb={a:m['qe'] for a,m in self.anchor_masks.items()},
            anchor_dirs=tuple(self.dataset_root/'references'/self.case_id/'anchors'/a for a in self.anchor_masks))


def load_case(dataset_root, case_id, *, references=True):
    root = Path(dataset_root).resolve()
    if not isinstance(case_id,str) or Path(case_id).name != case_id or case_id in ('.','..'):
        raise ValueError('case_id must be one path component')
    public_root = root/'public'/case_id
    doc = _json(public_root/'case.json')
    if doc.get('case_id') != case_id:
        raise ValueError('case identity mismatch')
    with np.load(public_root/'camera.npz', allow_pickle=False) as a:
        poses=np.asarray(a['poses_c2w']);times=np.asarray(a['timestamps_s'])
    if poses.shape != (len(times),4,4) or len(times)!=doc['frame_count'] or not np.isfinite(poses).all() or not np.isfinite(times).all():
        raise ValueError('invalid camera trajectory')
    for event in doc['events'].values():
        index=event['frame_index']
        if type(index) is not int or not 0 <= index < len(times):
            raise ValueError('event outside trajectory')
        if abs(float(event['timestamp_s'])-float(times[index]))>1e-6:
            raise ValueError('event time differs from camera timestamp')
    rgb=_rgb(public_root/'q0.png')
    if rgb.shape[:2] != (doc['intrinsics']['height'],doc['intrinsics']['width']):
        raise ValueError('input resolution differs from calibration')
    frames={};anchors={};observability={};ref=None
    if references:
        rr=root/'references'/case_id;ref=_json(rr/'reference.json')
        if ref.get('case_id')!=case_id:
            raise ValueError('reference identity mismatch')
        for role in ('q0','qe'):
            f=ref['frames'][role];camera=f
            depth=np.load(_asset(rr,f['depth']),allow_pickle=False)
            valid=_mask(_asset(rr,f['valid']));im=_rgb(_asset(rr,f['rgb']))
            if depth.dtype!=np.float32 or depth.shape!=(camera['depth_camera']['height'],camera['depth_camera']['width']) or valid.shape!=depth.shape:
                raise ValueError('depth, dtype, valid mask or camera mismatch')
            if im.shape[:2]!=(camera['rgb_camera']['height'],camera['rgb_camera']['width']):
                raise ValueError('reference RGB calibration mismatch')
            if np.any(valid & (~np.isfinite(depth) | (depth<=0))):
                raise ValueError('invalid values marked as valid depth')
            if f['depth_unit'] not in ('metre','source_pose_unit','DA3_estimated_unit'):
                raise ValueError('unsupported depth unit')
            frames[role]=ReferenceFrame(im,depth,valid,camera,f['depth_unit'])
        for anchor in ref['anchors']:
            aid=anchor['anchor_id']
            if aid in anchors:raise ValueError('duplicate anchor ID')
            masks={role:_mask(_asset(rr,anchor[role+'_mask'])) for role in ('q0','qe')}
            if any(masks[role].shape!=frames[role].rgb.shape[:2] for role in masks):
                raise ValueError('anchor mask resolution mismatch')
            anchors[aid]=masks
        if not anchors:raise ValueError('reference has no anchors')
        observability={name:_mask(_asset(rr,path)) for name,path in ref['observability'].items()}
        if any(mask.shape!=frames['qe'].depth.shape for mask in observability.values()):
            raise ValueError('observability is not in QE depth space')
        requested=np.linalg.inv(poses[doc['events']['q0']['frame_index']])@poses[doc['events']['qe1']['frame_index']]
        truth=np.linalg.inv(np.asarray(frames['q0'].camera['T_world_camera']))@np.asarray(frames['qe'].camera['T_world_camera'])
        if not np.allclose(requested,truth,atol=1e-5,rtol=0):raise ValueError('QE does not match reference pose')
    return Case(root,case_id,doc,rgb,poses,times,ref,frames,anchors,observability)
