"""UE midpoint-QE Peek: extend the first excursion before revisiting QE."""
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from src.trajectory.templates import Trajectory


def extend_midpoint_peek(short):
    from src.trajectory.long_templates import infer_short_motion, _allocate_segment_frames, LONG_REVISIT_PAIRS
    source=short.poses_world_camera; qi=short.key_indices['qe1']; q0=source[0]; qe=source[qi]
    far_index=int(np.argmax(np.linalg.norm(source[:qi+1,:3,3]-q0[:3,3],axis=1)))
    if far_index<=0: raise ValueError('midpoint Peek requires a first excursion')
    prefix=source[:far_index+1]
    distance=np.r_[0,np.cumsum(np.linalg.norm(np.diff(prefix[:,:3,3],axis=0),axis=1))]
    keep=np.r_[True,np.diff(distance)>1e-10]; knots=distance[keep]; samples=prefix[keep]
    def outbound(scale):
        parts=[]
        for lo,hi in [(0,.3),(.3,.6),(.6,1)]:
            t=np.linspace(lo,hi,201)*knots[-1]; part=np.repeat(q0[None],len(t),axis=0)
            for axis in range(3):part[:,axis,3]=q0[axis,3]+scale*(np.interp(t,knots,samples[:,axis,3])-q0[axis,3])
            part[:,:3,:3]=Slerp(knots,Rotation.from_matrix(samples[:,:3,:3]))(t).as_matrix()
            parts.append(part)
        parts[0][0]=q0
        return parts
    def make(scale):
        out=outbound(scale); far=out[-1][-1]; alpha=np.linspace(0,1,401)
        back=np.repeat(qe[None],len(alpha),axis=0)
        back[:,:3,3]=(1-alpha[:,None])*far[:3,3]+alpha[:,None]*qe[:3,3]
        back[:,:3,:3]=Slerp([0,1],Rotation.from_matrix([far[:3,:3],qe[:3,:3]]))(alpha).as_matrix()
        back[0]=far;back[-1]=qe
        return out+[back,back[::-1].copy(),out[2][::-1].copy(),out[1][::-1].copy(),out[0][::-1].copy()]
    motion=infer_short_motion(short); budget=motion.linear_speed*60*float(short.parameters.get('long_speed_ratio',1.))
    length=lambda parts:sum(np.linalg.norm(np.diff(p[:,:3,3],axis=0),axis=1).sum() for p in parts)
    lo,hi=0.,max(4.,budget/max(knots[-1],1e-8))
    for _ in range(40):
        mid=(lo+hi)/2
        if length(make(mid))>budget:hi=mid
        else:lo=mid
    parts=make(lo); weights=np.array([length([p]) for p in parts]); counts=_allocate_segment_frames(weights,959,16)
    # QE and W share the midpoint pose, with one explicitly recorded 1/16s tick.
    poses=np.repeat(q0[None],961,axis=0); indices=[];left=0
    for i,(part,n) in enumerate(zip(parts,counts)):
        d=np.r_[0,np.cumsum(np.linalg.norm(np.diff(part[:,:3,3],axis=0),axis=1))];keep=np.r_[True,np.diff(d)>1e-10]
        t=np.linspace(0,d[keep][-1],int(n)+1)
        for axis in range(3):poses[left:left+n+1,axis,3]=np.interp(t,d[keep],part[keep,axis,3])
        poses[left:left+n+1,:3,:3]=Slerp(d[keep],Rotation.from_matrix(part[keep,:3,:3]))(t).as_matrix()
        poses[left]=part[0];poses[left+n]=part[-1];left+=int(n);indices.append(left)
        if i==3: poses[left+1]=qe;left+=1
    names=['qc_first','qa_first','qb_first','qe1','qb_revisit','qa_revisit','qc_revisit','q0_return']
    keys={'q0':0,**dict(zip(names,indices)),'w':indices[3]+1};keys=dict(sorted(keys.items(),key=lambda x:x[1]))
    params={**short.parameters,'horizon_profile':'long','long_extension_contract':'orbia.long_six_template.v1',
        'trajectory_revision':'ue_peek_midpoint_qe_v1','qe_retiming_policy':'ue_peek_midpoint_v1',
        'source_qe_frame_index':qi,'source_qe_event':'qe1','preserved_prefix_frame_count':1,
        'midpoint_pose_hold_frames':1,'source_first_excursion_frame':far_index,'exploration_amplitude_units':lo,
        'minimum_revisit_gap_s':5.,'collision_validation':'required_downstream',
        'revisit_pairs':[{'name':p.name,'first_event':p.first_event,'revisit_event':p.revisit_event} for p in LONG_REVISIT_PAIRS]}
    return Trajectory(template_name=short.template_name,difficulty=short.difficulty,duration_s=961/16,fps=16,poses_world_camera=poses,key_indices=keys,parameters=params)
