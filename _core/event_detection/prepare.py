
from __future__ import annotations

from pathlib import Path

import numpy as np



POINTS=450

class IMUAccumulator:
    def __init__(self,starts):
        self.starts=np.unique(np.asarray(starts,np.int64));n=len(self.starts)
        self.sum=np.zeros((n,POINTS),np.float64);self.count=np.zeros((n,POINTS),np.uint16)
        self.sid=np.full((n,POINTS),-1,np.int64);self.mass=np.zeros(n,np.float64)
        self.quality=np.zeros((n,8),np.float64);self.role=np.zeros((n,2),np.float64);self.boundary=np.zeros((n,2),np.float64)
        self.local=np.zeros(n,np.float64);self.context=np.zeros(n,np.float64)
    def add(self,row,source_id):
        starts=np.asarray(row['frame_start_tick'],np.int64);ix=np.searchsorted(self.starts,starts)
        mask=np.asarray(row['point_mask'],bool)&np.asarray(row['valid'],bool)[:,None]
        p=np.asarray(row['probability'],np.float64);roles=np.asarray(row['role_probability'],np.float64)
        bounds=np.asarray(row['boundary_probability'],np.float64);quality=np.asarray(row['frame_quality'],np.float64)
        ids=np.asarray(row['segment_ids'],np.int64)
        for j,i in enumerate(ix):
            m=mask[j];mass=int(m.sum())
            if not mass:continue
            if not np.isfinite(p[j]) or not 0<=p[j]<=1:raise ValueError('Invalid IMU prediction')
            sid=(source_id+1)*1_000_000_000+int(ids[j]);old=self.sid[i,m]
            self.sum[i,m]+=p[j];self.count[i,m]+=1;self.sid[i,m]=np.where(old<0,sid,np.minimum(old,sid))
            self.mass[i]+=mass;self.role[i]+=roles[j]*mass;self.boundary[i]+=bounds[j]*mass;self.quality[i]+=quality[j]*mass
            self.local[i]+=float(row['local_probability'][j])*mass
            self.context[i]+=float(row['context_probability'][j])*mass
    def finish(self):
        support=self.count>0;p=np.clip(self.sum/np.maximum(self.count,1),1e-6,1-1e-6);mass=np.maximum(self.mass,1)[:,None]
        bp=np.clip(self.boundary/mass,1e-6,1-1e-6)
        local=np.clip(self.local/np.maximum(self.mass,1),1e-6,1-1e-6)
        context=np.clip(self.context/np.maximum(self.mass,1),1e-6,1-1e-6)
        return dict(frame_start_tick=self.starts,imu_logits=np.where(support,np.log(p)-np.log1p(-p),0).astype(np.float32),
            imu_support=support,imu_segment_ids=self.sid,imu_role_prob=(self.role/mass).astype(np.float32),
            imu_quality=(self.quality/mass).astype(np.float32),imu_boundary_logits=(np.log(bp)-np.log1p(-bp)).astype(np.float32),
            imu_local_logits=(np.log(local)-np.log1p(-local)).astype(np.float32),
            imu_context_logits=(np.log(context)-np.log1p(-context)).astype(np.float32))

def join_imu(starts,source):
    n=len(starts);old=source['frame_start_tick'];ix=np.searchsorted(old,starts);inside=ix<len(old)
    if len(old):inside&=old[np.minimum(ix,len(old)-1)]==starts
    else:inside[:]=False
    result={}
    for name,shape,dtype in [('imu_logits',(n,450),np.float32),('imu_support',(n,450),bool),('imu_segment_ids',(n,450),np.int64),
                            ('imu_role_prob',(n,2),np.float32),('imu_quality',(n,8),np.float32),('imu_boundary_logits',(n,2),np.float32),
                            ('imu_local_logits',(n,),np.float32),('imu_context_logits',(n,),np.float32)]:
        a=np.full(shape,-1 if name=='imu_segment_ids' else 0,dtype=dtype);a[inside]=source[name][ix[inside]];result[name]=a
    return result

def native_array(root,entry,prefix,key):
    value=entry.get('reused_arrays',{}).get(prefix+'__'+key)
    path=Path(value) if value else Path(root)/entry['path']/(prefix+'__'+key+'.npy')
    if not path.is_absolute():path=Path(root)/path
    return np.load(path,allow_pickle=False)

def native_frames(root,entry):
    ticks=native_array(root,entry,'imu','ticks').astype(np.int64)
    if not len(ticks) or np.any(np.diff(ticks)!=1):raise ValueError('Native session must retain physical point grid')
    origin=int(ticks[0])//450*450;n=(int(ticks[-1])-origin)//450+1;out={'ticks':origin+np.arange(n*450,dtype=np.int64).reshape(n,450)}
    pos=ticks-origin
    for name,key,dtype in [('target','target_all',np.float32),('known','supervision_all',bool),('observed','observed',bool)]:
        a=np.zeros(n*450,dtype);a[pos]=native_array(root,entry,'target',key);out[name]=a.reshape(n,450)
    return out

def majority_sid(ids,support):
    result=np.full(len(ids),-1,np.int64)
    for j in range(len(ids)):
        valid=ids[j,support[j]]
        if len(valid):
            values,counts=np.unique(valid,return_counts=True);result[j]=values[counts.argmax()]
    return result

def fused_segments(starts,imu_sid,ppg_sid,imu_support,ppg_support):
    """Context can continue if either physical modality links adjacent frames."""
    iv=imu_support.any(-1);pv=ppg_support.astype(bool);valid=iv|pv;result=np.full(len(starts),-1,np.int64);current=0
    for j in range(len(starts)):
        if not valid[j]:continue
        link=j>0 and valid[j-1] and starts[j]-starts[j-1]==450 and (
            (iv[j] and iv[j-1] and imu_sid[j]>=0 and imu_sid[j]==imu_sid[j-1]) or
            (pv[j] and pv[j-1] and ppg_sid[j]>=0 and ppg_sid[j]==ppg_sid[j-1]))
        if not link:current+=1
        result[j]=current
    return result

def join_ppg(starts,records,base):
    scalar_fields=('ppg_logits','ppg_start_logits','ppg_end_logits','ppg_quality','wave_coverage','feature_coverage',
        'ppg_quality_raw_max','ppg_quality_expert','raw_coverage','spectral_coverage')
    logit_fields=('ppg_logits','ppg_start_logits','ppg_end_logits')
    n=len(starts);sums={k:np.zeros(n,np.float64) for k in scalar_fields};embedding=np.zeros((n,64),np.float64)
    mass=np.zeros(n,np.int32);sid=np.full(n,-1,np.int64)
    for ri,entry in enumerate(records):
        with np.load(Path(base)/entry['path'],allow_pickle=False) as z:
            ticks=np.asarray(z['frame_start_tick'],np.int64);ix=np.searchsorted(starts,ticks);inside=ix<n
            inside[inside]&=starts[ix[inside]]==ticks[inside]
            valid=inside&np.asarray(z['ppg_support'],bool);dst=ix[valid]
            # The original adapter's internal expert policy is unchanged.
            for name in scalar_fields:
                value=z[name][valid].astype(np.float64)
                if name in logit_fields:value=1/(1+np.exp(-np.clip(value,-60,60)))
                sums[name][dst]+=value
            embedding[dst]+=z['ppg_embedding'][valid]
            mass[dst]+=1;sid[dst]=np.where(sid[dst]<0,ri+1,np.minimum(sid[dst],ri+1))
    out={name:(value/np.maximum(mass,1)).astype(np.float32) for name,value in sums.items()}
    for name in logit_fields:
        p=np.clip(out[name],1e-6,1-1e-6);out[name]=np.where(mass>0,np.log(p)-np.log1p(-p),0).astype(np.float32)
    out['ppg_embedding']=(embedding/np.maximum(mass,1)[:,None]).astype(np.float32)
    out['ppg_support']=mass>0
    return out,sid


