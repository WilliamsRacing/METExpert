
from __future__ import annotations





from pathlib import Path
import sys


import numpy as np
import torch


SCHEMA = "v6-imu-ppgv9-frozen-evidence.1"
POINTS = 450
PPG_FIELDS = ("ppg_logits", "ppg_embedding", "ppg_support", "ppg_quality",
              "wave_coverage", "feature_coverage", "ticks", "target", "known", "observed")























class FrameAccumulator:
    """Merge overlapping source frames by physical point contribution.

    Point probabilities retain partial support. Embedding/quality/role features
    are averaged by supported point count, without using target/known masks.
    """
    def __init__(self, starts):
        self.starts=np.unique(np.asarray(starts,np.int64))
        n=len(self.starts)
        self.probability_sum=np.zeros((n,POINTS),np.float64)
        self.point_count=np.zeros((n,POINTS),np.uint16)
        self.segment=np.full((n,POINTS),-1,np.int64)
        self.mass=np.zeros(n,np.float64)
        self.embedding=np.zeros((n,128),np.float64)
        self.quality=np.zeros((n,8),np.float64)
        self.role=np.zeros((n,3),np.float64)
        self.conflicting_points=0

    def add(self, starts, probability, embedding, mask, quality, role_probability, segment_ids):
        starts=np.asarray(starts,np.int64);mask=np.asarray(mask,bool)
        if mask.shape!=(len(starts),POINTS):raise ValueError("Expected point mask [F,450]")
        probability=np.asarray(probability,np.float64)
        if probability.shape!=(len(starts),):raise ValueError("Expected frame probabilities")
        segment_ids=np.asarray(segment_ids,np.int64)
        if segment_ids.shape==(len(starts),):segment_ids=np.broadcast_to(segment_ids[:,None],mask.shape)
        if segment_ids.shape!=mask.shape or np.any(mask&(segment_ids<0)):
            raise ValueError("Supported points require explicit physical segment ownership")
        indices=np.searchsorted(self.starts,starts)
        if np.any(indices>=len(self.starts)) or np.any(self.starts[indices]!=starts):
            raise ValueError("Unknown global frame tick")
        features=[np.asarray(embedding,np.float64),np.asarray(quality,np.float64),np.asarray(role_probability,np.float64)]
        if [x.shape for x in features]!=[(len(starts),128),(len(starts),8),(len(starts),3)]:
            raise ValueError("Wrong IMU feature dimensions")
        active=mask.any(-1)
        if not np.isfinite(probability[active]).all() or np.any((probability[active]<0)|(probability[active]>1)):
            raise ValueError("Nonfinite or out-of-range supported IMU probability")
        if any(not np.isfinite(x[active]).all() for x in features):
            raise ValueError("Nonfinite supported IMU features")
        for j,i in enumerate(indices):
            selected=mask[j];mass=int(selected.sum())
            if not mass:continue
            self.conflicting_points+=int((selected&(self.point_count[i]>0)&(self.segment[i]!=segment_ids[j])).sum())
            self.probability_sum[i,selected]+=probability[j]
            self.point_count[i,selected]+=1
            old=self.segment[i,selected];new=segment_ids[j,selected]
            self.segment[i,selected]=np.where(old<0,new,np.minimum(old,new))
            self.mass[i]+=mass
            self.embedding[i]+=features[0][j]*mass
            self.quality[i]+=features[1][j]*mass
            self.role[i]+=features[2][j]*mass

    def finish(self):
        support=self.point_count>0
        p=self.probability_sum/np.maximum(self.point_count,1)
        p=np.clip(p,1e-6,1-1e-6)
        logits=np.where(support,np.log(p)-np.log1p(-p),0.).astype(np.float32)
        mass=np.maximum(self.mass,1)[:,None]
        return dict(frame_start_tick=self.starts,imu_logits=logits,imu_support=support,
                    imu_segment_ids=self.segment,imu_embedding=(self.embedding/mass).astype(np.float32),
                    imu_quality=(self.quality/mass).astype(np.float32),
                    imu_role_prob=(self.role/mass).astype(np.float32))


@torch.inference_mode()
def infer_person(config, manifest, checkpoint, group, person, global_ids):
    sys.path.insert(0,str(Path(config["imu_project"])/"src"))
    from motion_encoder.data import SequenceDataset
    from motion_encoder.model import IMUV6Model, ModelConfig
    own=dict(manifest,records=[r for r in manifest["records"] if r["person"]==person])
    dataset=SequenceDataset(own,split=group,training=False)
    starts=[int(r["start_tick"])+np.arange(int(r["frame_count"]),dtype=np.int64)*POINTS for r in dataset.records]
    accumulator=FrameAccumulator(np.concatenate(starts) if starts else np.empty(0,np.int64))
    if not starts:return accumulator.finish(),dict(source_records=0,conflicting_points=0)
    device=torch.device(config.get("device","cuda:0"))
    model=IMUV6Model(ModelConfig(**checkpoint["model_config"])).to(device)
    model.load_state_dict(checkpoint["state_dict"],strict=True);model.eval()
    core=int(config.get("imu_core_frames",16));radius=model.temporal_radius
    requests=[(ri,start,min(core,int(r["frame_count"])-start)) for ri,r in enumerate(dataset.records)
              for start in range(0,int(r["frame_count"]),core)]
    batch_size=int(config.get("imu_inference_batch_size",config.get("imu_batch_size",8)))
    amp=bool(config.get("amp",True) and device.type=="cuda")
    for offset in range(0,len(requests),batch_size):
        chunk=requests[offset:offset+batch_size]
        windows=[dataset.get_window(ri,start-radius,core+2*radius) for ri,start,count in chunk]
        keys=("imu","point_mask","frame_quality","segment_ids")
        batch={k:torch.stack([w[k] for w in windows]).to(device) for k in keys}
        with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=amp):
            out=model(batch["imu"],batch["point_mask"],batch["frame_quality"],batch["segment_ids"])
        for j,(ri,start,count) in enumerate(chunk):
            sl=slice(radius,radius+count);record=dataset.records[ri]
            ticks=int(record["start_tick"])+np.arange(start,start+count,dtype=np.int64)*POINTS
            valid=out["valid"][j,sl].cpu().numpy()
            mask=batch["point_mask"][j,sl].cpu().numpy()&valid[:,None]
            sid=batch["segment_ids"][j,sl].cpu().numpy()
            sid=np.where(sid>=0,(global_ids[record["source_key"]]+1)*1_000_000_000+sid,-1)
            accumulator.add(ticks,out["probability"][j,sl].float().cpu().numpy(),
                            out["embedding"][j,sl].float().cpu().numpy(),mask,
                            batch["frame_quality"][j,sl].float().cpu().numpy(),
                            out["logits"][j,sl].float().softmax(-1).cpu().numpy(),sid)
    dataset.close();del model
    if device.type=="cuda":torch.cuda.empty_cache()
    return accumulator.finish(),dict(source_records=len(dataset.records),conflicting_points=accumulator.conflicting_points)


def join_imu(ppg, imu):
    """Join only on person-owned absolute frame ticks; labels do not affect it."""
    ticks=np.asarray(ppg["ticks"],np.int64)
    if ticks.ndim!=2 or ticks.shape[1]!=POINTS or np.any(ticks!=ticks[:,:1]+np.arange(POINTS)):
        raise ValueError("PPG/native frame point clock is not a 15 Hz grid")
    starts=ticks[:,0];n=len(starts);source=np.asarray(imu["frame_start_tick"],np.int64)
    ix=np.searchsorted(source,starts);inside=ix<len(source)
    if len(source):inside&=source[np.minimum(ix,len(source)-1)]==starts
    else:inside[:]=False
    result={}
    for key,shape,dtype in (("imu_logits",(n,POINTS),np.float32),("imu_support",(n,POINTS),bool),
                            ("imu_segment_ids",(n,POINTS),np.int64),("imu_embedding",(n,128),np.float32),
                            ("imu_quality",(n,8),np.float32),("imu_role_prob",(n,3),np.float32)):
        value=np.full(shape,-1 if key=="imu_segment_ids" else 0,dtype=dtype)
        value[inside]=imu[key][ix[inside]]
        result[key]=value
    return result



