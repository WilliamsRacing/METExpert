"""Fit only a small fusion head; expert weights and input contracts are frozen."""
from __future__ import annotations












import numpy as np
import torch







FIELDS = ("imu_logits", "imu_embedding", "imu_support", "imu_quality", "imu_role_prob",
          "ppg_logits", "ppg_embedding", "ppg_support", "ppg_quality", "wave_coverage", "feature_coverage")











def collate(rows):
    length=max(len(r['ppg_logits']) for r in rows);out={}
    for key in rows[0]:
        if key=='_index':out[key]=torch.tensor([int(r[key]) for r in rows]);continue
        values=[]
        for row in rows:
            x=row[key];value=np.zeros((length,*x.shape[1:]),dtype=x.dtype)
            if key=='imu_segment_ids':value.fill(-1)
            value[:len(x)]=x;values.append(value)
        out[key]=torch.from_numpy(np.stack(values))
    return out





















