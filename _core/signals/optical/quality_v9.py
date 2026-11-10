
import numpy as np
NAMES=tuple('family_coverage_'+str(i) for i in range(5))+('physiological_observed_fraction','mean_pc_available_fraction','hr_history_coverage','hr_history_available','best_raw_sqi','raw_observed_column_fraction','raw_available_channel_fraction')


def quality_context(values,original_mask,details,quality,spectral_mask):
    x=np.asarray(values);m=np.asarray(original_mask)>0
    history=np.where(m[:,25]&np.isfinite(x[:,25]),np.clip(x[:,25],0,1),0)
    q=np.clip(np.nan_to_num(quality),0,1);sm=np.clip(np.nan_to_num(spectral_mask),0,1)
    available=(q>0)&(sm.sum(-1)>0)
    extra=np.stack((details['reliability'],details['feature_mask'].mean(-1),history,m[:,25],q.max(-1),sm.mean((-1,-2)),available.mean(-1)),-1)
    return np.concatenate((details['family_coverage'],extra),-1).astype(np.float32)
