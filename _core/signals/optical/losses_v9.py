
from __future__ import annotations
import torch
from torch.nn import functional as F

LOSS_BATCH_KEYS=('core_mask','frame_valid','label_weight','target','start_target','end_target','boundary_mask','physiology_features','physiology_mask','identity_weight','frame_start_ms','frame_end_ms')
LOSS_OUTPUT_KEYS=('feature_logits','raw_logits','feature_start_logits','feature_end_logits','raw_start_logits','raw_end_logits','feature_support','raw_support','raw_feature_prediction','raw_observation_quality')


def combine_loss_batches(batches,outputs):
    b={k:torch.cat([x[k] for x in batches],0) for k in LOSS_BATCH_KEYS if all(k in x for x in batches)}
    o={k:torch.cat([x[k] for x in outputs],0) for k in LOSS_OUTPUT_KEYS}
    return b,o


def segmentation_loss_v9(output,batch,config,pos_weight,boundary_pos_weight):
    tc=config['training'];dev=output['feature_logits'].device
    if any(v.device!=dev for k,v in batch.items() if k in LOSS_BATCH_KEYS and isinstance(v,torch.Tensor)):
        raise ValueError('V9 loss expects a once-transferred device batch')
    clean=lambda x:torch.nan_to_num(x.float()).clamp(0,1)
    base=clean(batch['core_mask'])*clean(batch['frame_valid'])
    known=torch.isfinite(batch['target'])&(batch['target']>=0)
    target=torch.nan_to_num(batch['target'].float()).clamp(0,1)
    confidence=clean(batch['label_weight'])
    identity=batch.get('identity_weight',torch.ones_like(base)).float()
    if identity.ndim==1:identity=identity[:,None]
    zero=sum(torch.nan_to_num(output[e+'_logits'].float()).sum()*0 for e in ('feature','raw'))
    total=zero;stats={}
    for e in ('feature','raw'):
        valid=base*output[e+'_support'].bool()
        count=(valid*known*(confidence>0)).sum().clamp_min(1)
        weight=valid*known*confidence*identity
        logit=torch.where(weight>0,output[e+'_logits'].float(),torch.zeros_like(target))
        err=F.binary_cross_entropy_with_logits(logit,target,pos_weight=target.new_tensor(float(pos_weight)),reduction='none')
        bce=(err*weight).sum()/count
        p=logit.sigmoid();positive=(target*weight).sum()
        dice=torch.where(positive>0,1-(2*(p*target*weight).sum()+1)/((p*weight).sum()+positive+1),bce*0)
        boundary=zero
        for j,side in enumerate(('start','end')):
            truth=batch[side+'_target'].float();v=valid*clean(batch['boundary_mask'][...,j])*torch.isfinite(truth)
            bw=v*torch.where(truth>0,torch.ones_like(confidence),confidence)*identity
            z=torch.where(bw>0,output[e+'_'+side+'_logits'].float(),torch.zeros_like(truth))
            err=F.binary_cross_entropy_with_logits(z,torch.nan_to_num(truth).clamp(0,1),pos_weight=z.new_tensor(float(boundary_pos_weight[j])),reduction='none')
            boundary=boundary+(err*bw).sum()/(bw>0).sum().clamp_min(1)/2
        smooth=zero
        if float(tc.get('temporal_smooth_weight',0))>0:
            same=(target[:,1:]==target[:,:-1])&known[:,1:]&known[:,:-1]
            continuous=batch['frame_start_ms'][:,1:]==batch['frame_end_ms'][:,:-1]
            good=valid[:,1:]*valid[:,:-1]*same*continuous
            boundary_exclusion=(batch['start_target'].abs()+batch['end_target'].abs())>0
            # Exclude +/- two frames around annotated boundaries.
            buffer=F.max_pool1d(boundary_exclusion[:,None].float(),5,1,2).squeeze(1)>0
            good*=~(buffer[:,1:]|buffer[:,:-1])
            sw=good*torch.minimum(confidence[:,1:],confidence[:,:-1])*torch.minimum(identity.expand_as(base)[:,1:],identity.expand_as(base)[:,:-1])
            smooth=((p[:,1:]-p[:,:-1]).square()*sw).sum()/(good>0).sum().clamp_min(1)
        branch=bce+float(tc.get('dice_weight',.05))*dice+float(tc.get('boundary_weight',.2))*boundary+float(tc.get('temporal_smooth_weight',0))*smooth
        total=total+branch
        # Diagnostic means on known, supported frames; weak negatives retain their
        # existing label confidence in the objective. These are not precision/F1.
        negative=valid*known*(confidence>0)*(target==0)
        positive_mask=valid*known*(confidence>0)*(target==1)
        stats[e+'_background_probability']=(p*negative).sum().detach()/negative.sum().clamp_min(1)
        stats[e+'_positive_probability']=(p*positive_mask).sum().detach()/positive_mask.sum().clamp_min(1)
        stats.update({e+'_bce':bce.detach(),e+'_dice':dice.detach(),e+'_boundary':boundary.detach(),e+'_smooth':smooth.detach(),e+'_loss':branch.detach(),e+'_valid_frames':(valid*known*(confidence>0)).sum().detach(),e+'_weighted_frames':weight.sum().detach()})
    prediction=output['raw_feature_prediction'].float();truth=batch['physiology_features'].float()
    m=base[...,None]*output['raw_support'][...,None]*clean(batch['physiology_mask'])*torch.isfinite(truth)
    q=clean(output['raw_observation_quality'].detach())[...,None]
    err=F.smooth_l1_loss(torch.where(m>0,prediction,torch.zeros_like(prediction)),torch.nan_to_num(truth.detach()),reduction='none')
    physiology=(err*m*q*identity[...,None]).sum()/m.sum().clamp_min(1)
    total=total+float(tc.get('physiology_aux_weight',.02))*physiology
    stats['physiology_aux']=physiology.detach();stats['loss']=total.detach()
    return total,stats
