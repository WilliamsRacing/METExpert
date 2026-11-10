
from __future__ import annotations
import copy, json, math, os, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from .events import CANDIDATE_FEATURE_NAMES, DEFAULT_RULE, generate_candidates, candidate_features, candidate_training_targets, event_interval, interval_iou, select_candidates


def _plain(value):
    if isinstance(value,dict):return {str(k):_plain(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [_plain(v) for v in value]
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    return value


def _write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(_plain(value),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');os.replace(temp,path)


class EventRanker(nn.Module):
    def __init__(self,input_dim=None,architecture='logistic',hidden=16,dropout=.1):
        super().__init__();self.input_dim=input_dim or len(CANDIDATE_FEATURE_NAMES)
        if architecture not in ('logistic','mlp'):raise ValueError('Unsupported event architecture')
        self.architecture=architecture
        self.body=(nn.Identity() if architecture=='logistic' else nn.Sequential(
            nn.Linear(self.input_dim,hidden),nn.LayerNorm(hidden),nn.GELU(),nn.Dropout(dropout)))
        dim=self.input_dim if architecture=='logistic' else hidden
        self.existence=nn.Linear(dim,1);self.quality=nn.Linear(dim,1)

    def forward(self,x):
        if x.shape[-1]!=self.input_dim or not bool(torch.isfinite(x).all()):raise ValueError('Invalid event features')
        h=self.body(x.float())
        return dict(logits=self.existence(h).squeeze(-1),quality_logits=self.quality(h).squeeze(-1))


def model_scores(out,quality_power=.25):
    return out['logits'].sigmoid()*out['quality_logits'].sigmoid().pow(float(quality_power))


def ranker_loss(out,target,quality,groups,quality_weight=.2,ranking_weight=.1):
    """Natural candidate frequency: no positive-class weight or resampling."""
    loss=F.binary_cross_entropy_with_logits(out['logits'],target)
    qloss=F.mse_loss(out['quality_logits'].sigmoid(),quality)
    pair=(groups[:,None]==groups[None,:])&(groups[:,None]>=0)
    difference=quality[:,None]-quality[None,:];pair &= difference>.10
    if bool(pair.any()):
        gap=out['quality_logits'][:,None]-out['quality_logits'][None,:]
        ranking=(F.softplus(-gap[pair])*difference[pair]).mean()
    else:ranking=loss*0
    return loss+float(quality_weight)*qloss+float(ranking_weight)*ranking


def _scale(x):
    mean=x.mean(0,dtype=np.float64).astype(np.float32)
    scale=np.maximum(x.std(0,dtype=np.float64),.05).astype(np.float32)
    return mean,scale


def _candidate_bank(sessions,labels,rule):
    bank=[];features=[];targets=[];qualities=[];groups=[];owners=[];validity=[]
    group_map={};people=set()
    for session in sessions:
        metadata=session.get('metadata',{});person=metadata.get('person')
        if person is None:raise ValueError('Event training evidence lacks person')
        if metadata.get('group')=='development':raise ValueError('Original dev6 cannot fit/select ranker')
        if metadata.get('imu_oof') is False:raise ValueError('Event fitting requires IMU identity exclusion')
        people.add(person);candidates=generate_candidates(session,rule)
        x=candidate_features(session,candidates)
        y,valid,quality,truth_groups=candidate_training_targets(session,candidates,labels.participant_events(person))
        for j,truth in enumerate(truth_groups):
            if truth>=0:
                key=(person,int(truth));group_map.setdefault(key,len(group_map));truth_groups[j]=group_map[key]
        begin=sum(len(z) for z in targets)
        bank.append(dict(person=person,candidates=candidates,start=begin,count=len(candidates),valid=valid))
        features.append(x);targets.append(y);qualities.append(quality);groups.append(truth_groups)
        owners.extend([person]*len(candidates));validity.append(valid)
    if not features or not sum(map(len,features)):raise ValueError('No ranker candidates')
    return (bank,np.concatenate(features),np.concatenate(targets),np.concatenate(qualities),
            np.concatenate(groups),np.asarray(owners),np.concatenate(validity),sorted(people))


def _truth_scope(sessions,labels):
    by_person={}
    for session in sessions:by_person.setdefault(session['metadata']['person'],[]).append(session)
    truth={}
    for person,own in by_person.items():
        eligible=[]
        for e in labels.participant_events(person):
            start,end=event_interval(e);visible=False
            for s in own:
                ticks=np.asarray(s['ticks']).reshape(-1)
                a,b=np.searchsorted(ticks,[math.ceil(start*15/1000),math.ceil(end*15/1000)])
                if b>a and np.any((s['known'].astype(bool)&s['observed'].astype(bool)).reshape(-1)[a:b]):
                    visible=True;break
            if visible:eligible.append((start,end))
        truth[person]=eligible
    return truth


def match_counts(events,truth,people=None):
    """Maximum cardinality then IoU, with the official strict >.25 rule."""
    people=sorted(truth) if people is None else list(people);tp=fp=fn=0;per_person={}
    for person in people:
        predicted=[e for e in events if e.get('participant_key')==person];actual=truth[person]
        n,m=len(predicted),len(actual)
        if n and m:
            iou=np.asarray([[interval_iou((e['start_ms'],e['end_ms']),t) for t in actual] for e in predicted])
            legal=iou>.25;utility=legal*(min(n,m)+1+iou)
            row,col=linear_sum_assignment(-utility);matched=int(legal[row,col].sum())
        else:matched=0
        own=dict(TP=matched,FP=n-matched,FN=m-matched)
        own['F1']=2*matched/max(2*matched+own['FP']+own['FN'],1);per_person[person]=own
        tp+=own['TP'];fp+=own['FP'];fn+=own['FN']
    return dict(TP=tp,FP=fp,FN=fn,F1=2*tp/max(2*tp+fp+fn,1),per_person=per_person)


def _selected(bank,scores,threshold,settings,people=None):
    allowed=set(people) if people is not None else None;events=[]
    for row in bank:
        if allowed is not None and row['person'] not in allowed:continue
        own=scores[row['start']:row['start']+row['count']]
        events.extend(select_candidates(row['candidates'],own,threshold=threshold,
            nms_iou=float(settings.get('nms_iou',.5)),nms_ios=float(settings.get('nms_ios',.8))))
    return events


def grouped_folds(people,count=4,seed=20261029,atomic_groups=()):
    people=set(people);groups=[];seen=set()
    for group in atomic_groups:
        group=sorted(people&set(group))
        if group:
            if seen&set(group):raise ValueError('Atomic event groups overlap')
            groups.append(group);seen.update(group)
    groups.extend([[p] for p in sorted(people-seen)])
    if len(groups)<2:raise ValueError('Ranker selection needs at least two independent groups')
    rng=np.random.default_rng(seed);rng.shuffle(groups);groups.sort(key=lambda g:-len(g))
    folds=[[] for _ in range(min(int(count),len(groups)))]
    for group in groups:folds[min(range(len(folds)),key=lambda i:len(folds[i]))].extend(group)
    return [sorted(group) for group in folds]


def _fit(x,y,q,groups,fit_mask,config,settings,seed,epochs,device,checkpoints=()):
    torch.manual_seed(int(seed));mean,scale=_scale(x[fit_mask])
    bx=torch.from_numpy(((x[fit_mask]-mean)/scale).clip(-20,20)).to(device)
    by=torch.from_numpy(y[fit_mask]).to(device);bq=torch.from_numpy(q[fit_mask]).to(device)
    bg=torch.from_numpy(groups[fit_mask]).to(device)
    model=EventRanker(architecture=config['architecture'],hidden=int(settings.get('hidden',16)),dropout=float(settings.get('dropout',.1))).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=float(settings.get('lr',.01)),weight_decay=float(config['weight_decay']))
    batch_size=int(settings.get('batch_size',128));snapshots={};history=[]
    for epoch in range(1,int(epochs)+1):
        model.train();rng=torch.Generator().manual_seed(seed+epoch)
        order=torch.randperm(len(bx),generator=rng);values=[]
        for first in range(0,len(order),batch_size):
            ids=order[first:first+batch_size].to(device)
            optimizer.zero_grad(set_to_none=True);out=model(bx[ids])
            loss=ranker_loss(out,by[ids],bq[ids],bg[ids],settings.get('quality_weight',.2),settings.get('ranking_weight',.1))
            if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite ranker loss')
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step();values.append(float(loss.detach()))
        history.append(dict(epoch=epoch,loss=float(np.mean(values))))
        if epoch in checkpoints:snapshots[epoch]={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    return model,mean,scale,snapshots,history


@torch.inference_mode()
def _scores(model,x,mean,scale,device,quality_power):
    model.eval();result=[]
    for begin in range(0,len(x),512):
        batch=torch.from_numpy(((x[begin:begin+512]-mean)/scale).clip(-20,20)).to(device)
        result.append(model_scores(model(batch),quality_power).cpu().numpy())
    return np.concatenate(result) if result else np.empty(0,np.float32)


def fit_ranker(training_sessions,labels,settings,seed,output_dir):
    """Fit/select only original train people; return final model and safe payload.

    The selected threshold maximizes held-person event F1, tie-breaking by fewer
    FP then a higher threshold.  Thresholds and small model grids are declared
    in settings.  Old dev6 or held test evidence is rejected before fitting.
    """
    output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=True)
    settings=dict(settings);rule=DEFAULT_RULE|dict(settings.get('candidate_rule',{}))
    forbidden=set(settings.get('forbidden_people',[]))
    # Reject ownership before proposal generation, feature or annotation access.
    requested={s.get('metadata',{}).get('person') for s in training_sessions}
    if None in requested:raise ValueError('Event training evidence lacks person')
    if requested&forbidden:raise ValueError('Forbidden identity in event fitting')
    expected=settings.get('fit_people')
    if expected is not None and set(expected)!=requested:raise ValueError('Event fit identity scope changed')
    bank,x,y,q,groups,owners,valid,people=_candidate_bank(training_sessions,labels,rule)
    if not valid.any() or not ((y[valid]>.5).any() and (y[valid]<.5).any()):raise ValueError('Event ranker needs both known candidate classes')
    folds=grouped_folds(people,settings.get('selection_folds',4),seed,settings.get('atomic_groups',[['HNU21028','HNU21029']]))
    truth=_truth_scope(training_sessions,labels)
    epochs=sorted(set(int(e) for e in settings.get('epoch_candidates',[15,30,50])))
    if not epochs or min(epochs)<1:raise ValueError('Invalid ranker epoch candidates')
    thresholds=[float(t) for t in settings.get('threshold_candidates',[.35,.5,.65,.8,.9])]
    minimum_precision=float(settings.get('minimum_selection_precision',0.))
    if not 0<=minimum_precision<=1:raise ValueError('Invalid minimum selection precision')
    if minimum_precision>0 and 1. not in thresholds:thresholds.append(1.)
    device=torch.device(settings.get('device','cpu'));quality_power=float(settings.get('quality_power',.25))
    configs=[dict(architecture=a,weight_decay=float(w))
        for a in settings.get('architectures',['logistic','mlp']) for w in settings.get('weight_decay_candidates',[.01,.1])]
    trials=[];best=None
    _write(output_dir/'sampling.json',dict(candidates=len(y),valid=int(valid.sum()),positive=int((y[valid]>.5).sum()),
        negative=int((y[valid]<.5).sum()),pos_weight=1.,resampling=False,people=people,selection_groups=folds,
        candidate_source='IMU8 + exact original PPG v9r2; same generator at fitting and deployment'))
    for ci,config in enumerate(configs):
        predictions={e:np.full(len(x),np.nan,np.float32) for e in epochs}
        for fi,held in enumerate(folds):
            held_mask=np.isin(owners,held);fit_mask=(~held_mask)&valid
            if not fit_mask.any():raise ValueError('Empty ranker fit fold')
            model,mean,scale,states,_=_fit(x,y,q,groups,fit_mask,config,settings,seed+ci*100+fi,max(epochs),device,epochs)
            for epoch,state in states.items():
                model.load_state_dict(state);predictions[epoch][held_mask]=_scores(model,x[held_mask],mean,scale,device,quality_power)
        for epoch,score in predictions.items():
            if not np.isfinite(score).all():raise RuntimeError('Incomplete held-person event scores')
            for threshold in thresholds:
                report=match_counts(_selected(bank,score,threshold,settings),truth)
                precision=report['TP']/max(report['TP']+report['FP'],1)
                empty=report['TP']+report['FP']==0
                eligible=bool(empty or precision>=minimum_precision)
                record=dict(config=config,epochs=epoch,threshold=threshold,report=report,
                    precision=precision,eligible=eligible,minimum_selection_precision=minimum_precision,empty_rescue=empty)
                trials.append(record)
                # Simpler logistic and fewer epochs are preferred on exact ties.
                rank=(eligible,report['F1'],-report['FP'],threshold,config['architecture']=='logistic',-epoch,-config['weight_decay'])
                if best is None or rank>best['rank']:best=dict(rank=rank,record=record,scores=score.copy())
        _write(output_dir/'selection_trials.json',dict(trials=trials,selection='Approximate grouped development selection on fixed IMU-OOF evidence; not strict upstream nested validation'))
        print(json.dumps(dict(phase='training_v8_event_ranker_selection',architecture=config['architecture'],weight_decay=config['weight_decay'],best_f1=best['record']['report']['F1'])),flush=True)
    chosen=best['record']
    model,mean,scale,_,history=_fit(x,y,q,groups,valid,chosen['config'],settings,seed+9000,chosen['epochs'],device)
    payload=dict(format='fusion-v8.event-ranker.1',model_config=dict(input_dim=len(CANDIDATE_FEATURE_NAMES),
        architecture=chosen['config']['architecture'],hidden=int(settings.get('hidden',16)),dropout=float(settings.get('dropout',.1))),
        state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},mean=mean.tolist(),scale=scale.tolist(),
        feature_names=list(CANDIDATE_FEATURE_NAMES),threshold=chosen['threshold'],selected_epochs=chosen['epochs'],
        selected_weight_decay=chosen['config']['weight_decay'],quality_power=quality_power,settings=_plain(settings),
        candidate_rule=rule,nms_iou=float(settings.get('nms_iou',.5)),nms_ios=float(settings.get('nms_ios',.8)),
        fit_people=people,selection_people=people,forbidden_people=sorted(forbidden),selection_groups=folds,
        selection_report=chosen['report'],selection='Approximate grouped train-only development selection; upstream IMU OOF models may have fitted other held-group identities',
        selection_criterion='Maximum held-person event F1 among candidates meeting the predeclared precision minimum; threshold 1 disables rescue as conservative fallback',
        minimum_selection_precision=minimum_precision,selection_precision=chosen['precision'],rescue_enabled=chosen['threshold']<1,
        full_oof=False,imu_oof=True,ppg_oof=False,ppg_mode='exact_original_frozen_v9_r2',pos_weight=1.,
        positive_candidates=int((y[valid]>.5).sum()),negative_candidates=int((y[valid]<.5).sum()),candidate_count=len(y),valid_candidates=int(valid.sum()),
        independent_test_evaluated=False,ppg_training=False)
    torch.save(payload,output_dir/'component.pt');_write(output_dir/'history.json',history)
    _write(output_dir/'selection.json',{k:v for k,v in payload.items() if k!='state_dict'})
    _write(output_dir/'progress.json',dict(phase='complete',epochs=chosen['epochs'],selected_architecture=chosen['config']['architecture']))
    return model.eval(),payload


@torch.inference_mode()
def predict_ranker(session,candidates,payload,device='cpu'):
    if payload.get('format')!='fusion-v8.event-ranker.1':raise ValueError('Unsupported ranker component')
    if payload['feature_names']!=list(CANDIDATE_FEATURE_NAMES):raise ValueError('Event feature contract changed')
    x=candidate_features(session,candidates);device=torch.device(device)
    model=EventRanker(**payload['model_config']).to(device);model.load_state_dict(payload['state_dict']);model.eval()
    return _scores(model,x,np.asarray(payload['mean'],np.float32),np.asarray(payload['scale'],np.float32),device,payload.get('quality_power',.25))
