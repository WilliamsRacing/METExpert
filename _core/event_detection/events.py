
from __future__ import annotations
import math
import numpy as np


DEFAULT_RULE = dict(threshold=.4, low_threshold=.15, merge_gap_seconds=60.,
                    min_duration_seconds=90.)
_BASE_NAMES = (
    'log_duration','source_imu','source_ppg',
    'imu_mean','imu_max','imu_p90','imu_std','ppg_mean','ppg_max','ppg_p90','ppg_std',
    'imu_coverage','ppg_coverage','imu_high_fraction','ppg_high_fraction',
    'imu_low_fraction','ppg_low_fraction','imu_longest_low_fraction','ppg_longest_low_fraction',
    'disagreement','conditional_same','conditional_different',
    'imu_start_boundary','imu_end_boundary','ppg_start_boundary','ppg_end_boundary',
    'ppg_quality_raw_max','ppg_quality_expert','raw_coverage','spectral_coverage','feature_coverage',
    *(f'imu_quality_{j}' for j in range(8)),
    'imu_local_mean','imu_context_mean','imu_local_present','imu_context_present')
_SHAPE_NAMES = tuple(f'{source}_third{j}_{kind}'
                     for source in ('imu','ppg') for j in range(3)
                     for kind in ('mean','coverage'))
_CONTEXT_NAMES = tuple(f'{source}_{side}_{seconds}s_{kind}'
                       for seconds in (120,300) for side in ('before','after')
                       for source in ('imu','ppg') for kind in ('mean','coverage','inside_delta'))
_PEER_NAMES = ('other_source_max_iou','other_source_max_ios','other_source_count')
CANDIDATE_FEATURE_NAMES = _BASE_NAMES + _SHAPE_NAMES + _CONTEXT_NAMES + _PEER_NAMES


def sigmoid(x):
    return 1./(1.+np.exp(-np.clip(np.asarray(x,np.float64),-60.,60.)))


def interval_iou(a,b):
    inter=max(0.,min(a[1],b[1])-max(a[0],b[0]))
    union=max(a[1],b[1])-min(a[0],b[0])
    return inter/union if union>0 else 0.


def interval_ios(a,b):
    inter=max(0.,min(a[1],b[1])-max(a[0],b[0]))
    denominator=min(a[1]-a[0],b[1]-b[0])
    return inter/denominator if denominator>0 else 0.


def _runs(mask):
    change=np.diff(np.r_[False,np.asarray(mask,bool),False].astype(np.int8))
    return zip(np.flatnonzero(change==1),np.flatnonzero(change==-1))


def _segments(session, key, n, points):
    ids=np.asarray(session.get(key,session['segment_ids']),np.int64)
    if ids.shape==(n,):ids=np.broadcast_to(ids[:,None],(n,points))
    if ids.shape!=(n,points):raise ValueError('Physical segment IDs must align: '+key)
    return ids.reshape(-1)


def expert_arrays(session):
    ticks=np.asarray(session['ticks'],np.int64)
    if ticks.ndim!=2 or not len(ticks):raise ValueError('ticks must be nonempty [F,L]')
    n,points=ticks.shape;flat=ticks.reshape(-1)
    if np.any(np.diff(flat)<=0):raise ValueError('Candidate clocks must be ordered and unique')
    physical=np.asarray(session.get('physical_mask',np.ones_like(ticks,bool)),bool)
    if physical.shape!=ticks.shape:raise ValueError('Physical mask must align with ticks')
    imask=np.asarray(session['imu_support'],bool)&physical
    pmask=np.broadcast_to(np.asarray(session['ppg_support'],bool)[:,None],ticks.shape)&physical
    if imask.shape!=ticks.shape:raise ValueError('IMU support must be per physical point')
    il=np.asarray(session['imu_logits'])
    if il.shape==(n,):il=np.broadcast_to(il[:,None],ticks.shape)
    pl=np.broadcast_to(np.asarray(session['ppg_logits'])[:,None],ticks.shape)
    if il.shape!=ticks.shape:raise ValueError('IMU logits must be [F] or [F,L]')
    if not np.isfinite(il[imask]).all() or not np.isfinite(pl[pmask]).all():
        raise ValueError('Nonfinite supported expert output')
    isid=_segments(session,'imu_segment_ids',n,points)
    psid=_segments(session,'ppg_segment_ids',n,points)
    imask=imask.reshape(-1)&(isid>=0);pmask=pmask.reshape(-1)&(psid>=0)
    ip=np.where(imask,sigmoid(il.reshape(-1)),0.)
    pp=np.where(pmask,sigmoid(pl.reshape(-1)),0.)
    return flat,ip,pp,imask,pmask,isid,psid


def _decode(ticks,p,support,sid,source,rule):
    high=float(rule['threshold']);low=float(rule['low_threshold'])
    gap=float(rule['merge_gap_seconds']);minimum=float(rule['min_duration_seconds'])
    if not 0<=low<=high<=1 or min(gap,minimum)<0:raise ValueError('Invalid candidate rule')
    result=[]
    for left,right in _runs(support&(sid>=0)):
        cuts=np.flatnonzero((np.diff(ticks[left:right])!=1)|(np.diff(sid[left:right])!=0))+left+1
        edges=np.r_[left,cuts,right]
        for a,b in zip(edges[:-1],edges[1:]):
            spans=[]
            for lo,hi in _runs(p[a:b]>=low):
                lo+=a;hi+=a
                if not np.any(p[lo:hi]>=high):continue
                if spans and ticks[lo]-ticks[spans[-1][1]-1]-1<=gap*15:
                    spans[-1]=(spans[-1][0],hi)
                else:spans.append((lo,hi))
            for lo,hi in spans:
                start,end=int(ticks[lo]),int(ticks[hi-1])+1
                if (end-start)/15<minimum:continue
                result.append(dict(start_tick=start,end_tick=end,start_ms=round(start*1000/15),
                    end_ms=round(end*1000/15),score=float(p[lo:hi].mean()),source=source,
                    sources=[source],source_segments={source:int(sid[lo])}))
    return result


def generate_candidates(session,rule=None):
    """Only IMU8 and the exact frozen PPG outputs can create candidates."""
    rule=DEFAULT_RULE|dict(rule or {})
    ticks,ip,pp,imask,pmask,isid,psid=expert_arrays(session)
    items=_decode(ticks,ip,imask,isid,'imu',rule)+_decode(ticks,pp,pmask,psid,'ppg',rule)
    meta=session.get('metadata',{});person=meta.get('person',session.get('person'))
    session_key=meta.get('recording_id',meta.get('session_index',meta.get('continuous_id')))
    unique={}
    for item in items:
        item.update(participant_key=person,session_key=session_key)
        key=item['start_tick'],item['end_tick']
        if key not in unique:unique[key]=item
        else:
            old=unique[key];old['sources']=sorted(set(old['sources']+item['sources']))
            old['source']='+'.join(old['sources']);old['score']=max(old['score'],item['score'])
            old['source_segments'].update(item['source_segments'])
    return sorted(unique.values(),key=lambda e:(e['start_tick'],e['end_tick']))


def _longest(mask,ticks,sid):
    mask=np.asarray(mask,bool)
    if not mask.any():return 0
    cut=(np.diff(ticks)!=1)|(np.diff(sid)!=0)
    # Run lengths with an explicit false separator at each physical discontinuity.
    edge=np.r_[True,cut]|~mask
    indices=np.arange(len(mask));last=np.maximum.accumulate(np.where(edge,indices,-1))
    length=np.where(mask,indices-last+np.where(edge,1,0),0)
    # A false predecessor resets the next sample to a run of one.
    starts=mask & np.r_[True,(~mask[:-1])|cut]
    last=np.maximum.accumulate(np.where(starts,indices,-1))
    return int(np.where(mask,indices-last+1,0).max())


def _mean(values,valid):
    selected=np.asarray(values)[valid]
    return float(selected.mean()) if len(selected) else 0.


def candidate_features(session,candidates):
    """No label, known, observed or true-role field is inspected here."""
    ticks,ip,pp,imask,pmask,isid,psid=expert_arrays(session)
    n,points=session['ticks'].shape;output=[]
    frame_imask=imask.reshape(n,points).any(-1)
    frame_pmask=pmask.reshape(n,points).any(-1)
    def frame_field(key,mask,fa,fb,width=None,fallback=None):
        if key not in session:
            if fallback and fallback in session:key=fallback
            else:return np.zeros(width,np.float64) if width else 0.
        value=np.nan_to_num(np.asarray(session[key],np.float64),nan=0.,posinf=0.,neginf=0.)
        if value.shape[0]!=n:raise ValueError('Candidate field has incorrect frame dimension: '+key)
        if value.ndim==2 and width is None:value=value.mean(-1)
        valid=mask[fa:fb];own=value[fa:fb]
        return own[valid].mean(0) if valid.any() else (np.zeros(width) if width else 0.)
    def boundary(key,fa,fb,mask,side):
        if key not in session:return 0.
        value=np.asarray(session[key]);value=value[:,side] if value.ndim==2 else value
        a,b=(max(0,fa-1),min(n,fa+2)) if side==0 else (max(0,fb-2),min(n,fb+1))
        source_sid=isid if key.startswith('imu') else psid
        source_valid=imask if key.startswith('imu') else pmask
        own=np.flatnonzero(source_valid[fa*points:fb*points])
        if not len(own):return 0.
        anchor=fa*points+int(own[0 if side==0 else -1])
        same=(source_sid.reshape(n,points)==source_sid[anchor]).any(-1)
        valid=mask[a:b]&same[a:b];v=sigmoid(value[a:b])
        return float(v[valid].max()) if valid.any() else 0.
    for c in candidates:
        start=int(c.get('start_tick',round(c['start_ms']*15/1000)))
        end=int(c.get('end_tick',round(c['end_ms']*15/1000)))
        lo,hi=np.searchsorted(ticks,[start,end]);duration=max(end-start,1)
        fa,fb=lo//points,min(n,(hi+points-1)//points)
        x,y=ip[lo:hi],pp[lo:hi];a,b=imask[lo:hi],pmask[lo:hi]
        def stats(v,valid):
            z=v[valid]
            return [float(z.mean()),float(z.max()),float(np.quantile(z,.9)),float(z.std())] if len(z) else [0.]*4
        inside_means=(_mean(x,a),_mean(y,b))
        overlap=a&b
        role=frame_field('imu_role_prob',frame_imask,fa,fb,2)
        quality=frame_field('imu_quality',frame_imask,fa,fb,8)
        source=set(c['sources'])
        row=[math.log1p(duration/15)/math.log1p(3600),'imu' in source,'ppg' in source,
             *stats(x,a),*stats(y,b),float(a.sum()/duration),float(b.sum()/duration),
             float(((x>=.5)&a).sum()/duration),float(((y>=.5)&b).sum()/duration),
             float(((x>=.2)&a).sum()/duration),float(((y>=.2)&b).sum()/duration),
             _longest((x>=.2)&a,ticks[lo:hi],isid[lo:hi])/duration,
             _longest((y>=.2)&b,ticks[lo:hi],psid[lo:hi])/duration,
             _mean(np.abs(x-y),overlap),*role,
             boundary('imu_boundary_logits',fa,fb,frame_imask,0),boundary('imu_boundary_logits',fa,fb,frame_imask,1),
             boundary('ppg_start_logits',fa,fb,frame_pmask,0),boundary('ppg_end_logits',fa,fb,frame_pmask,1),
             frame_field('ppg_quality_raw_max',frame_pmask,fa,fb),
             frame_field('ppg_quality_expert',frame_pmask,fa,fb,fallback='ppg_quality'),
             frame_field('raw_coverage',frame_pmask,fa,fb),
             frame_field('spectral_coverage',frame_pmask,fa,fb,fallback='wave_coverage'),
             frame_field('feature_coverage',frame_pmask,fa,fb),*quality]
        for key in ('imu_local_logits','imu_context_logits'):
            row.append(float(sigmoid(frame_field(key,frame_imask,fa,fb))) if key in session else 0.)
        row.extend([float('imu_local_logits' in session),float('imu_context_logits' in session)])
        # Three ordered bins retain shape rather than just candidate mean.
        for values,valid in ((ip,imask),(pp,pmask)):
            for j in range(3):
                left=start+duration*j//3;right=start+duration*(j+1)//3
                u,v=np.searchsorted(ticks,[left,right]);mask=valid[u:v]
                row.extend([_mean(values[u:v],mask),float(mask.sum()/max(right-left,1))])
        # A before/after window can only use the same source physical run that
        # touches the corresponding candidate edge.  It cannot cross a hole.
        for seconds in (120,300):
            for side in ('before','after'):
                left,right=(start-seconds*15,start) if side=='before' else (end,end+seconds*15)
                u,v=np.searchsorted(ticks,[left,right])
                for mi,(values,valid,sid) in enumerate(((ip,imask,isid),(pp,pmask,psid))):
                    own=np.flatnonzero(valid[lo:hi])
                    anchor=lo+int(own[0 if side=='before' else -1]) if len(own) else None
                    mask=valid[u:v]&(sid[u:v]==sid[anchor]) if anchor is not None else np.zeros(v-u,bool)
                    # A time hole with reused IDs still prevents context jumps.
                    if anchor is not None:
                        if side=='before':
                            between=np.flatnonzero((np.diff(ticks[u:anchor+1])!=1)|(np.diff(sid[u:anchor+1])!=0)|(~valid[u:anchor]))
                            if len(between):mask[:min(len(mask),int(between[-1])+1)]=False
                        else:
                            between=np.flatnonzero((np.diff(ticks[anchor:v])!=1)|(np.diff(sid[anchor:v])!=0)|(~valid[anchor+1:v]))
                            if len(between):mask[max(0,anchor+int(between[0])+1-u):]=False
                    mean=_mean(values[u:v],mask);coverage=float(mask.sum()/max(right-left,1))
                    row.extend([mean,coverage,inside_means[mi]-mean if mask.any() else 0.])
        others=[other for other in candidates if other is not c and set(other['sources'])-source]
        pair=(c['start_ms'],c['end_ms'])
        row.extend([max([interval_iou(pair,(o['start_ms'],o['end_ms'])) for o in others] or [0.]),
                    max([interval_ios(pair,(o['start_ms'],o['end_ms'])) for o in others] or [0.]),
                    math.log1p(sum(interval_iou(pair,(o['start_ms'],o['end_ms']))>0 for o in others))])
        if len(row)!=len(CANDIDATE_FEATURE_NAMES) or not np.isfinite(row).all():
            raise ValueError(f'Invalid candidate features: {len(row)}/{len(CANDIDATE_FEATURE_NAMES)}')
        output.append(row)
    return np.asarray(output,np.float32).reshape(-1,len(CANDIDATE_FEATURE_NAMES))


def event_interval(event):
    start=event.get('start_ms',event.get('start_epoch_ms'))
    end=event.get('end_ms',event.get('end_epoch_ms'))
    if start is None or end is None or end<=start:raise ValueError('Invalid truth interval')
    return float(start),float(end)


def candidate_training_targets(session,candidates,events,iou_threshold=.25,min_negative_known=.95):
    """Training labels only.  Positive truncated observations stay usable.

    Partial records are not converted to negative background: negatives require
    >=95% reliable known observation.  Existence, best IoU, and truth IDs remain
    separate so duplicate alternatives can receive a quality ranking loss.
    """
    person=session.get('metadata',{}).get('person',session.get('person'))
    events=[e for e in events if e.get('participant_key',e.get('person',person))==person]
    intervals=[event_interval(e) for e in events]
    ticks=np.asarray(session['ticks'],np.int64).reshape(-1)
    known=(np.asarray(session['known'],bool)&np.asarray(session['observed'],bool)).reshape(-1)
    physical=np.asarray(session.get('physical_mask',session['observed']),bool).reshape(-1)
    target=np.zeros(len(candidates),np.float32);valid=np.zeros(len(candidates),bool)
    quality=np.zeros(len(candidates),np.float32);groups=np.full(len(candidates),-1,np.int64)
    for i,c in enumerate(candidates):
        pair=(c['start_ms'],c['end_ms']);ious=np.asarray([interval_iou(pair,e) for e in intervals])
        if len(ious):quality[i]=float(ious.max());groups[i]=int(ious.argmax()) if quality[i]>0 else -1
        target[i]=quality[i]>iou_threshold
        lo,hi=np.searchsorted(ticks,[math.ceil(pair[0]*15/1000),math.ceil(pair[1]*15/1000)])
        duration=max((pair[1]-pair[0])*15/1000,1.)
        known_mass=int(known[lo:hi].sum());observed_fraction=physical[lo:hi].sum()/duration
        valid[i]=(target[i]>.5 and known_mass>0 and observed_fraction>=.1) or (known_mass/duration>=min_negative_known)
    return target,valid,quality,groups


def _same_context(a,b,strict_session=True):
    pa=a.get('participant_key',a.get('person'));pb=b.get('participant_key',b.get('person'))
    if pa is not None and pb is not None and pa!=pb:return False
    sa,sb=a.get('session_key'),b.get('session_key')
    return not (strict_session and sa is not None and sb is not None and sa!=sb)


def duplicated(a,b,nms_iou=.5,nms_ios=.8,strict_session=True):
    if not _same_context(a,b,strict_session):return False
    aa=(a['start_ms'],a['end_ms']);bb=(b['start_ms'],b['end_ms'])
    return interval_iou(aa,bb)>nms_iou or interval_ios(aa,bb)>=nms_ios


def select_candidates(candidates,scores,threshold=.5,nms_iou=.5,nms_ios=.8):
    scores=np.asarray(scores,np.float64)
    if scores.shape!=(len(candidates),) or not np.isfinite(scores).all():raise ValueError('Scores must align')
    if not 0<=threshold<=1 or not 0<nms_iou<=1 or not 0<nms_ios<=1:raise ValueError('Invalid selection thresholds')
    # One is a declared conservative fallback: no rescue events, including a
    # float32 sigmoid that has numerically saturated at exactly one.
    if threshold==1:return []
    selected=[]
    order=sorted(range(len(candidates)),key=lambda i:(-scores[i],candidates[i]['start_ms'],candidates[i]['end_ms']))
    for i in order:
        if scores[i]<threshold:continue
        event=dict(candidates[i],score=float(scores[i]))
        if any(duplicated(event,old,nms_iou,nms_ios) for old in selected):continue
        selected.append(event)
    return sorted(selected,key=lambda e:(str(e.get('participant_key','')),e['start_ms'],e['end_ms']))


def merge_anchor_rescue(anchor_events,rescue_events,nms_iou=.5,nms_ios=.8):
    """Anchor dictionaries and boundaries are preserved, including their score.

    The anchor has priority across physically overlapping native records; rescue
    candidates are already ranked without labels.  No truth ID is consulted.
    """
    anchor=[dict(e) for e in anchor_events];rescue=[]
    for event in sorted(rescue_events,key=lambda e:(-float(e.get('score',0)),e['start_ms'])):
        if any(duplicated(event,old,nms_iou,nms_ios,strict_session=False) for old in anchor+rescue):continue
        rescue.append(dict(event))
    return sorted(anchor+rescue,key=lambda e:(str(e.get('participant_key','')),e['start_ms'],e['end_ms']))
