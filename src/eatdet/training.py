"""Train the complete detector from prepared competition recordings.

The public recipe fixes the selected training budgets and all operating rules.
Only training identities fit normalization, PCA, neural weights and event scores.
The optical expert uses a separate validation group for its independent EMA
snapshot selection. The output is the same five-component inference layout as
the supplied model directory.
"""
from __future__ import annotations

import copy
import gc
import hashlib
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    temporary.replace(path)


def save_weights(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp');torch.save(value,temporary);temporary.replace(path)


def _seed(seed):
    random.seed(int(seed));np.random.seed(int(seed));torch.manual_seed(int(seed))
    if torch.cuda.is_available():torch.cuda.manual_seed_all(int(seed))


def _default_config():
    return read_json(Path(__file__).resolve().parents[2]/'configs/training.json')


def load_training_inputs(prepared):
    """Validate identity ownership before opening any numerical recording."""
    if isinstance(prepared,dict):p=dict(prepared)
    else:
        path=Path(prepared)
        if path.is_dir():path=path/'prepared.json'
        p=read_json(path)
        for key in ('imu_manifest','ppg_cache_root','annotation_root','split_file','native_root'):
            if p.get(key) and not Path(p[key]).is_absolute():p[key]=str((path.parent/p[key]).resolve())
    for key in ('imu_manifest','ppg_cache_root','annotation_root','split_file','native_root'):
        if not p.get(key):raise ValueError('Training preparation requires '+key)
    split=read_json(p['split_file']);train=set(split.get('train',[]));val=set(split.get('val',[]));test=set(split.get('test',[]))
    if not train or not val or train&val or (train|val)&test:
        raise ValueError('Nonempty disjoint training, validation and test identity lists are required')
    if len(split['train'])!=len(train) or len(split['val'])!=len(val):raise ValueError('Duplicate split identity')
    groups=split.get('shared_source_atomic_groups',[])
    for group in groups:
        if sum(bool(set(group)&s) for s in (train,val,test))>1:
            raise ValueError('A shared-source identity group crosses data splits')
    if not (Path(p['annotation_root'])/'events.csv').is_file():raise FileNotFoundError('Annotations require events.csv')
    return p,split


def train_optical(prepared, split, config, folder, destination, device, smoke=False):
    """Fit numeric PCA, both optical experts, EMA snapshots and optical fusion."""
    from train_optical import fit_model
    from signals.optical.data import load_records, validate_splits
    from signals.optical.postprocess_v9 import fallback_postprocessor
    c=copy.deepcopy(config)
    if smoke:
        c['training'].update(epochs=1,finetune_epochs=0,minimum_epochs=1,num_workers=0)
    # Avoid loading protected test recordings in the trainer's record enumeration.
    records=[r for r in load_records(prepared['ppg_cache_root']) if r['subject_id'] in set(split['train'])|set(split['val'])]
    validate_splits(dict(split,test=[]),records)
    path=fit_model(records,split,Path(prepared['ppg_cache_root']),folder,c,device,debug_batches=1 if smoke else None)
    checkpoint=torch.load(path,map_location='cpu',weights_only=True)
    # The optical expert's deployed rule is a fixed, uncalibrated quality-aware
    # mixture. It is part of the published configuration rather than retuned by
    # the downstream multimodal training stages.
    state=fallback_postprocessor(c);candidate=copy.deepcopy(next(iter(state['candidates'].values())))
    candidate.update(alpha=.3,calibrated=False,quality_correction=True)
    if 'candidate_id' in candidate:candidate['candidate_id']='U1'
    state.update(selected_candidate='U1',candidates={'U1':candidate},fusion_weights=[.3,.7])
    checkpoint['postprocess']=state
    payload={key:checkpoint[key] for key in ('format','model_state','config','scaler','expert_epochs','postprocess')}
    payload['config']={'model':checkpoint['config']['model']}
    save_weights(destination,payload)
    del checkpoint;gc.collect()
    if device.type=='cuda':torch.cuda.empty_cache()
    return destination


def train_primary_imu(prepared, split, config, destination, device, amp, smoke=False):
    from motion_encoder.model import IMUV6Model, ModelConfig
    from motion_encoder.data import SequenceDataset, fit_normalization
    from motion_encoder.train import training_loss, augment_training, on_device, model_forward, worker_init
    from context_encoder.data import base_manifest
    c=copy.deepcopy(config);_seed(c['seed'])
    manifest=base_manifest(prepared['imu_manifest']);manifest['split']=copy.deepcopy(split)
    manifest['records']=[r for r in manifest['records'] if r['person'] in set(split['train'])|set(split['val'])]
    manifest['normalization']=fit_normalization(manifest,c['normalization_points_per_person'],c['seed'])
    dataset=SequenceDataset(manifest,split='train',frames=c['train_frames'],training=True,
        samples_per_epoch=min(c['samples_per_epoch'],c['batch_size']) if smoke else c['samples_per_epoch'],seed=c['seed'])
    workers=0 if smoke else int(c.get('workers',0))
    loader=DataLoader(dataset,batch_size=c['batch_size'],shuffle=False,num_workers=workers,
        pin_memory=device.type=='cuda',multiprocessing_context='spawn' if workers else None,worker_init_fn=worker_init)
    model=IMUV6Model(ModelConfig(**c.get('model',{}))).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=c['lr'],weight_decay=c['weight_decay'])
    epochs=1 if smoke else int(c['epochs']);history=[]
    try:
        for epoch in range(1,epochs+1):
            dataset.set_epoch(epoch);model.train();losses=[]
            factor=min(1.,epoch/2)*(.1+.9*.5*(1+math.cos(math.pi*(epoch-1)/max(1,int(c.get('schedule_epochs',30))-1))))
            for group in optimizer.param_groups:group['lr']=c['lr']*factor
            for supplied in loader:
                batch=on_device(supplied,device);batch=augment_training(batch,manifest['normalization'])
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=amp):
                    out=model_forward(model,batch);loss,parts=training_loss(out,batch,c['core_frames'],**c['loss'])
                if not parts['frames']:continue
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite primary IMU loss')
                loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step();losses.append(float(loss.detach()))
            if not losses:raise ValueError('No supervised primary IMU training frames')
            history.append(dict(epoch=epoch,loss=float(np.mean(losses))))
            print(json.dumps(dict(stage='primary_imu',**history[-1])),flush=True)
        save_weights(destination,dict(format='imu-v6.1',state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},
            model_config=model.config.to_dict(),normalization=manifest['normalization'],fit_people=split['train'],epoch=epochs))
        return history
    finally:
        dataset.close();del model
        if device.type=='cuda':torch.cuda.empty_cache()


def train_context_imus(prepared, split, config, folder, destination, device, amp, seed, smoke=False):
    from event_detection.common import make_folds
    from context_encoder.train import train_component
    c=copy.deepcopy(config);budgets=c.pop('fold_epochs');full_budget=int(c.pop('full_epochs'))
    folds=make_folds(split,len(budgets),seed);checkpoints={}
    for fold in folds+[dict(fold_id=len(folds),train_people=split['train'],prediction_people=split['val'])]:
        fi=fold['fold_id'];fit=list(fold['train_people'])
        forbidden=split['val']+split.get('test',[])+(fold['prediction_people'] if fi<len(folds) else [])
        settings=dict(c,base_manifest=prepared['imu_manifest'],train_people=fit,selection_people=[],
            forbidden_people=sorted(set(forbidden)),annotation_root=prepared['annotation_root'],
            output_root=str(folder/('full' if fi==len(folds) else f'group_{fi+1}')),
            seed=seed+fi*101+17,device=str(device),amp=amp,cpu_threads=torch.get_num_threads(),
            epochs=1 if smoke else int(full_budget if fi==len(folds) else budgets[fi]),schedule_epochs=30)
        if smoke:settings.update(samples_per_epoch=settings['batch_size'],workers=0)
        receipt=train_component(settings);source=Path(receipt['component_path'])
        if fi==len(folds):
            ck=torch.load(source,map_location='cpu',weights_only=True)
            save_weights(destination,{key:ck[key] for key in ('format','complete','state_dict','model_config','normalization','fit_people','epoch')})
        else:checkpoints[fi]=source
    return folds,checkpoints


class FusionEvidenceDataset(Dataset):
    def __init__(self,sessions,core_frames=128):
        self.records=[]
        for session in sessions:
            source=session['primary_evidence'];n=len(source['ppg_logits'])
            for start in range(0,n,int(core_frames)):
                self.records.append({key:value[start:start+core_frames].copy() for key,value in source.items()
                    if isinstance(value,np.ndarray) and value.ndim and len(value)==n})
    def __len__(self):return len(self.records)
    def __getitem__(self,index):return self.records[index]


def train_primary_fusion(sessions,config,destination,device,smoke=False):
    from primary_fusion.model import FusionModel
    from primary_fusion.train import collate, FIELDS
    c=dict(config);_seed(c['seed']);model=FusionModel(dropout=c['dropout']).to(device)
    dataset=FusionEvidenceDataset(sessions,c.get('core_frames',128))
    if not len(dataset):raise ValueError('No primary fusion training evidence')
    generator=torch.Generator();loader=DataLoader(dataset,batch_size=c['batch_size'],shuffle=True,collate_fn=collate,generator=generator)
    optimizer=torch.optim.AdamW(model.parameters(),lr=c['lr'],weight_decay=c['weight_decay']);history=[]
    for epoch in range(1,(1 if smoke else int(c['epochs']))+1):
        generator.manual_seed(c['seed']+epoch);model.train();losses=[]
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            out=model(**{k:batch[k].to(device) for k in FIELDS})
            mask=batch['known'].to(device).bool()&out['supported'];target=batch['target'].to(device).float()
            if not bool(mask.any()):continue
            loss=F.binary_cross_entropy_with_logits(out['logits'][mask],target[mask])
            loss+=c['calibration_penalty']*(model.temperature.log().square().mean()+model.bias.square().mean())
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite primary fusion loss')
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step();losses.append(float(loss.detach()))
            if smoke:break
        if not losses:raise ValueError('No supervised primary fusion frames')
        history.append(dict(epoch=epoch,loss=float(np.mean(losses))))
    save_weights(destination,dict(format='imu-v6-ppgv9-fusion.1',state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},
        config=dict(dropout=c['dropout']),epoch=len(history)))
    return history


def train_event_ranker(sessions,labels,config,destination,device,seed,smoke=False):
    from event_detection.events import CANDIDATE_FEATURE_NAMES, DEFAULT_RULE
    from event_detection.ranker import _candidate_bank, _fit
    c=dict(config);rule=DEFAULT_RULE|dict(c.get('candidate_rule',{}));bank,x,y,q,groups,owners,valid,people=_candidate_bank(sessions,labels,rule)
    if not valid.any() or not ((y[valid]>.5).any() and (y[valid]<.5).any()):
        raise ValueError('Event fitting requires known positive and negative candidates')
    architecture=dict(architecture=c['architecture'],weight_decay=c['weight_decay']);epochs=1 if smoke else int(c['epochs'])
    model,mean,scale,_,history=_fit(x,y,q,groups,valid,architecture,c,seed+9000,epochs,device)
    payload=dict(format='fusion-v8.event-ranker.1',state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},
        model_config=dict(input_dim=len(CANDIDATE_FEATURE_NAMES),architecture=c['architecture'],hidden=c['hidden'],dropout=c['dropout']),
        mean=mean.tolist(),scale=scale.tolist(),feature_names=list(CANDIDATE_FEATURE_NAMES),threshold=c['threshold'],
        candidate_rule=rule,nms_iou=c['nms_iou'],nms_ios=c['nms_ios'],quality_power=c['quality_power'])
    save_weights(destination,payload)
    return dict(epochs=epochs,candidates=len(y),positive=int((y[valid]>.5).sum()),negative=int((y[valid]<.5).sum()),history=history)


def train_model(prepared,output_root,config=None,*,device='cuda:0',workers=None,smoke=False,ppg_checkpoint=None):
    """Execute optical, primary IMU, context OOF/full and fusion training."""
    from .inference import build_sessions
    from signals.competition.annotations import CompetitionAnnotations
    p,split=load_training_inputs(prepared);c=_default_config() if config is None else (read_json(config) if not isinstance(config,dict) else copy.deepcopy(config))
    device=torch.device(device);amp=bool(c.get('amp',True) and device.type=='cuda')
    if device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('Requested CUDA device is unavailable')
    if amp and not torch.cuda.is_bf16_supported():raise RuntimeError('BF16 requires a supported CUDA device')
    if workers is not None:
        for key in ('primary_imu','context_imu'):c[key]['workers']=int(workers)
        c['ppg']['training']['num_workers']=int(workers)
    torch.set_num_threads(int(c.get('cpu_threads',8)));torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    root=Path(output_root).resolve();weights=root/'weights'
    if (root/'complete.json').exists() or any((weights/name).exists() for name in ('imu_primary.pt','imu_context.pt','primary_fusion.pt','event_ranker.pt')):
        raise FileExistsError('Choose a fresh training output directory')
    weights.mkdir(parents=True,exist_ok=True);started=time.time()
    write_json(root/'training_config.json',c)
    optical=weights/'ppg.pt';primary=weights/'imu_primary.pt';context=weights/'imu_context.pt';fusion=weights/'primary_fusion.pt';ranker=weights/'event_ranker.pt'
    if ppg_checkpoint:
        ck=torch.load(ppg_checkpoint,map_location='cpu',weights_only=True)
        if set(ck['scaler']['fit_subjects'])!=set(split['train']):raise ValueError('Optical PCA ownership differs from training split')
        save_weights(optical,ck)
    else:train_optical(p,split,c['ppg'],root/'training/optical',optical,device,smoke)
    primary_history=train_primary_imu(p,split,c['primary_imu'],primary,device,amp,smoke)
    primary_sessions=list(build_sessions(p,primary_checkpoint=primary,ppg_checkpoint=optical,people=split['train'],device=str(device),amp=amp,include_labels=True))
    fusion_history=train_primary_fusion(primary_sessions,c['primary_fusion'],fusion,device,smoke)
    del primary_sessions;gc.collect()
    folds,checkpoints=train_context_imus(p,split,c['context_imu'],root/'training/context',context,device,amp,c['seed'],smoke)
    sessions=[]
    for fold in folds:
        rows=list(build_sessions(p,context_checkpoint=checkpoints[fold['fold_id']],ppg_checkpoint=optical,
            people=fold['prediction_people'],device=str(device),amp=amp,include_labels=True))
        for row in rows:
            row['metadata'].update(group='imu_oof_frozen_ppg',imu_oof=True,ppg_oof=False,fold_id=fold['fold_id'])
        sessions.extend(rows)
    labels=CompetitionAnnotations(p['annotation_root'])
    ranker_receipt=train_event_ranker(sessions,labels,c['event_ranker'],ranker,torch.device('cpu'),c['seed'],smoke)
    del sessions;gc.collect()
    names=dict(ppg='ppg.pt',imu_primary='imu_primary.pt',imu_context='imu_context.pt',primary_fusion='primary_fusion.pt',event_ranker='event_ranker.pt')
    manifest=dict(format='eating-detection.weights.1',decoder=c['decoder'],components={key:dict(path='weights/'+name,sha256=hashlib.sha256((weights/name).read_bytes()).hexdigest(),bytes=(weights/name).stat().st_size)
        for key,name in names.items()})
    write_json(weights/'manifest.json',manifest)
    receipt=dict(completed=True,smoke_only=bool(smoke),weights=str(weights),elapsed_seconds=time.time()-started,
        primary_imu_epochs=len(primary_history),primary_fusion_epochs=len(fusion_history),context_folds=len(folds),
        event_ranker={k:v for k,v in ranker_receipt.items() if k!='history'})
    write_json(root/'complete.json',receipt)
    return receipt
