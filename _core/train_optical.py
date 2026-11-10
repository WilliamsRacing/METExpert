
from __future__ import annotations
from data_pipeline import PROJECT  # Make this standalone entry use the shared runtime.
import argparse, copy, gc, json, math, os, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.utils.checkpoint import checkpoint as activation_checkpoint
from signals.optical.common import read_json, seed_everything, choose_device, load_checkpoint, model_inputs, prepare_device_batch, cache_identity_aliases, validate_cache_config
from signals.optical.data import load_records, validate_splits, canonical_frame_masks, fit_v8_pca, SequenceDataset
from signals.optical.feature_pca import FeaturePCA
from signals.optical.model import build_model
from signals.optical.losses import estimate_loss_weights
from signals.optical.losses_v9 import segmentation_loss_v9, combine_loss_batches, LOSS_OUTPUT_KEYS
from signals.optical.expert_checkpoints import ExpertCheckpointBank
from signals.optical.selection_v9 import select_experts
from signals.optical.inference import infer_frames, event_report, save_predictions
from signals.optical.loader_v9 import inference_loader
from signals.optical.training_v9 import atomic_json, atomic_torch, cache_fingerprint, code_fingerprint, payload_hash, sha256, capture_rng, restore_rng, ModelEMA, StepSchedule, precision_policy, runtime_versions, inference_state_fingerprint


def resolve_config(config,diagnostic=False):
    c=copy.deepcopy(config);t=c['training']
    if c['model'].get('version')!='v9':raise ValueError('Use an explicit V9 configuration')
    micro=int(t['batch_size']);acc=int(t.get('gradient_accumulation',1));eff=micro*acc
    if eff!=8 and not diagnostic:raise ValueError('V9 requires GLOBAL effective batch8')
    if min(micro,acc,int(t['epochs']))<1:raise ValueError('Positive batch/accumulation/epochs required')
    if int(t.get('minimum_epochs',24))>int(t['epochs']) and not diagnostic:raise ValueError('minimum_epochs exceeds main epochs')
    if c['model'].get('causal'):raise ValueError('V9 uses a noncausal model with explicit halo')
    halo=(c['sequence_frames']-c['core_frames'])//2
    required=max(sum(c['model']['dilations']),sum(c['model']['short_dilations']))
    if halo<required or int(c['inference']['halo_frames'])<required:raise ValueError('Training/inference halo is smaller than TCN support')
    if t.get('loss_normalization','valid_count')!='valid_count':raise ValueError('V9 requires valid_count loss normalization')
    if c['features'].get('whiten',False):raise ValueError('V9 unwhitened PCA contract')
    h=c.get('hard_examples',{})
    if int(h.get('maximum_extra_per_event',2))!=2 or float(h.get('min_positive_transition_fraction',.5))!=.5:raise ValueError('V9 fixed replay caps are2/event and at least50%positive/transition')
    if int(c.get('crossfit',{}).get('groups',3))!=3:raise ValueError('V9 crossfit uses3 groups')
    if float(c.get('postprocess',{}).get('f1_tolerance',.005))!=.005:raise ValueError('V9 fixed near-peak F1 tolerance is.005')
    t['effective_batch_size']=eff;t['ema_decay_resolved']=float(t.get('ema_reference_decay',.99))**(eff/4)
    if c['features'].get('availability_mode','loading_fraction')!=c['model'].get('pca_availability_mode','loading_fraction'):raise ValueError('PCA/model availability contracts differ')
    if not 0<float(c['features']['pca_variance'])<=1:raise ValueError('Invalid PCA variance')
    r2=c['model'].get('revision')=='r2_precision_repair'
    calibration=c.get('postprocess',{}).get('calibration_enabled',not r2)
    if c.get('postprocess',{}).get('enabled',True) and calibration and not c.get('crossfit',{}).get('enabled',True):
        raise ValueError('Calibrated postprocessing requires training-identity crossfit')
    if r2 and c.get('postprocess',{}).get('event_verifier',False):
        raise ValueError('r2 event verifier is not part of the validated candidate policy')
    c['resolved_plan']='2.0' if r2 else '1.1';c['diagnostic_only']=bool(diagnostic)
    return c


def close_loader(loader):
    iterator=getattr(loader,'_iterator',None)
    if iterator is not None:iterator._shutdown_workers();loader._iterator=None
    close=getattr(getattr(loader,'dataset',None),'close',None)
    if close:close()


def subject_factors(records,ownership):
    counts={}
    for rec,owned in zip(records,ownership):
        with np.load(rec['_path'],allow_pickle=False) as z:
            count=int((owned&(z['target']>=0)&(z['label_weight']>0)).sum())
        counts[rec['subject_id']]=counts.get(rec['subject_id'],0)+count
    total=sum(counts.values());s=len(counts)
    weights={k:float(np.clip(total/(s*max(v,1)),.5,2.)) for k,v in counts.items()}
    return weights,counts


def _stage_optimizer(model,c,stage,ordinary_updates):
    t=c['training'];fine=stage=='finetune';h=c.get('hard_examples',{})
    epochs=int(t.get('finetune_epochs',10) if fine else t['epochs'])
    frac=float(h.get('finetune_fraction',.25) if fine else h.get('replay_fraction',.2))
    enabled=bool(h.get('enabled',True));start=int(h.get('start_epoch',11))
    # Predeclared upper bound uses successful updates; actual replay quotas can be smaller.
    normal_steps=ordinary_updates*epochs
    replay_epochs=epochs if fine else max(0,epochs-start+1)
    extra=math.ceil(ordinary_updates*frac/(1-frac))*replay_epochs if enabled else 0
    peak=float(t.get('finetune_lr',1.5e-5) if fine else t['lr'])
    minimum=float(t.get('finetune_min_lr',1.5e-6) if fine else t.get('min_lr',2e-6))
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=peak,weight_decay=float(t['weight_decay']))
    warmup=ordinary_updates*float(t.get('finetune_warmup_epochs',1) if fine else t.get('warmup_epochs',3))
    scheduler=StepSchedule(optimizer,peak,minimum,normal_steps+extra,math.ceil(warmup))
    return optimizer,scheduler


def fit_model(records,splits,cache,run,c,device,*,auxiliary=False,initial_crossfit=None,debug_batches=None):
    """Fit one main or fixed-schedule auxiliary model; returns selected artifact path."""
    from signals.optical.hard_examples_v9 import IdentityReplayBatchSampler, mine_hard_examples
    run=Path(run);run.mkdir(parents=True,exist_ok=True)
    c=resolve_config(c,debug_batches is not None);t=c['training'];h=c.get('hard_examples',{})
    train_records=[r for r in records if r['subject_id'] in splits['train']]
    validation_records=[r for r in records if r['subject_id'] in splits['val']]
    if not train_records:raise ValueError('No training identity records')
    if set(splits['train'])&set(splits['val']):raise ValueError('Leaking training/selection identities')
    if not auxiliary and not validation_records:raise ValueError('Main model requires disjoint inner validation identities')
    ckpath=run/'last.pt';resume=load_checkpoint(ckpath) if ckpath.exists() else None
    if (run/'best.pt').exists() and resume is None:raise ValueError('best.pt without complete resumable last.pt; inspect run')
    fingerprint=cache_fingerprint(cache);code=code_fingerprint()
    if resume:
        for key,value in [('config',c),('splits',splits),('cache_fingerprint',fingerprint),('code_fingerprint',code),('auxiliary',auxiliary),('debug_batches',debug_batches)]:
            if resume.get(key)!=value:raise ValueError('Unsafe V9 resume drift: '+key)
    seed_everything(int(c['seed']))
    if not (run/'config_requested.json').exists():atomic_json(run/'config_requested.json',c)
    atomic_json(run/'config_resolved.json',c)
    atomic_json(run/'config.json',c);atomic_json(run/'splits.json',splits)
    ownership=canonical_frame_masks(train_records)
    scaler=FeaturePCA.from_dict(resume['scaler']) if resume else fit_v8_pca(train_records,ownership,
        variance=float(c['features']['pca_variance']),availability_mode=c['features'].get('availability_mode','loading_fraction'))
    if set(scaler.fit_subjects)!=set(splits['train']):raise ValueError('PCA fit identities differ from model training identities')
    atomic_json(run/'pca.json',scaler.to_dict());atomic_json(run/'scaler.json',scaler.to_dict())
    atomic_json(run/'feature_schema.json',{'input_names':scaler.names,'output_names':scaler.output_names,'output_dim':scaler.output_dim,'fit_subjects':scaler.fit_subjects,'variance':c['features']['pca_variance'],'outer_labels_used':False})
    model=build_model(c['model'],scaler.output_dim).to(device)
    if not resume and c.get('self_supervised',{}).get('enabled',False):
        from pretrain_optical import pretrain_for_subjects
        pretrained=pretrain_for_subjects(cache,train_records,splits['train'],c,run/'self_supervised',device)
        pretrained=load_checkpoint(pretrained)
        if set(pretrained['train_subjects'])!=set(splits['train']):raise ValueError('SSL identity leakage')
        model.waveform.pulse.load_state_dict(pretrained['pulse_state'])
    pw,bw,balance=estimate_loss_weights(train_records,float(t.get('max_pos_weight',4)),frame_masks=ownership,
        boundary_maximum=float(t.get('max_boundary_pos_weight',15)),positive_power=float(t.get('pos_weight_power',1.)))
    identity,counts=subject_factors(train_records,ownership)
    ds=SequenceDataset(train_records,scaler,c['sequence_frames'],c['core_frames'],False,True,
        lru_records=int(t.get('lru_records',4)),frame_masks=ownership)
    if not len(ds):raise ValueError('No owned supervised cores')
    sampler=IdentityReplayBatchSampler(ds,batch_size=int(t['batch_size']),seed=int(c['seed']),effective_batch_size=int(t['effective_batch_size']))
    if not resume and initial_crossfit is not None and h.get('enabled',True):
        pool=mine_hard_examples(initial_crossfit,train_records,ds,splits['train'],source='training_crossfit',model_version='v9',decoder=c['decoder'],minimum_observed_fraction=float(h.get('minimum_observed_fraction',.8)),boundary_buffer_s=float(h.get('negative_boundary_buffer_s',120)),provenance={'excluded_subjects':splits['train'],'crossfit_source':'verified_inner_crossfit_provenance.json'})
        atomic_json(run/'hard_examples'/'initial_crossfit_pool.json',pool)
    else:pool=resume.get('hard_pool',[]) if resume else []
    workers=int(t.get('num_workers',4));kwargs={'persistent_workers':True,'prefetch_factor':int(t.get('prefetch_factor',2))} if workers else {}
    # Workers are recreated deterministically per epoch with seed/epoch; persistent workers are retained.
    generator=torch.Generator().manual_seed(int(c['seed']))
    dl=DataLoader(ds,batch_sampler=sampler,num_workers=workers,pin_memory=device.type=='cuda',generator=generator,**kwargs)
    val_loader=inference_loader(validation_records,scaler,c,device) if not auxiliary else None
    mining_loader=None
    use_amp,amp_dtype,scale_amp=precision_policy(c,device)
    amp=torch.amp.GradScaler('cuda',enabled=scale_amp)
    ema=ModelEMA(model,float(t.get('ema_reference_decay',.99)),4)
    bank=None if auxiliary else ExpertCheckpointBank(model,float(t.get('expert_selection_min_delta',1e-4)))
    ordinary_updates=math.ceil(len(ds)/int(t['effective_batch_size']))
    stage='main';stage_done=0;global_epoch=0;history=[];optimizer,scheduler=_stage_optimizer(model,c,stage,ordinary_updates)
    metadata={'versions':runtime_versions(),'device':str(device),'precision':str(amp_dtype),'seed':int(c['seed']),
        'pca_payload_sha256':payload_hash(scaler.to_dict()),'parameters':sum(p.numel() for p in model.parameters()),
        'pos_weight':pw,'boundary_pos_weight':bw,'balance':balance,'identity_weights':identity,'identity_known_frames':counts,
        'cache_fingerprint':fingerprint,'code_fingerprint':code,'auxiliary_fixed_final_ema':auxiliary,'debug_only':debug_batches is not None,
        'optimizer_update_budget':{'ordinary_updates_per_epoch':ordinary_updates,'stage_schedule':scheduler.state_dict()},
        'resume_granularity':'complete_epoch; interrupted epoch is replayed from its saved RNG and hard pool',
        'loader_augmentation':'deterministic per sample epoch in V9; no mutable worker RNG requirement',
        'cpu_threads':torch.get_num_threads(),'no_outer_labels_for_training':True}
    if resume:
        model.load_state_dict(resume['model_state']);ema.load_state_dict(resume['ema_state']);stage=resume['stage']
        optimizer,scheduler=_stage_optimizer(model,c,stage,ordinary_updates)
        optimizer.load_state_dict(resume['optimizer_state']);scheduler.load_state_dict(resume['scheduler_state']);amp.load_state_dict(resume['amp_state'])
        if bank:bank.load_state_dict(resume['expert_checkpoint_bank'])
        sampler.load_state_dict(resume['sampler_state']);generator.set_state(resume['loader_generator_state'].cpu())
        stage_done=int(resume['stage_completed']);global_epoch=int(resume['epoch']);history=resume['history'];restore_rng(resume['rng_state'])
    if resume and (resume['metadata']['device']!=str(device) or resume['metadata']['precision']!=str(amp_dtype)):raise ValueError('Resume device/precision contract changed; use the same hardware mode')
    atomic_json(run/'run_metadata.json',metadata)
    def save_last(fit_complete=False):
        value={'format':'ppg-v9-r2-full-training-v1' if c['model'].get('revision')=='r2_precision_repair' else 'ppg-v9-full-training-v1','config':c,'splits':splits,'scaler':scaler.to_dict(),
            'cache_fingerprint':fingerprint,'code_fingerprint':code,'model_state':model.state_dict(),'ema_state':ema.state_dict(),
            'optimizer_state':optimizer.state_dict(),'scheduler_state':scheduler.state_dict(),'amp_state':amp.state_dict(),
            'rng_state':capture_rng(),'loader_generator_state':generator.get_state(),'sampler_state':sampler.state_dict(),
            'hard_pool':pool,'stage':stage,'stage_completed':stage_done,'epoch':global_epoch,'history':history,'metadata':metadata,
            'auxiliary':auxiliary,'debug_batches':debug_batches,'fit_complete':fit_complete,'identity_aliases':cache_identity_aliases(cache)}
        if bank:value['expert_checkpoint_bank']=bank.state_dict()
        atomic_torch(ckpath,value)
    def save_best():
        states=ema.model.state_dict() if auxiliary else bank.compose(ema.model)
        epochs={'feature':global_epoch,'raw':global_epoch} if auxiliary else bank.epochs()
        atomic_torch(run/'best.pt',{'format':'ppg-v9-r2-inference-v1' if c['model'].get('revision')=='r2_precision_repair' else 'ppg-v9-inference-v1','model_state':states,'config':c,'splits':splits,'scaler':scaler.to_dict(),
            'epoch':max(epochs.values()),'expert_epochs':epochs,'epoch_interpretation':'fixed_final_ema' if auxiliary else 'maximum_of_independently_selected_expert_epochs',
            'weights_source':'fixed_final_ema' if auxiliary else 'independent_best_ema','identity_aliases':cache_identity_aliases(cache),'metadata':metadata})
    if resume and resume.get('fit_complete'):
        # Rebuild the derived artifact from the atomically saved source-of-truth bank.
        valid_best=False
        if (run/'best.pt').exists():
            try:
                existing=load_checkpoint(run/'best.pt')
                expected=ema.model.state_dict() if auxiliary else bank.compose(ema.model)
                valid_best=(existing.get('config')==c and existing.get('splits')==splits and existing.get('scaler')==scaler.to_dict()
                    and set(existing['model_state'])==set(expected) and all(torch.equal(existing['model_state'][k].cpu(),v.cpu()) for k,v in expected.items()))
            except (OSError,RuntimeError,KeyError,ValueError):valid_best=False
        if not valid_best:save_best()
        for loader in (dl,val_loader):
            if loader is not None:close_loader(loader)
        return run/'best.pt'
    try:
        while True:
            budget=int(t.get('finetune_epochs',10) if stage=='finetune' else t['epochs'])
            stop=stage_done>=budget
            if stage=='main' and not auxiliary and stage_done>=int(t.get('minimum_epochs',24)) and bank.all_stalled(int(t.get('patience',12))):stop=True
            if stop:
                if stage=='main' and int(t.get('finetune_epochs',10))>0:
                    model.load_state_dict(ema.model.state_dict() if auxiliary else bank.compose(ema.model))
                    stage='finetune';stage_done=0;optimizer,scheduler=_stage_optimizer(model,c,stage,ordinary_updates)
                    ema=ModelEMA(model,float(t.get('ema_reference_decay',.99)),4);amp=torch.amp.GradScaler('cuda',enabled=scale_amp)
                    save_last();continue
                break
            replay_on=bool(h.get('enabled',True)) and (stage=='finetune' or stage_done>=int(h.get('start_epoch',11))-1)
            refresh=int(h.get('refresh_epochs',5))
            if replay_on and ((stage=='main' and (stage_done-(int(h.get('start_epoch',11))-1))%refresh==0) or (stage=='finetune' and stage_done%refresh==0)):
                # Initial crossfit pool is consumed first; subsequent refreshes only use current TRAIN identities.
                if not (initial_crossfit is not None and stage=='main' and stage_done==int(h.get('start_epoch',11))-1 and pool):
                    if mining_loader is None:mining_loader=inference_loader(train_records,scaler,c,device)
                    atomic_json(run/'progress.json',{'phase':'hard_mining','stage':stage,'epoch':global_epoch+1})
                    frame=infer_frames(ema.model,train_records,scaler,c,device,loader=mining_loader)
                    pool=mine_hard_examples(frame,train_records,ds,splits['train'],source='training_ema',model_version='v9',decoder=c['decoder'],minimum_observed_fraction=float(h.get('minimum_observed_fraction',.8)),boundary_buffer_s=float(h.get('negative_boundary_buffer_s',120)))
                atomic_json(run/'hard_examples'/f'pool_epoch_{global_epoch+1:03d}.json',pool)
            fraction=float(h.get('finetune_fraction',.25) if stage=='finetune' else h.get('replay_fraction',.2)) if replay_on else 0.
            sampler.set_pool(pool,replay_fraction=fraction,version=f'{stage}:{global_epoch+1}')
            sampler.set_epoch(global_epoch)
            if hasattr(ds,'set_epoch'):ds.set_epoch(global_epoch,int(c['seed']))
            model.train();optimizer.zero_grad(set_to_none=True);started=time.perf_counter();diagnostics=[];timing={'data_wait_s':0.,'h2d_ms':0.,'forward_ms':0.,'backward_ms':0.,'optimizer_ms':0.};cuda_times=[]
            iterator=iter(dl);n_batches=len(dl) if debug_batches is None else min(len(dl),debug_batches)
            done=0;updates=0;skipped=0;actual_samples=0
            if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
            while done<n_batches:
                group=[];outputs=[];wait=time.perf_counter()
                group_size=min(int(t.get('gradient_accumulation',1)),n_batches-done)
                for _ in range(group_size):
                    b=next(iterator);timing['data_wait_s']+=time.perf_counter()-wait
                    sid=[train_records[int(ri)]['subject_id'] for ri in b['record_index']]
                    b['identity_weight']=torch.tensor([identity[s] for s in sid],dtype=torch.float32)
                    if device.type=='cuda':
                        stamps=[torch.cuda.Event(enable_timing=True) for _ in range(3)];stamps[0].record()
                    db=prepare_device_batch(b,device)
                    if device.type=='cuda':stamps[1].record()
                    def forward_for_loss(current_batch):
                        with torch.autocast(device_type=device.type,dtype=amp_dtype,enabled=use_amp):
                            out=model(**model_inputs(current_batch,device))
                        return {k:out[k] for k in LOSS_OUTPUT_KEYS}
                    # Fallback micro4 x2 uses activation rematerialization; exact effective-batch loss statistics.
                    out=activation_checkpoint(forward_for_loss,db,use_reentrant=False,preserve_rng_state=True) if group_size>1 else forward_for_loss(db)
                    if device.type=='cuda':stamps[2].record()
                    group.append({k:v for k,v in db.items() if k not in ('spectrogram','raw','spectrogram_mask','raw_mask')});outputs.append(out)
                    done+=1;actual_samples+=len(sid);wait=time.perf_counter()
                    if device.type=='cuda':cuda_times.append((stamps,'forward'))
                loss_batch,loss_output=combine_loss_batches(group,outputs)
                with torch.autocast(device_type=device.type,enabled=False):loss,stats=segmentation_loss_v9(loss_output,loss_batch,c,pw,bw)
                if device.type=='cuda':
                    backward_start=torch.cuda.Event(enable_timing=True);backward_end=torch.cuda.Event(enable_timing=True);backward_start.record()
                amp.scale(loss).backward()
                if device.type=='cuda':
                    backward_end.record();cuda_times.append(((backward_start,backward_end),'backward'));opt_start=torch.cuda.Event(enable_timing=True);opt_end=torch.cuda.Event(enable_timing=True);opt_start.record()
                amp.unscale_(optimizer)
                for name,params in model.expert_parameter_groups().items():stats[name+'_grad_norm']=torch.nn.utils.clip_grad_norm_(params,float(t.get('grad_clip',1))).detach()
                prior=amp.get_scale();amp.step(optimizer);amp.update();success=not scale_amp or amp.get_scale()>=prior
                if success:
                    scheduler.step();ema.update(model,sum(x['target'].shape[0] for x in group));updates+=1
                else:skipped+=1
                optimizer.zero_grad(set_to_none=True)
                if device.type=='cuda':opt_end.record();cuda_times.append(((opt_start,opt_end),'optimizer'))
                # One compact device->CPU diagnostic transfer per effective update.
                keys=list(stats);vals=torch.stack([stats[k].float() for k in keys]).detach().cpu().tolist();diagnostics.append({k:(float(v) if math.isfinite(v) else None) for k,v in zip(keys,vals)})
                if diagnostics[-1]['loss'] is None:raise FloatingPointError('Nonfinite V9 loss; checkpoint retained at prior complete epoch')
                if done==group_size or done%int(t.get('progress_every',25))==0 or done==n_batches:
                    progress={'phase':'training','stage':stage,'epoch':global_epoch+1,'stage_epoch':stage_done+1,'batch':done,'total_batches':n_batches,'loss':diagnostics[-1]['loss'],'optimizer_updates':scheduler.steps,'elapsed_seconds':time.perf_counter()-started}
                    atomic_json(run/'progress.json',progress);print(json.dumps(progress),flush=True)
                del group,outputs,loss_batch,loss_output,out,db,loss
            if device.type=='cuda':
                torch.cuda.synchronize(device)
                for events,kind in cuda_times:
                    if kind=='optimizer':timing['optimizer_ms']+=events[0].elapsed_time(events[1])
                    elif kind=='backward':timing['backward_ms']+=events[0].elapsed_time(events[1])
                    else:
                        timing['h2d_ms']+=events[0].elapsed_time(events[1]);timing['forward_ms']+=events[1].elapsed_time(events[2])
                timing['peak_gpu_bytes']=torch.cuda.max_memory_allocated(device)
            global_epoch+=1;stage_done+=1
            entry={'epoch':global_epoch,'stage':stage,'stage_epoch':stage_done,'train_loss':float(np.mean([d['loss'] for d in diagnostics])),
                'diagnostics':{k:(float(np.mean([d[k] for d in diagnostics if d[k] is not None])) if any(d[k] is not None for d in diagnostics) else None) for k in diagnostics[0]},'optimizer_updates':updates,'skipped_amp_updates':skipped,
                'samples_exposed':actual_samples,'lr':optimizer.param_groups[0]['lr'],'ema_updates':ema.updates,'ema_samples':ema.samples,
                'timing':timing,'sampler':sampler.last_audit}
            if not auxiliary:
                atomic_json(run/'progress.json',{'phase':'validation','stage':stage,'epoch':global_epoch})
                vstart=time.perf_counter();vf=infer_frames(ema.model,validation_records,scaler,c,device,loader=val_loader)
                selections=select_experts(vf,validation_records,c);bank.update(ema.model,selections,global_epoch)
                metrics=event_report(vf,validation_records,c['decoder'])[0]
                entry.update(val_event_f1=float(metrics['F1']),val=metrics,expert_selection=selections,expert_checkpoints=bank.summary(),validation_seconds=time.perf_counter()-vstart)
                atomic_json(run/'expert_checkpoints.json',{'experts':bank.summary(),'expert_epochs':bank.epochs(),'fit_subjects':splits['val'],'outer_labels_used':False,'selection_score_note':'fixed-eight-rule mean F1, not final event F1'})
            entry['seconds']=time.perf_counter()-started;history.append(entry)
            atomic_json(run/'history.json',history);atomic_json(run/'hard_examples'/f'exposure_epoch_{global_epoch:03d}.json',sampler.last_audit)
            # Save source-of-truth online/EMA/bank/RNG before derived best.
            save_last();save_best()
            atomic_json(run/'progress.json',{'phase':'epoch_complete','epoch':global_epoch,'stage':stage,'train_loss':entry['train_loss'],'val_event_f1':entry.get('val_event_f1'),'expert_epochs':bank.epochs() if bank else None,'updated_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())})
            print(json.dumps({'epoch':global_epoch,'stage':stage,'train_loss':entry['train_loss'],'val_event_f1':entry.get('val_event_f1'),'seconds':entry['seconds']}),flush=True)
        save_last(fit_complete=True);save_best()
        atomic_json(run/'progress.json',{'phase':'model_fit_complete','epoch':global_epoch,'auxiliary':auxiliary,'expert_epochs':bank.epochs() if bank else {'feature':global_epoch,'raw':global_epoch}})
    finally:
        for loader in (dl,val_loader,mining_loader):
            if loader is not None:close_loader(loader)
    del model,ema,optimizer,bank;gc.collect()
    if device.type=='cuda':torch.cuda.empty_cache()
    return run/'best.pt'


def crossfit_models(records,splits,cache,run,c,device,debug_batches=None):
    root=Path(run)/'inner_crossfit';root.mkdir(parents=True,exist_ok=True)
    subjects=sorted(map(str,splits['train']));rng=np.random.default_rng(int(c['seed']));rng.shuffle(subjects)
    groups=[list(map(str,x)) for x in np.array_split(subjects,3)]
    if min(map(len,groups))<1:raise ValueError('Crossfit requires at least three training identities')
    frame_parts=[];proof=[]
    for i,held in enumerate(groups):
        trained=sorted(set(subjects)-set(held));folder=root/f'group_{i+1:02d}'
        aux_splits={'train':trained,'val':held,'test':sorted(set(splits['val'])|set(splits['test'])),'crossfit_fold':i,'scope':'train_identity_crossfit_fixed_final_ema'}
        ckpath=fit_model(records,aux_splits,cache,folder/'run',c,device,auxiliary=True,debug_batches=debug_batches)
        output=folder/'heldout_frame_probabilities.csv.gz';entry={'fold_id':i,'train_subjects':trained,'predict_subjects':held,'checkpoint_fingerprint':inference_state_fingerprint(load_checkpoint(ckpath)),'fixed_final_ema':True,'held_labels_used_for_training_or_selection':False}
        receipt=folder/'prediction_provenance.json'
        reusable=False
        if output.exists() and receipt.exists():
            previous=read_json(receipt)
            if {k:previous.get(k) for k in entry}!=entry:raise ValueError('Crossfit prediction provenance drift')
            reusable=previous.get('prediction_sha256')==sha256(output)
        if reusable:
            frame=pd.read_csv(output)
        else:
            ck=load_checkpoint(ckpath);scaler=FeaturePCA.from_dict(ck['scaler']);model=build_model(c['model'],scaler.output_dim).to(device);model.load_state_dict(ck['model_state'])
            held_records=[r for r in records if r['subject_id'] in held]
            frame=infer_frames(model,held_records,scaler,c,device);frame['crossfit_fold']=i
            temporary=output.with_name(output.name+'.tmp');frame.to_csv(temporary,index=False,compression='gzip');os.replace(temporary,output)
            atomic_json(receipt,{**entry,'prediction_sha256':sha256(output)});del model;gc.collect()
            if device.type=='cuda':torch.cuda.empty_cache()
        if set(frame.subject_id.astype(str))!=set(held):raise ValueError('Crossfit identity export mismatch')
        frame_parts.append(frame);proof.append(entry)
    joined=pd.concat(frame_parts,ignore_index=True).sort_values(['subject_id','start_ms'])
    if joined.duplicated(['subject_id','start_ms']).any():raise ValueError('Duplicate crossfit predictions')
    joined.to_csv(root/'frame_probabilities.csv.gz',index=False,compression='gzip')
    provenance={'training_subjects':subjects,'inner_subjects':list(splits['val']),'outer_subjects':list(splits['test']),'crossfit_folds':proof}
    atomic_json(root/'provenance.json',provenance)
    return joined,provenance






