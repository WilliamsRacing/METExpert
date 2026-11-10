
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .model import IMUV8Model, ModelConfig
from .data import SequenceDataset, base_manifest, prepare_fold_manifest


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)


def _now():
    return datetime.now(timezone.utc).isoformat()


def training_loss(out, batch, core_frames=16, role_weight=.1, dice_weight=.1,
                  local_weight=.2, context_weight=.2, joint_weight=.1,
                  boundary_weight=.05, minimum_supervision_coverage=.5):
    logits = out["logits"].float()
    valid = (batch["label_mask"].bool() & out["valid"]
             & (batch["point_mask"].float().mean(-1) >= minimum_supervision_coverage))
    if core_frames < 1 or core_frames > logits.shape[1]:
        raise ValueError("Invalid supervised core")
    first = (logits.shape[1]-core_frames)//2
    core = torch.zeros_like(valid)
    core[:, first:first+core_frames] = True
    valid &= core
    if not bool(valid.any()):
        zero = logits.sum()*0
        return zero, dict(total=0., bce=0., role=0., dice=0., boundary=0., frames=0)
    fraction = batch["target_fraction"].float().clamp(0, 1)
    bce = F.binary_cross_entropy_with_logits(logits[valid], fraction[valid])
    local_bce=F.binary_cross_entropy_with_logits(out['local_logits'].float()[valid],fraction[valid])
    context_bce=F.binary_cross_entropy_with_logits(out['context_logits'].float()[valid],fraction[valid])
    joint_bce=F.binary_cross_entropy_with_logits(out['joint_logits'].float()[valid],fraction[valid])
    role_mask = valid & ((batch["target"] == 1) | (batch["target"] == 2))
    role = (F.cross_entropy(out["role_logits"].float()[role_mask],
                            batch["target"][role_mask].long()-1)
            if bool(role_mask.any()) else out["role_logits"].sum()*0)
    p = torch.where(valid, logits.sigmoid(), torch.zeros_like(logits))
    y = torch.where(valid, fraction, torch.zeros_like(fraction))
    meal = y.sum(-1) > 0
    per_bag = 1-(2*(p*y).sum(-1)+1)/(p.sum(-1)+y.sum(-1)+1)
    dice = per_bag[meal].mean() if bool(meal.any()) else logits.sum()*0
    boundary = F.binary_cross_entropy_with_logits(
        out["boundary_logits"].float()[valid], batch["boundary"].float()[valid],
        pos_weight=logits.new_tensor([8., 8.]))
    total = bce+local_weight*local_bce+context_weight*context_bce+joint_weight*joint_bce+role_weight*role+dice_weight*dice+boundary_weight*boundary
    return total, dict(total=float(total.detach()), bce=float(bce.detach()),local_bce=float(local_bce.detach()),context_bce=float(context_bce.detach()),joint_bce=float(joint_bce.detach()),
                       role=float(role.detach()), dice=float(dice.detach()),
                       boundary=float(boundary.detach()), frames=int(valid.sum()))


def physical_augment(batch, normalization, max_degrees=10., rotation_probability=.8,
                     gain_std=.03, missing_frame_probability=.05):
    """One proper 3D installation rotation per whole sequence.

    Raw ACC, GYRO, gravity and linear ACC use the same proper rotation. ACC views
    share the same gain, preserving raw_ACC = gravity + linear_ACC. No reflection
    is used, and neither labels nor role metadata select a transform.
    """
    result = dict(batch)
    x = batch["imu"]
    b, t, c, length = x.shape
    if c != 12:
        raise ValueError("Physical multiview augmentation requires 12 channels")
    mean = x.new_tensor(normalization["mean"])[None, None, :, None]
    std = x.new_tensor(normalization["std"])[None, None, :, None]
    physical = x*std+mean
    axis = torch.randn(b, 3, device=x.device, dtype=x.dtype)
    axis = axis/axis.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    selected = torch.rand(b, device=x.device) < rotation_probability
    angle = (torch.rand(b, device=x.device, dtype=x.dtype)*2-1)*(float(max_degrees)*math.pi/180)
    angle = torch.where(selected, angle, torch.zeros_like(angle))
    zero = torch.zeros_like(axis[:, 0])
    a, d, e = axis.unbind(-1)
    skew = torch.stack((zero, -e, d, e, zero, -a, -d, a, zero), -1).reshape(b, 3, 3)
    identity = torch.eye(3, device=x.device, dtype=x.dtype)[None]
    outer = axis[:, :, None]*axis[:, None, :]
    cosine, sine = angle.cos()[:, None, None], angle.sin()[:, None, None]
    rotation = cosine*identity+(1-cosine)*outer+sine*skew
    groups = physical.reshape(b, t, 4, 3, length)
    groups = torch.einsum("bij,btgjl->btgil", rotation, groups)
    ga = 1+float(gain_std)*torch.randn(b, device=x.device, dtype=x.dtype)
    gg = 1+float(gain_std)*torch.randn(b, device=x.device, dtype=x.dtype)
    gain = torch.stack((ga, gg, ga, ga), -1)[:, None, :, None, None]
    physical = (groups*gain).reshape_as(x)
    mask = batch["point_mask"].clone()
    quality = batch["frame_quality"].clone()
    drop = torch.rand(b, device=x.device) < missing_frame_probability
    frame = torch.randint(t, (b,), device=x.device)
    for i in drop.nonzero(as_tuple=False).flatten().tolist():
        mask[i, frame[i]] = False
        quality[i, frame[i]] = 0
        quality[i, frame[i], 3] = 1
    result["imu"] = torch.where(mask[:, :, None], (physical-mean)/std, torch.zeros_like(x))
    result["point_mask"], result["frame_quality"] = mask, quality
    return result


def _batch(batch, device):
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
            for k, v in batch.items()}


def _forward(model, batch):
    return model(batch["imu"], batch["point_mask"], batch["frame_quality"], batch["segment_ids"])


def _worker_init(_):
    torch.set_num_threads(1)


def _payload(model, manifest, config, epoch, complete):
    return dict(
        format="imu-v8.component.1", component="imu", complete=bool(complete),
        state_dict=model.state_dict(), model_config=model.config.to_dict(),
        normalization=manifest["normalization"],
        fit_people=list(config["train_people"]),
        selection_people=list(config.get("selection_people", [])),
        selected_on="inner_train_identity_frame_bce" if config.get("selection_people") else "locked_epoch_budget_no_outer_validation",
        epoch=int(epoch), config=config,
        input_contract=dict(base_cache_signature=manifest.get("signature"),
                            frame_points=450, sample_rate_hz=15, input_channels=12,
                            ordered_subtokens=6, output_frame_seconds=30,
                            normalization_fit_people=manifest["normalization"]["people"]),
        role_order=["same", "different"], embedding_dim=model.config.embedding_dim,
        external_validation_used_for_fit=False, test_read=False,
    )


def _save(path, payload):
    path = Path(path)
    tmp = path.with_name(path.name+".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


@torch.inference_mode()
def _selection_bce(model, manifest, people, device, core=16, batch_size=8, amp=False):
    """Natural nonoverlapping cores from inner identities, person mean BCE.

    Selection has its own dataset but reuses training-owned normalization. No
    outer validation identity is accepted by prepare_fold_manifest upstream.
    """
    dataset=SequenceDataset(manifest,people=people,frames=core+2*model.temporal_radius,
        core_frames=core,training=False,normalization=manifest['normalization'])
    requests=[(ri,start,min(core,row['frame_count']-start)) for ri,row in enumerate(dataset.records)
              for start in range(0,row['frame_count'],core)]
    totals={};model.eval();h=model.temporal_radius
    try:
        for first in range(0,len(requests),batch_size):
            chunk=requests[first:first+batch_size]
            windows=[dataset.get_window(ri,start-h,core+2*h) for ri,start,count in chunk]
            batch={key:torch.stack([w[key] for w in windows]).to(device)
                   for key in ('imu','point_mask','frame_quality','segment_ids','label_mask','target_fraction')}
            with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=amp):out=_forward(model,batch)
            for j,(ri,start,count) in enumerate(chunk):
                sl=slice(h,h+count)
                valid=batch['label_mask'][j,sl].bool()&out['valid'][j,sl]&(batch['point_mask'][j,sl].float().mean(-1)>=.5)
                if not bool(valid.any()):continue
                losses=F.binary_cross_entropy_with_logits(out['logits'][j,sl].float()[valid],batch['target_fraction'][j,sl].float()[valid],reduction='sum')
                person=dataset.records[ri]['person'];old=totals.setdefault(person,[0.,0]);old[0]+=float(losses);old[1]+=int(valid.sum())
        per_person={p:loss/count for p,(loss,count) in totals.items() if count}
        if not per_person:raise RuntimeError('No known supported inner-selection frames')
        return dict(person_mean_bce=float(np.mean(list(per_person.values()))),per_person=per_person,
                    effective_frames=sum(v[1] for v in totals.values()),people=sorted(per_person),
                    selection_scope='original training identities only; no resampling')
    finally:dataset.close()


def train_component(config):
    """Fit requested epochs; optionally choose by inner training-identity BCE.

    Required config: base_manifest, train_people, output_root. selection_people
    defaults empty; both fit and inner-selection must be old training identities.
    Returns metadata including component_path, not a model or a large state dict.
    """
    config = json.loads(json.dumps(config, default=str))
    outroot = Path(config["output_root"])
    outroot.mkdir(parents=True, exist_ok=True)
    if any((outroot/name).exists() for name in ("component.pt", "complete.json", "last.pt")):
        raise FileExistsError("Independent component output already contains weights")
    lock = None
    if os.name == "posix":
        import fcntl
        lock = (outroot/"train.lock").open("a+")
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    seed = int(config.get("seed", 20261029))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(int(config.get("cpu_threads", 4)))
    epochs = int(config.get("epochs", 15))
    if epochs < 1 or epochs > 30:
        raise ValueError("Epoch budget must be between 1 and 30")
    manifest = prepare_fold_manifest(
        config["base_manifest"], config["train_people"], config.get("selection_people", []),
        points_per_person=int(config.get("normalization_points_per_person", 10000)), seed=seed,
        forbidden_people=config.get("forbidden_people", []))
    _write(outroot/"config.json", config)
    _write(outroot/"fold_manifest.json", manifest)
    device = torch.device(config.get("device", "cuda:0"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    amp = bool(config.get("amp", True) and device.type == "cuda")
    if amp and not torch.cuda.is_bf16_supported():
        raise RuntimeError("Requested BF16 is unsupported")
    model = IMUV8Model(ModelConfig(**config.get("model", {}))).to(device)
    core = int(config.get("core_frames", 16))
    frames = int(config.get("train_frames", core+2*(model.temporal_radius+1)))
    if frames < core+2*model.temporal_radius:
        raise ValueError("Training sequence does not contain full temporal halo")
    dataset = SequenceDataset(
        manifest, people=config["train_people"], frames=frames, core_frames=core,
        training=True, samples_per_epoch=int(config.get("samples_per_epoch", 2048)),
        seed=seed, sampling=config.get("sampling"),
        annotation_root=config.get("annotation_root"))
    _write(outroot/"sampling.json", dataset.sampling_summary())
    workers = int(config.get("workers", 2))
    loader = DataLoader(
        dataset, batch_size=int(config.get("batch_size", 8)), shuffle=False,
        num_workers=workers, pin_memory=device.type == "cuda",
        multiprocessing_context="spawn" if workers else None,
        persistent_workers=False, worker_init_fn=_worker_init)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config.get("lr", 3e-4)),
        weight_decay=float(config.get("weight_decay", 1e-3)))
    history = [];best_epoch=0;best_bce=float("inf")
    _write(outroot/"architecture.json", dict(
        model=model.config.to_dict(), parameters=sum(p.numel() for p in model.parameters()),
        temporal_radius=model.temporal_radius))
    try:
        for epoch in range(1, epochs+1):
            started = time.monotonic()
            dataset.set_epoch(epoch)
            model.train()
            factor = min(1., epoch/2)*(.1+.9*.5*(1+math.cos(math.pi*(epoch-1)/max(1, int(config.get("schedule_epochs",30))-1))))
            for group in optimizer.param_groups:
                group["lr"] = float(config.get("lr", 3e-4))*factor
            totals, steps = {}, 0
            for step, supplied in enumerate(loader, 1):
                batch = _batch(supplied, device)
                if config.get("augment", True):
                    batch = physical_augment(batch, manifest["normalization"], **config.get("augmentation", {}))
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
                    output = _forward(model, batch)
                    loss, parts = training_loss(output, batch, core_frames=core, **config.get("loss", {}))
                if parts["frames"]:
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Nonfinite IMU V8 loss")
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                    optimizer.step()
                    steps += 1
                    for key, value in parts.items():
                        totals[key] = totals.get(key, 0.)+value
                if step == 1 or step % 10 == 0:
                    status = dict(phase="training", component='context_encoder', epoch=epoch, epochs=epochs,
                                  step=step, total_steps=len(loader), loss=parts, pid=os.getpid(),
                                  fit_people=config["train_people"], updated_utc=_now(),
                                  fixed_epoch_budget=not bool(config.get('selection_people')), external_validation_used_for_fit=False)
                    _write(outroot/"progress.json", status)
                    print(json.dumps(status), flush=True)
            if not steps:
                raise RuntimeError("No physically supported known core frames")
            history.append(dict(epoch=epoch, loss={k: v/steps for k, v in totals.items()},
                                steps=steps, seconds=time.monotonic()-started,
                                lr=optimizer.param_groups[0]["lr"]))
            if config.get('selection_people'):
                selection=_selection_bce(model,manifest,config['selection_people'],device,core,int(config.get('batch_size',8)),amp)
                history[-1]['selection']=selection
                if selection['person_mean_bce']<best_bce:
                    best_bce=selection['person_mean_bce'];best_epoch=epoch
                    _save(outroot/'best.pt',_payload(model,manifest,config,epoch,True))
            _write(outroot/"history.json", history)
            _save(outroot/"last.pt", _payload(model, manifest, config, epoch, False))
        destination = outroot/"component.pt"
        if config.get('selection_people'):
            chosen=torch.load(outroot/'best.pt',map_location=device,weights_only=True)
            _save(destination,chosen)
        else:
            best_epoch=epochs
            _save(destination, _payload(model, manifest, config, epochs, True))
        result = dict(component_path=str(destination), completed=True, epochs=epochs,best_epoch=best_epoch,selected_epoch=best_epoch,
                      selection_bce=best_bce if config.get("selection_people") else None,
                      fit_people=config["train_people"],
                      normalization_fit_people=manifest["normalization"]["people"],
                      selected_on="inner_train_identity_frame_bce" if config.get("selection_people") else "locked_epoch_budget_no_outer_validation",
                      finished_utc=_now(), test_read=False)
        _write(outroot/"complete.json", result)
        _write(outroot/"progress.json", dict(phase="complete", **result))
        return result
    finally:
        dataset.close()


def load_component(path, device="cpu"):
    payload = torch.load(path, map_location=device, weights_only=True)
    if payload.get("format") != "imu-v8.component.1" or not payload.get("complete"):
        raise ValueError("A completed IMU V8 component is required")
    if set(payload["normalization"]["people"])-set(payload["fit_people"]):
        raise ValueError("Normalization contains a non-fit identity")
    model = IMUV8Model(ModelConfig(**payload["model_config"])).to(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    return model.eval(), payload


@torch.inference_mode()
def predict_records(model, base, people, normalization, device=None, core_frames=16,
                    batch_size=8, amp=False, require_excluded_people=()):
    """Return label-free frame evidence. This function does not select weights.

    For OOF callers pass require_excluded_people=people; overlap with
    normalization-fit people is rejected before any signal array is opened.
    """
    people = list(people)
    if set(require_excluded_people) & set(normalization["people"]):
        raise ValueError("OOF target identities occur in normalization fit people")
    device = torch.device(device if device is not None else next(model.parameters()).device)
    model = model.to(device).eval()
    manifest = base_manifest(base)
    dataset = SequenceDataset(
        manifest, people=people, frames=core_frames+2*model.temporal_radius,
        training=False, normalization=normalization)
    found = {row["person"] for row in dataset.records}
    missing = set(people)-found
    if missing:
        raise ValueError(f"Missing requested physical-cache identities: {sorted(missing)}")
    radius = model.temporal_radius
    requests = [(ri, start, min(core_frames, row["frame_count"]-start))
                for ri, row in enumerate(dataset.records)
                for start in range(0, row["frame_count"], core_frames)]
    collected = {}
    keys = ("imu", "point_mask", "frame_quality", "segment_ids")
    try:
        for first in range(0, len(requests), batch_size):
            chunk = requests[first:first+batch_size]
            windows = [dataset.get_window(ri, start-radius, core_frames+2*radius)
                       for ri, start, count in chunk]
            batch = {key: torch.stack([w[key] for w in windows]).to(device) for key in keys}
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=bool(amp and device.type == "cuda")):
                out = _forward(model, batch)
            for j, (ri, start, count) in enumerate(chunk):
                row = dataset.records[ri]
                target = collected.setdefault(ri, {})
                sl = slice(radius, radius+count)
                values = {k: out[k][j, sl].float().cpu().numpy()
                          for k in ("probability", "logits", "embedding", "role_probability")}
                values['local_probability']=out['local_logits'][j,sl].sigmoid().float().cpu().numpy()
                values['context_probability']=out['context_logits'][j,sl].sigmoid().float().cpu().numpy()
                values["boundary_probability"] = out["boundary_logits"][j, sl].sigmoid().float().cpu().numpy()
                values["valid"] = out["valid"][j, sl].cpu().numpy()
                values["point_mask"] = batch["point_mask"][j, sl].cpu().numpy()
                values["frame_quality"] = batch["frame_quality"][j, sl].float().cpu().numpy()
                local = batch["segment_ids"][j, sl].cpu().numpy()
                values["segment_ids"] = np.where(local >= 0, (ri+1)*1_000_000_000+local, -1)
                values["frame_start_tick"] = row["start_tick"]+np.arange(start, start+count, dtype=np.int64)*450
                for key, value in values.items():
                    target.setdefault(key, []).append(value)
        result = []
        for ri in sorted(collected):
            row = dataset.records[ri]
            value = {k: np.concatenate(v) for k, v in collected[ri].items()}
            value.update(person=row["person"], source_key=row["source_key"],
                         start_tick=row["start_tick"], frame_count=row["frame_count"])
            result.append(value)
        return result
    finally:
        dataset.close()







