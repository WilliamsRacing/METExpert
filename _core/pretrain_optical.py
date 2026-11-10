
from __future__ import annotations
from data_pipeline import PROJECT

import argparse
import hashlib
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Sampler

from signals.optical.spectral_convnext_v8 import SpectralConvNeXtV8, clean_column_mask


def masked_patch_mask(column_mask, *, frequency_bins=37, block_size=(4, 4), fraction=.3, generator=None):
    """Select only observed pixels, in 4x4 blocks (edge blocks can be smaller)."""
    if column_mask.ndim != 2 or not 0 < fraction < 1:
        raise ValueError("expected [N,time] support and masking fraction in (0,1)")
    batch, time = column_mask.shape
    bh, bw = map(int, block_size)
    if min(bh, bw) < 1:
        raise ValueError("mask block dimensions must be positive")
    height, width = math.ceil(frequency_bins/bh), math.ceil(time/bw)
    # CPU RNG is checkpointable and independent of CUDA launch order.
    selection = torch.zeros(batch, height*width, dtype=torch.bool)
    observed = (column_mask.detach().cpu() > 0)[:, None, :].expand(batch, frequency_bins, time)
    eligible = F.max_pool2d(observed.float()[:, None], (bh,bw), stride=(bh,bw), ceil_mode=True).flatten(1) > 0
    for bi in range(batch):
        indices = torch.nonzero(eligible[bi], as_tuple=False).flatten()
        if not indices.numel():
            continue
        count = max(1, int(round(float(fraction)*indices.numel())))
        chosen = indices[torch.randperm(indices.numel(), generator=generator)[:count]]
        selection[bi, chosen] = True
    expanded = selection.reshape(batch, height, width).repeat_interleave(bh,1).repeat_interleave(bw,2)
    return (expanded[:, :frequency_bins, :time] & observed).to(column_mask.device)


class MaskedSpectrumPretrainer(nn.Module):
    def __init__(self, *, dim=64, widths=(24,48,96), depths=(1,1,2), expansion=2,
                 dropout=.15, drop_path=.05, block_size=(4,4), mask_fraction=.3):
        super().__init__()
        self.pulse = SpectralConvNeXtV8(dim, widths, depths, expansion, dropout, drop_path)
        self.mask_token = nn.Parameter(torch.zeros(1))
        self.decoder = nn.Sequential(nn.Conv2d(widths[-1], widths[-1], 3, padding=1), nn.GELU(),
                                     nn.Conv2d(widths[-1], 1, 1))
        self.block_size = tuple(block_size)
        self.mask_fraction = float(mask_fraction)

    def spatial_features(self, spectrogram, real_column_mask):
        """Same trainable encoder blocks; preserve all 23 time positions."""
        real = clean_column_mask(spectrogram, real_column_mask)
        valid = real[:, None, None, :] > 0
        x = torch.where(real[:,None,:] > 0, spectrogram, torch.zeros_like(spectrogram))[:,None]
        for down, norm, stage in zip(self.pulse.down, self.pulse.down_norm, self.pulse.stages):
            x = down(x)
            x = norm(x.permute(0,2,3,1)).permute(0,3,1,2)
            x = torch.where(valid, F.gelu(x), torch.zeros_like(x))
            for block in stage:
                x = block(x, real)
        return x, real

    def forward(self, spectrogram, real_column_mask, *, artificial_mask=None, generator=None):
        real = clean_column_mask(spectrogram, real_column_mask)
        if artificial_mask is None:
            artificial_mask = masked_patch_mask(real, block_size=self.block_size,
                                                fraction=self.mask_fraction, generator=generator)
        if artificial_mask.shape != spectrogram.shape:
            raise ValueError("artificial mask must align with [N,37,23] spectra")
        observed = (real[:,None,:] > 0).expand_as(spectrogram)
        if bool((artificial_mask & ~observed).any()):
            raise ValueError("cannot reconstruct naturally missing positions")
        eligible = real.gt(0).float().mean(-1) >= .8
        artificial_mask = artificial_mask.bool() & observed & eligible[:,None,None]
        clean = torch.where(observed, spectrogram, torch.zeros_like(spectrogram))
        masked = torch.where(artificial_mask, self.mask_token.to(clean.dtype), clean)
        features, _ = self.spatial_features(masked, real)
        reconstruction = F.interpolate(self.decoder(features).float(), size=(37,23),
                                       mode="bilinear", align_corners=False)[:,0]
        elementwise = F.huber_loss(reconstruction, clean.float(), reduction="none")
        denominator = artificial_mask.sum()
        numerator = (elementwise * artificial_mask).sum()
        loss = numerator / denominator.clamp_min(1)
        return {"loss": loss, "loss_numerator": numerator, "loss_denominator": denominator,
                "reconstruction": reconstruction, "real_column_mask": real,
                "artificial_mask": artificial_mask, "spatial_shape": tuple(features.shape),
                "eligible_frames": eligible, "masked_fraction": artificial_mask.sum()/observed.sum().clamp_min(1)}


class SpectrumFrameDataset(Dataset):
    """Only spectra, support and route metadata are read; no eating labels."""
    def __init__(self, records, subjects, minimum_column_fraction=.8):
        allowed = {str(s) for s in subjects}
        self.records = [r for r in records if str(r["subject_id"]) in allowed]
        if not allowed or {str(r["subject_id"]) for r in self.records} != allowed:
            raise ValueError("SSL training identities are empty or absent from records")
        self.items = []
        self.groups = {}
        self._spectra = {}
        self._masks = {}
        valid_channels = {1, *range(3,21)}
        for ri, rec in enumerate(self.records):
            with np.load(rec["_path"], allow_pickle=False) as data:
                mask = np.asarray(data["spectrogram_mask"], np.float32)
                quality = np.asarray(data["quality"], np.float32)
                channels = np.asarray(data["channel_ids"], int)
                raw_mask = np.asarray(data["raw_mask"], np.float32)
                owned = np.asarray(data["canonical_mask"], bool) if "canonical_mask" in data else np.ones(len(mask), bool)
            eligible = ((mask > 0).mean(-1) >= minimum_column_fraction) & (quality > 0)
            eligible &= (raw_mask.mean(-1) >= .5) & np.isin(channels, list(valid_channels)) & owned[:,None]
            spectrum = np.load(rec["_spectrogram_path"], mmap_mode="r", allow_pickle=False)
            frames, routes = np.nonzero(eligible)
            record_items = []
            for fi, ci in zip(frames, routes):
                if not np.isfinite(spectrum[fi,ci,:,mask[fi,ci] > 0]).all():
                    continue
                index = len(self.items)
                self.items.append((ri,int(fi),int(ci)))
                record_items.append(index)
            if record_items:
                self.groups.setdefault(str(rec["subject_id"]), {})[ri] = record_items
        if not self.items or set(self.groups) != allowed:
            raise ValueError("each SSL training identity needs >=1 eligible observed spectrum")
        self.training_subjects = sorted(allowed)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        ri, fi, ci = self.items[index]
        rec = self.records[ri]
        # Bounded caches: opening an mmap does not load the entire spectrum.
        if ri not in self._spectra:
            if len(self._spectra) >= 8:
                old = next(iter(self._spectra))
                self._spectra.pop(old)
                self._masks.pop(old)
            self._spectra[ri] = np.load(rec["_spectrogram_path"], mmap_mode="r", allow_pickle=False)
            with np.load(rec["_path"], allow_pickle=False) as data:
                self._masks[ri] = np.asarray(data["spectrogram_mask"], np.float32)
        return (torch.from_numpy(np.array(self._spectra[ri][fi,ci], dtype=np.float32)),
                torch.from_numpy(np.array(self._masks[ri][fi,ci], dtype=np.float32)))


class IdentitySegmentFrameSampler(Sampler):
    """Round-robin identities and continuous segments with replacement inside."""
    def __init__(self, dataset, seed=20261011):
        self.dataset, self.seed, self.epoch = dataset, int(seed), 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        rng = np.random.default_rng(self.seed+self.epoch)
        subjects = list(self.dataset.groups)
        rng.shuffle(subjects)
        records = {s: list(self.dataset.groups[s]) for s in subjects}
        for values in records.values():
            rng.shuffle(values)
        count = {s:0 for s in subjects}
        for i in range(len(self.dataset)):
            subject = subjects[i % len(subjects)]
            recs = records[subject]
            ri = recs[count[subject] % len(recs)]
            count[subject] += 1
            candidates = self.dataset.groups[subject][ri]
            yield int(candidates[int(rng.integers(len(candidates)))])


def _source_fingerprint(cache, records, subjects):
    allowed = {str(s) for s in subjects}
    manifest = Path(cache) / "manifest.json"
    payload = {"subjects": sorted(allowed),
        "records": [{key:r.get(key) for key in ("subject_id","recording_id","continuous_id","n_frames", "v8_build_signature", "original_fields_digest")}
                    for r in records if str(r["subject_id"]) in allowed],
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest() if manifest.exists() else None}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def pretrain_for_subjects(cache, records, subjects, config, out, device):
    """Train/resume optional SSL, returning its audited checkpoint path.

    Call independently for each auxiliary-model training allow-list. It must
    never reuse a full-fold checkpoint for a held-out crossfit identity.
    """
    options = dict(config.get("self_supervised", config.get("pretraining", config.get("ssl", {}))))
    if not options.get("enabled", False):
        return None
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(device)
    seed = int(config["seed"])
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    dataset = SpectrumFrameDataset(records, subjects)
    sampler = IdentitySegmentFrameSampler(dataset, seed)
    batch_size = int(options.get("batch_size",256))
    microbatch = int(options.get("microbatch_size",batch_size))
    if batch_size < 1 or microbatch < 1 or batch_size % microbatch:
        raise ValueError("SSL frame batch must be positive and divisible by frame microbatch")
    accumulation = batch_size // microbatch
    epochs = int(options.get("epochs",40))
    warmup = int(options.get("warmup_epochs",5))
    lr = float(options.get("lr",3e-4))
    model_config = config["model"]
    model = MaskedSpectrumPretrainer(dim=model_config.get("dim",64),
        widths=tuple(model_config.get("spectral_widths",(24,48,96))),
        depths=tuple(model_config.get("spectral_depths",(1,1,2))),
        expansion=model_config.get("spectral_expansion",2), dropout=model_config.get("dropout",.15),
        drop_path=model_config.get("spectral_drop_path",.05),
        block_size=tuple(options.get("block_size",(4,4))), mask_fraction=float(options.get("mask_fraction",.3))).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=float(options.get("weight_decay",.002)))
    updates_per_epoch = math.ceil(len(dataset)/batch_size)
    total_updates, warmup_updates = epochs*updates_per_epoch, warmup*updates_per_epoch
    def schedule(step):
        if step < warmup_updates:
            return max(1., step+1)/max(1,warmup_updates)
        ratio = (step-warmup_updates)/max(1,total_updates-warmup_updates)
        return .5*(1+math.cos(math.pi*min(1.,ratio)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    amp = device.type == "cuda" and bool(options.get("amp",True))
    bf16 = amp and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if bf16 else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=amp and not bf16)
    generator = torch.Generator().manual_seed(seed)
    loader_generator = torch.Generator().manual_seed(seed)
    workers = int(options.get("num_workers",4))
    loader = DataLoader(dataset, batch_size=microbatch, sampler=sampler, num_workers=workers,
                        pin_memory=device.type=="cuda", persistent_workers=workers>0,
                        generator=loader_generator, **({"prefetch_factor":2} if workers else {}))
    fingerprint = _source_fingerprint(cache, records, subjects)
    contract = {"training_subjects": sorted(str(s) for s in subjects), "source_fingerprint": fingerprint,
                "options": options, "model": model_config, "seed": seed}
    last = out/"last.pt"
    begin, success_steps, history = 0, 0, []
    if last.exists():
        checkpoint = torch.load(last, map_location="cpu", weights_only=True)
        if checkpoint["contract"] != contract:
            raise ValueError("SSL checkpoint identity/source/config contract differs")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["amp_scaler"])
        random.setstate(checkpoint["rng"]["python"])
        nstate = checkpoint["rng"]["numpy"]
        np.random.set_state((nstate[0],np.asarray(nstate[1],np.uint32),nstate[2],nstate[3],nstate[4]))
        torch.set_rng_state(checkpoint["rng"]["torch"])
        if device.type == "cuda" and checkpoint["rng"].get("cuda") is not None:
            torch.cuda.set_rng_state_all(checkpoint["rng"]["cuda"])
        generator.set_state(checkpoint["mask_generator"])
        loader_generator.set_state(checkpoint["loader_generator"])
        begin, success_steps = int(checkpoint["epoch"]), int(checkpoint["success_steps"])
        history = checkpoint["history"]
    model.train()
    for epoch in range(begin, epochs):
        sampler.set_epoch(epoch)
        optimizer.zero_grad(set_to_none=True)
        total_loss, total_pixels, group_pixels = 0., 0, 0
        for batch_index, (spectrum, real_mask) in enumerate(loader):
            spectrum = spectrum.to(device, non_blocking=True)
            real_mask = real_mask.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type,dtype=dtype,enabled=amp):
                output = model(spectrum, real_mask, generator=generator)
            # Backpropagate raw masked-pixel numerator, then divide accumulated
            # gradients by the entire group pixel count (unequal tail safe).
            scaler.scale(output["loss_numerator"]).backward()
            pixels = int(output["loss_denominator"].detach().cpu())
            group_pixels += pixels
            total_pixels += pixels
            total_loss += float(output["loss_numerator"].detach().cpu())
            finish = (batch_index+1)%accumulation==0 or batch_index+1==len(loader)
            if not finish:
                continue
            scaler.unscale_(optimizer)
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad.div_(max(1,group_pixels))
            nn.utils.clip_grad_norm_(model.parameters(),1.)
            before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= before:
                scheduler.step()
                success_steps += 1
            optimizer.zero_grad(set_to_none=True)
            group_pixels = 0
        history.append({"epoch":epoch+1,"masked_huber":total_loss/max(1,total_pixels),
                        "masked_pixels":total_pixels,"success_steps":success_steps,
                        "frame_exposure":len(dataset)})
        nstate = np.random.get_state()
        rng = {"python":random.getstate(),"numpy":[nstate[0],nstate[1].tolist(),int(nstate[2]),int(nstate[3]),float(nstate[4])],"torch":torch.get_rng_state(),
               "cuda":torch.cuda.get_rng_state_all() if device.type=="cuda" else None}
        payload = {"version":"v9_masked_stft_pretraining", "epoch":epoch+1,"success_steps":success_steps,
            "model":model.state_dict(),"pulse_state":model.pulse.state_dict(),
            "optimizer":optimizer.state_dict(),"scheduler":scheduler.state_dict(),"amp_scaler":scaler.state_dict(),
            "rng":rng,"mask_generator":generator.get_state(),"loader_generator":loader_generator.get_state(),
            "contract":contract,"train_subjects":contract["training_subjects"],
            "source_fingerprint":fingerprint,"config":config,"history":history,
            "head_policy":"discard reconstruction head; initialize pulse only; supervised encoder remains trainable"}
        temporary = last.with_suffix(".pt.tmp")
        torch.save(payload,temporary)
        temporary.replace(last)
        (out/"history.json").write_text(json.dumps(history,indent=2),encoding="utf-8")
    return str(last)









