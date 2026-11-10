
from __future__ import annotations
from collections import defaultdict
import numpy as np
import torch


def _cpu_array(value):
    if isinstance(value, torch.Tensor):
        if value.device.type != 'cpu':
            raise ValueError('Build V9 span metadata from the CPU batch before H2D')
        return value.detach().numpy()
    return np.asarray(value)


def pack_temporal_spans(support):
    """Return exact-length groups as (length, flattened frame indices)."""
    support = _cpu_array(support).astype(bool)
    if support.ndim != 2:
        raise ValueError('Temporal support must be [batch, frames]')
    groups = defaultdict(list)
    frames = support.shape[1]
    for row, valid in enumerate(support):
        change = np.diff(np.r_[False, valid, False].astype(np.int8))
        for start, stop in zip(np.flatnonzero(change == 1), np.flatnonzero(change == -1)):
            groups[int(stop - start)].append(np.arange(start, stop, dtype=np.int64) + row * frames)
    return [(length, torch.from_numpy(np.stack(groups[length]))) for length in sorted(groups)]


def build_temporal_spans(batch):
    """Exactly mirror V9 expert observation eligibility on CPU, before H2D."""
    valid = _cpu_array(batch['frame_valid']) > 0
    if 'original_feature_mask' in batch:
        mask = _cpu_array(batch['original_feature_mask'])
        feature = np.delete(np.isfinite(mask) & (mask > 0), 25, axis=-1).any(-1)
    else:
        feature = (_cpu_array(batch['feature_mask']) > 0).any(-1)
    if 'physical_feature_support' in batch:
        flag = _cpu_array(batch['physical_feature_support'])
        feature &= np.isfinite(flag) & (flag > 0)
    mask = np.clip(np.nan_to_num(_cpu_array(batch['raw_mask']), nan=0., posinf=0., neginf=0.), 0, 1)
    quality = np.nan_to_num(_cpu_array(batch['quality']), nan=0., posinf=0., neginf=0.)
    channel = np.clip(_cpu_array(batch['channel_ids']).astype(np.int64), 0, 44)
    wave = np.zeros(45, bool)
    wave[[*range(3, 9), 12, 13, 14, 18, 19, 20, 1, 9, 10, 11, 15, 16, 17]] = True
    raw_channel = (mask.mean(-1) >= .5) & (quality > 0) & wave[channel]
    if 'spectrogram_mask' in batch:
        smask = np.clip(np.nan_to_num(_cpu_array(batch['spectrogram_mask']), nan=0., posinf=0., neginf=0.), 0, 1)
        if 'spectrogram' in batch:
            smask = smask * np.isfinite(_cpu_array(batch['spectrogram'])).all(-2)
        raw_channel &= smask.sum(-1) > 0
    else:
        raise ValueError('V9 requires cached numerical spectra and observed-column masks')
    return {'feature': pack_temporal_spans(valid & feature),
            'raw': pack_temporal_spans(valid & raw_channel.any(-1))}


def prepare_device_batch(batch, device, *, non_blocking=True):
    """Transfer tensors once; preserve Python metadata and CPU-built span groups."""
    device = torch.device(device)
    spans = batch.get('temporal_spans')
    if spans is None:
        spans = build_temporal_spans(batch)
    result = {key: value.to(device, non_blocking=non_blocking)
              if isinstance(value, torch.Tensor) else value for key, value in batch.items()
              if key != 'temporal_spans'}
    result['temporal_spans'] = {
        name: [(length, indices.to(device, non_blocking=non_blocking)) for length, indices in groups]
        for name, groups in spans.items()}
    return result
