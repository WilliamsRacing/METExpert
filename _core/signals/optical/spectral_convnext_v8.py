
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def clean_column_mask(spectrogram, column_mask):
    if spectrogram.ndim < 3 or tuple(spectrogram.shape[-2:]) != (37, 23):
        raise ValueError("V8 spectrograms must end in [37,23]")
    if tuple(column_mask.shape) != (*spectrogram.shape[:-2], 23):
        raise ValueError("V8 spectrogram mask must align with its 23 time columns")
    mask = torch.nan_to_num(column_mask.float(), nan=0., posinf=0., neginf=0.).clamp(0, 1)
    return mask * torch.isfinite(spectrogram).all(-2)


class MaskedGRN(nn.Module):
    """Global response normalization over observed positions only (NHWC).

    The response sum uses float32 even under autocast. Gamma/beta start at zero,
    so the GRN itself starts as the identity on the measured support.
    """
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, channels))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, channels))
        self.eps = float(eps)

    def forward(self, x, column_mask):
        mask = column_mask[:, None, :, None].to(x.dtype)
        valid = mask > 0
        clean = torch.where(valid, x, torch.zeros_like(x))
        # vector_norm has a well-defined zero gradient for an all-zero channel.
        response = torch.linalg.vector_norm(clean.float() * mask.float().sqrt(), ord=2,
                                            dim=(1, 2), keepdim=True)
        normalized = response / (response.mean(-1, keepdim=True) + self.eps)
        y = clean + self.gamma.to(x.dtype) * clean * normalized.to(x.dtype) + self.beta.to(x.dtype)
        return torch.where(valid, y, torch.zeros_like(y))


class DropPath(nn.Module):
    def __init__(self, probability=0.):
        super().__init__()
        if not 0 <= probability < 1:
            raise ValueError("drop_path must be in [0,1)")
        self.probability = float(probability)

    def forward(self, x):
        if not self.training or self.probability == 0:
            return x
        keep = 1 - self.probability
        sample = x.new_empty(x.shape[0], 1, 1, 1).bernoulli_(keep)
        return x * sample / keep


class MaskedConvNeXtBlock(nn.Module):
    def __init__(self, channels, expansion=2, dropout=.15, drop_path=0., kernel=(5, 3)):
        super().__init__()
        if expansion < 1 or isinstance(expansion, bool) or int(expansion) != expansion:
            raise ValueError("spectral expansion must be a positive integer")
        if len(kernel) != 2 or any(int(k) != k or k < 1 or k % 2 != 1 for k in kernel):
            raise ValueError("spectral kernel must contain two positive odd integers")
        hidden = int(expansion) * channels
        self.depthwise = nn.Conv2d(channels, channels, tuple(kernel),
            padding=(kernel[0] // 2, kernel[1] // 2), groups=channels)
        self.norm = nn.LayerNorm(channels, eps=1e-6)
        self.expand = nn.Linear(channels, hidden)
        self.grn = MaskedGRN(hidden)
        self.contract = nn.Linear(hidden, channels)
        self.dropout = nn.Dropout(dropout)
        self.drop_path = DropPath(drop_path)

    def forward(self, x, column_mask):
        valid = column_mask[:, None, None, :] > 0
        x = torch.where(valid, x, torch.zeros_like(x))
        y = self.depthwise(x)
        y = torch.where(valid, y, torch.zeros_like(y)).permute(0, 2, 3, 1)
        y = F.gelu(self.expand(self.norm(y)))
        y = self.grn(y, column_mask)
        y = self.dropout(self.contract(y)).permute(0, 3, 1, 2)
        return torch.where(valid, x + self.drop_path(y), torch.zeros_like(x))


class SpectralConvNeXtV8(nn.Module):
    """One physical channel: [37,23] -> 64-D; shared between the top channels."""
    def __init__(self, dim=64, widths=(24, 48, 96), depths=(1, 1, 2),
                 expansion=2, dropout=.15, drop_path=.05, kernel=(5, 3)):
        super().__init__()
        if len(widths) != 3 or len(depths) != 3 or any(
                isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in (*widths, *depths)):
            raise ValueError("V8 spectral widths/depths need three positive integers each")
        self.output_dim = int(dim)
        self.down = nn.ModuleList()
        self.down_norm = nn.ModuleList()
        self.stages = nn.ModuleList()
        previous, block_index = 1, 0
        probabilities = torch.linspace(0., float(drop_path), sum(depths)).tolist()
        for width, depth in zip(widths, depths):
            self.down.append(nn.Conv2d(previous, width, 3, stride=(2, 1), padding=1))
            self.down_norm.append(nn.LayerNorm(width, eps=1e-6))
            self.stages.append(nn.ModuleList([
                MaskedConvNeXtBlock(width, expansion, dropout, probabilities[block_index + i], kernel)
                for i in range(depth)]))
            block_index += depth
            previous = width
        self.project = nn.Linear(widths[-1] * 5, dim)
        self.out_norm = nn.LayerNorm(dim)

    def forward(self, spectrogram, column_mask):
        if spectrogram.ndim != 3:
            raise ValueError("V8 spectral channel input must have shape [N,37,23]")
        mask = clean_column_mask(spectrogram, column_mask)
        valid = mask[:, None, None, :] > 0
        clean = torch.where(mask[:, None, :] > 0, spectrogram, torch.zeros_like(spectrogram))
        x = clean[:, None]
        for down, norm, stage in zip(self.down, self.down_norm, self.stages):
            x = down(x)
            x = norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            x = torch.where(valid, F.gelu(x), torch.zeros_like(x))
            for block in stage:
                x = block(x, mask)
        # Frequency position is retained. Only observed time columns are pooled.
        x = (x * mask[:, None, None, :]).sum(-1) / mask.sum(-1).clamp_min(1.)[:, None, None]
        x = self.out_norm(self.project(x.flatten(1)))
        return x * (mask.sum(-1) > 0)[:, None]


class QualityChannelEncoderV8(nn.Module):
    """V5 physical-ID/wavelength quality pooling around the new shared encoder."""
    def __init__(self, dim=64, widths=(24, 48, 96), depths=(1, 1, 2),
                 expansion=2, dropout=.15, drop_path=.05, kernel=(5, 3)):
        super().__init__()
        self.wave_encoder = "convnextv2_stft"
        self.pulse = SpectralConvNeXtV8(dim, widths, depths, expansion, dropout, drop_path, kernel)
        self.wavelength = nn.Embedding(4, dim, padding_idx=0)
        self.channel_identity = nn.Embedding(45, dim, padding_idx=0)
        nn.init.normal_(self.wavelength.weight, std=.02)
        nn.init.normal_(self.channel_identity.weight, std=.02)
        with torch.no_grad():
            self.wavelength.weight[0].zero_()
            self.channel_identity.weight[0].zero_()
        self.attention = nn.Sequential(nn.Linear(dim + 1, max(dim // 2, 1)), nn.GELU(),
                                      nn.Linear(max(dim // 2, 1), 1))
        lut = torch.zeros(45, dtype=torch.long)
        lut[3:9] = 1
        lut[[12, 13, 14, 18, 19, 20]] = 2
        lut[[1, 9, 10, 11, 15, 16, 17]] = 3
        self.register_buffer("wavelength_lut", lut)

    def observation(self, raw_mask, quality, channel_ids, spectrogram_mask=None):
        if raw_mask.shape[:-1] != quality.shape or channel_ids.shape != quality.shape:
            raise ValueError("V8 channel observation arrays are not aligned")
        coverage = torch.nan_to_num(raw_mask.float(), nan=0., posinf=0., neginf=0.).clamp(0, 1).mean(-1)
        wave = self.wavelength_lut[channel_ids.long().clamp(0, 44)]
        q = torch.nan_to_num(quality.float(), nan=0., posinf=0., neginf=0.).clamp(0, 1)
        available = (coverage >= .5) & (q > 0) & (wave > 0)
        if spectrogram_mask is None or tuple(spectrogram_mask.shape) != (*quality.shape, 23):
            raise ValueError("V8 STFT requires aligned spectrogram_mask ending in 23 columns")
        smask = torch.nan_to_num(spectrogram_mask.float(), nan=0., posinf=0., neginf=0.).clamp(0, 1)
        available = available & (smask.sum(-1) > 0)
        return available, coverage, q, wave

    def forward(self, raw, raw_mask, quality, channel_ids, spectrogram=None, spectrogram_mask=None):
        if raw.shape[:-1] != raw_mask.shape[:-1] or tuple(raw.shape[:-1]) != tuple(quality.shape):
            raise ValueError("V8 raw placeholders/masks must align with channel quality")
        if spectrogram is None or spectrogram_mask is None or tuple(spectrogram.shape) != (*quality.shape, 37, 23):
            raise ValueError("V8 requires aligned spectrograms ending in 37x23")
        smask = clean_column_mask(spectrogram, spectrogram_mask)
        available, _, q, wave = self.observation(raw_mask, quality, channel_ids, smask)
        selected = available.reshape(-1).nonzero(as_tuple=False).squeeze(-1)
        if selected.numel():
            encoded = self.pulse(spectrogram.reshape(-1, 37, 23).index_select(0, selected),
                                 smask.reshape(-1, 23).index_select(0, selected))
            emb = encoded.new_zeros(available.numel(), self.wavelength.embedding_dim).index_copy(0, selected, encoded)
        else:
            emb = raw.new_zeros(available.numel(), self.wavelength.embedding_dim)
        emb = emb.reshape(*quality.shape, -1)
        emb = emb + self.wavelength(wave) + self.channel_identity(channel_ids.long().clamp(0, 44))
        logits = self.attention(torch.cat((emb, q[..., None]), -1)).squeeze(-1) + q.clamp_min(1e-5).log()
        logits = logits.masked_fill(~available, -1e4)
        weights = logits.softmax(-1) * available
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-6)
        return (emb * weights[..., None]).sum(-2), weights