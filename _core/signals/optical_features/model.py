from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F

class TemporalBlock(nn.Module):
    def __init__(self, dim, dilation, dropout, causal=False):
        super().__init__()
        self.causal = causal
        self.padding = 2 * dilation if causal else dilation
        self.norm = nn.LayerNorm(dim)
        self.depthwise = nn.Conv1d(dim, dim, 3, dilation=dilation, groups=dim)
        self.mix = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(2 * dim, dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, valid):
        y = self.norm(x).transpose(1, 2)
        y = F.pad(y, (self.padding, 0) if self.causal else (self.padding, self.padding))
        y = self.depthwise(y).transpose(1, 2)
        return (x + self.dropout(self.mix(y))) * valid[..., None]


class SpanMultiScaleTemporal(nn.Module):
    """Parallel short/long TCNs evaluated separately on contiguous valid spans.

    With 30s frames, dilations 1/2/4 cover 15 frames (7.5min), and
    1/2/4/8/16 cover 63 frames (31.5min). Available context remains limited
    by each cached continuous segment. Internal invalid frames are barriers,
    rather than just zero activations that a dilated convolution can jump.
    """
    def __init__(self, dim, short_dilations, long_dilations, dropout, causal):
        super().__init__()
        for name, values in (("short_dilations", short_dilations), ("dilations", long_dilations)):
            if not values or any(isinstance(d, bool) or not isinstance(d, int) or d < 1 for d in values):
                raise ValueError(f"{name} must contain positive integer dilations")
        self.short = nn.ModuleList([TemporalBlock(dim, d, dropout, causal) for d in short_dilations])
        self.long = nn.ModuleList([TemporalBlock(dim, d, dropout, causal) for d in long_dilations])
        self.mix = nn.Sequential(nn.Linear(2 * dim, dim), nn.LayerNorm(dim), nn.GELU())

    def _span(self, x):
        valid = x.new_ones(x.shape[:2])
        short, long = x, x
        for block in self.short:
            short = block(short, valid)
        for block in self.long:
            long = block(long, valid)
        return self.mix(torch.cat((short, long), -1))

    def forward(self, x, frame_valid):
        valid = frame_valid > 0
        if x.ndim != 3 or tuple(valid.shape) != tuple(x.shape[:2]):
            raise ValueError("Temporal input must be [batch, frames, dim] with aligned frame_valid")
        if bool(valid.all()):
            return self._span(x)
        rows = []
        # Most cached examples have one span plus end padding. Only tiny
        # boundary-index vectors move to Python; no signal leaves the device.
        for bi in range(x.shape[0]):
            padded = F.pad(valid[bi].to(torch.int8), (1, 1))
            change = padded[1:] - padded[:-1]
            starts = (change == 1).nonzero(as_tuple=False).flatten().tolist()
            ends = (change == -1).nonzero(as_tuple=False).flatten().tolist()
            pieces, cursor = [], 0
            for start, end in zip(starts, ends):
                if start > cursor:
                    pieces.append(x[bi:bi + 1, cursor:start] * 0.)
                pieces.append(self._span(x[bi:bi + 1, start:end]))
                cursor = end
            if cursor < x.shape[1]:
                pieces.append(x[bi:bi + 1, cursor:] * 0.)
            if not pieces:
                pieces = [x[bi:bi + 1] * 0.]
            rows.append(torch.cat(pieces, 1))
        return torch.cat(rows, 0)



def build_model(config, num_features):
    if config.get('version') != 'v8':
        raise ValueError('This independent source package supports V8 only')
    from .model_v8 import PPGEatingNetV8
    keys=('dim','feature_width','dropout','dilations','short_dilations','input_mode','causal',
          'spectral_widths','spectral_depths','spectral_expansion','spectral_drop_path',
          'initial_fusion_weights','num_original_features')
    return PPGEatingNetV8(num_features=num_features,**{k:config[k] for k in keys if k in config})
