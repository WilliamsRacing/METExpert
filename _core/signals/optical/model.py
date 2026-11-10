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

    def forward(self, x, frame_valid, packed_spans=None):
        if x.ndim != 3 or tuple(frame_valid.shape) != tuple(x.shape[:2]):
            raise ValueError("Temporal input must be [batch, frames, dim] with aligned validity")
        if packed_spans is None:
            if frame_valid.device.type != 'cpu':
                raise ValueError("V9 CUDA temporal inputs require CPU-prepared temporal_spans")
            from .data_v9 import pack_temporal_spans
            packed_spans = pack_temporal_spans(frame_valid > 0)
        # Group spans of identical length. No padded activations, LayerNorm
        # offsets, or dilated kernels can carry information over a true hole.
        flat = x.reshape(-1, x.shape[-1])
        output = flat * 0.
        for length, indices in packed_spans:
            if not isinstance(length, int) or length < 1 or indices.ndim != 2 or indices.shape[1] != length:
                raise ValueError("Malformed exact-length temporal span metadata")
            indices = indices.to(device=x.device, dtype=torch.long, non_blocking=True)
            selected = flat.index_select(0, indices.reshape(-1)).reshape(indices.shape[0], length, x.shape[-1])
            encoded = self._span(selected)
            # Autocast LayerNorm may return FP32 from BF16/FP16 inputs.
            # Preserve that output precision; index_copy requires equal dtypes.
            if output.dtype != encoded.dtype:
                output = output.to(dtype=encoded.dtype)
            output = output.index_copy(0, indices.reshape(-1), encoded.reshape(-1, x.shape[-1]))
        return output.reshape_as(x)



def build_model(config, num_features):
    if config.get('version') not in ('v8', 'v9'):
        raise ValueError('This source package supports V9 and its explicit V8 architecture ablation')
    keys=('dim','feature_width','dropout','dilations','short_dilations','input_mode','causal',
          'spectral_widths','spectral_depths','spectral_expansion','spectral_drop_path','spectral_kernel',
          'initial_fusion_weights','num_original_features')
    if config.get('version') == 'v9':
        from .model_v9 import PPGEatingNetV9
        keys += ('numeric_residual', 'numeric_gain_init', 'family_coverage_input', 'pca_availability_mode')
        return PPGEatingNetV9(num_features=num_features, **{k: config[k] for k in keys if k in config})
    from .model_v8 import PPGEatingNetV8
    return PPGEatingNetV8(num_features=num_features, **{k: config[k] for k in keys if k in config})
