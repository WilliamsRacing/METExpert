
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass(frozen=True)
class ModelConfig:
    input_channels: int = 12
    quality_dim: int = 8
    local_channels: int = 64
    embedding_dim: int = 128
    local_kernel_size: int = 15
    local_dilations: tuple[int, ...] = (1, 2, 4)
    temporal_dilations: tuple[int, ...] = (1, 2, 4, 8)
    dropout: float = 0.15

    def to_dict(self) -> dict:
        return asdict(self)


class _LocalResidualBlock(nn.Module):
    """Per-point normalization avoids statistics from other frames or samples."""

    def __init__(self, channels: int, kernel: int, dilation: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.depthwise = nn.Conv1d(
            channels, channels, kernel, dilation=dilation,
            padding=dilation * (kernel - 1) // 2, groups=channels,
        )
        self.expand = nn.Conv1d(channels, channels * 2, 1)
        self.contract = nn.Conv1d(channels * 2, channels, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        valid = mask[:, None, :]
        y = self.norm(x.transpose(1, 2)).transpose(1, 2)
        y = torch.where(valid, y, torch.zeros_like(y))
        y = self.depthwise(y)
        y = self.contract(self.dropout(F.gelu(self.expand(y))))
        return torch.where(valid, x + self.dropout(y), torch.zeros_like(x))


class _SegmentDepthwiseConv(nn.Module):
    """Depthwise k=3 convolution with strict same-contiguous-run support.

    Unlike masking after an ordinary dilated convolution, this also blocks a
    dilation from jumping over an invalid frame into another valid run.
    """

    def __init__(self, channels: int, dilation: int):
        super().__init__()
        self.dilation = int(dilation)
        self.weight = nn.Parameter(torch.empty(3, channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        nn.init.normal_(self.weight, mean=0.0, std=math.sqrt(1.0 / 3.0))

    def forward(self, x: Tensor, valid: Tensor, runs: Tensor) -> Tensor:
        _, length, _ = x.shape
        indices = torch.arange(length, device=x.device)
        result = torch.zeros_like(x)
        for tap, offset in enumerate((-self.dilation, 0, self.dilation)):
            source = indices + offset
            inside = (source >= 0) & (source < length)
            source = source.clamp(0, max(0, length - 1))
            shifted = x.index_select(1, source)
            allowed = (
                inside[None, :] & valid & valid.index_select(1, source)
                & (runs == runs.index_select(1, source))
            )
            shifted = torch.where(allowed[..., None], shifted, torch.zeros_like(shifted))
            result = result + shifted * self.weight[tap]
        return torch.where(
            valid[..., None], result + self.bias, torch.zeros_like(result)
        )


class _TemporalResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.depthwise = _SegmentDepthwiseConv(channels, dilation)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(channels * 2, channels),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, valid: Tensor, runs: Tensor) -> Tensor:
        y = torch.where(valid[..., None], self.norm(x), torch.zeros_like(x))
        y = self.depthwise(y, valid, runs)
        y = self.mlp(y)
        return torch.where(
            valid[..., None], x + self.dropout(y), torch.zeros_like(x)
        )


class IMUV6Model(nn.Module):
    """Three-class meal detector with local motion and 15.5-minute context.

    forward(imu, point_mask, frame_quality=None, segment_ids=None)
      imu:           [B,T,C,450] (other positive point lengths are also accepted)
      point_mask:    [B,T,450], True means physically usable
      frame_quality: [B,T,Q], defaults to zero physical-quality covariates
      segment_ids:   [B,T], negative means invalid; changes split temporal context

    Returns logits [B,T,3], boundary [B,T,2], embedding [B,T,128], valid [B,T],
    probability [B,T]. Invalid outputs are zero and must remain unsupervised.
    Background=0, same-hand meal=1, different-hand meal=2. Unknown-hand positive
    supervision belongs in the trainer as -log(p1+p2), not a fourth class.
    """

    def __init__(self, config: ModelConfig | None = None, **kwargs):
        super().__init__()
        if config is not None and kwargs:
            raise ValueError("Pass ModelConfig or keyword configuration, not both")
        self.config = config if config is not None else ModelConfig(**kwargs)
        cfg = self.config
        if min(cfg.input_channels, cfg.local_channels, cfg.embedding_dim) < 1:
            raise ValueError("Channel dimensions must be positive")
        if cfg.quality_dim < 0:
            raise ValueError("quality_dim must be nonnegative")
        if cfg.local_kernel_size < 1 or cfg.local_kernel_size % 2 == 0:
            raise ValueError("local_kernel_size must be a positive odd number")
        if not cfg.temporal_dilations or any(int(d) < 1 for d in cfg.temporal_dilations):
            raise ValueError("temporal_dilations must contain positive integers")
        if not cfg.local_dilations or any(int(d) < 1 for d in cfg.local_dilations):
            raise ValueError("local_dilations must contain positive integers")
        self.temporal_radius = sum(int(d) for d in cfg.temporal_dilations)
        self.temporal_receptive_frames = 2 * self.temporal_radius + 1
        self.local_receptive_points = (
            1 + (cfg.local_kernel_size - 1) * sum(cfg.local_dilations)
        )

        self.point_projection = nn.Conv1d(cfg.input_channels, cfg.local_channels, 1)
        self.local_blocks = nn.ModuleList([
            _LocalResidualBlock(
                cfg.local_channels, cfg.local_kernel_size, int(d), cfg.dropout
            ) for d in cfg.local_dilations
        ])
        # Mean, standard deviation, max retain complementary motion statistics.
        self.local_projection = nn.Sequential(
            nn.Linear(cfg.local_channels * 3, cfg.embedding_dim),
            nn.LayerNorm(cfg.embedding_dim), nn.GELU(),
        )
        self.context_projection = nn.Linear(
            cfg.embedding_dim + cfg.quality_dim + 1, cfg.embedding_dim
        )
        self.temporal_blocks = nn.ModuleList([
            _TemporalResidualBlock(cfg.embedding_dim, int(d), cfg.dropout)
            for d in cfg.temporal_dilations
        ])
        self.output_norm = nn.LayerNorm(cfg.embedding_dim)
        self.classifier = nn.Sequential(
            nn.Linear(cfg.embedding_dim, 64), nn.GELU(), nn.Linear(64, 3)
        )
        self.boundary_head = nn.Sequential(
            nn.Linear(cfg.embedding_dim, 64), nn.GELU(), nn.Linear(64, 2)
        )
        # Starts/ends are sparse; this prior is learned normally during training.
        nn.init.constant_(self.boundary_head[-1].bias, -3.0)

    @staticmethod
    def _contiguous_runs(valid: Tensor, segment_ids: Tensor | None) -> Tensor:
        if segment_ids is None:
            segment_ids = torch.zeros_like(valid, dtype=torch.long)
        first = torch.ones_like(valid[:, :1])
        changed = (
            ~valid[:, :-1]
            | (segment_ids[:, 1:] != segment_ids[:, :-1])
        )
        starts = valid & torch.cat((first, changed), dim=1)
        run_ids = starts.long().cumsum(dim=1)
        return torch.where(valid, run_ids, torch.full_like(run_ids, -1))

    def _local_encode(self, imu: Tensor, mask: Tensor) -> Tensor:
        count = mask.sum(dim=-1, keepdim=True).clamp_min(1).to(imu.dtype)
        valid = mask[:, None, :]
        x = torch.where(valid, imu, torch.zeros_like(imu))
        x = self.point_projection(x)
        x = torch.where(valid, x, torch.zeros_like(x))
        for block in self.local_blocks:
            x = block(x, mask)
        mean = x.sum(dim=-1) / count
        variance = (
            torch.where(valid, (x - mean[..., None]).square(), torch.zeros_like(x))
            .sum(dim=-1) / count
        )
        std = (variance + 1e-6).sqrt()
        maximum = x.masked_fill(~valid, -torch.inf).amax(dim=-1)
        maximum = torch.where(
            mask.any(dim=-1, keepdim=True), maximum, torch.zeros_like(maximum)
        )
        pooled = torch.cat((mean, std, maximum), dim=-1)
        return self.local_projection(pooled)

    def forward(
        self, imu: Tensor, point_mask: Tensor,
        frame_quality: Tensor | None = None, segment_ids: Tensor | None = None,
    ) -> dict[str, Tensor]:
        cfg = self.config
        if imu.ndim != 4:
            raise ValueError("imu must have shape [B,T,C,L]")
        batch, length, channels, points = imu.shape
        if length < 1 or points < 1 or channels != cfg.input_channels:
            raise ValueError("Empty sequence or wrong IMU channel count")
        if point_mask.shape != (batch, length, points):
            raise ValueError("point_mask must have shape [B,T,L]")
        if not imu.is_floating_point():
            raise TypeError("imu must have a floating dtype")
        # Sanitize before any projection: 0*NaN would still be NaN.
        mask = point_mask.to(device=imu.device, dtype=torch.bool)
        mask = mask & torch.isfinite(imu).all(dim=2)
        valid = mask.any(dim=-1)
        if segment_ids is not None:
            if segment_ids.shape != (batch, length):
                raise ValueError("segment_ids must have shape [B,T]")
            segment_ids = segment_ids.to(device=imu.device, dtype=torch.long)
            valid = valid & (segment_ids >= 0)
        mask = mask & valid[..., None]
        clean = torch.where(mask[:, :, None, :], imu, torch.zeros_like(imu))
        if frame_quality is None:
            quality = imu.new_zeros(batch, length, cfg.quality_dim)
        else:
            if frame_quality.shape != (batch, length, cfg.quality_dim):
                raise ValueError("frame_quality must have shape [B,T,Q]")
            quality = frame_quality.to(device=imu.device, dtype=imu.dtype)
            quality = torch.nan_to_num(quality, nan=0.0, posinf=0.0, neginf=0.0)
            quality = torch.where(valid[..., None], quality, torch.zeros_like(quality))

        local = self._local_encode(
            clean.reshape(batch * length, channels, points),
            mask.reshape(batch * length, points),
        ).reshape(batch, length, cfg.embedding_dim)
        coverage = mask.to(imu.dtype).mean(dim=-1, keepdim=True)
        x = self.context_projection(torch.cat((local, quality, coverage), dim=-1))
        x = torch.where(valid[..., None], x, torch.zeros_like(x))
        runs = self._contiguous_runs(valid, segment_ids)
        for block in self.temporal_blocks:
            x = block(x, valid, runs)
        embedding = torch.where(
            valid[..., None], self.output_norm(x), torch.zeros_like(x)
        )
        logits = torch.where(
            valid[..., None], self.classifier(embedding),
            embedding.new_zeros(batch, length, 3),
        )
        boundary = torch.where(
            valid[..., None], self.boundary_head(embedding),
            embedding.new_zeros(batch, length, 2),
        )
        probability = torch.where(
            valid, logits.softmax(dim=-1)[..., 1:].sum(dim=-1),
            logits.new_zeros(batch, length),
        )
        return dict(
            logits=logits, boundary=boundary, boundary_logits=boundary,
            embedding=embedding, valid=valid, probability=probability,
        )

