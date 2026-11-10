
from __future__ import annotations
from dataclasses import asdict, dataclass
import math
import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass(frozen=True)
class ModelConfig:
    input_channels: int = 12
    quality_dim: int = 8
    local_channels: int = 64
    embedding_dim: int = 128
    subtoken_count: int = 6
    local_kernel_size: int = 15
    local_dilations: tuple[int, ...] = (1, 2, 4)
    subtoken_dilations: tuple[int, ...] = (1, 2)
    short_dilations: tuple[int, ...] = (1, 2)
    temporal_dilations: tuple[int, ...] = (1, 2, 4, 8)
    dropout: float = .15
    token_dim: int = 64

    def to_dict(self):
        return asdict(self)


class PointBlock(nn.Module):
    def __init__(self, dim, kernel, dilation, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.depthwise = nn.Conv1d(dim, dim, kernel, dilation=dilation,
                                  padding=dilation*(kernel-1)//2, groups=dim)
        self.mlp = nn.Sequential(nn.Conv1d(dim, 2*dim, 1), nn.GELU(),
                                 nn.Dropout(dropout), nn.Conv1d(2*dim, dim, 1))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask):
        m = mask[:, None]
        y = self.norm(x.transpose(1, 2)).transpose(1, 2)
        y = torch.where(m, y, torch.zeros_like(y))
        y = self.mlp(self.depthwise(y))
        return torch.where(m, x+self.drop(y), torch.zeros_like(x))


def contiguous_runs(valid, segments=None):
    if segments is None:
        segments = torch.zeros_like(valid, dtype=torch.long)
    changed = (~valid[:, :-1]) | (segments[:, 1:] != segments[:, :-1])
    starts = valid & torch.cat((torch.ones_like(valid[:, :1]), changed), 1)
    ids = starts.long().cumsum(1)
    return torch.where(valid, ids, torch.full_like(ids, -1))


class SegmentConv(nn.Module):
    def __init__(self, dim, dilation):
        super().__init__()
        self.dilation = int(dilation)
        self.weight = nn.Parameter(torch.empty(3, dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        nn.init.normal_(self.weight, std=math.sqrt(1/3))

    def forward(self, x, valid, segments):
        t = x.shape[1]
        index = torch.arange(t, device=x.device)
        y = torch.zeros_like(x)
        for tap, offset in enumerate((-self.dilation, 0, self.dilation)):
            source = index+offset
            inside = (source >= 0) & (source < t)
            source = source.clamp(0, t-1)
            good = (inside[None] & valid & valid.index_select(1, source)
                    & (segments == segments.index_select(1, source)))
            value = torch.where(good[..., None], x.index_select(1, source), torch.zeros_like(x))
            y = y+value*self.weight[tap]
        return torch.where(valid[..., None], y+self.bias, torch.zeros_like(y))


class TemporalBlock(nn.Module):
    def __init__(self, dim, dilation, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.depthwise = SegmentConv(dim, dilation)
        self.mlp = nn.Sequential(nn.Linear(dim, dim*2), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(dim*2, dim))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, valid, runs):
        y = torch.where(valid[..., None], self.norm(x), torch.zeros_like(x))
        y = self.mlp(self.depthwise(y, valid, runs))
        return torch.where(valid[..., None], x+self.drop(y), torch.zeros_like(x))


def _pool(values, mask):
    """Masked mean/std/max over the last axis, never let padding affect stats."""
    m = mask[:, None]
    count = mask.sum(-1, keepdim=True).clamp_min(1).to(values.dtype)
    mean = torch.where(m, values, 0.).sum(-1)/count
    variance = torch.where(m, (values-mean[..., None]).square(), 0.).sum(-1)/count
    maximum = values.masked_fill(~m, -torch.inf).amax(-1)
    maximum = torch.where(mask.any(-1)[:, None], maximum, 0.)
    return torch.cat((mean, (variance+1e-6).sqrt(), maximum), -1)


class IMUV8Model(nn.Module):
    """All-frame statistics retained; token TCN runs across frame boundaries.

    local/context/joint probabilities have independent losses. A small gate
    mixes their probabilities; there is no bounded additive context correction.
    Role is conditional on eating and never hard-selects the meal predictor.
    """
    def __init__(self, config=None, **kwargs):
        super().__init__()
        if config is not None and kwargs: raise ValueError('Use config or keywords')
        self.config = ModelConfig(**config) if isinstance(config,dict) else (config if config is not None else ModelConfig(**kwargs))
        c=self.config
        if c.subtoken_count!=6 or c.input_channels!=12 or c.local_channels<1:
            raise ValueError('V8 requires 12 channels, six 5-second tokens')
        if c.local_kernel_size%2!=1: raise ValueError('Local kernel must be odd')
        self.token_radius=sum(c.subtoken_dilations)
        # Token halo plus context halo: both are sequential dependencies.
        self.temporal_radius=math.ceil(self.token_radius/c.subtoken_count)+max(sum(c.short_dilations),sum(c.temporal_dilations))
        self.temporal_receptive_frames=2*self.temporal_radius+1
        self.local_receptive_points=1+(c.local_kernel_size-1)*sum(c.local_dilations)
        self.project=nn.Conv1d(c.input_channels,c.local_channels,1)
        self.local_blocks=nn.ModuleList(PointBlock(c.local_channels,c.local_kernel_size,d,c.dropout) for d in c.local_dilations)
        self.whole_projection=nn.Sequential(nn.Linear(3*c.local_channels,c.embedding_dim),nn.LayerNorm(c.embedding_dim),nn.GELU())
        self.token_projection=nn.Sequential(nn.Linear(3*c.local_channels,c.token_dim),nn.LayerNorm(c.token_dim),nn.GELU())
        self.token_blocks=nn.ModuleList(TemporalBlock(c.token_dim,d,c.dropout) for d in c.subtoken_dilations)
        self.token_frame_projection=nn.Sequential(nn.Linear(3*c.token_dim,c.embedding_dim),nn.LayerNorm(c.embedding_dim),nn.GELU())
        self.local_projection=nn.Sequential(nn.Linear(2*c.embedding_dim,c.embedding_dim),nn.LayerNorm(c.embedding_dim),nn.GELU())
        self.context_projection=nn.Linear(c.embedding_dim+c.quality_dim+1,c.embedding_dim)
        self.short_blocks=nn.ModuleList(TemporalBlock(c.embedding_dim,d,c.dropout) for d in c.short_dilations)
        self.long_blocks=nn.ModuleList(TemporalBlock(c.embedding_dim,d,c.dropout) for d in c.temporal_dilations)
        self.mix=nn.Sequential(nn.Linear(3*c.embedding_dim,c.embedding_dim),nn.LayerNorm(c.embedding_dim),nn.GELU())
        self.local_meal=nn.Linear(c.embedding_dim,1)
        self.context_meal=nn.Linear(c.embedding_dim,1)
        self.joint_meal=nn.Linear(c.embedding_dim,1)
        self.probability_gate=nn.Sequential(nn.Linear(2*c.embedding_dim+c.quality_dim+1,32),nn.GELU(),nn.Linear(32,3))
        nn.init.zeros_(self.probability_gate[-1].weight)
        with torch.no_grad(): self.probability_gate[-1].bias.copy_(torch.tensor([0.,0.,math.log(2.)]))
        self.role_head=nn.Linear(c.embedding_dim,2)
        self.boundary_head=nn.Linear(c.embedding_dim,2)
        nn.init.constant_(self.boundary_head.bias,-3.)

    def forward(self,imu,point_mask,frame_quality=None,segment_ids=None):
        if imu.ndim!=4: raise ValueError('imu must be [B,T,C,L]')
        b,t,c,length=imu.shape;cfg=self.config
        if c!=cfg.input_channels or t<1 or length%cfg.subtoken_count:
            raise ValueError('Invalid IMU shape')
        if point_mask.shape!=(b,t,length): raise ValueError('Invalid point mask')
        mask=point_mask.bool()&torch.isfinite(imu).all(2)
        valid=mask.any(-1)
        if segment_ids is None: segment_ids=torch.zeros((b,t),dtype=torch.long,device=imu.device)
        if segment_ids.shape!=(b,t): raise ValueError('Invalid segment ids')
        valid&=segment_ids>=0;mask&=valid[...,None]
        quality=imu.new_zeros(b,t,cfg.quality_dim) if frame_quality is None else torch.nan_to_num(frame_quality.to(imu),nan=0.,posinf=0.,neginf=0.)
        if quality.shape!=(b,t,cfg.quality_dim): raise ValueError('Invalid quality')
        quality=torch.where(valid[...,None],quality,0.)
        x=torch.where(mask[:,:,None],imu,0.).reshape(b*t,c,length)
        pm=mask.reshape(b*t,length)
        x=torch.where(pm[:,None],self.project(x),0.)
        for block in self.local_blocks: x=block(x,pm)
        whole=self.whole_projection(_pool(x,pm)).reshape(b,t,-1)
        pieces=cfg.subtoken_count;width=length//pieces
        values=x.reshape(b*t,cfg.local_channels,pieces,width).permute(0,2,1,3).reshape(b*t*pieces,cfg.local_channels,width)
        token_mask=pm.reshape(b*t*pieces,width)
        tokens=self.token_projection(_pool(values,token_mask)).reshape(b,t*pieces,cfg.token_dim)
        token_valid=token_mask.any(-1).reshape(b,t*pieces)
        token_segments=segment_ids.repeat_interleave(pieces,1)
        token_runs=contiguous_runs(token_valid,token_segments)
        tokens=torch.where(token_valid[...,None],tokens,0.)
        for block in self.token_blocks: tokens=block(tokens,token_valid,token_runs)
        # Continuous token features retain order through the token TCN. Masked
        # frame aggregation adds no fixed slot-specific positional parameters.
        token_frame=self.token_frame_projection(_pool(tokens.reshape(b*t,pieces,cfg.token_dim).transpose(1,2),token_valid.reshape(b*t,pieces))).reshape(b,t,-1)
        local=self.local_projection(torch.cat((whole,token_frame),-1))
        local=torch.where(valid[...,None],local,0.)
        coverage=mask.float().mean(-1,keepdim=True)
        base=torch.where(valid[...,None],self.context_projection(torch.cat((local,quality,coverage),-1)),0.)
        runs=contiguous_runs(valid,segment_ids);short,long=base,base
        for block in self.short_blocks: short=block(short,valid,runs)
        for block in self.long_blocks: long=block(long,valid,runs)
        embedding=torch.where(valid[...,None],self.mix(torch.cat((base,short,long),-1)),0.)
        local_logit=self.local_meal(local).squeeze(-1)
        context_logit=self.context_meal(long).squeeze(-1)
        joint_logit=self.joint_meal(embedding).squeeze(-1)
        weights=self.probability_gate(torch.cat((local,embedding,quality,coverage),-1)).softmax(-1)
        expert_logits=torch.stack((local_logit,context_logit,joint_logit),-1)
        probability=torch.where(valid,(weights*expert_logits.sigmoid()).sum(-1),0.)
        logits=torch.where(valid,torch.logit(probability.clamp(1e-6,1-1e-6)),0.)
        role=torch.where(valid[...,None],self.role_head(embedding),0.)
        boundary=torch.where(valid[...,None],self.boundary_head(embedding),0.)
        return dict(logits=logits,meal_logits=logits,probability=probability,role_logits=role,
            role_probability=torch.where(valid[...,None],role.softmax(-1),0.),boundary_logits=boundary,
            boundary=boundary,embedding=embedding,valid=valid,
            local_logits=torch.where(valid,local_logit,0.),context_logits=torch.where(valid,context_logit,0.),
            joint_logits=torch.where(valid,joint_logit,0.),expert_weights=torch.where(valid[...,None],weights,0.))


