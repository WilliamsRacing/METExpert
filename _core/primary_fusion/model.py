
from __future__ import annotations

import math
import torch
from torch import Tensor, nn


def _masked_weights(scores: Tensor, available: Tensor) -> Tensor:
    weights = scores.float().masked_fill(~available, -1e4).softmax(-1) * available
    return weights / weights.sum(-1, keepdim=True).clamp_min(1e-12)


def _finite_active(value: Tensor, active: Tensor, name: str) -> Tensor:
    """Ignore missing evidence before arithmetic; reject corrupt active evidence."""
    expanded = active
    while expanded.ndim < value.ndim:
        expanded = expanded.unsqueeze(-1)
    clean = torch.where(expanded, value, torch.zeros_like(value))
    if not bool(torch.isfinite(clean).all()):
        raise ValueError(f"Nonfinite active {name}")
    return clean.float()


class FusionModel(nn.Module):
    """Calibrated convex logit fusion with exact point-level availability.

    Expert outputs/normalization remain frozen outside this module. IMU logits
    may be frame-level [B,F] or point-level [B,F,L]. Physical IMU support is
    [B,F,L]; PPG support is [B,F]. Quality describes evidence and is separate
    from availability. No unknown/missing modality is turned into background.
    """

    def __init__(self, dropout: float = .1):
        super().__init__()
        self.imu_projection = nn.Linear(128, 16)
        self.ppg_projection = nn.Linear(64, 16)
        # 32 projected dimensions plus 18 scalar/quality/role covariates.
        self.gate = nn.Sequential(
            nn.Linear(50, 32), nn.GELU(), nn.Dropout(dropout), nn.Linear(32, 2)
        )
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)
        # .5 + 2.5*sigmoid(logit(.2)) == 1 at initialization.
        self.temperature_parameter = nn.Parameter(torch.full((2,), math.log(.2 / .8)))
        self.bias_parameter = nn.Parameter(torch.zeros(2))

    @property
    def temperature(self) -> Tensor:
        return .5 + 2.5 * self.temperature_parameter.sigmoid()

    @property
    def bias(self) -> Tensor:
        return 3 * self.bias_parameter.tanh()

    def forward(
        self, imu_logits: Tensor, imu_embedding: Tensor, imu_support: Tensor,
        imu_quality: Tensor, imu_role_prob: Tensor, ppg_logits: Tensor,
        ppg_embedding: Tensor, ppg_support: Tensor, ppg_quality: Tensor,
        wave_coverage: Tensor, feature_coverage: Tensor,
    ) -> dict[str, Tensor]:
        if imu_support.ndim != 3 or imu_support.shape[-1] < 1:
            raise ValueError("imu_support must have shape [B,F,L]")
        shape = imu_support.shape[:2]
        if ppg_logits.shape != shape or ppg_support.shape != shape:
            raise ValueError("PPG frame logits/support must align with IMU frames")
        if imu_embedding.shape != (*shape, 128) or ppg_embedding.shape != (*shape, 64):
            raise ValueError("Expected IMU128 and PPG64 frame embeddings")
        if imu_quality.shape != (*shape, 8) or imu_role_prob.shape != (*shape, 3):
            raise ValueError("Expected eight IMU quality and three role inputs")
        for name, value in (("ppg_quality", ppg_quality), ("wave_coverage", wave_coverage),
                            ("feature_coverage", feature_coverage)):
            if value.shape != shape:
                raise ValueError(f"{name} must have shape [B,F]")
        imask = imu_support.bool()
        iavailable, pavailable = imask.any(-1), ppg_support.bool()
        if imu_logits.shape == shape:
            iframe = _finite_active(imu_logits, iavailable, "IMU logit")
            ipoint = iframe[..., None].expand_as(imask)
        elif imu_logits.shape == imask.shape:
            ipoint = _finite_active(imu_logits, imask, "IMU point logit")
            iframe = ipoint.sum(-1) / imask.sum(-1).clamp_min(1)
        else:
            raise ValueError("imu_logits must have shape [B,F] or [B,F,L]")
        plogit = _finite_active(ppg_logits, pavailable, "PPG logit")
        iembedding = _finite_active(imu_embedding, iavailable, "IMU embedding")
        pembedding = _finite_active(ppg_embedding, pavailable, "PPG embedding")

        def covariate(value, active):
            clean = torch.nan_to_num(value.float(), nan=0., posinf=0., neginf=0.).clamp(0, 1)
            while active.ndim < clean.ndim:
                active = active.unsqueeze(-1)
            return torch.where(active, clean, torch.zeros_like(clean))

        iq = covariate(imu_quality, iavailable)
        role = covariate(imu_role_prob, iavailable)
        pq = covariate(ppg_quality, pavailable)
        wc = covariate(wave_coverage, pavailable)
        fc = covariate(feature_coverage, pavailable)
        iproject = torch.where(iavailable[..., None], self.imu_projection(iembedding),
                               iembedding.new_zeros(*shape, 16))
        pproject = torch.where(pavailable[..., None], self.ppg_projection(pembedding),
                               pembedding.new_zeros(*shape, 16))
        disagreement = torch.where(iavailable & pavailable, (iframe - plogit).abs(), 0.)
        scalar = torch.cat((torch.stack((iframe, plogit, imask.float().mean(-1), pq, wc, fc), -1),
                            iq, role, disagreement[..., None]), -1)
        features = torch.cat((iproject, pproject, scalar), -1)
        available = torch.stack((iavailable, pavailable), -1)
        frame_weights = _masked_weights(self.gate(features), available)
        point_available = torch.stack((imask, pavailable[..., None].expand_as(imask)), -1)
        effective = frame_weights[..., None, :] * point_available
        effective = effective / effective.sum(-1, keepdim=True).clamp_min(1e-12)
        iz = ipoint / self.temperature[0] + self.bias[0]
        pz = plogit[..., None] / self.temperature[1] + self.bias[1]
        logits = effective[..., 0] * iz + effective[..., 1] * pz
        supported = point_available.any(-1)
        logits = torch.where(supported, logits, torch.zeros_like(logits))
        probability = torch.where(supported, logits.sigmoid(), torch.zeros_like(logits))
        return dict(logits=logits, probability=probability, supported=supported,
                    frame_weights=frame_weights, point_weights=effective,
                    temperature=self.temperature, bias=self.bias, gate_scalars=scalar)
