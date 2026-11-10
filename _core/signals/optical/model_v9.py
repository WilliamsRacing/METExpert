
from __future__ import annotations

import torch
from torch import nn
from .model import SpanMultiScaleTemporal
from .spectral_convnext_v8 import QualityChannelEncoderV8, clean_column_mask

EXPERT_NAMES = ("feature", "stft")
EXPERT_STATE_NAMES = ("feature", "raw")


def late_fusion_probabilities(probabilities, support, reliability, prior, *, strict_selected=True, validate_prior=True):
    """Fuse [...,2] in [indicator,STFT] order; unsupported output is NaN.

    A zero prior is a strict exclusion. A selected missing expert never supplies
    sigmoid(0) evidence and never turns on an excluded expert.
    """
    if not strict_selected:
        raise ValueError("V9 always uses strict selected-expert support")
    if probabilities.shape != support.shape or support.shape != reliability.shape or probabilities.shape[-1] != 2:
        raise ValueError("V9 fusion needs aligned [...,2] expert arrays")
    prior = torch.as_tensor(prior, dtype=probabilities.dtype, device=probabilities.device)
    if prior.shape != (2,):
        raise ValueError("V9 fusion prior must contain exactly two experts")
    if validate_prior and (not torch.isfinite(prior).all() or bool((prior < 0).any()) or not torch.isclose(
            prior.sum(), prior.new_tensor(1.), atol=1e-6, rtol=1e-6)):
        raise ValueError("V9 fusion prior must be a finite nonnegative two-expert simplex")
    usable = support.bool() & torch.isfinite(probabilities) & (prior > 0)
    q = torch.nan_to_num(reliability, nan=0., posinf=0., neginf=0.).clamp(0, 1)
    weighted = prior * (.5 + .5 * q) * usable
    weights = weighted / weighted.sum(-1, keepdim=True).clamp_min(1e-12)
    clean = torch.where(usable, probabilities.clamp(0, 1), torch.zeros_like(probabilities))
    valid = usable.any(-1)
    fused = (weights * clean).sum(-1)
    return torch.where(valid, fused, torch.full_like(fused, float("nan"))), weights, valid


class PPGEatingNetV9(nn.Module):
    """Two disjoint experts with fold-specific PCA input dimension."""
    def __init__(self, num_features, dim=64, wave_width=48, feature_width=64,
                 dropout=.15, dilations=(1, 2, 4, 8, 16), input_mode="fused", causal=False,
                 short_dilations=(1, 2, 4), wave_encoder="convnextv2_stft",
                 spectral_widths=(24, 48, 96), spectral_depths=(1, 1, 2),
                 spectral_expansion=2, spectral_drop_path=.05, spectral_kernel=(5, 3),
                 num_original_features=68, original_support_excluded_indices=(25,),
                 fusion_weights=(.5, .5), initial_fusion_weights=None,
                 reliability_context_dim=12, modality_dropout=0.,
                 numeric_residual=True, numeric_gain_init=.1,
                 family_coverage_input=True, pca_availability_mode="loading_fraction"):
        super().__init__()
        if isinstance(num_features, bool) or not isinstance(num_features, int) or num_features < 1:
            raise ValueError("V9 needs a positive fold-specific transformed feature count")
        if isinstance(num_original_features, bool) or not isinstance(num_original_features, int) or num_original_features < 1:
            raise ValueError("num_original_features must be a positive integer")
        if input_mode not in ("fused", "raw", "features"):
            raise ValueError("input_mode must be fused/raw/features")
        if wave_encoder not in ("convnextv2_stft", "convnextv2", "stft2d"):
            raise ValueError("V9 uses only its masked ConvNeXt STFT encoder")
        if modality_dropout != 0:
            raise ValueError("V9 experts have independent supervision; modality_dropout is fixed at zero")
        if num_original_features != 68 or tuple(original_support_excluded_indices) != (25,):
            raise ValueError('V9 requires the fixed 68-column schema with history coverage at column 25')
        if pca_availability_mode not in ('loading_fraction', 'legacy_binary'):
            raise ValueError('Unknown V9 PCA availability mode')
        self.family_coverage_input = bool(family_coverage_input)
        self.pca_availability_mode = pca_availability_mode
        self.input_mode = input_mode
        self.num_features = num_features
        self.num_original_features = num_original_features
        self.physiology_num_features = 24
        self.reliability_context_dim = reliability_context_dim
        physical = torch.ones(num_original_features, dtype=torch.bool)
        for index in original_support_excluded_indices:
            if not isinstance(index, int) or not 0 <= index < num_original_features:
                raise ValueError("original support exclusion index is outside the original feature schema")
            physical[index] = False
        if not bool(physical.any()):
            raise ValueError("At least one original physical feature is required")
        self.register_buffer("physiological_feature_mask", physical)
        self.waveform = QualityChannelEncoderV8(dim, spectral_widths, spectral_depths,
            spectral_expansion, dropout, spectral_drop_path, spectral_kernel)
        self.physiology = nn.Sequential(
            nn.Linear(2 * num_features + num_original_features + (5 if self.family_coverage_input else 0), feature_width),
            nn.LayerNorm(feature_width), nn.GELU(), nn.Dropout(dropout), nn.Linear(feature_width, dim))
        self.numeric_residual = bool(numeric_residual)
        self.numeric_gain = nn.Parameter(torch.tensor(float(numeric_gain_init)), requires_grad=self.numeric_residual)
        self.numeric_physiology = nn.Sequential(
            nn.Linear(int(physical.sum()) + num_original_features, 64), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(64, dim))
        if not self.numeric_residual:
            for parameter in self.numeric_physiology.parameters():
                parameter.requires_grad_(False)
        self.feature_temporal = SpanMultiScaleTemporal(dim, short_dilations, dilations, dropout, causal)
        self.raw_temporal = SpanMultiScaleTemporal(dim, short_dilations, dilations, dropout, causal)
        self.feature_eating_head = nn.Linear(dim, 1)
        self.feature_boundary_head = nn.Linear(dim, 2)
        self.raw_head = nn.Linear(dim, 3)
        self.raw_physiology = nn.Linear(dim, 24)
        self.register_buffer("fusion_weights", torch.tensor([.5, .5], dtype=torch.float32))
        self.set_fusion_weights(fusion_weights if initial_fusion_weights is None else initial_fusion_weights)

    @staticmethod
    def _expert_for_state_key(key):
        prefix = key.split(".", 1)[0]
        if prefix == "fusion_weights":
            return None
        if prefix in ("physiology", "physiological_feature_mask", "numeric_physiology", "numeric_gain") or prefix.startswith("feature_"):
            return "feature"
        if prefix == "waveform" or prefix.startswith("raw_"):
            return "raw"
        raise ValueError(f"Unassigned V9 expert state: {key}")

    def expert_parameter_groups(self):
        groups = {name: [] for name in EXPERT_STATE_NAMES}
        seen = set()
        for key, parameter in self.named_parameters(remove_duplicate=False):
            if not parameter.requires_grad:
                continue
            name = self._expert_for_state_key(key)
            if name is None or id(parameter) in seen:
                raise ValueError(f"Shared or unassigned V9 trainable parameter: {key}")
            groups[name].append(parameter)
            seen.add(id(parameter))
        if seen != {id(p) for p in self.parameters() if p.requires_grad}:
            raise ValueError("V9 expert parameter groups do not cover all trainable parameters")
        return groups

    def expert_parameter_group(self, name):
        """Singular convenience alias; the optimizer uses expert_parameter_groups."""
        if name == "stft":
            name = "raw"
        if name not in EXPERT_STATE_NAMES:
            raise ValueError(f"Unknown V9 expert: {name}")
        return self.expert_parameter_groups()[name]

    def expert_state_keys(self, name):
        if name == "stft":
            name = "raw"
        if name not in EXPERT_STATE_NAMES:
            raise ValueError(f"Unknown V9 expert: {name}")
        return tuple(key for key in self.state_dict() if self._expert_for_state_key(key) == name)

    def set_fusion_weights(self, weights):
        prior = torch.as_tensor(weights, dtype=self.fusion_weights.dtype, device=self.fusion_weights.device)
        if prior.shape != (2,) or not torch.isfinite(prior).all() or bool((prior < 0).any()) or not torch.isclose(
                prior.sum(), prior.new_tensor(1.), atol=1e-6, rtol=1e-6):
            raise ValueError("V9 fusion weights must be two finite nonnegative values summing to one")
        with torch.no_grad():
            self.fusion_weights.copy_(prior)

    def forward(self, raw, raw_mask, quality, channel_ids, features, feature_mask, frame_valid,
                spectrogram=None, spectrogram_mask=None, routing_context=None,
                reliability_context=None, original_feature_mask=None,
                physical_feature_support=None, feature_quality=None, numeric_features=None,
                family_coverage=None, temporal_spans=None):
        if raw.ndim != 4 or raw_mask.ndim != 4 or raw.shape[:3] != raw_mask.shape[:3]:
            raise ValueError("V9 raw and observation mask need [batch,frames,channels,samples]")
        shape = raw.shape[:2]
        dim = self.feature_eating_head.in_features
        if tuple(frame_valid.shape) != tuple(shape) or tuple(features.shape) != (*shape, self.num_features) or feature_mask.shape != features.shape:
            raise ValueError("V9 PCA values/masks and frame validity are not aligned")
        valid = torch.isfinite(frame_valid) & (frame_valid > 0)
        clean_feature_mask = torch.nan_to_num(feature_mask.float(), nan=0., posinf=0., neginf=0.).clamp(0, 1)
        clean_feature_mask = clean_feature_mask * torch.isfinite(features)
        if self.pca_availability_mode == 'legacy_binary':
            clean_feature_mask = (clean_feature_mask > 0).float()
        clean_features = torch.where(clean_feature_mask > 0, torch.nan_to_num(features.float(),
            nan=0., posinf=0., neginf=0.), torch.zeros_like(features, dtype=torch.float32))
        if original_feature_mask is None:
            original_mask = raw.new_zeros(*shape, self.num_original_features)
            observed_support = clean_feature_mask.sum(-1) > 0
            observed_coverage = clean_feature_mask.mean(-1)
        else:
            if tuple(original_feature_mask.shape) != (*shape, self.num_original_features):
                raise ValueError("V9 original_feature_mask must match the original engineering schema")
            original_mask = torch.nan_to_num(original_feature_mask.float(), nan=0., posinf=0., neginf=0.).clamp(0, 1)
            physical_mask = original_mask[..., self.physiological_feature_mask]
            observed_support = physical_mask.sum(-1) > 0
            observed_coverage = physical_mask.mean(-1)
        if physical_feature_support is not None:
            if tuple(physical_feature_support.shape) != tuple(shape):
                raise ValueError("physical_feature_support must align with frames")
            observed_support = observed_support & torch.isfinite(physical_feature_support) & (physical_feature_support > 0)
        physical_support = observed_support & valid
        feature_support = physical_support & (self.input_mode != "raw")
        if feature_quality is None:
            fq = observed_coverage
        else:
            if tuple(feature_quality.shape) != tuple(shape):
                raise ValueError("feature_quality must align with frames")
            fq = torch.nan_to_num(feature_quality.float(), nan=0., posinf=0., neginf=0.).clamp(0, 1)
        fq = fq * feature_support
        # Routing/quality metadata never has a free learned path to the class label.
        if reliability_context is not None and tuple(reliability_context.shape) != (*shape, self.reliability_context_dim):
            raise ValueError("V9 reliability_context has an unexpected schema")
        zero_embedding = raw.new_zeros(*shape, dim)
        if self.input_mode != "raw":
            components = [clean_features, clean_feature_mask, original_mask]
            if self.family_coverage_input:
                if family_coverage is None or tuple(family_coverage.shape) != (*shape, 5):
                    raise ValueError("V9 requires five per-family observation fractions")
                coverage = torch.nan_to_num(family_coverage.float(), nan=0., posinf=0., neginf=0.).clamp(0, 1)
                components.append(coverage)
            feature_embedding = self.physiology(torch.cat(components, -1))
            if self.numeric_residual:
                numeric_dim = int(self.num_original_features - 1)
                if numeric_features is None or tuple(numeric_features.shape) != (*shape, numeric_dim):
                    raise ValueError("V9 requires 67 scaled physiological values excluding hr_history_coverage")
                numeric_mask = original_mask[..., self.physiological_feature_mask]
                numeric = torch.where(numeric_mask > 0, torch.nan_to_num(numeric_features.float(),
                    nan=0., posinf=0., neginf=0.), torch.zeros_like(numeric_features, dtype=torch.float32))
                feature_embedding = feature_embedding + self.numeric_gain * self.numeric_physiology(torch.cat((numeric, original_mask), -1))
            feature_embedding = self.feature_temporal(feature_embedding * feature_support[..., None], feature_support,
                None if temporal_spans is None else temporal_spans['feature'])
            feature_prediction = torch.cat((self.feature_eating_head(feature_embedding),
                                             self.feature_boundary_head(feature_embedding)), -1) * feature_support[..., None]
        else:
            feature_embedding = zero_embedding
            feature_prediction = raw.new_zeros(*shape, 3)
        if self.input_mode != "features":
            if spectrogram is None or spectrogram_mask is None:
                raise ValueError("V9 requires cached numerical STFT values and masks")
            effective_mask = torch.nan_to_num(raw_mask.float(), nan=0., posinf=0., neginf=0.) * valid[..., None, None]
            effective_smask = clean_column_mask(spectrogram, spectrogram_mask) * valid[..., None, None]
            available, _, clean_quality, _ = self.waveform.observation(effective_mask, quality, channel_ids, effective_smask)
            raw_support = available.any(-1) & valid
            best_quality = clean_quality.masked_fill(~available, 0.).max(-1).values * raw_support
            raw_embedding, weights = self.waveform(raw, effective_mask, quality, channel_ids, spectrogram, effective_smask)
            raw_embedding = self.raw_temporal(raw_embedding * raw_support[..., None], raw_support,
                None if temporal_spans is None else temporal_spans["raw"])
            raw_prediction = self.raw_head(raw_embedding) * raw_support[..., None]
            raw_feature_prediction = self.raw_physiology(raw_embedding) * raw_support[..., None]
        else:
            raw_support = torch.zeros_like(valid)
            raw_embedding = zero_embedding
            raw_prediction = raw.new_zeros(*shape, 3)
            raw_feature_prediction = raw.new_zeros(*shape, 24)
            weights = raw.new_zeros(*shape, raw.shape[2])
            best_quality = raw.new_zeros(shape)
        support = torch.stack((feature_support, raw_support), -1)
        expert_logits = torch.stack((feature_prediction, raw_prediction), -2)
        expert_probabilities = torch.where(support[..., None], expert_logits.sigmoid(), torch.full_like(expert_logits, float("nan")))
        reliability = torch.stack((fq, best_quality), -1)
        prior = self.fusion_weights
        if self.input_mode == "features":
            prior = prior.new_tensor([1., 0.])
        elif self.input_mode == "raw":
            prior = prior.new_tensor([0., 1.])
        probability, actual_weights, prediction_valid = late_fusion_probabilities(
            expert_probabilities[..., 0], support, reliability, prior, validate_prior=False)
        boundary_probability = (torch.nan_to_num(expert_probabilities[..., 1:], nan=0.) * actual_weights[..., None]).sum(-2)
        p = torch.cat((torch.nan_to_num(probability, nan=.5)[..., None], boundary_probability), -1)
        prediction = torch.logit(p.clamp(1e-6, 1 - 1e-6)) * prediction_valid[..., None]
        embeddings = torch.stack((feature_embedding, raw_embedding), -2)
        embedding = (embeddings * actual_weights[..., None]).sum(-2)
        return {"logits": prediction[..., 0], "start_logits": prediction[..., 1], "end_logits": prediction[..., 2],
            "prediction_valid": prediction_valid, "embedding": embedding, "channel_weights": weights,
            "feature_embedding": feature_embedding, "raw_embedding": raw_embedding,
            "feature_logits": feature_prediction[..., 0], "feature_start_logits": feature_prediction[..., 1],
            "feature_end_logits": feature_prediction[..., 2], "feature_support": feature_support,
            "physical_feature_support": physical_support, "selected_feature_support": feature_support,
            "raw_logits": raw_prediction[..., 0], "raw_start_logits": raw_prediction[..., 1],
            "raw_end_logits": raw_prediction[..., 2], "raw_support": raw_support, "spectral_support": raw_support,
            "expert_probabilities": expert_probabilities[..., 0], "expert_start_probabilities": expert_probabilities[..., 1],
            "expert_end_probabilities": expert_probabilities[..., 2], "expert_support": support,
            "expert_reliability": reliability, "expert_fusion_weights": actual_weights,
            "fusion_feature_support": feature_support, "fusion_raw_support": raw_support,
            "raw_gate": actual_weights[..., 1], "raw_residual": raw.new_zeros(shape),
            "raw_boundary_residual": raw.new_zeros(*shape, 2), "raw_feature_prediction": raw_feature_prediction,
            "raw_observation_quality": best_quality, "feature_observation_quality": fq}
