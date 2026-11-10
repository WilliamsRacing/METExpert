
from __future__ import annotations

import copy
import numpy as np
import torch
from torch import nn
from .model import build_model
from .feature_pca import FeaturePCA, VERSION
from .quality_v9 import quality_context
from .data_v9 import build_temporal_spans
from .schema import V5_FEATURE_NAMES
# This is the unchanged native physical schema. Importing standalone V9's
# feature extractor would unnecessarily load its pandas-based TXT parser.
from signals.optical_features.features_v8 import NEW_FEATURE_NAMES


class PPGV9RuntimeAdapter(nn.Module):
    v2_runtime = True

    def __init__(self, checkpoint):
        super().__init__()
        if checkpoint.get("format") != "ppg-v9-r2-inference-v1":
            raise ValueError("Expected a complete V9 r2 inference checkpoint")
        self.config = copy.deepcopy(checkpoint["config"])
        if (self.config["model"].get("version") != "v9" or
                self.config["model"].get("revision") != "r2_precision_repair"):
            raise ValueError("Unsupported PPG model/revision")
        if self.config["model"].get("dim",64)!=64:
            raise ValueError("V2 DynamicFusion requires a 64-dimensional PPG embedding")
        self.pca = FeaturePCA.from_dict(checkpoint["scaler"])
        if self.pca.payload.get("version") != VERSION:
            raise ValueError("Wrong V9 PCA version")
        if tuple(self.pca.names) != V5_FEATURE_NAMES + tuple(NEW_FEATURE_NAMES):
            raise ValueError("V9 requires its exact ordered 68-column physical schema")
        self.core = build_model(self.config["model"], self.pca.output_dim)
        self.core.load_state_dict(checkpoint["model_state"], strict=True)
        self.num_original_features = 68
        self.postprocess = copy.deepcopy(checkpoint["postprocess"])
        if self.postprocess.get("schema_version") not in ("v9_postprocess_1", "v9_postprocess_2"):
            raise ValueError("Unsupported V9 postprocess schema")
        self.candidate = self.postprocess["candidates"][self.postprocess["selected_candidate"]]
        alpha = float(self.candidate["alpha"])
        if not np.isfinite(alpha) or not 0 <= alpha <= 1:
            raise ValueError("V9 fusion alpha must be on [0,1]")
        for expert in ("feature", "raw"):
            p = self.postprocess["calibration"]["experts"][expert]
            if not .5 <= float(p["temperature"]) <= 3 or not -2 <= float(p["bias"]) <= 2:
                raise ValueError("Invalid frozen V9 calibration")

    def _frozen_frames(self, out):
        support = out["expert_support"].bool()
        logits = torch.stack((out["feature_logits"], out["raw_logits"]), -1).float()
        if self.candidate["calibrated"]:
            params = self.postprocess["calibration"]["experts"]
            temperatures = logits.new_tensor([params[e]["temperature"] for e in ("feature", "raw")])
            biases = logits.new_tensor([params[e]["bias"] for e in ("feature", "raw")])
            logits = logits / temperatures + biases
        probabilities = logits.sigmoid()
        alpha = float(self.candidate["alpha"])
        prior = logits.new_tensor([alpha, 1 - alpha])
        usable = support & (prior > 0)
        if not torch.isfinite(probabilities[usable]).all():
            raise ValueError("Nonfinite selected supported V9 scores")
        q = torch.nan_to_num(out["expert_reliability"].float(), nan=0., posinf=0., neginf=0.).clamp(0, 1)
        evidence = .5 + .5 * q if self.candidate["quality_correction"] else torch.ones_like(q)
        unnormalized = prior * evidence * usable
        weights = unnormalized / unnormalized.sum(-1, keepdim=True).clamp_min(1e-12)
        valid = usable.any(-1)
        probability = (torch.where(usable, probabilities, torch.zeros_like(probabilities)) * weights).sum(-1)
        embeddings = torch.stack((out["feature_embedding"], out["raw_embedding"]), -2)
        embedding = (torch.where(usable[..., None], embeddings, torch.zeros_like(embeddings)) * weights[..., None]).sum(-2)
        result = dict(out)
        result.update(logits=torch.logit(probability.clamp(1e-6, 1 - 1e-6)) * valid,
                      probability=probability, embedding=embedding,
                      prediction_valid=valid, expert_fusion_weights=weights, raw_gate=weights[..., 1])
        # V9 calibration adjusts eating scores only, exactly as apply_calibration.
        for side in ("start", "end"):
            branch = torch.stack((out["feature_" + side + "_logits"], out["raw_" + side + "_logits"]), -1).sigmoid()
            p = (torch.where(usable, branch, torch.zeros_like(branch)) * weights).sum(-1)
            result[side + "_logits"] = torch.logit(p.clamp(1e-6, 1 - 1e-6)) * valid
        return result

    def forward(self, raw, raw_mask, quality, channel_ids, features, feature_mask,
                frame_valid, spectrogram=None, spectrogram_mask=None,
                routing_context=None, segment_ids=None, **kwargs):
        if spectrogram is None or spectrogram_mask is None:
            raise ValueError("V9 requires numerical STFT and observed-column masks")
        values = features.detach().cpu().numpy()
        details = self.pca.transform_details(values, feature_mask.detach().cpu().numpy())
        shape = values.shape[:-1]
        context = quality_context(values.reshape(-1, 68), details["raw_mask"].reshape(-1, 68),
            {k: v.reshape((-1,) + v.shape[len(shape):]) for k, v in details.items()},
            quality.detach().cpu().numpy().reshape(-1, quality.shape[-1]),
            spectrogram_mask.detach().cpu().numpy().reshape(-1, *spectrogram_mask.shape[-2:]))
        inputs = dict(raw=raw, raw_mask=raw_mask, quality=quality, channel_ids=channel_ids,
                      frame_valid=frame_valid, spectrogram=spectrogram, spectrogram_mask=spectrogram_mask)
        for name, key in (("features", "features"), ("feature_mask", "feature_mask"),
                          ("original_feature_mask", "raw_mask"), ("physical_feature_support", "supported"),
                          ("feature_quality", "reliability"), ("numeric_features", "numeric_features"),
                          ("family_coverage", "family_coverage")):
            inputs[name] = torch.as_tensor(details[key], device=features.device)
        inputs["reliability_context"] = torch.as_tensor(context.reshape(*shape, 12), device=features.device)
        if segment_ids is None:
            raise ValueError("V4 requires explicit PPG physical segment IDs")
        if tuple(segment_ids.shape) != tuple(shape):
            raise ValueError("PPG segment IDs are not aligned with frames")
        # CPU metadata is built before the core's CUDA forward; masks never jump holes.
        rows = []
        for bi in range(shape[0]):
            ids = segment_ids[bi].detach().cpu().numpy()
            cuts = np.r_[0, np.flatnonzero(ids[1:] != ids[:-1]) + 1, shape[1]]
            pieces = []
            for a, b in zip(cuts[:-1], cuts[1:]):
                piece = {k: v[bi:bi+1, int(a):int(b)] for k, v in inputs.items()}
                piece["frame_valid"] = piece["frame_valid"] * (int(ids[a]) >= 0)
                cpu = {k: v.detach().cpu() for k, v in piece.items()}
                piece["temporal_spans"] = build_temporal_spans(cpu)
                pieces.append(self._frozen_frames(self.core(**piece)))
            rows.append({k: torch.cat([p[k] for p in pieces], 1) for k in pieces[0]})
        return {k: torch.cat([r[k] for r in rows], 0) for k in rows[0]}
