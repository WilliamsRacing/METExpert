
from __future__ import annotations

import copy
import torch
from torch import nn

from .model import build_model
from .feature_pca import FeaturePCA




class PPGV8RuntimeAdapter(nn.Module):
    """Expose V8 as the stable V1 PPG expert interface.

    The adapter receives the 68 physical features produced by the V2 runtime
    preprocessing and returns the same ``logits``, ``embedding`` and support
    fields consumed by the unchanged V1 point-level fusion.
    """

    v2_runtime = True

    def __init__(self, checkpoint):
        super().__init__()
        config = copy.deepcopy(checkpoint["config"])
        pca_payload = checkpoint.get("scaler") or checkpoint.get("pca")
        if pca_payload is None:
            raise ValueError("V2 PPG checkpoint must contain its PCA payload")
        self.pca = FeaturePCA.from_dict(pca_payload)
        self.core = build_model(config["model"], self.pca.output_dim)
        self.core.load_state_dict(checkpoint["model_state"], strict=True)
        self.config = config
        self.num_original_features = len(self.pca.names)

    def forward(self, raw, raw_mask, quality, channel_ids, features, feature_mask,
                frame_valid, spectrogram=None, spectrogram_mask=None,
                routing_context=None, segment_ids=None, **kwargs):
        details = self.pca.transform_details(features.detach().cpu().numpy(),
                                             feature_mask.detach().cpu().numpy())
        transformed = {key: torch.as_tensor(details[key], device=features.device)
                       for key in ("features", "feature_mask", "supported", "reliability")}
        inputs = dict(raw=raw, raw_mask=raw_mask, quality=quality, channel_ids=channel_ids,
            features=transformed["features"], feature_mask=transformed["feature_mask"],
            frame_valid=frame_valid, spectrogram=spectrogram, spectrogram_mask=spectrogram_mask,
            original_feature_mask=feature_mask,
            physical_feature_support=transformed["supported"], feature_quality=transformed["reliability"])
        if segment_ids is None:
            return self.core(**inputs)
        rows = []
        for bi in range(features.shape[0]):
            ids = segment_ids[bi].detach().cpu().tolist()
            cuts = [0] + [i for i in range(1, len(ids)) if ids[i] != ids[i-1]] + [len(ids)]
            pieces = [self.core(**{k: v[bi:bi+1, a:b] for k, v in inputs.items()})
                      for a, b in zip(cuts[:-1], cuts[1:])]
            rows.append({k: torch.cat([piece[k] for piece in pieces], 1) for k in pieces[0]})
        return {k: torch.cat([row[k] for row in rows], 0) for k in rows[0]}
