from __future__ import annotations
import json
import random
from pathlib import Path
import numpy as np
import torch


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def choose_device(name="auto"):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; install a CUDA PyTorch build or use --device cpu")
    device=torch.device(name)
    if device.type=="cuda":
        device=torch.device("cuda",torch.cuda.current_device() if device.index is None else device.index)
        torch.cuda.set_device(device)
    return device


MODEL_FIELDS = ("raw", "raw_mask", "quality", "channel_ids", "features", "feature_mask", "frame_valid")


def model_inputs(batch, device):
    """Re-use an already moved device batch or build its CPU spans before H2D."""
    fields = MODEL_FIELDS + tuple(k for k in (
        'spectrogram', 'spectrogram_mask', 'routing_context', 'reliability_context',
        'original_feature_mask', 'physical_feature_support', 'feature_quality',
        'numeric_features', 'family_coverage') if k in batch)
    from .data_v9 import prepare_device_batch
    moved = prepare_device_batch(batch, device)
    return {**{k: moved[k] for k in fields}, 'temporal_spans': moved['temporal_spans']}


from .data_v9 import prepare_device_batch


def input_support(batch, input_mode="fused"):
    raw = batch["raw_mask"].sum((-1, -2)) > 0
    from .schema import FEATURE_DIM, V4_FEATURE_DIM, V5_FEATURE_DIM, V5_SUPPORT_INDICES
    mask = batch["feature_mask"]
    # V4 sidecar reliability metadata cannot manufacture physiological support.
    if mask.shape[-1] == V4_FEATURE_DIM:
        mask = mask[..., :FEATURE_DIM]
    elif mask.shape[-1] == V5_FEATURE_DIM:
        mask = mask[..., list(V5_SUPPORT_INDICES)]
    if "spectrogram_mask" in batch:
        raw = raw & (batch["spectrogram_mask"].sum((-1, -2)) > 0)
    feature = batch.get("physical_feature_support", mask.sum(-1) > 0) > 0
    return feature if input_mode == "features" else raw if input_mode == "raw" else raw | feature


def validate_cache_config(records, config):
    for rec in records:
        with np.load(rec["_path"], allow_pickle=False) as z:
            if abs(float(z["fs"]) - config["fs"]) > .01:
                raise ValueError("Cache fs differs from training config")
            if abs(float(z["frame_seconds"]) - config["frame_seconds"]) > .01:
                raise ValueError("Cache frame duration differs from training config")
            if z["channel_ids"].shape[1] != config["top_k"]:
                raise ValueError("Cache top_k differs from training config")
            if config['model'].get('version') in ('v8', 'v9'):
                from .schema import V5_FEATURE_NAMES
                from .features_v8 import NEW_FEATURE_NAMES
                if tuple(z['feature_names'].astype(str)) != V5_FEATURE_NAMES + tuple(NEW_FEATURE_NAMES):
                    raise ValueError('V8 requires exact 68-column schema')
                if str(z['schema_version']) != 'ppg-eating-cache-v8' or 'canonical_mask' not in z:
                    raise ValueError('Incomplete V8 cache contract')
                if any(k.startswith('papagei') for k in z.files):
                    raise ValueError('V8 cache must not contain pretrained embeddings')
                from .cache_io import load_spectrogram
                load_spectrogram(rec)


def load_checkpoint(path):
    # Only this package's tensor/primitive checkpoint format is supported.
    return torch.load(path, map_location="cpu", weights_only=True)


def cache_identity_aliases(cache):
    path = Path(cache) / "subject_mapping.json"
    return read_json(path).get("aliases", {}) if path.exists() else {}


def validate_checkpoint_identity(checkpoint, cache):
    previous = checkpoint.get("identity_aliases", {})
    if not previous:
        return  # Synthetic cache has no personal alias table.
    current = cache_identity_aliases(cache)
    if not current:
        raise ValueError("Inference cache lacks subject_mapping.json; reuse the training identity mapping")
    changed = [alias for alias in previous.keys() & current.keys() if previous[alias] != current[alias]]
    if changed:
        raise ValueError("Identity IDs changed: reprepare with --subject-mapping from the training cache")
