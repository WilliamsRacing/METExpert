from __future__ import annotations
import math
import numpy as np
from .data import _validated_frame_masks
from .losses_v9 import segmentation_loss_v9 as segmentation_loss

def estimate_loss_weights(records, maximum=15, frame_masks=None, *, positive_power=1., boundary_maximum=None):
    """Estimate only from training-owned frames; V1 ratio/caps remain the defaults.

    Classification uses clip((weighted_negative/weighted_positive)**power).
    Boundary ratios retain an independently configurable cap and no power.
    Input support/augmentation can reduce actual supervised counts per batch.
    """
    import numpy as np
    boundary_maximum = maximum if boundary_maximum is None else boundary_maximum
    if not math.isfinite(positive_power) or positive_power <= 0:
        raise ValueError("positive_power must be finite and positive")
    if any(not math.isfinite(value) or value < 1 for value in (maximum, boundary_maximum)):
        raise ValueError("Loss weight caps must be finite and at least 1")
    ownership = _validated_frame_masks(records, frame_masks)
    positive = negative = 0.; bp = np.zeros(2); bn = np.zeros(2)
    for ri, rec in enumerate(records):
        with np.load(rec["_path"], allow_pickle=False) as z:
            y, w = z["target"], z["label_weight"]
            owned = ownership[ri]
            positive += float(w[(y == 1) & owned].sum())
            negative += float(w[(y == 0) & owned].sum())
            for j, key in enumerate(("start_target", "end_target")):
                yy, mm = z[key], z["boundary_mask"][:, j].astype(bool) & owned
                bp[j] += float(np.sum(mm & (yy > 0)))
                bn[j] += float(np.sum(w[mm & (yy == 0)]))
    if positive <= 0 or negative <= 0:
        raise ValueError("Training groups need positive eating labels and background; check background policy and split")
    ratio = negative / positive
    pw = float(np.clip(ratio ** positive_power, 1, maximum))
    return pw, np.clip(bn / np.maximum(bp, 1), 1, boundary_maximum).tolist(), {
        "weighted_positive_frames": positive, "weighted_negative_frames": negative,
        "positive_boundary_frames": bp.tolist(), "classification_ratio": ratio,
        "pos_weight_power": positive_power, "max_pos_weight": maximum,
        "max_boundary_pos_weight": boundary_maximum,
        "owned_frames": sum(int(mask.sum()) for mask in ownership),
        "duplicate_frames_excluded": sum(len(mask) - int(mask.sum()) for mask in ownership)}
