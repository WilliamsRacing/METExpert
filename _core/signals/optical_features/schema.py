"""Versioned, PPG-only cached frame contract; no metadata is a model feature."""
from __future__ import annotations

SCHEMA_VERSION = "ppg-eating-cache-v1"
BASE_FEATURE_NAMES = (
    "hr_bpm_60", "rmssd_ms_60", "rmssd_cycle_60", "sdnn_ms_120",
    "pnn50_120", "sd1_sd2_120", "hf_lf_120", "ac_dc_60",
    "rise_fraction_60", "width_fraction_60", "prepeak_area_fraction_60",
    "r_red_ir_60",
)
FEATURE_NAMES = BASE_FEATURE_NAMES + tuple(n + "__relative_baseline" for n in BASE_FEATURE_NAMES)
FEATURE_DIM = len(FEATURE_NAMES)
# V3 cache remains immutable and 24-dimensional. V4 appends a sidecar only.
HR_REFERENCE_NAMES = ("hr_relative_pooled", "hr_history_coverage", "hr_history_relative_mad")
V4_FEATURE_NAMES = FEATURE_NAMES + HR_REFERENCE_NAMES
V4_FEATURE_DIM = len(V4_FEATURE_NAMES)
CANDIDATE_CHANNELS = (1,) + tuple(range(3, 21))
GREEN_CHANNELS = tuple(range(3, 9))
RED_CHANNELS = (12, 13, 14, 18, 19, 20)
IR_CHANNELS = (1, 9, 10, 11, 15, 16, 17)
FEATURE_LOOKBACK_SECONDS = (60, 60, 60, 120, 120, 120, 120, 60, 60, 60, 60, 60)
NPZ_REQUIRED_KEYS = (
    "raw", "raw_mask", "quality", "channel_ids", "feature_values", "feature_mask",
    "feature_names", "feature_channel_ids", "frame_start_ms", "frame_end_ms", "target",
    "start_target", "end_target", "label_weight", "boundary_mask", "meal_overlap_fraction",
    "recording_id", "subject_id", "continuous_id", "fs", "frame_seconds", "meals",
)


def wavelength_id(channel: int) -> int:
    """0 missing, 1 green, 2 red, 3 infrared (ambient PPG2 excluded)."""
    if channel in GREEN_CHANNELS:
        return 1
    if channel in RED_CHANNELS:
        return 2
    if channel in IR_CHANNELS:
        return 3
    return 0

# V5 full independent cache: physical physiology and reliability are explicit.
V5_NEW_FEATURE_NAMES = (
    "hr_bpm_300", "hr_short_long_diff", "hr_slope_bpm_min_300",
    "rmssd_ms_300", "sdnn_ms_300", "cvnn_300", "sd2_ms_300",
    "lf_ms2_300", "hf_ms2_300", "hf_lf_300", "ln_rmssd_short_long_diff",
    "amplitude_robust_cv_120", "width_fraction_mad_120",
)
V5_FEATURE_NAMES = V4_FEATURE_NAMES + V5_NEW_FEATURE_NAMES
V5_FEATURE_DIM = len(V5_FEATURE_NAMES)
V5_SUPPORT_INDICES = tuple(i for i in range(V5_FEATURE_DIM) if i != 25)
V5_SCHEMA_VERSION = "ppg-eating-cache-v5"
