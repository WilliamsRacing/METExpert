
from __future__ import annotations

import numpy as np
from scipy import stats

from .multiscale_features import NEW_FEATURE_NAMES, QUALITY_METRIC_NAMES, QUALITY_WINDOWS_SECONDS, extract_channel_beats, interval_spectrum, _window
from .preprocess import channel_quality
from .schema import CANDIDATE_CHANNELS

FAMILY_NAMES = (
    "hr_level", "hr_trend", "time_hrv", "frequency_hrv",
    "rmssd_contrast", "morphology",
)
FAMILY_QUALITY_NAMES = (
    "support_fraction", "time_span_s", "max_untrusted_gap_s",
    "valid_observation_fraction", "sqi",
)
FAMILY_FEATURE_INDICES = ((0, 1), (2,), (3, 4, 5, 6), (7, 8, 9), (10,), (11, 12))


def _validate_beats(beats):
    """Reject malformed cached arrays instead of silently restoring support."""
    interval_names = ("interval_start_s", "interval_end_s", "interval_time_s",
                      "interval_s", "interval_valid", "interval_available_s")
    morph_names = ("morph_time_s", "morph_available_s", "amplitude", "width_fraction")
    result = {key: np.asarray(value) for key, value in beats.items()}
    for names in (interval_names, morph_names):
        if any(name not in result for name in names):
            raise ValueError("beat_cache lacks required interval or morphology arrays")
        if any(result[name].ndim != 1 for name in names):
            raise ValueError("beat-cache arrays must be one-dimensional")
        if len({len(result[name]) for name in names}) != 1:
            raise ValueError("beat-cache arrays must have aligned lengths")
    for name in ("interval_start_s", "interval_end_s", "interval_time_s",
                 "interval_available_s", "morph_time_s", "morph_available_s"):
        x = result[name]
        if np.any(~np.isfinite(x)) or np.any(x < 0) or np.any(np.diff(x) < 0):
            raise ValueError("beat-cache timestamps must be finite increasing segment times")
    if (np.any(result["interval_end_s"] <= result["interval_start_s"])
            or np.any(result["interval_available_s"] < result["interval_end_s"])
            or np.any(result["morph_available_s"] < result["morph_time_s"])):
        raise ValueError("beat-cache availability precedes the observation")
    result["interval_valid"] = result["interval_valid"].astype(bool)
    return result


def _ppi_window(beats, end_s, seconds, *, minimum_coverage=.80,
                minimum_span_fraction=.90, maximum_gap=5., minimum_valid_fraction=0.,
                required_bins=0):
    left = end_s - seconds
    if left < -1e-7:
        return {}
    starts, ends = beats["interval_start_s"], beats["interval_end_s"]
    lo = int(np.searchsorted(starts, left, side="left"))
    hi = int(np.searchsorted(beats["interval_available_s"], end_s, side="right"))
    if hi <= lo:
        return {}
    valid = (beats["interval_valid"][lo:hi]
             & np.isfinite(beats["interval_s"][lo:hi])
             & (beats["interval_s"][lo:hi] >= 1. / 3.5)
             & (beats["interval_s"][lo:hi] <= 1. / .6)
             & (ends[lo:hi] <= end_s))
    ids = np.arange(lo, hi)[valid]
    if len(ids) < 15 or valid.mean() < minimum_valid_fraction:
        return {}
    rr = beats["interval_s"][ids]
    time = beats["interval_time_s"][ids]
    coverage = float(rr.sum() / seconds)
    span = float(time[-1] - time[0])
    gaps = np.r_[starts[ids[0]] - left,
                 starts[ids[1:]] - ends[ids[:-1]], end_s - ends[ids[-1]]]
    max_gap = float(np.maximum(gaps, 0.).max(initial=0.))
    if (coverage < minimum_coverage or span < minimum_span_fraction * seconds
            or max_gap > maximum_gap):
        return {}
    if required_bins:
        n_bins = 10 if seconds == 300 else 4
        counts, _ = np.histogram(time, np.linspace(left, end_s, n_bins + 1))
        if np.count_nonzero(counts >= 2) < required_bins:
            return {}
    sdnn = float(np.std(rr, ddof=1))
    result = {"hr": float(60. / np.median(rr)), "sdnn": sdnn,
              "cvnn": sdnn / float(np.mean(rr)), "rr": rr, "time": time,
              "quality": np.array([coverage, span, max_gap, valid.mean()])}
    # Deliberately do not take diff(rr): rejected intervals must break adjacency.
    differences = np.diff(rr)[np.diff(ids) == 1]
    if len(differences) >= 15:
        result["rmssd"] = float(np.sqrt(np.mean(differences ** 2)))
        sd1sq = float(np.var(differences, ddof=1) / 2.)
        result["sd2"] = float(np.sqrt(max(2. * sdnn ** 2 - sd1sq, 0.)))
    return result


def _hr_window(beats, end_s, seconds):
    # Robust location/trend tolerates bounded missing intervals; it does not
    # interpolate them or grant the same eligibility to frequency PRV.
    return _ppi_window(beats, end_s, seconds, minimum_coverage=.60,
                       minimum_span_fraction=.80 if seconds == 300 else .75,
                       maximum_gap=30. if seconds == 300 else 15.,
                       minimum_valid_fraction=.60,
                       required_bins=8 if seconds == 300 else 3)


def _hr_trend(beats, end_s):
    if end_s < 300.:
        return None
    points = [(end, _hr_window(beats, end, 60))
              for end in np.arange(end_s - 240., end_s + .01, 30.)]
    accepted = [(end, item) for end, item in points if item]
    if len(accepted) < 7 or accepted[-1][0] - accepted[0][0] < 180.:
        return None
    t = np.array([item[0] for item in accepted])
    # At least one estimate from every chronological third of the sampled ends.
    bins = np.minimum(((t - (end_s - 240.)) / 90.).astype(int), 2)
    if len(np.unique(bins)) < 3:
        return None
    hr = np.array([item[1]["hr"] for item in accepted])
    slope = float(stats.theilslopes(hr, (t - t[0]) / 60.).slope)
    # Describes the accepted subwindow estimates, not an interpolated PPI series.
    q = np.array([len(accepted) / 9., t[-1] - t[0],
                  np.diff(t).max(initial=0.), len(accepted) / 9.])
    return slope, q


def _morph_family(beats, end_s):
    if end_s < 120.:
        return {}, {}
    left = end_s - 120.
    lo = int(np.searchsorted(beats["morph_time_s"], left, side="left"))
    hi = int(np.searchsorted(beats["morph_available_s"], end_s, side="right"))
    if hi <= lo:
        return {}, {}
    t = beats["morph_time_s"][lo:hi]
    result, diagnostics = {}, {}
    for field, name in (("amplitude", "amplitude_robust_cv_120"),
                        ("width_fraction", "width_fraction_mad_120")):
        x = beats[field][lo:hi]
        valid = np.isfinite(x) & (x > 1e-8)
        if field == "width_fraction":
            valid &= x < 1.
        tx, xx = t[valid], x[valid]
        if len(xx) < 30 or valid.mean() < .60:
            continue
        span = float(tx[-1] - tx[0])
        max_gap = float(np.diff(np.r_[left, tx, end_s]).max(initial=0.))
        counts, _ = np.histogram(tx, np.linspace(left, end_s, 7))
        bin_fraction = float(np.mean(counts >= 2))
        if span < 90. or max_gap > 20. or bin_fraction < 5. / 6.:
            continue
        median = float(np.median(xx))
        mad = float(np.median(np.abs(xx - median)))
        result[name] = (1.4826 * mad / median if field == "amplitude" else mad)
        diagnostics[name] = np.array([bin_fraction, span, max_gap, valid.mean()])
    return result, diagnostics


def extract_multiscale_features(segment_raw, segment_support, frame_end_indices,
                                fs=25., channel_ids=None, frame_quality=None,
                                quality_frame_end_indices=None, *, beat_cache=None,
                                segment_length_samples=None):
    """V5-compatible inputs plus optional pre-extracted per-channel beat cache.

    Cached mode: pass raw/support=None, beat_cache={str(id): beat_arrays},
    channel_ids, frame_quality and quality_frame_end_indices. These arrays must
    belong to the SAME continuous segment. No history carries across calls.

    Returns V5 values/mask/channels and quality_metrics with their original
    dimensions. Additional feature_quality[F,13,5] and family_quality[F,6,5]
    describe the actual selected routes. family_channels is 0 for none, -1 for
    a family whose available scalars use different channels, otherwise the id.
    family_mask means ANY scalar in that family is usable, not all scalars.
    Legacy quality_metrics remains strict V5 300/120 s PPI diagnostic only; it
    must NOT be used as a mask for V6 HR/trend or morphology.
    """
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError("fs must be positive")
    raw = None if segment_raw is None else np.asarray(segment_raw)
    support = None if segment_support is None else np.asarray(segment_support, dtype=bool)
    frame_ends = np.asarray(frame_end_indices, dtype=np.int64)
    if (frame_ends.ndim != 1 or np.any(frame_ends < 0)
            or np.any(np.diff(frame_ends) <= 0)):
        raise ValueError("frame ends must be nonnegative strictly increasing")
    if raw is not None:
        if raw.ndim != 2 or support is None or support.shape != raw.shape:
            raise ValueError("raw/support must be aligned 2D arrays")
        length = len(raw)
        if channel_ids is None:
            channel_ids = np.arange(1, 21) if raw.shape[1] == 20 else CANDIDATE_CHANNELS
    else:
        if support is not None or beat_cache is None or channel_ids is None:
            raise ValueError("cached mode requires beat_cache and explicit channel_ids")
        if frame_quality is None:
            raise ValueError("cached mode requires observed frame_quality")
        length = int(segment_length_samples if segment_length_samples is not None
                     else frame_ends.max(initial=0))
    if np.any(frame_ends > length):
        raise ValueError("frame end exceeds this continuous segment")
    ids = np.asarray(channel_ids, dtype=np.int16)
    if (ids.ndim != 1 or len(np.unique(ids)) != len(ids)
            or (raw is not None and len(ids) != raw.shape[1])):
        raise ValueError("unique channel ids must align to input columns")
    columns = [i for i, channel in enumerate(ids) if int(channel) in CANDIDATE_CHANNELS]
    if beat_cache is None:
        beats = {int(ids[c]): extract_channel_beats(raw[:, c], support[:, c], fs)
                 for c in columns}
    else:
        mapped = {int(key): value for key, value in beat_cache.items()}
        missing = [int(ids[c]) for c in columns if int(ids[c]) not in mapped]
        if missing:
            raise ValueError(f"beat_cache lacks physical channels: {missing}")
        beats = {int(ids[c]): _validate_beats(mapped[int(ids[c])]) for c in columns}
    q = q_ends = q_prefix = None
    if frame_quality is not None:
        q = np.asarray(frame_quality, dtype=float)
        q_ends = np.asarray(frame_ends if quality_frame_end_indices is None
                            else quality_frame_end_indices, dtype=np.int64)
        if (q.shape != (len(q_ends), len(ids)) or np.any(~np.isfinite(q))
                or q_ends.ndim != 1 or np.any(np.diff(q_ends) <= 0)):
            raise ValueError("frame_quality must align to increasing quality frame ends")
        q_prefix = np.vstack([np.zeros((1, len(ids))), np.cumsum(q, axis=0)])
    frame_size = int(round(30. * fs))

    def quality(end, seconds):
        left = end - int(round(seconds * fs))
        if left < 0:
            return np.zeros(len(ids))
        if q is not None:
            a = int(np.searchsorted(q_ends, left + frame_size, side="left"))
            b = int(np.searchsorted(q_ends, end, side="right"))
            if (b - a == seconds // 30 and b > a
                    and q_ends[a] == left + frame_size and q_ends[b - 1] == end
                    and np.all(np.diff(q_ends[a:b]) == frame_size)):
                return (q_prefix[b] - q_prefix[a]) / (b - a)
        scores = np.zeros(len(ids))
        if raw is not None:
            for c in columns:
                scores[c] = channel_quality(raw[left:end, c], fs, support[left:end, c])
        return scores

    shape = (len(frame_ends), len(NEW_FEATURE_NAMES))
    values, masks = np.zeros(shape, np.float32), np.zeros(shape, np.float32)
    chosen = np.zeros(shape, np.int16)
    feature_q = np.zeros(shape + (5,), np.float32)
    legacy_q = np.zeros((len(frame_ends), 2, 5), np.float32)
    name_index = {name: i for i, name in enumerate(NEW_FEATURE_NAMES)}
    for row, end in enumerate(frame_ends):
        end, end_s = int(end), float(end / fs)
        qualities = {seconds: quality(end, seconds) for seconds in (60, 120, 300)}
        for wi, seconds in enumerate(QUALITY_WINDOWS_SECONDS):
            if end_s < seconds:
                continue
            scores = qualities[seconds]
            for c in sorted(columns, key=lambda c: (-scores[c], int(ids[c]))):
                if scores[c] < .30:
                    break
                item = _window(beats[int(ids[c])], end_s, seconds)
                if item:
                    legacy_q[row, wi] = np.r_[item["quality"], scores[c]]
                    break
        for family, feature_indices in enumerate(FAMILY_FEATURE_INDICES):
            seconds = 120 if family == 5 else 300
            if end_s < seconds:
                continue
            scores = qualities[seconds]
            for c in sorted(columns, key=lambda c: (-scores[c], int(ids[c]))):
                score = float(scores[c])
                if score < .30:
                    break
                b, result, audit = beats[int(ids[c])], {}, {}
                if family == 0:
                    long = _hr_window(b, end_s, 300)
                    if long:
                        result[NEW_FEATURE_NAMES[0]] = long["hr"]
                        audit[NEW_FEATURE_NAMES[0]] = long["quality"]
                        short = _hr_window(b, end_s, 60)
                        if short and qualities[60][c] >= .30:
                            result[NEW_FEATURE_NAMES[1]] = short["hr"] - long["hr"]
                            audit[NEW_FEATURE_NAMES[1]] = long["quality"]
                elif family == 1:
                    trend = _hr_trend(b, end_s)
                    if trend is not None:
                        result[NEW_FEATURE_NAMES[2]], audit[NEW_FEATURE_NAMES[2]] = trend
                elif family in (2, 3, 4):
                    # Keep the original strict 80%/90%/5 s PRV criteria.
                    long = _ppi_window(b, end_s, 300)
                    if long:
                        if family == 2:
                            result = {NEW_FEATURE_NAMES[4]: long["sdnn"] * 1000.,
                                      NEW_FEATURE_NAMES[5]: long["cvnn"]}
                            if "rmssd" in long:
                                result[NEW_FEATURE_NAMES[3]] = long["rmssd"] * 1000.
                                result[NEW_FEATURE_NAMES[6]] = long["sd2"] * 1000.
                        elif family == 3:
                            result = interval_spectrum(long["time"], long["rr"])
                        elif qualities[60][c] >= .30:
                            short = _ppi_window(b, end_s, 60)
                            if (short.get("rmssd", 0.) > 1e-8
                                    and long.get("rmssd", 0.) > 1e-8):
                                result[NEW_FEATURE_NAMES[10]] = float(
                                    np.log(short["rmssd"]) - np.log(long["rmssd"]))
                        audit = {name: long["quality"] for name in result}
                else:
                    result, audit = _morph_family(b, end_s)
                for name, value in result.items():
                    j = name_index[name]
                    if not masks[row, j] and np.isfinite(value):
                        values[row, j], masks[row, j], chosen[row, j] = value, 1., ids[c]
                        feature_q[row, j] = np.r_[audit[name], score]
                if masks[row, list(feature_indices)].all():
                    break
    family_channels = np.zeros((len(frame_ends), len(FAMILY_NAMES)), np.int16)
    family_mask = np.zeros_like(family_channels, dtype=np.float32)
    family_q = np.zeros(family_channels.shape + (5,), np.float32)
    for family, indices in enumerate(FAMILY_FEATURE_INDICES):
        for row in range(len(frame_ends)):
            valid_indices = [j for j in indices if masks[row, j]]
            if not valid_indices:
                continue
            channels = np.unique(chosen[row, valid_indices])
            family_channels[row, family] = channels[0] if len(channels) == 1 else -1
            family_mask[row, family] = 1.
            quality_rows = feature_q[row, valid_indices]
            # Conservative summary; feature_quality retains exact scalar audit.
            family_q[row, family] = quality_rows.min(axis=0)
            family_q[row, family, 2] = quality_rows[:, 2].max()
    return {"values": values, "mask": masks, "channels": chosen,
            "quality_metrics": legacy_q, "feature_quality": feature_q,
            "family_channels": family_channels, "family_mask": family_mask,
            "family_quality": family_q, "family_names": np.array(FAMILY_NAMES),
            "family_quality_names": np.array(FAMILY_QUALITY_NAMES),
            "beat_cache": {str(key): value for key, value in beats.items()}}
