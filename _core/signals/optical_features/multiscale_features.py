
from __future__ import annotations

import numpy as np
from scipy import ndimage, signal, stats
from scipy.integrate import trapezoid

from .preprocess import channel_quality
from .schema import CANDIDATE_CHANNELS

NEW_FEATURE_NAMES = (
    "hr_bpm_300", "hr_short_long_diff", "hr_slope_bpm_min_300",
    "rmssd_ms_300", "sdnn_ms_300", "cvnn_300", "sd2_ms_300",
    "lf_ms2_300", "hf_ms2_300", "hf_lf_300",
    "ln_rmssd_short_long_diff", "amplitude_robust_cv_120",
    "width_fraction_mad_120",
)
QUALITY_METRIC_NAMES = (
    "interval_time_coverage", "time_span_s", "max_untrusted_gap_s",
    "valid_interval_fraction", "sqi",
)
QUALITY_WINDOWS_SECONDS = (300, 120)


def _integral_band(frequency: np.ndarray, psd: np.ndarray,
                   low: float, high: float) -> float:
    knots = np.r_[low, frequency[(frequency > low) & (frequency < high)], high]
    return float(trapezoid(np.interp(knots, frequency, psd), knots))


def interval_spectrum(interval_time_s: np.ndarray,
                      interval_s: np.ndarray) -> dict[str, float]:
    """Full observed-span Hann periodogram of 4 Hz interpolated PPI, in ms^2.

    Caller enforces coverage and gap eligibility. This helper additionally
    rejects non-increasing timestamps and gaps >5 s between usable interval
    centers. There is no 512-point/128-second Welch segmentation cap.
    """
    t = np.asarray(interval_time_s, dtype=float)
    rr = np.asarray(interval_s, dtype=float)
    if (len(t) < 40 or t.shape != rr.shape or not np.isfinite(t).all()
            or not np.isfinite(rr).all() or np.any(np.diff(t) <= 0)
            or t[-1] - t[0] < 270. or np.max(np.diff(t)) > 5.):
        return {}
    grid = np.arange(t[0], t[-1], .25)
    x = signal.detrend(np.interp(grid, t, rr * 1000.), type="linear")
    f, p = signal.periodogram(x, fs=4., window="hann", detrend=False,
                              scaling="density")
    lf, hf = _integral_band(f, p, .04, .15), _integral_band(f, p, .15, .40)
    result = {"lf_ms2_300": lf, "hf_ms2_300": hf}
    if lf > 1e-8:
        result["hf_lf_300"] = hf / lf
    return result


def _empty_beats() -> dict[str, np.ndarray]:
    names = ("peak_index", "peak_time_s", "peak_available_s", "interval_start_s",
             "interval_end_s", "interval_time_s", "interval_s", "interval_valid",
             "interval_available_s", "morph_time_s", "morph_available_s",
             "amplitude", "width_fraction")
    return {name: np.empty(0, dtype=(np.int64 if name == "peak_index" else
                                   bool if name == "interval_valid" else float))
            for name in names}


def extract_channel_beats(raw: np.ndarray, support: np.ndarray,
                          fs: float = 25.) -> dict[str, np.ndarray]:
    """Detect a reusable beat series without borrowing another physical channel.

    Prominence amplitude and normalized half-prominence width describe local
    filtered pulse morphology. They are intentionally not raw AC/DC or pressure.
    The first usable completed 10 s block sets optical polarity. A channel that
    starts flat/missing can recover; no estimate is available before calibration.
    Bad raw samples and the following 2 s filter settling period invalidate PPI.
    """
    x = np.asarray(raw, dtype=float)
    observed = np.asarray(support, dtype=bool) & np.isfinite(x)
    if x.ndim != 1 or observed.shape != x.shape or fs <= 0:
        raise ValueError("raw/support must be aligned 1D arrays and fs positive")
    n = len(x)
    calibration = int(round(10. * fs))
    if n < calibration:
        return _empty_beats()
    indices = np.arange(n)
    last_good = np.maximum.accumulate(np.where(observed, indices, -1))
    # Hold the latest observation; never interpolate from a future sample.
    filled = np.where(last_good >= 0, x[np.maximum(last_good, 0)], 0.)
    filled = np.nan_to_num(filled)
    sos = signal.butter(3, [.5, min(5., .45 * fs)], btype="bandpass", fs=fs,
                        output="sos")
    y, _ = signal.sosfilt(sos, filled, zi=signal.sosfilt_zi(sos) * filled[0])
    calibration_end = None
    # Inspect completed fixed blocks in chronological order. Initial missing or
    # flat data must not permanently disable a channel that later recovers.
    # A later successful block cannot retroactively make an earlier feature
    # available: every peak carries this calibration's decision time below.
    for stop in range(calibration, n + 1, calibration):
        start = stop - calibration
        usable_initial = observed[start:stop].copy()
        usable_initial[:int(round(2. * fs))] = False
        first = y[start:stop][usable_initial]
        if len(first) >= fs * 2 and float(np.std(first)) >= 1e-8:
            calibration_end = stop
            if np.mean((first - np.mean(first)) ** 3) < 0:
                y = -y
            break
    if calibration_end is None:
        return _empty_beats()
    # Trailing exponential energy, not a whole-record amplitude statistic.
    alpha = 1. - np.exp(-1. / (5. * fs))
    energy = signal.lfilter([alpha], [1., -(1. - alpha)], y * y)
    threshold = .35 * np.sqrt(np.maximum(energy, 1e-16))
    half_context = max(2, int(round(1.24 * fs)))
    wlen = 2 * half_context + 1
    peaks, properties = signal.find_peaks(y, prominence=threshold, wlen=wlen)
    if len(peaks) < 3:
        return _empty_beats()
    # Finite-radius suppression avoids find_peaks(distance=...) global ranking.
    radius = max(1, int(np.floor(.25 * fs)))
    local_max = ndimage.maximum_filter1d(y, size=radius * 2 + 1,
                                         mode="constant", cval=-np.inf)
    accepted = ((y[peaks] >= local_max[peaks]) & observed[peaks]
                & (peaks >= int(round(2. * fs)))
                & (peaks + half_context < n))
    peaks = peaks[accepted]
    prominence = properties["prominences"][accepted]
    left_bases = properties["left_bases"][accepted]
    right_bases = properties["right_bases"][accepted]
    if len(peaks) < 3:
        return _empty_beats()
    denominator = y[peaks - 1] - 2. * y[peaks] + y[peaks + 1]
    correction = np.divide(.5 * (y[peaks - 1] - y[peaks + 1]), denominator,
                           out=np.zeros(len(peaks)), where=np.abs(denominator) > 1e-10)
    precise = peaks + np.clip(correction, -.5, .5)
    peak_t = precise / fs
    available = np.maximum((peaks + half_context + 1) / fs, calibration_end / fs)
    intervals = np.diff(peak_t)
    trailing = np.lib.stride_tricks.sliding_window_view(
        np.pad(intervals, (10, 0), mode="edge"), 11)
    reference = np.median(trailing, axis=-1)
    last_bad = np.maximum.accumulate(np.where(~observed, indices, -int(2 * fs) - 1))
    settled = observed & ((indices - last_bad) > int(round(2. * fs)))
    settled[:int(round(2. * fs))] = False
    bad_prefix = np.r_[0, np.cumsum(~settled)]
    interval_valid = ((intervals >= 1. / 3.5) & (intervals <= 1. / .6)
                      & (np.abs(intervals - reference) <= .30 * reference)
                      & ((bad_prefix[peaks[1:] + 1] - bad_prefix[peaks[:-1]]) == 0))
    widths, _, _, _ = signal.peak_widths(
        y, peaks, rel_height=.5,
        prominence_data=(prominence, left_bases, right_bases))
    center = np.arange(1, len(peaks) - 1)
    morph_valid = (interval_valid[:-1] & interval_valid[1:]
                   & np.isfinite(widths[1:-1]) & (prominence[1:-1] > 1e-8))
    center = center[morph_valid]
    period = .5 * (intervals[center - 1] + intervals[center])
    return {
        "peak_index": peaks.astype(np.int64), "peak_time_s": peak_t,
        "peak_available_s": available,
        "interval_start_s": peak_t[:-1], "interval_end_s": peak_t[1:],
        "interval_time_s": .5 * (peak_t[:-1] + peak_t[1:]),
        "interval_s": intervals, "interval_valid": interval_valid,
        "interval_available_s": available[1:],
        "morph_time_s": peak_t[center],
        "morph_available_s": available[center + 1],
        "amplitude": prominence[center],
        "width_fraction": widths[center] / fs / period,
    }


def _window(beats: dict[str, np.ndarray], end_s: float, seconds: int,
            *, minimum_span_fraction: float = .90) -> dict:
    """PPI eligibility, adjacent-only differences, and descriptive diagnostics."""
    left = end_s - seconds
    if left < -1e-7:
        return {}
    starts, ends = beats["interval_start_s"], beats["interval_end_s"]
    # Require both pulse endpoints to lie inside the requested physical window.
    lo = int(np.searchsorted(starts, left, side="left"))
    hi = int(np.searchsorted(beats["interval_available_s"], end_s, side="right"))
    if hi <= lo:
        return {}
    valid = beats["interval_valid"][lo:hi]
    ids = np.arange(lo, hi)[valid]
    if len(ids) < 15:
        return {}
    rr = beats["interval_s"][ids]
    coverage = float(np.sum(rr) / seconds)
    t = beats["interval_time_s"][ids]
    span = float(t[-1] - t[0])
    gaps = np.r_[starts[ids[0]] - left,
                 starts[ids[1:]] - ends[ids[:-1]], end_s - ends[ids[-1]]]
    max_gap = float(np.maximum(gaps, 0.).max(initial=0.))
    if coverage < .80 or span < minimum_span_fraction * seconds or max_gap > 5.:
        return {}
    consecutive = np.diff(ids) == 1
    differences = np.diff(rr)[consecutive]
    sdnn = float(np.std(rr, ddof=1))
    result = {"hr": float(60. / np.median(rr)), "sdnn": sdnn,
              "cvnn": sdnn / float(np.mean(rr)), "rr": rr, "time": t,
              "quality": np.array([coverage, span, max_gap, valid.mean()], dtype=float)}
    if len(differences) >= 15:
        rmssd = float(np.sqrt(np.mean(differences ** 2)))
        sd1_squared = float(np.var(differences, ddof=1) / 2.)
        result["rmssd"] = rmssd
        result["sd2"] = float(np.sqrt(max(2. * sdnn ** 2 - sd1_squared, 0.)))
    return result


def _robust_hr_slope(beats: dict[str, np.ndarray], end_s: float) -> float:
    ends = np.arange(end_s - 240., end_s + .01, 30.)
    points = [(t, _window(beats, t, 60)) for t in ends]
    points = [(t, item["hr"]) for t, item in points if item]
    if len(points) < 5 or points[-1][0] - points[0][0] < 120.:
        return np.nan
    t, hr = np.asarray(points, dtype=float).T
    return float(stats.theilslopes(hr, (t - t[0]) / 60.).slope)


def _morphology(beats: dict[str, np.ndarray], end_s: float) -> dict[str, float]:
    lo = np.searchsorted(beats["morph_time_s"], end_s - 120., side="left")
    hi = np.searchsorted(beats["morph_available_s"], end_s, side="right")
    if hi <= lo or hi - lo < 15:
        return {}
    amplitude = beats["amplitude"][lo:hi]
    widths = beats["width_fraction"][lo:hi]
    median = float(np.median(amplitude))
    result = {}
    if median > 1e-8:
        result["amplitude_robust_cv_120"] = float(
            1.4826 * np.median(np.abs(amplitude - median)) / median)
    widths = widths[np.isfinite(widths) & (widths > 0.) & (widths < 1.)]
    if len(widths) >= 15:
        result["width_fraction_mad_120"] = float(np.median(np.abs(widths - np.median(widths))))
    return result


def extract_multiscale_features(segment_raw: np.ndarray, segment_support: np.ndarray,
                                frame_end_indices: np.ndarray, fs: float = 25.,
                                channel_ids=None, frame_quality=None,
                                quality_frame_end_indices=None) -> dict:
    """Return the 13 aligned new features and serializable per-channel beats.

    Arrays have [sample,physical_channel] layout; frame ends are exclusive
    sample indices on this *one continuous* segment. `frame_quality`, if given,
    contains SQI on the 30 s intervals ending at `quality_frame_end_indices`.
    Only already-ended, fully contained quality intervals are averaged. If a
    complete quality history is unavailable, window-local channel_quality is
    computed instead. No selected-channel concatenation or label is accepted.

    Result: values/mask [F,13], channels [F,13], quality_metrics [F,2,5]
    (300 s,120 s; columns QUALITY_METRIC_NAMES), and beat_cache {str(id): arrays}.
    Values outside mask are zero. Channel 2 is always excluded.
    """
    raw = np.asarray(segment_raw)
    support = np.asarray(segment_support, dtype=bool)
    if raw.ndim != 2 or support.shape != raw.shape or fs <= 0:
        raise ValueError("raw/support must be aligned 2D arrays and fs positive")
    frame_ends = np.asarray(frame_end_indices, dtype=np.int64)
    if frame_ends.ndim != 1 or np.any(frame_ends < 0) or np.any(frame_ends > len(raw)):
        raise ValueError("frame ends must be exclusive valid segment indices")
    if len(frame_ends) > 1 and np.any(np.diff(frame_ends) <= 0):
        raise ValueError("frame ends must be strictly increasing")
    if channel_ids is None:
        channel_ids = np.arange(1, 21) if raw.shape[1] == 20 else CANDIDATE_CHANNELS
    ids = np.asarray(channel_ids, dtype=np.int16)
    if len(ids) != raw.shape[1] or len(np.unique(ids)) != len(ids):
        raise ValueError("physical channel ids must uniquely match raw columns")
    columns = [i for i, channel in enumerate(ids) if int(channel) in CANDIDATE_CHANNELS]
    beats = {int(ids[c]): extract_channel_beats(raw[:, c], support[:, c], fs) for c in columns}
    q = q_ends = q_prefix = None
    if frame_quality is not None:
        q = np.asarray(frame_quality, dtype=float)
        q_ends = np.asarray(frame_ends if quality_frame_end_indices is None else
                            quality_frame_end_indices, dtype=np.int64)
        if (q.shape != (len(q_ends), raw.shape[1]) or np.any(~np.isfinite(q))
                or np.any(np.diff(q_ends) <= 0)):
            raise ValueError("frame_quality must align to increasing quality frame ends")
        q_prefix = np.vstack([np.zeros((1, q.shape[1])), np.cumsum(q, axis=0)])
    frame_size = int(round(30. * fs))

    def quality(end: int, seconds: int) -> np.ndarray:
        left = end - int(round(seconds * fs))
        if q is not None:
            a = int(np.searchsorted(q_ends, left + frame_size, side="left"))
            b = int(np.searchsorted(q_ends, end, side="right"))
            # Require exactly tiled 30 s support; never borrow a later SQI.
            if (b - a == seconds // 30 and b > a
                    and q_ends[a] == left + frame_size and q_ends[b - 1] == end
                    and np.all(np.diff(q_ends[a:b]) == frame_size)):
                return (q_prefix[b] - q_prefix[a]) / (b - a)
        scores = np.zeros(raw.shape[1], dtype=float)
        for c in columns:
            scores[c] = channel_quality(raw[left:end, c], fs, support[left:end, c])
        return scores

    values = np.zeros((len(frame_ends), len(NEW_FEATURE_NAMES)), dtype=np.float32)
    masks = np.zeros_like(values)
    chosen = np.zeros_like(values, dtype=np.int16)
    diagnostics = np.zeros((len(frame_ends), 2, 5), dtype=np.float32)
    name_index = {name: i for i, name in enumerate(NEW_FEATURE_NAMES)}
    for row, end in enumerate(frame_ends):
        end_s = float(end / fs)
        for window_index, seconds in enumerate(QUALITY_WINDOWS_SECONDS):
            if end < int(round(seconds * fs)):
                continue
            scores = quality(int(end), seconds)
            order = sorted(columns, key=lambda c: (-scores[c], int(ids[c])))
            for c in order:
                score = float(scores[c])
                if score < .30:
                    break
                channel = int(ids[c])
                b = beats[channel]
                metrics = _window(b, end_s, seconds)
                if not metrics:
                    continue
                result = {}
                if seconds == 300:
                    result = {"hr_bpm_300": metrics["hr"],
                              "hr_slope_bpm_min_300": _robust_hr_slope(b, end_s),
                              "sdnn_ms_300": metrics["sdnn"] * 1000.,
                              "cvnn_300": metrics["cvnn"]}
                    if "rmssd" in metrics:
                        result["rmssd_ms_300"] = metrics["rmssd"] * 1000.
                        result["sd2_ms_300"] = metrics["sd2"] * 1000.
                    result.update(interval_spectrum(metrics["time"], metrics["rr"]))
                    short = _window(b, end_s, 60)
                    if short:
                        result["hr_short_long_diff"] = short["hr"] - metrics["hr"]
                        if short.get("rmssd", 0.) > 1e-8 and metrics.get("rmssd", 0.) > 1e-8:
                            result["ln_rmssd_short_long_diff"] = float(
                                np.log(short["rmssd"]) - np.log(metrics["rmssd"]))
                else:
                    result = _morphology(b, end_s)
                    if not result:
                        continue
                for name, value in result.items():
                    if np.isfinite(value):
                        j = name_index[name]
                        values[row, j] = value
                        masks[row, j] = 1.
                        chosen[row, j] = channel
                diagnostics[row, window_index] = np.r_[metrics["quality"], score]
                break
    return {"values": values, "mask": masks, "channels": chosen,
            "quality_metrics": diagnostics,
            "beat_cache": {str(key): value for key, value in beats.items()}}
