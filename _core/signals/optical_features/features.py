"""Causal (past-only) PPG features and label-free same-channel rolling baselines."""
from __future__ import annotations

from collections import defaultdict, deque
import numpy as np
from scipy import ndimage, signal
from scipy.integrate import trapezoid

from .preprocess import channel_quality, filtered_wave
from .schema import BASE_FEATURE_NAMES, FEATURE_LOOKBACK_SECONDS, CANDIDATE_CHANNELS, RED_CHANNELS, IR_CHANNELS


def _integral_band(f: np.ndarray, power: np.ndarray, low: float, high: float) -> float:
    knots = np.r_[low, f[(f > low) & (f < high)], high]
    return float(trapezoid(np.interp(knots, f, power), knots))


def _pulse_metrics(raw: np.ndarray, support: np.ndarray, fs: float, seconds: int,
                   quality_score: float | None = None) -> dict[str, float]:
    """Local pulse estimates, not ECG HRV or calibrated physical blood-flow values."""
    score = channel_quality(raw, fs, support) if quality_score is None else quality_score
    if score < .35:
        return {}
    wave = filtered_wave(raw, fs)
    centered = wave / max(wave.std(), 1e-8)
    # Label-blind polarity convention favors a positively skewed optical pulse.
    if np.mean(centered ** 3) < 0:
        wave = -wave
    y = wave / max(wave.std(), 1e-8)
    freq, power = signal.periodogram(y, fs=fs)
    allowed = (freq >= .6) & (freq <= 3.5)
    dominant = float(freq[np.flatnonzero(allowed)[np.argmax(power[allowed])]])
    peaks, _ = signal.find_peaks(y, distance=max(1, int(.60 * fs / dominant)), prominence=.35)
    if len(peaks) < 17:
        return {}
    precise = peaks.astype(float)
    for j, position in enumerate(peaks):
        if 0 < position < len(y) - 1:
            denominator = y[position - 1] - 2 * y[position] + y[position + 1]
            if abs(denominator) > 1e-10:
                precise[j] += np.clip(.5 * (y[position - 1] - y[position + 1]) / denominator, -.5, .5)
    intervals = np.diff(precise) / fs
    median = ndimage.median_filter(intervals, size=11, mode="nearest")
    bad_prefix = np.r_[0, np.cumsum(~support)]
    valid = (intervals >= 1 / 3.5) & (intervals <= 1 / .6)
    valid &= np.abs(intervals - median) <= .30 * median
    valid &= (bad_prefix[peaks[1:] + 1] - bad_prefix[peaks[:-1]]) == 0
    if valid.sum() < 15 or valid.mean() < .60:
        return {}
    rr = intervals[valid]
    result = {"hr_bpm": float(60. / np.median(rr))}
    differences = np.diff(intervals)[valid[:-1] & valid[1:]]
    if len(differences) >= 15:
        rmssd = float(np.sqrt(np.mean(differences ** 2)))
        sdnn = float(np.std(rr, ddof=1))
        sd1 = float(np.std(differences, ddof=1) / np.sqrt(2))
        sd2 = float(np.sqrt(max(2 * sdnn ** 2 - sd1 ** 2, 0.)))
        result.update(rmssd_ms=rmssd * 1000., rmssd_cycle=rmssd / np.median(rr),
                      sdnn_ms=sdnn * 1000., pnn50=float(np.mean(np.abs(differences) > .05)))
        if sd2 > 1e-8:
            result["sd1_sd2"] = sd1 / sd2
    t = (precise[:-1] + precise[1:]) / (2 * fs)
    kept_t = t[valid]
    if seconds >= 120 and len(rr) >= 40 and valid.mean() >= .70:
        coverage = (kept_t[-1] - kept_t[0]) / seconds
        gap = np.diff(kept_t).max(initial=0.)
        if coverage >= .80 and gap <= 10.:
            uniform_t = np.arange(kept_t[0], kept_t[-1], .25)
            interval_signal = signal.detrend(np.interp(uniform_t, kept_t, rr * 1000.))
            f, p = signal.welch(interval_signal, fs=4., nperseg=min(512, len(interval_signal)))
            lf, hf = _integral_band(f, p, .04, .15), _integral_band(f, p, .15, .40)
            if lf > 1e-8:
                result["hf_lf"] = hf / lf
    troughs = [left + int(np.argmin(wave[left:right])) for left, right in zip(peaks[:-1], peaks[1:])]
    morph = []
    templates = []
    for j in range(1, len(peaks) - 1):
        if not (valid[j - 1] and valid[j]):
            continue
        left, right, peak = troughs[j - 1], troughs[j], int(peaks[j])
        if right - left < 5 or not left < peak < right or not support[left:right + 1].all():
            continue
        beat = wave[left:right + 1] - np.linspace(wave[left], wave[right], right - left + 1)
        position = peak - left
        height = float(beat[position])
        if height <= 1e-8:
            continue
        dc = float(np.median(raw[left:right + 1]))
        rise_fraction = (precise[j] - left) / (right - left)
        low = np.flatnonzero(beat[:position] < height / 2.)
        high = np.flatnonzero(beat[position + 1:] < height / 2.)
        width_fraction = np.nan
        if len(low) and len(high):
            a, b = int(low[-1]), int(position + 1 + high[0])
            rising = a + (height / 2. - beat[a]) / max(beat[a + 1] - beat[a], 1e-12)
            falling = b - 1 + (height / 2. - beat[b - 1]) / min(beat[b] - beat[b - 1], -1e-12)
            width_fraction = (falling - rising) / (right - left)
        morph.append((height / dc if dc > 1. else np.nan, rise_fraction, width_fraction))
        template = np.interp(np.linspace(0, 1, 100), np.linspace(0, 1, len(beat)), beat / height)
        templates.append(np.maximum(template, 0.))
    if len(morph) >= 15:
        morph = np.asarray(morph)
        for column, name in enumerate(("ac_dc", "rise_fraction", "width_fraction")):
            if np.isfinite(morph[:, column]).sum() >= 15:
                result[name] = float(np.nanmedian(morph[:, column]))
        template = np.median(np.stack(templates), axis=0)
        peak = int(np.argmax(template))
        area = float(trapezoid(template))
        if area > 1e-8:
            result["prepeak_area_fraction"] = float(trapezoid(template[:peak + 1]) / area)
    return result


def _best_window(raw: np.ndarray, support: np.ndarray, fs: float,
                 qualities: dict[int, float] | None = None) -> tuple[int, dict]:
    quality = [((channel_quality(raw[:, channel - 1], fs, support[:, channel - 1])
                 if qualities is None else qualities[channel]), channel)
               for channel in CANDIDATE_CHANNELS]
    quality.sort(key=lambda item: (-item[0], item[1]))
    # If the best SQI fails peak-count/coverage requirements, try the next good channel.
    for score, channel in quality:
        if score < .35:
            break
        metrics = _pulse_metrics(raw[:, channel - 1], support[:, channel - 1], fs,
                                 int(round(len(raw) / fs)), quality_score=score)
        if "hr_bpm" in metrics:
            return channel, metrics
    return 0, {}


def _ratio_feature(raw: np.ndarray, support: np.ndarray, fs: float,
                   qualities: dict[int, float] | None = None) -> tuple[float, tuple[int, int]]:
    optical = {}
    for channel in RED_CHANNELS + IR_CHANNELS:
        score = (channel_quality(raw[:, channel - 1], fs, support[:, channel - 1])
                 if qualities is None else qualities[channel])
        dc = float(np.median(raw[:, channel - 1]))
        if score < .40 or dc <= 1.:
            continue
        y = filtered_wave(raw[:, channel - 1], fs)
        ac = float(np.sqrt(np.mean(y ** 2)))
        if ac > 1e-8:
            optical[channel] = (score, y, ac / dc)
    candidates = []
    for red in RED_CHANNELS:
        for infrared in IR_CHANNELS:
            if red not in optical or infrared not in optical:
                continue
            qr, yr, ar = optical[red]
            qi, yi, ai = optical[infrared]
            correlation = abs(float(np.corrcoef(yr, yi)[0, 1]))
            if np.isfinite(correlation) and correlation >= .50 and ai > 1e-12:
                candidates.append((min(qr, qi) * correlation, red, infrared, ar / ai))
    if not candidates:
        return np.nan, (0, 0)
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    _, red, infrared, ratio = candidates[0]
    return float(ratio), (red, infrared)


def extract_features(segment_raw: np.ndarray, segment_support: np.ndarray,
                     end_index: int, fs: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One frame's features use [frame_end-lookback, frame_end), never future or gaps."""
    values = np.zeros(len(BASE_FEATURE_NAMES), dtype=np.float32)
    mask = np.zeros(len(BASE_FEATURE_NAMES), dtype=np.float32)
    channels = np.zeros((len(BASE_FEATURE_NAMES), 2), dtype=np.int16)
    windows = {}
    for seconds in (60, 120):
        size = int(round(seconds * fs))
        if end_index >= size:
            raw = segment_raw[end_index - size:end_index]
            support = segment_support[end_index - size:end_index]
            # These immutable window-local scores serve pulse selection and
            # red/IR pairing; never reuse a score across windows or recordings.
            quality_scores = {channel: channel_quality(raw[:, channel - 1], fs, support[:, channel - 1])
                              for channel in CANDIDATE_CHANNELS}
            windows[seconds] = (*_best_window(raw, support, fs, quality_scores), raw, support, quality_scores)
    for index, (name, seconds) in enumerate(zip(BASE_FEATURE_NAMES, FEATURE_LOOKBACK_SECONDS)):
        if seconds not in windows:
            continue
        channel, metrics, raw, support, quality_scores = windows[seconds]
        if name == "r_red_ir_60":
            value, pair = _ratio_feature(raw, support, fs, quality_scores)
            channels[index] = pair
        else:
            key = name.rsplit("_", 1)[0]
            value = metrics.get(key, np.nan)
            channels[index, 0] = channel
        if np.isfinite(value):
            values[index] = float(value)
            mask[index] = 1.
        else:
            channels[index] = 0
    return values, mask, channels


class RollingBaseline:
    """Past 30min, same-channel history, ≥5min covered; no label-selected baseline."""
    def __init__(self, frame_seconds: float = 30., history_seconds: float = 1800., minimum_seconds: float = 300.):
        self.history = defaultdict(deque)
        self.frame_seconds = float(frame_seconds)
        self.history_ms = history_seconds * 1000.
        self.minimum = max(2, int(np.ceil(minimum_seconds / frame_seconds)))

    def transform(self, end_ms: int, values: np.ndarray, mask: np.ndarray,
                  channels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        normalized, normalized_mask = np.zeros_like(values), np.zeros_like(mask)
        for index, name in enumerate(BASE_FEATURE_NAMES):
            if not mask[index]:
                continue
            key = (index, int(channels[index, 0]), int(channels[index, 1]))
            history = self.history[key]
            # The entire baseline estimate precedes the current feature's input window.
            before_ms = end_ms - FEATURE_LOOKBACK_SECONDS[index] * 1000.
            while history and history[0][0] < before_ms - self.history_ms:
                history.popleft()
            prior = [(t, v) for t, v in history if t <= before_ms]
            if len(prior) >= self.minimum and prior[-1][0] - prior[0][0] >= (self.minimum - 1) * self.frame_seconds * 1000.:
                base = float(np.median([v for _, v in prior]))
                if abs(base) > 1e-6:
                    normalized[index] = (values[index] - base) / abs(base)
                    normalized_mask[index] = 1.
            history.append((int(end_ms), float(values[index])))
        return np.r_[values, normalized].astype(np.float32), np.r_[mask, normalized_mask].astype(np.float32)
