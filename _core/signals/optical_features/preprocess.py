"""Read only PPG fields; recover callback time, gaps, and label-blind channel quality."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import numpy as np
from scipy import signal

from .schema import CANDIDATE_CHANNELS


def _text_offset(path: Path) -> int:
    with path.open("rb") as stream:
        first_line = stream.readline(4096)
        if first_line.startswith((b"ACC_TIME\tPPG_TIME\tGYRO_TIME\tPPG1\t", b"PPG_TIME\tPPG1\t")):
            return 0
        stream.seek(0)
        prefix = stream.read(40 * 1024 * 1024)
    for marker in (b"ACC_TIME\tPPG_TIME\tGYRO_TIME\tPPG1\t", b"PPG_TIME\tPPG1\t"):
        offset = prefix.find(marker)
        if offset >= 0:
            return offset
    raise ValueError("No supported PPG TSV header in the first 40 MiB; binary payload not guessed")


def _deduplicate_batches(array: np.ndarray, block_starts: list[int], conflict_policy: str = "error") -> tuple[np.ndarray, dict]:
    """Remove exact repeats; optionally quarantine ALL ambiguous callback versions.

    Quarantine never guesses which version is correct. Its timestamps are
    returned so reconstruction can force gaps even for short missing batches.
    The strict default and non-conflicting retained rows remain unchanged.
    """
    if conflict_policy not in ("error", "quarantine"):
        raise ValueError("Invalid conflicting callback policy")
    starts = np.unique(np.r_[0, np.flatnonzero(np.diff(array[:, 0]) != 0) + 1,
                             np.asarray(block_starts, dtype=np.int64)])
    ends = np.r_[starts[1:], len(array)]
    keep = np.ones(len(starts), bool)
    seen: dict[int, list[int]] = {}
    all_versions: dict[int, list[int]] = {}
    quarantined = set()
    for i, (left, right) in enumerate(zip(starts, ends)):
        current = array[left:right]
        timestamp = int(current[0, 0])
        all_versions.setdefault(timestamp, []).append(i)
        if timestamp in quarantined:
            keep[i] = False
            continue
        previous = seen.setdefault(timestamp, [])
        if not previous:
            previous.append(i)
            continue
        matched = False
        for j in list(previous):
            other = array[starts[j]:ends[j]]
            count = min(len(current), len(other))
            if np.array_equal(current[:count], other[:count]):
                if len(current) <= len(other):
                    keep[i] = False
                else:
                    keep[j] = False
                    previous.remove(j)
                    previous.append(i)
                matched = True
                break
        if not matched:
            if conflict_policy == "error":
                raise ValueError(f"Conflicting PPG callback at timestamp {timestamp}; refusing ambiguous timing")
            quarantined.add(timestamp)
            keep[all_versions[timestamp]] = False
            seen[timestamp] = []
    mask = np.repeat(keep, ends - starts)
    if not mask.any():
        raise ValueError("No unambiguous PPG samples remain after callback quarantine")
    quarantine_callbacks = np.isin(array[starts, 0], sorted(quarantined))
    quarantine_samples = np.repeat(quarantine_callbacks, ends - starts)
    audit = {"exact_duplicate_callbacks_removed": int(((~keep) & ~quarantine_callbacks).sum()),
             "exact_duplicate_samples_removed": int(((~mask) & ~quarantine_samples).sum()),
             "conflicting_callback_policy": conflict_policy,
             "quarantined_callback_groups": len(quarantined),
             "quarantined_callback_versions": int(quarantine_callbacks.sum()),
             "quarantined_samples_removed": int(quarantine_samples.sum()),
             "quarantined_callback_timestamps_ms": sorted(quarantined),
             "quarantine_reason": "All versions of each non-identical same-timestamp callback excluded; no version selected; reconstruction forces a gap"}
    return array[mask], audit


def read_ppg_record(directory: str | Path, conflict_policy: str = "error") -> tuple[np.ndarray, list[dict]]:
    import pandas as pd
    """Return [PPG_TIME, PPG1..20]. No ACC/GYRO value columns are parsed."""
    directory = Path(directory)
    info_path = directory / "info.json"
    declared = []
    if info_path.exists():
        info = json.loads(info_path.read_text(encoding="utf-8-sig"))
        declared = [item["filename"] for item in info.get("files", [])
                    if Path(item.get("filename", "")).suffix.lower() in (".txt", ".dat", ".tsv")]
    discovered = sorted(p.name for p in directory.iterdir()
                        if p.is_file() and p.suffix.lower() in (".txt", ".dat", ".tsv") and not p.name.startswith("._"))
    names = declared or discovered
    if declared and sorted(declared) != discovered:
        raise ValueError("info.json file list and discovered sensor files differ")
    if not names:
        raise ValueError("No supported TXT/DAT/TSV exports")
    pieces, audits, block_starts = [], [], []
    count = 0
    columns = ["PPG_TIME"] + [f"PPG{i}" for i in range(1, 21)]
    for name in names:
        path = directory / name
        if path.resolve().parent != directory.resolve():
            raise ValueError("Unsafe info.json member path")
        offset = _text_offset(path)
        audit = {"filename": name, "bytes": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns,
                 "binary_prefix_bytes": offset, "rows": 0, "valid_ppg_rows": 0,
                 "invalid_time_rows": 0, "repeated_headers": 0, "parsed_columns": columns}
        block_starts.append(count)
        with path.open("rb") as stream:
            stream.seek(offset)
            # String chunks allow explicit repeated-export headers, with numeric validation below.
            reader = pd.read_csv(stream, sep="\t", usecols=columns, dtype=str,
                                 chunksize=100000, on_bad_lines="error", keep_default_na=False)
            for frame in reader:
                repeated = frame.PPG_TIME.eq("PPG_TIME").to_numpy()
                cuts = np.r_[0, np.flatnonzero(repeated), len(frame)]
                for left, right in zip(cuts[:-1], cuts[1:]):
                    if left < len(frame) and repeated[left]:
                        audit["repeated_headers"] += 1
                        block_starts.append(count)
                        left += 1
                    part = frame.iloc[left:right]
                    if part.empty:
                        continue
                    values = part[columns].apply(pd.to_numeric, errors="raise").to_numpy(dtype=np.int64)
                    good = values[:, 0] > 0
                    audit["rows"] += len(values)
                    audit["valid_ppg_rows"] += int(good.sum())
                    audit["invalid_time_rows"] += int((~good).sum())
                    if good.any():
                        pieces.append(values[good])
                        count += int(good.sum())
        audits.append(audit)
    if not pieces:
        raise ValueError("No positive PPG timestamps")
    data = np.concatenate(pieces)
    data, dedup = _deduplicate_batches(data, [v for v in block_starts if v < len(data)], conflict_policy=conflict_policy)
    audits[-1].update(dedup)
    return data, audits


def _rate_change_boundaries(rates: np.ndarray) -> list[int]:
    """Detect only sustained rate changes, not an isolated delayed callback.

    Boundary ``j`` starts the callback whose outgoing interval is ``rates[j]``.
    Two 9-callback windows need seven consistent estimates on either side and
    a >1.5-fold median difference. Adjacent candidate positions describe one
    transition; locate it at the largest immediate rate jump in that band.
    """
    width, required = 9, 7
    rates = np.asarray(rates, float)
    if len(rates) < width * 2:
        return []
    windows = np.lib.stride_tricks.sliding_window_view(rates, width)
    valid = np.isfinite(windows) & (windows >= 12.) & (windows <= 105.)
    masked = np.where(valid, windows, np.nan)
    # Avoid all-NaN warnings for sparse/bad windows without accepting them.
    count = valid.sum(axis=1)
    median = np.zeros(len(windows), float)
    good = count >= required
    median[good] = np.nanmedian(masked[good], axis=1)
    coherent = good & (np.sum(valid & (np.abs(windows - median[:, None]) <= .20 * median[:, None]), axis=1) >= required)
    n_boundaries = len(rates) - 2 * width + 1
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = median[width:width + n_boundaries] / median[:n_boundaries]
    candidate_mask = (coherent[:n_boundaries] & coherent[width:width + n_boundaries]
                      & ((ratio > 1.5) | (ratio < (1. / 1.5))))
    candidates = (np.flatnonzero(candidate_mask) + width).tolist()
    if not candidates:
        return []
    result = []
    cuts = np.r_[0, np.flatnonzero(np.diff(candidates) > 1) + 1, len(candidates)]
    for left, right in zip(cuts[:-1], cuts[1:]):
        positions = candidates[left:right]
        jump = [abs(np.log(rates[j] / rates[j - 1]))
                if np.isfinite(rates[j]) and np.isfinite(rates[j - 1]) and rates[j] > 0 and rates[j - 1] > 0 else -np.inf
                for j in positions]
        result.append(positions[int(np.argmax(jump))])
    return result


def reconstruct_time(timestamps: np.ndarray, quarantined_callback_times_ms=None) -> tuple[float, list[tuple[int, int]], np.ndarray, dict]:
    """Rebuild each continuous sampling-rate regime; keep callback samples.

    The first returned rate is only a recording overview. ``segment_fs_hz`` in
    the audit is aligned with returned safe segments and must be used when
    deciding their observed durations. Nothing here depends on meal labels.
    """
    timestamps = np.asarray(timestamps, dtype=np.int64)
    if len(timestamps) < 3:
        raise ValueError("Too few timestamps")
    starts = np.r_[0, np.flatnonzero(np.diff(timestamps) != 0) + 1]
    ends = np.r_[starts[1:], len(timestamps)]
    counts = ends - starts
    delta = np.diff(timestamps[starts]) / 1000.0
    stable = (delta >= .005) & (delta <= 2.0)
    rates = counts[:-1][stable] / delta[stable]
    if len(rates) < 2:
        raise ValueError("Cannot infer sampling rate from callback timestamps")
    fs = float(np.median(rates))
    if 100 < fs <= 105:
        fs = 100.  # Millisecond callback quantization near a nominal 100 Hz.
    if not 12 <= fs <= 100:
        raise ValueError(f"Unsupported inferred PPG sampling rate: {fs:.4f} Hz")
    # Acquisition gaps must not allow the dominant rate in another period to
    # shrink an entire later period into one-callback fragments.
    quarantine_times = np.unique(np.asarray(quarantined_callback_times_ms if quarantined_callback_times_ms is not None else [], dtype=np.int64))
    callback_times = timestamps[starts]
    quarantine_cuts = (np.searchsorted(quarantine_times, callback_times[1:])
                       > np.searchsorted(quarantine_times, callback_times[:-1]))
    rough_boundaries = np.r_[0, np.flatnonzero((delta <= 0) | (delta > 2.) | quarantine_cuts) + 1, len(starts)]
    time_ms = np.empty(len(timestamps), float)
    segments, local_rates = [], []
    rate_changes = []
    for rough_left, rough_right in zip(rough_boundaries[:-1], rough_boundaries[1:]):
        local_delta = delta[rough_left:rough_right - 1]
        with np.errstate(divide="ignore", invalid="ignore"):
            local_rate = counts[rough_left:rough_right - 1] / local_delta
        changes = _rate_change_boundaries(local_rate)
        regime_boundaries = np.r_[rough_left, np.asarray(changes, dtype=int) + rough_left, rough_right]
        rate_changes.extend(int(timestamps[starts[rough_left + j]]) for j in changes)
        for left, right in zip(regime_boundaries[:-1], regime_boundaries[1:]):
            estimates = counts[left:right - 1] / delta[left:right - 1]
            eligible = np.isfinite(estimates) & (estimates >= 12.) & (estimates <= 105.)
            # Extremely short pieces cannot support frames. Keep their samples
            # with the overview rate rather than pretending to infer a rate.
            segment_fs = min(100., float(np.median(estimates[eligible]))) if eligible.sum() >= 2 else fs
            expected = counts[left:right - 1] / segment_fs
            local_split = np.abs(delta[left:right - 1] - expected) > np.maximum(.20, .35 * expected)
            boundaries = np.r_[left, np.flatnonzero(local_split) + left + 1, right]
            for first, stop in zip(boundaries[:-1], boundaries[1:]):
                sample_left, sample_right = int(starts[first]), int(ends[stop - 1])
                local_counts = counts[first:stop]
                time_ms[sample_left:sample_right] = np.repeat(timestamps[starts[first:stop]].astype(float), local_counts)
                time_ms[sample_left:sample_right] += (
                    np.arange(sample_left, sample_right) - np.repeat(starts[first:stop], local_counts)
                ) * (1000. / segment_fs)
                segments.append((sample_left, sample_right))
                local_rates.append(segment_fs)
    # Small timestamp jitter must not create overlapping reconstructed samples.
    safe_segments, safe_rates = [], []
    for (left, right), segment_fs in zip(segments, local_rates):
        incompatible = np.flatnonzero(np.diff(time_ms[left:right]) <= 0) + left + 1
        points = np.r_[left, incompatible, right]
        safe_segments.extend((int(a), int(b)) for a, b in zip(points[:-1], points[1:]))
        safe_rates.extend([segment_fs] * (len(points) - 1))
    audit = {"anchor_convention": "callback timestamp approximates first sample", "inferred_fs": fs,
             "batch_count": len(starts), "modal_batch_size": int(np.bincount(counts).argmax()),
             "fs_p10": float(np.quantile(rates, .1)), "fs_p90": float(np.quantile(rates, .9)),
             "backward_callbacks": int((delta < 0).sum()), "continuous_segments": len(safe_segments),
             "segment_fs_hz": safe_rates, "sustained_rate_change_callback_ms": rate_changes,
             "quarantined_callback_timestamps_ms": quarantine_times.tolist(),
             "forced_quarantine_gap_splits": int(quarantine_cuts.sum()),
             "sampling_rate_policy": "Split >2s/backward acquisition gaps, then sustained >1.5-fold changes using 9-callback/7-consistent windows; estimate each regime separately; overview fs is not a segment duration rate",
             "precision_note": "Within-callback timing is reconstructed; subsecond boundary precision is not guaranteed"}
    return fs, safe_segments, time_ms, audit


def resample_segment(times_ms: np.ndarray, data: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Uniform segment grid, never bridge a segment gap; mask indicates observed support."""
    step = 1000.0 / fs
    grid = np.arange(times_ms[0], times_ms[-1] + .1 * step, step)
    values = np.empty((len(grid), 20), dtype=float)
    support = np.ones_like(values, dtype=bool)
    for channel in range(20):
        x = data[:, channel].astype(float)
        good = np.isfinite(x) & (x != 0)
        values[:, channel] = np.interp(grid, times_ms, np.nan_to_num(x))
        # Bad observations cannot be silently filled by interpolation.
        support[:, channel] = np.interp(grid, times_ms, good.astype(float)) >= .999
    return grid, values, support


def filtered_wave(raw: np.ndarray, fs: float) -> np.ndarray:
    raw = np.asarray(raw, float)
    if len(raw) < max(30, int(fs * 2)):
        return np.zeros_like(raw)
    sos = signal.butter(3, [.5, min(5., .45 * fs)], btype="bandpass", fs=fs, output="sos")
    return signal.sosfiltfilt(sos, signal.detrend(raw))


def robust_normalize(filtered: np.ndarray) -> np.ndarray:
    center = np.median(filtered)
    scale = 1.4826 * np.median(np.abs(filtered - center))
    if scale <= 1e-8:
        return np.zeros_like(filtered, dtype=np.float32)
    return np.clip((filtered - center) / scale, -8., 8.).astype(np.float32)


def channel_quality(raw: np.ndarray, fs: float, mask: np.ndarray | None = None) -> float:
    """PPG-only engineering SQI: ACF, beat template, spectral concentration, entropy."""
    raw = np.asarray(raw, float)
    if mask is None:
        mask = np.isfinite(raw) & (raw != 0)
    if len(raw) < 4 * fs or np.mean(mask) < .85 or np.std(raw) < 1e-6:
        return 0.
    x = np.nan_to_num(raw)
    flat = np.mean(np.diff(x) == 0)
    if np.mean(x == 0) > .05 or flat > .20:
        return 0.
    extrema = (x == np.min(x)) | (x == np.max(x))
    if np.mean(extrema[:-2] & extrema[1:-1] & extrema[2:]) > .02:
        return 0.
    y = filtered_wave(x, fs)
    y = (y - y.mean()) / max(y.std(), 1e-8)
    f, power = signal.periodogram(y, fs=fs, window="hann")
    domain = (f >= .5) & (f <= 5.)
    cardiac = (f >= .6) & (f <= 3.5)
    if not cardiac.any() or power[domain].sum() <= 1e-12:
        return 0.
    frequency = float(f[np.flatnonzero(cardiac)[np.argmax(power[cardiac])]])
    harmonics = domain & ((abs(f - frequency) <= .18) | (abs(f - 2 * frequency) <= .18) | (abs(f - 3 * frequency) <= .18))
    concentration = float(power[harmonics].sum() / power[domain].sum())
    prob = power[domain] / power[domain].sum()
    entropy = -float(np.sum(prob * np.log(np.maximum(prob, 1e-20)))) / np.log(len(prob))
    ac = signal.fftconvolve(y, y[::-1], mode="full")[len(y) - 1:]
    ac /= max(ac[0], 1e-12)
    lo, hi = max(2, int(fs / 3.5)), min(len(y) - 2, int(fs / .6))
    local, _ = signal.find_peaks(ac[lo:hi + 1])
    if len(local):
        period = lo + int(local[np.argmax(ac[lo + local])])
        acf = float(np.clip(ac[period], 0., 1.))
    else:
        period, acf = fs / frequency, 0.
    peaks, _ = signal.find_peaks(y, distance=max(1, int(.65 * period)), prominence=.35)
    # Resample every eligible beat together; numerical thresholds and ordering
    # match the original per-beat interpolation/normalization calculation.
    left, right = peaks[:-1], peaks[1:]
    seconds = (right - left) / fs
    good = (seconds >= 1 / 3.5) & (seconds <= 1 / .6)
    left, right = left[good], right[good]
    positions = left[:, None] + np.linspace(0, 1, 40)[None, :] * (right - left)[:, None]
    beats = np.interp(positions.ravel(), np.arange(len(y)), y).reshape(-1, 40)
    beats -= beats.mean(axis=1, keepdims=True)
    norm = np.linalg.norm(beats, axis=1)
    beats = beats[norm > 1e-8] / norm[norm > 1e-8, None]
    template = 0.
    if len(beats) >= 4:
        beats = np.stack(beats)
        others = beats.sum(axis=0)[None] - beats
        template = float(np.clip(np.median(np.sum(beats * others, axis=1) / np.maximum(np.linalg.norm(others, axis=1), 1e-12)), 0., 1.))
    ff, pp = signal.welch(signal.detrend(x), fs=fs, nperseg=min(len(x), int(4 * fs)))
    relative = float(pp[(ff >= .5) & (ff <= 5)].sum() / max(pp[(ff >= .1) & (ff <= min(10, .49 * fs))].sum(), 1e-20))
    return float(np.clip(.30 * acf + .30 * template + .20 * concentration + .15 * relative + .05 * (1 - entropy), 0., 1.))


def top_channels(raw: np.ndarray, mask: np.ndarray, fs: float, top_k: int = 3, min_sqi: float = .30) -> tuple[np.ndarray, np.ndarray]:
    """Input [sample,20]; deterministic ties by channel number, no labels or IMU."""
    scores = np.array([channel_quality(raw[:, c - 1], fs, mask[:, c - 1]) for c in CANDIDATE_CHANNELS])
    order = np.argsort(-scores, kind="stable")
    chosen, qualities = np.zeros(top_k, np.int16), np.zeros(top_k, np.float32)
    for slot, index in enumerate(order[:top_k]):
        if scores[index] >= min_sqi:
            chosen[slot] = CANDIDATE_CHANNELS[index]
            qualities[slot] = scores[index]
    return chosen, qualities
