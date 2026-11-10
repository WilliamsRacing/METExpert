from __future__ import annotations

import math
import numpy as np
from scipy import signal


def modality_runs(callbacks, config):
    """Reconstruct repeated callback clocks with a robust local sample interval."""
    if not callbacks: return []
    clocks = np.array([c for c, _ in callbacks], np.int64)
    counts = np.array([len(v[0]) for _, v in callbacks])
    steps = np.diff(clocks) / counts[:-1]
    quarantined = np.asarray(getattr(callbacks, "quarantined_intervals_ms", []), dtype=np.int64).reshape(-1, 2)
    # Do not infer a slower sample rate across a deliberately removed packet.
    crosses = np.zeros(len(steps), dtype=bool)
    if len(quarantined):
        # Intervals are sorted and merged by load_callbacks. O(N log Q), no N*Q matrix.
        candidate = np.searchsorted(quarantined[:,1], clocks[:-1], side="right")
        safe = np.minimum(candidate, len(quarantined)-1)
        crosses = (candidate < len(quarantined)) & (quarantined[safe,0] < clocks[1:])
        steps[crosses] = np.nan
    fallback = 1000 / float(config.get("fallback_native_hz", 106.383))
    if fallback <= 0: raise ValueError("fallback_native_hz must be positive")
    gap = float(config.get("max_interpolation_gap_ms", 250))
    runs = []; times = []; values = []; previous_end = None

    def flush():
        if times:
            runs.append((np.concatenate(times), np.concatenate(values)))
            times.clear(); values.clear()

    for i, (clock, entry) in enumerate(callbacks):
        near = steps[max(0, i - 4):min(len(steps), i + 5)]
        good = near[(near > 0) & (near <= gap)]
        step = float(np.median(good)) if len(good) else fallback
        # Packet-sized gaps must not be stretched across the missing duration.
        if i < len(steps) and 0 < steps[i] <= min(gap, 2 * step): step = float(steps[i])
        tt = clock + np.arange(len(entry[0])) * step
        value = entry[0]
        if len(quarantined):
            q = int(np.searchsorted(quarantined[:,0], clock, side="right"))
            if q < len(quarantined):
                keep = tt < quarantined[q,0];tt=tt[keep];value=value[keep]
        if not len(tt): continue
        barrier = False
        if previous_end is not None and len(quarantined):
            q = int(np.searchsorted(quarantined[:,1],previous_end,side="right"))
            barrier = q < len(quarantined) and quarantined[q,0] < tt[0]
        if previous_end is not None and (clock - previous_end > gap or barrier): flush()
        if previous_end is not None and clock <= previous_end:
            raise ValueError("Expanded callback times overlap; ambiguous packet timing")
        times.append(tt); values.append(value); previous_end = tt[-1]
    flush()
    return runs


def build_device_imu(acc, gyro, lo, hi, config):
    """15 Hz device axes, retained gravity, no hand transform or AHRS."""
    if (hi - lo) / 1000 > config.get("max_session_seconds", 172800):
        raise MemoryError("Session too long; provide explicit session_manifest")
    cutoff = float(config.get("lowpass_hz", 6))
    if not 0 < cutoff < 7.5: raise ValueError("lowpass_hz must be between 0 and 7.5 Hz")
    scales = [config.get("acc_scale_g"), config.get("gyro_scale_dps")]
    if any(v is None or not np.isfinite(v) or v <= 0 for v in scales):
        raise ValueError("Explicit positive acc_scale_g / gyro_scale_dps required")
    first, last = (int(lo) * 15 + 999) // 1000, (int(hi) * 15 + 999) // 1000
    ticks = np.arange(first, last, dtype=np.int64); n = len(ticks)
    data = np.zeros((n, 6), np.float32); observed = np.zeros((n, 2), bool)
    quality = np.zeros((n, 2), bool); source_run = np.full((n, 2), -1, np.int32)
    target = ticks.astype(np.float64) * (1000 / 15)
    for m, callbacks in enumerate((acc, gyro)):
        for group, (tt, xx) in enumerate(modality_runs(callbacks, config)):
            if len(tt) < 4: continue
            rate = 1000 / float(np.median(np.diff(tt)))
            if rate < 15 or rate <= 2 * cutoff: continue  # Do not upsample an insufficient native rate into apparent 15 Hz support.
            indices = np.flatnonzero((target >= tt[0]) & (target <= tt[-1]))
            if not len(indices): continue
            physical = xx * scales[m]
            sos = signal.butter(4, cutoff, fs=rate, btype="lowpass", output="sos")
            try: filtered = signal.sosfiltfilt(sos, physical, axis=0)
            except ValueError: continue  # Short runs cannot support the specified zero-phase filter.
            data[indices, m * 3:m * 3 + 3] = np.stack([np.interp(target[indices], tt, filtered[:, c]) for c in range(3)], 1)
            observed[indices, m] = True; source_run[indices, m] = group
            limit = config.get("saturation_counts")
            good = np.isfinite(physical).all(1)
            if limit is not None: good &= (np.abs(xx) < float(limit)).all(1)
            nearest = np.searchsorted(tt, target[indices]).clip(0, len(tt) - 1)
            before = np.maximum(nearest - 1, 0)
            quality[indices, m] = good[nearest] & good[before]
    support = observed.all(1) & quality.all(1)
    segment = np.full(n, -1, np.int32)
    if n:
        starts=support & np.r_[True,(~support[:-1]) | np.any(source_run[1:]!=source_run[:-1],axis=1)]
        groups=np.cumsum(starts,dtype=np.int32)-1;segment[support]=groups[support]
    return dict(data=data, ticks=ticks, observed_mask=observed, imu_support=support, continuous_segment=segment)
