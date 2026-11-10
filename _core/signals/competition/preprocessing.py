from __future__ import annotations
import numpy as np
from signals.bme.signals import expand_packet_timestamps, _filter_and_resample, remove_gravity, device_to_shm, continuous_runs
from signals.preprocessing import decode_shm, preprocess
from signals.optical_features.preprocess import reconstruct_time, resample_segment, channel_quality, filtered_wave, robust_normalize
from signals.optical_features.features import extract_features, RollingBaseline
from signals.optical_features.hr_reference import compute_hr_reference
from signals.optical_features.multiscale_features_v6 import extract_multiscale_features
from signals.optical_features.routing import route_sequence
from signals.optical_features.schema import CANDIDATE_CHANNELS
from signals.optical_features.time_frequency import waveform_spectrogram
from signals.optical_features.features_v8 import extract_features_v8
def _resample_imu(data, scale):
    if data is None or len(data[0]) < 3:
        return []
    clocks, values, rows, owners = data
    callback_starts = np.r_[0, np.flatnonzero(np.diff(clocks) != 0) + 1]
    callback_counts = np.diff(np.r_[callback_starts, len(clocks)])
    callback_delta = np.diff(clocks[callback_starts]).astype(np.float64)
    steps = callback_delta / callback_counts[:-1]
    boundaries = [0]
    for j, delta in enumerate(callback_delta):
        nearby = steps[max(0, j - 4):min(len(steps), j + 5)]
        good = nearby[(nearby > 0) & np.isfinite(nearby)]
        expected = (np.median(good) if len(good) else 1000 / 106.383) * callback_counts[j]
        if delta <= 0 or delta > expected + 250:
            boundaries.append(int(callback_starts[j + 1]))
    boundaries.append(len(clocks))
    times = np.empty(len(clocks), np.float64)
    for a, b in zip(boundaries[:-1], boundaries[1:]):
        times[a:b] = expand_packet_timestamps(clocks[a:b])
    valid = np.isfinite(times)
    times, values, rows, owners = times[valid], values[valid], rows[valid], owners[valid]
    splits = np.r_[0, np.flatnonzero((np.diff(times) <= 0) | (np.diff(times) > 250)) + 1, len(times)]
    result = []
    for a, b in zip(splits[:-1], splits[1:]):
        block = np.tile(values[a:b].astype(np.float64) * scale, (1, 2))
        sampled = _filter_and_resample(times[a:b], block)
        if sampled is None:
            continue
        ticks, signal, _ = sampled
        nearest = np.searchsorted(times[a:b], ticks.astype(float) * 1000 / 15).clip(0, b - a - 1) + a
        result.append((ticks, signal[:, :3], rows[nearest], owners[nearest]))
    return result


def build_imu(acc, gyro, worn, direction, start_ms, end_ms, orientation_changes=(), *, acc_scale=None, gyro_scale=None):
    from signals.bme.signals import ACC_G_PER_COUNT, GYRO_DEG_PER_COUNT
    first, last = (start_ms * 15 + 999) // 1000, (end_ms * 15 + 999) // 1000
    ticks = np.arange(first, last, dtype=np.int64)
    n = len(ticks)
    values, observations = np.zeros((n, 6), np.float32), np.zeros((n, 2), bool)
    source_rows, source_owners = np.full((n, 2), -1, np.int64), np.full((n, 2), -1, np.int32)
    for j, (data, scale) in enumerate(((acc, ACC_G_PER_COUNT if acc_scale is None else acc_scale), (gyro, GYRO_DEG_PER_COUNT if gyro_scale is None else gyro_scale))):
        for tt, signal, row, owner in _resample_imu(data, scale):
            selected = (tt >= first) & (tt < last)
            index = tt[selected] - first
            values[index, j * 3:j * 3 + 3] = signal[selected]
            observations[index, j] = True
            source_rows[index, j], source_owners[index, j] = row[selected], owner[selected]
    observed = observations.all(-1)
    states, processed = np.zeros(n, np.uint8), np.zeros_like(values)
    segments = np.full(n, -1, np.int32)
    quality = np.zeros(n, bool)
    # Only a real hand/direction change resets orientation state. Repeated
    # profile uploads must not manufacture a signal boundary.
    changes, previous = [], (worn, direction)
    for change in sorted(orientation_changes):
        if tuple(change[1:3]) != previous:
            changes.append(change)
            previous = tuple(change[1:3])
    orientation_cuts = [0] + [int(np.searchsorted(ticks, (int(t) * 15 + 999) // 1000)) for t, *_ in changes] + [n]
    orientation_values = [(worn, direction)] + [tuple(c[1:3]) for c in changes]
    group = 0
    for (left, right), (current_hand, current_direction) in zip(zip(orientation_cuts[:-1], orientation_cuts[1:]), orientation_values):
        if right <= left:
            continue
        known = current_hand in {"left", "right"} and current_direction in {"standard", "rot180"}
        rounded = None
        if known:
            # Reuse the original AHRS grouping: gaps <=1s retain state while
            # long gaps reset it. Smoothing/TCN runs still stop at every gap.
            physical = remove_gravity(values[left:right], observed[left:right])
            rounded = device_to_shm(physical, current_hand, current_direction)
        previous_end, seen = None, 0
        for local_a, local_b in continuous_runs(observed[left:right]):
            a, b = local_a + left, local_b + left
            segments[a:b] = group
            group += 1
            if not known:
                continue
            if previous_end is None or a - previous_end > 15:
                seen = 0
            output, state = preprocess(decode_shm(rounded[local_a:local_b].tobytes(), source="compact-memory-f32"))
            processed[a:b], states[a:b] = output.astype(np.float32), state
            quality[a:b] = np.isfinite(output).all(-1) & (state != 0)
            quality[a:min(b, a + max(45 - seen, 0))] = False
            seen += b - a
            previous_end = b
    return {"ticks": ticks, "data": processed, "state": states, "observed": observed,
            "quality": quality, "supported": quality,
            "segment": segments, "source_row": source_rows, "source_owner": source_owners,
            "modality_observed": observations}


class PPGCacheMismatch(ValueError):
    """The cached PPG cannot be combined with the current physical timeline."""


def build_ppg(data, start_ms, end_ms, audit, *, base_cache=None):
    names_shapes = {"raw": (3, 750), "raw_mask": (3, 750), "quality": (3,), "channel_ids": (3,),
                    "features": (68,), "feature_mask": (68,), "routing_context": (2,),
                    "spectrogram": (3, 37, 23), "spectrogram_mask": (3, 23),
                    "source_row": (750,), "source_owner": (750,)}
    starts = np.arange((start_ms // 30000) * 30000, ((end_ms + 29999) // 30000) * 30000, 30000, dtype=np.int64)
    if base_cache is None:
        result = {name: np.zeros((len(starts), *shape), np.float32) for name, shape in names_shapes.items()}
        result["channel_ids"] = result["channel_ids"].astype(np.int16)
        result["source_row"] = np.full((len(starts), 750), -1, np.int64)
        result["source_owner"] = np.full((len(starts), 750), -1, np.int32)
        result.update(frame_start_ms=starts, segment=np.full(len(starts), -1, np.int32),
                      observed_fraction=np.zeros(len(starts), np.float32), frame_valid=np.zeros(len(starts), bool),
                      observed=np.zeros((len(starts), 450), bool))
    else:
        if not np.array_equal(starts,base_cache["frame_start_ms"]) or base_cache["features"].shape!=(len(starts),40):
            raise PPGCacheMismatch("V1 PPG cache must have the same frame timeline and original 40-column schema")
        result=dict(base_cache,features=np.zeros((len(starts),68),np.float32),feature_mask=np.zeros((len(starts),68),np.float32))
        result["features"][:,:27]=base_cache["features"][:,:27]
        result["feature_mask"][:,:27]=base_cache["feature_mask"][:,:27]
    if data is None or len(data[0]) < 3:
        if base_cache is not None and base_cache["frame_valid"].any():raise PPGCacheMismatch("V1 PPG frames require their original raw PPG for the V8 supplement")
        return result
    clocks, optical, source_rows, source_owners = data
    try:
        quarantine = [t for item in audit for t in item.get("callbacks", {}).get("quarantined_callback_timestamps_ms", [])]
        _, segments, times, clock_audit = reconstruct_time(clocks, quarantine)
    except ValueError as exc:
        if base_cache is not None and base_cache["frame_valid"].any():raise PPGCacheMismatch("Raw PPG clock differs from the reusable V1 session") from exc
        audit.append({"modality": "ppg", "unsupported_clock": str(exc)})
        return result
    audit.append({"modality": "ppg", "time_reconstruction": clock_audit})
    observed=np.zeros((len(starts),450),bool)
    expected_segment=np.full(len(starts),-1,np.int32)
    for gid, (a, b) in enumerate(segments):
        if b - a < 3:
            continue
        grid, raw, support = resample_segment(times[a:b], optical[a:b, :20], 25.)
        point_ticks = starts[:, None] * 15 // 1000 + np.arange(450, dtype=np.int64)[None, :]
        point_ms = point_ticks.astype(np.float64) * 1000 / 15
        observed |= (point_ms >= grid[0]) & (point_ms < grid[-1] + 40)
        if base_cache is None:
            result["observed"] = observed
            result["observed_fraction"] = observed.mean(-1).astype(np.float32)
        eligible = np.flatnonzero((starts >= grid[0]) & (starts + 30000 <= grid[-1] + 40.1))
        if not len(eligible):
            continue
        expected_segment[eligible]=gid
        frame_starts = starts[eligible]
        # Original V5 performs a second interpolation onto an epoch-aligned
        # 40ms grid, including its full pre-frame history.
        aligned_first = float(frame_starts[0] - np.floor((frame_starts[0] - grid[0]) / 40.) * 40.)
        aligned = np.arange(aligned_first, float(frame_starts[-1] + 30000), 40.)
        raw = np.stack([np.interp(aligned, grid, raw[:, c]) for c in range(20)], 1)
        support = np.stack([np.interp(aligned, grid, support[:, c].astype(float)) >= .999 for c in range(20)], 1)
        grid = aligned
        ends = np.rint((frame_starts + 30000 - aligned_first) / 40.).astype(np.int64)
        quality = np.array([[channel_quality(raw[e - 750:e, c - 1], 25, support[e - 750:e, c - 1])
                             for c in CANDIDATE_CHANNELS] for e in ends], np.float32)
        if base_cache is None:
            selected, selected_quality, context = route_sequence(quality, CANDIDATE_CHANNELS, frame_starts)
            baseline = RollingBaseline()
            features, masks = [], []
            for e, t in zip(ends, frame_starts + 30000):
                v, m, channels = extract_features(raw, support, int(e), 25.)
                v, m = baseline.transform(int(t), v, m, channels)
                features.append(v); masks.append(m)
            features, masks = np.array(features), np.array(masks)
            hr, hr_mask = compute_hr_reference(frame_starts + 30000, features[:, 0], masks[:, 0])
            prefix,prefix_mask=np.column_stack((features,hr)),np.column_stack((masks,hr_mask))
        else:
            prefix,prefix_mask=base_cache["features"][eligible,:27],base_cache["feature_mask"][eligible,:27]
        candidate = np.asarray(CANDIDATE_CHANNELS) - 1
        multi = extract_multiscale_features(raw[:, candidate], support[:, candidate], ends, 25.,
                                             channel_ids=CANDIDATE_CHANNELS, frame_quality=quality,
                                             quality_frame_end_indices=ends)
        extra = extract_features_v8(raw[:, candidate], support[:, candidate], ends, 25.,
                                    channel_ids=CANDIDATE_CHANNELS, frame_quality=quality,
                                    quality_frame_end_indices=ends, beat_cache=multi["beat_cache"])
        result["features"][eligible] = np.column_stack((prefix, multi["values"], extra["values"]))
        result["feature_mask"][eligible] = np.column_stack((prefix_mask, multi["mask"], extra["mask"]))
        if base_cache is not None:continue
        result["channel_ids"][eligible], result["quality"][eligible] = selected, selected_quality
        result["routing_context"][eligible] = context
        result["segment"][eligible] = gid
        result["frame_valid"][eligible] = True
        for index, e, ids in zip(eligible, ends, selected):
            nearest = np.searchsorted(times[a:b], grid[e - 750:e]).clip(0, b - a - 1) + a
            result["source_row"][index], result["source_owner"][index] = source_rows[nearest], source_owners[nearest]
            for k, channel in enumerate(ids):
                if channel:
                    wave = raw[e - 750:e, channel - 1]
                    mask = support[e - 750:e, channel - 1]
                    result["raw"][index, k] = robust_normalize(filtered_wave(wave, 25.)) * mask
                    result["raw_mask"][index, k] = mask
            spec, smask = waveform_spectrogram(result["raw"][index], result["raw_mask"][index])
            result["spectrogram"][index], result["spectrogram_mask"][index] = spec, smask
    if base_cache is not None:
        segment_frames=int(np.count_nonzero(expected_segment!=base_cache["segment"]))
        observation_points=int(np.count_nonzero(observed!=base_cache["observed"]))
        validity_frames=int(np.count_nonzero((expected_segment>=0)!=base_cache["frame_valid"]))
        if segment_frames or observation_points or validity_frames:
            raise PPGCacheMismatch("V1 cache and raw PPG have different physical segments/observation masks: "
                f"segment_frames={segment_frames}, observation_points={observation_points}, validity_frames={validity_frames}")
    return result



def build_standard_imu(acc, gyro, start_ms, end_ms, *, orientation="standard", acc_scale=None, gyro_scale=None):
    """Normalize units and use the canonical device axes without wear-hand metadata.

    In the extracted coordinate helper, 'right' selects the base device-to-model
    transform (no additional left-hand reflection). It is a coordinate convention,
    not a claim about the participant's wearing hand.
    """
    return build_imu(acc, gyro, "right", orientation, start_ms, end_ms,
                     acc_scale=acc_scale, gyro_scale=gyro_scale)
