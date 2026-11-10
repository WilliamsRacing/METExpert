"""Label-free event decoding shared by training and exported Windows bundles."""
import numpy as np


def runs(mask):
    edges = np.diff(np.r_[False, np.asarray(mask, bool), False].astype(np.int8))
    return zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))


def decode(ticks, probability, supported, *, threshold=.5, low_threshold=None,
           min_duration_seconds=30, merge_gap_seconds=60, participant_key='', device='', source='fusion',
           short_min_duration_seconds=None, short_high_fraction=.75, short_high_duration_seconds=60,
           event_high_fraction=0.):
    ticks, probability, supported = np.asarray(ticks), np.asarray(probability), np.asarray(supported, bool)
    if not (ticks.ndim == 1 and ticks.shape == probability.shape == supported.shape):
        raise ValueError('Decoder expects aligned flat arrays')
    if len(ticks) > 1 and np.any(np.diff(ticks) <= 0):
        raise ValueError('Decoder ticks must be unique and increasing')
    low = threshold if low_threshold is None else low_threshold
    if not 0 <= low <= threshold <= 1 or min_duration_seconds < 0 or merge_gap_seconds < 0:
        raise ValueError('Invalid decoder thresholds/durations')
    if (not 0 <= event_high_fraction <= 1 or not 0 <= short_high_fraction <= 1
        or short_high_duration_seconds < 0 or (short_min_duration_seconds is not None
        and short_min_duration_seconds < 0)):
        raise ValueError('Invalid event evidence rules')
    if not np.isfinite(probability[supported]).all():
        raise ValueError('Nonfinite supported probability')
    events = []
    for left, right in runs(supported):
        cuts = [left] + (np.flatnonzero(np.diff(ticks[left:right]) != 1) + left + 1).tolist() + [right]
        for a, b in zip(cuts[:-1], cuts[1:]):
            merged = []
            for start, end in runs(probability[a:b] >= low):
                start, end = start + a, end + a
                if not np.any(probability[start:end] >= threshold):
                    continue
                if merged and ticks[start] - ticks[merged[-1][1]-1] - 1 <= merge_gap_seconds * 15:
                    merged[-1] = (merged[-1][0], end)
                else:
                    merged.append((start, end))
            for start, end in merged:
                duration=(end-start)/15
                high_seconds=float(np.count_nonzero(probability[start:end]>=threshold))/15
                fraction=high_seconds/duration
                accepted=duration>=min_duration_seconds
                if short_min_duration_seconds is not None:
                    accepted |= (duration>=short_min_duration_seconds and fraction>=short_high_fraction
                                 and high_seconds>=short_high_duration_seconds)
                if accepted and fraction>=event_high_fraction:
                    events.append(dict(event_id=f'{participant_key}:{device}:{source}:{int(ticks[start])}',
                        participant_key=participant_key, device=device, source=source,
                        start_ms=int(round(int(ticks[start])*1000/15)),
                        end_ms=int(round((int(ticks[end-1])+1)*1000/15)),
                        score=float(probability[start:end].mean()),
                        continuous_id=f'{participant_key}:{device}:{int(ticks[a])}',
                        _high_fraction=fraction,_high_seconds=high_seconds,_duration_seconds=duration))
    return events
