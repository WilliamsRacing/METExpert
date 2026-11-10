
from __future__ import annotations

import math
import numpy as np
from scipy import signal

from .multiscale_features import extract_channel_beats, _integral_band
from .multiscale_features_v6 import _validate_beats
from .preprocess import channel_quality
from .schema import CANDIDATE_CHANNELS

NEW_FEATURE_NAMES = (
    'prv_power_004_040_300', 'hf_normalized_300',
    'prv_spectral_entropy_300', 'lf_peak_hz_300', 'hf_peak_hz_300',
    'prv_spectral_centroid_300', 'ppi_mad_ms_300', 'ppi_iqr_ms_300',
    'ppi_median_abs_diff_ms_300', 'cvsd_300', 'pnn50_300',
    'ppi_permutation_entropy_300', 'sd1_ms_300', 'poincare_sd1_sd2_300',
    'poincare_area_ms2_300', 'guzik_index_300', 'porta_index_300',
    'poincare_lag5_ratio_300', 'ppg_spectral_entropy_60',
    'ppg_spectral_centroid_hz_60', 'ppg_spectral_bandwidth_hz_60',
    'ppg_spectral_flatness_60', 'ppg_h2_h1_60', 'ppg_h3_h1_60',
    'ppg_fundamental_fraction_60', 'amplitude_median_abs_diff_norm_120',
    'width_fraction_cv_120', 'template_correlation_iqr_120',
)
FAMILY_NAMES = ('frequency_prv', 'time_variability', 'poincare',
                'waveform_spectrum', 'morphology')
FAMILY_INDICES = (tuple(range(0, 6)), tuple(range(6, 12)),
                  tuple(range(12, 18)), tuple(range(18, 25)), tuple(range(25, 28)))
FAMILY_SECONDS = (300, 300, 300, 60, 120)
QUALITY_NAMES = ('support_fraction', 'time_span_s', 'max_untrusted_gap_s',
                 'valid_observation_fraction', 'sqi')
EXTRACTION_SPEC = {
    'version': 'v8-engineering-features-1', 'n_features': 28,
    'frequency_prv': {'seconds': 300, 'min_span_s': 270,
                      'min_coverage': .8, 'max_gap_s': 5., 'resample_hz': 4.},
    'permutation_entropy': {'dimension': 3, 'lag': 1, 'min_patterns': 100,
                            'max_tie_fraction': .2, 'tie_atol_s': 1e-10},
    'poincare': {'ddof': 1, 'guzik': 'squared_positive_differences_percent',
                 'porta': 'negative_nonzero_differences_percent', 'lag': 5},
    'waveform': {'seconds': 60, 'band_hz': [.5, 5.],
                 'min_observed_fraction': .95, 'max_unobserved_gap_s': .2,
                 'harmonic_half_width': 'max(0.10,0.10*f0)',
                 'flatness_floor_fraction_of_mean': 1e-12},
    'morphology': {'seconds': 120, 'min_beats': 30, 'template_samples': 48},
    'minimum_sqi': .30,
}
_EPS = 1e-12


def _filtered_channel(raw, support, fs):
    """Same causal filtering/polarity decision as the shared beat extractor."""
    x = np.asarray(raw, float)
    observed = np.asarray(support, bool) & np.isfinite(x)
    idx = np.arange(len(x))
    prev = np.maximum.accumulate(np.where(observed, idx, -1))
    filled = np.where(prev >= 0, x[np.maximum(prev, 0)], 0.)
    sos = signal.butter(3, [.5, min(5., .45 * fs)], btype='bandpass', fs=fs, output='sos')
    if not len(x):
        return x.copy(), observed
    y, _ = signal.sosfilt(sos, filled, zi=signal.sosfilt_zi(sos) * filled[0])
    calibration = max(1, int(round(10. * fs)))
    for stop in range(calibration, len(x) + 1, calibration):
        begin = stop - calibration
        good = observed[begin:stop].copy()
        good[:int(round(2. * fs))] = False
        first = y[begin:stop][good]
        if len(first) >= 2 * fs and np.std(first) >= 1e-8:
            if np.mean((first - np.mean(first)) ** 3) < 0:
                y = -y
            break
    last_bad = np.maximum.accumulate(np.where(~observed, idx, -int(2 * fs) - 1))
    settled = observed & ((idx - last_bad) > int(round(2. * fs)))
    settled[:int(round(2. * fs))] = False
    return y, settled


def _guard_refinement(beats, y, observed, fs):
    """Recheck the existing parabola correction; never modify caller's cache."""
    b = {key: np.asarray(value).copy() for key, value in beats.items()}
    peaks = np.asarray(b.get('peak_index', []), int)
    if not len(peaks):
        b['morph_peak_id'] = np.empty(0, np.int64)
        return b
    if np.any(peaks < 1) or np.any(peaks + 1 >= len(y)):
        raise ValueError('peak indices outside raw segment')
    old_t = b['peak_time_s'].copy()
    three_valid = observed[peaks - 1] & observed[peaks] & observed[peaks + 1]
    den = y[peaks - 1] - 2. * y[peaks] + y[peaks + 1]
    delta = np.divide(.5 * (y[peaks - 1] - y[peaks + 1]), den,
                      out=np.zeros(len(peaks)), where=np.abs(den) > 1e-10)
    refine = (three_valid & (den < -1e-10) & np.isfinite(delta)
              & (np.abs(delta) <= .5))
    precise = (peaks + np.where(refine, delta, 0.)) / fs
    rr = np.diff(precise)
    # Morphology records retain the identities of their original centre peaks.
    mt = b['morph_time_s']
    mid = np.searchsorted(old_t, mt)
    if len(mt) and (np.any(mid >= len(peaks)) or not np.allclose(old_t[mid], mt, atol=1e-7)):
        raise ValueError('morphology times do not match cached peak identities')
    b['morph_peak_id'] = mid.astype(np.int64)
    b['morph_time_s'] = precise[mid]
    b['peak_time_s'] = precise
    b['interval_start_s'], b['interval_end_s'] = precise[:-1], precise[1:]
    b['interval_time_s'], b['interval_s'] = (precise[:-1] + precise[1:]) / 2., rr
    trailing = np.lib.stride_tricks.sliding_window_view(np.pad(rr, (10, 0), mode='edge'), 11)
    reference = np.median(trailing, axis=-1)
    b['interval_valid'] &= (three_valid[:-1] & three_valid[1:]
                            & (rr >= 1. / 3.5) & (rr <= 1. / .6)
                            & (np.abs(rr - reference) <= .30 * reference))
    b['_refined_valid'] = refine
    return b


def _ppi_data(beats, end_s, seconds=300):
    """Select a complete same-channel window and retain uncompressed ids."""
    left = end_s - seconds
    if left < -1e-7:
        return None
    lo = int(np.searchsorted(beats['interval_start_s'], left, side='left'))
    hi = int(np.searchsorted(beats['interval_available_s'], end_s, side='right'))
    if hi <= lo:
        return None
    rr_all = beats['interval_s'][lo:hi]
    valid = (beats['interval_valid'][lo:hi] & np.isfinite(rr_all)
             & (rr_all >= 1. / 3.5) & (rr_all <= 1. / .6)
             & (beats['interval_end_s'][lo:hi] <= end_s))
    selected = np.flatnonzero(valid)
    if len(selected) < 40:
        return None
    ids = lo + selected
    rr = rr_all[selected]
    times = beats['interval_time_s'][ids]
    coverage, span = float(rr.sum() / seconds), float(times[-1] - times[0])
    starts, ends = beats['interval_start_s'], beats['interval_end_s']
    gaps = np.r_[starts[ids[0]] - left, starts[ids[1:]] - ends[ids[:-1]], end_s - ends[ids[-1]]]
    max_gap = float(np.maximum(gaps, 0.).max(initial=0.))
    if coverage < .8 or span < .9 * seconds or max_gap > 5.:
        return None
    return {'rr': rr, 'time': times, 'rr_all': rr_all, 'valid': valid,
            'quality': np.array([coverage, span, max_gap, valid.mean()], float)}


def _continuous_pairs(rr, valid, lag=1):
    """Every intervening ORIGINAL interval must remain valid."""
    rr, valid = np.asarray(rr, float), np.asarray(valid, bool)
    n = len(rr) - lag
    if n <= 0:
        return np.empty(0), np.empty(0)
    ok = np.logical_and.reduce([valid[j:j+n] for j in range(lag + 1)])
    return rr[:n][ok], rr[lag:][ok]


def _poincare_metrics(rr_all, valid):
    first, second = _continuous_pairs(rr_all, valid)
    if len(first) < 15:
        return {}
    # Millisecond units; direct rotated coordinates on the SAME valid pairs.
    d = (second - first) * 1000.
    sd1 = float(np.std(d / np.sqrt(2.), ddof=1))
    sd2 = float(np.std((first + second) * 1000. / np.sqrt(2.), ddof=1))
    result = {NEW_FEATURE_NAMES[12]: sd1,
              NEW_FEATURE_NAMES[14]: float(np.pi * sd1 * sd2)}
    if sd2 > 1e-10:
        result[NEW_FEATURE_NAMES[13]] = sd1 / sd2
    total = float(np.sum(d * d))
    if total > 1e-10:
        result[NEW_FEATURE_NAMES[15]] = float(100. * np.sum(d[d > 0.] ** 2) / total)
    nonzero = np.abs(d) > 1e-7
    if np.count_nonzero(nonzero):
        result[NEW_FEATURE_NAMES[16]] = float(100. * np.count_nonzero(d < -1e-7) / np.count_nonzero(nonzero))
    a5, b5 = _continuous_pairs(rr_all, valid, lag=5)
    if len(a5) >= 15:
        s1 = float(np.std((b5 - a5) / np.sqrt(2.), ddof=1))
        s2 = float(np.std((b5 + a5) / np.sqrt(2.), ddof=1))
        if s2 > 1e-10:
            result[NEW_FEATURE_NAMES[17]] = s1 / s2
    return result


def _permutation_entropy(rr_all, valid):
    if len(rr_all) < 3:
        return np.nan, {'permutation_patterns': 0, 'permutation_tie_fraction': 0.}
    windows = np.lib.stride_tricks.sliding_window_view(rr_all, 3)
    ok = np.lib.stride_tricks.sliding_window_view(valid, 3).all(axis=1)
    windows = windows[ok]
    if not len(windows):
        return np.nan, {'permutation_patterns': 0, 'permutation_tie_fraction': 0.}
    ties = (np.isclose(windows[:, 0], windows[:, 1], rtol=0., atol=1e-10)
            | np.isclose(windows[:, 1], windows[:, 2], rtol=0., atol=1e-10)
            | np.isclose(windows[:, 0], windows[:, 2], rtol=0., atol=1e-10))
    diag = {'permutation_patterns': int((~ties).sum()),
            'permutation_tie_fraction': float(ties.mean())}
    if (~ties).sum() < 100 or ties.mean() > .2:
        return np.nan, diag
    order = np.argsort(windows[~ties], axis=1)
    _, counts = np.unique(order, axis=0, return_counts=True)
    p = counts / counts.sum()
    return float(-np.sum(p * np.log(p)) / np.log(math.factorial(3))), diag


def _time_metrics(data):
    rr = data['rr']
    median = float(np.median(rr))
    result = {NEW_FEATURE_NAMES[6]: float(np.median(np.abs(rr - median)) * 1000.),
              NEW_FEATURE_NAMES[7]: float(np.diff(np.quantile(rr, [.25, .75]))[0] * 1000.)}
    a, b = _continuous_pairs(data['rr_all'], data['valid'])
    d = b - a
    if len(d) >= 15:
        result[NEW_FEATURE_NAMES[8]] = float(np.median(np.abs(d)) * 1000.)
        result[NEW_FEATURE_NAMES[9]] = float(np.sqrt(np.mean(d ** 2)) / np.mean(rr))
        result[NEW_FEATURE_NAMES[10]] = float(np.mean(np.abs(d) > .050))
    entropy, diag = _permutation_entropy(data['rr_all'], data['valid'])
    if np.isfinite(entropy):
        result[NEW_FEATURE_NAMES[11]] = entropy
    return result, diag


def _entropy(p):
    p = np.asarray(p, float)
    total = float(p.sum())
    if len(p) < 2 or total <= _EPS or not np.isfinite(total):
        return np.nan
    normalized = p / total
    positive = normalized > 0.
    return float(-np.sum(normalized[positive] * np.log(normalized[positive])) / np.log(len(p)))


def _frequency_metrics(data):
    t, rr = data['time'], data['rr']
    if t[-1] - t[0] < 270. or np.diff(t).max(initial=0.) > 5.:
        return {}
    grid = np.arange(t[0], t[-1], .25)
    x = signal.detrend(np.interp(grid, t, rr * 1000.), type='linear')
    f, p = signal.periodogram(x, fs=4., window='hann', detrend=False, scaling='density')
    lf, hf = _integral_band(f, p, .04, .15), _integral_band(f, p, .15, .40)
    total = lf + hf
    if not np.isfinite(total) or total <= _EPS:
        return {}
    band = (f >= .04) & (f <= .40)
    fb, pb = f[band], p[band]
    result = {NEW_FEATURE_NAMES[0]: total, NEW_FEATURE_NAMES[1]: hf / total,
              NEW_FEATURE_NAMES[2]: _entropy(pb),
              NEW_FEATURE_NAMES[5]: float(np.sum(fb * pb) / pb.sum())}
    for i, lo, hi, power in ((3, .04, .15, lf), (4, .15, .40, hf)):
        selected = (f >= lo) & (f <= hi)
        if power > _EPS and selected.sum():
            result[NEW_FEATURE_NAMES[i]] = float(f[selected][np.argmax(p[selected])])
    return result


def _maximum_invalid_run(valid, fs):
    padded = np.r_[True, np.asarray(valid, bool), True]
    starts = np.flatnonzero(padded[:-1] & ~padded[1:])
    ends = np.flatnonzero(~padded[:-1] & padded[1:])
    return float(np.max(ends - starts, initial=0) / fs)


def _waveform_metrics(y, settled, beats, end, fs):
    size = int(round(60. * fs))
    if end < size:
        return {}, None
    x, valid = y[end-size:end].copy(), settled[end-size:end]
    maximum_gap = _maximum_invalid_run(valid, fs)
    if valid.mean() < .95 or maximum_gap > .2 or not valid[0] or not valid[-1]:
        return {}, None
    # Only isolated <=0.2 s samples inside an observed continuous window may be
    # interpolated. Acquisition breaks remain the caller's hard segment boundary.
    if not valid.all():
        ix = np.arange(size)
        x[~valid] = np.interp(ix[~valid], ix[valid], x[valid])
    x = signal.detrend(x, type='linear')
    f, p = signal.periodogram(x, fs=fs, window='hann', detrend=False, scaling='density')
    band = (f >= .5) & (f <= 5.)
    fb, pb = f[band], p[band]
    power = _integral_band(f, p, .5, 5.)
    if power <= _EPS or pb.sum() <= _EPS:
        return {}, None
    centroid = float(np.sum(fb * pb) / pb.sum())
    floor = max(float(pb.mean()) * 1e-12, np.finfo(float).tiny)
    result = {NEW_FEATURE_NAMES[18]: _entropy(pb), NEW_FEATURE_NAMES[19]: centroid,
              NEW_FEATURE_NAMES[20]: float(np.sqrt(np.sum(pb * (fb-centroid) ** 2) / pb.sum())),
              NEW_FEATURE_NAMES[21]: float(np.exp(np.mean(np.log(np.maximum(pb, floor)))) / pb.mean())}
    quality = np.array([valid.mean(), 60., maximum_gap, valid.mean()], float)
    data = _ppi_data(beats, end / fs, 60)
    if data is not None:
        estimate = float(1. / np.median(data['rr']))
        neighbourhood = max(.10, .10 * estimate)
        search = (f >= max(.5, estimate-neighbourhood)) & (f <= min(5., estimate+neighbourhood))
        if search.any():
            f0 = float(f[search][np.argmax(p[search])])
            half = max(.10, .10 * f0)
            if f0-half >= .5 and f0+half <= 5.:
                p1 = _integral_band(f, p, f0-half, f0+half)
                if p1 > _EPS:
                    result[NEW_FEATURE_NAMES[24]] = float(np.clip(p1 / power, 0., 1.))
                    for harmonic, feature in ((2, 22), (3, 23)):
                        low, high = harmonic*f0-half, harmonic*f0+half
                        if low >= .5 and high <= 5.:
                            pk = _integral_band(f, p, low, high)
                            result[NEW_FEATURE_NAMES[feature]] = float(np.sqrt(max(pk, 0.) / p1))
    return result, quality


def _morphology_metrics(y, settled, beats, end, fs):
    end_s, left = end / fs, end / fs - 120.
    if left < 0:
        return {}, None
    lo = int(np.searchsorted(beats['morph_time_s'], left, side='left'))
    hi = int(np.searchsorted(beats['morph_available_s'], end_s, side='right'))
    if hi-lo < 30:
        return {}, None
    t = beats['morph_time_s'][lo:hi]
    amp = beats['amplitude'][lo:hi]
    width = beats['width_fraction'][lo:hi]
    ids = beats['morph_peak_id'][lo:hi]
    finite = np.isfinite(amp) & (amp > 1e-8) & np.isfinite(width) & (width > 0.) & (width < 1.)
    # Check centre-adjacent original PPI after V8's guarded peak refinement.
    int_valid = beats['interval_valid']
    finite &= (ids > 0) & (ids < len(int_valid))
    clipped = np.clip(ids, 1, max(1, len(int_valid)-1))
    finite &= int_valid[clipped-1] & int_valid[clipped]
    tx = t[finite]
    if len(tx) < 30 or finite.mean() < .6:
        return {}, None
    span = float(tx[-1] - tx[0])
    gap = float(np.diff(np.r_[left, tx, end_s]).max(initial=0.))
    count, _ = np.histogram(tx, np.linspace(left, end_s, 7))
    coverage = float(np.mean(count >= 2))
    if span < 90. or gap > 20. or coverage < 5. / 6.:
        return {}, None
    result = {}
    xx, ww, accepted_ids = amp[finite], width[finite], ids[finite]
    consecutive = np.diff(accepted_ids) == 1
    if consecutive.sum() >= 15 and np.median(xx) > 1e-8:
        result[NEW_FEATURE_NAMES[25]] = float(np.median(np.abs(np.diff(xx)[consecutive])) / np.median(xx))
    if np.mean(ww) > 1e-8:
        result[NEW_FEATURE_NAMES[26]] = float(np.std(ww, ddof=1) / np.mean(ww))
    profiles = []
    peak_t = beats['peak_time_s']
    for peak_id in accepted_ids:
        a = .5 * (peak_t[peak_id-1] + peak_t[peak_id])
        b = .5 * (peak_t[peak_id] + peak_t[peak_id+1])
        if a < left or b > end_s:
            continue
        ia, ib = int(np.ceil(a*fs)), int(np.floor(b*fs)) + 1
        if ib-ia < 5 or not settled[ia:ib].all():
            continue
        # Sample only within this complete pulse and frame: no outside
        # endpoint borrowing and no unobserved interpolation neighbours.
        samples = np.linspace(ia, ib-1, 48)
        pulse = np.interp(samples, np.arange(ia, ib), y[ia:ib])
        pulse -= pulse.mean()
        norm = np.linalg.norm(pulse)
        if norm > 1e-8:
            profiles.append(pulse / norm)
    if len(profiles) >= 30:
        profiles = np.asarray(profiles)
        template = np.median(profiles, axis=0)
        template -= template.mean()
        norm = np.linalg.norm(template)
        if norm > 1e-8:
            correlations = np.clip(profiles @ (template/norm), -1., 1.)
            result[NEW_FEATURE_NAMES[27]] = float(np.quantile(correlations, .75) - np.quantile(correlations, .25))
    quality = np.array([coverage, span, gap, finite.mean()], float)
    return result, quality


def extract_features_v8(segment_raw, segment_support, frame_end_indices, fs=25.,
                        channel_ids=None, frame_quality=None,
                        quality_frame_end_indices=None, *, beat_cache=None):
    """Calculate 28 V8 extras from one raw continuous [N,20] physical segment.

    Can accept a different channel count with explicit physical channel_ids;
    ambient channel 2 is always excluded. Optional cached beats must refer to
    the same raw segment, with sample-zero time origin. Reusable SQI rows are
    accepted only when they completely tile each past 60/120/300 s window.

    Returns values/mask/channels [F,28], feature_quality [F,28,5], family_channels
    [F,5] (0 unavailable, -1 scalar fallback routes differ), family_mask [F,5],
    family_quality [F,5,5], scalar observation counts and audit reasons. Zero
    values outside mask are placeholders, never real measurements.
    """
    raw = np.asarray(segment_raw)
    support = np.asarray(segment_support, bool)
    if raw.ndim != 2 or support.shape != raw.shape or not np.isfinite(fs) or fs <= 10.:
        raise ValueError('aligned physical raw/support arrays and fs >10 Hz are required')
    support = support & np.isfinite(raw)
    ends_in = np.asarray(frame_end_indices)
    ends = ends_in.astype(np.int64)
    if (ends.ndim != 1 or not np.array_equal(ends, ends_in) or np.any(ends < 0)
            or np.any(ends > len(raw)) or np.any(np.diff(ends) <= 0)):
        raise ValueError('frame ends must be increasing exclusive integer indices within this segment')
    if channel_ids is None:
        if raw.shape[1] == 20:
            channel_ids = np.arange(1, 21)
        elif raw.shape[1] == len(CANDIDATE_CHANNELS):
            channel_ids = CANDIDATE_CHANNELS
        else:
            raise ValueError('nonstandard channel count requires physical channel_ids')
    ids = np.asarray(channel_ids, np.int16)
    if ids.ndim != 1 or len(ids) != raw.shape[1] or len(np.unique(ids)) != len(ids):
        raise ValueError('unique channel ids must match raw columns')
    columns = [i for i, c in enumerate(ids) if int(c) in CANDIDATE_CHANNELS]
    cached = None if beat_cache is None else {int(k): v for k, v in beat_cache.items()}
    contexts = {}
    def context(column):
        if column not in contexts:
            physical = int(ids[column])
            if cached is not None and physical not in cached:
                raise ValueError('beat cache missing a requested physical channel')
            b = extract_channel_beats(raw[:, column], support[:, column], fs) if cached is None else _validate_beats(cached[physical])
            y, settled = _filtered_channel(raw[:, column], support[:, column], fs)
            b = _guard_refinement(b, y, support[:, column], fs)
            contexts[column] = (b, y, settled)
        return contexts[column]
    q = qe = None
    if frame_quality is not None:
        q = np.asarray(frame_quality, float)
        qe = np.asarray(ends if quality_frame_end_indices is None else quality_frame_end_indices, np.int64)
        if (qe.ndim != 1 or q.shape != (len(qe), len(ids)) or not np.isfinite(q).all()
                or np.any(np.diff(qe) <= 0) or np.any(qe < 0) or np.any(qe > len(raw))):
            raise ValueError('frame_quality must align with observed increasing quality frame ends')
    frame_size = int(round(30. * fs))
    def quality(end, seconds):
        left = end-int(round(seconds*fs))
        if q is not None:
            a, b = np.searchsorted(qe, [left+frame_size, end], side='left')
            b = int(np.searchsorted(qe, end, side='right'))
            if (b-a == seconds//30 and b>a and qe[a] == left+frame_size
                    and qe[b-1] == end and np.all(np.diff(qe[a:b]) == frame_size)):
                return q[a:b].mean(axis=0)
        scores = np.zeros(len(ids))
        for c in columns:
            scores[c] = channel_quality(raw[left:end, c], fs, support[left:end, c])
        return scores
    shape = (len(ends), 28)
    values, mask = np.zeros(shape, np.float32), np.zeros(shape, np.float32)
    channels = np.zeros(shape, np.int16)
    feature_quality = np.zeros(shape + (5,), np.float32)
    observation_count = np.zeros(shape, np.int32)
    failure_code = np.full(shape, 1, np.uint8)
    name_index = {name: i for i, name in enumerate(NEW_FEATURE_NAMES)}
    route_attempts = np.zeros((len(ends), 5), np.int16)
    permutation_diagnostics = []
    for row, end_ in enumerate(ends):
        end = int(end_)
        quality_cache, computed = {}, {}
        for family, (indices, seconds) in enumerate(zip(FAMILY_INDICES, FAMILY_SECONDS)):
            if end < int(round(seconds*fs)):
                continue
            failure_code[row, list(indices)] = 2
            if seconds not in quality_cache:
                quality_cache[seconds] = quality(end, seconds)
            scores = quality_cache[seconds]
            ordered = sorted(columns, key=lambda c: (-scores[c], int(ids[c])))
            for c in ordered:
                sqi = float(scores[c])
                if sqi < .30:
                    continue
                route_attempts[row, family] += 1
                failure_code[row, [i for i in indices if not mask[row, i]]] = 3
                key = (c, seconds)
                if key not in computed:
                    beats, y, settled = context(c)
                    if seconds == 300:
                        data = _ppi_data(beats, end/fs)
                        if data is None:
                            computed[key] = ({}, None, 0)
                        else:
                            result = _frequency_metrics(data)
                            time_values, diag = _time_metrics(data)
                            result.update(time_values)
                            result.update(_poincare_metrics(data['rr_all'], data['valid']))
                            permutation_diagnostics.append({'frame': int(row), 'channel': int(ids[c]), **diag})
                            computed[key] = (result, data['quality'], len(data['rr']))
                    elif seconds == 60:
                        result, diag = _waveform_metrics(y, settled, beats, end, fs)
                        computed[key] = (result, diag, int(np.count_nonzero(settled[end-int(60*fs):end])))
                    else:
                        result, diag = _morphology_metrics(y, settled, beats, end, fs)
                        n = int(np.count_nonzero((beats['morph_time_s'] >= end/fs-120.) & (beats['morph_available_s'] <= end/fs)))
                        computed[key] = (result, diag, n)
                result, diag, n_observations = computed[key]
                # Preserve a specific last-attempt reason for optional measures.
                for missing_i in indices:
                    if mask[row, missing_i] or NEW_FEATURE_NAMES[missing_i] in result:
                        continue
                    if missing_i == 11:
                        failure_code[row, missing_i] = 4
                    elif missing_i in (22, 23, 24):
                        failure_code[row, missing_i] = 5
                    elif missing_i in (13, 15, 16, 17):
                        failure_code[row, missing_i] = 6
                for name, value in result.items():
                    i = name_index[name]
                    if i not in indices or mask[row, i] or not np.isfinite(value):
                        continue
                    values[row, i], mask[row, i], channels[row, i] = value, 1., ids[c]
                    feature_quality[row, i] = np.r_[diag, sqi]
                    observation_count[row, i] = n_observations
                    failure_code[row, i] = 0
                if np.all(mask[row, list(indices)]):
                    break
    family_channels = np.zeros((len(ends), 5), np.int16)
    family_mask = np.zeros((len(ends), 5), np.float32)
    family_quality = np.zeros((len(ends), 5, 5), np.float32)
    for f, indices in enumerate(FAMILY_INDICES):
        for row in range(len(ends)):
            good = [i for i in indices if mask[row, i]]
            if good:
                unique = np.unique(channels[row, good])
                family_channels[row, f] = unique[0] if len(unique)==1 else -1
                family_mask[row, f] = 1.
                family_quality[row, f] = feature_quality[row, good].mean(axis=0)
    return {'values': values, 'mask': mask, 'channels': channels,
            'feature_names': NEW_FEATURE_NAMES, 'feature_quality': feature_quality,
            'family_channels': family_channels, 'family_mask': family_mask,
            'family_quality': family_quality, 'observation_count': observation_count,
            'failure_code': failure_code, 'route_attempts': route_attempts,
            'audit': {'spec': EXTRACTION_SPEC, 'family_names': FAMILY_NAMES,
                      'quality_names': QUALITY_NAMES,
                      'failure_codes': {0: 'available', 1: 'insufficient_window',
                                        2: 'no_qualified_channel', 3: 'ineligible_or_undefined_measurement',
                                        4: 'permutation_insufficient_contiguous_patterns_or_excess_ties',
                                        5: 'harmonic_missing_ppi_or_neighbourhood_outside_band_or_zero_power',
                                        6: 'poincare_insufficient_pairs_or_zero_denominator'},
                      'permutation_diagnostics': permutation_diagnostics}}


compute_segment_features = extract_features_v8
