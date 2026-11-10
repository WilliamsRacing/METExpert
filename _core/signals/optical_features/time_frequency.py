"""Fixed numerical PPG STFT: no image rendering, labels or learned preprocessing."""
from __future__ import annotations
import numpy as np
from scipy import fft, signal

STFT_SPEC = {"version": "v5-stft-1", "fs": 25., "frame_samples": 750,
             "window_samples": 200, "hop_samples": 25, "n_fft": 200,
             "window": "periodic_hann", "center": False, "frequency_min": .5,
             "frequency_max": 5., "column_min_coverage": .80,
             "transform": "log1p(one_sided_PSD)", "units_before_log": "normalized_amplitude_squared/Hz",
             "shape": [37, 23], "dtype": "float32"}


def waveform_spectrogram(raw, raw_mask, fs=25., minimum_coverage=.80):
    """Return [...,37,23] float32 PSD and [...,23] observed-column bool mask.

    The preceding raw representation is independently bandpass filtered and
    robustly normalized within each physical-channel 30 s frame. Zero fill is
    applied only to unavailable samples, with every 8 s window audited.
    """
    x = np.asarray(raw, np.float32)
    observed = np.asarray(raw_mask, bool)
    if x.shape != observed.shape or x.shape[-1] != 750 or float(fs) != 25.:
        raise ValueError("V5 STFT requires aligned 750-point 25 Hz frames")
    if not 0 <= minimum_coverage <= 1:
        raise ValueError("Invalid STFT coverage threshold")
    finite = np.isfinite(x)
    observed = observed & finite
    x = np.where(observed, x, 0.)
    windows = np.lib.stride_tricks.sliding_window_view(x, 200, axis=-1)[..., ::25, :]
    support = np.lib.stride_tricks.sliding_window_view(observed, 200, axis=-1)[..., ::25, :]
    valid = support.mean(axis=-1) >= minimum_coverage
    hann = signal.windows.hann(200, sym=False).astype(np.float32)
    transform = fft.rfft(windows * hann, n=200, axis=-1)
    power = (transform.real ** 2 + transform.imag ** 2) / (fs * np.sum(hann ** 2))
    power[..., 1:-1] *= 2.
    frequencies = fft.rfftfreq(200, 1. / fs)
    keep = (frequencies >= .5) & (frequencies <= 5.)
    result = np.moveaxis(np.log1p(power[..., keep]), -1, -2)
    result = np.where(valid[..., None, :], result, 0.).astype(np.float32)
    if result.shape[-2:] != (37, 23) or not np.isfinite(result).all():
        raise ValueError("Invalid STFT shape or nonfinite transform")
    return result, valid
