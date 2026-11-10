from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

SAMPLE_RATE = 15
AXES = 6
VERY_SMALL = 1e-6
SMOOTH_LENGTH = 15
SMOOTH_SIGMA = 10.0
TREND_WINDOW = 150
REST_WINDOW = 15
ACCEL_REST_THRESHOLD = 0.008
GYRO_REST_THRESHOLD_DEG = 0.04 * 180.0 / np.pi


def read_shm(path: str | Path) -> NDArray[np.float64]:
    """Read the transformed Clemson Shimmer format as Ax, Ay, Az, yaw, pitch, roll."""
    path = Path(path)
    return decode_shm(path.read_bytes(), source=str(path))


def decode_shm(payload: bytes, *, source: str = "<memory>") -> NDArray[np.float64]:
    """Decode Clemson's six-axis float32 SHM payload from memory."""
    values = np.frombuffer(payload, dtype="<f4")
    if values.size % AXES:
        raise ValueError(f"{source} contains {values.size} floats, not a multiple of six")
    raw = values.reshape(-1, AXES)
    result = np.empty_like(raw, dtype=np.float64)
    result[:, 0] = -raw[:, 1]
    result[:, 1] = -raw[:, 0]
    result[:, 2] = -raw[:, 2]
    result[:, 3] = raw[:, 5]
    result[:, 4] = raw[:, 3]
    result[:, 5] = raw[:, 4]
    return result


def smooth(raw: NDArray[np.float64]) -> NDArray[np.float64]:
    if raw.ndim != 2 or raw.shape[1] != AXES:
        raise ValueError(f"Expected [samples, 6], received {raw.shape}")
    count = raw.shape[0]
    if count <= SMOOTH_LENGTH:
        return raw.copy()

    radii = np.arange(SMOOTH_LENGTH - 1, -1, -1, dtype=np.float64)
    kernel = np.exp(-(radii * radii) / (2.0 * SMOOTH_SIGMA * SMOOTH_SIGMA))
    kernel /= kernel.sum()
    output = np.empty_like(raw)
    for axis in range(AXES):
        output[:, axis] = np.convolve(raw[:, axis], kernel, mode="full")[:count]
    output[:SMOOTH_LENGTH] = raw[:SMOOTH_LENGTH]
    return output


def _remove_accelerometer_trend(data: NDArray[np.float64]) -> None:
    count = data.shape[0]
    if count <= TREND_WINDOW:
        return
    for axis in range(3):
        cumulative = np.concatenate(([0.0], np.cumsum(data[:, axis], dtype=np.float64)))
        centers = np.arange(TREND_WINDOW // 2, count - TREND_WINDOW // 2)
        means = (
            cumulative[centers + TREND_WINDOW // 2]
            - cumulative[centers - TREND_WINDOW // 2]
        ) / TREND_WINDOW
        data[centers, axis] -= means


def motion_states(
    raw: NDArray[np.float64], preprocessed: NDArray[np.float64]
) -> NDArray[np.uint8]:
    """Return 0=device off, 1=rest, 2=motion using the original EatMon rules."""
    count = raw.shape[0]
    states = np.full(count, 2, dtype=np.uint8)
    if count == 0:
        return states

    zero_accel = np.all(np.abs(raw[:, :3]) <= VERY_SMALL, axis=1)
    constant_accel = np.zeros(count, dtype=bool)
    if count > 1:
        norm_squared = np.sum(raw[1:, :3] ** 2, axis=1)
        unchanged = np.all(np.abs(raw[1:, :3] - raw[:-1, :3]) <= VERY_SMALL, axis=1)
        constant_accel[1:] = (norm_squared >= 0.9) & unchanged
    off = zero_accel | constant_accel
    states[off] = 0
    preprocessed[constant_accel, :3] = 0.0

    if count < REST_WINDOW:
        return states
    cumulative = np.vstack((np.zeros((1, AXES)), np.cumsum(preprocessed, axis=0)))
    cumulative_sq = np.vstack(
        (np.zeros((1, AXES)), np.cumsum(preprocessed * preprocessed, axis=0))
    )
    starts = np.arange(0, count - REST_WINDOW + 1)
    totals = cumulative[starts + REST_WINDOW] - cumulative[starts]
    totals_sq = cumulative_sq[starts + REST_WINDOW] - cumulative_sq[starts]
    means = totals / REST_WINDOW
    variances = np.maximum(totals_sq / REST_WINDOW - means * means, 0.0)
    deviations = np.sqrt(variances)
    centers = starts + REST_WINDOW // 2
    rest_center = np.zeros(count, dtype=bool)
    eligible = ~off[centers]
    rest_center[centers] = eligible & (
        (deviations[:, :3].sum(axis=1) < ACCEL_REST_THRESHOLD)
        & (deviations[:, 3:].sum(axis=1) < GYRO_REST_THRESHOLD_DEG)
    )

    # The C loop can mark the preceding seven positions as rest after each position
    # initializes itself. This forward-looking reduction preserves that ordering.
    padded = np.pad(rest_center.astype(np.int32), (0, REST_WINDOW // 2))
    cumulative_rest = np.concatenate(([0], np.cumsum(padded)))
    future_rest = (
        cumulative_rest[REST_WINDOW // 2 + 1 : REST_WINDOW // 2 + 1 + count]
        - cumulative_rest[:count]
    )
    states[future_rest > 0] = 1
    return states


def preprocess(
    raw: NDArray[np.float64], *, apply_smoothing: bool = True, remove_accel_trend: bool = False
) -> tuple[NDArray[np.float64], NDArray[np.uint8]]:
    data = smooth(raw) if apply_smoothing else raw.copy()
    if remove_accel_trend:
        _remove_accelerometer_trend(data)
    states = motion_states(raw, data)
    return data, states
