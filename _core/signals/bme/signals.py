from __future__ import annotations

import io
import hashlib
import math
import tempfile
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Iterator

import imufusion
import numpy as np
from numpy.typing import NDArray
from scipy import signal

SAMPLE_RATE = 15
# Hardware-defined raw-count conversions. The gyroscope scale is supplied in
# rad/s per count, while EatMon and imufusion expect deg/s.
ACC_G_PER_COUNT = 0.00024414
GYRO_RAD_PER_COUNT = 0.00122173
GYRO_DEG_PER_COUNT = GYRO_RAD_PER_COUNT * 180.0 / math.pi
ACC_COUNTS_PER_G = 1.0 / ACC_G_PER_COUNT
GYRO_COUNTS_PER_DPS = 1.0 / GYRO_DEG_PER_COUNT
LOWPASS_HZ = 6.0
MAX_INTERPOLATION_GAP_MS = 250.0
MAX_MODALITY_GAP_MS = 1000.0
FUSION_RESET_GAP_MS = 1000.0
FUSION_WARMUP_SAMPLES = 3 * SAMPLE_RATE

EXPECTED_HEADER = (
    "ACC_TIME",
    "PPG_TIME",
    "GYRO_TIME",
    *(f"PPG{i}" for i in range(1, 45)),
    "ACC_X",
    "ACC_Y",
    "ACC_Z",
    "GYRO_X",
    "GYRO_Y",
    "GYRO_Z",
)
EXPECTED_HEADER_BYTES = "\t".join(EXPECTED_HEADER).encode("ascii")
# Keep all three packet clocks while retaining only six IMU value columns.
# Layout: ACC_TIME, PPG_TIME, GYRO_TIME, ACC_X/Y/Z, GYRO_X/Y/Z.
USECOLS = (0, 1, 2, 47, 48, 49, 50, 51, 52)


@dataclass
class ResampledChunk:
    ticks: NDArray[np.int64]
    values: NDArray[np.float32]
    priority: int
    recording_id: str
    zip_name: str
    qc: dict[str, Any] = field(default_factory=dict)
    ppg_valid: NDArray[np.bool_] | None = None


@dataclass
class RecordingReadResult:
    chunks: list[ResampledChunk]
    qc: dict[str, Any]


class SensorZipSource:
    """Open child sensor ZIPs from an extracted directory or an outer ZIP."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self._outer: zipfile.ZipFile | None = None
        self._paths: dict[str, Path] = {}
        self._members: dict[str, str] = {}

    def __enter__(self) -> SensorZipSource:
        if self.root.is_dir():
            for path in self.root.rglob("*.zip"):
                self._add_unique(self._paths, path.name, path)
        elif self.root.is_file() and self.root.suffix.lower() == ".zip":
            self._outer = zipfile.ZipFile(self.root)
            for name in self._outer.namelist():
                path = PurePosixPath(name)
                if path.name.lower().endswith(".zip"):
                    self._add_unique(self._members, path.name, name)
        else:
            raise FileNotFoundError(self.root)
        if not self.available_names:
            raise ValueError(f"No child sensor ZIPs found in {self.root}")
        return self

    @staticmethod
    def _add_unique(target: dict[str, Any], name: str, value: Any) -> None:
        if name in target:
            raise ValueError(f"Duplicate sensor ZIP basename: {name}")
        target[name] = value

    @property
    def available_names(self) -> set[str]:
        return set(self._paths) | set(self._members)

    def archive_sha256(self, basename: str) -> str:
        """Hash the exact source ZIP, including info.json and other provenance."""
        if basename in self._paths:
            handle = self._paths[basename].open("rb")
        elif basename in self._members and self._outer is not None:
            handle = self._outer.open(self._members[basename])
        else:
            raise FileNotFoundError(basename)
        digest = hashlib.sha256()
        with handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    @contextmanager
    def open_child(self, basename: str) -> Iterator[zipfile.ZipFile]:
        if basename in self._paths:
            with zipfile.ZipFile(self._paths[basename]) as archive:
                yield archive
            return
        if basename not in self._members or self._outer is None:
            raise FileNotFoundError(f"Sensor ZIP is not present in source: {basename}")
        info = self._outer.getinfo(self._members[basename])
        if info.file_size > 2 * 1024**3:
            raise ValueError(f"Nested ZIP is unexpectedly large: {basename}")
        nested = io.BytesIO(self._outer.read(info))
        with zipfile.ZipFile(nested) as archive:
            yield archive

    def close(self) -> None:
        if self._outer is not None:
            self._outer.close()
            self._outer = None

    def __exit__(self, *exc: Any) -> None:
        self.close()


def _safe_text_members(archive: zipfile.ZipFile) -> list[str]:
    names = archive.namelist()
    if len(names) != len(set(names)):
        raise ValueError("Sensor ZIP contains duplicate members")
    result = []
    for name in names:
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or "\\" in name or ":" in name:
            raise ValueError(f"Unsafe sensor ZIP member: {name!r}")
        if path.name.lower().endswith(".txt"):
            result.append(name)
    if not result:
        raise ValueError("Sensor ZIP contains no TXT member")
    return sorted(result)


def _numeric_row(line: bytes) -> list[int] | None:
    parts = line.strip().split(b"\t")
    if len(parts) != len(EXPECTED_HEADER):
        return None
    try:
        return [int(parts[index]) for index in USECOLS]
    except ValueError:
        return None


def _fallback_rows(payload: bytes) -> tuple[NDArray[np.int64], int]:
    values: list[list[int]] = []
    rejected = 0
    for line in payload.splitlines():
        row = _numeric_row(line)
        if row is None:
            rejected += 1
        else:
            values.append(row)
    if not values:
        return np.empty((0, len(USECOLS)), dtype=np.int64), rejected
    return np.asarray(values, dtype=np.int64), rejected


def parse_sensor_payload(payload: bytes) -> tuple[NDArray[np.int64], dict[str, Any]]:
    if b"\x00" in payload[:4096]:
        raise ValueError("unsupported_binary_payload")
    first, separator, remainder = payload.partition(b"\n")
    first = first.lstrip(b"\xef\xbb\xbf").rstrip(b"\r")
    header_ok = first == EXPECTED_HEADER_BYTES
    first_numeric = _numeric_row(first)
    if header_ok:
        numeric_payload = remainder
    elif first_numeric is not None:
        numeric_payload = payload
    else:
        numeric_payload = remainder if separator else b""

    fallback = False
    rejected_rows = 0
    if numeric_payload.strip():
        try:
            rows = np.loadtxt(
                io.BytesIO(numeric_payload),
                dtype=np.int64,
                delimiter="\t",
                usecols=USECOLS,
                ndmin=2,
            )
        except (ValueError, UnicodeDecodeError):
            fallback = True
            rows, rejected_rows = _fallback_rows(numeric_payload)
    else:
        rows = np.empty((0, len(USECOLS)), dtype=np.int64)
    return rows, {
        "header_recovered": not header_ok,
        "fallback_parser": fallback,
        "rejected_rows": rejected_rows,
        "numeric_rows": int(rows.shape[0]),
    }


def expand_packet_timestamps(timestamps: NDArray[np.int64]) -> NDArray[np.float64]:
    """Expand repeated packet timestamps into per-sample millisecond times."""
    timestamps = np.asarray(timestamps, dtype=np.int64)
    output = np.full(timestamps.shape, np.nan, dtype=np.float64)
    valid_indices = np.flatnonzero(timestamps > 0)
    if valid_indices.size == 0:
        return output
    values = timestamps[valid_indices]
    starts = np.r_[0, np.flatnonzero(np.diff(values) != 0) + 1]
    ends = np.r_[starts[1:], values.size]

    candidate_steps = []
    for group_index in range(len(starts) - 1):
        start, end = int(starts[group_index]), int(ends[group_index])
        delta = float(values[starts[group_index + 1]] - values[start])
        if delta > 0:
            candidate_steps.append(delta / max(1, end - start))
    default_step = float(np.median(candidate_steps)) if candidate_steps else 1000.0 / 106.383

    for group_index, (start, end) in enumerate(zip(starts, ends, strict=True)):
        start, end = int(start), int(end)
        step = default_step
        if group_index + 1 < len(starts):
            delta = float(values[starts[group_index + 1]] - values[start])
            if delta > 0:
                step = delta / max(1, end - start)
        output[valid_indices[start:end]] = values[start] + np.arange(end - start) * step
    return output


def _filter_and_resample(
    times_ms: NDArray[np.float64], values: NDArray[np.float64]
) -> tuple[NDArray[np.int64], NDArray[np.float32], float] | None:
    if times_ms.size < 2:
        return None
    delta_ms = np.diff(times_ms)
    positive = delta_ms[delta_ms > 0]
    if positive.size == 0:
        return None
    native_rate = 1000.0 / float(np.median(positive))
    filtered = values
    if native_rate > 2.0:
        cutoff = min(LOWPASS_HZ, native_rate * 0.45)
        if cutoff > 0 and cutoff < native_rate / 2:
            sos = signal.butter(8, cutoff, btype="lowpass", fs=native_rate, output="sos")
            try:
                filtered = signal.sosfiltfilt(sos, values, axis=0)
            except ValueError:
                filtered = signal.sosfilt(sos, values, axis=0)

    first_tick = math.ceil(times_ms[0] * SAMPLE_RATE / 1000.0)
    last_tick = math.floor(times_ms[-1] * SAMPLE_RATE / 1000.0)
    if last_tick < first_tick:
        return None
    ticks = np.arange(first_tick, last_tick + 1, dtype=np.int64)
    target_ms = ticks.astype(np.float64) * 1000.0 / SAMPLE_RATE
    result = np.empty((ticks.size, 6), dtype=np.float32)
    for axis in range(6):
        result[:, axis] = np.interp(target_ms, times_ms, filtered[:, axis]).astype(np.float32)
    return ticks, result, native_rate


def _presence_on_ticks(
    timestamps_ms: NDArray[np.float64], ticks: NDArray[np.int64]
) -> NDArray[np.bool_]:
    """Project a modality packet clock onto the 15 Hz target ticks.

    A gap over one second is kept as missing. This is a coverage mask, not an
    interpolation rule and does not assert that the modality values are valid.
    """
    values = np.asarray(timestamps_ms, dtype=np.float64)
    values = values[np.isfinite(values) & (values > 0)]
    result = np.zeros(ticks.shape, dtype=np.bool_)
    if values.size == 0 or ticks.size == 0:
        return result
    values = np.unique(np.sort(values))
    breaks = np.flatnonzero(np.diff(values) > MAX_MODALITY_GAP_MS) + 1
    starts = np.r_[0, breaks]
    ends = np.r_[breaks, values.size]
    target_ms = ticks.astype(np.float64) * 1000.0 / SAMPLE_RATE
    for start, end in zip(starts, ends, strict=True):
        result |= (target_ms >= values[start]) & (target_ms <= values[end - 1])
    return result


def _resample_rows(
    rows: NDArray[np.int64], priority: int, recording_id: str, zip_name: str
) -> tuple[list[ResampledChunk], dict[str, Any]]:
    acc_present = rows[:, 0] > 0 if rows.size else np.zeros(0, dtype=bool)
    ppg_present = rows[:, 1] > 0 if rows.size else np.zeros(0, dtype=bool)
    gyro_present = rows[:, 2] > 0 if rows.size else np.zeros(0, dtype=bool)
    joint = acc_present & gyro_present
    qc: dict[str, Any] = {
        "raw_rows": int(rows.shape[0]),
        "acc_rows": int(np.count_nonzero(acc_present)),
        "ppg_rows": int(np.count_nonzero(ppg_present)),
        "gyro_rows": int(np.count_nonzero(gyro_present)),
        "joint_rows": int(np.count_nonzero(joint)),
        "continuous_blocks": 0,
        "native_rate_hz_median": None,
    }
    if np.count_nonzero(joint) < 2:
        return [], qc

    acc_times = expand_packet_timestamps(rows[:, 0])
    gyro_times = expand_packet_timestamps(rows[:, 2])
    indices = np.flatnonzero(joint & np.isfinite(acc_times) & np.isfinite(gyro_times))
    times = (acc_times[indices] + gyro_times[indices]) / 2.0
    values = rows[indices, 3:].astype(np.float64)
    values[:, :3] *= ACC_G_PER_COUNT
    values[:, 3:] *= GYRO_DEG_PER_COUNT

    delta_ms = np.diff(times)
    breaks = np.flatnonzero((delta_ms <= 0) | (delta_ms > MAX_INTERPOLATION_GAP_MS)) + 1
    starts = np.r_[0, breaks]
    ends = np.r_[breaks, times.size]
    chunks: list[ResampledChunk] = []
    rates = []
    for start, end in zip(starts, ends, strict=True):
        resampled = _filter_and_resample(times[start:end], values[start:end])
        if resampled is None:
            continue
        ticks, block, native_rate = resampled
        ppg_valid = _presence_on_ticks(
            expand_packet_timestamps(rows[:, 1]), ticks
        )
        rates.append(native_rate)
        chunks.append(
            ResampledChunk(
                ticks=ticks,
                values=block,
                priority=priority,
                recording_id=recording_id,
                zip_name=zip_name,
                qc={"native_rate_hz": native_rate},
                ppg_valid=ppg_valid,
            )
        )
    qc["continuous_blocks"] = len(chunks)
    if rates:
        qc["native_rate_hz_median"] = float(np.median(rates))
    return chunks, qc


def read_and_resample_recording(
    source: SensorZipSource, record: dict[str, Any]
) -> RecordingReadResult:
    chunks: list[ResampledChunk] = []
    member_reports = []
    with source.open_child(record["zip_name"]) as archive:
        for member in _safe_text_members(archive):
            with archive.open(member) as handle:
                prefix = handle.read(4096)
                if b"\x00" in prefix:
                    member_reports.append(
                        {
                            "member": member,
                            "unsupported_binary_payload": True,
                            "header_recovered": False,
                            "fallback_parser": False,
                            "rejected_rows": 0,
                            "numeric_rows": 0,
                            "raw_rows": 0,
                            "acc_rows": 0,
                            "ppg_rows": 0,
                            "gyro_rows": 0,
                            "joint_rows": 0,
                            "continuous_blocks": 0,
                            "native_rate_hz_median": None,
                        }
                    )
                    continue
                payload = prefix + handle.read()
            rows, parse_qc = parse_sensor_payload(payload)
            member_chunks, resample_qc = _resample_rows(
                rows,
                int(record["upload_ms"]),
                str(record["uniqueid"]),
                str(record["zip_name"]),
            )
            chunks.extend(member_chunks)
            member_reports.append({"member": member, **parse_qc, **resample_qc})
    return RecordingReadResult(
        chunks=chunks,
        qc={
            "recording_id": record["uniqueid"],
            "zip_name": record["zip_name"],
            "members": member_reports,
            "chunk_count": len(chunks),
        },
    )


def continuous_runs(valid: NDArray[np.bool_]) -> list[tuple[int, int]]:
    padded = np.r_[False, np.asarray(valid, dtype=bool), False]
    changes = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    return [(int(start), int(end)) for start, end in zip(starts, ends, strict=True)]


def remove_gravity(
    device_values: NDArray[np.float32], valid: NDArray[np.bool_]
) -> NDArray[np.float32]:
    if device_values.ndim != 2 or device_values.shape[1] != 6:
        raise ValueError(f"Expected [samples, 6], got {device_values.shape}")
    if valid.shape != (device_values.shape[0],):
        raise ValueError("Validity mask does not match device values")
    output = np.zeros_like(device_values, dtype=np.float32)
    settings = imufusion.AhrsSettings(
        sample_rate=SAMPLE_RATE,
        convention=imufusion.CONVENTION_NWU,
        gain=0.5,
        gyroscope_range=2000.0,
        acceleration_rejection=10.0,
        magnetic_rejection=0.0,
        rejection_timeout=5.0,
    )
    bias_settings = imufusion.BiasSettings(
        sample_rate=SAMPLE_RATE,
        stationary_threshold=3.0,
        stationary_period=5.0,
    )
    runs = continuous_runs(valid)
    grouped_runs: list[list[tuple[int, int]]] = []
    max_reset_gap = round(FUSION_RESET_GAP_MS * SAMPLE_RATE / 1000.0)
    for run in runs:
        if grouped_runs and run[0] - grouped_runs[-1][-1][1] <= max_reset_gap:
            grouped_runs[-1].append(run)
        else:
            grouped_runs.append([run])
    for group in grouped_runs:
        ahrs = imufusion.Ahrs()
        ahrs.set_settings(settings)
        bias = imufusion.Bias()
        bias.set_settings(bias_settings)
        seen = 0
        previous_index: int | None = None
        for index in range(group[0][0], group[-1][1]):
            if not valid[index]:
                continue
            if previous_index is not None:
                ahrs.set_sample_period((index - previous_index) / SAMPLE_RATE)
            accelerometer = device_values[index, :3].astype(np.float64)
            gyroscope = bias.update(device_values[index, 3:].astype(np.float64))
            ahrs.update_no_magnetometer(gyroscope, accelerometer)
            if seen >= FUSION_WARMUP_SAMPLES:
                output[index, :3] = ahrs.get_linear_acceleration()
                output[index, 3:] = gyroscope
            seen += 1
            previous_index = index
    return output


def device_to_internal(
    device_values: NDArray[np.float32], hand: str, orientation: str = "standard"
) -> NDArray[np.float32]:
    values = np.asarray(device_values, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 6:
        raise ValueError(f"Expected [samples, 6], got {values.shape}")
    if hand not in {"left", "right"}:
        raise ValueError(f"Unsupported hand: {hand}")
    if orientation not in {"standard", "rot180"}:
        raise ValueError(f"Unsupported orientation: {orientation}")
    device = values.copy()
    if orientation == "rot180":
        device[:, [0, 1, 3, 4]] *= -1.0
    internal = np.empty_like(device)
    internal[:, 0] = -device[:, 0]
    internal[:, 1] = device[:, 1]
    internal[:, 2] = -device[:, 2]
    internal[:, 3] = device[:, 5]
    internal[:, 4] = -device[:, 4]
    internal[:, 5] = device[:, 3]
    if hand == "left":
        internal[:, :3] *= np.asarray([1.0, -1.0, 1.0], dtype=np.float32)
        internal[:, 3:] *= np.asarray([-1.0, 1.0, -1.0], dtype=np.float32)
    return internal


def internal_to_shm(internal: NDArray[np.float32]) -> NDArray[np.float32]:
    internal = np.asarray(internal, dtype=np.float32)
    raw = np.empty_like(internal)
    raw[:, 0] = -internal[:, 1]
    raw[:, 1] = -internal[:, 0]
    raw[:, 2] = -internal[:, 2]
    raw[:, 3] = internal[:, 4]
    raw[:, 4] = internal[:, 5]
    raw[:, 5] = internal[:, 3]
    return raw.astype("<f4", copy=False)


def device_to_shm(
    device_values: NDArray[np.float32], hand: str, orientation: str = "standard"
) -> NDArray[np.float32]:
    return internal_to_shm(device_to_internal(device_values, hand, orientation))


def shm_to_device(raw_shm: NDArray[np.float32], primary_hand: str) -> NDArray[np.float32]:
    raw = np.asarray(raw_shm, dtype=np.float32)
    if raw.ndim != 2 or raw.shape[1] != 6:
        raise ValueError(f"Expected [samples, 6], got {raw.shape}")
    internal = np.empty_like(raw)
    internal[:, 0] = -raw[:, 1]
    internal[:, 1] = -raw[:, 0]
    internal[:, 2] = -raw[:, 2]
    internal[:, 3] = raw[:, 5]
    internal[:, 4] = raw[:, 3]
    internal[:, 5] = raw[:, 4]
    if primary_hand == "left":
        internal[:, :3] *= np.asarray([1.0, -1.0, 1.0], dtype=np.float32)
        internal[:, 3:] *= np.asarray([-1.0, 1.0, -1.0], dtype=np.float32)
    device = np.empty_like(internal)
    device[:, 0] = -internal[:, 0]
    device[:, 1] = internal[:, 1]
    device[:, 2] = -internal[:, 2]
    device[:, 3] = internal[:, 5]
    device[:, 4] = -internal[:, 4]
    device[:, 5] = internal[:, 3]
    return device


def temporary_shm(values: NDArray[np.float32]) -> tempfile.NamedTemporaryFile:
    handle = tempfile.NamedTemporaryFile(suffix=".shm", delete=False)
    np.asarray(values, dtype="<f4").tofile(handle)
    handle.close()
    return handle
