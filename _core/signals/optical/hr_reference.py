
from __future__ import annotations

from collections import deque
import hashlib
import json
from pathlib import Path

import numpy as np

from .schema import FEATURE_NAMES, HR_REFERENCE_NAMES, V4_FEATURE_NAMES

VERSION = "ppg-v4-hr-reference-v1"
SPEC = {
    "current_lookback_ms": 60000, "history_ms": 1800000,
    "frame_ms": 30000, "inclusive_grid_slots": 61,
    "minimum_valid_frames": 10, "minimum_span_ms": 270000,
    "minimum_abs_reference": 1e-6, "scope": "single_original_continuous_segment",
    "channel_pooling": "HR_only", "labels_used": False,
}


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def feature_digest(values, mask, names, starts, ends):
    """Fingerprint original feature arrays and frame alignment, not labels/raw."""
    h = hashlib.sha256()
    for array in (values, mask, np.asarray(names).astype("U"), starts, ends):
        a = np.ascontiguousarray(array)
        h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


def compute_hr_reference(frame_end_ms, hr, hr_mask):
    """Generate once on the segment time axis; never restart per model window.

    At endpoint t, only estimates ending in [t-60s-30min, t-60s]
    are eligible. Endpoints are unique, on an uninterrupted 30s grid.
    The current estimate is inserted only after constructing its features.
    """
    end = np.asarray(frame_end_ms)
    hr = np.asarray(hr)
    observed = np.asarray(hr_mask).astype(bool)
    if end.ndim != 1 or hr.shape != end.shape or observed.shape != end.shape:
        raise ValueError("HR/time arrays must be aligned one-dimensional arrays")
    if not np.all(np.isfinite(end)) or not np.all(end == np.rint(end)):
        raise ValueError("Frame endpoints must be finite integer milliseconds")
    end = end.astype(np.int64)
    if len(end) > 1 and np.any(np.diff(end) != SPEC["frame_ms"]):
        raise ValueError("HR history requires unique uninterrupted 30s frames within one segment")
    values = np.zeros((len(end), 3), dtype=np.float32)
    masks = np.zeros((len(end), 3), dtype=bool)
    masks[:, 1] = True  # Zero coverage is observed metadata, not physiological support.
    history = deque()
    for i, t in enumerate(end):
        upper = int(t) - SPEC["current_lookback_ms"]
        lower = upper - SPEC["history_ms"]
        while history and history[0][0] < lower:
            history.popleft()
        past = [entry for entry in history if entry[0] <= upper]
        values[i, 1] = len(past) / SPEC["inclusive_grid_slots"]
        if len(past) > SPEC["inclusive_grid_slots"]:
            raise ValueError("Duplicate or invalid HR history time grid")
        if len(past) >= SPEC["minimum_valid_frames"] and past[-1][0] - past[0][0] >= SPEC["minimum_span_ms"]:
            samples = np.asarray([entry[1] for entry in past], dtype=np.float64)
            baseline = float(np.median(samples))
            if np.isfinite(baseline) and abs(baseline) > SPEC["minimum_abs_reference"]:
                mad = float(np.median(np.abs(samples - baseline))) / abs(baseline)
                if np.isfinite(mad):
                    values[i, 2] = mad; masks[i, 2] = True
                if observed[i] and np.isfinite(hr[i]):
                    relative = (float(hr[i]) - baseline) / abs(baseline)
                    if np.isfinite(relative):
                        values[i, 0] = relative; masks[i, 0] = True
        if observed[i] and np.isfinite(hr[i]):
            history.append((int(t), float(hr[i])))
    return values, masks


def record_key(record):
    return (str(record["subject_id"]), str(record["recording_id"]), str(record["continuous_id"]))


def _contained_sidecar_path(folder, relative):
    path = (folder / relative).resolve()
    if not path.is_relative_to(folder.resolve()):
        raise ValueError("Sidecar path escapes its manifest directory")
    return path


def attach_sidecar(records, cache_dir, sidecar_dir):
    """Check all source signatures and sidecar hashes before attaching paths."""
    folder = Path(sidecar_dir).resolve(); manifest_path = folder / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"{manifest_path}: run build_v4_hr_reference.py for this cache")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("version") != VERSION or manifest.get("spec") != SPEC:
        raise ValueError("Incompatible V4 HR sidecar generation specification")
    if manifest.get("feature_names") != list(HR_REFERENCE_NAMES) or manifest.get("input_feature_names") != list(V4_FEATURE_NAMES):
        raise ValueError("V4 sidecar feature names/order mismatch")
    if manifest.get("source_records_sha256") != sha256_file(Path(cache_dir) / "records.jsonl"):
        raise ValueError("HR sidecar belongs to a different source cache; generate a sidecar for this cache")
    entries = manifest.get("records", [])
    if len(entries) != len(records):
        raise ValueError("HR sidecar record count differs from source cache")
    index = {tuple(entry["key"]): entry for entry in entries}
    if len(index) != len(entries):
        raise ValueError("Duplicate HR sidecar record identity")
    digest = sha256_file(manifest_path)
    result = []
    for record in records:
        entry = index.get(record_key(record))
        if entry is None or int(entry["n_frames"]) != int(record["n_frames"]):
            raise ValueError("Missing/misaligned HR sidecar record")
        source = Path(record["_path"]).resolve(); stat = source.stat()
        if str(source) != entry["source_path"] or stat.st_size != entry["source_bytes"] or stat.st_mtime_ns != entry["source_mtime_ns"]:
            raise ValueError("Original V3 source cache changed since HR sidecar generation")
        path = _contained_sidecar_path(folder, entry["path"])
        if sha256_file(path) != entry["sha256"]:
            raise ValueError("HR sidecar content fingerprint mismatch")
        rec = dict(record)
        rec.update(_feature_sidecar=str(path), _feature_sidecar_manifest_sha256=digest,
                   _feature_sidecar_sha256=entry["sha256"],
                   _feature_sidecar_source_digest=entry["source_feature_digest"])
        result.append(rec)
    return result


def append_sidecar(record, values, mask, names, starts, ends):
    if tuple(np.asarray(names).astype(str)) != FEATURE_NAMES:
        raise ValueError("V4 sidecar requires original ordered 24-feature V3 cache")
    if feature_digest(values, mask, names, starts, ends) != record["_feature_sidecar_source_digest"]:
        raise ValueError("Original feature/time fingerprint no longer matches the HR sidecar")
    path = Path(record["_feature_sidecar"])
    if sha256_file(path) != record["_feature_sidecar_sha256"]:
        raise ValueError("HR sidecar changed after loading the record manifest")
    with np.load(path, allow_pickle=False) as z:
        if tuple(z["feature_names"].astype(str)) != HR_REFERENCE_NAMES:
            raise ValueError("Unexpected HR sidecar feature names/order")
        if not np.array_equal(z["frame_start_ms"], starts) or not np.array_equal(z["frame_end_ms"], ends):
            raise ValueError("HR sidecar frame timestamps are not aligned")
        extra, extra_mask = z["feature_values"], z["feature_mask"]
        if extra.shape != (len(starts), 3) or extra_mask.shape != extra.shape:
            raise ValueError("HR sidecar arrays have invalid dimensions")
        if not np.isfinite(extra).all():
            raise ValueError("HR sidecar values must be finite with missing entries masked")
    # Concatenation preserves every original value and mask without rescaling.
    return np.concatenate((values, extra), axis=1), np.concatenate((mask, extra_mask.astype(mask.dtype)), axis=1), list(V4_FEATURE_NAMES)


def build_sidecar(cache_dir, output_dir):
    from .data import load_records
    cache = Path(cache_dir).resolve(); output = Path(output_dir).resolve()
    records = load_records(cache)
    if (output / "manifest.json").exists():
        attach_sidecar(records, cache, output)
        return json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    if output.exists() and any(output.iterdir()):
        raise ValueError("Refusing to replace an incomplete/nonempty sidecar directory")
    output.mkdir(parents=True, exist_ok=True); (output / "records").mkdir()
    entries = []
    all_count = np.zeros(3, np.int64); total_frames = 0
    seen = set()
    for i, record in enumerate(records):
        key = record_key(record)
        if key in seen: raise ValueError("Duplicate continuous-segment identity")
        seen.add(key)
        source = Path(record["_path"]).resolve(); stat = source.stat()
        # Intentionally never read raw/raw_mask, meals, target or boundary members.
        with np.load(source, allow_pickle=False) as z:
            names = z["feature_names"]
            if tuple(names.astype(str)) != FEATURE_NAMES:
                raise ValueError("HR sidecar expects original 24-feature V3 cache")
            values, mask = z["feature_values"], z["feature_mask"]
            starts, ends = z["frame_start_ms"], z["frame_end_ms"]
            if values.shape != (int(record["n_frames"]), len(FEATURE_NAMES)) or mask.shape != values.shape:
                raise ValueError("Source feature dimensions disagree with record manifest")
            if not np.all(ends - starts == SPEC["frame_ms"]):
                raise ValueError("Source frame duration must be 30 seconds")
            extra, extra_mask = compute_hr_reference(ends, values[:, 0], mask[:, 0])
            source_digest = feature_digest(values, mask, names, starts, ends)
        name = f"records/{i:05d}_{hashlib.sha256(chr(0).join(key).encode()).hexdigest()[:16]}.npz"
        destination = output / name
        np.savez_compressed(destination, feature_values=extra, feature_mask=extra_mask,
                            feature_names=np.asarray(HR_REFERENCE_NAMES), frame_start_ms=starts, frame_end_ms=ends)
        # Check source stat stayed stable while it was read; source is read-only.
        after = source.stat()
        if (after.st_size, after.st_mtime_ns) != (stat.st_size, stat.st_mtime_ns):
            raise RuntimeError("Source cache changed during HR sidecar build")
        entries.append(dict(key=list(key), path=name, n_frames=len(starts), sha256=sha256_file(destination),
                            source_path=str(source), source_bytes=stat.st_size, source_mtime_ns=stat.st_mtime_ns,
                            source_feature_digest=source_digest))
        total_frames += len(starts); all_count += extra_mask.sum(axis=0)
    manifest = dict(version=VERSION, spec=SPEC, feature_names=list(HR_REFERENCE_NAMES),
                    input_feature_names=list(V4_FEATURE_NAMES), source_cache=str(cache),
                    source_records_sha256=sha256_file(cache / "records.jsonl"), records=entries,
                    record_count=len(entries), source_frames_including_overlap=total_frames,
                    added_available_frames_including_overlap=all_count.tolist(),
                    generation_reads_labels=False, original_cache_modified=False)
    pending = output / "manifest.pending.json"
    pending.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    pending.replace(output / "manifest.json")
    return manifest
