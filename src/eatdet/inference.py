"""Signal-to-event inference for the delivered eating detector.

Only physical signal, availability and timestamp arrays are read for prediction.
The same numerical evidence builder is used by the training entry point.
"""
from __future__ import annotations

from collections import OrderedDict, defaultdict
from pathlib import Path
import csv
import gc
import hashlib
import json
import math
import os
import tempfile
from typing import Iterable

import numpy as np
import torch


POINTS = 450
DEFAULT_DECODER = dict(threshold=.5, low_threshold=.2,
    min_duration_seconds=450, merge_gap_seconds=60,
    short_min_duration_seconds=300, short_high_fraction=.5,
    short_high_duration_seconds=30, event_high_fraction=0.)
COMPONENT_FILES = {"imu_primary": "imu_primary.pt", "imu_context": "imu_context.pt",
                   "ppg": "ppg.pt", "primary_fusion": "primary_fusion.pt", "event_ranker": "event_ranker.pt"}


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _checkpoint(value):
    if isinstance(value, dict):
        return value
    return torch.load(Path(value), map_location="cpu", weights_only=True)


def verify_weights(weights_dir):
    """Verify the component inventory, file hashes and decoder parameters."""
    root = Path(weights_dir).expanduser().resolve()
    manifest = _read(root / "manifest.json")
    if manifest.get("format") != "eating-detection.weights.1":
        raise ValueError("Unsupported model weight manifest")
    components = manifest.get("components", {})
    if set(components) != set(COMPONENT_FILES):
        raise ValueError("Weight manifest must identify all five trained components")
    resolved = {}
    for name, expected_name in COMPONENT_FILES.items():
        item = components[name]
        relative = Path(item["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Weight manifest paths must be package-relative")
        path = ((root.parent / relative) if relative.parts[:1] == (root.name,) else (root / relative)).resolve()
        if not path.is_relative_to(root) or path.name != expected_name:
            raise ValueError("Weight component is outside the model directory: " + name)
        if not path.is_file():
            raise FileNotFoundError(path)
        expected = item.get("sha256", "")
        if len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
            raise ValueError("Malformed weight SHA256: " + name)
        h = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                h.update(block)
        if h.hexdigest() != expected:
            raise ValueError("Weight checksum mismatch: " + name)
        if "bytes" in item and int(item["bytes"]) != path.stat().st_size:
            raise ValueError("Weight size mismatch: " + name)
        resolved[name] = str(path)
    rule = DEFAULT_DECODER | dict(manifest.get("decoder", {}))
    if any(k not in DEFAULT_DECODER for k in rule):
        raise ValueError("Unknown decoder parameter")
    if any(v is not None and (not isinstance(v, (int, float)) or not math.isfinite(v)) for v in rule.values()):
        raise ValueError("Decoder parameters must be finite numerical values")
    if any(v is None and k not in ("low_threshold", "short_min_duration_seconds") for k, v in rule.items()):
        raise ValueError("Required decoder parameters cannot be null")
    low = rule["threshold"] if rule["low_threshold"] is None else rule["low_threshold"]
    if not 0 <= low <= rule["threshold"] <= 1:
        raise ValueError("Invalid decoder probability thresholds")
    if any(rule[k] is not None and rule[k] < 0 for k in ("min_duration_seconds", "merge_gap_seconds", "short_min_duration_seconds", "short_high_duration_seconds")):
        raise ValueError("Decoder durations must be nonnegative")
    if any(not 0 <= rule[k] <= 1 for k in ("short_high_fraction", "event_high_fraction")):
        raise ValueError("Invalid decoder evidence fraction")
    return dict(manifest, resolved_components=resolved, decoder=rule, verified=True)


@torch.inference_mode()
def check_models(weights_dir, *, device="cpu"):
    """Load and execute every delivered trained model on numerical test signals."""
    from motion_encoder.model import IMUV6Model, ModelConfig as PrimaryConfig
    from context_encoder.model import IMUV8Model, ModelConfig as ContextConfig
    from signals.optical.adapter import PPGV9RuntimeAdapter
    from primary_fusion.model import FusionModel
    from event_detection.ranker import EventRanker, model_scores
    from event_detection.events import CANDIDATE_FEATURE_NAMES
    manifest = verify_weights(weights_dir)
    files = manifest["resolved_components"]
    checkpoints = {k: _checkpoint(p) for k, p in files.items()}
    for name, ck in checkpoints.items():
        state = ck.get("model_state" if name == "ppg" else "state_dict", {})
        if not state or any(not isinstance(v, torch.Tensor) or not bool(torch.isfinite(v).all()) for v in state.values()):
            raise ValueError("Invalid or nonfinite model tensors: " + name)
    device = torch.device(device)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(min(previous_threads, 4))
    try:
        n = 4
        signal = torch.zeros((1, n, 12, POINTS), device=device)
        mask = torch.ones((1, n, POINTS), dtype=torch.bool, device=device)
        quality = torch.ones((1, n, 8), device=device)
        ids = torch.zeros((1, n), dtype=torch.long, device=device)
        models = (IMUV6Model(PrimaryConfig(**checkpoints["imu_primary"]["model_config"])),
                  IMUV8Model(ContextConfig(**checkpoints["imu_context"]["model_config"])))
        outputs = []
        for name, model in zip(("imu_primary", "imu_context"), models):
            model.load_state_dict(checkpoints[name]["state_dict"], strict=True)
            out = model.to(device).eval()(signal, mask, quality, ids)
            if out["probability"].shape != (1, n) or not bool(torch.isfinite(out["probability"]).all()):
                raise ValueError("Invalid IMU forward: " + name)
            outputs.append(out)
        optical = PPGV9RuntimeAdapter(checkpoints["ppg"]).to(device).eval()
        ppg = optical(raw=torch.zeros((1, n, 3, 1), device=device),
            raw_mask=torch.ones((1, n, 3, 750), dtype=torch.bool, device=device),
            quality=torch.ones((1, n, 3), device=device),
            channel_ids=torch.arange(3, device=device)[None, None].expand(1, n, -1),
            features=torch.zeros((1, n, 68), device=device),
            feature_mask=torch.ones((1, n, 68), dtype=torch.bool, device=device),
            frame_valid=torch.ones((1, n), device=device),
            spectrogram=torch.zeros((1, n, 3, 37, 23), device=device),
            spectrogram_mask=torch.ones((1, n, 3, 23), dtype=torch.bool, device=device),
            segment_ids=ids)
        if ppg["embedding"].shape != (1, n, 64) or not bool(torch.isfinite(ppg["logits"]).all()):
            raise ValueError("Invalid PPG forward")
        ck = checkpoints["primary_fusion"]
        fusion = FusionModel(dropout=float(ck.get("config", {}).get("dropout", .1))).to(device).eval()
        fusion.load_state_dict(ck["state_dict"], strict=True)
        fusion_out = fusion(imu_logits=torch.logit(outputs[0]["probability"].clamp(1e-6, 1-1e-6)), imu_embedding=outputs[0]["embedding"],
            imu_support=mask, imu_quality=quality, imu_role_prob=outputs[0]["logits"].float().softmax(-1),
            ppg_logits=ppg["logits"], ppg_embedding=ppg["embedding"], ppg_support=ppg["prediction_valid"],
            ppg_quality=torch.ones((1, n), device=device), wave_coverage=torch.ones((1, n), device=device),
            feature_coverage=torch.ones((1, n), device=device))
        if fusion_out["probability"].shape != (1, n, POINTS) or not bool(torch.isfinite(fusion_out["probability"]).all()):
            raise ValueError("Invalid primary fusion forward")
        rank_ck = checkpoints["event_ranker"]
        if rank_ck["feature_names"] != list(CANDIDATE_FEATURE_NAMES):
            raise ValueError("Event feature order differs from trained weights")
        ranker = EventRanker(**rank_ck["model_config"]).to(device).eval()
        ranker.load_state_dict(rank_ck["state_dict"], strict=True)
        scores = model_scores(ranker(torch.zeros((2, len(CANDIDATE_FEATURE_NAMES)), device=device)), rank_ck.get("quality_power", .25))
        if scores.shape != (2,) or not bool(torch.isfinite(scores).all()):
            raise ValueError("Invalid event ranker forward")
        return dict(passed=True, weight_hashes_verified=True, safe_weight_loading=True,
                    trained_components=list(COMPONENT_FILES), forward_checks=5, device=str(device))
    finally:
        torch.set_num_threads(previous_threads)


def _manifest(value):
    if isinstance(value, dict):
        return dict(value)
    path = Path(value).expanduser().resolve()
    if path.is_dir():
        path = path / "manifest.json"
    data = _read(path)
    data["root"] = str(path.parent)
    return data


def _prepared(value):
    if isinstance(value, dict):
        return dict(value)
    path = Path(value).expanduser().resolve()
    if path.is_dir():
        path = path / "prepared.json"
    data = _read(path)
    for key in ("imu_manifest", "native_root", "ppg_cache_root"):
        if key in data and not Path(data[key]).is_absolute():
            data[key] = str((path.parent / data[key]).resolve())
    return data


class _SignalWindows:
    """Read the four numerical IMU fields, independent of training annotations."""
    def __init__(self, manifest, person, normalization, max_open=3):
        self.root = Path(manifest["root"])
        self.records = [(i, r) for i, r in enumerate(manifest["records"])
                        if str(r["person"]) == str(person)]
        self.mean = np.asarray(normalization["mean"], np.float32)
        self.std = np.asarray(normalization["std"], np.float32)
        if self.mean.shape != (12,) or self.std.shape != (12,) or not np.isfinite(self.mean).all() or not np.isfinite(self.std).all() or np.any(self.std <= 0):
            raise ValueError("IMU normalization must contain twelve finite scales")
        self.open = OrderedDict()
        self.max_open = int(max_open)

    def arrays(self, ri):
        if ri not in self.open:
            _, row = self.records[ri]
            folder = self.root / row["path"]
            self.open[ri] = {k: np.load(folder / (k + ".npy"), mmap_mode="r", allow_pickle=False)
                            for k in ("imu", "point_mask", "frame_quality", "segment_ids")}
        self.open.move_to_end(ri)
        while len(self.open) > self.max_open:
            _, arrays = self.open.popitem(last=False)
            for a in arrays.values():
                if getattr(a, "_mmap", None) is not None:
                    a._mmap.close()
        return self.open[ri]

    def window(self, ri, first, frames):
        _, row = self.records[ri]
        source = self.arrays(ri)
        lo, hi = max(0, first), min(int(row["frame_count"]), first + frames)
        where = slice(lo - first, hi - first)
        values = dict(imu=np.zeros((frames, 12, POINTS), np.float32),
                      point_mask=np.zeros((frames, POINTS), bool),
                      frame_quality=np.zeros((frames, 8), np.float32),
                      segment_ids=np.full(frames, -1, np.int64))
        if hi > lo:
            for name in values:
                values[name][where] = source[name][lo:hi]
        x = (values["imu"] - self.mean[None, :, None]) / self.std[None, :, None]
        values["imu"] = np.where(values["point_mask"][:, None], x, 0.).astype(np.float32)
        return {k: torch.from_numpy(a.copy()) for k, a in values.items()}

    def close(self):
        for arrays in self.open.values():
            for a in arrays.values():
                if getattr(a, "_mmap", None) is not None:
                    a._mmap.close()
        self.open.clear()


@torch.inference_mode()
def infer_imu_person(manifest, checkpoint, person, *, contextual=False,
                     device="cpu", batch_size=8, amp=False):
    """Run one trained IMU component on all recordings belonging to a person."""
    if isinstance(batch_size, bool) or int(batch_size) != batch_size or int(batch_size) < 1:
        raise ValueError("IMU batch size must be a positive integer")
    manifest, ck = _manifest(manifest), _checkpoint(checkpoint)
    if contextual:
        from context_encoder.model import IMUV8Model, ModelConfig
        from event_detection.prepare import IMUAccumulator
        model = IMUV8Model(ModelConfig(**ck["model_config"]))
    else:
        from motion_encoder.model import IMUV6Model, ModelConfig
        from primary_fusion.prepare import FrameAccumulator
        model = IMUV6Model(ModelConfig(**ck["model_config"]))
    model.load_state_dict(ck["state_dict"], strict=True)
    device = torch.device(device)
    model = model.to(device).eval().requires_grad_(False)
    windows = _SignalWindows(manifest, person, ck["normalization"])
    starts = [int(row["start_tick"]) + np.arange(int(row["frame_count"]), dtype=np.int64) * POINTS
              for _, row in windows.records]
    clock = np.concatenate(starts) if starts else np.empty(0, np.int64)
    acc = IMUAccumulator(clock) if contextual else FrameAccumulator(clock)
    radius, core = int(model.temporal_radius), 16
    requests = [(ri, start, min(core, int(row["frame_count"]) - start))
                for ri, (_, row) in enumerate(windows.records)
                for start in range(0, int(row["frame_count"]), core)]
    amp = bool(amp and device.type == "cuda")
    try:
        for first in range(0, len(requests), int(batch_size)):
            own = requests[first:first + int(batch_size)]
            rows = [windows.window(ri, start - radius, core + 2 * radius)
                    for ri, start, count in own]
            batch = {k: torch.stack([row[k] for row in rows]).to(device)
                     for k in ("imu", "point_mask", "frame_quality", "segment_ids")}
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
                out = model(batch["imu"], batch["point_mask"], batch["frame_quality"], batch["segment_ids"])
            for j, (ri, start, count) in enumerate(own):
                global_index, record = windows.records[ri]
                sl = slice(radius, radius + count)
                ticks = int(record["start_tick"]) + np.arange(start, start + count, dtype=np.int64) * POINTS
                valid = out["valid"][j, sl].cpu().numpy()
                mask = batch["point_mask"][j, sl].cpu().numpy() & valid[:, None]
                local_ids = batch["segment_ids"][j, sl].cpu().numpy()
                sid = np.where(local_ids >= 0, (global_index + 1) * 1_000_000_000 + local_ids, -1)
                probability = out["probability"][j, sl].float().cpu().numpy()
                quality = batch["frame_quality"][j, sl].float().cpu().numpy()
                if contextual:
                    row = dict(frame_start_tick=ticks, point_mask=mask, valid=valid,
                        probability=probability,
                        role_probability=out["role_probability"][j, sl].float().cpu().numpy(),
                        boundary_probability=out["boundary_logits"][j, sl].sigmoid().float().cpu().numpy(),
                        frame_quality=quality, segment_ids=sid,
                        local_probability=out["local_logits"][j, sl].sigmoid().float().cpu().numpy(),
                        context_probability=out["context_logits"][j, sl].sigmoid().float().cpu().numpy())
                    acc.add(row, global_index)
                else:
                    acc.add(ticks, probability, out["embedding"][j, sl].float().cpu().numpy(),
                            mask, quality, out["logits"][j, sl].float().softmax(-1).cpu().numpy(), sid)
        return acc.finish()
    finally:
        windows.close()
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()


@torch.inference_mode()
def infer_ppg_person(cache_root, checkpoint, person, *, device="cpu"):
    """Infer continuous PPG records with the trained feature and spectral experts."""
    from signals.optical.data import load_records
    from signals.optical.cache_io import load_spectrogram
    from signals.optical.adapter import PPGV9RuntimeAdapter
    record_file = Path(cache_root) / "records.jsonl"
    if record_file.is_file() and not record_file.read_text(encoding="utf-8").strip():
        return []
    records = [r for r in load_records(cache_root) if str(r["subject_id"]) == str(person)]
    if not records:
        return []
    device = torch.device(device)
    model = PPGV9RuntimeAdapter(_checkpoint(checkpoint)).to(device).eval().requires_grad_(False)
    results = []
    try:
        for rec in records:
            with np.load(rec["_path"], allow_pickle=False) as z:
                arrays = {k: z[k].copy() for k in ("raw_mask", "quality", "channel_ids", "feature_values", "feature_mask", "frame_start_ms", "spectrogram_mask")}
            spectra = load_spectrogram(rec)
            n = len(arrays["frame_start_ms"])
            outputs = defaultdict(list)
            for first in range(0, n, 32):
                count = min(32, n - first)
                begin, length, halo = first - 32, 96, 32
                lo, hi = max(0, begin), min(n, begin + length)
                where = slice(lo - begin, hi - begin)
                batch = {}
                for old, new in (("raw_mask", "raw_mask"), ("quality", "quality"), ("channel_ids", "channel_ids"),
                                 ("feature_values", "features"), ("feature_mask", "feature_mask"), ("spectrogram_mask", "spectrogram_mask")):
                    values = arrays[old]
                    value = np.zeros((length, *values.shape[1:]), values.dtype)
                    value[where] = values[lo:hi]
                    batch[new] = torch.from_numpy(value)[None].to(device)
                value = np.zeros((length, 3, 37, 23), np.float32)
                value[where] = spectra[lo:hi]
                batch["spectrogram"] = torch.from_numpy(value)[None].to(device)
                batch["raw"] = torch.zeros((1, length, 3, 1), device=device)
                valid = torch.zeros((1, length), device=device)
                valid[:, where] = 1
                batch["frame_valid"] = valid
                ids = torch.full((1, length), -1, device=device, dtype=torch.long)
                ids[:, where] = 0
                batch["segment_ids"] = ids
                out = model(**batch)
                sl = slice(halo, halo + count)
                get = lambda x: x[0, sl].detach().float().cpu().numpy()
                for name, key in (("ppg_logits", "logits"), ("ppg_start_logits", "start_logits"),
                                  ("ppg_end_logits", "end_logits"), ("ppg_embedding", "embedding")):
                    outputs[name].append(get(out[key]))
                outputs["ppg_support"].append(out["prediction_valid"][0, sl].cpu().numpy().astype(bool))
                outputs["ppg_quality_raw_max"].append(get(batch["quality"].float().amax(-1)))
                outputs["ppg_quality_expert"].append(get((out["expert_reliability"].float() * out["expert_fusion_weights"].float()).sum(-1)))
                outputs["raw_coverage"].append(get(batch["raw_mask"].float().mean((-1, -2))))
                outputs["spectral_coverage"].append(get(batch["spectrogram_mask"].float().mean((-1, -2))))
                outputs["feature_coverage"].append(get(batch["feature_mask"].float().mean(-1)))
            data = {k: np.concatenate(v) for k, v in outputs.items()}
            data["ppg_quality"] = data["ppg_quality_expert"]
            data["wave_coverage"] = data["spectral_coverage"]
            data["frame_start_tick"] = arrays["frame_start_ms"].astype(np.int64) * 15 // 1000
            results.append(dict(person=str(person), recording_id=str(rec["recording_id"]),
                                continuous_id=str(rec["continuous_id"]), arrays=data))
            if getattr(spectra, "_mmap", None) is not None:
                spectra._mmap.close()
        return results
    finally:
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


def _join_ppg(starts, records):
    """Join continuous expert predictions to a session's absolute point clock."""
    names = ("ppg_logits", "ppg_start_logits", "ppg_end_logits", "ppg_quality", "wave_coverage", "feature_coverage",
             "ppg_quality_raw_max", "ppg_quality_expert", "raw_coverage", "spectral_coverage")
    n = len(starts)
    sums = {k: np.zeros(n, np.float64) for k in names}
    embedding, mass, sid = np.zeros((n, 64), np.float64), np.zeros(n, np.int32), np.full(n, -1, np.int64)
    for ri, rec in enumerate(records):
        data = rec["arrays"]
        clock = data["frame_start_tick"]
        ix = np.searchsorted(starts, clock)
        inside = ix < n
        inside[inside] &= starts[ix[inside]] == clock[inside]
        valid = inside & data["ppg_support"].astype(bool)
        dest = ix[valid]
        for name in names:
            value = data[name][valid].astype(np.float64)
            if name in ("ppg_logits", "ppg_start_logits", "ppg_end_logits"):
                value = 1 / (1 + np.exp(-np.clip(value, -60, 60)))
            sums[name][dest] += value
        embedding[dest] += data["ppg_embedding"][valid]
        mass[dest] += 1
        sid[dest] = np.where(sid[dest] < 0, ri + 1, np.minimum(sid[dest], ri + 1))
    out = {k: (v / np.maximum(mass, 1)).astype(np.float32) for k, v in sums.items()}
    for name in ("ppg_logits", "ppg_start_logits", "ppg_end_logits"):
        p = np.clip(out[name], 1e-6, 1 - 1e-6)
        out[name] = np.where(mass > 0, np.log(p) - np.log1p(-p), 0).astype(np.float32)
    out["ppg_embedding"] = (embedding / np.maximum(mass, 1)[:, None]).astype(np.float32)
    out["ppg_support"] = mass > 0
    return out, sid


def _native_session(root, entry, include_labels):
    from event_detection.prepare import native_array
    ticks = native_array(root, entry, "imu", "ticks").astype(np.int64)
    if not len(ticks) or np.any(np.diff(ticks) != 1):
        raise ValueError("Each native session must contain its complete physical grid")
    origin = int(ticks[0]) // POINTS * POINTS
    frames = (int(ticks[-1]) - origin) // POINTS + 1
    arrays = {"ticks": origin + np.arange(frames * POINTS, dtype=np.int64).reshape(frames, POINTS)}
    positions = ticks - origin
    observed = np.zeros(frames * POINTS, bool)
    observed[positions] = native_array(root, entry, "target", "observed").astype(bool)
    arrays["physical_mask"] = observed.reshape(frames, POINTS)
    if include_labels:
        arrays["observed"] = arrays["physical_mask"].copy()
        for name, key, dtype in (("known", "supervision_all", bool), ("target", "target_all", np.float32)):
            values = np.zeros(frames * POINTS, dtype)
            values[positions] = native_array(root, entry, "target", key)
            arrays[name] = values.reshape(frames, POINTS)
    return arrays


def build_sessions(prepared, *, primary_checkpoint=None, context_checkpoint=None,
                   ppg_checkpoint, people=None, device="cpu", batch_size=8,
                   amp=False, include_labels=False):
    """Yield aligned signal evidence, optionally with labels for training only.

    Each result contains the contextual IMU/PPG numerical arrays and a separate
    ``primary_evidence`` dictionary for the primary probability-fusion head.
    Either IMU component may be omitted while preparing its training stage.
    """
    from event_detection.prepare import join_imu as join_context, majority_sid, fused_segments
    from primary_fusion.prepare import join_imu as join_primary
    cfg = _prepared(prepared)
    manifest = _manifest(cfg["imu_manifest"])
    native_root = Path(cfg["native_root"])
    entries = _read(native_root / "manifest.json")["sessions"]
    available = {str(e["person"]) for e in entries}
    people = sorted(available if people is None else {str(p) for p in people})
    if set(people) - available:
        raise ValueError("Requested people are absent from the prepared signals")
    primary_ck = _checkpoint(primary_checkpoint) if primary_checkpoint is not None else None
    context_ck = _checkpoint(context_checkpoint) if context_checkpoint is not None else None
    ppg_ck = _checkpoint(ppg_checkpoint)
    for person in people:
        primary = infer_imu_person(manifest, primary_ck, person, device=device, batch_size=batch_size, amp=amp) if primary_ck is not None else None
        context = infer_imu_person(manifest, context_ck, person, contextual=True, device=device, batch_size=batch_size, amp=amp) if context_ck is not None else None
        records = infer_ppg_person(cfg["ppg_cache_root"], ppg_ck, person, device=device)
        by_record = defaultdict(list)
        for rec in records:
            by_record[rec["recording_id"]].append(rec)
        for si, entry in enumerate(entries):
            if str(entry["person"]) != person:
                continue
            arrays = _native_session(native_root, entry, include_labels)
            starts = arrays["ticks"][:, 0]
            recording_id = str(entry.get("recording_id", Path(entry["path"]).name))
            ppg, p_sid = _join_ppg(starts, by_record.get(recording_id, []))
            arrays.update(ppg)
            arrays["ppg_support"] &= arrays["physical_mask"].any(-1)
            arrays["ppg_segment_ids"] = p_sid
            if context is not None:
                arrays.update(join_context(starts, context))
                arrays["imu_support"] &= arrays["physical_mask"]
                imu_sid = majority_sid(arrays["imu_segment_ids"], arrays["imu_support"])
            else:
                n = len(starts)
                arrays.update(imu_logits=np.zeros((n, POINTS), np.float32), imu_support=np.zeros((n, POINTS), bool),
                    imu_segment_ids=np.full((n, POINTS), -1, np.int64), imu_quality=np.zeros((n, 8), np.float32),
                    imu_role_prob=np.zeros((n, 2), np.float32), imu_boundary_logits=np.zeros((n, 2), np.float32),
                    imu_local_logits=np.zeros(n, np.float32), imu_context_logits=np.zeros(n, np.float32))
                imu_sid = np.full(n, -1, np.int64)
            arrays["segment_ids"] = fused_segments(starts, imu_sid, p_sid, arrays["imu_support"], arrays["ppg_support"])
            arrays["metadata"] = dict(person=person, recording_id=recording_id, session_index=si)
            if primary is not None:
                own = dict(ticks=arrays["ticks"], **join_primary(arrays, primary))
                own["imu_support"] &= arrays["physical_mask"]
                for key in ("ppg_logits", "ppg_embedding", "ppg_support", "feature_coverage"):
                    own[key] = arrays[key]
                own["ppg_quality"] = arrays["ppg_quality_raw_max"]
                own["wave_coverage"] = arrays["raw_coverage"]
                own["physical_mask"] = arrays["physical_mask"]
                own["metadata"] = dict(arrays["metadata"])
                if include_labels:
                    for key in ("target", "known", "observed"):
                        own[key] = arrays[key]
                arrays["primary_evidence"] = own
            yield arrays
        del primary, context, records
        gc.collect()


@torch.inference_mode()
def primary_probabilities(evidence, checkpoint, *, device="cpu", batch_size=256):
    """Evaluate the trained per-frame primary fusion gate."""
    if isinstance(batch_size, bool) or int(batch_size) != batch_size or int(batch_size) < 1:
        raise ValueError("Primary-fusion batch size must be a positive integer")
    from primary_fusion.model import FusionModel
    from primary_fusion.predict import FIELDS
    ck = _checkpoint(checkpoint)
    device = torch.device(device)
    model = FusionModel(dropout=float(ck.get("config", {}).get("dropout", .1)))
    model.load_state_dict(ck["state_dict"], strict=True)
    model = model.to(device).eval().requires_grad_(False)
    outputs = defaultdict(list)
    frames = len(evidence["ticks"])
    for first in range(0, frames, batch_size):
        batch = {k: torch.from_numpy(np.array(evidence[k][first:first + batch_size], copy=True))[None].to(device) for k in FIELDS}
        out = model(**batch)
        for name in ("probability", "supported", "frame_weights"):
            outputs[name].append(out[name][0].detach().cpu().numpy())
    return {k: np.concatenate(v) for k, v in outputs.items()}


def primary_events(session, checkpoint, *, decoder=None, device="cpu"):
    from signals.competition.decoder import decode
    evidence = session.get("primary_evidence", session)
    out = primary_probabilities(evidence, checkpoint, device=device)
    support = (out["supported"] & evidence["physical_mask"]).reshape(-1)
    sid = evidence["imu_segment_ids"].reshape(-1)
    ppg = np.broadcast_to(evidence["ppg_support"][:, None], evidence["ticks"].shape).reshape(-1)
    cut = support[1:] & support[:-1] & (sid[1:] >= 0) & (sid[:-1] >= 0) & (sid[1:] != sid[:-1]) & ~(ppg[1:] & ppg[:-1])
    edges = np.r_[0, np.flatnonzero(cut) + 1, len(support)]
    ticks = evidence["ticks"].reshape(-1)
    probability = out["probability"].reshape(-1)
    rule = DEFAULT_DECODER | dict(decoder or {})
    person = evidence["metadata"]["person"]
    result = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        result.extend(decode(ticks[lo:hi], probability[lo:hi], support[lo:hi], participant_key=person, **rule))
    return result


def predict_primary_sessions(sessions, primary_fusion_checkpoint, decoder=None, *, device="cpu"):
    result = []
    for session in sessions:
        result.extend(primary_events(session, primary_fusion_checkpoint, decoder=decoder, device=device))
    return sorted(result, key=lambda e: (str(e["participant_key"]), e["start_ms"], e["end_ms"]))


def predict_from_sessions(sessions, event_ranker_checkpoint, primary_fusion_checkpoint,
                          decoder=None, *, device="cpu"):
    """Apply both trained event paths to numerical expert evidence."""
    from event_detection.events import generate_candidates, select_candidates, merge_anchor_rescue
    from event_detection.ranker import predict_ranker
    payload = _checkpoint(event_ranker_checkpoint)
    anchor, rescue = [], []
    for session in sessions:
        anchor.extend(primary_events(session, primary_fusion_checkpoint, decoder=decoder, device=device))
        candidates = generate_candidates(session, rule=payload["candidate_rule"])
        scores = predict_ranker(session, candidates, payload, device="cpu")
        rescue.extend(select_candidates(candidates, scores, threshold=payload["threshold"],
            nms_iou=payload["nms_iou"], nms_ios=payload["nms_ios"]))
    return merge_anchor_rescue(anchor, rescue, nms_iou=payload["nms_iou"], nms_ios=payload["nms_ios"])


def write_predictions(events, destination):
    """Validate a complete competition CSV before atomically publishing it."""
    from signals.competition.official.output import validate_result
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="", suffix=".csv",
                prefix="." + destination.name + ".", dir=destination.parent, delete=False) as stream:
            temporary = Path(stream.name)
            writer = csv.writer(stream)
            writer.writerow(["externalid", "beforeTime", "afterTime"])
            count = 0
            for event in sorted(events, key=lambda e: (str(e["participant_key"]), e["start_ms"], e["end_ms"])):
                # Preserve input values for the official strict type/format
                # checks; coercing float timestamps to int would hide mistakes.
                writer.writerow([event["participant_key"], event["start_ms"], event["end_ms"]])
                count += 1
            stream.flush()
            os.fsync(stream.fileno())
        if validate_result(temporary) != count:
            raise RuntimeError("Competition CSV row count changed during validation")
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return destination


def predict(prepared, weights_dir, output_csv, *, people=None, device="cpu",
            batch_size=8, amp=None, decoder=None):
    """Run prepared signals through every delivered trained component."""
    weights = Path(weights_dir).expanduser().resolve()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if amp is None:
        amp = bool(torch.device(device).type == 'cuda' and torch.cuda.is_bf16_supported())
    verified = verify_weights(weights)
    required = ("imu_primary.pt", "imu_context.pt", "ppg.pt", "primary_fusion.pt", "event_ranker.pt")
    for name in required:
        if not (weights / name).is_file():
            raise FileNotFoundError(weights / name)
    sessions = build_sessions(prepared, primary_checkpoint=weights / "imu_primary.pt",
        context_checkpoint=weights / "imu_context.pt", ppg_checkpoint=weights / "ppg.pt",
        people=people, device=device, batch_size=batch_size, amp=amp, include_labels=False)
    events = predict_from_sessions(sessions, weights / "event_ranker.pt", weights / "primary_fusion.pt",
                                   decoder=verified["decoder"] if decoder is None else decoder, device=device)
    write_predictions(events, output_csv)
    return events
