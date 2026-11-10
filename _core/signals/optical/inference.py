"""Core-only sequence inference, overlap deduplication, and event-level reporting."""
from __future__ import annotations
import inspect
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from .common import model_inputs, input_support, write_json, prepare_device_batch
from .data import SequenceDataset
from .events import events_from_frames, match_events, truth_from_records, events_to_dataframe, PreparedFrameDecoder


@torch.inference_mode()
def infer_frames(model, records, scaler, config, device, batch_size=None, *,
                 progress_every=50, progress_callback=None, loader=None):
    """Return the same frame table; emit optional progress separately on stdout."""
    inference = config.get("inference", {})
    effective_batch_size = (inference.get("batch_size", config["training"]["batch_size"])
                            if batch_size is None else batch_size)
    if effective_batch_size < 1 or progress_every < 1:
        raise ValueError("Inference batch_size and progress_every must be positive")
    sequence_frames, core_frames = config["sequence_frames"], config["core_frames"]
    if "core_frames" in inference:
        core_frames = int(inference["core_frames"])
        halo = int(inference.get("halo_frames", 32))
        if core_frames < 1 or halo < 32 or config["model"].get("causal", False):
            raise ValueError("Expanded V3/V4 inference requires noncausal model and >=32-frame halo")
        if str(config["model"].get("version", "v1")).lower() not in ("v3", "v4", "v5", "v6", "v7", "v8", "v9"):
            raise ValueError("Expanded inference is validated only for the V3/V4 local TCN")
        required_halo = max(sum(config["model"].get("dilations", (1, 2, 4, 8, 16))),
                            sum(config["model"].get("short_dilations", (1, 2, 4))))
        if halo < required_halo:
            raise ValueError("Inference halo does not cover the configured temporal receptive field")
        sequence_frames = core_frames + 2 * halo
    if loader is None:
        from .loader_v9 import inference_loader
        loader=inference_loader(records,scaler,config,device)
        if batch_size is not None:
            from copy import deepcopy
            adjusted=deepcopy(config);adjusted['inference']['batch_size']=batch_size
            loader=inference_loader(records,scaler,adjusted,device)
    dataset=loader.dataset
    from .training_v9 import precision_policy
    use_amp,amp_dtype,_=precision_policy(config,device)
    export_branches = str(config["model"].get("version", "v1")).lower() in ("v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9")
    v8 = config["model"].get("version") in ("v8", "v9")
    model.eval(); rows = []; start = time.monotonic()
    for batch_index, batch in enumerate(loader, start=1):
        device_batch=prepare_device_batch(batch,device)
        with torch.autocast(device_type=device.type,dtype=amp_dtype,enabled=use_amp):
            out = model(**model_inputs(device_batch, device))
        p = out["logits"].float().sigmoid().cpu().numpy()
        sp, ep = out["start_logits"].float().sigmoid().cpu().numpy(), out["end_logits"].float().sigmoid().cpu().numpy()
        core = (batch["core_mask"] * batch["frame_valid"]).numpy() > 0
        support = input_support(batch, config["model"].get("input_mode", "fused")).numpy()
        q = batch["quality"].max(-1).values.numpy()
        channel_weights = out["channel_weights"].float().cpu().numpy()
        branch = None
        if export_branches:
            # These are diagnostics/fusion outputs, never label-derived inputs.
            feature_support = out["feature_support"] if "feature_support" in out else input_support(batch, "features").to(device)
            raw_support = out["raw_support"] if "raw_support" in out else input_support(batch, "raw").to(device)
            if config["model"].get("input_mode", "fused") == "features":
                raw_support = torch.zeros_like(raw_support, dtype=torch.bool)
            elif config["model"].get("input_mode", "fused") == "raw":
                feature_support = torch.zeros_like(feature_support, dtype=torch.bool)
            # The encoder may reject an input with only sparse waveform
            # samples even when common.input_support saw a nonzero mask.
            if v8:
                support = out["prediction_valid"].detach().float().cpu().numpy().astype(bool)
                reliability = out["expert_reliability"].detach().float().cpu().numpy()
                experts_supported = out["expert_support"].detach().float().cpu().numpy().astype(bool)
                q = np.where(experts_supported, reliability, 0.).max(-1)
            else:
                support &= (feature_support | raw_support).detach().float().cpu().numpy().astype(bool)
            branch = {
                "feature_logit": out["feature_logits"].detach().float().cpu().numpy(),
                "raw_logit": out["raw_logits"].detach().float().cpu().numpy(),
                "feature_prob": out["feature_logits"].sigmoid().detach().float().cpu().numpy(),
                "raw_prob": out["raw_logits"].sigmoid().detach().float().cpu().numpy(),
                "raw_gate": out["raw_gate"].detach().float().cpu().numpy(),
                "raw_residual": out["raw_residual"].detach().float().cpu().numpy(),
                "feature_supported": feature_support.detach().float().cpu().numpy().astype(bool),
                "raw_supported": raw_support.detach().float().cpu().numpy().astype(bool),
            }
        v8_arrays = None
        if v8:
            v8_arrays = {key: out[key].detach().float().cpu().numpy() for key in (
                "expert_probabilities", "expert_start_probabilities", "expert_end_probabilities",
                "expert_support", "expert_reliability")}
        for bi in range(len(p)):
            rec = records[int(batch["record_index"][bi])]
            for ti in np.flatnonzero(core[bi]):
                usable = bool(support[bi, ti])
                row = {"subject_id": rec["subject_id"], "recording_id": rec["recording_id"],
                             "continuous_id": rec["continuous_id"],
                             "start_ms": int(batch["frame_start_ms"][bi, ti]),
                             "end_ms": int(batch["frame_end_ms"][bi, ti]),
                             "eating_prob": float(p[bi, ti]) if usable else 0.,
                             "start_prob": float(sp[bi, ti]) if usable else 0.,
                             "end_prob": float(ep[bi, ti]) if usable else 0.,
                             "quality": float(q[bi, ti]) if usable else 0., "supported": usable,
                             "target": float(batch["target"][bi, ti]),
                             "label_weight": float(batch["label_weight"][bi, ti]),
                             "channels": ";".join(str(int(x)) for x in batch["channel_ids"][bi, ti]),
                             "channel_weights": ";".join(f"{x:.5f}" for x in channel_weights[bi, ti])}
                if "reliability_context" in batch:
                    row["hr_history_coverage"] = float(batch["reliability_context"][bi, ti, 7])
                    row["hr_history_available"] = bool(batch["reliability_context"][bi, ti, 8] > 0)
                if branch is not None:
                    feature_usable = usable and bool(branch["feature_supported"][bi, ti])
                    raw_usable = usable and bool(branch["raw_supported"][bi, ti])
                    row.update({"feature_logit": float(branch["feature_logit"][bi, ti]) if feature_usable else np.nan,
                                "raw_logit": float(branch["raw_logit"][bi, ti]) if raw_usable else np.nan,
                                "feature_prob": float(branch["feature_prob"][bi, ti]) if feature_usable else 0.,
                                "raw_prob": float(branch["raw_prob"][bi, ti]) if raw_usable else 0.,
                                "raw_gate": float(branch["raw_gate"][bi, ti]) if raw_usable else 0.,
                                "raw_residual": float(branch["raw_residual"][bi, ti]) if raw_usable else 0.,
                                "feature_supported": feature_usable, "raw_supported": raw_usable})
                if v8_arrays is not None:
                    row["quality_source"] = "max_supported_expert_reliability_v8"
                    for j, expert in enumerate(("feature", "raw")):
                        ok = bool(v8_arrays["expert_support"][bi, ti, j])
                        row[expert + "_supported"] = ok
                        row[expert + "_logit"] = float(branch[expert + "_logit"][bi, ti]) if ok else np.nan
                        for source, suffix in (("expert_probabilities", "_prob"),
                                               ("expert_start_probabilities", "_start_prob"),
                                               ("expert_end_probabilities", "_end_prob")):
                            row[expert + suffix] = float(v8_arrays[source][bi, ti, j]) if ok else np.nan
                        row[expert + "_reliability"] = float(v8_arrays["expert_reliability"][bi, ti, j]) if ok else 0.
                rows.append(row)
        if batch_index == 1 or batch_index % progress_every == 0 or batch_index == len(loader):
            progress = {"phase": "inference", "batch": batch_index, "total_batches": len(loader),
                        "frames": len(rows), "elapsed_seconds": time.monotonic() - start}
            if progress_callback is not None:
                progress_callback(progress)
            else:
                print(json.dumps(progress), flush=True)
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("No inference frames")
    # Epoch-aligned 30s frames make duplicated source records exactly identifiable.
    frame = frame.sort_values(["subject_id", "start_ms", "quality", "recording_id"],
                              ascending=[True, True, False, True], kind="stable")
    duplicate_count = sum(len(m) - int(m.sum()) for m in dataset.frame_masks)
    duplicate_count += int(frame.duplicated(["subject_id", "start_ms"]).sum())
    frame = frame.drop_duplicates(["subject_id", "start_ms"], keep="first").reset_index(drop=True)
    frame.attrs["duplicate_source_frames_removed"] = duplicate_count
    if v8:
        # Use the identical probability arithmetic for validation/deployment
        # exports, avoiding float32-logit round trips near decoder thresholds.
        from .event_detection import fuse_frame_table
        frame = fuse_frame_table(frame, model.fusion_weights.detach().cpu().tolist(),
                                 strict_selected=True)
    return frame


def decode_frame_table(frame, decoder):
    if str(decoder.get("search_version", "v1")).lower().startswith(("v7", "v8", "v9")):
        return FrameDecodeCache(frame, physical_continuity=True).decode(decoder)
    if str(decoder.get("search_version", "v1")).lower().startswith("v6"):
        # V6 internal validation uses the same actual-gap/support barriers as deployment.
        return FrameDecodeCache(frame, segment_groups=True).decode(decoder)
    accepted = set(inspect.signature(events_from_frames).parameters) - {"_prepared"}
    params = {k: v for k, v in decoder.items() if k in accepted}
    events = []
    for subject, group in frame.groupby("subject_id", sort=True):
        group = group.sort_values("start_ms")
        events.extend(events_from_frames(group.start_ms.to_numpy(), group.end_ms.to_numpy(),
                       group.eating_prob.to_numpy(), subject_id=str(subject),
                       start_probs=group.start_prob.to_numpy(), end_probs=group.end_prob.to_numpy(),
                       quality=group.quality.to_numpy(), **params))
    return events


class FrameDecodeCache:
    """One immutable probability-table snapshot scoped to a single search.

    segment_groups=False reproduces inner validation's subject-only grouping.
    True reproduces deployment/OOF calibration's continuous-segment grouping,
    including its explicit unsupported-frame barrier and final event sorting.
    Never share an instance across checkpoints, epochs or changed frame tables.
    """
    def __init__(self, frame, *, segment_groups=False, physical_continuity=False):
        self.physical_continuity = bool(physical_continuity)
        if physical_continuity and segment_groups:
            raise ValueError("Physical continuity and source-ID grouping are mutually exclusive")
        self.segment_groups = bool(segment_groups)
        if physical_continuity:
            if "supported" not in frame:
                raise ValueError("V7 physical decoding requires explicit support barriers")
            if frame.duplicated(["subject_id", "start_ms"]).any():
                raise ValueError("V7 event decoding requires canonical unique subject/timestamp frames")
        self.groups = []
        keys = ["subject_id", "recording_id", "continuous_id"] if segment_groups else "subject_id"
        for key, group in frame.groupby(keys, sort=True):
            group = group.sort_values("start_ms")
            prob = group.eating_prob.to_numpy(dtype=float, copy=True)
            if segment_groups or physical_continuity:
                prob[~group.supported.to_numpy(dtype=bool)] = np.nan
            subject = key[0] if segment_groups else key
            self.groups.append(PreparedFrameDecoder(
                group.start_ms.to_numpy(), group.end_ms.to_numpy(), prob, str(subject),
                start_probs=group.start_prob.to_numpy(), end_probs=group.end_prob.to_numpy(),
                quality=group.quality.to_numpy()))

    def decode(self, decoder):
        accepted = set(inspect.signature(events_from_frames).parameters) - {"_prepared"}
        params = {key: value for key, value in decoder.items() if key in accepted}
        events = [event for group in self.groups for event in group.decode(**params)]
        if self.segment_groups or self.physical_continuity:
            events.sort(key=lambda event: (str(event.subject_id), event.start_ms, event.end_ms))
        return events


def truth_in_coverage(records, frame):
    # Keep original event boundaries. Partial acquisition is not a new, shorter meal.
    candidates = truth_from_records(records)
    selected, coverage = [], []
    for ev in candidates:
        g = frame[(frame.subject_id == ev.subject_id) & (frame.end_ms > ev.start_ms) & (frame.start_ms < ev.end_ms)]
        if g.empty:
            continue
        observed = np.maximum(0, np.minimum(g.end_ms.to_numpy(), ev.end_ms) - np.maximum(g.start_ms.to_numpy(), ev.start_ms))
        fraction = float(observed.sum() / (ev.end_ms - ev.start_ms))
        selected.append(ev)
        coverage.append({"subject_id": ev.subject_id, "meal_id": ev.meal_id,
                         "start_ms": ev.start_ms, "end_ms": ev.end_ms, "observed_fraction": fraction})
    return selected, coverage


def event_report(frame, records, decoder, *, prepared=None, prepared_truth=None):
    predicted = (decode_frame_table(frame, decoder) if prepared is None else prepared.decode(decoder))
    truth, coverage = truth_in_coverage(records, frame) if prepared_truth is None else prepared_truth
    result = match_events(predicted, truth, iou_threshold=.25)
    report = dict(result["metrics"])
    report.update({"iou_rule": "strictly greater than 0.25; one-to-one maximum-cardinality matching",
                   "evaluation_scope": "full original meal intervals overlapping cached observation coverage",
                   "observed_frames": len(frame), "unsupported_frames": int((~frame.supported).sum()),
                   "duplicate_source_frames_removed": frame.attrs.get("duplicate_source_frames_removed", 0),
                   "fully_observed_meals_90pct": sum(c["observed_fraction"] >= .9 for c in coverage),
                   "partially_observed_meals": sum(c["observed_fraction"] < .9 for c in coverage)})
    matched_truth = {(ev.subject_id, ev.start_ms, ev.end_ms) for _, ev, _ in result["matched"]}
    scenarios = {}
    for scenario in ("dominant", "non_dominant", "unknown"):
        sub = [e for e in truth if e.scenario == scenario]
        hit = sum((e.subject_id, e.start_ms, e.end_ms) in matched_truth for e in sub)
        scenarios[scenario] = {"true_events": len(sub), "matched_events": hit,
                               "recall": hit / len(sub) if sub else None}
    report["wear_scenario_recall"] = scenarios
    report["wear_scenario_note"] = "Scenario is provided only by positive meal metadata; background FP cannot be honestly assigned by this annotation alone."
    return report, predicted, coverage


def tune_decoder(frame,records,base,*,precompute=True):
    from .selection_v8 import _rules, fixed_rule_panel
    config={'decoder':base}
    if str(base.get('search_version','')).endswith('checkpoint'):
        decoder,metrics,rows,_=fixed_rule_panel(frame,records,config)
        return decoder,metrics,rows
    rows=_rules(frame,records,config,False)
    chosen=max(range(len(rows)),key=lambda i:rows[i]['metrics']['F1'])
    return rows[chosen]['decoder'],rows[chosen]['metrics'],rows


def save_predictions(directory, frame, events, report=None, coverage=None):
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    frame.to_csv(directory / "frame_probabilities.csv.gz", index=False, compression="gzip")
    table = events_to_dataframe(events)
    table.to_csv(directory / "events.csv", index=False, encoding="utf-8-sig")
    write_json(directory / "events.json", table.replace({np.nan: None}).to_dict("records"))
    if report is not None:
        write_json(directory / "evaluation.json", report)
    if coverage is not None:
        write_json(directory / "truth_coverage.json", coverage)

