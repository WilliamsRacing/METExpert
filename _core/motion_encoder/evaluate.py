
from __future__ import annotations

import copy
import ast
import csv
from dataclasses import dataclass
import hashlib
from itertools import product
import json
from pathlib import Path
import sys
from types import ModuleType
from typing import Iterable, Mapping, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment


FIXED_DECODER = dict(threshold=.5, low_threshold=.2, min_duration_seconds=450,
                     merge_gap_seconds=60, short_min_duration_seconds=300,
                     short_high_fraction=.5, short_high_duration_seconds=30,
                     event_high_fraction=0.)
# Predeclared small selection grid; not a test-set search.
DEFAULT_GRID = dict(threshold=(.4, .5, .6), low_threshold=(.2,),
                    min_duration_seconds=(300, 450), merge_gap_seconds=(60,))


_REFERENCES = {}


def _reference(root=None):
    """Load unchanged pure definitions without importing PPG training/HDF5.

    AST selection removes only unrelated top-level imports/definitions, not any
    node inside the original selected functions. Source SHA256 values are kept
    in every report, including the original maximum-cardinality strict matcher.
    """
    root = Path(__file__).resolve().parents[1]
    if root in _REFERENCES:
        return _REFERENCES[root]
    sources = (
        ('signals/multimodal/storage.py', ("fingerprint", "read_jsonl")),
        ('signals/multimodal/labels.py', ("hand", "LabelPolicy")),
        ('signals/competition/annotations.py', ("subject", "intervals", "CompetitionAnnotations")),
        ('signals/competition/decoder.py', ("runs", "decode")),
        ('optical_training/events.py', ("Event", "interval_iou", "_metrics", "match_events")),
        ('event_metrics/evaluate.py', ("predicted_events", "evaluation_scope", "score", "_rank", "select_decoder")),
    )
    name = "_imu_v6_native_contract_" + hashlib.sha256(str(root).encode()).hexdigest()[:12]
    metric = ModuleType(name)
    metric.__dict__.update(np=np, product=product, Path=Path, json=json,
                           hashlib=hashlib, csv=csv, dataclass=dataclass,
                           Sequence=Sequence, linear_sum_assignment=linear_sum_assignment,
                           LABEL_VERSION="unused-native-constructor")
    # dataclasses resolves annotation modules through sys.modules.
    sys.modules[name] = metric
    hashes = {}
    for relative, wanted in sources:
        path = root / relative
        payload = path.read_bytes()
        hashes[relative] = hashlib.sha256(payload).hexdigest()
        tree = ast.parse(payload.decode("utf-8-sig"), filename=str(path))
        definitions = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in wanted]
        if {n.name for n in definitions} != set(wanted):
            raise ValueError(f"Original metric contract definitions missing from {path}")
        pure = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)] + definitions,
                          type_ignores=[])
        ast.fix_missing_locations(pure)
        exec(compile(pure, str(path), "exec"), metric.__dict__)
    metric.source_sha256 = hashes
    metric.reference_root = str(root)
    _REFERENCES[root] = (metric, metric.CompetitionAnnotations)
    return _REFERENCES[root]


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _native_array(root, entry, prefix, name):
    key = f"{prefix}__{name}"
    path = entry.get("reused_arrays", {}).get(key)
    path = Path(path) if path else root / entry["path"] / (key + ".npy")
    if not path.is_absolute():
        path = root / path
    # Evaluation retains many native sessions at once. Plain loading closes the
    # file immediately; retaining one mmap per array can exceed Linux's FD limit.
    return np.load(path, allow_pickle=False)


def _prediction_blocks(predictions, people):
    """Validate all inputs, rejecting rather than silently reading test people."""
    grouped = {p: [] for p in people}
    for supplied in predictions:
        row = dict(supplied)
        person = str(row["person"])
        if person not in grouped:
            raise ValueError(f"Prediction outside the locked evaluation people: {person}")
        starts = np.asarray(row["frame_start_tick"], dtype=np.int64)
        probability = np.asarray(row["probability"], dtype=np.float64)
        valid = np.asarray(row["valid"], dtype=bool)
        if starts.ndim != 1 or probability.shape != starts.shape or valid.shape != starts.shape:
            raise ValueError("Expected aligned one-dimensional starts/probability/valid")
        mask = row.get("point_mask")
        if mask is None:
            # Legal for synthetic/full-support callers. Production should supply
            # the actual mask, otherwise valid necessarily means all 450 points.
            mask = np.broadcast_to(valid[:, None], (len(starts), 450))
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != (len(starts), 450):
            raise ValueError("point_mask must have shape [frames,450]")
        segment = row.get("segment_ids", row.get("segment"))
        segment = (np.full(len(starts), -1, np.int64) if segment is None else
                   np.asarray(segment, dtype=np.int64))
        if segment.shape != starts.shape:
            raise ValueError("segment_ids must have shape [frames]")
        if not np.isfinite(probability[valid]).all() or np.any(
                (probability[valid] < 0) | (probability[valid] > 1)):
            raise ValueError("Valid probabilities must be finite and in [0,1]")
        grouped[person].append(dict(starts=starts, probability=probability,
                                    valid=valid, mask=mask, segment=segment))
    joined = {}
    for person, blocks in grouped.items():
        if not blocks:
            joined[person] = None
            continue
        values = {k: np.concatenate([b[k] for b in blocks], axis=0)
                  for k in blocks[0]}
        order = np.lexsort((values["segment"], values["starts"]))
        joined[person] = {k: v[order] for k, v in values.items()}
    return joined


def _project(ticks, observed, block):
    n = len(ticks)
    sums = np.zeros(n, np.float64)
    counts = np.zeros(n, np.uint32)
    owner = np.full(n, -1, np.int64)
    conflicts = np.zeros(n, bool)
    if n and block is not None:
        # A prediction covers [start,start+450); its origin need not be aligned
        # with the native session origin. Duplicate frames contribute pointwise.
        first = np.searchsorted(block["starts"], int(ticks[0]) - 449, side="left")
        last = np.searchsorted(block["starts"], int(ticks[-1]) + 1, side="left")
        offset = np.arange(450, dtype=np.int64)
        for j in range(first, last):
            if not block["valid"][j]:
                continue
            desired = block["starts"][j] + offset
            ix = np.searchsorted(ticks, desired)
            in_range = ix < n
            good = in_range & block["mask"][j]
            good[in_range] &= ticks[ix[in_range]] == desired[in_range]
            indices = ix[good]
            if not len(indices):
                continue
            sid = int(block["segment"][j])
            existing = counts[indices] > 0
            if sid >= 0:
                conflicts[indices] |= existing & (owner[indices] >= 0) & (owner[indices] != sid)
                owner[indices] = np.where(owner[indices] < 0, sid,
                                          np.minimum(owner[indices], sid))
            sums[indices] += block["probability"][j]
            counts[indices] += 1
    supported = (counts > 0) & observed
    probability = (sums / np.maximum(counts, 1)).astype(np.float32)
    probability[~supported] = 0.
    return probability, supported, owner, dict(
        duplicate_contribution_points=int((counts > 1).sum()),
        conflicting_raw_segment_points=int(conflicts.sum()),
        raw_supported_native_unobserved_points=int(((counts > 0) & ~observed).sum()))


def _role_rows(rows, labels, role):
    """Conditional diagnostics only: keep every eligible background prediction."""
    selected = copy.copy(labels)
    relation = {"same": True, "different": False, "unknown_role": None}[role]
    selected.events = [e for e in labels.events if labels.same_hand(e) is relation]
    out = []
    for source in rows:
        row = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in source.items()}
        include = np.zeros(len(row["ticks"]), bool)
        exclude = include.copy()
        for event in labels.participant_events(row["person"]):
            inside = ((row["ticks"] * 1000 >= int(event["start_epoch_ms"]) * 15) &
                      (row["ticks"] * 1000 < int(event["end_epoch_ms"]) * 15))
            if labels.same_hand(event) is relation:
                include |= inside
            else:
                exclude |= inside
        eligible = ~(exclude & ~include)
        row["known"] &= eligible
        row["supported"] &= eligible
        row["target"] = row["target"] * include
        out.append(row)
    return out, selected


class NativeEvaluator:
    """Fixed native scope for the explicitly allowed people.

    ``predictions`` is a list of blocks with person, frame_start_tick[F],
    probability[F], valid[F], optional globally unique segment_ids[F], and
    point_mask[F,450]. Empty/missing prediction blocks still retain native truth.
    Native arrays are read-only; no PPG signals, routes or checkpoints are loaded.
    """
    def __init__(self, native_root, annotation_root, people: Iterable[str],
                 reference_root=None, forbidden_people=()):
        self.native_root = Path(native_root).resolve()
        self.people = tuple(sorted(set(map(str, people))))
        if not self.people:
            raise ValueError("Explicit nonempty evaluation people are required")
        if set(self.people) & set(forbidden_people):
            raise ValueError("Evaluation people include a forbidden/test identity")
        self.metric, annotation_class = _reference(reference_root)
        self.labels = annotation_class(annotation_root)
        if not self.labels.annotated:
            raise ValueError("Original native annotations are required")
        manifest = _read(self.native_root / "manifest.json")
        if manifest.get("schema") not in {"fast-release-v2-cache-2", "fast-release-v2-cache-3"} or not manifest.get("complete"):
            raise ValueError("Complete native V2/V5 cache required")
        self.entries = [e for e in manifest["sessions"] if e["person"] in self.people]
        available = {e["person"] for e in self.entries}
        if set(self.people) - available:
            raise ValueError(f"Missing native sessions: {sorted(set(self.people)-available)}")
        # Copy only tiny event metadata. The reference's participant_events reads
        # annotations only; excluded people's numerical arrays are never opened.
        self.labels = copy.copy(self.labels)
        self.labels.events = [e for e in self.labels.events if e["participant_key"] in self.people]
        self.labels.unknown = [e for e in self.labels.unknown if e["participant_key"] in self.people]

    def rows(self, predictions):
        grouped = _prediction_blocks(predictions, self.people)
        rows, audit = [], dict(duplicate_contribution_points=0,
                              conflicting_raw_segment_points=0,
                              raw_supported_native_unobserved_points=0)
        for entry in self.entries:
            ticks = np.asarray(_native_array(self.native_root, entry, "imu", "ticks"), np.int64)
            known = np.asarray(_native_array(self.native_root, entry, "target", "supervision_all"), bool)
            observed = np.asarray(_native_array(self.native_root, entry, "target", "observed"), bool)
            target = np.asarray(_native_array(self.native_root, entry, "target", "target_all"), np.float32)
            if any(a.shape != ticks.shape for a in (known, observed, target)):
                raise ValueError("Native ticks/targets must align")
            probability, support, segment, counts = _project(ticks, observed, grouped[entry["person"]])
            for key, value in counts.items():
                audit[key] += value
            # Additional raw physical segment boundaries affect decoding only.
            # Scope construction below still sees all original native samples.
            split = (support[1:] & support[:-1] & (segment[1:] >= 0) &
                     (segment[:-1] >= 0) & (segment[1:] != segment[:-1]))
            cuts = np.r_[0, np.flatnonzero(split) + 1, len(ticks)]
            for a, b in zip(cuts[:-1], cuts[1:]):
                if b > a:
                    rows.append(dict(person=entry["person"], ticks=ticks[a:b],
                                     probability=probability[a:b], supported=support[a:b],
                                     known=known[a:b], observed=observed[a:b], target=target[a:b]))
        audit.update(native_sessions=len(self.entries), evaluation_people=list(self.people),
                     native_observed_points=sum(int(r["observed"].sum()) for r in rows),
                     prediction_supported_observed_points=sum(int(r["supported"].sum()) for r in rows),
                     denominator_source="immutable native known/observed plus full original events",
                     original_metric_sources_sha256=self.metric.source_sha256,
                     raw_support_changes_truth_denominator=False)
        return rows, audit

    def _report(self, rows, rule, audit, diagnostics=True):
        report = dict(all=self.metric.score(rows, self.labels, rule), decoder=dict(rule),
                      support_audit=audit)
        if diagnostics:
            for role in ("same", "different", "unknown_role"):
                scoped, labels = _role_rows(rows, self.labels, role)
                report[role] = self.metric.score(scoped, labels, rule)
            report["per_person"] = {p: self.metric.score(
                [r for r in rows if r["person"] == p], self.labels, rule) for p in self.people}
            report["events"] = self.metric.predicted_events(rows, rule)
            report["role_diagnostics_note"] = (
                "Conditional label-aware diagnostics retain background false positives; "
                "they are not deployable role routing or overall performance.")
        return report

    def evaluate(self, predictions, rule=None, *, diagnostics=True):
        rows, audit = self.rows(predictions)
        return self._report(rows, FIXED_DECODER if rule is None else dict(rule), audit, diagnostics)

    def select_decoder(self, predictions, grid=None, *, fixed=None, diagnostics=True):
        """Use only this evaluator's explicit inner-selection/validation people."""
        rows, audit = self.rows(predictions)
        config = dict(decoder=dict(FIXED_DECODER if fixed is None else fixed),
                      decoder_grid=dict(DEFAULT_GRID if grid is None else grid))
        rule, _ = self.metric.select_decoder(rows, self.labels, config)
        return rule, self._report(rows, rule, audit, diagnostics)


def score_native(predictions, native_root, annotation_root, people, *,
                 rule=None, reference_root=None, forbidden_people=(), diagnostics=True):
    return NativeEvaluator(native_root, annotation_root, people, reference_root,
                           forbidden_people).evaluate(predictions, rule, diagnostics=diagnostics)
