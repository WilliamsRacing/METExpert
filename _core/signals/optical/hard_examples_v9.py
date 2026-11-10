
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict, deque
from pathlib import Path

import numpy as np
from torch.utils.data import Sampler


def _pool_digest(pool):
    payload = json.dumps(pool, sort_keys=True, ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def identity_loss_weights(dataset):
    """Return bounded loss-numerator factors and non-resampled frame counts.

    Only canonical known, confidence-positive core frames enter N_s. The
    independent ``label_weight`` array is not clipped/mutated by these factors.
    """
    counts = Counter()
    by_record = defaultdict(list)
    for ri, start in dataset.items:
        by_record[int(ri)].append(int(start))
    for ri, starts in by_record.items():
        rec = dataset.records[ri]
        with np.load(rec["_path"], allow_pickle=False) as data:
            valid = (np.asarray(data["target"]) >= 0) & (np.asarray(data["label_weight"]) > 0)
        valid &= np.asarray(dataset.frame_masks[ri], bool)
        for start in starts:
            counts[str(rec["subject_id"])] += int(valid[start:start + dataset.core_frames].sum())
    subjects = sorted({str(rec["subject_id"]) for rec in dataset.records})
    total = sum(counts.values())
    factors = {s: float(np.clip(total / (len(subjects) * counts[s]), .5, 2.))
               if counts[s] else .5 for s in subjects}
    return {"weights": factors, "known_valid_core_frames": {s: int(counts[s]) for s in subjects},
            "total_known_valid_core_frames": int(total), "normalization": "valid_frame_count",
            "factor_scope": "loss_numerator_only_separate_from_label_confidence"}


class IdentityReplayBatchSampler(Sampler):
    """Identity-interleaved ordinary traversal plus limited event replay.

    ``replay_fraction`` means replay / (ordinary + replay), so .20 appends
    floor(.25*N) samples. Pools are already time/event based, but the sampler
    also merges overlapping same-subject candidates and enforces <=2 extra
    exposures per merged event per epoch. Background extras never outnumber
    positive/transition extras. Short pools are not silently oversampled.
    """
    def __init__(self, dataset, batch_size=8, seed=20261011, effective_batch_size=None):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.effective_batch_size = int(effective_batch_size or batch_size)
        if self.batch_size < 1 or self.effective_batch_size < self.batch_size:
            raise ValueError("positive microbatch and effective batch >= microbatch required")
        if self.effective_batch_size % self.batch_size:
            raise ValueError("effective batch must be divisible by microbatch")
        self.seed = int(seed)
        if self.seed < 0:
            raise ValueError("seed must be nonnegative")
        self.subjects = [str(dataset.records[ri]["subject_id"]) for ri, _ in dataset.items]
        self.epoch = 0
        self.pool = []
        self.replay_fraction = 0.
        self.pool_version = None
        self.last_audit = {}
        self._batches = None

    def set_epoch(self, epoch):
        self.epoch = int(epoch)
        self._batches = None

    def set_pool(self, pool, replay_fraction=.2, version=None):
        if not 0 <= float(replay_fraction) <= .25:
            raise ValueError("V9 replay fraction must be within [0,.25]")
        validated = []
        for candidate in pool or []:
            row = dict(candidate)
            index = int(row["item_index"])
            if not 0 <= index < len(self.dataset):
                raise ValueError("hard item index is outside this dataset")
            subject = str(row["subject_id"])
            if subject != self.subjects[index]:
                raise ValueError("hard pool identity does not match the training item")
            if row.get("kind") not in ("positive", "transition", "background"):
                raise ValueError("hard kind must be positive, transition or background")
            if not row.get("event_id"):
                raise ValueError("hard replay requires a nonempty event_id")
            row.update(item_index=index, subject_id=subject)
            validated.append(row)
        # Give temporally overlapping proposals one common budget. A positive
        # and its boundary-transition proposal already share the meal event ID.
        grouped = defaultdict(list)
        for row in validated:
            grouped[(row["subject_id"], "background" if row["kind"] == "background" else "meal")].append(row)
        for (subject, category), rows in grouped.items():
            event_groups = {}
            for row in rows:
                event_groups.setdefault(str(row["event_id"]), []).append(row)
            intervals = []
            for event, members in event_groups.items():
                a = min(float(r.get("start_ms", r.get("candidate_start_ms", 0))) for r in members)
                b = max(float(r.get("end_ms", r.get("candidate_end_ms", a))) for r in members)
                intervals.append((a, b, event, members))
            intervals.sort(key=lambda value: (value[0], value[1], value[2]))
            components = []
            for a, b, event, members in intervals:
                match = None
                for component in components:
                    overlap = max(0., min(b, component[1]) - max(a, component[0]))
                    shortest = min(b-a, component[1]-component[0])
                    if event == component[2] or (shortest > 0 and overlap / shortest >= .5):
                        match = component
                        break
                if match is None:
                    components.append([a, b, event, list(members)])
                else:
                    match[0], match[1] = min(match[0], a), max(match[1], b)
                    match[3].extend(members)
            for _, _, event, members in components:
                for row in members:
                    row["_budget_event_id"] = subject + ":" + category + ":" + event
        self.pool = validated
        self.replay_fraction = float(replay_fraction)
        self.pool_version = version or _pool_digest(validated)
        self._batches = None

    def _extra(self, rng):
        budget = int(math.floor(len(self.dataset) * self.replay_fraction / (1-self.replay_fraction)))
        grouped = defaultdict(list)
        # Remove duplicate item/event entries before selecting a replay window.
        seen = set()
        for row in self.pool:
            key = (row["_budget_event_id"], row["item_index"])
            if key not in seen:
                grouped[row["_budget_event_id"]].append(row)
                seen.add(key)
        events = list(grouped)
        rng.shuffle(events)
        # First distribute at most one candidate from each event, then a second.
        queues = {"positive": [], "background": []}
        for turn in range(2):
            for event in events:
                rows = grouped[event]
                row = rows[int(rng.integers(len(rows)))]
                queues["background" if row["kind"] == "background" else "positive"].append(row)
        positive = queues["positive"][:budget]
        # At most half extras can be weak/confirmed background. If positives
        # are scarce, leave the unused budget empty instead of weakening guards.
        n_background = min(len(queues["background"]), len(positive), budget // 2)
        n_positive = min(len(positive), budget - n_background)
        chosen = positive[:n_positive] + queues["background"][:n_background]
        rng.shuffle(chosen)
        return chosen, budget

    def _make(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        extra, requested = self._extra(rng)
        ordinary = defaultdict(list)
        for index, subject in enumerate(self.subjects):
            ordinary[subject].append((index, "ordinary", None))
        for values in ordinary.values():
            rng.shuffle(values)
        replay = defaultdict(list)
        for row in extra:
            replay[row["subject_id"]].append((row["item_index"], "replay", row))
        # Randomly intersperse additions while preserving a complete ordinary
        # pass: the ordinary/replay source remains explicit in audit counters.
        queues = {}
        for subject in ordinary:
            rows = ordinary[subject] + replay[subject]
            rng.shuffle(rows)
            queues[subject] = deque(rows)
        order, effective_groups = [], []
        exposure, ordinary_count, replay_count = Counter(), Counter(), Counter()
        while queues:
            group = []
            used = set()
            for _ in range(self.effective_batch_size):
                available = [s for s, q in queues.items() if q]
                if not available:
                    break
                fresh = [s for s in available if s not in used]
                choices = fresh or available
                # Remaining long records must eventually be consumed; select
                # the largest queue among fresh subjects with random tie break.
                rng.shuffle(choices)
                subject = max(choices, key=lambda s: len(queues[s]))
                index, source, row = queues[subject].popleft()
                group.append(index)
                order.append(index)
                used.add(subject)
                exposure[subject] += 1
                (ordinary_count if source == "ordinary" else replay_count)[subject] += 1
                if not queues[subject]:
                    del queues[subject]
            effective_groups.append({"size": len(group), "distinct_identities": len(used),
                                     "target_distinct": min(4, len(group), len(set(self.subjects)))})
        self._batches = [order[i:i+self.batch_size] for i in range(0, len(order), self.batch_size)]
        event_counts = Counter(row["_budget_event_id"] for row in extra)
        self.last_audit = {"epoch": self.epoch, "seed": self.seed,
            "ordinary_samples": len(self.dataset), "replay_samples_requested": requested,
            "replay_samples": len(extra), "replay_fraction_actual": len(extra)/max(1,len(order)),
            "replay_positive_transition": sum(row["kind"] != "background" for row in extra),
            "replay_background": sum(row["kind"] == "background" for row in extra),
            "event_extra_exposure": dict(event_counts), "identity_exposure": dict(exposure),
            "identity_ordinary_exposure": dict(ordinary_count), "identity_replay_exposure": dict(replay_count),
            "pool_version": self.pool_version, "pool_sha256": _pool_digest(self.pool),
            "effective_batch_size": self.effective_batch_size, "microbatch_size": self.batch_size,
            "effective_groups": effective_groups,
            "groups_below_four_identities": sum(x["distinct_identities"] < x["target_distinct"] for x in effective_groups),
            "shortfall_reason": "insufficient eligible event budgets and positive/transition quota" if len(extra)<requested else None,
            "diversity_note": "tail groups may have fewer identities after complete-traversal queues are exhausted"}
        return self._batches

    def __iter__(self):
        return iter(self._make() if self._batches is None else self._batches)

    def __len__(self):
        return len(self._make() if self._batches is None else self._batches)

    def state_dict(self):
        return {"epoch": self.epoch, "seed": self.seed, "batch_size": self.batch_size,
                "effective_batch_size": self.effective_batch_size, "pool": self.pool,
                "replay_fraction": self.replay_fraction, "pool_version": self.pool_version,
                "dataset_item_count": len(self.dataset), "last_audit": self.last_audit}

    def load_state_dict(self, state):
        if (int(state["seed"]) != self.seed or int(state["batch_size"]) != self.batch_size or
                int(state["effective_batch_size"]) != self.effective_batch_size or
                int(state["dataset_item_count"]) != len(self.dataset)):
            raise ValueError("sampler resume contract differs from checkpoint")
        self.set_pool(state["pool"], state["replay_fraction"], state.get("pool_version"))
        self.set_epoch(state["epoch"])
        self.last_audit = dict(state.get("last_audit", {}))


def _metadata(dataset):
    """Read only labels/observation metadata once, without decompressing spectra."""
    by_record, item_intervals = {}, defaultdict(list)
    for ri, rec in enumerate(dataset.records):
        with np.load(rec["_path"], allow_pickle=False) as data:
            target = np.asarray(data["target"], float)
            weight = np.asarray(data["label_weight"], float)
            start = np.asarray(data["frame_start_ms"], float)
            end = np.asarray(data["frame_end_ms"], float)
        owned = np.asarray(dataset.frame_masks[ri], bool)
        by_record[ri] = {"target": target, "weight": weight, "start": start, "end": end, "owned": owned}
    for index, (ri, start) in enumerate(dataset.items):
        data = by_record[int(ri)]
        stop = min(start + dataset.core_frames, len(data["start"]))
        pick = np.flatnonzero(data["owned"][start:stop]) + start
        if len(pick):
            item_intervals[str(dataset.records[ri]["subject_id"])].append(
                (index, float(data["start"][pick[0]]), float(data["end"][pick[-1]])))
    canonical = {}
    for ri, data in by_record.items():
        subject = str(dataset.records[ri]["subject_id"])
        for fi in np.flatnonzero(data["owned"]):
            canonical[(subject, float(data["start"][fi]))] = (float(data["target"][fi]), float(data["weight"][fi]))
    return canonical, item_intervals


def mine_hard_examples(frame, records, dataset, allowed_subjects, *, model_version="v9",
                       source="training_ema", decoder=None, minimum_observed_fraction=.8,
                       high_threshold=.6, low_threshold=.2, boundary_buffer_s=120.,
                       min_background_duration_s=120., min_quality=.15, provenance=None):
    """Create small replay indices using only explicitly allowed train identities.

    The returned dictionaries carry original confidence and event/time reason.
    Predictions must be canonical (one row per identity/timestamp). Unknown
    labels, unsupported signal and boundary-protected background are excluded.
    ``source=training_crossfit`` additionally requires an excluded-subject proof.
    """
    from .events import truth_from_records, match_events
    allowed = {str(s) for s in allowed_subjects}
    dataset_subjects = {str(r["subject_id"]) for r in dataset.records}
    record_subjects = {str(r["subject_id"]) for r in records}
    frame_subjects = set(frame.subject_id.astype(str))
    if not allowed or not dataset_subjects <= allowed or not record_subjects <= allowed or not frame_subjects <= allowed:
        raise ValueError("hard mining contains an identity outside the training allow-list")
    required = {"subject_id", "start_ms", "end_ms", "eating_prob", "supported"}
    if not required <= set(frame):
        raise ValueError("hard mining needs canonical timestamp/probability/support columns")
    if frame.duplicated(["subject_id", "start_ms"]).any():
        raise ValueError("hard mining requires canonical unique timestamps")
    provenance = dict(provenance or {})
    if source in ("training_crossfit", "training_identity_crossfit"):
        proof = provenance.get("excluded_subjects")
        if proof is None or not frame_subjects <= {str(s) for s in proof}:
            raise ValueError("crossfit mining requires excluded_subjects provenance for every predicted identity")
    canonical, items = _metadata(dataset)
    truth = truth_from_records(records)
    unmatched = None
    if decoder is not None:
        from .inference import decode_frame_table
        match = match_events(decode_frame_table(frame, decoder), truth, .25)
        unmatched = {(e.subject_id, e.start_ms, e.end_ms) for e in match["unmatched_truth"]}
    table = frame.copy()
    table["subject_id"] = table.subject_id.astype(str)
    cached = [canonical.get((str(s), float(t)), (-1., 0.)) for s, t in zip(table.subject_id, table.start_ms)]
    table["_target"] = [v[0] for v in cached]
    table["_weight"] = [v[1] for v in cached]
    table["_eligible"] = (table.supported.astype(bool) & (table._target >= 0) & (table._weight > 0) &
                           np.isfinite(table.eating_prob) & (table.end_ms > table.start_ms))
    if "quality" in table:
        table["_eligible"] &= np.isfinite(table.quality) & (table.quality >= min_quality)
    candidates = []
    for event in truth:
        group = table[(table.subject_id == event.subject_id) & (table.end_ms > event.start_ms) &
                      (table.start_ms < event.end_ms)].sort_values("start_ms")
        good = group[group._eligible & (group._target > 0)]
        durations = np.maximum(0., np.minimum(good.end_ms.to_numpy(), event.end_ms) -
                               np.maximum(good.start_ms.to_numpy(), event.start_ms))
        observed = float(durations.sum() / (event.end_ms-event.start_ms))
        if observed < minimum_observed_fraction or good.empty:
            continue
        score = good.eating_prob.to_numpy(float)
        reasons = []
        if float(np.average(score, weights=durations)) < .5:
            reasons.append("low_observed_meal_probability")
        if float(score.max()) < high_threshold:
            reasons.append("no_trusted_high_trigger")
        if unmatched is not None and (event.subject_id, event.start_ms, event.end_ms) in unmatched:
            reasons.append("missed_by_frozen_event_decoder")
        low = score < low_threshold
        high = score >= high_threshold
        broken = bool(high.any() and low.mean() >= .2)
        if broken:
            reasons.append("in_meal_probability_breaks")
        if not reasons:
            continue
        candidates.append({"subject_id": event.subject_id, "kind": "transition" if broken else "positive",
                           "event_id": "meal:" + (event.meal_id or f"{event.start_ms:.0f}:{event.end_ms:.0f}"),
                           "start_ms": event.start_ms, "end_ms": event.end_ms, "reason": reasons,
                           "observed_fraction": observed, "priority": float(1-score.mean()),
                           "label_weight_min": float(good._weight.min()), "label_weight_max": float(good._weight.max())})
    # One high-probability run is one event budget, including if it crosses
    # source-record boundaries at genuinely contiguous observed timestamps.
    for subject, group in table.groupby("subject_id", sort=True):
        group = group.sort_values("start_ms")
        valid = group._eligible.to_numpy(bool) & (group._target.to_numpy() == 0)
        starts, ends = group.start_ms.to_numpy(float), group.end_ms.to_numpy(float)
        for event in truth:
            if event.subject_id == subject:
                valid &= ~((ends > event.start_ms-boundary_buffer_s*1000) &
                           (starts < event.end_ms+boundary_buffer_s*1000))
        valid &= group.eating_prob.to_numpy(float) >= high_threshold
        run = []
        def finish(indices):
            if not indices:
                return
            segment = group.iloc[indices]
            a, b = float(segment.start_ms.iloc[0]), float(segment.end_ms.iloc[-1])
            if b-a < min_background_duration_s*1000:
                return
            candidates.append({"subject_id": str(subject), "kind": "background",
                "event_id": f"background:{a:.0f}:{b:.0f}", "start_ms": a, "end_ms": b,
                "reason": ["sustained_high_known_background_outside_meal_protection"],
                "observed_fraction": 1., "priority": float(segment.eating_prob.mean()),
                "label_weight_min": float(segment._weight.min()), "label_weight_max": float(segment._weight.max())})
        for index, ok in enumerate(valid):
            contiguous = bool(run) and starts[index] <= ends[run[-1]] + 1.
            if not ok or (run and not contiguous):
                finish(run)
                run = []
            if ok:
                run.append(index)
        finish(run)
    output = []
    for candidate in candidates:
        for index, a, b in items[candidate["subject_id"]]:
            if b <= candidate["start_ms"] or a >= candidate["end_ms"]:
                continue
            output.append({**candidate, "item_index": int(index), "core_start_ms": a, "core_end_ms": b,
                           "model_version": str(model_version), "source": str(source),
                           "training_subjects": sorted(allowed), "provenance": provenance,
                           "confidence_policy": "retain_original_cached_label_weight_no_promotion",
                           "boundary_protection_s": float(boundary_buffer_s)})
    output.sort(key=lambda row: (row["subject_id"], row["event_id"], row["item_index"]))
    return output


def save_hard_pool(path, pool, *, metadata=None):
    """Atomic JSON audit sidecar; does not touch physical preprocessing caches."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"version": "v9_event_replay_1", "pool_sha256": _pool_digest(pool),
               "candidate_core_entries": len(pool), "pool": pool, "metadata": dict(metadata or {})}
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)
    return payload

