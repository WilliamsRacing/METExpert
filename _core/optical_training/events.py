"""Timestamp-aware event decoding and the competition's strict IoU scoring.

This module needs NumPy, SciPy and pandas, but not PyTorch. Event intervals use
milliseconds and half-open boundaries [start_ms, end_ms). Neither smoothing nor
gap merging may bridge unobserved or invalid-quality intervals. The caller must
deduplicate overlapping recording windows before passing frames to the decoder.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class Event:
    subject_id: str
    start_ms: float
    end_ms: float
    score: float = 1.0
    meal_id: str = ""
    scenario: str = "unknown"

    def __post_init__(self) -> None:
        object.__setattr__(self, "subject_id", str(self.subject_id))
        object.__setattr__(self, "meal_id", str(self.meal_id))
        object.__setattr__(self, "scenario", str(self.scenario or "unknown"))
        object.__setattr__(self, "start_ms", float(self.start_ms))
        object.__setattr__(self, "end_ms", float(self.end_ms))
        object.__setattr__(self, "score", float(self.score))
        if not np.isfinite(self.start_ms) or not np.isfinite(self.end_ms):
            raise ValueError("Event timestamps must be finite milliseconds.")
        if self.end_ms <= self.start_ms:
            raise ValueError("Event end_ms must be strictly greater than start_ms.")
        if not np.isfinite(self.score):
            raise ValueError("Event score must be finite.")


def interval_iou(a: Event | Sequence[float], b: Event | Sequence[float]) -> float:
    """Temporal intersection over union; touching intervals have zero overlap."""
    sa, ea = (a.start_ms, a.end_ms) if isinstance(a, Event) else a
    sb, eb = (b.start_ms, b.end_ms) if isinstance(b, Event) else b
    if not np.all(np.isfinite([sa, ea, sb, eb])) or ea <= sa or eb <= sb:
        raise ValueError("IoU needs two finite, positive-duration intervals.")
    overlap = max(0.0, min(ea, eb) - max(sa, sb))
    union = (ea - sa) + (eb - sb) - overlap
    return float(overlap / union)


def _metrics(matched: list, n_pred: int, n_truth: int) -> dict:
    tp = len(matched)
    fp, fn = n_pred - tp, n_truth - tp
    sensitivity = tp / n_truth if n_truth else 0.0
    precision = tp / n_pred if n_pred else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    return {
        "TP": tp, "FP": fp, "FN": fn,
        "sensitivity": sensitivity, "precision": precision, "PPV": precision,
        "F1": f1,
        "start_mae_s": float(np.mean([abs(p.start_ms - t.start_ms) / 1000.0
                                       for p, t, _ in matched])) if tp else None,
        "end_mae_s": float(np.mean([abs(p.end_ms - t.end_ms) / 1000.0
                                     for p, t, _ in matched])) if tp else None,
        "mean_matched_iou": float(np.mean([iou for _, _, iou in matched])) if tp else None,
        "n_pred": n_pred, "n_truth": n_truth,
    }


def match_events(pred: Sequence[Event], truth: Sequence[Event],
                 iou_threshold: float = 0.25) -> dict:
    """One-to-one, same-subject matching with **IoU > threshold**.

    The assignment maximizes the number of valid matches first, then total IoU.
    This avoids an IoU-greedy match consuming the only partner of another event.
    ``matched`` contains (prediction, truth, IoU) triples. Boundary MAE uses only
    matched events. Empty denominators give zero rates; no hits give None MAEs.
    """
    if not 0 <= iou_threshold < 1:
        raise ValueError("iou_threshold must be in [0, 1).")
    pred, truth = list(pred), list(truth)
    matched, used_pred, used_truth = [], set(), set()
    subjects = sorted({e.subject_id for e in pred} | {e.subject_id for e in truth}, key=str)
    for subject in subjects:
        pi = [i for i, e in enumerate(pred) if e.subject_id == subject]
        ti = [i for i, e in enumerate(truth) if e.subject_id == subject]
        if not pi or not ti:
            continue
        ious = np.asarray([[interval_iou(pred[i], truth[j]) for j in ti] for i in pi])
        valid = ious > iou_threshold
        cardinality_weight = min(len(pi), len(ti)) + 1.0
        # Each prediction can instead occupy a zero-weight dummy column.
        weights = np.zeros((len(pi), len(ti) + len(pi)), dtype=float)
        weights[:, :len(ti)] = np.where(valid, cardinality_weight + ious,
                                       -cardinality_weight)
        rows, cols = linear_sum_assignment(-weights)
        for row, col in zip(rows, cols):
            if col < len(ti) and valid[row, col]:
                i, j = pi[row], ti[col]
                used_pred.add(i)
                used_truth.add(j)
                matched.append((pred[i], truth[j], float(ious[row, col])))
    matched.sort(key=lambda x: (str(x[1].subject_id), x[1].start_ms, x[0].start_ms))
    return {
        "matched": matched,
        "unmatched_pred": [e for i, e in enumerate(pred) if i not in used_pred],
        "unmatched_truth": [e for i, e in enumerate(truth) if i not in used_truth],
        "metrics": _metrics(matched, len(pred), len(truth)),
    }


def _vector(values, n: int, name: str, default: float | None = None) -> np.ndarray:
    if values is None:
        if default is None:
            raise ValueError(f"{name} is required.")
        return np.full(n, default, dtype=float)
    result = np.asarray(values, dtype=float)
    if result.ndim != 1 or len(result) != n:
        raise ValueError(f"{name} must be a one-dimensional vector of length {n}.")
    return result


def _observed_runs(starts: np.ndarray, ends: np.ndarray,
                   valid: np.ndarray) -> list[tuple[int, int]]:
    """Return [first, last_exclusive) runs with complete valid time coverage."""
    runs, begin = [], None
    for i in range(len(starts)):
        contiguous = i > 0 and abs(starts[i] - ends[i - 1]) <= 1e-6
        if not valid[i] or (begin is not None and not contiguous):
            if begin is not None:
                runs.append((begin, i))
                begin = None
        if valid[i] and begin is None:
            begin = i
    if begin is not None:
        runs.append((begin, len(starts)))
    return runs


def _smooth(prob: np.ndarray, width: int) -> np.ndarray:
    if width <= 1:
        return prob.copy()
    # Edge replication keeps the first/last complete frame valid. Each run is
    # processed separately, so filtering never mixes measurements across gaps.
    pad = width // 2
    padded = np.pad(prob, (pad, pad), mode="edge")
    return np.median(np.lib.stride_tricks.sliding_window_view(padded, width), axis=1)


class PreparedFrameDecoder:
    """Private immutable frame snapshot; reuse gap-local median filtering.

    A new instance is created for every supplied probability table. Arrays are
    copied and read-only: callers cannot mutate a frame table and silently reuse
    stale smoothing. Cache keys contain both quality threshold and median width;
    all other decoding and matching behavior remains in events_from_frames.
    """
    def __init__(self, starts, ends, probabilities, subject_id, *,
                 start_probs=None, end_probs=None, quality=None):
        self.subject_id = str(subject_id)
        def snapshot(value):
            if value is None:
                return None
            result = np.array(value, dtype=float, copy=True)
            result.flags.writeable = False
            return result
        self.starts = snapshot(starts)
        self.ends = snapshot(ends)
        self.prob = snapshot(probabilities)
        self.sp = snapshot(start_probs)
        self.ep = snapshot(end_probs)
        self.quality = snapshot(quality)
        self._runs = {}

    def _smoothed_runs(self, starts, ends, prob, quality, min_quality, width):
        # events_from_frames has already performed the complete validation.
        if (starts is not self.starts or ends is not self.ends or prob is not self.prob
                or (self.quality is not None and quality is not self.quality)):
            raise ValueError("Prepared decoder belongs to a different frame snapshot")
        key = (float(min_quality), int(width))
        if key not in self._runs:
            valid = np.isfinite(prob) & np.isfinite(quality) & (quality >= min_quality)
            rows = []
            for begin, stop in _observed_runs(starts, ends, valid):
                smooth = _smooth(prob[begin:stop], width)
                smooth.flags.writeable = False
                rows.append((begin, stop, smooth))
            self._runs[key] = tuple(rows)
        return self._runs[key]

    def decode(self, **params):
        return events_from_frames(self.starts, self.ends, self.prob,
            subject_id=self.subject_id, start_probs=self.sp, end_probs=self.ep,
            quality=self.quality, _prepared=self, **params)


def events_from_frames(frame_start_ms, frame_end_ms, probabilities, subject_id: str,
                       start_probs=None, end_probs=None, threshold: float = 0.5,
                       low_threshold: float = 0.35, min_duration_s: float = 120,
                       max_duration_s: float = 7200, merge_gap_s: float = 60,
                       quality=None, min_quality: float = 0.15,
                       boundary_threshold: float = 0.5, smoothing_frames: int = 3,
                       boundary_search_s: float = 120,
                       min_high_duration_s: float = 0,
                       min_high_fraction: float = 0,
                       min_mean_probability: float = 0, max_bridge_frames: int | None = None, *,
                       _prepared: PreparedFrameDecoder | None = None) -> list[Event]:
    """Median smoothing, hysteresis and optional edge-only boundary refinement.

    Inputs must be sorted, non-overlapping frames for a *single subject*.
    NaN probabilities, insufficient quality, or actual unobserved gaps split the
    timeline. A short below-low-threshold background gap can be merged only
    within one fully observed, valid run. Boundary heads can trim a decoded
    event near its two edges; they never repeatedly split its interior.

    Predicted boundaries have frame resolution. ``score`` is the time-weighted
    mean of the original (unsmoothed) eating probabilities in the event.
    Optional evidence requirements are checked separately for every final
    maximum-duration slice, after boundary refinement. High evidence uses the
    run-local median-smoothed probabilities and true overlap durations; mean
    evidence uses original probabilities. Zero requirements preserve V1.
    """
    starts = np.asarray(frame_start_ms, dtype=float)
    if starts.ndim != 1:
        raise ValueError("frame_start_ms must be one-dimensional.")
    n = len(starts)
    ends = _vector(frame_end_ms, n, "frame_end_ms")
    prob = _vector(probabilities, n, "probabilities")
    qual = _vector(quality, n, "quality", default=1.0)
    sp = _vector(start_probs, n, "start_probs", default=0.0)
    ep = _vector(end_probs, n, "end_probs", default=0.0)
    if not np.all(np.isfinite([threshold, low_threshold, min_duration_s,
                               max_duration_s, merge_gap_s, min_quality,
                               boundary_threshold, boundary_search_s,
                               min_high_duration_s, min_high_fraction,
                               min_mean_probability])):
        raise ValueError("All decoder thresholds and durations must be finite.")
    if not 0 <= low_threshold <= threshold <= 1:
        raise ValueError("Require 0 <= low_threshold <= threshold <= 1.")
    if not 0 <= min_quality <= 1 or not 0 <= boundary_threshold <= 1:
        raise ValueError("Quality and boundary thresholds must be in [0, 1].")
    if min_duration_s < 0 or max_duration_s <= 0 or max_duration_s < min_duration_s:
        raise ValueError("Require 0 <= min_duration_s <= max_duration_s, with max > 0.")
    if min_high_duration_s < 0:
        raise ValueError("min_high_duration_s cannot be negative.")
    if not 0 <= min_high_fraction <= 1 or not 0 <= min_mean_probability <= 1:
        raise ValueError("Evidence fraction and mean probability must be in [0, 1].")
    if merge_gap_s < 0 or boundary_search_s < 0:
        raise ValueError("Gap and boundary search durations cannot be negative.")
    if not isinstance(smoothing_frames, (int, np.integer)) or smoothing_frames < 1 or smoothing_frames % 2 == 0:
        raise ValueError("smoothing_frames must be a positive odd integer.")
    if not np.all(np.isfinite(starts)) or not np.all(np.isfinite(ends)):
        raise ValueError("All frame timestamps must be finite.")
    if np.any(ends <= starts) or (n > 1 and np.any(starts[1:] < ends[:-1] - 1e-6)):
        raise ValueError("Frames must be sorted, positive-duration and non-overlapping.")
    for values, name in [(prob, "probabilities"), (qual, "quality"),
                         (sp, "start_probs"), (ep, "end_probs")]:
        finite = values[np.isfinite(values)]
        if np.any((finite < 0) | (finite > 1)):
            raise ValueError(f"Finite {name} values must be in [0, 1].")
    valid = np.isfinite(prob) & np.isfinite(qual) & (qual >= min_quality)
    result = []
    if _prepared is None:
        runs = ((begin, stop, _smooth(prob[begin:stop], smoothing_frames))
                for begin, stop in _observed_runs(starts, ends, valid))
    else:
        runs = _prepared._smoothed_runs(starts, ends, prob, qual, min_quality, smoothing_frames)
    for begin, stop, smoothed in runs:
        above_low = smoothed >= low_threshold
        candidates, first = [], None
        for local in range(len(smoothed) + 1):
            active = local < len(smoothed) and above_low[local]
            if active and first is None:
                first = local
            if not active and first is not None:
                if np.any(smoothed[first:local] >= threshold):
                    candidates.append((begin + first, begin + local))
                first = None
        merged = []
        for left, right in candidates:
            if merged and starts[left] - ends[merged[-1][1] - 1] <= merge_gap_s * 1000 and (max_bridge_frames is None or left - merged[-1][1] <= max_bridge_frames):
                # Every intervening frame is valid and fully observed here.
                previous_left, previous_right = merged[-1]
                if np.all(smoothed[previous_right - begin:left - begin] < low_threshold):
                    merged[-1] = (previous_left, right)
                    continue
            merged.append((left, right))
        for left, right in merged:
            refined_left, refined_right = left, right
            if start_probs is not None and boundary_search_s > 0:
                indices = np.arange(left, right)
                indices = indices[starts[indices] <= starts[left] + boundary_search_s * 1000]
                scores = np.nan_to_num(sp[indices], nan=0.0, posinf=0.0, neginf=0.0)
                if len(indices) and scores.max() >= boundary_threshold:
                    refined_left = int(indices[np.argmax(scores)])
            if end_probs is not None and boundary_search_s > 0:
                indices = np.arange(left, right)
                indices = indices[ends[indices] >= ends[right - 1] - boundary_search_s * 1000]
                scores = np.nan_to_num(ep[indices], nan=0.0, posinf=0.0, neginf=0.0)
                if len(indices) and scores.max() >= boundary_threshold:
                    refined_right = int(indices[np.argmax(scores)]) + 1
            # A noisy edge prediction cannot invalidate an otherwise long event.
            if refined_left < refined_right and (
                ends[refined_right - 1] - starts[refined_left] >= min_duration_s * 1000
            ):
                left, right = refined_left, refined_right
            current_start = starts[left]
            final_end = ends[right - 1]
            # Split only at the configured maximum duration; no boundary head
            # noise introduces additional internal event edges.
            while current_start < final_end:
                current_end = min(final_end, current_start + max_duration_s * 1000)
                if current_end - current_start >= min_duration_s * 1000:
                    weights = np.maximum(0.0, np.minimum(ends[left:right], current_end)
                                         - np.maximum(starts[left:right], current_start))
                    score = float(np.average(prob[left:right], weights=weights))
                    # Include valid below-low frames in the denominator after
                    # gap merging. Evidence cannot leak in from another slice.
                    high = smoothed[left - begin:right - begin] >= threshold
                    high_seconds = float(weights[high].sum() / 1000.0)
                    slice_seconds = (current_end - current_start) / 1000.0
                    high_fraction = high_seconds / slice_seconds
                    if (high_seconds + 1e-9 >= min_high_duration_s
                            and high_fraction + 1e-12 >= min_high_fraction
                            and score + 1e-12 >= min_mean_probability):
                        result.append(Event(str(subject_id), float(current_start),
                                            float(current_end), score=score))
                current_start = current_end
    return result


def events_to_dataframe(events: Iterable[Event]) -> pd.DataFrame:
    columns = ["subject_id", "start_ms", "end_ms", "score", "meal_id", "scenario"]
    result = pd.DataFrame([asdict(e) for e in events], columns=columns)
    if len(result):
        result = result.sort_values(["subject_id", "start_ms", "end_ms"]).reset_index(drop=True)
    return result


def truth_from_records(records: Iterable[Mapping] | str | Path) -> list[Event]:
    """Deduplicate full meal labels across recording/cache metadata.

    No coverage clipping is performed: callers select observed-support labels
    explicitly and report truncated/partial support separately. Subject and
    scenario identifiers are bookkeeping, never model inputs.
    """
    if isinstance(records, (str, Path)):
        with Path(records).open("r", encoding="utf-8") as handle:
            records = [json.loads(line) for line in handle if line.strip()]
    unique = {}
    for record in records:
        subject = str(record["subject_id"])
        meals = record.get("meals", [])
        if isinstance(meals, str):
            meals = json.loads(meals)
        for meal in meals:
            scenario = next((str(meal[key]) for key in
                             ("wear_scenario", "mealwear_scenario", "scenario")
                             if meal.get(key) is not None), "unknown")
            event = Event(subject, float(meal["start_ms"]), float(meal["end_ms"]),
                          meal_id=str(meal.get("meal_id", meal.get("uniqueid", ""))),
                          scenario=scenario)
            key = (subject, event.start_ms, event.end_ms)
            if key not in unique or (unique[key].scenario == "unknown" and scenario != "unknown"):
                unique[key] = event
    return sorted(unique.values(), key=lambda e: (e.subject_id, e.start_ms, e.end_ms))


def evaluate_by_scenario(pred: Sequence[Event], truth: Sequence[Event],
                         iou_threshold: float = 0.25) -> dict:
    """Report truth-scenario recall, using one global one-to-one assignment.

    Negative/background intervals do not have meal-wear labels, so assigning
    false positives to a meal scenario (and reporting scenario precision/F1)
    would be unjustified. Only overall precision and F1 are reported.
    """
    matched_result = match_events(pred, truth, iou_threshold)
    rows = []
    for scenario in sorted({t.scenario for t in truth}, key=str):
        subset = [t for t in truth if t.scenario == scenario]
        hits = [m for m in matched_result["matched"] if m[1].scenario == scenario]
        values = _metrics(hits, len(hits), len(subset))
        rows.append({"scenario": scenario, "n_truth": len(subset), "TP": len(hits),
                     "FN": len(subset) - len(hits), "sensitivity": values["sensitivity"],
                     "start_mae_s": values["start_mae_s"], "end_mae_s": values["end_mae_s"]})
    return {"overall": matched_result["metrics"], "by_scenario": rows,
            "note": "Scenario precision/F1 unavailable: background wear scenarios are unlabelled."}


def _find_column(table: pd.DataFrame, names: Sequence[str], required: bool = True):
    for name in names:
        if name in table.columns:
            return name
    if required:
        raise ValueError(f"Frame table needs one of these columns: {', '.join(names)}.")
    return None


def tune_decoder(frame_table: pd.DataFrame, truth: Sequence[Event], grid=None) -> dict:
    """Tune three decoder parameters on held-out **validation subjects only**.

    ``grid`` is a mapping with threshold, min_duration_s and merge_gap_s lists,
    or an iterable of parameter dictionaries. The low threshold is always
    0.7 * threshold; smoothing/quality/max duration/boundary defaults are fixed.
    Returns JSON-serializable ``params``, ``metrics`` and all ``search`` rows.
    A caller must freeze these parameters before evaluating test subjects.
    """
    if "subject_id" not in frame_table:
        raise ValueError("Frame table needs subject_id.")
    start_col = _find_column(frame_table, ("frame_start_ms", "start_ms"))
    end_col = _find_column(frame_table, ("frame_end_ms", "end_ms"))
    prob_col = _find_column(frame_table, ("eating_prob", "probability", "probabilities", "prob", "p_eating"))
    start_prob_col = _find_column(frame_table, ("start_probability", "start_prob", "p_start"), False)
    end_prob_col = _find_column(frame_table, ("end_probability", "end_prob", "p_end"), False)
    quality_col = _find_column(frame_table, ("quality",), False)
    allowed = ("threshold", "min_duration_s", "merge_gap_s")
    defaults = {"threshold": (0.4, 0.5, 0.6), "min_duration_s": (60, 120, 180),
                "merge_gap_s": (0, 30, 60)}
    if grid is None:
        grid = defaults
    if isinstance(grid, Mapping):
        if set(grid) - set(allowed):
            raise ValueError(f"Only {allowed} may be tuned.")
        candidates = [dict(zip(allowed, values)) for values in
                      product(*(grid.get(key, defaults[key]) for key in allowed))]
    else:
        candidates = list(grid)
        if any(set(c) - set(allowed) or set(c) != set(allowed) for c in candidates):
            raise ValueError(f"Each grid candidate must contain exactly {allowed}.")
    if not candidates:
        raise ValueError("Decoder grid cannot be empty.")
    groups = [(subject, group.sort_values([start_col, end_col]))
              for subject, group in frame_table.groupby("subject_id", sort=True)]
    supported_subjects = {str(subject) for subject, _ in groups}
    if {str(t.subject_id) for t in truth} - supported_subjects:
        raise ValueError("Truth contains subjects missing from validation frame_table.")
    rows, best, best_key = [], None, None
    for candidate in candidates:
        params = {key: float(candidate[key]) for key in allowed}
        params["low_threshold"] = 0.7 * params["threshold"]
        predicted = []
        for subject, group in groups:
            predicted.extend(events_from_frames(
                group[start_col], group[end_col], group[prob_col], str(subject),
                start_probs=group[start_prob_col] if start_prob_col else None,
                end_probs=group[end_prob_col] if end_prob_col else None,
                quality=group[quality_col] if quality_col else None, **params))
        metrics = match_events(predicted, truth)["metrics"]
        row = {"params": params, "metrics": metrics}
        rows.append(row)
        boundary_mae = (metrics["start_mae_s"] or 0) + (metrics["end_mae_s"] or 0)
        key = (metrics["F1"], metrics["sensitivity"], metrics["precision"], -boundary_mae,
               -params["merge_gap_s"], params["threshold"], params["min_duration_s"])
        if best_key is None or key > best_key:
            best, best_key = row, key
    return {"params": best["params"], "metrics": best["metrics"], "search": rows,
            "note": "Validation-subject tuning only; freeze params before test evaluation."}
