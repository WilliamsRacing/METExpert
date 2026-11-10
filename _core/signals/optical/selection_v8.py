
from __future__ import annotations

import math
import numpy as np

EXPERTS = ("feature", "raw")
ALPHAS = tuple(i / 10. for i in range(11))
HIGHS = (.50, .55, .60, .65)
LOWS = (.20, .30, .40)
DURATIONS = (120, 300)
BRIDGES = (0, 30)
F1_TOLERANCE = .005
SCOPE = "inner_validation_identities_only"


def decoder_candidates_v8(base, checkpoint=False):
    fixed = dict(base)
    for key in ("selection_f1_tolerance", "selection_stability_penalty"):
        fixed.pop(key, None)
    fixed.update(search_version="v8_checkpoint" if checkpoint else "v8",
                 smoothing_frames=3, boundary_search_s=0, min_quality=.15,
                 max_duration_s=7200, max_bridge_frames=1,
                 min_high_duration_s=0, min_high_fraction=0., min_mean_probability=0.)
    highs = (.55, .65) if checkpoint else HIGHS
    lows = (.20, .40) if checkpoint else LOWS
    bridges = (0,) if checkpoint else BRIDGES
    for high in highs:
        for low in lows:
            for duration in DURATIONS:
                for gap in bridges:
                    yield dict(fixed, threshold=high, low_threshold=low,
                               min_duration_s=duration, merge_gap_s=gap)


def _f1(tp, fp, fn):
    d = 2 * tp + fp + fn
    return float(2 * tp / d) if d else 0.


def identity_stability(predicted, truth, subjects):
    from .events import match_events
    subject_ids = sorted({str(s) for s in subjects})
    if ({str(e.subject_id) for e in predicted} | {str(e.subject_id) for e in truth}) - set(subject_ids):
        raise ValueError("V8 validation events include an undeclared identity")
    counts = []
    for subject in subject_ids:
        m = match_events([e for e in predicted if str(e.subject_id) == subject],
                         [e for e in truth if str(e.subject_id) == subject],
                         iou_threshold=.25)["metrics"]
        counts.append(dict(subject_id=subject, **{k: int(m[k]) for k in ("TP", "FP", "FN")},
                           F1=float(m["F1"])))
    totals = {k: sum(row[k] for row in counts) for k in ("TP", "FP", "FN")}
    micro = _f1(**{k.lower(): v for k, v in totals.items()})
    loo = ([{"omitted_subject_id": row["subject_id"],
             "F1": _f1(totals["TP"] - row["TP"], totals["FP"] - row["FP"],
                       totals["FN"] - row["FN"])} for row in counts] if len(counts) > 1 else [])
    values = [r["F1"] for r in loo]
    return {"selection_micro_f1": micro, "selection_loo_min": float(min(values)) if values else micro,
            "selection_loo_mean": float(np.mean(values)) if values else micro,
            "selection_loo_std": float(np.std(values)) if values else 0.,
            "selection_subject_count": len(counts), "selection_identity_counts": counts,
            "selection_loo_f1": loo,
            "selection_loo_note": "delete-one-identity pooled sensitivity; not worst single identity or new CV"}


def frame_average_precision(target, probability):
    y, p = np.asarray(target, float), np.asarray(probability, float)
    if y.ndim != 1 or y.shape != p.shape or not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError("AP requires finite aligned 1-D vectors")
    if not np.isin(y, (0., 1.)).all():
        raise ValueError("AP targets must be binary")
    positives = float(y.sum())
    if not len(y) or not positives:
        return 0.
    order = np.argsort(-p, kind="stable")
    y, p = y[order], p[order]
    ends = np.r_[np.flatnonzero(np.diff(p) != 0), len(p) - 1]
    tp = np.cumsum(y)[ends]
    return float(np.sum((tp / (ends + 1)) * np.diff(np.r_[0., tp])) / positives)


def _rules(frame, records, config, checkpoint):
    from .inference import FrameDecodeCache, truth_in_coverage, event_report
    prepared = FrameDecodeCache(frame, physical_continuity=True)
    truth = truth_in_coverage(records, frame)
    rows = []
    for decoder in decoder_candidates_v8(config["decoder"], checkpoint=checkpoint):
        metrics, predicted, _ = event_report(frame, records, decoder, prepared=prepared,
                                              prepared_truth=truth)
        metrics.update(identity_stability(predicted, truth[0], frame.subject_id.unique()))
        if abs(metrics["F1"] - metrics["selection_micro_f1"]) > 1e-12:
            raise RuntimeError("Identity event counts differ from pooled matching")
        rows.append({"decoder": decoder, "metrics": metrics})
    return rows


def _rule_tie(decoder):
    return (int(decoder["merge_gap_s"] != 0),
            abs(HIGHS.index(float(decoder["threshold"])) - HIGHS.index(.55))
            + abs(LOWS.index(float(decoder["low_threshold"])) - LOWS.index(.20)),
            int(decoder["min_duration_s"] != 300))


def fixed_rule_panel(frame, records, config):
    rows = _rules(frame, records, config, checkpoint=True)
    if len(rows) != 8:
        raise RuntimeError("V8 checkpoint panel must contain exactly eight fixed rules")
    values = np.asarray([row["metrics"]["F1"] for row in rows], dtype=float)
    i = min(range(len(rows)), key=lambda k: (-values[k],
                rows[k]["metrics"]["selection_loo_std"], _rule_tie(rows[k]["decoder"]), k))
    summary = {"mean_fixed_rule_F1": float(values.mean()), "std_fixed_rule_F1": float(values.std()),
               "best_fixed_rule_F1": float(values.max()), "fixed_rule_F1": values.tolist(),
               "n_fixed_rules": 8}
    return rows[i]["decoder"], rows[i]["metrics"], rows, summary


def select_experts(frame, records, config):
    from .event_detection import fuse_frame_table
    subjects = sorted(str(s) for s in frame.subject_id.unique())
    result = {}
    for index, expert in enumerate(EXPERTS):
        table = fuse_frame_table(frame, [float(index == 0), float(index == 1)])
        supported = table.supported.to_numpy(dtype=bool)
        y, p = table.target.to_numpy(float), table.eating_prob.to_numpy(float)
        observed = supported & np.isfinite(y) & (y >= 0) & np.isfinite(p)
        receipt = {"expert": expert, "scope": SCOPE, "fit_subjects": subjects,
                   "caller_responsible_for_inner_split": True, "outer_labels_used": False,
                   "supported_frames": int(supported.sum()), "supervised_frames": int(observed.sum()),
                   "selection_policy": "mean_fixed_eight_rule_F1_then_AP_then_negative_BCE_v8",
                   "strict_selected_expert": True}
        if not observed.any():
            decoder = next(decoder_candidates_v8(config["decoder"], checkpoint=True))
            metrics = dict(receipt, F1=0., mean_fixed_rule_F1=0., std_fixed_rule_F1=0.,
                           best_fixed_rule_F1=0., fixed_rule_F1=[0.] * 8, n_fixed_rules=8,
                           frame_AP=0., frame_BCE=1e6, selection_score=-1.,
                           selection_unavailable=True)
            result[expert] = {"score": [-1., -1., -1e6], "metrics": metrics,
                              "decoder": decoder, "selection_trials": []}
            continue
        labels = (y[observed] >= .5).astype(float)
        ap = frame_average_precision(labels, p[observed])
        clipped = np.clip(p[observed], 1e-6, 1 - 1e-6)
        bce = float(np.mean(-labels * np.log(clipped) - (1 - labels) * np.log(1 - clipped)))
        decoder, metrics, trials, summary = fixed_rule_panel(table, records, config)
        score = [summary["mean_fixed_rule_F1"], ap, -bce]
        metrics = dict(metrics, **summary, **receipt, frame_AP=ap, frame_BCE=bce,
                       selection_score=score[0], selection_unavailable=False,
                       event_F1_is_descriptive_best_fixed_rule=True)
        result[expert] = {"score": score, "metrics": metrics, "decoder": decoder,
                          "selection_trials": trials}
    return result


def candidate_grid_key(row):
    d = row["decoder"]
    return (ALPHAS.index(round(float(row["alpha"]), 10)),
            HIGHS.index(float(d["threshold"])), LOWS.index(float(d["low_threshold"])),
            DURATIONS.index(int(d["min_duration_s"])), BRIDGES.index(int(d["merge_gap_s"])))


def select_stable_candidate(rows):
    """Near-peak pooled F1, deletion minimum, local median, F1, preset ties.

    The function never changes the 0.005 tolerance based on any observed score.
    Neighborhood varies one of alpha/high/low by one grid step; duration and
    bridging remain fixed. It reuses the existing candidate table.
    """
    if not rows:
        raise ValueError("V8 selection requires candidates")
    indexed = {}
    for i, row in enumerate(rows):
        key = candidate_grid_key(row)
        if key in indexed:
            raise ValueError("Duplicate V8 fusion/decoder candidate")
        indexed[key] = i
        m = row["metrics"]
        for k in ("F1", "selection_loo_min"):
            if not math.isfinite(float(m[k])) or not 0 <= float(m[k]) <= 1:
                raise ValueError("Invalid V8 selection metric: " + k)
        if abs(float(m.get("selection_micro_f1", m["F1"])) - float(m["F1"])) > 1e-12:
            raise ValueError("Pooled and identity count F1 differ")
    best_f1 = max(float(row["metrics"]["F1"]) for row in rows)
    for key, i in indexed.items():
        neighbor_indices = [i]
        for axis in (0, 1, 2):
            for step in (-1, 1):
                adjacent = list(key)
                adjacent[axis] += step
                found = indexed.get(tuple(adjacent))
                if found is not None:
                    neighbor_indices.append(found)
        values = [float(rows[j]["metrics"]["F1"]) for j in neighbor_indices]
        rows[i]["metrics"].update(
            selection_neighborhood_median=float(np.median(values)),
            selection_neighborhood_size=len(neighbor_indices),
            selection_best_micro_f1=best_f1, selection_f1_tolerance=F1_TOLERANCE,
            selection_near_optimal=bool(rows[i]["metrics"]["F1"] + F1_TOLERANCE + 1e-12 >= best_f1),
            selection_policy="within_0.005_then_loo_min_then_local_median_then_f1_then_preset_v8",
            selection_score=float(rows[i]["metrics"]["F1"]))
    eligible = [i for i, row in enumerate(rows) if row["metrics"]["selection_near_optimal"]]
    def rank(i):
        row, m = rows[i], rows[i]["metrics"]
        a, h, low, duration, gap = candidate_grid_key(row)
        return (-float(m["selection_loo_min"]), -float(m["selection_neighborhood_median"]),
                -float(m["F1"]), int(gap != 0), abs(h - 1) + abs(low),
                abs(a - 5), int(DURATIONS[duration] != 300), i)
    return min(eligible, key=rank)
