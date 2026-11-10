
from __future__ import annotations

import inspect
import itertools
import math
import numpy as np
from .events import PreparedFrameDecoder, events_from_frames, match_events, truth_from_records

EXPERTS = ("feature", "raw")
ALPHAS = tuple(i / 10. for i in range(11))
HIGHS, LOWS = (.5, .6, .7), (.2, .3)
DURATIONS, BRIDGES, HIGH_DURATIONS = (120, 300), (0, 30), (30, 60)
F1_TOLERANCE = .005
R2_REVISION = "r2_precision_repair"
R2_HIGHS, R2_LOWS = (.55, .65, .75), (.3, .4, .5)


def is_precision_revision(config):
    return (config or {}).get("model", {}).get("revision") == R2_REVISION


def validate_frame_table(frame):
    required = {"subject_id", "start_ms", "end_ms", "feature_prob", "raw_prob",
                "feature_supported", "raw_supported", "feature_reliability", "raw_reliability"}
    if required - set(frame):
        raise ValueError("V9 frame table missing columns: " + str(sorted(required - set(frame))))
    if frame.empty:
        raise ValueError("V9 needs a nonempty frame table")
    if frame.duplicated(["subject_id", "start_ms"]).any():
        raise ValueError("Duplicate canonical subject/timestamp frames")
    for _, group in frame.groupby("subject_id", sort=True):
        group = group.sort_values("start_ms", kind="stable")
        starts, ends = group.start_ms.to_numpy(float), group.end_ms.to_numpy(float)
        if not np.isfinite(np.r_[starts, ends]).all() or np.any(ends <= starts):
            raise ValueError("Frame times must be finite with positive duration")
        if len(starts) > 1 and np.any(starts[1:] < ends[:-1] - 1e-6):
            raise ValueError("V9 canonical frames must not overlap")


def fuse_frame_table(frame, alpha, *, quality_correction=True):
    alpha = float(alpha)
    if not np.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("Alpha must lie on [0,1]")
    result = frame.copy()
    result.attrs = dict(frame.attrs)
    prior = np.array([alpha, 1. - alpha])
    support = np.column_stack([frame[x + "_supported"].fillna(False).to_numpy(bool) for x in EXPERTS])
    reliability = np.column_stack([frame[x + "_reliability"].to_numpy(float) for x in EXPERTS])
    reliability = np.clip(np.nan_to_num(reliability, nan=0., posinf=0., neginf=0.), 0., 1.)
    usable = support & (prior > 0)
    scores = np.column_stack([frame[x + "_prob"].to_numpy(float) for x in EXPERTS])
    if np.any(~np.isfinite(scores[usable])) or np.any((scores[usable] < 0) | (scores[usable] > 1)):
        raise ValueError("Selected supported scores must be finite probabilities")
    evidence = (.5 + .5 * reliability) if quality_correction else np.ones_like(reliability)
    total = evidence * usable * prior
    norm = total.sum(-1, keepdims=True)
    weights = np.divide(total, norm, out=np.zeros_like(total), where=norm > 0)
    valid = norm[:, 0] > 0
    for suffix, column in (("_prob", "eating_prob"), ("_start_prob", "start_prob"), ("_end_prob", "end_prob")):
        values = np.column_stack([frame[x + suffix].to_numpy(float) if x + suffix in frame
                                  else np.zeros(len(frame)) for x in EXPERTS])
        if np.any(~np.isfinite(values[usable])):
            raise ValueError("Supported selected score has NaN")
        result[column] = np.where(valid, (np.where(usable, values, 0.) * weights).sum(-1), 0.)
    result["supported"] = valid
    result["quality"] = np.where(valid, np.max(np.where(usable, reliability, 0.), axis=1), 0.)
    for j, expert in enumerate(EXPERTS):
        result[expert + "_fusion_weight"] = weights[:, j]
    result["raw_gate"] = weights[:, 1]
    result["raw_residual"] = result.eating_prob.to_numpy() - np.where(usable[:, 0], scores[:, 0], 0.)
    result.attrs.update(model_version="v9", fusion_prior_weights=prior.tolist(),
                        fusion_strict_selected=True, quality_correction=bool(quality_correction))
    return result


def decoder_candidates_v9(base=None, checkpoint=False, *, revision=None):
    fixed = dict(base or {})
    fixed = {k: v for k, v in fixed.items() if k in inspect.signature(events_from_frames).parameters}
    fixed.pop("subject_id", None)
    fixed.update(search_version="v9_checkpoint" if checkpoint else "v9",
                 smoothing_frames=3, boundary_search_s=0, min_quality=.15 if checkpoint else 0.,
                 max_duration_s=7200, max_bridge_frames=1,
                 min_high_fraction=0., min_mean_probability=0.)
    if revision == R2_REVISION:
        # The same score scale, quality barrier and evidence requirements are
        # used in snapshot selection and deployment. Only finite thresholds vary.
        fixed.update(search_version="v9_r2_checkpoint" if checkpoint else "v9_r2",
                     min_quality=.15, min_high_duration_s=60,
                     min_high_fraction=.15, min_mean_probability=.45, merge_gap_s=0)
        highs, lows = ((.55, .65), (.3, .4)) if checkpoint else (R2_HIGHS, R2_LOWS)
        for high, low, duration in itertools.product(highs, lows, DURATIONS):
            yield dict(fixed, threshold=high, low_threshold=low, min_duration_s=duration)
        return
    if checkpoint:
        # Legacy receipts retain their exact original V8 fixed 8-rule panel.
        for high, low, duration in itertools.product((.55, .65), (.2, .4), DURATIONS):
            yield dict(fixed, threshold=high, low_threshold=low, min_duration_s=duration,
                       merge_gap_s=0, min_high_duration_s=0)
    else:
        for high, low, duration, gap, high_duration in itertools.product(
                HIGHS, LOWS, DURATIONS, BRIDGES, HIGH_DURATIONS):
            yield dict(fixed, threshold=high, low_threshold=low, min_duration_s=duration,
                       merge_gap_s=gap, min_high_duration_s=high_duration)


class FrameDecodeCache:
    """Copy once; median filtering and support barriers are reused for 48 rules."""
    def __init__(self, frame):
        if "supported" not in frame:
            raise ValueError("Physical-continuity decoding requires explicit support")
        self.groups = []
        for subject, group in frame.groupby("subject_id", sort=True):
            group = group.sort_values("start_ms", kind="stable")
            p = group.eating_prob.to_numpy(float, copy=True)
            p[~group.supported.fillna(False).to_numpy(bool)] = np.nan
            self.groups.append(PreparedFrameDecoder(
                group.start_ms.to_numpy(float), group.end_ms.to_numpy(float), p, str(subject),
                start_probs=group.start_prob.to_numpy(float) if "start_prob" in group else None,
                end_probs=group.end_prob.to_numpy(float) if "end_prob" in group else None,
                quality=np.clip(np.nan_to_num(group.quality.to_numpy(float), nan=0.), 0., 1.)))

    def decode(self, decoder):
        accepted = set(inspect.signature(events_from_frames).parameters) - {"_prepared"}
        params = {k: v for k, v in decoder.items() if k in accepted}
        output = [event for group in self.groups for event in group.decode(**params)]
        return sorted(output, key=lambda event: (str(event.subject_id), event.start_ms, event.end_ms))


def truth_in_coverage(records, frame):
    truth = truth_from_records(records)
    output, coverage = [], []
    for event in truth:
        group = frame[(frame.subject_id.astype(str) == str(event.subject_id))
                      & (frame.end_ms > event.start_ms) & (frame.start_ms < event.end_ms)]
        if group.empty:
            continue
        duration = np.maximum(0., np.minimum(group.end_ms.to_numpy(float), event.end_ms)
                              - np.maximum(group.start_ms.to_numpy(float), event.start_ms)).sum()
        output.append(event)
        coverage.append(dict(subject_id=event.subject_id, meal_id=event.meal_id,
                             start_ms=event.start_ms, end_ms=event.end_ms,
                             observed_fraction=float(duration / (event.end_ms - event.start_ms))))
    return output, coverage


def score_events(predicted, truth, subjects):
    result = match_events(predicted, truth, iou_threshold=.25)
    metrics = dict(result["metrics"])
    ids = sorted(set(str(s) for s in subjects))
    event_ids = {e.subject_id for e in predicted} | {e.subject_id for e in truth}
    if event_ids - set(ids):
        raise ValueError("Undeclared event identity")
    counts = []
    for subject in ids:
        tp = sum(str(t.subject_id) == subject for _, t, _ in result["matched"])
        npred = sum(str(p.subject_id) == subject for p in predicted)
        ntruth = sum(str(t.subject_id) == subject for t in truth)
        counts.append(dict(subject_id=subject, TP=int(tp), FP=int(npred - tp), FN=int(ntruth - tp),
                           F1=_f1(tp, npred - tp, ntruth - tp)))
    loo = [dict(omitted_subject_id=row["subject_id"],
                F1=_f1(metrics["TP"] - row["TP"], metrics["FP"] - row["FP"], metrics["FN"] - row["FN"]))
           for row in counts] if len(ids) > 1 else []
    values = [row["F1"] for row in loo]
    metrics.update(selection_micro_f1=float(metrics["F1"]),
                   selection_loo_min=float(min(values)) if values else float(metrics["F1"]),
                   selection_loo_mean=float(np.mean(values)) if values else float(metrics["F1"]),
                   selection_loo_std=float(np.std(values)) if values else 0.,
                   selection_subject_count=len(ids), selection_identity_counts=counts,
                   selection_loo_f1=loo,
                   selection_loo_note="Delete-one-identity sensitivity only; no retraining or new CV")
    return metrics


def _f1(tp, fp, fn):
    denominator = 2 * tp + fp + fn
    return float(2 * tp / denominator) if denominator else 0.


def _grid_axes(rows):
    # Derive axes from the declared trial list, so old 48-rule receipts and the
    # revised 18-rule panel use identical one-coordinate neighborhood semantics.
    return tuple(sorted(set(values)) for values in (
        [float(r["alpha"]) for r in rows],
        [float(r["decoder"]["threshold"]) for r in rows],
        [float(r["decoder"]["low_threshold"]) for r in rows],
        [float(r["decoder"]["min_duration_s"]) for r in rows],
        [float(r["decoder"]["merge_gap_s"]) for r in rows],
        [float(r["decoder"]["min_high_duration_s"]) for r in rows]))


def _grid_key(row, axes):
    d = row["decoder"]
    values = (row["alpha"], d["threshold"], d["low_threshold"],
              d["min_duration_s"], d["merge_gap_s"], d["min_high_duration_s"])
    return tuple(axis.index(float(value)) for axis, value in zip(axes, values))


def select_stable_candidate(rows):
    if not rows:
        raise ValueError("No V9 selection candidates")
    indexed = {}
    axes = _grid_axes(rows)
    for index, row in enumerate(rows):
        key = _grid_key(row, axes)
        if key in indexed:
            raise ValueError("Duplicate V9 alpha/rule grid point")
        indexed[key] = index
        if not all(math.isfinite(float(row["metrics"][k])) and 0 <= float(row["metrics"][k]) <= 1
                   for k in ("F1", "selection_loo_min")):
            raise ValueError("Invalid candidate metrics")
    peak = max(float(row["metrics"]["F1"]) for row in rows)
    for key, index in indexed.items():
        neighbors = [index]
        # One alpha, high, low, duration, gap OR high-duration step; other axes fixed.
        for axis in range(6):
            for step in (-1, 1):
                neighbor = list(key)
                neighbor[axis] += step
                if tuple(neighbor) in indexed:
                    neighbors.append(indexed[tuple(neighbor)])
        values = [float(rows[n]["metrics"]["F1"]) for n in neighbors]
        rows[index]["metrics"].update(selection_neighborhood_median=float(np.median(values)),
            selection_neighborhood_size=len(values), selection_near_optimal=bool(
                float(rows[index]["metrics"]["F1"]) + F1_TOLERANCE + 1e-12 >= peak),
            selection_best_micro_f1=peak, selection_f1_tolerance=F1_TOLERANCE,
            selection_policy="near_peak_0.005_then_delete_identity_min_then_grid_neighbor_median_then_f1_then_lexicographic_v9")
    eligible = [i for i, row in enumerate(rows) if row["metrics"]["selection_near_optimal"]]
    return min(eligible, key=lambda i: (-float(rows[i]["metrics"]["selection_loo_min"]),
               -float(rows[i]["metrics"]["selection_neighborhood_median"]),
               -float(rows[i]["metrics"]["F1"]), _grid_key(rows[i], axes)))


def search_fusion_decoder(frame, records, config, *, quality_correction):
    validate_frame_table(frame)
    truth, _ = truth_in_coverage(records, frame)
    subjects = frame.subject_id.astype(str).unique()
    rows = []
    revision = R2_REVISION if is_precision_revision(config) else None
    rules = list(decoder_candidates_v9(config.get("decoder", {}), revision=revision))
    for alpha in ALPHAS:
        fused = fuse_frame_table(frame, alpha, quality_correction=quality_correction)
        prepared = FrameDecodeCache(fused)
        for decoder in rules:
            predicted = prepared.decode(decoder)
            rows.append(dict(alpha=alpha, fusion_weights=[alpha, 1. - alpha], decoder=decoder,
                             metrics=score_events(predicted, truth, subjects)))
    expected_rules = 18 if revision == R2_REVISION else 48
    if len(rows) != len(ALPHAS) * expected_rules:
        raise RuntimeError("V9 finite candidate panel size differs from its declared revision")
    selected = select_stable_candidate(rows)
    return rows[selected], rows, dict(n_candidates=len(rows), selected_index=selected,
        selected_alpha=float(rows[selected]["alpha"]), selected_decoder=rows[selected]["decoder"],
        peak_training_event_F1=max(float(r["metrics"]["F1"]) for r in rows),
        selected_training_event_F1=float(rows[selected]["metrics"]["F1"]),
        F1_tolerance=F1_TOLERANCE, quality_correction=bool(quality_correction),
        revision=revision or "plan_1.1", n_alphas=len(ALPHAS), n_decoder_rules=len(rules),
        fit_scope="training_identity_crossfit_predictions", outer_labels_used=False,
        neighborhood="Self and one grid step on any one of alpha/high/low/duration/gap/high-duration")


def fixed_rule_panel(frame, records, config):
    truth, _ = truth_in_coverage(records, frame)
    prepared = FrameDecodeCache(frame)
    rows = []
    for decoder in decoder_candidates_v9(config.get("decoder"), checkpoint=True,
            revision=R2_REVISION if is_precision_revision(config) else None):
        events = prepared.decode(decoder)
        rows.append(dict(decoder=decoder, metrics=score_events(events, truth, frame.subject_id.unique())))
    values = np.array([row["metrics"]["F1"] for row in rows])
    selected = min(range(len(rows)), key=lambda i: (-values[i], rows[i]["metrics"]["selection_loo_std"], i))
    summary = dict(mean_fixed_rule_F1=float(values.mean()), std_fixed_rule_F1=float(values.std()),
                   best_fixed_rule_F1=float(values.max()), fixed_rule_F1=values.tolist(), n_fixed_rules=8)
    return rows[selected]["decoder"], rows[selected]["metrics"], rows, summary


def select_experts(frame, records, config):
    from .selection_v8 import frame_average_precision
    result = {}
    for index, expert in enumerate(EXPERTS):
        fused = fuse_frame_table(frame, float(index == 0), quality_correction=False)
        valid = fused.supported.to_numpy(bool) & np.isfinite(fused.target) & (fused.target >= 0) & (fused.target <= 1)
        y = (fused.loc[valid, "target"].to_numpy(float) >= .5).astype(float)
        p = fused.loc[valid, "eating_prob"].to_numpy(float)
        if not len(y):
            result[expert] = dict(score=[-1., -1., -1e6], metrics=dict(selection_score=-1.,
                                  frame_AP=0., frame_BCE=1e6, selection_unavailable=True),
                                  decoder=next(decoder_candidates_v9(config.get("decoder"), True,
                                      revision=R2_REVISION if is_precision_revision(config) else None)),
                                  selection_trials=[])
            continue
        ap = frame_average_precision(y, p)
        p = np.clip(p, 1e-7, 1 - 1e-7)
        bce = float(np.mean(-y * np.log(p) - (1 - y) * np.log1p(-p)))
        decoder, metrics, trials, summary = fixed_rule_panel(fused, records, config)
        metrics.update(summary, frame_AP=ap, frame_BCE=bce, selection_score=summary["mean_fixed_rule_F1"],
            selection_unavailable=False, expert=expert, outer_labels_used=False,
            fit_subjects=sorted(frame.subject_id.astype(str).unique()), scope="inner_validation_identities_only")
        result[expert] = dict(score=[summary["mean_fixed_rule_F1"], ap, -bce],
                              metrics=metrics, decoder=decoder, selection_trials=trials)
    return result



