
from __future__ import annotations

import hashlib
import json
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from .events import Event, interval_iou, match_events
from .selection_v9 import FrameDecodeCache, fuse_frame_table, score_events, truth_in_coverage

FEATURE_NAMES = ["log_duration_s", "feature_mean", "feature_q90", "raw_mean", "raw_q90",
                 "fused_mean", "high_duration_fraction", "longest_high_fraction",
                 "expert_disagreement", "feature_available_fraction", "raw_quality",
                 "joint_support_fraction"]
THRESHOLDS = (.2, .3, .4, .5, .6, .7, .8)


def _weighted_mean(values, weights, default=0.):
    valid = np.isfinite(values) & (weights > 0)
    return float(np.average(values[valid], weights=weights[valid])) if valid.any() else float(default)


def _weighted_quantile(values, weights, q=.9):
    valid = np.isfinite(values) & (weights > 0)
    if not valid.any():
        return 0.
    order = np.argsort(values[valid], kind="stable")
    values, weights = values[valid][order], weights[valid][order]
    return float(values[min(np.searchsorted(np.cumsum(weights), q * weights.sum()), len(values) - 1)])


def _physical_runs(frame):
    assigned = {}
    for subject, group in frame.groupby("subject_id", sort=True):
        group = group.sort_values("start_ms", kind="stable")
        last_end, run = None, -1
        any_support = group.feature_supported.fillna(False).to_numpy(bool) | group.raw_supported.fillna(False).to_numpy(bool)
        for (_, row), supported in zip(group.iterrows(), any_support):
            if not supported:
                last_end = None
                continue
            if last_end is None or abs(float(row.start_ms) - last_end) > 1e-6:
                run += 1
            assigned[(str(subject), float(row.start_ms))] = run
            last_end = float(row.end_ms)
    return assigned


def candidate_features(event, frame):
    group = frame[(frame.subject_id.astype(str) == str(event.subject_id)) &
                  (frame.end_ms > event.start_ms) & (frame.start_ms < event.end_ms)].sort_values("start_ms")
    overlaps = np.maximum(0., np.minimum(group.end_ms.to_numpy(float), event.end_ms)
                          - np.maximum(group.start_ms.to_numpy(float), event.start_ms))
    fs = group.feature_supported.fillna(False).to_numpy(bool)
    rs = group.raw_supported.fillna(False).to_numpy(bool)
    fp, rp = group.feature_prob.to_numpy(float), group.raw_prob.to_numpy(float)
    fused = group.eating_prob.to_numpy(float)
    total = overlaps.sum()
    if total <= 0:
        raise ValueError("Candidate has no observed frames")
    high = np.isfinite(fused) & (fused >= .5) & group.supported.fillna(False).to_numpy(bool)
    longest, current, previous_end = 0., 0., None
    for (_, row), is_high, weight in zip(group.iterrows(), high, overlaps):
        contiguous = previous_end is not None and abs(float(row.start_ms) - previous_end) <= 1e-6
        current = (current if contiguous else 0.) + weight if is_high else 0.
        longest = max(longest, current)
        previous_end = float(row.end_ms)
    feature_availability = (group["feature_availability"].to_numpy(float)
                            if "feature_availability" in group else group.feature_reliability.to_numpy(float))
    raw_quality = group.raw_reliability.to_numpy(float)
    values = [np.log1p((event.end_ms - event.start_ms) / 1000.),
              _weighted_mean(fp, overlaps * fs), _weighted_quantile(fp, overlaps * fs),
              _weighted_mean(rp, overlaps * rs), _weighted_quantile(rp, overlaps * rs),
              _weighted_mean(fused, overlaps * group.supported.to_numpy(bool)),
              float(overlaps[high].sum() / total), float(longest / total),
              _weighted_mean(np.abs(fp - rp), overlaps * fs * rs),
              _weighted_mean(feature_availability, overlaps * fs),
              _weighted_mean(raw_quality, overlaps * rs),
              float(overlaps[fs & rs].sum() / total)]
    return np.asarray(values, float)


def generate_candidates(calibrated_frame, alpha):
    fused = fuse_frame_table(calibrated_frame, alpha, quality_correction=False)
    sources = [("feature", 1.), ("raw", 0.), ("fused", float(alpha))]
    run_ids = _physical_runs(calibrated_frame)
    unique = {}
    for source, weight in sources:
        prepared = FrameDecodeCache(fuse_frame_table(calibrated_frame, weight, quality_correction=False))
        for high in (.4, .5):
            for low in (.15, .25):
                decoder = dict(threshold=high, low_threshold=low, min_duration_s=60,
                               max_duration_s=7200, merge_gap_s=30, smoothing_frames=3,
                               min_quality=0., boundary_search_s=0, max_bridge_frames=1,
                               min_high_duration_s=0)
                for event in prepared.decode(decoder):
                    key = (str(event.subject_id), float(event.start_ms), float(event.end_ms))
                    if key not in unique:
                        run = run_ids.get((str(event.subject_id), float(event.start_ms)))
                        if run is None:
                            raise ValueError("Candidate starts outside observed support")
                        unique[key] = dict(event=event, continuous_run=int(run), sources=[])
                    unique[key]["sources"].append(dict(source=source, high=high, low=low))
    rows = [unique[k] for k in sorted(unique)]
    for row in rows:
        row["features"] = candidate_features(row["event"], fused)
    return rows


def temporal_nms(rows, scores, threshold, *, overlap_threshold=.5):
    order = sorted(range(len(rows)), key=lambda i: (-float(scores[i]), str(rows[i]["event"].subject_id),
                    rows[i]["event"].start_ms, rows[i]["event"].end_ms))
    keep = []
    for index in order:
        if float(scores[index]) < float(threshold):
            continue
        event = rows[index]["event"]
        if any(str(rows[j]["event"].subject_id) == str(event.subject_id)
               and rows[j]["continuous_run"] == rows[index]["continuous_run"]
               and interval_iou(event, rows[j]["event"]) > overlap_threshold for j in keep):
            continue
        keep.append(index)
    return [Event(rows[i]["event"].subject_id, rows[i]["event"].start_ms,
                  rows[i]["event"].end_ms, score=float(scores[i])) for i in keep]


def _candidate_labels(rows, frame, truth):
    labels, weights, meal_keys = [], [], []
    for row in rows:
        event = row["event"]
        meals = [meal for meal in truth if str(meal.subject_id) == str(event.subject_id)]
        ious = [interval_iou(event, meal) for meal in meals]
        best = int(np.argmax(ious)) if ious else -1
        group = frame[(frame.subject_id.astype(str) == str(event.subject_id)) &
                      (frame.end_ms > event.start_ms) & (frame.start_ms < event.end_ms)]
        overlap = np.maximum(0., np.minimum(group.end_ms.to_numpy(float), event.end_ms)
                             - np.maximum(group.start_ms.to_numpy(float), event.start_ms))
        y, confidence = group.target.to_numpy(float), group.label_weight.to_numpy(float)
        if best >= 0 and ious[best] > .25:
            label = 1
            positive = (y > 0) & np.isfinite(y) & (confidence > 0)
            weight = _weighted_mean(confidence, overlap * positive, default=1.)
            meal = meals[best]
            meal_key = (str(meal.subject_id), float(meal.start_ms), float(meal.end_ms))
        elif not any(x > 0 for x in ious):
            known = np.isfinite(y) & (y >= 0) & (y < .5) & (confidence > 0)
            # Unknown/positive frame evidence may not be promoted to a certain negative.
            if not np.all(known[overlap > 0]):
                label, weight, meal_key = -1, 0., None
            else:
                label, weight, meal_key = 0, _weighted_mean(confidence, overlap), None
        else:
            label, weight, meal_key = -1, 0., None
        labels.append(label)
        weights.append(weight)
        meal_keys.append(meal_key)
    labels, weights = np.asarray(labels, int), np.asarray(weights, float)
    for key in {key for key in meal_keys if key is not None}:
        indices = np.array([value == key for value in meal_keys])
        denominator = weights[indices].sum()
        if denominator > 0:
            weights[indices] /= denominator
    return labels, weights


def _scores(rows, state):
    if not rows:
        return np.zeros(0, float)
    x = np.stack([row["features"] for row in rows])
    x = (x - np.asarray(state["feature_mean"])) / np.asarray(state["feature_scale"])
    return expit(x @ np.asarray(state["coef"]) + float(state["intercept"]))


def fit_verifier(calibrated_frame, records, alpha, *, regularization=.01):
    rows = generate_candidates(calibrated_frame, alpha)
    truth, _ = truth_in_coverage(records, calibrated_frame)
    labels, weights = _candidate_labels(rows, calibrated_frame, truth)
    accepted = (labels >= 0) & (weights > 0)
    digest_data = [(row["event"].subject_id, row["event"].start_ms, row["event"].end_ms,
                    row["features"].tolist(), int(label), float(weight))
                   for row, label, weight in zip(rows, labels, weights)]
    state = dict(schema_version="v9_event_verifier_1", enabled=False, alpha=float(alpha),
                 scope="training_identity_crossfit_predictions", outer_labels_used=False,
                 feature_names=FEATURE_NAMES, regularization=float(regularization),
                 n_candidates=len(rows), n_positive=int(np.sum(labels == 1)),
                 n_negative=int(np.sum(labels == 0)), n_ambiguous_or_unknown=int(np.sum(labels < 0)),
                 proposal_table_fingerprint=hashlib.sha256(json.dumps(digest_data, sort_keys=True).encode()).hexdigest())
    hit = sum(any(interval_iou(row["event"], meal) > .25 for row in rows
                  if str(row["event"].subject_id) == str(meal.subject_id)) for meal in truth)
    state["candidate_recall_ceiling"] = dict(covered_truth=int(hit), n_truth=len(truth),
                                          recall=float(hit / len(truth)) if truth else 0.,
                                          definition="Any proposal has strict IoU > .25; not one-to-one final recall")
    if not accepted.any() or len(np.unique(labels[accepted])) < 2:
        state["unavailable_reason"] = "Verifier requires positive and trusted/weak negative training candidates"
        return state
    x = np.stack([row["features"] for row in rows])[accepted]
    y, w = labels[accepted].astype(float), weights[accepted]
    w = w / w.sum()
    mean = np.average(x, weights=w, axis=0)
    scale = np.sqrt(np.average((x - mean) ** 2, weights=w, axis=0))
    scale = np.where(scale > 1e-6, scale, 1.)
    x = (x - mean) / scale
    def objective(theta):
        score = x @ theta[:-1] + theta[-1]
        loss = np.sum(w * (np.logaddexp(0., score) - y * score))
        loss += regularization * np.sum(theta[:-1] ** 2)
        error = w * (expit(score) - y)
        return float(loss), np.r_[x.T @ error + 2 * regularization * theta[:-1], error.sum()]
    fit = minimize(objective, np.zeros(len(FEATURE_NAMES) + 1), method="L-BFGS-B",
                   jac=True, options={"maxiter": 500, "ftol": 1e-12})
    if not fit.success or not np.isfinite(fit.x).all():
        state["unavailable_reason"] = "Verifier optimizer failed: " + str(fit.message)
        return state
    state.update(enabled=True, feature_mean=mean.tolist(), feature_scale=scale.tolist(),
                 coef=fit.x[:-1].tolist(), intercept=float(fit.x[-1]), nms_iou=.5,
                 fit_subjects=sorted(calibrated_frame.subject_id.astype(str).unique()))
    scores = _scores(rows, state)
    trials = []
    for threshold in THRESHOLDS:
        events = temporal_nms(rows, scores, threshold)
        metrics = score_events(events, truth, calibrated_frame.subject_id.unique())
        trials.append(dict(threshold=threshold, metrics=metrics))
    peak = max(row["metrics"]["F1"] for row in trials)
    for i, row in enumerate(trials):
        nearby = trials[max(0, i - 1):min(len(trials), i + 2)]
        row["metrics"]["selection_neighborhood_median"] = float(np.median([x["metrics"]["F1"] for x in nearby]))
    eligible = [i for i, row in enumerate(trials) if row["metrics"]["F1"] + .005 + 1e-12 >= peak]
    selected = min(eligible, key=lambda i: (-trials[i]["metrics"]["selection_loo_min"],
               -trials[i]["metrics"]["selection_neighborhood_median"], -trials[i]["metrics"]["F1"],
               trials[i]["threshold"]))
    state.update(threshold=trials[selected]["threshold"], threshold_trials=trials,
                 selected_training_metrics=trials[selected]["metrics"],
                 training_metrics_note="Fit and threshold select reuse training crossfit table; not independent validation")
    return state


def apply_verifier(calibrated_frame, state):
    if not state.get("enabled"):
        raise ValueError("Cannot apply disabled/unavailable V9 verifier")
    if state.get("feature_names") != FEATURE_NAMES:
        raise ValueError("Verifier feature schema mismatch")
    rows = generate_candidates(calibrated_frame, float(state["alpha"]))
    scores = _scores(rows, state)
    events = temporal_nms(rows, scores, float(state["threshold"]), overlap_threshold=float(state.get("nms_iou", .5)))
    return events, dict(candidate_count=len(rows), accepted_count=len(events),
                        nms_iou=float(state.get("nms_iou", .5)),
                        events_are_retained_candidates_without_boundary_expansion=True)

