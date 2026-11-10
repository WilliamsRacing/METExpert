
from __future__ import annotations

import numpy as np

EXPERTS = ("feature", "raw")
FUSION_PANEL = tuple((i / 10., 1. - i / 10.) for i in range(11))


def _priors(value):
    prior = (np.array([float(value), 1. - float(value)], dtype=float)
             if np.isscalar(value) else np.asarray(value, dtype=float))
    if prior.shape != (2,) or not np.isfinite(prior).all() or (prior < 0).any() or not np.isclose(prior.sum(), 1., atol=np.finfo(np.float32).eps, rtol=0):
        raise ValueError("V8 priors must be a finite nonnegative length-2 simplex")
    return prior


def fuse_frame_table(frame, priors, *, strict_selected=True):
    if not strict_selected:
        raise ValueError("V8 always strictly disables zero-prior experts")
    prior = _priors(priors)
    result = frame.copy()
    result.attrs = dict(frame.attrs)
    result.attrs.update(fusion_strict_selected=True, fusion_prior_weights=prior.tolist(),
                        fusion_expert_order=list(EXPERTS), model_version="v8")
    p = np.stack([frame[x + "_prob"].to_numpy(dtype=float) for x in EXPERTS], -1)
    q = np.stack([frame[x + "_reliability"].to_numpy(dtype=float) for x in EXPERTS], -1)
    supported = np.stack([frame[x + "_supported"].fillna(False).to_numpy(dtype=bool)
                          for x in EXPERTS], -1)
    usable = supported & (prior[None, :] > 0)
    if np.any(~np.isfinite(p[usable])) or np.any((p[usable] < 0) | (p[usable] > 1)):
        raise ValueError("A selected supported expert has invalid probabilities")
    q = np.clip(np.nan_to_num(q, nan=0., posinf=0., neginf=0.), 0., 1.)
    evidence = (.5 + .5 * q) * usable
    unnormalized = evidence * prior[None, :]
    total = unnormalized.sum(-1, keepdims=True)
    weights = np.divide(unnormalized, np.maximum(total, 1e-12))
    valid = total[:, 0] > 0
    for suffix, column in (("_prob", "eating_prob"), ("_start_prob", "start_prob"),
                           ("_end_prob", "end_prob")):
        source = [frame[x + suffix].to_numpy(dtype=float) if x + suffix in frame
                  else np.zeros(len(frame), dtype=float) for x in EXPERTS]
        v = np.stack(source, -1)
        if np.any(~np.isfinite(v[usable])):
            raise ValueError("A selected supported expert has a nonfinite boundary score")
        result[column] = np.where(valid, np.clip((np.where(usable, v, 0.) * weights).sum(-1),
                                               1e-6, 1 - 1e-6), 0.)
    result["supported"] = valid
    result["quality"] = np.where(weights > 0, q, 0.).max(-1)
    result["raw_gate"] = weights[:, 1]
    result["raw_residual"] = result.eating_prob.to_numpy() - np.where(usable[:, 0], p[:, 0], 0.)
    for j, expert in enumerate(EXPERTS):
        result[expert + "_fusion_weight"] = weights[:, j]
    return result


def select_fusion_decoder(frame, records, config, *, checkpoint=False, weights=None):
    """Freeze two independently selected experts before calling this function.

    Returns (decoder, metrics, trial_rows), matching the earlier training API.
    No labels from outer identities may be passed; fit identity IDs are recorded.
    """
    from .selection_v8 import ALPHAS, _rules, select_stable_candidate, F1_TOLERANCE
    if checkpoint:
        raise ValueError("V8 checkpoint snapshots are selected independently, not by fusion grid")
    if weights is not None and not np.allclose(np.asarray(weights), np.asarray(FUSION_PANEL)):
        raise ValueError("V8 final fusion uses the predeclared alpha 0:0.1:1 grid")
    rows = []
    for index, alpha in enumerate(ALPHAS):
        prior = [alpha, 1. - alpha]
        table = fuse_frame_table(frame, prior)
        for trial in _rules(table, records, config, checkpoint=False):
            rows.append(dict(trial, alpha=float(alpha), fusion_weights=prior,
                             fusion_candidate_index=index))
    if len(rows) != 528:
        raise RuntimeError("V8 final selection must evaluate all 528 fixed candidates")
    chosen = select_stable_candidate(rows)
    selected = rows[chosen]
    metrics = dict(selected["metrics"])
    peak_f1 = max(float(row["metrics"]["F1"]) for row in rows)
    subjects = sorted(str(s) for s in frame.subject_id.unique())
    summary = {
        "scope": "inner_validation_identities_only", "fit_subjects": subjects,
        "caller_responsible_for_inner_split": True, "outer_labels_used": False,
        "expert_order": list(EXPERTS), "n_fusion_priors": 11,
        "n_rules_per_prior": 48, "n_candidates": 528, "selected_index": chosen,
        "selected_alpha": selected["alpha"], "selected_weights": selected["fusion_weights"],
        "selected_decoder": dict(selected["decoder"]), "exact_peak_event_F1": peak_f1,
        "selected_event_F1": float(metrics["F1"]), "F1_tolerance": F1_TOLERANCE,
        "F1_sacrificed_from_peak": float(peak_f1 - metrics["F1"]),
        "near_optimal_candidate_count": sum(bool(r["metrics"]["selection_near_optimal"]) for r in rows),
        "selection_policy": metrics["selection_policy"],
        "deletion_note": "Minimum pooled F1 after deleting one validation identity; not new CV",
        "neighborhood_note": "Self and one alpha/high/low grid step; same duration and bridge",
        "strict_selected": True, "temperature_calibration": "not_applied",
    }
    metrics.update(fusion_weights=list(selected["fusion_weights"]),
                   fusion_scope="inner_validation_identities_only",
                   fusion_expert_order=list(EXPERTS), fusion_strict_selected=True,
                   selection_summary=summary)
    return dict(selected["decoder"]), metrics, rows
