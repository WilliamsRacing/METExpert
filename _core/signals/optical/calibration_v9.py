
from __future__ import annotations

import hashlib
import json
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

EXPERTS = ("feature", "raw")


def table_fingerprint(frame):
    columns = [c for c in ("subject_id", "start_ms", "end_ms", "crossfit_fold",
               "target", "label_weight", "feature_logit", "raw_logit",
               "feature_prob", "raw_prob", "feature_supported", "raw_supported")
               if c in frame]
    canonical = frame[columns].sort_values(["subject_id", "start_ms"], kind="stable")
    return hashlib.sha256(canonical.to_csv(index=False, float_format="%.12g").encode()).hexdigest()


def expert_logits(frame, expert):
    support = frame[expert + "_supported"].fillna(False).to_numpy(bool)
    if expert + "_logit" in frame:
        z = frame[expert + "_logit"].to_numpy(float)
    else:
        p = frame[expert + "_prob"].to_numpy(float)
        if np.any(~np.isfinite(p[support])) or np.any((p[support] < 0) | (p[support] > 1)):
            raise ValueError("Supported " + expert + " probability is invalid")
        safe = np.clip(np.where(support, p, .5), 1e-7, 1. - 1e-7)
        z = np.log(safe) - np.log1p(-safe)
    if not np.isfinite(z[support]).all():
        raise ValueError("Supported " + expert + " logit is invalid")
    return np.where(support, z, 0.)


def apply_calibration(frame, state):
    result = frame.copy()
    result.attrs = dict(frame.attrs)
    for expert in EXPERTS:
        support = result[expert + "_supported"].fillna(False).to_numpy(bool)
        parameters = state["experts"][expert]
        temperature = float(parameters["temperature"])
        bias = float(parameters["bias"])
        if not .5 <= temperature <= 3. or not -2. <= bias <= 2.:
            raise ValueError("Calibration state lies outside fixed V9 bounds")
        calibrated = expit(expert_logits(result, expert) / temperature + bias)
        result[expert + "_prob"] = np.where(support, calibrated, np.nan)
    result.attrs["calibration_scope"] = "training_identity_crossfit_predictions"
    return result


def _diagnostics(y, p, subject):
    def summary(indices):
        target, score = y[indices], p[indices]
        bins = []
        for index in range(10):
            mask = (score >= index / 10.) & ((score < (index + 1) / 10.) if index < 9 else (score <= 1.))
            bins.append({"lower": index / 10., "upper": (index + 1) / 10.,
                         "n": int(mask.sum()), "mean_score": float(score[mask].mean()) if mask.any() else None,
                         "observed_target_fraction": float(target[mask].mean()) if mask.any() else None})
        return {"known_frames": int(len(target)),
                "unweighted_brier": float(np.mean((score - target) ** 2)) if len(target) else None,
                "reliability_bins": bins}
    rows = {str(s): summary(subject == s) for s in np.unique(subject)}
    return dict(summary(np.ones(len(y), bool)), by_subject=rows,
                note="Known, supported natural-distribution frames; confidence not used in Brier; not true physiological probability")


def fit_calibrators(frame, *, regularization=.01):
    if regularization < 0 or not np.isfinite(regularization):
        raise ValueError("Calibration regularization must be finite and nonnegative")
    required = {"subject_id", "target", "label_weight", "start_ms", "end_ms"}
    if not required.issubset(frame):
        raise ValueError("Calibration requires natural-distribution targets and label_weight")
    subject = frame.subject_id.astype(str).to_numpy()
    y_all = frame.target.to_numpy(float)
    confidence = frame.label_weight.to_numpy(float)
    duration = (frame.end_ms.to_numpy(float) - frame.start_ms.to_numpy(float)) / 1000.
    if (duration <= 0).any() or not np.isfinite(duration).all():
        raise ValueError("Calibration frames need positive finite durations")
    if not np.isfinite(confidence).all() or (confidence < 0).any():
        raise ValueError("Invalid original label confidence")
    result = {"schema_version": "v9_bounded_calibration_1",
              "scope": "training_identity_crossfit_predictions", "outer_labels_used": False,
              "temperature_bounds": [.5, 3.], "bias_bounds": [-2., 2.],
              "regularization": float(regularization), "fit_subjects": sorted(set(subject)),
              "source_fingerprint": table_fingerprint(frame), "experts": {}}
    for expert in EXPERTS:
        support = frame[expert + "_supported"].fillna(False).to_numpy(bool)
        valid = support & np.isfinite(y_all) & (y_all >= 0) & (y_all <= 1) & (confidence > 0)
        y, z, ids = y_all[valid], expert_logits(frame, expert)[valid], subject[valid]
        raw_weight = confidence[valid] * duration[valid]
        weights = raw_weight.copy()
        for identity in np.unique(ids):
            selected = ids == identity
            weights[selected] /= raw_weight[selected].sum()
        if len(weights):
            weights /= weights.sum()
        if not len(y) or not np.any(y > 0) or not np.any(y < 1):
            temperature, bias = 1., 0.
            status = "identity_fallback_insufficient_known_class_support"
            optimization = {"success": False, "message": status}
        else:
            def objective(theta):
                log_temperature, shift = theta
                scaled_z = z * np.exp(-log_temperature)
                logit = scaled_z + shift
                loss = np.sum(weights * (np.logaddexp(0., logit) - y * logit))
                loss += regularization * (log_temperature ** 2 + shift ** 2)
                error = weights * (expit(logit) - y)
                grad = np.array([np.sum(error * -scaled_z), np.sum(error)])
                grad += 2. * regularization * theta
                return float(loss), grad
            optimized = minimize(objective, np.zeros(2), method="L-BFGS-B", jac=True,
                                 bounds=[(np.log(.5), np.log(3.)), (-2., 2.)],
                                 options={"maxiter": 300, "ftol": 1e-12, "gtol": 1e-8})
            good = bool(optimized.success and np.isfinite(optimized.x).all())
            temperature, bias = (float(np.exp(optimized.x[0])), float(optimized.x[1])) if good else (1., 0.)
            status = "fitted" if good else "identity_fallback_optimizer_failed"
            optimization = {"success": good, "message": str(optimized.message),
                            "iterations": int(optimized.nit), "objective": float(optimized.fun)}
        before, after = expit(z), expit(z / temperature + bias)
        result["experts"][expert] = {
            "temperature": float(np.clip(temperature, .5, 3.)), "bias": float(np.clip(bias, -2., 2.)),
            "status": status, "supported_known_frames": int(valid.sum()),
            "fit_subjects": sorted(set(ids)), "optimization": optimization,
            "before": _diagnostics(y, before, ids), "after": _diagnostics(y, after, ids)}
    return result



def assess_calibration_transfer(frame, state, *, max_brier_increase=0.):
    """Gate optional auxiliary calibration on disjoint main-model validation.

    This is a selection diagnostic, not a new calibration fit. It explicitly
    checks the deployed score scale instead of assuming auxiliary logits and
    early-selected main-model logits have interchangeable distributions.
    """
    if not np.isfinite(max_brier_increase) or max_brier_increase < 0:
        raise ValueError("Calibration Brier tolerance must be finite and nonnegative")
    calibrated = apply_calibration(frame, state)
    y = frame.target.to_numpy(float) if "target" in frame else np.full(len(frame), np.nan)
    confidence = frame.label_weight.to_numpy(float) if "label_weight" in frame else np.ones(len(frame))
    rows = {}
    for expert in EXPERTS:
        support = frame[expert + "_supported"].fillna(False).to_numpy(bool)
        valid = support & np.isfinite(y) & (y >= 0) & (y <= 1) & np.isfinite(confidence) & (confidence > 0)
        before = frame[expert + "_prob"].to_numpy(float)
        after = calibrated[expert + "_prob"].to_numpy(float)
        known = int(valid.sum())
        before_brier = float(np.mean((before[valid] - y[valid]) ** 2)) if known else None
        after_brier = float(np.mean((after[valid] - y[valid]) ** 2)) if known else None
        background = valid & (y <= .1)
        count = int(background.sum())
        before_mass = float(np.mean(before[background] >= .3)) if count else None
        after_mass = float(np.mean(after[background] >= .3)) if count else None
        improves_or_preserves = bool(known and after_brier <= before_brier + max_brier_increase + 1e-12)
        continuation_floor = bool(count and after_mass > .5 and after_mass > before_mass + .2)
        rows[expert] = dict(known_frames=known, background_frames=count,
            before_brier=before_brier, after_brier=after_brier,
            background_fraction_ge_0_3_before=before_mass,
            background_fraction_ge_0_3_after=after_mass,
            brier_preserved=improves_or_preserves,
            background_continuation_floor_detected=continuation_floor,
            accepted=bool(improves_or_preserves and not continuation_floor))
    return dict(accepted=all(row["accepted"] for row in rows.values()), experts=rows,
        fit_scope="no_refit_main_inner_validation_transfer_diagnostic",
        max_brier_increase=float(max_brier_increase), outer_labels_used=False,
        background_guard="reject if >=0.3 background fraction rises >0.20 and exceeds 0.50",
        note="Passing this guard does not establish calibrated physiological probabilities")
