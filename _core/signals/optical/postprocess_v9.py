
from __future__ import annotations

import copy
import json
import numpy as np
from .calibration_v9 import apply_calibration, fit_calibrators, table_fingerprint, assess_calibration_transfer
from .selection_v9 import FrameDecodeCache, F1_TOLERANCE, fuse_frame_table, score_events, search_fusion_decoder, truth_in_coverage, validate_frame_table, is_precision_revision, R2_REVISION


def validate_crossfit_provenance(train_frame, train_records, inner_frame,
                                 inner_records, provenance):
    if not isinstance(provenance, dict):
        raise ValueError("V9 postprocessing requires explicit crossfit provenance")
    required = {"training_subjects", "inner_subjects", "outer_subjects", "crossfit_folds"}
    if required - set(provenance):
        raise ValueError("Crossfit provenance missing: " + str(sorted(required - set(provenance))))
    training = set(map(str, provenance["training_subjects"]))
    inner = set(map(str, provenance["inner_subjects"]))
    outer = set(map(str, provenance["outer_subjects"]))
    if not training or not inner or training & inner or training & outer or inner & outer:
        raise ValueError("Training/internal/outer identities must be nonempty where needed and disjoint")
    if set(train_frame.subject_id.astype(str)) != training or set(inner_frame.subject_id.astype(str)) != inner:
        raise ValueError("Prediction identity sets do not match provenance")
    if set(str(r["subject_id"]) for r in train_records) - training:
        raise ValueError("Training records contain internal/outer identities")
    if set(str(r["subject_id"]) for r in inner_records) - inner:
        raise ValueError("Internal records contain training/outer identities")
    if "crossfit_fold" not in train_frame:
        raise ValueError("Training prediction rows must carry crossfit_fold provenance")
    folds = provenance["crossfit_folds"]
    if len(folds) != 3:
        raise ValueError("V9 plan 1.1 requires exactly three fixed training-side crossfit folds")
    coverage, identifiers = set(), set()
    audited = []
    for index, fold in enumerate(folds):
        identifier = str(fold.get("fold_id", index))
        if identifier in identifiers:
            raise ValueError("Duplicate crossfit fold ID")
        identifiers.add(identifier)
        fit = set(map(str, fold["train_subjects"]))
        predicted = set(map(str, fold["predict_subjects"]))
        if not predicted or fit & predicted or predicted - training or fit != training - predicted:
            raise ValueError("Auxiliary fit must use only the other training identities, excluding its prediction group")
        if coverage & predicted:
            raise ValueError("Crossfit identity predicted by more than one auxiliary model")
        actual = set(train_frame.loc[train_frame.crossfit_fold.astype(str) == identifier, "subject_id"].astype(str))
        if actual != predicted:
            raise ValueError("Crossfit row fold tags disagree with held-out identities")
        coverage.update(predicted)
        audited.append(dict(fold_id=identifier, train_subjects=sorted(fit),
                            predict_subjects=sorted(predicted),
                            checkpoint_fingerprint=fold.get("checkpoint_fingerprint"),
                            held_out_identity_labels_used_for_auxiliary_training=False))
    if coverage != training or set(train_frame.crossfit_fold.astype(str)) != identifiers:
        raise ValueError("Crossfit groups must cover every training identity exactly once")
    return dict(training_subjects=sorted(training), inner_subjects=sorted(inner),
                outer_subjects=sorted(outer), crossfit_folds=audited,
                fit_scope="training_identity_crossfit_predictions",
                inner_scope="fixed_complete_candidate_selection_only",
                outer_labels_used=False, source_table_fingerprint=table_fingerprint(train_frame),
                note="Identity manifests and row tags are checked; caller preserves checkpoint/training proof")


def fallback_candidate(config=None):
    if is_precision_revision(config):
        return dict(candidate_id="C0", calibrated=False, quality_correction=True, alpha=.5,
            decoder=dict(search_version="v9_r2_c0", threshold=.65, low_threshold=.4,
                min_duration_s=300, max_duration_s=7200, merge_gap_s=0, smoothing_frames=3,
                min_quality=.15, boundary_search_s=0, boundary_threshold=.5,
                min_high_duration_s=60, min_high_fraction=.15, min_mean_probability=.45,
                max_bridge_frames=1), complexity=0,
            training_selection="Predeclared uncalibrated evidence-guarded fallback")
    return dict(candidate_id="C0", calibrated=False, quality_correction=True, alpha=.5,
        decoder=dict(search_version="v9_c0", threshold=.6, low_threshold=.2,
                     min_duration_s=300, max_duration_s=7200, merge_gap_s=0,
                     smoothing_frames=3, min_quality=.15, boundary_search_s=0,
                     boundary_threshold=.5, min_high_duration_s=0,
                     min_high_fraction=0., min_mean_probability=0., max_bridge_frames=1),
        complexity=0, training_selection="Predeclared uncalibrated V8-quality fallback")


def _apply_candidate(frame, candidate, calibration, decode=True):
    scores = apply_calibration(frame, calibration) if candidate["calibrated"] else frame
    fused = fuse_frame_table(scores, float(candidate["alpha"]),
                             quality_correction=bool(candidate["quality_correction"]))
    verifier_receipt = None
    if decode:
        if candidate["candidate_id"] == "C3":
            from .verifier_v9 import apply_verifier
            events, verifier_receipt = apply_verifier(scores, candidate["verifier"])
        else:
            events = FrameDecodeCache(fused).decode(candidate["decoder"])
    else:
        events = []
    fused.attrs.update(postprocess_candidate=candidate["candidate_id"],
                       postprocess_version=("v9_r2_precision_repair" if candidate.get("revision") == R2_REVISION
                                            else "v9_plan_1.1"))
    return dict(frame=fused, events=events, candidate_id=candidate["candidate_id"],
                verifier_receipt=verifier_receipt)


def _fit_postprocessor_legacy(train_frame, train_records, inner_frame, inner_records, config,
                      *, provenance=None):
    validate_frame_table(train_frame)
    validate_frame_table(inner_frame)
    train_records, inner_records = list(train_records), list(inner_records)
    options = config.get("postprocess", {})
    if provenance is None:
        provenance = options.get("provenance")
    audit = validate_crossfit_provenance(train_frame, train_records, inner_frame, inner_records, provenance)
    if "label_weight" not in train_frame:
        raise ValueError("Training crossfit table must preserve original label_weight, not replay frequency")
    calibration = fit_calibrators(train_frame,
                                 regularization=float(options.get("calibration_regularization", .01)))
    calibrated_train = apply_calibration(train_frame, calibration)
    candidates = {"C0": fallback_candidate()}
    training_trials, training_selection = {}, {}
    for identifier, quality in (("C1", False), ("C2", True)):
        chosen, rows, summary = search_fusion_decoder(calibrated_train, train_records, config,
                                                     quality_correction=quality)
        candidates[identifier] = dict(candidate_id=identifier, calibrated=True,
            quality_correction=quality, alpha=float(chosen["alpha"]),
            decoder=copy.deepcopy(chosen["decoder"]), complexity=int(identifier[1:]),
            training_metrics=copy.deepcopy(chosen["metrics"]), training_selection=summary)
        training_selection[identifier] = summary
        if options.get("save_search_trials", True):
            training_trials[identifier] = rows
    verifier_status = "disabled_before_R6"
    if options.get("event_verifier", False):
        from .verifier_v9 import fit_verifier
        verifier = fit_verifier(calibrated_train, train_records, candidates["C1"]["alpha"],
                               regularization=.01)
        verifier_status = "available" if verifier.get("enabled") else verifier.get("unavailable_reason")
        if verifier.get("enabled"):
            candidates["C3"] = dict(candidate_id="C3", calibrated=True, quality_correction=False,
                alpha=candidates["C1"]["alpha"], decoder=None, complexity=3, verifier=verifier)
    inner_truth, coverage = truth_in_coverage(inner_records, inner_frame)
    inner_rows = []
    for identifier in sorted(candidates):
        output = _apply_candidate(inner_frame, candidates[identifier], calibration)
        metrics = score_events(output["events"], inner_truth, inner_frame.subject_id.astype(str).unique())
        if identifier == "C3":
            from .verifier_v9 import generate_candidates
            from .events import interval_iou
            calibrated_inner = apply_calibration(inner_frame, calibration)
            proposals = generate_candidates(calibrated_inner, candidates[identifier]["alpha"])
            covered = sum(any(str(row["event"].subject_id) == str(meal.subject_id)
                              and interval_iou(row["event"], meal) > .25 for row in proposals)
                          for meal in inner_truth)
            metrics["candidate_recall_ceiling"] = dict(covered_truth=int(covered), n_truth=len(inner_truth),
                recall=float(covered / len(inner_truth)) if inner_truth else 0.)
        inner_rows.append(dict(candidate_id=identifier, complexity=candidates[identifier]["complexity"],
                               metrics=metrics))
    non_verifier_peak = max(row["metrics"]["F1"] for row in inner_rows if row["candidate_id"] != "C3")
    admissible = [row for row in inner_rows if row["candidate_id"] != "C3"
                  or row["metrics"]["F1"] + 1e-12 >= non_verifier_peak + .005]
    for row in inner_rows:
        row["eligible_verifier_gain"] = bool(row in admissible)
        if row["candidate_id"] == "C3":
            row["minimum_required_F1_gain_over_best_nonverifier"] = .005
    peak = max(row["metrics"]["F1"] for row in admissible)
    eligible = [row for row in admissible if row["metrics"]["F1"] + F1_TOLERANCE + 1e-12 >= peak]
    winner = min(eligible, key=lambda row: (-row["metrics"]["selection_loo_min"],
                 row["complexity"], -row["metrics"]["F1"], row["candidate_id"]))
    state = dict(schema_version="v9_postprocess_1", plan_version="1.1",
        selected_candidate=winner["candidate_id"], expert_order=["feature", "raw"],
        calibration=calibration, candidates=candidates,
        training_selection=training_selection, training_trials=training_trials,
        inner_selection=dict(rows=inner_rows, selected_candidate=winner["candidate_id"],
            selected_metrics=winner["metrics"], peak_internal_F1=float(peak), F1_tolerance=.005,
            policy="near_peak_then_delete_identity_min_then_simpler_then_F1; C3 needs >=0.005 improvement",
            fit_subjects=sorted(inner_frame.subject_id.astype(str).unique()),
            fixed_candidates_only=True, outer_labels_used=False),
        verifier_status=verifier_status, provenance=audit,
        receipt=dict(training_frame_fingerprint=table_fingerprint(train_frame),
                     inner_frame_fingerprint=table_fingerprint(inner_frame),
                     inner_observed_meals=len(coverage),
                     calibrated_probabilities_are_not_claimed_true_physiological_probabilities=True,
                     training_optimal_F1_is_not_independent_validation=True,
                     developer_OOF_not_new_blind_test=True))
    selected = candidates[winner["candidate_id"]]
    state["decoder"] = copy.deepcopy(selected.get("decoder")) or {"search_version": "v9_event_verifier", "uses_candidate_verifier": True}
    state["event_verifier"] = copy.deepcopy(candidates.get("C3", {}).get("verifier"))
    state["fusion_weights"] = [selected["alpha"], 1. - selected["alpha"]]
    # Guarantee portable exports contain no ndarray/NaN/optimizer Python objects.
    json.dumps(state, allow_nan=False)
    return state


def _identity_calibration():
    return dict(schema_version="v9_bounded_calibration_1", scope="disabled_identity_mapping",
        outer_labels_used=False,
        experts={name: dict(temperature=1., bias=0., status="identity_calibration_disabled")
                 for name in ("feature", "raw")})


def _r2_inner_provenance(train_records, inner_frame, inner_records, provenance):
    inner = set(inner_frame.subject_id.astype(str))
    declared_inner = {str(r["subject_id"]) for r in inner_records}
    training = {str(r["subject_id"]) for r in train_records}
    if not inner or inner != declared_inner or training & inner:
        raise ValueError("Main inner validation must match its records and exclude every training identity")
    outer = set()
    if provenance is not None:
        if not isinstance(provenance, dict):
            raise ValueError("V9 r2 provenance must be an identity manifest")
        if set(map(str, provenance.get("inner_subjects", inner))) != inner:
            raise ValueError("Internal predictions disagree with the r2 identity manifest")
        declared_training = set(map(str, provenance.get("training_subjects", training)))
        if training and declared_training != training:
            raise ValueError("Training records disagree with the r2 identity manifest")
        training = declared_training
        outer = set(map(str, provenance.get("outer_subjects", [])))
        if training & inner or outer & (training | inner):
            raise ValueError("Training/internal/outer identities overlap in r2 provenance")
    return dict(training_subjects=sorted(training), inner_subjects=sorted(inner),
        outer_subjects=sorted(outer), fit_scope="main_inner_validation_identities_only",
        inner_scope="expert_snapshot_and_finite_postprocessing_selection",
        outer_labels_used=False, training_side_crossfit_used=False,
        note="The same fixed internal identities select snapshots and final rules; this is development selection, not an independent calibration validation")


def _fit_postprocessor_r2(train_frame, train_records, inner_frame, inner_records, config,
                          *, provenance=None):
    validate_frame_table(inner_frame)
    train_records, inner_records = list(train_records), list(inner_records)
    options = config.get("postprocess", {})
    if provenance is None:
        provenance = options.get("provenance")
    if train_frame is None:
        audit = _r2_inner_provenance(train_records, inner_frame, inner_records, provenance)
        fit_frame, fit_records = inner_frame, inner_records
    else:
        validate_frame_table(train_frame)
        audit = validate_crossfit_provenance(train_frame, train_records, inner_frame,
                                             inner_records, provenance)
        audit["training_side_crossfit_used"] = True
        fit_frame, fit_records = train_frame, train_records
    scope = audit["fit_scope"]
    candidates = {"C0": fallback_candidate(config)}
    selections, trials = {}, {}
    # The uncalibrated optimized route is unconditional. A failed calibration
    # can never force deployment back to one fixed alpha/decoder combination.
    for identifier, quality in (("U0", False), ("U1", True)):
        chosen, rows, summary = search_fusion_decoder(fit_frame, fit_records, config,
                                                     quality_correction=quality)
        summary["fit_scope"] = scope
        candidates[identifier] = dict(candidate_id=identifier, calibrated=False,
            quality_correction=quality, alpha=float(chosen["alpha"]),
            decoder=copy.deepcopy(chosen["decoder"]), complexity=1 + int(quality),
            training_metrics=copy.deepcopy(chosen["metrics"]), training_selection=summary)
        selections[identifier] = summary
        if options.get("save_search_trials", True):
            trials[identifier] = rows
    calibration = _identity_calibration()
    transfer = dict(accepted=False, status="disabled_by_fixed_revision_config")
    if options.get("calibration_enabled", False):
        if train_frame is None:
            raise ValueError("Optional r2 calibration requires training-side crossfit predictions; main internal labels cannot fit it")
        calibration = fit_calibrators(train_frame,
            regularization=float(options.get("calibration_regularization", .01)))
        transfer = assess_calibration_transfer(inner_frame, calibration,
            max_brier_increase=float(options.get("calibration_max_brier_increase", 0.)))
        if transfer["accepted"]:
            calibrated_train = apply_calibration(train_frame, calibration)
            for identifier, quality in (("C1", False), ("C2", True)):
                chosen, rows, summary = search_fusion_decoder(calibrated_train, train_records, config,
                                                             quality_correction=quality)
                summary["fit_scope"] = "training_identity_crossfit_predictions"
                candidates[identifier] = dict(candidate_id=identifier, calibrated=True,
                    quality_correction=quality, alpha=float(chosen["alpha"]),
                    decoder=copy.deepcopy(chosen["decoder"]), complexity=3 + int(quality),
                    training_metrics=copy.deepcopy(chosen["metrics"]), training_selection=summary)
                selections[identifier] = summary
                if options.get("save_search_trials", True):
                    trials[identifier] = rows
    if options.get("event_verifier", False):
        raise ValueError("V9 r2 precision repair keeps event verifier disabled; use a separately declared ablation")
    inner_truth, coverage = truth_in_coverage(inner_records, inner_frame)
    inner_rows = []
    for identifier in sorted(candidates):
        candidates[identifier]["revision"] = R2_REVISION
        output = _apply_candidate(inner_frame, candidates[identifier], calibration)
        metrics = score_events(output["events"], inner_truth, inner_frame.subject_id.astype(str).unique())
        inner_rows.append(dict(candidate_id=identifier,
                               complexity=candidates[identifier]["complexity"], metrics=metrics))
    peak = max(row["metrics"]["F1"] for row in inner_rows)
    eligible = [row for row in inner_rows if row["metrics"]["F1"] + F1_TOLERANCE + 1e-12 >= peak]
    winner = min(eligible, key=lambda row: (-row["metrics"]["selection_loo_min"],
                 row["complexity"], -row["metrics"]["F1"], row["candidate_id"]))
    selected = candidates[winner["candidate_id"]]
    state = dict(schema_version="v9_postprocess_2", plan_version=R2_REVISION,
        calibrated=bool(selected["calibrated"]), fit_scope=scope,
        selected_candidate=winner["candidate_id"], expert_order=["feature", "raw"],
        calibration=calibration, calibration_transfer=transfer, candidates=candidates,
        training_selection=selections, training_trials=trials,
        inner_selection=dict(rows=inner_rows, selected_candidate=winner["candidate_id"],
            selected_metrics=winner["metrics"], peak_internal_F1=float(peak), F1_tolerance=F1_TOLERANCE,
            policy="near_peak_then_delete_identity_min_then_simpler_then_F1",
            fit_subjects=sorted(inner_frame.subject_id.astype(str).unique()),
            fixed_candidates_only=True, finite_grid_fitted_on=scope, outer_labels_used=False,
            independent_validation=False),
        verifier_status="disabled_in_r2_precision_repair", provenance=audit,
        decoder=copy.deepcopy(selected["decoder"]), event_verifier=None,
        fusion_weights=[selected["alpha"], 1. - selected["alpha"]],
        receipt=dict(training_frame_fingerprint=table_fingerprint(train_frame) if train_frame is not None else None,
            fit_frame_fingerprint=table_fingerprint(fit_frame), inner_frame_fingerprint=table_fingerprint(inner_frame),
            inner_observed_meals=len(coverage), calibrated_probabilities_are_not_claimed_true_physiological_probabilities=True,
            training_optimal_F1_is_not_independent_validation=True,
            developer_OOF_not_new_blind_test=True))
    json.dumps(state, allow_nan=False)
    return state


def fit_postprocessor(train_frame, train_records, inner_frame, inner_records, config,
                      *, provenance=None):
    if is_precision_revision(config):
        return _fit_postprocessor_r2(train_frame, train_records, inner_frame, inner_records,
                                     config, provenance=provenance)
    return _fit_postprocessor_legacy(train_frame, train_records, inner_frame, inner_records,
                                     config, provenance=provenance)


def apply_postprocessor(frame, state, *, decode=True):
    validate_frame_table(frame)
    if state.get("schema_version") not in ("v9_postprocess_1", "v9_postprocess_2"):
        raise ValueError("Unsupported V9 postprocessor schema")
    identifier = state["selected_candidate"]
    if identifier not in state["candidates"]:
        raise ValueError("Selected V9 candidate missing from portable state")
    candidate = state["candidates"][identifier]
    output = _apply_candidate(frame, candidate, state["calibration"], decode=decode)
    # Stable CSV contract across folds that selected different C candidates.
    # Preserve original model scores; never silently calibrate their columns twice.
    calibrated = apply_calibration(frame, state["calibration"])
    exported = output["frame"]
    for expert in ("feature", "raw"):
        exported[expert + "_postprocess_prob"] = exported[expert + "_prob"].to_numpy(float)
        exported[expert + "_calibrated_prob"] = calibrated[expert + "_prob"].to_numpy(float)
        exported[expert + "_prob"] = frame[expert + "_prob"].to_numpy(float)
    exported["postprocess_candidate"] = identifier
    exported["calibration_applied"] = bool(candidate["calibrated"])
    exported["quality_correction"] = bool(candidate["quality_correction"])
    exported["fusion_alpha"] = float(candidate["alpha"])
    exported.attrs["expert_probability_export_contract"] = (
        "*_prob=original; *_calibrated_prob=calibrated; *_postprocess_prob=actually used; eating_prob=frozen fusion")
    return output


def evaluate_postprocessor(frame, records, state):
    output = apply_postprocessor(frame, state)
    truth, coverage = truth_in_coverage(records, frame)
    metrics = score_events(output["events"], truth, frame.subject_id.astype(str).unique())
    metrics.update(postprocess_candidate=state["selected_candidate"],
                   iou_rule="strictly greater than 0.25; one-to-one maximum-cardinality matching",
                   evaluation_scope="Full original meal intervals overlapping recorded observation",
                   fully_observed_meals_90pct=sum(row["observed_fraction"] >= .9 for row in coverage),
                   partially_observed_meals=sum(row["observed_fraction"] < .9 for row in coverage),
                   observed_frames=len(frame), unsupported_frames=int((~output["frame"].supported).sum()),
                   outer_labels_used_for_fitting=False)
    return dict(output, metrics=metrics, coverage=coverage)



def fallback_postprocessor(config=None):
    """Predeclared C0 without fitting; safe for R1/R2/R3 and auxiliary exports."""
    candidate = fallback_candidate(config)
    revised = is_precision_revision(config)
    if revised:
        candidate["revision"] = R2_REVISION
    return dict(schema_version="v9_postprocess_2" if revised else "v9_postprocess_1",
                plan_version=R2_REVISION if revised else "1.1",
                selected_candidate="C0", expert_order=["feature", "raw"],
                decoder=copy.deepcopy(candidate["decoder"]), fusion_weights=[.5, .5],
                calibration=dict(schema_version="v9_bounded_calibration_1",
                    scope="not_fitted_predeclared_fallback",
                    experts={name: dict(temperature=1., bias=0., status="identity_not_fitted")
                             for name in ("feature", "raw")}),
                candidates={"C0": candidate}, training_selection={}, training_trials={},
                inner_selection=dict(selected_candidate="C0", rows=[],
                                     scope="no_selection_predeclared_fallback"),
                verifier_status="disabled_before_R6",
                provenance=dict(fit_scope="no_fitting_predeclared_C0", outer_labels_used=False),
                receipt=dict(no_data_used_for_fitting=True))


simple_fallback_state = fallback_postprocessor



