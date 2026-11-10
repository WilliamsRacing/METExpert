
from __future__ import annotations

import numpy as np

VERSION = "v8-grouped-pca-v1"
GROUP_NAMES = ("hr_time_prv", "frequency_prv", "poincare_complexity",
               "morphology_optical", "ppg_spectrum")
DIRECT_NAMES = ("hr_bpm_60", "hr_relative_pooled", "hr_short_long_diff",
                "rmssd_ms_300", "hf_lf_300", "r_red_ir_60__relative_baseline")
RELIABILITY_NAMES = ("hr_history_coverage",)


def feature_group(name):
    """Every physical column has exactly one declared family; unknowns fail."""
    base = str(name).split("__relative_baseline")[0]
    if name in RELIABILITY_NAMES:
        return None
    if base.startswith("ppg_"):
        return "ppg_spectrum"
    if base.startswith(("prv_", "lf_", "hf_")):
        return "frequency_prv"
    if base.startswith(("sd1", "sd2", "poincare_", "guzik_", "porta_", "ppi_permutation_")):
        return "poincare_complexity"
    if base.startswith(("hr_", "rmssd", "sdnn", "pnn", "cvnn", "cvsd", "ppi_", "ln_rmssd")):
        return "hr_time_prv"
    if base.startswith(("ac_dc", "rise_", "width_", "prepeak_", "r_red_ir",
                        "amplitude_", "template_")):
        return "morphology_optical"
    raise ValueError("Unassigned V8 feature family: " + str(name))


def _weighted_quantile(values, weights, fraction):
    order = np.argsort(values, kind="stable")
    v, w = values[order], weights[order]
    cumulative = np.cumsum(w)
    return float(v[min(np.searchsorted(cumulative, fraction * cumulative[-1]), len(v) - 1)])


class FeaturePCA:
    """Signed-log/robust scaling followed by unwhitened per-family PCA.

    names describes the raw cache schema. output_names/output_dim describe the
    projected values. transform returns only those values; transform_details
    preserves the original missingness, family support, and physical values.
    """
    def __init__(self, payload):
        self.payload = payload
        self.names = list(payload["feature_names"])
        self.output_names = list(payload["output_names"])
        self.output_dim = len(self.output_names)
        self.median = np.asarray(payload["median"], dtype=np.float64)
        self.scale = np.asarray(payload["scale"], dtype=np.float64)
        self.groups = list(payload["groups"])
        self.direct = list(payload["direct"])
        self.fit_subjects = list(payload["fit_subjects"])
        self._validate()

    @staticmethod
    def transform_base(x):
        return np.sign(x) * np.log1p(np.abs(x))

    @classmethod
    def fit(cls, x, mask, feature_names, subject_ids, *, training_subjects,
            variance_threshold=.95, sample_stride=2, max_rows_per_subject=4000):
        x, mask = np.asarray(x, dtype=np.float64), np.asarray(mask, dtype=bool)
        names = [str(n) for n in feature_names]
        subjects = np.asarray(subject_ids).astype(str)
        declared = sorted({str(s) for s in training_subjects})
        if x.ndim != 2 or mask.shape != x.shape or x.shape[1] != len(names):
            raise ValueError("PCA needs aligned N x D physical values, masks and names")
        if subjects.shape != (len(x),) or not len(x) or len(set(names)) != len(names):
            raise ValueError("PCA needs nonempty rows, unique names and row identity IDs")
        if not declared or sorted(set(subjects.tolist())) != declared:
            raise ValueError("PCA observed identities must equal the declared training identities")
        if not 0 < variance_threshold <= 1 or sample_stride < 1 or max_rows_per_subject < 2:
            raise ValueError("Invalid PCA fitting settings")
        unknown_direct = [n for n in DIRECT_NAMES if n not in names]
        if unknown_direct:
            raise ValueError("Missing V8 direct features: " + repr(unknown_direct))
        family = [feature_group(n) for n in names]
        selected, sample_audit = [], []
        for subject in declared:
            all_indices = np.flatnonzero(subjects == subject)
            pick = all_indices[::int(sample_stride)]
            if len(pick) > max_rows_per_subject:
                pick = pick[np.linspace(0, len(pick) - 1, int(max_rows_per_subject), dtype=int)]
            selected.append(pick)
            sample_audit.append({"subject_id": subject, "available_rows": int(len(all_indices)),
                                 "sampled_rows": int(len(pick)), "total_weight": 1.})
        indices = np.concatenate(selected)
        x, mask, subjects = x[indices], mask[indices], subjects[indices]
        observed = mask & np.isfinite(x)
        transformed = cls.transform_base(np.where(observed, x, 0.))
        weights = np.zeros(len(x), dtype=np.float64)
        for subject in declared:
            found = subjects == subject
            weights[found] = 1. / int(found.sum())
        weights /= weights.sum()
        median, scale, coverage, dropped, constant = [], [], [], {}, []
        for j, name in enumerate(names):
            good = observed[:, j]
            coverage.append(float(weights[good].sum()))
            if not good.any():
                median.append(0.); scale.append(1.); constant.append(True)
                dropped[name] = "all_missing_in_training"
                continue
            v, w = transformed[good, j], weights[good]
            median.append(_weighted_quantile(v, w, .5))
            scale.append(max(_weighted_quantile(v, w, .75) - _weighted_quantile(v, w, .25), .05))
            is_constant = len(v) < 2 or float(np.ptp(v)) <= 1e-8
            constant.append(is_constant)
            if is_constant and name not in DIRECT_NAMES and name not in RELIABILITY_NAMES:
                dropped[name] = "near_constant_in_training"
        median, scale = np.asarray(median), np.asarray(scale)
        z = np.where(observed, np.clip((transformed - median) / scale, -10., 10.), 0.)
        groups, output_names, position = [], [], 0
        for group in GROUP_NAMES:
            all_cols = [j for j, value in enumerate(family)
                        if value == group and names[j] not in DIRECT_NAMES]
            cols = [j for j in all_cols if not constant[j]]
            item = {"name": group, "all_input_indices": all_cols, "input_indices": cols,
                    "input_names": [names[j] for j in cols], "output_start": position}
            if cols:
                matrix = z[:, cols]
                center = np.sum(matrix * weights[:, None], axis=0)
                centered = matrix - center
                covariance = centered.T @ (centered * weights[:, None])
                values, vectors = np.linalg.eigh(covariance)
                order = np.argsort(values)[::-1]
                values, vectors = np.maximum(values[order], 0.), vectors[:, order].T
                total = float(values.sum())
                if total <= 1e-12:
                    count = 0
                else:
                    count = min(len(values), int(np.searchsorted(np.cumsum(values) / total,
                                                                 variance_threshold)) + 1)
                components = vectors[:count]
                for component in components:
                    if component[np.argmax(np.abs(component))] < 0:
                        component *= -1
                ratios = values / total if total else np.zeros_like(values)
                item.update(center=center.tolist(), components=components.tolist(),
                            explained_variance=values[:count].tolist(),
                            explained_variance_ratio=ratios[:count].tolist(),
                            retained_variance=float(ratios[:count].sum()),
                            full_variance=float(total), n_components=int(count))
            else:
                item.update(center=[], components=[], explained_variance=[],
                            explained_variance_ratio=[], retained_variance=0.,
                            full_variance=0., n_components=0)
            count = item["n_components"]
            output_names.extend(group + "__pc_" + str(i + 1) for i in range(count))
            position += count
            item["output_stop"] = position
            groups.append(item)
        direct = []
        for name in DIRECT_NAMES:
            j = names.index(name)
            direct.append({"name": name, "input_index": j, "output_index": position,
                           "observed_in_training": coverage[j] > 0})
            output_names.append(name + "__direct")
            position += 1
        payload = {
            "version": VERSION, "feature_names": names, "output_names": output_names,
            "transform": "signed_log1p_then_identity_balanced_median_iqr_then_group_pca",
            "median": median.tolist(), "scale": scale.tolist(), "clip": 10.,
            "whiten": False, "variance_threshold": float(variance_threshold),
            "group_order": list(GROUP_NAMES), "groups": groups, "direct": direct,
            "reliability_fields_excluded_from_pca": list(RELIABILITY_NAMES),
            "weighted_training_coverage": coverage, "dropped_columns": dropped,
            "fit_subjects": declared, "scope": "training_identities_only_no_labels",
            "sampling": {"stride": int(sample_stride),
                         "max_rows_per_subject": int(max_rows_per_subject),
                         "policy": "deterministic_temporal_thinning_equal_total_identity_weight",
                         "identities": sample_audit},
            "n_fit_rows": int(len(indices)), "input_dim": len(names),
            "output_dim": len(output_names),
        }
        return cls(payload)

    def _validate(self):
        p = self.payload
        if p.get("version") != VERSION or p.get("whiten") is not False:
            raise ValueError("Unsupported PCA checkpoint contract")
        if len(set(self.names)) != len(self.names) or self.median.shape != (len(self.names),):
            raise ValueError("Malformed PCA feature schema")
        if self.scale.shape != self.median.shape or not np.isfinite(self.median).all() or not np.isfinite(self.scale).all() or (self.scale <= 0).any():
            raise ValueError("Malformed PCA scaler")
        if [g["name"] for g in self.groups] != list(GROUP_NAMES):
            raise ValueError("PCA groups changed")
        position = 0
        for group in self.groups:
            cols, count = group["input_indices"], int(group["n_components"])
            if any(not 0 <= int(j) < len(self.names) for j in cols):
                raise ValueError("PCA input index outside schema")
            if group["output_start"] != position or group["output_stop"] != position + count:
                raise ValueError("PCA output slices are not contiguous")
            center = np.asarray(group["center"], dtype=float)
            components = np.asarray(group["components"], dtype=float).reshape(count, len(cols))
            if center.shape != (len(cols),) or not np.isfinite(center).all() or not np.isfinite(components).all():
                raise ValueError("Malformed PCA center/components")
            position += count
        if [d["name"] for d in self.direct] != list(DIRECT_NAMES):
            raise ValueError("PCA direct fields changed")
        if len(self.output_names) != position + len(DIRECT_NAMES):
            raise ValueError("PCA dimension mismatch")

    def transform_details(self, x, mask):
        x, mask = np.asarray(x, dtype=np.float64), np.asarray(mask, dtype=bool)
        if x.ndim < 2 or mask.shape != x.shape or x.shape[-1] != len(self.names):
            raise ValueError("PCA input must end in its exact physical feature dimension")
        shape, flat = x.shape[:-1], x.reshape(-1, len(self.names))
        observed = mask.reshape(flat.shape) & np.isfinite(flat)
        logged = self.transform_base(np.where(observed, flat, 0.))
        physical = np.where(observed, np.clip((logged - self.median) / self.scale, -10., 10.), 0.)
        values = np.zeros((len(flat), self.output_dim), np.float64)
        output_mask = np.zeros_like(values, dtype=bool)
        family_coverage = np.zeros((len(flat), len(GROUP_NAMES)), np.float64)
        for k, group in enumerate(self.groups):
            cols, count = group["input_indices"], group["n_components"]
            if not cols or not count:
                continue
            coverage = observed[:, cols].mean(-1)
            valid = coverage > 0
            center = np.asarray(group["center"])
            components = np.asarray(group["components"])
            projected = (physical[:, cols] - center) @ components.T
            begin, end = group["output_start"], group["output_stop"]
            values[:, begin:end] = np.where(valid[:, None], projected, 0.)
            output_mask[:, begin:end] = valid[:, None]
            family_coverage[:, k] = coverage
        direct_masks = []
        for direct in self.direct:
            j, k = direct["input_index"], direct["output_index"]
            valid = observed[:, j] & bool(direct["observed_in_training"])
            values[:, k] = np.where(valid, physical[:, j], 0.)
            output_mask[:, k] = valid
            direct_masks.append(valid)
        support_columns = sorted({j for group in self.groups if group["n_components"]
                                  for j in group["input_indices"]}
                                 | {d["input_index"] for d in self.direct if d["observed_in_training"]})
        supported = output_mask.any(-1)
        reliability = (observed[:, support_columns].mean(-1) if support_columns
                       else np.zeros(len(flat), dtype=float))
        def restore(a):
            return a.reshape(shape + a.shape[1:]).astype(np.float32)
        return {"features": restore(values), "feature_mask": restore(output_mask),
                "raw_mask": restore(observed), "family_coverage": restore(family_coverage),
                "supported": supported.reshape(shape),
                "reliability": reliability.reshape(shape).astype(np.float32),
                "scaled_physical": restore(physical)}

    def transform(self, x, mask):
        return self.transform_details(x, mask)["features"]

    def to_dict(self):
        import copy
        return copy.deepcopy(self.payload)

    @classmethod
    def from_dict(cls, payload):
        import copy
        return cls(copy.deepcopy(payload))


GroupedFeaturePCA = FeaturePCA
