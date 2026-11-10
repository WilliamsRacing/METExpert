

from pathlib import Path
import json
import os

import numpy as np




from signals.optical_features.schema import V5_FEATURE_NAMES
from signals.optical_features.features_v8 import NEW_FEATURE_NAMES

SCHEMA = "fast-release-v2-cache-3"
CACHE_SCHEMAS={SCHEMA,"fast-release-v2-cache-2","fast-release-v4-cache-view-1"}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def frame_labels(labels, person, ppg):
    out = labels.frame_targets(person, ppg["frame_start_ms"], ppg["observed"].mean(-1))
    # V8 weak-background convention; partial meal frames can still supervise boundaries.
    out["label_weight"] *= np.where(out["target"] == 0, .25, 1.).astype(np.float32)
    return out


def _save_session(root, person, index, lo, hi, imu, ppg, labels, reused_arrays=None):
    relative = Path("sessions") / f"{person}-{index:05d}"
    directory = root / relative; directory.mkdir(parents=True, exist_ok=True)
    reused_arrays=dict(reused_arrays or {})
    ppg = dict(ppg)
    ppg["raw_mask"] = ppg["raw_mask"].mean(-1, keepdims=True).astype(np.float32)
    ppg["raw"] = np.zeros((*ppg["quality"].shape, 1), np.float32)
    point_observed = imu["observed"].copy()
    ticks = imu["ticks"]
    offset = ticks - int(ppg["frame_start_ms"][0]) * 15 // 1000
    valid = (offset >= 0) & (offset < ppg["observed"].size)
    point_observed[valid] |= ppg["observed"].ravel()[offset[valid]]
    targets = labels.targets(person, ticks, point_observed)
    targets["observed"] = point_observed
    imu_keys = ("ticks", "data", "supported", "segment", "observed")
    ppg_keys = ("raw", "raw_mask", "quality", "channel_ids", "features", "feature_mask", "frame_valid",
                "spectrogram", "spectrogram_mask", "routing_context", "segment", "frame_start_ms", "observed")
    for prefix, arrays, keys in (("imu", imu, imu_keys), ("ppg", ppg, ppg_keys),
                                  ("target", targets, tuple(targets))):
        for key in keys:
            if f"{prefix}__{key}" not in reused_arrays:
                np.save(directory / f"{prefix}__{key}.npy", arrays[key], allow_pickle=False)
    starts = ppg["frame_start_ms"]
    point_index = starts[:, None] * 15 // 1000 + np.arange(450) - int(ticks[0])
    good = (point_index >= 0) & (point_index < len(ticks))
    clipped = point_index.clip(0, len(ticks)-1)
    known = targets["supervision_all"][clipped] & good
    ip = (imu["supported"][clipped] & good).any(-1)
    feature_support = ppg["feature_mask"][:, [i for i in range(68) if i != 25]].any(-1)
    pp = ppg["frame_valid"] & (feature_support | (ppg["spectrogram_mask"].sum((1, 2)) > 0))
    candidate = known.any(-1) & (ip | pp)
    positive = ((targets["target_all"][clipped] > 0) & known).any(-1)
    np.save(directory / "candidates.npy", np.column_stack((starts[candidate], positive[candidate])), allow_pickle=False)
    imu_known = known & (targets["imu_relation"][clipped] >= 0) & imu["supported"][clipped] & good
    imu_candidate = imu_known.any(-1)
    imu_positive = ((targets["target_all"][clipped] > 0) & imu_known).any(-1)
    np.save(directory / "imu_candidates.npy", np.column_stack((starts[imu_candidate], imu_positive[imu_candidate])), allow_pickle=False)
    fl = frame_labels(labels, person, ppg)
    records = []
    ids = ppg["segment"]
    cuts = np.r_[0, np.flatnonzero(np.diff(ids)) + 1, len(ids)]
    for a, b in zip(cuts[:-1], cuts[1:]):
        if ids[a] < 0:
            continue
        cid = f"{person}-{index:05d}-{int(a):05d}"
        path = root / "ppg" / "records" / (cid + ".npz")
        path.parent.mkdir(parents=True, exist_ok=True)
        spectrum = root / "ppg" / "spectrograms" / (cid + ".npy")
        spectrum.parent.mkdir(parents=True, exist_ok=True)
        reused_spectrum=reused_arrays.get("ppg__spectrogram")
        if not reused_spectrum:np.save(spectrum, ppg["spectrogram"][a:b], allow_pickle=False)
        np.savez(path, raw_mask=ppg["raw_mask"][a:b], quality=ppg["quality"][a:b],
            channel_ids=ppg["channel_ids"][a:b], feature_values=ppg["features"][a:b],
            feature_mask=ppg["feature_mask"][a:b], feature_names=np.asarray(V5_FEATURE_NAMES+NEW_FEATURE_NAMES),
            frame_start_ms=starts[a:b], frame_end_ms=starts[a:b]+30000,
            spectrogram_mask=ppg["spectrogram_mask"][a:b], canonical_mask=np.ones(b-a, bool),
            fs=np.array(25.), frame_seconds=np.array(30.), schema_version=np.array("ppg-eating-cache-v8"),
            **{k: v[a:b] for k, v in fl.items()})
        meals = [dict(start_ms=e["start_epoch_ms"], end_ms=e["end_epoch_ms"], meal_id=e["event_id"])
                 for e in labels.participant_events(person)
                 if e["start_epoch_ms"] < starts[b-1]+30000 and e["end_epoch_ms"] > starts[a]]
        entry=dict(path=path.relative_to(root/"ppg").as_posix(),
            spectrogram_path=reused_spectrum or spectrum.relative_to(root/"ppg").as_posix(), subject_id=person,
            recording_id=f"{person}-{index:05d}", continuous_id=cid, n_frames=int(b-a), meals=meals)
        if reused_spectrum:entry["spectrogram_slice"]=[int(a),int(b)]
        records.append(entry)
    session=dict(person=person, start_ms=int(lo), end_ms=int(hi), path=relative.as_posix())
    if reused_arrays:session["reused_arrays"]=reused_arrays
    return session, records












