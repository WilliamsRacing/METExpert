"""Label-independent raw-device IMU preparation and identity-balanced sequences.

The existing source cache is opened read-only. Only train/validation identities
are materialized; test numerical arrays are never opened. Labels never choose
signal values, physical masks, gravity filtering, or segment boundaries.
"""
from __future__ import annotations

from collections import OrderedDict, defaultdict
from pathlib import Path
import hashlib
import json

import os

import numpy as np
from scipy.signal import butter, sosfiltfilt
import torch
from torch.utils.data import Dataset


SCHEMA = "imu-v6-raw-multiview-1"
POINTS = 450
FS = 15
CHANNELS = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z",
            "gravity_x", "gravity_y", "gravity_z", "linear_x", "linear_y", "linear_z"]
ARRAY_NAMES = ("imu", "point_mask", "frame_quality", "segment_ids", "target",
               "target_fraction", "boundary", "label_mask")
QUALITY_NAMES = ["supported_fraction", "acc_observed_fraction", "gyro_observed_fraction",
                 "maximum_missing_fraction", "finite_fraction", "acc_near_8g_fraction",
                 "gyro_near_2293dps_fraction", "exact_repeat_fraction"]


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _runs(mask):
    edges = np.diff(np.r_[False, np.asarray(mask, bool), False].astype(np.int8))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def _physical_runs(mask, segments):
    mask = np.asarray(mask, bool); segments = np.asarray(segments)
    for a, b in _runs(mask & (segments >= 0)):
        cuts = np.r_[a, a + 1 + np.flatnonzero(np.diff(segments[a:b]) != 0), b]
        yield from zip(cuts[:-1], cuts[1:])


def multiview(raw, supported, segments):
    """Raw ACC/GYRO plus low-frequency gravity estimate and residual ACC.

    Gravity is a 0.3 Hz second-order zero-phase lowpass within physically valid
    runs. Very short runs use their median ACC rather than unstable padding.
    This is an estimated view, not an AHRS/orientation ground truth.
    """
    raw = np.asarray(raw, np.float32)
    supported = (np.asarray(supported, bool) & np.isfinite(raw).all(-1)
                 & (np.asarray(segments) >= 0))
    clean = np.where(supported[:, None], raw, 0.).astype(np.float32)
    gravity = np.zeros((len(raw), 3), np.float32)
    sos = butter(2, .3, fs=FS, output="sos")
    for a, b in _physical_runs(supported, segments):
        x = clean[a:b, :3]
        if b - a <= 12:
            gravity[a:b] = np.median(x, axis=0)
        else:
            gravity[a:b] = sosfiltfilt(sos, x, axis=0).astype(np.float32)
    values = np.concatenate((clean, gravity, clean[:, :3] - gravity), axis=1)
    return np.where(supported[:, None], values, 0.).astype(np.float32), supported


def frame_arrays(raw, ticks, observed_mask, supported, segments):
    """Build globally aligned frames; no annotation values enter this function."""
    raw = np.asarray(raw, np.float32); ticks = np.asarray(ticks, np.int64)
    observed_mask = np.asarray(observed_mask, bool)
    supported = np.asarray(supported, bool); segments = np.asarray(segments, np.int64)
    if raw.shape != (len(ticks), 6) or observed_mask.shape != (len(ticks), 2):
        raise ValueError("Raw-device cache shape mismatch")
    if len(ticks) and np.any(np.diff(ticks) != 1):
        raise ValueError("Raw-device session must retain a continuous 15 Hz grid")
    if not len(ticks):
        raise ValueError("Cannot frame an empty source session")
    features, valid = multiview(raw, supported, segments)
    origin = int(ticks[0]) // POINTS * POINTS
    count = (int(ticks[-1]) - origin) // POINTS + 1
    total = count * POINTS
    offset = int(ticks[0]) - origin
    sl = slice(offset, offset + len(ticks))
    x = np.zeros((total, 12), np.float32); x[sl] = features
    m = np.zeros(total, bool); m[sl] = valid
    seg = np.full(total, -1, np.int64); seg[sl] = np.where(valid, segments, -1)
    obs = np.zeros((total, 2), bool); obs[sl] = observed_mask
    finite = np.zeros(total, bool); finite[sl] = np.isfinite(raw).all(-1)
    sat_acc = np.zeros(total, bool); sat_acc[sl] = (np.abs(raw[:, :3]) >= 7.95).any(-1)
    sat_gyro = np.zeros(total, bool); sat_gyro[sl] = (np.abs(raw[:, 3:]) >= 2280.).any(-1)
    repeats = np.zeros(total, bool)
    if len(raw) > 1:
        repeats[offset + 1:offset + len(raw)] = (raw[1:] == raw[:-1]).all(-1)
    fm = m.reshape(count, POINTS)
    fs = seg.reshape(count, POINTS)
    frame_segments = np.full(count, -1, np.int64)
    # A frame containing separate physical segments retains only the largest
    # segment. Selection uses support only, never meal labels or hand roles.
    for i in range(count):
        ids, mass = np.unique(fs[i, fm[i]], return_counts=True)
        if len(ids):
            selected = ids[int(np.argmax(mass))]
            fm[i] &= fs[i] == selected
            frame_segments[i] = selected
    x = x.reshape(count, POINTS, 12)
    x = np.where(fm[..., None], x, 0.).transpose(0, 2, 1).copy()
    missing = np.empty(count, np.float32)
    for i in range(count):
        missing[i] = max((b-a for a,b in _runs(~fm[i])), default=0) / POINTS
    quality = np.stack((fm.mean(-1), obs.reshape(count, POINTS, 2).mean(1)[:, 0],
                        obs.reshape(count, POINTS, 2).mean(1)[:, 1], missing,
                        finite.reshape(count, POINTS).mean(-1),
                        sat_acc.reshape(count, POINTS).mean(-1),
                        sat_gyro.reshape(count, POINTS).mean(-1),
                        repeats.reshape(count, POINTS).mean(-1)), -1).astype(np.float32)
    return dict(imu=x, point_mask=fm.copy(), frame_quality=quality,
                segment_ids=frame_segments), origin, offset


def frame_labels(labels, person, ticks, origin, count, offset):
    """Meal fractions supervise occupancy, not fictitious individual bites."""
    ticks = np.asarray(ticks, np.int64)
    out = labels.targets(person, ticks, np.ones(len(ticks), bool))
    total = count * POINTS; sl = slice(offset, offset + len(ticks))
    known = np.zeros(total, bool); known[sl] = out["annotation_known"]
    y = np.zeros(total, np.float32); y[sl] = out["target_all"]
    role = np.full(total, -1, np.int8); role[sl] = out["imu_relation"]
    known = known.reshape(count, POINTS); y = y.reshape(count, POINTS)
    role = role.reshape(count, POINTS)
    fraction = y.mean(-1).astype(np.float32)
    target = np.where(fraction >= .5, 3, 0).astype(np.int64)
    for i in np.flatnonzero(fraction >= .5):
        roles = np.unique(role[i, known[i] & (y[i] > 0)])
        if len(roles) == 1 and roles[0] in (1, 2):
            target[i] = roles[0]
    label_mask = known.mean(-1) >= .999
    target[~label_mask] = -1
    boundary = np.zeros((count, 2), np.float32)
    starts = origin + np.arange(count, dtype=np.int64) * POINTS
    for e in labels.participant_events(person):
        if e.get("unknown", False):
            continue
        for j, key in enumerate(("start_epoch_ms", "end_epoch_ms")):
            # Exact rational comparison: a tick is 1000/15 ms.
            t15 = int(e[key]) * FS
            inside = (starts * 1000 <= t15) & ((starts + POINTS) * 1000 > t15)
            boundary[inside, j] = 1.
    return dict(target=target, target_fraction=fraction, boundary=boundary,
                label_mask=label_mask)





def fit_normalization(manifest, points_per_person=10000, seed=20261028):
    rng = np.random.default_rng(seed); root=Path(manifest["root"])
    by_person=defaultdict(list)
    for row in manifest["records"]:
        if row["person"] in manifest["split"]["train"] and row["supported_points"]:
            by_person[row["person"]].append(row)
    values=[]; counts={}
    for person, rows in sorted(by_person.items()):
        weights=np.asarray([r["supported_points"] for r in rows],float); weights/=weights.sum()
        budget=rng.multinomial(int(points_per_person),weights)
        person_values=[]
        for row, count in zip(rows,budget):
            if not count:
                continue
            folder=root/row["path"]
            mask=np.load(folder/"point_mask.npy",mmap_mode="r",allow_pickle=False)
            x=np.load(folder/"imu.npy",mmap_mode="r",allow_pickle=False)
            valid=np.flatnonzero(mask.reshape(-1))
            ix=rng.choice(valid,size=int(count),replace=len(valid)<count)
            person_values.append(np.asarray(x[ix//POINTS,:,ix%POINTS],np.float32))
            mask._mmap.close();x._mmap.close()
        own=np.concatenate(person_values);values.append(own);counts[person]=len(own)
    if not values:
        raise ValueError("No training-only supported normalization points")
    x=np.concatenate(values);mean=x.mean(0,dtype=np.float64);std=x.std(0,dtype=np.float64)
    std=np.maximum(std,1e-3)
    return dict(mean=mean.tolist(), std=std.tolist(), fit_scope="train_only",
                people=sorted(by_person), points_per_person=counts, total_points=len(x),seed=int(seed))


class SequenceDataset(Dataset):
    def __init__(self, manifest, split="train", frames=48, training=False,
                 samples_per_epoch=2048, seed=20261028, normalize=True, max_open=3):
        if isinstance(manifest,(str,Path)):
            path=Path(manifest);path=path/"manifest.json" if path.is_dir() else path
            manifest=_read(path);manifest["root"]=str(path.parent.resolve())
        self.manifest=manifest;self.root=Path(manifest["root"])
        if split not in ("train","val"):
            raise ValueError("IMU v6 experiment does not load the held-out test split")
        allowed=set(manifest["split"][split])
        self.records=[r for r in manifest["records"] if r["person"] in allowed]
        self.frames=int(frames);self.training=bool(training);self.samples_per_epoch=int(samples_per_epoch)
        self.seed=int(seed);self.epoch=0;self.normalize=normalize;self.max_open=max_open;self.open=OrderedDict()
        self.mean=np.asarray(manifest["normalization"]["mean"],np.float32)
        self.std=np.asarray(manifest["normalization"]["std"],np.float32)
        self.people=defaultdict(list);self.positive=defaultdict(list)
        self.validation_refs=[]
        for i,row in enumerate(self.records):
            if row["supported_points"] and (row["labeled_frames"] or not self.training):
                self.people[row["person"]].append(i)
            if row["positive_frames"]:
                self.positive[row["person"]].append(i)
            self.validation_refs.extend((i,start) for start in range(0,row["frame_count"],self.frames))
        if training and not self.people:
            raise ValueError("Training dataset has no physical IMU support")

    def arrays(self, record_index):
        index=int(record_index)
        if index not in self.open:
            self.open[index]={k:np.load(self.root/self.records[index]["path"]/(k+".npy"),
                                      mmap_mode="r",allow_pickle=False) for k in ARRAY_NAMES}
        self.open.move_to_end(index)
        while len(self.open)>self.max_open:
            _, old=self.open.popitem(last=False)
            for x in old.values():
                if getattr(x,"_mmap",None) is not None:x._mmap.close()
        return self.open[index]

    def set_epoch(self,epoch):self.epoch=int(epoch)

    def close(self):
        for arrays in self.open.values():
            for value in arrays.values():
                if getattr(value,"_mmap",None) is not None:value._mmap.close()
        self.open.clear()

    def __getstate__(self):
        state=dict(self.__dict__);state["open"]=OrderedDict();return state

    def __len__(self):
        return self.samples_per_epoch if self.training else len(self.validation_refs)

    def get_window(self,record_index,start_frame,frames=None):
        index=int(record_index);start=int(start_frame);frames=self.frames if frames is None else int(frames)
        row=self.records[index];a=self.arrays(index)
        lo=max(0,start);hi=min(row["frame_count"],start+frames)
        dst=slice(lo-start,hi-start)
        out={"imu":np.zeros((frames,12,POINTS),np.float32),
             "point_mask":np.zeros((frames,POINTS),bool),
             "frame_quality":np.zeros((frames,8),np.float32),
             "segment_ids":np.full(frames,-1,np.int64),"target":np.full(frames,-1,np.int64),
             "target_fraction":np.zeros(frames,np.float32),"boundary":np.zeros((frames,2),np.float32),
             "label_mask":np.zeros(frames,bool)}
        if hi>lo:
            for key in ARRAY_NAMES:out[key][dst]=a[key][lo:hi]
        if self.normalize:
            out["imu"]=(out["imu"]-self.mean[None,:,None])/self.std[None,:,None]
        out["imu"]=np.where(out["point_mask"][:,None,:],out["imu"],0.).astype(np.float32)
        out={k:torch.from_numpy(v.copy()) for k,v in out.items()}
        out["record_index"]=torch.tensor(index,dtype=torch.long)
        out["start_frame"]=torch.tensor(start,dtype=torch.long)
        return out

    def __getitem__(self,item):
        if isinstance(item,tuple):return self.get_window(item[0],item[1])
        if not self.training:return self.get_window(*self.validation_refs[int(item)])
        rng=np.random.default_rng(self.seed+self.epoch*1000003+int(item)*101)
        positive=bool(rng.random()<.5 and self.positive)
        pool=self.positive if positive else self.people
        person=rng.choice(sorted(pool));index=int(rng.choice(pool[person]));row=self.records[index]
        if positive:
            center=int(rng.choice(row["positive_frames"]))
        elif row["labeled_frames"]:
            center=int(rng.choice(row["labeled_frames"]))
        else:
            center=int(rng.integers(max(1,row["frame_count"])))
        # Positive anchors are distributed across the supervised central region.
        center_shift=int(rng.integers(max(1,self.frames//3),max(2,2*self.frames//3)))
        return self.get_window(index,center-center_shift)
