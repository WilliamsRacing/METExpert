
from __future__ import annotations
from collections import OrderedDict, defaultdict
import csv
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

POINTS = 450
ARRAYS = ("imu", "point_mask", "frame_quality", "segment_ids", "target",
          "target_fraction", "boundary", "label_mask")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def base_manifest(value):
    if isinstance(value, dict):
        return dict(value)
    path = Path(value)
    if path.is_dir():
        path = path/"manifest.json"
    result = read_json(path)
    result["root"] = str(path.parent.resolve())
    return result


def _close(values):
    for value in values.values():
        mapping = getattr(value, "_mmap", None)
        if mapping is not None:
            mapping.close()


def fit_normalization(manifest, people, points_per_person=10000, seed=20261029):
    people = set(people)
    rng = np.random.default_rng(seed)
    groups = defaultdict(list)
    for row in manifest["records"]:
        if row["person"] in people and row["supported_points"]:
            groups[row["person"]].append(row)
    means, second, counts = [], [], {}
    root = Path(manifest["root"])
    for person, rows in sorted(groups.items()):
        weights = np.asarray([r["supported_points"] for r in rows], float)
        budget = rng.multinomial(int(points_per_person), weights/weights.sum())
        own = []
        for row, count in zip(rows, budget):
            if not count:
                continue
            folder = root/row["path"]
            arrays = {k: np.load(folder/(k+".npy"), mmap_mode="r", allow_pickle=False)
                      for k in ("imu", "point_mask")}
            try:
                available = np.flatnonzero(arrays["point_mask"].reshape(-1))
                ix = rng.choice(available, size=int(count), replace=len(available)<count)
                own.append(np.asarray(arrays["imu"][ix//POINTS, :, ix%POINTS], np.float64).copy())
            finally:
                _close(arrays)
        values = np.concatenate(own)
        means.append(values.mean(0))
        second.append(np.square(values).mean(0))
        counts[person] = len(values)
    if not means:
        raise ValueError("No supported fit-person points for normalization")
    mean = np.mean(means, 0)
    std = np.sqrt(np.maximum(np.mean(second, 0)-mean**2, 1e-6))
    return dict(mean=mean.tolist(), std=std.tolist(), fit_scope="train_people_only",
                people=sorted(groups), points_per_person=counts, seed=int(seed))


def prepare_fold_manifest(base, train_people, selection_people=(), points_per_person=10000,
                          seed=20261029, forbidden_people=()):
    manifest = base_manifest(base)
    train, selection = list(train_people), list(selection_people)
    if not train or len(train) != len(set(train)) or len(selection) != len(set(selection)):
        raise ValueError("Nonempty unique train people and unique selection people required")
    if set(train) & set(selection) or (set(train)|set(selection)) & set(forbidden_people):
        raise ValueError("Fold identity overlap or forbidden identity")
    # The old outer validation people may be exported later, but cannot enter a
    # new IMU fit/inner selection through this training API.
    allowed = set(manifest["split"]["train"])
    if (set(train)|set(selection))-allowed:
        raise ValueError("IMU V8 fitting/selection is restricted to original training identities")
    result = dict(manifest)
    result["split"] = {"train": train, "selection": selection}
    result["normalization"] = fit_normalization(
        manifest, train, points_per_person=points_per_person, seed=seed)
    result["v8_fold_contract"] = dict(train_people=train, selection_people=selection,
                                     normalization_fit_people=result["normalization"]["people"],
                                     external_validation_used_for_fit=False)
    return result


def _events(annotation_root, people):
    rows = defaultdict(list)
    path = Path(annotation_root)/"events.csv"
    if not path.is_file():
        raise FileNotFoundError("Event-owned sampling requires original annotations/events.csv")
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            person = row["externalid"].strip().upper()
            if person in people:
                a, b = int(row["beforeTime"]), int(row["afterTime"])
                if a >= b:
                    raise ValueError("Invalid meal annotation")
                eid = row.get("event_id") or f"{person}:{a}:{b}"
                rows[person].append((a, b, eid))
    return rows


class SequenceDataset(Dataset):
    """Person -> meal/explicit-background/natural core sampling.

    Arrays and labels remain in the immutable V6 cache. Natural sampling is
    duration-weighted within each uniformly sampled person. Explicit background
    cores are wholly known, physically supported and guarded from true meals.
    """
    def __init__(self, manifest, people=None, frames=48, core_frames=16,
                 training=False, samples_per_epoch=2048, seed=20261029,
                 sampling=None, normalization=None, annotation_root=None, max_open=4):
        self.manifest = base_manifest(manifest)
        self.root = Path(self.manifest["root"])
        self.people_allowed = set(people if people is not None else self.manifest["split"]["train"])
        self.records = [r for r in self.manifest["records"] if r["person"] in self.people_allowed]
        self.frames, self.core_frames = int(frames), int(core_frames)
        if self.frames < self.core_frames or (self.frames-self.core_frames) % 2:
            raise ValueError("Sequence frames must contain a centered integral core")
        self.training, self.samples_per_epoch = bool(training), int(samples_per_epoch)
        self.seed, self.epoch, self.max_open = int(seed), 0, int(max_open)
        self.normalization = normalization or self.manifest["normalization"]
        self.mean = np.asarray(self.normalization["mean"], np.float32)
        self.std = np.maximum(np.asarray(self.normalization["std"], np.float32), 1e-3)
        self.open = OrderedDict()
        self.sampling = dict(natural=.5, meal=.25, background=.25, background_guard_seconds=120)
        self.sampling.update(sampling or {})
        self.fractions = np.array([self.sampling[k] for k in ("natural", "meal", "background")], float)
        if np.any(self.fractions < 0) or not np.isclose(self.fractions.sum(), 1):
            raise ValueError("Sampling fractions must be nonnegative and sum to one")
        self.natural, self.meals, self.background = defaultdict(list), defaultdict(dict), defaultdict(list)
        if not training:
            return
        events = _events(annotation_root or self.manifest["annotation_root"], self.people_allowed)
        guard = float(self.sampling["background_guard_seconds"])*1000
        for ri, row in enumerate(self.records):
            person = row["person"]
            if row["supported_points"] and row["labeled_frames"]:
                self.natural[person].append(ri)
            for center in row["positive_frames"]:
                middle = (row["start_tick"]+int(center)*POINTS+POINTS//2)*1000/15
                for a, b, eid in events[person]:
                    if a <= middle < b:
                        self.meals[person].setdefault(eid, []).append((ri, int(center)))
            a = self.arrays(ri)
            known = np.asarray(a["label_mask"], bool)
            full = np.asarray(a["point_mask"], bool).mean(-1) >= .5
            empty = np.asarray(a["target_fraction"]) == 0
            acceptable = known & full & empty
            half = self.core_frames//2
            # One potential background event per non-overlapping core bucket.
            for center in range(half, row["frame_count"]-(self.core_frames-half)+1, self.core_frames):
                left, right = center-half, center+(self.core_frames-half)
                if not acceptable[left:right].all():
                    continue
                start_ms = (row["start_tick"]+left*POINTS)*1000/15
                end_ms = (row["start_tick"]+right*POINTS)*1000/15
                if any(a0-guard < end_ms and b0+guard > start_ms for a0, b0, _ in events[person]):
                    continue
                self.background[person].append((ri, center))
        self.meals = {p: v for p, v in self.meals.items() if v}
        self.background = {p: v for p, v in self.background.items() if v}
        if not self.natural:
            raise ValueError("No known supported training sequence anchors")
        self.close()

    def sampling_summary(self):
        return dict(
            fractions=self.sampling, fit_people=sorted(self.people_allowed),
            natural_records=sum(map(len, self.natural.values())),
            meal_events=sum(map(len, self.meals.values())),
            background_cores=sum(map(len, self.background.values())))

    def arrays(self, ri):
        ri = int(ri)
        if ri not in self.open:
            folder = self.root/self.records[ri]["path"]
            self.open[ri] = {k: np.load(folder/(k+".npy"), mmap_mode="r", allow_pickle=False)
                             for k in ARRAYS}
        self.open.move_to_end(ri)
        while len(self.open) > self.max_open:
            _, old = self.open.popitem(last=False)
            _close(old)
        return self.open[ri]

    def close(self):
        for value in self.open.values():
            _close(value)
        self.open.clear()

    def __getstate__(self):
        out = dict(self.__dict__)
        out["open"] = OrderedDict()
        return out

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return self.samples_per_epoch if self.training else len(self.records)

    def get_window(self, ri, start_frame, frames=None):
        frames = self.frames if frames is None else int(frames)
        ri, start_frame = int(ri), int(start_frame)
        row, arrays = self.records[ri], self.arrays(ri)
        left, right = max(0, start_frame), min(row["frame_count"], start_frame+frames)
        out = dict(imu=np.zeros((frames, 12, POINTS), np.float32),
                   point_mask=np.zeros((frames, POINTS), bool),
                   frame_quality=np.zeros((frames, 8), np.float32),
                   segment_ids=np.full(frames, -1, np.int64), target=np.full(frames, -1, np.int64),
                   target_fraction=np.zeros(frames, np.float32), boundary=np.zeros((frames, 2), np.float32),
                   label_mask=np.zeros(frames, bool))
        if right > left:
            destination = slice(left-start_frame, right-start_frame)
            for key in ARRAYS:
                out[key][destination] = arrays[key][left:right]
        x = (out["imu"]-self.mean[None, :, None])/self.std[None, :, None]
        out["imu"] = np.where(out["point_mask"][:, None], x, 0).astype(np.float32)
        result = {k: torch.from_numpy(v.copy()) for k, v in out.items()}
        result["record_index"] = torch.tensor(ri)
        result["start_frame"] = torch.tensor(start_frame)
        return result

    def __getitem__(self, item):
        if not self.training:
            raise TypeError("Export uses explicit get_window requests")
        rng = np.random.default_rng(self.seed+self.epoch*1000003+int(item)*101)
        kind = ("natural", "meal", "background")[rng.choice(3, p=self.fractions)]
        if kind == "meal" and self.meals:
            person = rng.choice(sorted(self.meals))
            eid = rng.choice(sorted(self.meals[person]))
            refs = self.meals[person][eid]
            ri, center = refs[int(rng.integers(len(refs)))]
            core_start = (self.frames-self.core_frames)//2
            position = int(rng.integers(core_start, core_start+self.core_frames))
        elif kind == "background" and self.background:
            person = rng.choice(sorted(self.background))
            refs = self.background[person]
            ri, center = refs[int(rng.integers(len(refs)))]
            position = (self.frames-self.core_frames)//2+self.core_frames//2
        else:
            person = rng.choice(sorted(self.natural))
            refs = self.natural[person]
            weight = np.asarray([len(self.records[i]["labeled_frames"]) for i in refs], float)
            ri = int(rng.choice(refs, p=weight/weight.sum()))
            center = int(rng.choice(self.records[ri]["labeled_frames"]))
            position = (self.frames-self.core_frames)//2+self.core_frames//2
        result = self.get_window(ri, center-position)
        result["sampling_stratum"] = torch.tensor({"natural": 0, "meal": 1, "background": 2}[kind])
        return result

