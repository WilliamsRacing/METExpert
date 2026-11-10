"""Subject splits, training-only normalization and bounded-memory sequence reads."""
from __future__ import annotations

import json
import multiprocessing as mp
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler


def load_records(cache_dir, feature_sidecar=None):
    cache_dir = Path(cache_dir)
    path = cache_dir / "records.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path}: run prepare_data.py first")
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not records:
        raise ValueError("No cached continuous segments")
    for rec in records:
        key = next((k for k in ("path", "cache_path", "npz_path") if k in rec), None)
        if key is None:
            raise ValueError("Cache record lacks path")
        p = Path(rec[key])
        rec["_path"] = str(p if p.is_absolute() else cache_dir / p)
        if not Path(rec["_path"]).is_file():
            raise FileNotFoundError(rec["_path"])
        rec['subject_id'] = str(rec['subject_id'])
        if 'spectrogram_path' in rec:
            sp=Path(rec['spectrogram_path']);rec['_spectrogram_path']=str(sp if sp.is_absolute() else cache_dir/sp)
    if feature_sidecar is not None:
        from .hr_reference import attach_sidecar
        records = attach_sidecar(records, cache_dir, feature_sidecar)
    return records


def feature_sidecar_fingerprint(records):
    """Stable whole-sidecar identity even when records are a fold subset."""
    fingerprints = {r.get("_feature_sidecar_manifest_sha256") for r in records}
    if not fingerprints or fingerprints == {None}:
        return None
    if None in fingerprints or len(fingerprints) != 1:
        raise ValueError("Mixed or inconsistent HR feature sidecars")
    return next(iter(fingerprints))


def _features_with_sidecar(rec, values, mask, names, starts, ends):
    if "_feature_sidecar" not in rec:
        return values, mask, [str(n) for n in names]
    from .hr_reference import append_sidecar
    return append_sidecar(rec, values, mask, names, starts, ends)


def make_splits(records, seed=20261011, fold=None, n_folds=5):
    subjects = sorted({r["subject_id"] for r in records})
    if len(subjects) < 5:
        raise ValueError("At least 5 distinct identity groups are required; use synthetic smoke mode for code checks")
    # Split by linked identity; neither recording nor window random split is allowed.
    rng = np.random.default_rng(seed)
    rng.shuffle(subjects)
    if fold is None:
        nv = max(1, round(len(subjects) * .15)); nt = max(1, round(len(subjects) * .15))
        test, val, train = subjects[:nt], subjects[nt:nt + nv], subjects[nt + nv:]
    else:
        if not 2 <= n_folds <= len(subjects) or not 0 <= fold < n_folds:
            raise ValueError("Invalid outer cross-validation fold")
        chunks = [[str(s) for s in a] for a in np.array_split(subjects, n_folds)]
        test = chunks[fold]
        remainder = [s for j, a in enumerate(chunks) if j != fold for s in a]
        nv = max(1, round(len(remainder) * .2))
        val, train = remainder[:nv], remainder[nv:]
    result = {"seed": seed, "fold": fold, "n_folds": n_folds,
              "train": list(train), "val": list(val), "test": list(test)}
    validate_splits(result, records)
    return result


def validate_splits(splits, records):
    sets = [set(splits[k]) for k in ("train", "val", "test")]
    if not sets[0] or not sets[1] or any(sets[i] & sets[j] for i in range(3) for j in range(i)):
        raise ValueError("Empty or overlapping subject splits")
    available = {r["subject_id"] for r in records}
    if set.union(*sets) != available:
        raise ValueError("Split subjects do not exactly match cache subjects")


def canonical_frame_masks(records):
    """Own each (subject_id, frame_start_ms) once, using only PPG quality.

    Returns boolean arrays aligned with ``records`` and their local frames.
    Only timestamp and quality NPZ members are read; waveforms, features and
    labels cannot influence ownership. Highest ``quality.max`` wins. Equal
    qualities use recording/segment/path/index order for deterministic ties.
    Compute once and pass ``frame_masks=...`` to all training consumers to
    avoid repeated metadata I/O. Context frames remain readable by the dataset.
    """
    # V5 freezes original V3/V4 ownership even when stable routing changes SQI.
    stored = []
    for rec in records:
        with np.load(rec["_path"], allow_pickle=False) as z:
            if "canonical_mask" not in z:
                stored = []; break
            owned = np.asarray(z["canonical_mask"], dtype=bool)
            if owned.shape != (int(rec["n_frames"]),):
                raise ValueError("V5 canonical ownership shape mismatch")
            stored.append(owned)
    if stored:
        return stored
    masks, owners = [], {}
    for ri, rec in enumerate(records):
        with np.load(rec["_path"], allow_pickle=False) as z:
            timestamps = np.asarray(z["frame_start_ms"])
            quality = np.asarray(z["quality"], dtype=float)
        if timestamps.ndim != 1 or quality.ndim < 1 or quality.shape[0] != len(timestamps):
            raise ValueError("Frame timestamps and PPG quality have incompatible shapes")
        if len(timestamps) != int(rec["n_frames"]):
            raise ValueError("Record n_frames differs from cached timestamp count")
        if not np.all(np.isfinite(timestamps)):
            raise ValueError("Frame timestamps must be finite")
        quality = np.where(np.isfinite(quality), quality, -np.inf)
        score = quality.reshape(len(timestamps), -1).max(axis=1) if len(timestamps) else np.empty(0)
        masks.append(np.zeros(len(timestamps), dtype=bool))
        priority = (str(rec.get("recording_id", "")), str(rec.get("continuous_id", "")),
                    str(rec["_path"]), ri)
        subject = str(rec["subject_id"])
        for fi, timestamp in enumerate(timestamps):
            key = (subject, float(timestamp))
            previous = owners.get(key)
            if previous is None or score[fi] > previous[0] or (
                score[fi] == previous[0] and (priority, fi) < (previous[1], previous[3])
            ):
                if previous is not None:
                    masks[previous[2]][previous[3]] = False
                masks[ri][fi] = True
                owners[key] = (float(score[fi]), priority, ri, fi)
    return masks


def _validated_frame_masks(records, frame_masks=None):
    if frame_masks is None:
        return canonical_frame_masks(records)
    if len(frame_masks) != len(records):
        raise ValueError("frame_masks must have one array per record")
    result = [np.asarray(mask, dtype=bool) for mask in frame_masks]
    for rec, mask in zip(records, result):
        if mask.shape != (int(rec["n_frames"]),):
            raise ValueError("Each frame ownership mask must match that record's n_frames")
    return result


class FeatureScaler:
    """Signed-log, then median/IQR fitted only to training groups' observed values."""
    def __init__(self, names, median, scale):
        self.names = [str(name) for name in names]
        self.median = np.asarray(median, np.float32)
        self.scale = np.asarray(scale, np.float32)

    @staticmethod
    def transform_base(x):
        return np.sign(x) * np.log1p(np.abs(x))

    @classmethod
    def fit(cls, records, max_rows=200000, seed=20261011, frame_masks=None):
        if not records or max_rows < 1:
            raise ValueError("Scaler fitting needs records and a positive max_rows")
        ownership = _validated_frame_masks(records, frame_masks)
        rows, masks, names = [], [], None
        for ri, rec in enumerate(records):
            with np.load(rec["_path"], allow_pickle=False) as z:
                values, mask, current = _features_with_sidecar(
                    rec, z["feature_values"], z["feature_mask"], z["feature_names"],
                    z["frame_start_ms"], z["frame_end_ms"])
                if names is None:
                    names = current
                if current != names:
                    raise ValueError("Feature schema changed across cache segments")
                x = values[ownership[ri]]
                m = mask.astype(bool)[ownership[ri]]
                # Bound deterministic per-record samples before concatenation.
                step = max(1, len(x) // max(1, max_rows // len(records)))
                rows.append(x[::step]); masks.append(m[::step])
        x, mask = np.concatenate(rows), np.concatenate(masks)
        if len(x) > max_rows:
            pick = np.random.default_rng(seed).choice(len(x), max_rows, replace=False)
            x, mask = x[pick], mask[pick]
        x = cls.transform_base(x)
        median, scale = [], []
        for j in range(x.shape[1]):
            v = x[:, j][mask[:, j] & np.isfinite(x[:, j])]
            median.append(float(np.median(v)) if len(v) else 0.)
            scale.append(float(max(np.quantile(v, .75) - np.quantile(v, .25), .05)) if len(v) else 1.)
        return cls(names, median, scale)

    def transform(self, x, mask):
        z = (self.transform_base(np.nan_to_num(x)) - self.median) / self.scale
        return np.clip(z, -10, 10).astype(np.float32) * mask

    def to_dict(self):
        return {"feature_names": self.names, "median": self.median.tolist(), "scale": self.scale.tolist(),
                "transform": "signed_log1p_then_training_median_iqr"}

    @classmethod
    def from_dict(cls, d):
        return cls(d["feature_names"], d["median"], d["scale"])


def fit_v8_pca(records, ownership, variance=.95, availability_mode="loading_fraction"):
    from .feature_pca import FeaturePCA
    rows=[]; masks=[]; subjects=[]; names=None
    for rec, owned in zip(records,ownership):
        with np.load(rec['_path'],allow_pickle=False) as z:
            current=list(z['feature_names'].astype(str))
            if names is not None and current!=names:raise ValueError('V8 input feature order changed')
            names=current
            rows.append(z['feature_values'][owned]);masks.append(z['feature_mask'][owned])
            subjects.extend([rec['subject_id']]*int(owned.sum()))
    return FeaturePCA.fit(np.concatenate(rows),np.concatenate(masks),names,np.asarray(subjects),
        training_subjects=sorted({r['subject_id'] for r in records}), variance_threshold=float(variance), availability_mode=availability_mode)


class SequenceDataset(Dataset):
    """V8 bounded record LRU; spectra are mmap and no raw/embedding is loaded."""
    def __init__(self,records,scaler,sequence_frames=192,core_frames=128,causal=False,
                 augment=False,lru_records=4,frame_masks=None):
        if not isinstance(lru_records,int) or isinstance(lru_records,bool) or lru_records<1:
            raise ValueError('lru_records must be a positive integer')
        if core_frames<1 or sequence_frames<core_frames or (not causal and (sequence_frames-core_frames)%2):
            raise ValueError('Invalid context/core lengths')
        self.records,self.scaler=records,scaler
        self.sequence_frames,self.core_frames=sequence_frames,core_frames
        self.left=sequence_frames-core_frames if causal else (sequence_frames-core_frames)//2
        self.augment,self.lru_records=augment,lru_records
        # Spawn-context synchronization is valid for Windows spawn and Linux
        # fork/spawn workers. Epoch changes remain visible to persistent workers.
        ctx=mp.get_context('spawn')
        lock=ctx.RLock()
        self._augmentation_epoch=ctx.Value('q',0,lock=lock)
        self._augmentation_seed=ctx.Value('q',20261011,lock=lock)
        self.frame_masks=_validated_frame_masks(records,frame_masks);self.cache=OrderedDict()
        self.items=[(ri,start) for ri,r in enumerate(records) for start in range(0,int(r['n_frames']),core_frames)
                    if self.frame_masks[ri][start:start+core_frames].any()]
    def __len__(self):return len(self.items)
    def set_epoch(self, epoch, seed=20261011):
        epoch,seed=int(epoch),int(seed)
        if epoch<0 or seed<0:
            raise ValueError('Dataset augmentation epoch/seed must be nonnegative')
        with self._augmentation_epoch.get_lock():
            self._augmentation_epoch.value=epoch
            self._augmentation_seed.value=seed
    def augmentation_state(self):
        with self._augmentation_epoch.get_lock():
            return {'seed':int(self._augmentation_seed.value),'epoch':int(self._augmentation_epoch.value),
                    'policy':'local_numpy_generator_seed_epoch_item'}
    def __getstate__(self):
        state=self.__dict__.copy()
        # Never pickle a live numpy mmap or duplicate its loaded record buffers.
        state['cache']=OrderedDict()
        return state
    @staticmethod
    def _close_record(record):
        seen=set()
        for value in record.values():
            mapping=getattr(value,'_mmap',None)
            if mapping is not None and id(mapping) not in seen:
                mapping.close();seen.add(id(mapping))
    def close(self):
        """Release every mmap handle before Windows cleanup or worker exit."""
        for record in self.cache.values():self._close_record(record)
        self.cache.clear()
    def __del__(self):
        if hasattr(self,'cache'):self.close()
    def _load(self,ri):
        if ri in self.cache:
            self.cache.move_to_end(ri);return self.cache[ri]
        rec=self.records[ri]
        keys=('raw_mask','quality','channel_ids','feature_values','feature_mask','target','label_weight',
              'start_target','end_target','boundary_mask','frame_start_ms','frame_end_ms','spectrogram_mask')
        if rec.get('_numeric_dir'):
            directory=Path(rec['_numeric_dir'])
            names=np.load(directory/'feature_names.npy',mmap_mode='r',allow_pickle=False)
            if list(names.astype(str))!=self.scaler.names:
                if getattr(names,'_mmap',None) is not None:names._mmap.close()
                raise ValueError('Checkpoint feature schema differs from numeric sidecache')
            if getattr(names,'_mmap',None) is not None:names._mmap.close()
            d={key:np.load(directory/(key+'.npy'),mmap_mode='r',allow_pickle=False) for key in keys}
            if (directory/'routing_context.npy').is_file():
                d['routing_context']=np.load(directory/'routing_context.npy',mmap_mode='r',allow_pickle=False)
        else:
            with np.load(rec['_path'],allow_pickle=False) as z:
                if list(z['feature_names'].astype(str))!=self.scaler.names:raise ValueError('Checkpoint feature schema differs')
                d={key:z[key] for key in keys}
                if 'routing_context' in z:d['routing_context']=z['routing_context']
        from .cache_io import load_spectrogram
        d['spectrogram']=load_spectrogram(rec)
        self.cache[ri]=d
        while len(self.cache)>self.lru_records:
            _, evicted=self.cache.popitem(last=False)
            self._close_record(evicted)
        return d
    def __getitem__(self,index):
        with self._augmentation_epoch.get_lock():
            epoch=int(self._augmentation_epoch.value);seed=int(self._augmentation_seed.value)
        rng=np.random.default_rng(np.random.SeedSequence([seed,epoch,int(index)]))
        ri,core_start=self.items[index];d=self._load(ri)
        n,t=len(d['target']),self.sequence_frames
        start=core_start-self.left;lo,hi=max(0,start),min(n,start+t);a,b=lo-start,hi-start
        result={}
        for key,value in d.items():
            if key=='feature_values':continue
            fill=-1 if key=='target' else 0
            arr=np.full((t,*value.shape[1:]),fill,dtype=value.dtype);arr[a:b]=value[lo:hi];result[key]=arr
        values=np.zeros((t,len(self.scaler.names)),np.float32);values[a:b]=d['feature_values'][lo:hi]
        original_mask=result.pop('feature_mask').astype(np.float32)
        physical, original_observed=self.scaler.prepare_base(values,original_mask)
        original_mask=original_observed.astype(np.float32)
        from .schema import FEATURE_NAMES
        auxiliary_indices=[self.scaler.names.index(name) for name in FEATURE_NAMES]
        result['physiology_features']=physical[:,auxiliary_indices].copy()
        result['physiology_mask']=original_mask[:,auxiliary_indices].copy()
        if self.augment:
            if rng.random()<.3:
                k=int(rng.integers(result['quality'].shape[1]))
                observed=(result['spectrogram_mask'].sum(-1)>0)&(result['quality']>0)
                drop=observed.sum(-1)>1
                for key in ('raw_mask','quality','channel_ids','spectrogram_mask','spectrogram'):
                    result[key][drop,k]=0
            scale=float(rng.uniform(.9,1.1))**2
            result['spectrogram']=np.log1p(np.expm1(result['spectrogram'])*scale).astype(np.float32)
            result['spectrogram']*=result['spectrogram_mask'][:,:,None,:]
            original_mask*=rng.random(original_mask.shape)>.05
        details=self.scaler.project_scaled(physical,original_mask)
        result['features']=details['features'].astype(np.float32)
        result['feature_mask']=details['feature_mask'].astype(np.float32)
        result['original_feature_mask']=details['raw_mask']
        from .quality_v9 import quality_context
        result['reliability_context']=quality_context(values,original_mask,details,result['quality'],result['spectrogram_mask'])
        result['numeric_features']=details['numeric_features']
        result['family_coverage']=details['family_coverage']
        result['physical_feature_support']=details['supported'].astype(np.float32)
        result['feature_quality']=details['reliability'].astype(np.float32)
        result['raw']=np.zeros((t,result['quality'].shape[1],1),np.float32)
        result['raw_mask']=result['raw_mask'].astype(np.float32)
        result['boundary_mask']=result['boundary_mask'].astype(np.float32)
        result['frame_valid']=np.zeros(t,np.float32);result['frame_valid'][a:b]=1
        result['physical_feature_support']*=result['frame_valid']
        result['feature_quality']*=result['frame_valid']
        result['core_mask']=np.zeros(t,np.float32)
        core_count=min(self.core_frames,n-core_start)
        result['core_mask'][self.left:self.left+core_count]=self.frame_masks[ri][core_start:core_start+core_count]
        result['frame_indices']=np.full(t,-1,np.int64);result['frame_indices'][a:b]=np.arange(lo,hi)
        tensors={k:torch.from_numpy(np.ascontiguousarray(v)) for k,v in result.items()}
        tensors['record_index']=torch.tensor(ri)
        return tensors


class RecordGroupedSampler(Sampler[int]):
    """Shuffle records and their core blocks while retaining record-local I/O.

    ``SequenceDataset`` loads an entire compressed continuous segment into its
    bounded LRU. A global window shuffle repeatedly decompresses those same
    arrays. This sampler visits every dataset item once per epoch, keeping all
    core blocks from one record adjacent. Both the record order and the order
    of core blocks within each record change reproducibly with ``seed+epoch``.
    With ``interleave_records>1``, round-robin the active continuous segments,
    replacing a slot only when exhausted. Set dataset LRU at least that large;
    this mixes mini-batches without class sampling or repeated decompression.
    The default of 1 preserves the previous grouped order exactly.
    Use with DataLoader ``shuffle=False, num_workers=0`` and call ``set_epoch``
    before iterating each epoch. It does not sample by labels or discard data.
    """

    def __init__(self, dataset: SequenceDataset, seed: int = 20261011,
                 shuffle_within_record: bool = True, interleave_records: int = 1):
        self.seed = int(seed)
        if self.seed < 0:
            raise ValueError("Sampler seed must be nonnegative")
        if not isinstance(interleave_records, int) or interleave_records < 1:
            raise ValueError("interleave_records must be a positive integer")
        self.interleave_records = interleave_records
        self.epoch = 0
        self.shuffle_within_record = bool(shuffle_within_record)
        grouped = {}
        for item_index, (record_index, _) in enumerate(dataset.items):
            grouped.setdefault(int(record_index), []).append(item_index)
        self.groups = tuple(np.asarray(indices, dtype=np.int64)
                            for _, indices in sorted(grouped.items()))
        self.size = len(dataset.items)

    def set_epoch(self, epoch: int):
        epoch = int(epoch)
        if epoch < 0:
            raise ValueError("Sampler epoch must be nonnegative")
        self.epoch = epoch

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        order = iter(rng.permutation(len(self.groups)))
        if self.interleave_records == 1:
            # Preserve the V1 sequence and RNG consumption exactly.
            for group_index in order:
                indices = self.groups[int(group_index)]
                if self.shuffle_within_record:
                    indices = rng.permutation(indices)
                yield from (int(index) for index in indices)
            return

        def next_group():
            group_index = next(order, None)
            if group_index is None:
                return None
            indices = self.groups[int(group_index)]
            if self.shuffle_within_record:
                indices = rng.permutation(indices)
            return iter(int(index) for index in indices)

        # Fixed round-robin slots retain an active segment until exhausted.
        # Refill at the next round, matching an LRU of interleave_records.
        # No labels enter selection, and every core is still visited once.
        active = [next_group() for _ in range(min(self.interleave_records, len(self.groups)))]
        while any(group is not None for group in active):
            for slot, group in enumerate(active):
                if group is None:
                    continue
                index = next(group, None)
                if index is None:
                    active[slot] = next_group()
                    group = active[slot]
                    if group is None:
                        continue
                    index = next(group)
                yield index

    def __len__(self):
        return self.size
