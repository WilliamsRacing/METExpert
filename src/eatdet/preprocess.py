"""Competition TXT recordings to physical IMU frames and optical features.

Preprocessing is independent of meal annotations. When an annotation directory
is supplied, supervision is written alongside the same numerical arrays.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

import numpy as np


DEVICE_SETTINGS = dict(
    acc_scale_g=.00024414, gyro_scale_dps=.06999997270451806,
    fallback_native_hz=106.383, lowpass_hz=6,
    max_interpolation_gap_ms=250, session_gap_ms=3000,
    saturation_counts=32767, max_session_source_rows=5000000,
    max_session_seconds=172800, malformed_policy="error")
NATIVE_SETTINGS = dict(
    source_revision="legacy_v1", default_orientation="standard",
    max_session_source_rows=5000000, resolved_preprocessing_backend="cpu",
    cuda_preprocess_batch_frames=16)


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _write(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+".tmp")
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    os.replace(temporary,path)


def _sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def discover_people(data_root, people=None):
    """Return normalized external IDs and their original participant folders."""
    root=Path(data_root).resolve()
    if not root.is_dir():raise FileNotFoundError(root)
    inventory={}
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        files=sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower()=='.txt')
        if not files:continue
        person=''.join(folder.name.split()).upper()
        if not person or any(c in person for c in ',/\\\x00'):raise ValueError('Invalid participant folder')
        if person in inventory:raise ValueError('Duplicate normalized participant: '+person)
        inventory[person]=folder
    if people is not None:
        requested={''.join(str(p).split()).upper() for p in people}
        missing=requested-set(inventory)
        if missing:raise ValueError('Requested participant folders absent: '+', '.join(sorted(missing)))
        inventory={p:inventory[p] for p in sorted(requested)}
    if not inventory:raise ValueError('No participant folders with competition TXT recordings')
    return inventory


def _source_contract(inventory, annotation_root, split_file):
    files=[]
    for person,folder in inventory.items():
        for path in sorted(folder.iterdir()):
            if path.is_file() and path.suffix.lower()=='.txt':
                stat=path.stat();files.append(dict(person=person,path=str(path.resolve()),size=stat.st_size,mtime_ns=stat.st_mtime_ns))
    annotations=[]
    if annotation_root:
        for path in sorted(Path(annotation_root).iterdir()):
            if path.is_file():annotations.append(dict(name=path.name,sha256=_sha(path)))
    return dict(format='eating-detector.preparation.1',sources=files,
        annotations=annotations,split_sha256=_sha(split_file) if split_file else None,
        device_preprocessing=DEVICE_SETTINGS,native_preprocessing=NATIVE_SETTINGS)


def _save_imu_frames(device_root, imu_root, labels, people, split):
    from motion_encoder.data import frame_arrays, frame_labels, SCHEMA, QUALITY_NAMES, CHANNELS
    source=_read(device_root/'manifest.json');records=[]
    sessions=sorted((s for s in source['sessions'] if s['participant_key'] in people),
        key=lambda s:(s['participant_key'],int(s['start_ms']),s['key']))
    contract=dict(schema=SCHEMA,sample_rate_hz=15,frame_points=450,channels=CHANNELS,
        preprocessing=DEVICE_SETTINGS,source_manifest_sha256=_sha(device_root/'manifest.json'),
        annotation_fingerprint=labels.fingerprint)
    signature=hashlib.sha256(json.dumps(contract,sort_keys=True).encode()).hexdigest()
    for number,entry in enumerate(sessions):
        source_folder=device_root/entry['path'];folder=imu_root/'records'/f'{number:06d}'
        receipt=folder/'manifest.json'
        if receipt.exists():
            row=_read(receipt)
            if row['signature']!=signature or row['source_key']!=entry['key']:raise ValueError('Changed IMU preparation contract')
            if not all((folder/(name+'.npy')).exists() for name in ('imu','point_mask','frame_quality','segment_ids','target','target_fraction','boundary','label_mask')):
                raise ValueError('Incomplete IMU record: '+str(folder))
        else:
            arrays={name:np.load(source_folder/(name+'.npy'),allow_pickle=False) for name in
                ('data','ticks','observed_mask','imu_support','continuous_segment')}
            if not len(arrays['ticks']):continue
            framed,origin,offset=frame_arrays(arrays['data'],arrays['ticks'],arrays['observed_mask'],
                arrays['imu_support'],arrays['continuous_segment'])
            framed.update(frame_labels(labels,entry['participant_key'],arrays['ticks'],origin,len(framed['imu']),offset))
            folder.mkdir(parents=True,exist_ok=True)
            for name,value in framed.items():np.save(folder/(name+'.npy'),value,allow_pickle=False)
            known=framed['label_mask']&(framed['point_mask'].mean(-1)>=.5)
            row=dict(signature=signature,person=entry['participant_key'],source_key=entry['key'],
                recording_key=entry.get('recording_key'),start_tick=int(origin),frame_count=len(framed['imu']),
                path=folder.relative_to(imu_root).as_posix(),supported_points=int(framed['point_mask'].sum()),
                positive_frames=np.flatnonzero(known&(framed['target']>0)).tolist(),labeled_frames=np.flatnonzero(known).tolist())
            _write(receipt,row)
        records.append(row)
    manifest=dict(schema=SCHEMA,signature=signature,contract=contract,records=records,complete=True,
        root=str(imu_root),source_root=str(device_root),annotation_root=labels.root if hasattr(labels,'root') else '',
        split=split,quality_names=QUALITY_NAMES,
        normalization=dict(mean=[0.]*12,std=[1.]*12,people=[],fit_scope='checkpoint_supplied_at_inference'))
    _write(imu_root/'manifest.json',manifest)
    return manifest


def _prepare_native(inventory, native_root, labels, txt_workers, preprocess_workers):
    """Preserve subject-wide session clock and original optical feature state."""
    from signals.competition.parallel_source import ParallelSubjectSource
    from data_pipeline.data import _save_session
    native_root.mkdir(parents=True,exist_ok=True)
    sessions=[];records=[]
    pp=dict(NATIVE_SETTINGS,txt_read_workers=int(txt_workers),preprocess_workers=int(preprocess_workers))
    for person,folder in inventory.items():
        receipt=native_root/'people'/(person+'.json')
        if receipt.exists():
            saved=_read(receipt);sessions.extend(saved['sessions']);records.extend(saved['records']);continue
        own_sessions=[];own_records=[]
        with tempfile.TemporaryDirectory(prefix='source-',dir=native_root) as work:
            source=ParallelSubjectSource(folder,Path(work)/'subject.h5',workers=int(txt_workers),revision=pp['source_revision']).ingest()
            try:
                prepared=_prepared_sessions(source,pp,pp['default_orientation'],DEVICE_SETTINGS['acc_scale_g'],DEVICE_SETTINGS['gyro_scale_dps'])
                for index,((lo,hi),(imu,ppg,audit,timing)) in enumerate(zip(source.intervals,prepared)):
                    session,entries=_save_session(native_root,person,index,lo,hi,imu,ppg,labels)
                    own_sessions.append(session);own_records.extend(entries)
                    del imu,ppg
            finally:source.close()
        _write(receipt,dict(person=person,sessions=own_sessions,records=own_records))
        sessions.extend(own_sessions);records.extend(own_records)
    _write(native_root/'manifest.json',dict(schema='fast-release-v2-cache-3',root=str(native_root),sessions=sessions,people=list(inventory),complete=True))
    ppg_root=native_root/'ppg';ppg_root.mkdir(exist_ok=True)
    (ppg_root/'records.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records),encoding='utf-8')
    _write(ppg_root/'manifest.json',dict(schema_version='ppg-eating-cache-v8',records_count=len(records),complete=True,
        preview_only=False,errors=[],background_weight=.25,fs=25.,frame_seconds=30.,top_k=3))
    return sessions,records


def _prepare_native_session(path,files,lo,hi,maximum,direction,acc_scale,gyro_scale,
                    backend='cpu',cuda_batch_frames=16,source_revision='legacy_v1'):
    """Independent sessions retain all within-session AHRS/baseline state."""
    from signals.competition.source import SubjectSource
    from signals.competition.preprocessing import build_standard_imu, build_ppg
    started=time.perf_counter()
    source=SubjectSource(Path(path).parent,Path(path),lambda _:None,revision=source_revision)
    source.files=files
    audit=[]
    acc=source.modality('acc',lo,hi,maximum,audit)
    gyro=source.modality('gyro',lo,hi,maximum,audit)
    optical=source.modality('ppg',lo,hi,maximum,audit)
    read_seconds=time.perf_counter()-started
    imu=build_standard_imu(acc,gyro,lo,hi,orientation=direction,acc_scale=acc_scale,gyro_scale=gyro_scale)
    imu_seconds=time.perf_counter()-started-read_seconds
    ppg=build_ppg(optical,lo,hi,audit)
    return imu,ppg,audit,{'source_read':read_seconds,'imu':imu_seconds,
                         'ppg':time.perf_counter()-started-read_seconds-imu_seconds}

def _prepared_sessions(source,config,direction,acc_scale,gyro_scale):
    backend=config.get('resolved_preprocessing_backend','cpu')
    workers=1 if backend=='cuda' else int(config.get('preprocess_workers',1))
    batch_frames=int(config.get('cuda_preprocess_batch_frames',16))
    args=lambda lo,hi:(str(source.path),source.files,lo,hi,int(config['max_session_source_rows']),
                       direction,acc_scale,gyro_scale,backend,batch_frames,config.get('source_revision','legacy_v1'))
    if workers<=1:
        for lo,hi in source.intervals:
            yield _prepare_native_session(*args(lo,hi))
        return
    # Bound queued outputs: a long session can contain large waveform arrays.
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        pending={}
        for index in range(min(workers,len(source.intervals))):
            pending[index]=pool.submit(_prepare_native_session,*args(*source.intervals[index]))
        for index in range(len(source.intervals)):
            value=pending.pop(index).result()
            next_index=index+workers
            if next_index<len(source.intervals):
                pending[next_index]=pool.submit(_prepare_native_session,*args(*source.intervals[next_index]))
            yield value

def prepare(data_root, work_dir, *, annotation_root=None, split_file=None, people=None,
            workers=4, device_workers=1, preprocess_workers=1):
    """Materialize a self-contained preparation work directory and return its map.

    The work directory is an output location; no file in data_root is changed.
    Optional annotations use events.csv, policy.json, event_metadata.jsonl and
    unknown_intervals.csv. Optional split_file contains train, val and test IDs.
    """
    data_root=Path(data_root).resolve();root=Path(work_dir).resolve()
    if root==data_root or data_root in root.parents:raise ValueError('Work directory must be outside the original dataset')
    if not 1<=int(workers)<=8 or not 1<=int(preprocess_workers)<=8 or int(device_workers)<1:
        raise ValueError('Worker counts must be positive; TXT/session workers must not exceed 8')
    inventory=discover_people(data_root,people)
    root.mkdir(parents=True,exist_ok=True)
    contract=_source_contract(inventory,annotation_root,split_file)
    contract_file=root/'preparation_contract.json'
    if contract_file.exists() and _read(contract_file)!=contract:raise ValueError('Inputs changed; choose a new work directory')
    _write(contract_file,contract)
    from signals.competition.annotations import CompetitionAnnotations
    from sensor_io.cache import build_cache
    labels=CompetitionAnnotations(annotation_root)
    split=_read(split_file) if split_file else dict(train=list(inventory),val=[],test=[])
    if 'split' in split:split=split['split']
    parts=[set(split.get(k,[])) for k in ('train','val','test')]
    if any(parts[i]&parts[j] for i,j in ((0,1),(0,2),(1,2))):raise ValueError('Training, validation and test identities overlap')
    if split_file and not set(inventory)<=set.union(*parts):raise ValueError('Split must assign every prepared participant')
    device_root=root/'device';imu_root=root/'imu';native_root=root/'native'
    build_cache(dict(input_root=str(data_root),cache_root=str(device_root),preprocessing=DEVICE_SETTINGS,
        prepare_io=dict(workers=int(device_workers),worker_threads=1)),people=list(inventory))
    imu=_save_imu_frames(device_root,imu_root,labels,set(inventory),split)
    imu['annotation_root']=str(Path(annotation_root).resolve()) if annotation_root else None
    _write(imu_root/'manifest.json',imu)
    sessions,records=_prepare_native(inventory,native_root,labels,int(workers),int(preprocess_workers))
    prepared=dict(format='eating-detector.prepared.1',root=str(root),data_root=str(data_root),people=list(inventory),
        imu_manifest=str(imu_root/'manifest.json'),native_root=str(native_root),ppg_cache_root=str(native_root/'ppg'),
        annotation_root=str(Path(annotation_root).resolve()) if annotation_root else None,
        split_file=str(Path(split_file).resolve()) if split_file else None,
        sessions=len(sessions),ppg_records=len(records),complete=True)
    _write(root/'prepared.json',prepared)
    return prepared


def load_prepared(path):
    path=Path(path)
    if path.is_dir():path=path/'prepared.json'
    value=_read(path)
    if value.get('format')!='eating-detector.prepared.1' or not value.get('complete'):
        raise ValueError('A complete prepared.json is required')
    for key in ('imu_manifest','native_root','ppg_cache_root'):
        target=Path(value[key])
        if not target.is_absolute():target=path.parent/target
        if not target.exists():raise FileNotFoundError(target)
        value[key]=str(target.resolve())
    return value
