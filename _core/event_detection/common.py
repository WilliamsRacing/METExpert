
from __future__ import annotations
import hashlib, json, os, time
from pathlib import Path
import numpy as np
import torch

def plain(value):
    if isinstance(value,dict):return {str(k):plain(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [plain(v) for v in value]
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,Path):return str(value)
    return value

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(plain(value),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');os.replace(tmp,path)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def digest(value):return hashlib.sha256(json.dumps(plain(value),sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def save_npz(path,arrays):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.tmp')
    with tmp.open('wb') as f:np.savez(f,**arrays)
    os.replace(tmp,path)
def save_torch(path,payload):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.tmp')
    torch.save(payload,tmp);os.replace(tmp,path)
def progress(root,phase,**values):
    d=dict(phase=phase,pid=os.getpid(),unix_time=time.time(),**plain(values));write(Path(root)/'progress.json',d);print(json.dumps(d),flush=True)

def make_folds(split,count=4,seed=20261029):
    """Keep shared-source aliases atomic; assign only original train identities."""
    people=set(split['train']);groups=[];seen=set()
    for group in split.get('shared_source_atomic_groups',[]):
        own=sorted(people&set(group))
        if own:
            if seen&set(own):raise ValueError('Overlapping atomic groups')
            groups.append(own);seen.update(own)
    groups.extend([[p] for p in sorted(people-seen)])
    groups.sort(key=lambda g:(-len(g),digest(dict(seed=seed,group=g))))
    folds=[[] for _ in range(count)]
    for group in groups:
        i=min(range(count),key=lambda x:(len(folds[x]),x));folds[i].extend(group)
    result=[]
    for i,held in enumerate(folds):
        held=sorted(held);fit=sorted(people-set(held))
        result.append(dict(fold_id=i,train_people=fit,prediction_people=held,selection_people=[]))
    if set().union(*(set(r['prediction_people']) for r in result))!=people:raise AssertionError('Incomplete fold ownership')
    return result

def scope_digest(rows):
    result={}
    for person in sorted({r['person'] for r in rows}):
        own=[r for r in rows if r['person']==person]
        fields={k:np.concatenate([np.asarray(r[k]).reshape(-1) for r in own]) for k in ('ticks','known','observed','target')}
        keep=fields['known'].astype(bool)|fields['observed'].astype(bool);order=np.argsort(fields['ticks'][keep],kind='stable');h=hashlib.sha256()
        for key,dtype in (('ticks','<i8'),('known','u1'),('observed','u1'),('target','<f4')):
            h.update(key.encode());h.update(np.asarray(fields[key][keep][order],dtype=dtype).tobytes())
        result[person]=h.hexdigest()
    return result
