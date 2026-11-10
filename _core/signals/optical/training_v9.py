
from __future__ import annotations
import copy, hashlib, json, math, os, random, platform
from pathlib import Path
import numpy as np
import torch
from .common import write_json


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()


def payload_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def code_fingerprint():
    root=Path(__file__).resolve().parent
    files=sorted(p for p in root.rglob('*.py') if '__pycache__' not in p.parts and 'tests' not in p.parts)
    result={str(p.relative_to(root)).replace('\\','/'):sha256(p) for p in files}
    entry=root.parents[2]/'training/train_ppg_v9.py'
    if entry.is_file():result['training/train_ppg_v9.py']=sha256(entry)
    return result


def cache_fingerprint(cache):
    cache=Path(cache)
    files={name:sha256(cache/name) for name in ('manifest.json','records.jsonl','provenance.json','subject_mapping.json','meals.json','build_spec.json') if (cache/name).is_file()}
    if 'records.jsonl' not in files: raise FileNotFoundError('Cache requires records.jsonl')
    return {'files':files,'sha256':payload_hash(files)}


def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp');write_json(tmp,value);os.replace(tmp,path)


def atomic_torch(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp');torch.save(value,tmp);os.replace(tmp,path)


def capture_rng():
    n=np.random.get_state()
    d={'python':random.getstate(),'numpy':[n[0],n[1].tolist(),int(n[2]),int(n[3]),float(n[4])],'torch':torch.get_rng_state()}
    if torch.cuda.is_available() and torch.cuda.is_initialized(): d['cuda']=torch.cuda.get_rng_state_all()
    return d


def restore_rng(d):
    random.setstate(d['python']);n=d['numpy'];np.random.set_state((n[0],np.asarray(n[1],np.uint32),n[2],n[3],n[4]))
    torch.set_rng_state(d['torch'].cpu())
    if 'cuda' in d:
        if not torch.cuda.is_available() or len(d['cuda'])!=torch.cuda.device_count():raise ValueError('Resume CUDA RNG topology changed')
        torch.cuda.set_rng_state_all(d['cuda'])


class ModelEMA:
    def __init__(self,model,base_decay=.99,reference_batch=4):
        self.model=copy.deepcopy(model).eval().requires_grad_(False)
        self.base_decay=float(base_decay);self.reference_batch=int(reference_batch);self.updates=0;self.samples=0
    @torch.no_grad()
    def update(self,model,actual_sequences):
        decay=self.base_decay**(int(actual_sequences)/self.reference_batch)
        current=model.state_dict()
        for key,value in self.model.state_dict().items():
            src=current[key].detach()
            if key=='fusion_weights' or not value.is_floating_point():value.copy_(src)
            else:value.mul_(decay).add_(src,alpha=1-decay)
        self.updates+=1;self.samples+=int(actual_sequences)
    def state_dict(self):
        return {'base_decay':self.base_decay,'reference_batch':self.reference_batch,'updates':self.updates,'samples':self.samples,'model_state':self.model.state_dict()}
    def load_state_dict(self,d):
        if d['base_decay']!=self.base_decay or d['reference_batch']!=self.reference_batch:raise ValueError('EMA contract changed')
        self.model.load_state_dict(d['model_state']);self.updates=int(d['updates']);self.samples=int(d['samples'])


class StepSchedule:
    """Advance only after a successful optimizer update, including AMP skips."""
    def __init__(self,optimizer,peak,min_lr,total_steps,warmup_steps):
        self.optimizer=optimizer;self.peak=float(peak);self.min_lr=float(min_lr)
        self.total_steps=max(1,int(total_steps));self.warmup_steps=max(1,int(warmup_steps));self.steps=0
        self._apply()
    def _apply(self):
        if self.steps<self.warmup_steps:lr=self.peak*(self.steps+1)/self.warmup_steps
        else:
            x=min(1.,(self.steps-self.warmup_steps)/max(1,self.total_steps-self.warmup_steps))
            lr=self.min_lr+(self.peak-self.min_lr)*.5*(1+math.cos(math.pi*x))
        for group in self.optimizer.param_groups:group['lr']=lr
    def step(self):self.steps+=1;self._apply()
    def state_dict(self):return {k:getattr(self,k) for k in ('peak','min_lr','total_steps','warmup_steps','steps')}
    def load_state_dict(self,d):
        if any(d[k]!=getattr(self,k) for k in ('peak','min_lr','total_steps','warmup_steps')):raise ValueError('Scheduler contract changed')
        self.steps=int(d['steps']);self._apply()


def precision_policy(config,device):
    name=config['training'].get('precision','auto')
    if device.type!='cuda' or not config['training'].get('amp',True):return False,torch.float32,False
    if name=='auto':name='bf16' if torch.cuda.is_bf16_supported() else 'fp16'
    if name=='bf16' and not torch.cuda.is_bf16_supported():raise ValueError('Requested BF16 is not supported on this GPU')
    if name not in ('bf16','fp16','fp32'):raise ValueError('precision must be auto/bf16/fp16/fp32')
    return name!='fp32',{'bf16':torch.bfloat16,'fp16':torch.float16,'fp32':torch.float32}[name],name=='fp16'


def runtime_versions():
    import scipy, pandas
    return {'python':platform.python_version(),'torch':str(torch.__version__),'numpy':np.__version__,'scipy':scipy.__version__,'pandas':pandas.__version__,'cuda_runtime':torch.version.cuda}


def inference_state_fingerprint(checkpoint):
    """Tensor/content identity unaffected by torch.save archive serialization."""
    h=hashlib.sha256()
    h.update(payload_hash({key:checkpoint[key] for key in ('config','scaler','splits')}).encode())
    for key,value in sorted(checkpoint['model_state'].items()):
        x=value.detach().cpu().contiguous()
        h.update(key.encode());h.update(str(x.dtype).encode());h.update(str(tuple(x.shape)).encode())
        h.update(x.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()
