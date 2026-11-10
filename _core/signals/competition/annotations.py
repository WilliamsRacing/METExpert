"""Annotations are optional and never affect numerical inputs or decoding."""
from pathlib import Path
import csv
import json
import numpy as np
from signals.multimodal.labels import LabelPolicy
from signals.multimodal.storage import fingerprint, read_jsonl


def subject(value):
    result=''.join(str(value).split()).upper()
    if not result or any(c in result for c in ',/\\\x00'):raise ValueError('Invalid externalid')
    return result


def intervals(path):
    if not path.exists():return []
    rows=[]
    with path.open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f)
        if not {'externalid','beforeTime','afterTime'}<=set(reader.fieldnames or []):raise ValueError('Annotation CSV needs externalid,beforeTime,afterTime')
        for i,r in enumerate(reader):
            a,b=r['beforeTime'],r['afterTime']
            if any(len(v)!=13 or not v.isascii() or not v.isdigit() for v in (a,b)) or int(a)>=int(b):raise ValueError('Invalid annotation epoch interval')
            rows.append(dict(participant_key=subject(r['externalid']),start_epoch_ms=int(a),end_epoch_ms=int(b),event_id=r.get('event_id') or f'{subject(r["externalid"])}:{a}:{b}'))
    return rows


class CompetitionAnnotations(LabelPolicy):
    native=True
    def __init__(self,root=None):
        root=Path(root) if root else None
        self.annotated=bool(root and (root/'events.csv').exists())
        self.events=intervals(root/'events.csv') if self.annotated else []
        self.unknown=intervals(root/'unknown_intervals.csv') if self.annotated else []
        self.policy=json.loads((root/'policy.json').read_text(encoding='utf-8')) if self.annotated else {}
        if self.annotated and self.policy.get('coverage')!='all_observed':raise ValueError('Explicit all_observed annotation policy required')
        metadata=read_jsonl(root/'event_metadata.jsonl') if self.annotated else []
        by_id={r['event_id']:r for r in metadata}
        if len(by_id)!=len(metadata):raise ValueError('Duplicate event metadata IDs')
        ids=set()
        for e in self.events:
            if e['event_id'] in ids:raise ValueError('Duplicate annotation event ID')
            ids.add(e['event_id'])
            m=by_id.get(e['event_id'],{})
            e.update({k:m[k] for k in ('same_hand','source_row','source_table') if k in m})
        self.identity=json.loads((root/'identity_map.json').read_text(encoding='utf-8')) if self.annotated and (root/'identity_map.json').exists() else {}
        if self.policy.get('unknown_hand_positive_weight',1.)!=1.:
            raise ValueError('Native v1 requires weight 1 for known meals with unknown hand')
        self.fingerprint=fingerprint(dict(version='competition2.annotations.v1',events=self.events,unknown=self.unknown,policy=self.policy,identity=self.identity,annotated=self.annotated))

    def targets(self,key,ticks,observed):
        out=super().targets(key,ticks,observed)
        # 3 is known eating with unknown hand, distinct from unknown annotation.
        known=(out['target_all']>0)&out['supervision_all']&(out['imu_relation']==-1)
        out['imu_relation'][known]=3
        out['annotation_known']=out['supervision_all'].copy()
        return out

    def wholly_known(self,key,start,end):
        return self.annotated and not any(e['participant_key']==key and e['start_epoch_ms']<end and e['end_epoch_ms']>start for e in self.unknown)
