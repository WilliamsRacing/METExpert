"""Convert official dining tables into timestamp supervision for training."""
from __future__ import annotations

import ast
import csv
from datetime import datetime
import hashlib
import io
import json
from pathlib import Path
from zoneinfo import ZoneInfo


def _read_rows(path, allow_trailing_incomplete=False):
    path=Path(path);raw=path.read_bytes();text=None
    for encoding in ('utf-8-sig','gb18030'):
        try:text=raw.decode(encoding);break
        except UnicodeError:continue
    if text is None:raise ValueError('CSV encoding must be UTF-8 or GB18030: '+path.name)
    reader=csv.DictReader(io.StringIO(text,newline=''),strict=True);rows=[];truncated=False
    try:
        for row in reader:
            if None in row or any(v is None for v in row.values()):
                if allow_trailing_incomplete and reader.line_num>=text.count('\n'):
                    truncated=True;break
                raise ValueError(f'{path.name}: incomplete CSV row at line {reader.line_num}')
            rows.append({str(k).strip():str(v).strip() for k,v in row.items()})
    except csv.Error as exc:
        if allow_trailing_incomplete and 'unexpected end of data' in str(exc):truncated=True
        else:raise ValueError(f'{path.name}: malformed CSV at line {reader.line_num}') from exc
    return rows,dict(name=path.name,sha256=hashlib.sha256(raw).hexdigest(),encoding=encoding,
        complete_rows=len(rows),trailing_incomplete_record=truncated)


def _person(value):
    value=''.join(str(value or '').split()).upper()
    if value.lower() in ('','test','测试','nan','null','none','unknown'):return None
    if any(c in value for c in ',/\\\x00'):raise ValueError('Invalid participant ID')
    return value


def _hand(value):
    value=str(value or '').strip().lower()
    if '左' in value or 'left' in value:return 'left'
    if '右' in value or 'right' in value:return 'right'
    return None


def _timestamp(value, timezone):
    value=str(value or '').strip()
    if value.isascii() and value.isdigit() and len(value)==13:return int(value)
    parsed=datetime.fromisoformat(value)
    if parsed.tzinfo is None:parsed=parsed.replace(tzinfo=ZoneInfo(timezone))
    result=int(parsed.timestamp()*1000)
    if not 10**12<=result<10**13:raise ValueError('Expected 13-digit epoch milliseconds')
    return result


def _interval(row, timezone):
    text=row.get('timeFrame','')
    if text:
        try:value=json.loads(text)
        except json.JSONDecodeError:value=ast.literal_eval(text)
        if not isinstance(value,dict):raise ValueError('timeFrame must be a startTime/endTime object')
        a,b=_timestamp(value['startTime'],timezone),_timestamp(value['endTime'],timezone)
    else:a,b=_timestamp(row.get('beforeTime'),timezone),_timestamp(row.get('afterTime'),timezone)
    if a>=b:raise ValueError('Meal start must precede meal end')
    return a,b


def convert_tables(meal_tables, output_dir, *, users_csv=None, timezone='Asia/Shanghai',data_root=None,people=None):
    """Create events.csv, hand-relation metadata and explicit annotation policy.

    Tables are ordered by authority: the first table supplies complete original
    values; repeated IDs in later tables only fill empty fields. A truncated
    final secondary record is omitted while its complete prefix is retained.
    """
    tables=[Path(p) for p in meal_tables]
    if not tables:raise ValueError('At least one dining CSV is required')
    output=Path(output_dir).resolve()
    if any(output==p.resolve().parent for p in tables):raise ValueError('Write converted annotations to a separate directory')
    allowed=set(str(p).upper() for p in people) if people is not None else None
    if data_root is not None:
        from .preprocess import discover_people
        available=set(discover_people(data_root))
        if allowed is not None and not allowed<=available:raise ValueError('Unknown annotation participant filter')
        allowed=available if allowed is None else allowed
    sources=[]
    if users_csv:
        users,audit=_read_rows(users_csv);sources.append(audit)
    by_id={};order=[]
    for index,path in enumerate(tables):
        rows,audit=_read_rows(path,allow_trailing_incomplete=index>0);sources.append(audit)
        for row in rows:
            if not {'uniqueid','externalid'}<=row.keys():raise ValueError('Dining CSV needs uniqueid and externalid columns')
            event_id=row['uniqueid']
            if not event_id:raise ValueError('Dining event uniqueid cannot be empty')
            if event_id in by_id:
                previous=by_id[event_id]['row']
                for key,value in row.items():
                    if not previous.get(key) and value:previous[key]=value
                continue
            by_id[event_id]=dict(row=dict(row),source_table=path.name);order.append(event_id)
    events=[];metadata=[];skipped=[]
    for event_id in order:
        item=by_id[event_id];row=item['row'];person=_person(row.get('externalid'))
        if person is None or (allowed is not None and person not in allowed):
            skipped.append(dict(event_id=event_id,reason='not_in_input'));continue
        try:a,b=_interval(row,timezone)
        except (ValueError,TypeError,KeyError,SyntaxError) as exc:
            raise ValueError('Invalid interval in dining event '+event_id+': '+str(exc)) from exc
        worn,eating=_hand(row.get('wearHand')),_hand(row.get('dietaryHand'))
        same=None if worn is None or eating is None else worn==eating
        events.append(dict(externalid=person,beforeTime=a,afterTime=b,event_id=event_id))
        metadata.append(dict(event_id=event_id,same_hand=same,source_row=row,source_table=item['source_table']))
    events.sort(key=lambda r:(r['externalid'],r['beforeTime'],r['afterTime'],r['event_id']))
    output.mkdir(parents=True,exist_ok=True)
    fields=['externalid','beforeTime','afterTime','event_id']
    with (output/'events.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(events)
    with (output/'unknown_intervals.csv').open('w',encoding='utf-8',newline='') as f:csv.writer(f).writerow(fields)
    (output/'event_metadata.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in metadata),encoding='utf-8')
    policy=dict(coverage='all_observed',different_hand_weight=.25,unknown_hand_positive_weight=1.,
        unknown_priority=True,version='competition2.annotations.v1')
    (output/'policy.json').write_text(json.dumps(policy,ensure_ascii=False,indent=2),encoding='utf-8')
    (output/'identity_map.json').write_text('{}',encoding='utf-8')
    audit=dict(format='eating-detector.annotation-import.1',events=len(events),participants=len({r['externalid'] for r in events}),
        sources=sources,skipped=skipped,timezone=timezone,interval_preference='timeFrame, then beforeTime/afterTime',
        table_authority='Earlier tables supply values; later duplicate IDs fill empty fields')
    (output/'conversion.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    return dict(annotation_root=str(output),events=len(events),participants=audit['participants'])
