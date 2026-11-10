"""Official text input, exact int64 clocks, disk spool, callback quarantine."""
from pathlib import Path
import io
import re
import shutil
import contextlib
import hashlib
import h5py
import numpy as np
from .official.input import DataReader, InfoReader
from signals.bme.signals import EXPECTED_HEADER
from signals.optical_base.preprocess import _deduplicate_batches

FIELDS = {'acc': (0, 'acc', 47, 50), 'ppg': (1, 'ppg', 3, 47), 'gyro': (2, 'gyro', 50, 53)}
SPOOL_DTYPE = np.dtype([('clock','<i8',(3,)),('ppg','<f8',(20,)),
                         ('acc','<f8',(3,)),('gyro','<f8',(3,)),('line','<i8')])


def parse_blocks(text, file_id, batch_size=4096, malformed_rows=None):
    """Convert valid rows in bounded batches and quarantine bad-width rows.

    ``malformed_rows`` is an optional list populated with ``line``/``fields``
    records for rows whose tab-separated field count is not the official 53.
    Other validation failures remain fatal because silently dropping them could
    change clocks or signal values without an auditable reason.
    """
    header='\t'.join(EXPECTED_HEADER)
    lines=[];row_numbers=[];cuts=[];numeric_count=0;saw_header=False
    batch_size=max(1,int(batch_size))

    def convert(batch,rows,header_cuts):
        values=np.fromstring('\n'.join(batch).replace('\t',' '),sep=' ',dtype=np.float64)
        if values.size!=len(batch)*53:
            for line_no,line in zip(rows,batch):
                fields=line.split('\t')
                try:
                    if len(fields)!=53:raise ValueError(f'expected 53 fields, got {len(fields)}')
                    if any(not re.fullmatch(r'[+-]?[0-9]+',v) for v in fields[:3]):raise ValueError('clock must be an integer')
                    clock=[int(v) for v in fields[:3]]
                    if any(v!=0 and not 10**12<=v<10**13 for v in clock):raise ValueError('clock must be zero (absent) or 13-digit epoch milliseconds')
                    signal=[float(v) for v in fields[3:]]
                    if not np.isfinite(signal).all():raise ValueError('non-finite signal')
                except (ValueError,OverflowError) as exc:raise ValueError(f'{file_id}:{line_no}: {exc}') from exc
            raise ValueError(f'{file_id}:{rows[0]}: invalid numeric field')
        values=values.reshape(len(batch),53)
        clocks=values[:,:3]
        if not np.isfinite(values).all() or not np.equal(clocks,np.floor(clocks)).all() or not np.all((clocks==0)|((clocks>=10**12)&(clocks<10**13))):
            for line_no,line in zip(rows,batch):
                fields=line.split('\t')
                try:
                    if any(not re.fullmatch(r'[+-]?[0-9]+',v) for v in fields[:3]):raise ValueError('clock must be an integer')
                    clock=[int(v) for v in fields[:3]]
                    if any(v!=0 and not 10**12<=v<10**13 for v in clock):raise ValueError('clock must be zero (absent) or 13-digit epoch milliseconds')
                    if not np.isfinite([float(v) for v in fields[3:]]).all():raise ValueError('non-finite signal')
                except (ValueError,OverflowError) as exc:raise ValueError(f'{file_id}:{line_no}: {exc}') from exc
            raise ValueError(f'{file_id}:{rows[0]}: invalid numeric field')
        return clocks.astype(np.int64),values[:,3:],np.asarray(rows,np.int64),header_cuts

    for line_no,line in enumerate(io.StringIO(text),1):
        line=line.strip().lstrip('\ufeff')
        if not line:continue
        if line==header:
            saw_header=True;cuts.append(numeric_count);continue
        if not saw_header:raise ValueError(f'{file_id}:{line_no}: expected the official 53-column header')
        field_count=line.count('\t')+1
        if field_count!=53:
            if malformed_rows is not None:
                malformed_rows.append({'line': line_no, 'fields': field_count})
            continue
        for clock in line.split('\t',3)[:3]:
            digits=clock[1:] if clock[:1] in ('+','-') else clock
            if not digits.isascii() or not digits.isdigit():raise ValueError(f'{file_id}:{line_no}: clock must be an integer')
            value=int(clock)
            if value!=0 and not 10**12<=value<10**13:raise ValueError(f'{file_id}:{line_no}: clock must be zero (absent) or 13-digit epoch milliseconds')
        lines.append(line);row_numbers.append(line_no);numeric_count+=1
        if len(lines)>=batch_size:
            yield convert(lines,row_numbers,cuts);lines=[];row_numbers=[];cuts=[]
    if lines or cuts:yield convert(lines,row_numbers,cuts)


class SubjectSource:
    def __init__(self, person_dir, spool_path, log=print, progress=None, storage='hdf5', revision='legacy_v1'):
        from .exact_replay import LEGACY, REVISION
        if revision not in (LEGACY,REVISION):raise ValueError('Unknown source revision')
        self.revision=revision
        self.person_dir = Path(person_dir)
        self.path = Path(spool_path)
        self.log = log
        self.progress = progress
        self.storage = storage
        self.files = []
        self.file_hashes = {}
        self.skipped_rows = {}
        self.intervals = []

    def ingest(self):
        if self.storage == 'memmap':
            return self._ingest_memmap()
        with h5py.File(self.path, 'w') as h:
            for number, (file_id, text) in enumerate(DataReader(self.person_dir).iter_data()):
                self.file_hashes[file_id]=hashlib.sha256(text.encode('utf-8')).hexdigest()
                self.log(f'  TXT {number+1}: {file_id}')
                group = h.create_group(str(number))
                group.attrs['file_id'] = file_id
                datasets = {name: group.create_dataset(name, shape=(0,width), maxshape=(None,width),
                             dtype=dt, chunks=(4096,width), compression='lzf')
                             for name,width,dt in [('clock',3,'i8'),('acc',3,'f8'),('gyro',3,'f8'),('ppg',20,'f8')]}
                if self.revision=='exact_repeat_v1':
                    datasets['ppg_proof']=group.create_dataset('ppg_proof',shape=(0,44),maxshape=(None,44),dtype='f8',chunks=(4096,44),compression='lzf')
                line_ds = group.create_dataset('line', shape=(0,), maxshape=(None,), dtype='i8', chunks=(4096,))
                offset, header_cuts = 0, []
                skipped=[]
                for clock, value, lines, cuts in parse_blocks(text, file_id, malformed_rows=skipped):
                    header_cuts.extend(cuts)
                    size = len(clock)
                    if not size:
                        continue
                    # V5 candidate routing only uses physical channels 1..20;
                    # channels 21..44 are validated by the parser, not spooled.
                    arrays = {'clock': clock, 'ppg': value[:,:20], 'acc': value[:,44:47], 'gyro': value[:,47:50]}
                    if 'ppg_proof' in datasets:arrays['ppg_proof']=value[:,:44]
                    for key,array in arrays.items():
                        datasets[key].resize(offset+size,axis=0)
                        datasets[key][offset:offset+size] = array
                    line_ds.resize(offset+size,axis=0); line_ds[offset:offset+size] = lines
                    for col in range(3):
                        times = np.unique(clock[:,col]); times = times[times>0]
                        if len(times):
                            edges = np.r_[0,np.flatnonzero(np.diff(times)>2000)+1,len(times)]
                            self.intervals.extend((int(times[a]),int(times[b-1])+1000) for a,b in zip(edges[:-1],edges[1:]))
                    offset += size
                if skipped:
                    self.skipped_rows[file_id] = skipped
                    self.log(f'  TXT skipped {len(skipped)} malformed rows in {file_id}; first line {skipped[0]["line"]}')
                group.create_dataset('block_start_rows', data=np.unique(header_cuts), dtype='i8')
                self.files.append((str(number),file_id,offset))
                if self.progress:
                    self.progress('txt_file',file_id=file_id,rows=offset)
                del text
        merged = []
        for a,b in sorted(self.intervals):
            if merged and a <= merged[-1][1]+2000:
                merged[-1] = (merged[-1][0],max(merged[-1][1],b))
            else:
                merged.append((a,b))
        self.intervals = merged
        return self

    def _ingest_memmap(self):
        """Use uncompressed structured .npy maps for the transient source spool.

        HDF5 remains supported for the legacy Windows/runtime tests.  Native TXT
        preparation uses this path so gzip/chunk metadata writes do not sit
        between TXT parsing and numerical preprocessing.
        """
        self.path.mkdir(parents=True, exist_ok=True)
        for number, (file_id, text) in enumerate(DataReader(self.person_dir).iter_data()):
            self.file_hashes[file_id]=hashlib.sha256(text.encode('utf-8')).hexdigest()
            capacity=text.count('\n')+1
            count=0;offset=0;header_cuts=[]
            path = self.path / f'{number:06d}.npy'
            if capacity:
                dtype=np.dtype(SPOOL_DTYPE.descr+[('ppg_proof','<f8',(44,))]) if self.revision=='exact_repeat_v1' else SPOOL_DTYPE
                mapped = np.lib.format.open_memmap(path, mode='w+', dtype=dtype, shape=(capacity,))
            else:
                np.save(path,np.empty((0,),dtype=SPOOL_DTYPE));mapped=None
            skipped=[]
            for clock, values, lines, cuts in parse_blocks(text, file_id, batch_size=16_384, malformed_rows=skipped):
                size=len(clock)
                if size:
                    mapped['clock'][offset:offset+size]=clock
                    mapped['ppg'][offset:offset+size]=values[:,:20]
                    if 'ppg_proof' in mapped.dtype.names:mapped['ppg_proof'][offset:offset+size]=values[:,:44]
                    mapped['acc'][offset:offset+size]=values[:,44:47]
                    mapped['gyro'][offset:offset+size]=values[:,47:50]
                    mapped['line'][offset:offset+size]=lines
                header_cuts.extend(cuts);offset+=size
            if skipped:
                self.skipped_rows[file_id] = skipped
                self.log(f'  TXT skipped {len(skipped)} malformed rows in {file_id}; first line {skipped[0]["line"]}')
            count=offset
            if mapped is not None:mapped.flush();del mapped
            np.save(self.path / f'{number:06d}.blocks.npy', np.asarray(header_cuts, np.int64))
            self.files.append((str(path), file_id, count))
            if self.progress:
                self.progress('txt_file', file_id=file_id, rows=count)
            if count:
                view = np.lib.format.open_memmap(path, mode='r')
                for col in range(3):
                    times = np.unique(view['clock'][:count, col])
                    times = times[times > 0]
                    if len(times):
                        edges = np.r_[0, np.flatnonzero(np.diff(times) > 2000) + 1, len(times)]
                        self.intervals.extend((int(times[a]), int(times[b-1]) + 1000)
                                              for a, b in zip(edges[:-1], edges[1:]))
                del view
            del text
        merged = []
        for a, b in sorted(self.intervals):
            if merged and a <= merged[-1][1] + 2000:
                merged[-1] = (merged[-1][0], max(merged[-1][1], b))
            else:
                merged.append((a, b))
        self.intervals = merged
        return self

    def modality(self, modality, lo, hi, maximum, audit):
        col, field, _, _ = FIELDS[modality]
        clocks, values, lines, owners, boundaries = [], [], [], [], []
        total = 0
        quarantined = set()
        exact_removed=0
        with (h5py.File(self.path,'r') if self.storage!='memmap' else contextlib.nullcontext(None)) as h:
            for owner,(key,file_id,count) in enumerate(self.files):
                if count==0:continue
                group=h[key] if self.storage!='memmap' else np.load(key,mmap_mode='r',allow_pickle=False)
                cuts=(group['block_start_rows'][:] if self.storage!='memmap' else
                      np.load(str(Path(key).with_suffix('.blocks.npy')),allow_pickle=False))
                has_proof=('ppg_proof' in group if self.storage!='memmap' else 'ppg_proof' in group.dtype.names)
                proof_field='ppg_proof' if modality=='ppg' and self.revision=='exact_repeat_v1' and has_proof else field
                cc,vv,ll,rr = [],[],[],[]
                for offset in range(0,count,65536):
                    stop=min(count,offset+65536)
                    raw=group['clock'][offset:stop,col]
                    keep=(raw>0)&(raw>=lo-1000)&(raw<hi+1000)
                    if not keep.any():
                        continue
                    indices=np.flatnonzero(keep)
                    total+=len(indices)
                    if total>maximum:
                        raise MemoryError(f'{modality} session exceeds {maximum} source rows; increase max_session_source_rows')
                    cc.append(raw[indices]); vv.append(group[proof_field][offset:stop][indices])
                    ll.append(group['line'][offset:stop][indices]); rr.append(offset+indices)
                if not cc:
                    continue
                c,v,l,r=map(np.concatenate,(cc,vv,ll,rr))
                if self.revision=='exact_repeat_v1' and (modality!='ppg' or has_proof):
                    from .exact_replay import retained_rows
                    unique,removed=retained_rows(c,v,r,cuts)
                    exact_removed+=removed
                    c,v,l,r=c[unique],v[unique],l[unique],r[unique]
                if modality=='ppg':v=v[:,:20]
                backwards=np.flatnonzero(np.diff(c)<0)+1
                keep=np.ones(len(c),bool)
                for j in backwards:
                    bad=(c>=c[j])&(c<=c[j-1])
                    quarantined.update(c[bad].tolist()); keep &= ~bad
                c,v,l,r=c[keep],v[keep],l[keep],r[keep]
                if not len(c):
                    continue
                base=sum(len(x) for x in clocks)
                boundaries.append(base)
                boundaries.extend(base+int(np.searchsorted(r,x)) for x in cuts if 0<np.searchsorted(r,x)<len(c))
                clocks.append(c);values.append(v);lines.append(l);owners.append(np.full(len(c),owner,np.int32))
                if self.storage=='memmap':del group
        if not clocks:
            audit.append({'modality':modality,'source_revision':self.revision,
                'exact_replay_rows_removed':exact_removed,'retained_rows':0,
                'rollback_quarantined_timestamps':len(quarantined)})
            return None
        c,v,l,o=map(np.concatenate,(clocks,values,lines,owners))
        # Float64 represents every 13-digit integer exactly; the stored clocks stay int64.
        joined=np.column_stack((c,v))
        try:
            clean,info,keep=_deduplicate_batches(joined,sorted(set(boundaries)),'quarantine',return_mask=True)
        except ValueError as exc:
            audit.append({'modality':modality,'error':str(exc)})
            raise ValueError(f'{modality}: ambiguous source clock: {exc}') from exc
        quarantined.update(info['quarantined_callback_timestamps_ms'])
        if info['quarantined_callback_timestamps_ms']:
            raise ValueError(f'{modality}: conflicting overlapping sources; cannot infer device identity')
        audit.append({'modality':modality,'callbacks':{**info,'quarantined_callback_timestamps_ms':sorted(quarantined)}})
        c,v,l,o=c[keep],v[keep],l[keep],o[keep]
        selected=~np.isin(c,list(quarantined))
        c,v,l,o=c[selected],v[selected],l[selected],o[selected]
        audit.append({'modality':modality,'source_revision':self.revision,
            'exact_replay_rows_removed':exact_removed,'retained_rows':len(c),
            'rollback_quarantined_timestamps':len(quarantined)})
        order=np.argsort(c,kind='stable')
        return c[order],v[order],l[order],o[order]

    def close(self):
        if self.storage=='memmap':shutil.rmtree(self.path,ignore_errors=True)
        else:self.path.unlink(missing_ok=True)


def participants(data_root):
    root=Path(data_root)
    if not root.is_dir():
        raise FileNotFoundError(f'Place ACC_test_data beside the EXE: {root}')
    infos=InfoReader(root/'userinfobean.txt')
    seen=set()
    for directory in sorted(p for p in root.iterdir() if p.is_dir()):
        subject=''.join(directory.name.split()).upper()
        if subject in seen:
            raise ValueError('Duplicate normalized subject folder: '+subject)
        seen.add(subject)
        yield subject,directory,infos.read_info(subject)
