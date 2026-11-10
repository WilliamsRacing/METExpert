"""Bounded TXT parsing into independent HDF shards, assembled in source order."""
from concurrent.futures import ProcessPoolExecutor
import hashlib
import multiprocessing as mp
from pathlib import Path
import time

import h5py
import numpy as np

from .official.input import DataReader
from .source import SubjectSource, parse_blocks


class _SingleFileReader(DataReader):
    def __init__(self, path):
        super().__init__(Path(path).parent)
        self.path = Path(path)

    def _paths(self):
        return [self.path]


def _parse_file(path, destination, revision='legacy_v1'):
    started = time.perf_counter()
    file_id, text = next(_SingleFileReader(path).iter_data())
    digest = hashlib.sha256(text.encode('utf-8')).hexdigest()
    intervals = []
    with h5py.File(destination, 'w') as handle:
        group = handle.create_group('data')
        group.attrs['file_id'] = file_id
        datasets = {name: group.create_dataset(name, shape=(0, width), maxshape=(None, width),
            dtype=dtype, chunks=(4096, width), compression='lzf')
            for name, width, dtype in [('clock', 3, 'i8'), ('acc', 3, 'f8'), ('gyro', 3, 'f8'), ('ppg', 20, 'f8')]}
        if revision=='exact_repeat_v1':datasets['ppg_proof']=group.create_dataset('ppg_proof',shape=(0,44),maxshape=(None,44),dtype='f8',chunks=(4096,44),compression='lzf')
        lines_ds = group.create_dataset('line', shape=(0,), maxshape=(None,), dtype='i8', chunks=(4096,))
        offset, header_cuts = 0, []
        # Keep the reference block size: interval ownership must remain identical.
        skipped=[]
        for clock, value, lines, cuts in parse_blocks(text, file_id, malformed_rows=skipped):
            header_cuts.extend(cuts)
            size = len(clock)
            if not size:
                continue
            arrays = dict(clock=clock, ppg=value[:, :20], acc=value[:, 44:47], gyro=value[:, 47:50])
            if 'ppg_proof' in datasets:arrays['ppg_proof']=value[:,:44]
            for key, array in arrays.items():
                datasets[key].resize(offset + size, axis=0)
                datasets[key][offset:offset + size] = array
            lines_ds.resize(offset + size, axis=0)
            lines_ds[offset:offset + size] = lines
            for col in range(3):
                times = np.unique(clock[:, col]); times = times[times > 0]
                if len(times):
                    edges = np.r_[0, np.flatnonzero(np.diff(times) > 2000) + 1, len(times)]
                    intervals.extend((int(times[a]), int(times[b-1]) + 1000) for a, b in zip(edges[:-1], edges[1:]))
            offset += size
        group.create_dataset('block_start_rows', data=np.unique(header_cuts), dtype='i8')
    return dict(file_id=file_id, rows=offset, sha256=digest, intervals=intervals,
                skipped_rows=skipped,
                seconds=time.perf_counter() - started)


class ParallelSubjectSource(SubjectSource):
    def __init__(self, person_dir, spool_path, log=print, *, workers=4, **options):
        super().__init__(person_dir, spool_path, log, **options)
        if type(workers) is not int or not 1 <= workers <= 8:
            raise ValueError('TXT reader workers must be between 1 and 8')
        self.workers = workers

    def ingest(self):
        paths = sorted(p for p in self.person_dir.iterdir() if p.is_file() and p.suffix.lower() == '.txt')
        if self.workers == 1 or len(paths) < 2:
            return super().ingest()
        if self.storage != 'hdf5':
            raise ValueError('Parallel inference reader requires HDF shard storage')
        count = min(self.workers, len(paths))
        self.log(f'  TXT reader: {count} CPU processes; {len(paths)} files')
        shards = self.path.with_name(self.path.name + '.parts')
        shards.mkdir(exist_ok=False)
        self.files, self.file_hashes, self.intervals = [], {}, []
        with ProcessPoolExecutor(max_workers=count, mp_context=mp.get_context('spawn')) as pool:
            pending = {}
            for i in range(count):
                pending[i] = pool.submit(_parse_file, str(paths[i]), str(shards / f'{i:06d}.h5'),self.revision)
            with h5py.File(self.path, 'w') as handle:
                for i, path in enumerate(paths):
                    result = pending.pop(i).result()
                    # Each worker exclusively owns its shard. Parent stores links,
                    # so no arrays or complete text pass through process IPC.
                    target = shards / f'{i:06d}.h5'
                    handle[str(i)] = h5py.ExternalLink(str(target.resolve()), '/data')
                    self.files.append((str(i), result['file_id'], result['rows']))
                    self.file_hashes[result['file_id']] = result['sha256']
                    self.intervals.extend(result['intervals'])
                    if result['skipped_rows']:
                        self.skipped_rows[result['file_id']] = result['skipped_rows']
                        first=result['skipped_rows'][0]['line']
                        self.log(f"  TXT skipped {len(result['skipped_rows'])} malformed rows in {result['file_id']}; first line {first}")
                    self.log(f"  TXT {i+1}/{len(paths)}: {result['file_id']} ({result['seconds']:.2f}s)")
                    if self.progress:
                        self.progress('txt_file', file_id=result['file_id'], rows=result['rows'])
                    next_i = i + count
                    if next_i < len(paths):
                        pending[next_i] = pool.submit(_parse_file, str(paths[next_i]), str(shards / f'{next_i:06d}.h5'),self.revision)
        merged = []
        for a, b in sorted(self.intervals):
            if merged and a <= merged[-1][1] + 2000:
                merged[-1] = (merged[-1][0], max(merged[-1][1], b))
            else:
                merged.append((a, b))
        self.intervals = merged
        return self

    def close(self):
        super().close()
        import shutil
        shutil.rmtree(self.path.with_name(self.path.name + '.parts'), ignore_errors=True)
