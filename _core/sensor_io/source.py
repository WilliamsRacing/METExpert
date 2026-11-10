"""Bounded TXT spool; only device ACC/GYRO enter the numerical path."""
from __future__ import annotations

from pathlib import Path
import hashlib
import numpy as np

HEADER = ["ACC_TIME", "PPG_TIME", "GYRO_TIME"] + [f"PPG{i}" for i in range(1, 45)] + ["ACC_X", "ACC_Y", "ACC_Z", "GYRO_X", "GYRO_Y", "GYRO_Z"]
DTYPE = np.dtype([("clock", "<i8"), ("value", "<f8", (3,)), ("line", "<i8"), ("block", "<i4")])
SOURCE_REVISION = "callback-quarantine-v2"


class Callbacks(list):
    """Ordered unambiguous packets plus intervals reconstruction must not bridge."""
    def __init__(self, values=(), quarantined_intervals_ms=()):
        super().__init__(values)
        self.quarantined_intervals_ms = list(quarantined_intervals_ms)


def spool_txt(path, destination, config):
    """Stream one file in bounded chunks; PPG values are never parsed."""
    destination = Path(destination); destination.mkdir(parents=True, exist_ok=True)
    chunks = {k: [] for k in ("acc","gyro")}; clocks = []; malformed = []; block = -1
    numeric=[];line_numbers=[];blocks=[]
    header="\t".join(HEADER)

    def flush_numeric():
        if not numeric:return
        # Only eight selected fields are converted. Never parse the 44 PPG columns.
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("error",DeprecationWarning)
            try:values=np.fromstring("\n".join(numeric).replace("\t"," "),sep=" ",dtype=np.float64)
            except (ValueError,DeprecationWarning) as exc:
                raise ValueError(f"{path}: invalid ACC/GYRO numeric field in lines {line_numbers[0]}..{line_numbers[-1]}") from exc
        if values.size!=len(numeric)*8:
            raise ValueError(f"{path}: invalid ACC/GYRO numeric field in lines {line_numbers[0]}..{line_numbers[-1]}")
        values=values.reshape(-1,8);lines=np.asarray(line_numbers,np.int64);bb=np.asarray(blocks,np.int32)
        for modality,col,axes in (("acc",0,slice(2,5)),("gyro",1,slice(5,8))):
            present=values[:,col]>0
            if not np.isfinite(values[present,axes]).all():
                raise ValueError(f"{path}: non-finite {modality} in lines {lines[0]}..{lines[-1]}")
            if not present.any():continue
            result=np.empty(int(present.sum()),dtype=DTYPE)
            result["clock"]=values[present,col].astype(np.int64);result["value"]=values[present,axes]
            result["line"]=lines[present];result["block"]=bb[present]
            out=destination/f"{modality}-{len(chunks[modality]):06d}.npy"
            np.save(out,result,allow_pickle=False);chunks[modality].append(str(out))
        numeric.clear();line_numbers.clear();blocks.clear()

    with Path(path).open(encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, 1):
            line=line.rstrip("\r\n")
            if not line.strip():continue
            if line==header:
                block += 1; continue
            if block < 0: raise ValueError(f"{path}:{line_no}: official 53-column header required")
            field_count=line.count("\t")+1
            if field_count != 53:
                if config.get("malformed_policy", "error") != "quarantine":
                    raise ValueError(f"{path}:{line_no}: expected 53 columns, got {field_count}")
                malformed.append({"line": line_no, "columns": field_count}); continue
            fields=line.split("\t",3);tail=fields[3].rsplit("\t",6)[1:]
            for key, col in (("acc", 0), ("gyro", 2)):
                token = fields[col]
                if not token.isascii() or not token.isdigit():
                    raise ValueError(f"{path}:{line_no}: {key} clock must be integer epoch ms")
                clock = int(token)
                if clock == 0: continue
                if not 10**12 <= clock < 10**13: raise ValueError(f"{path}:{line_no}: invalid epoch clock")
            numeric.append("\t".join([fields[0],fields[2],*tail]));line_numbers.append(line_no);blocks.append(block)
            if len(numeric)>=65536:flush_numeric()
    flush_numeric()
    # Callback clocks are a compact metadata inventory, not the signal matrix.
    for paths in chunks.values():
        for p in paths:
            x = np.load(p, mmap_mode="r"); clocks.extend(np.unique(x["clock"]).tolist())
    return chunks, sorted(set(clocks)), malformed


def infer_sessions(clocks, config):
    t = np.asarray(sorted(set(clocks)), np.int64)
    if not len(t): return []
    cuts = np.r_[0, np.flatnonzero(np.diff(t) > config.get("session_gap_ms", 3000)) + 1, len(t)]
    return [(int(t[a]), int(t[b - 1]) + 1000) for a, b in zip(cuts[:-1], cuts[1:])]


def load_callbacks(spools, modality, lo, hi, config):
    """Exact replay is removed; ALL non-identical versions are quarantined.

    Neither a longer payload, a matching prefix, nor source order establishes
    which packet is genuine. Ambiguous clocks remain explicit missing intervals.
    """
    policy = config.get("callback_conflict_policy", "quarantine")
    if policy not in ("error", "quarantine"):
        raise ValueError("callback_conflict_policy must be error or quarantine")
    callbacks = {}; scanned = 0; versions = {}; quarantined = set()
    conflicting = set(); rollback_intervals = []
    for source in spools:
        pieces = []
        for p in source["chunks"][modality]:
            x = np.load(p, mmap_mode="r"); keep = (x["clock"] >= lo) & (x["clock"] < hi)
            if keep.any():
                scanned+=int(keep.sum())
                if scanned>config.get("max_session_source_rows",5000000):raise MemoryError("Session source scan exceeds max_session_source_rows")
                pieces.append(np.asarray(x[keep]))
        if not pieces: continue
        x = np.concatenate(pieces)
        cuts = np.r_[0, np.flatnonzero((np.diff(x["clock"]) != 0) | (np.diff(x["block"]) != 0)) + 1, len(x)]
        highest = -1
        for a, b in zip(cuts[:-1], cuts[1:]):
            clock = int(x["clock"][a]); value = x["value"][a:b].copy()
            versions.setdefault(clock, []).append(dict(source=source["file"],
                first_line=int(x["line"][a]), last_line=int(x["line"][b-1]),
                block=int(x["block"][a]), samples=len(value),
                payload_sha256=hashlib.sha256(value.astype("<f8", copy=False).tobytes()).hexdigest()))
            old = callbacks.get(clock)
            exact_replay = old is not None and np.array_equal(old[0], value)
            if exact_replay:
                # A replay of an entire equal packet need not reset physical time.
                highest = max(highest, clock)
                continue
            if old is not None:
                if policy == "error":
                    raise ValueError(f"Conflicting {modality} callback at {clock} in {source['file']}; callback_conflict_policy=error")
                conflicting.add(clock); quarantined.add(clock); callbacks.pop(clock, None)
            if clock < highest:
                if policy == "error":
                    raise ValueError(f"Clock rollback in {source['file']}: {modality} at {clock}")
                rollback_intervals.append((clock, highest))
            highest = max(highest, clock)
            if clock not in quarantined and old is None:
                callbacks[clock] = (value, source["file"], x["line"][a:b].copy())
    ordered_clocks = np.array(sorted(versions), dtype=np.int64)
    for left, right in rollback_intervals:
        quarantined.update(int(c) for c in ordered_clocks[(ordered_clocks>=left)&(ordered_clocks<=right)])
    for clock in quarantined: callbacks.pop(clock, None)
    intervals = []
    for clock in sorted(quarantined):
        index = int(np.searchsorted(ordered_clocks, clock, side="right"))
        if index < len(ordered_clocks): end = int(ordered_clocks[index])
        else:
            # No next anchor exists; use a conservative duration for the missing last packet.
            duration = max(v["samples"] for v in versions[clock]) * 1000 / float(config.get("fallback_native_hz",106.383))
            end = min(hi, int(np.ceil(clock+max(1., duration))))
        if intervals and clock <= intervals[-1][1]: intervals[-1][1] = max(intervals[-1][1], end)
        else: intervals.append([clock, end])
    retained = sum(len(entry[0]) for entry in callbacks.values())
    duplicates = sum(sum(v["samples"] for v in occurrences)-len(callbacks[c][0])
                     for c, occurrences in versions.items() if c in callbacks)
    audit = dict(source_revision=SOURCE_REVISION, modality=modality,
        conflicting_callback_policy=policy, scanned_rows=scanned, retained_rows=retained,
        duplicate_rows_removed=duplicates, conflicting_callback_groups=len(conflicting),
        quarantined_callback_groups=len(quarantined),
        quarantined_rows_removed=sum(v["samples"] for c in quarantined for v in versions[c]),
        quarantined_callback_timestamps_ms=sorted(quarantined), quarantined_intervals_ms=intervals,
        clock_rollback_intervals_ms=[list(v) for v in rollback_intervals],
        quarantined_versions=[dict(clock_ms=c, reason="non_identical_same_timestamp" if c in conflicting else "clock_rollback_interval",
                                   versions=versions[c]) for c in sorted(quarantined)],
        quarantine_reason="All versions excluded; no first/last/longest payload selected; reconstruction forces a missing gap")
    return Callbacks(sorted(callbacks.items()), intervals), audit
