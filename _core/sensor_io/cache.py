from __future__ import annotations

from pathlib import Path
import tempfile
import time
from .parallel import record_results
import numpy as np
from . import CACHE_SCHEMA
from .common import identity, read_json, write_json, digest
from .source import spool_txt, infer_sessions, load_callbacks, SOURCE_REVISION
from .preprocessing import build_device_imu


def build_cache(config, people=None):
    try:
        return _build_cache(config, people)
    except Exception as exc:
        root=Path(config["cache_root"]);status=root/"prepare_status.json"
        if status.exists() and read_json(status).get("state")=="preparing":
            write_json(status,dict(state="incomplete",source_revision=SOURCE_REVISION,
                                   error=f"{type(exc).__name__}: {exc}",
                                   recovery="Rerun prepare with the same configuration; completed recording receipts are reused"))
        raise


def _build_cache(config, people=None):
    root = Path(config["cache_root"]); raw = Path(config["input_root"])
    if not raw.is_dir(): raise FileNotFoundError(raw)
    pp = config["preprocessing"]; root.mkdir(parents=True, exist_ok=True)
    bounds = read_json(config["session_manifest"]) if config.get("session_manifest") else []
    if isinstance(bounds, dict): bounds = bounds["sessions"]
    aliases = read_json(config["identity_map"]) if config.get("identity_map") else {}
    aliases = {identity(k): identity(v) for k, v in aliases.items()}
    inventory = {}
    for folder in sorted(p for p in raw.iterdir() if p.is_dir()):
        if not any(p.suffix.lower() == ".txt" for p in folder.iterdir() if p.is_file()): continue
        key = aliases.get(identity(folder.name), identity(folder.name))
        inventory.setdefault(key, []).extend(sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".txt"))
    if people:
        requested = {aliases.get(identity(p), identity(p)) for p in people}
        if requested - inventory.keys(): raise ValueError(f"Missing people: {sorted(requested - inventory.keys())}")
        inventory = {p: files for p, files in inventory.items() if p in requested}
    if not inventory: raise ValueError("No participant TXT files found")
    build = dict(schema=CACHE_SCHEMA, unit="recording", input_root=str(raw.resolve()), preprocessing=pp, aliases=aliases, session_manifest=bounds)
    if (root / "build.json").exists() and read_json(root / "build.json") != build:
        raise ValueError("Cache configuration changed; choose a new cache_root")
    write_json(root / "build.json", build)
    write_json(root/"prepare_status.json",dict(state="preparing",source_revision=SOURCE_REVISION))
    jobs = [(person, recording_file) for person, files in inventory.items() for recording_file in files]
    io=config.get("prepare_io",{})
    workers=int(io.get("workers",1));threads=int(io.get("worker_threads",1))
    if workers<1 or threads<1:raise ValueError("prepare_io workers/worker_threads must be >=1")
    print(f"prepare: {len(jobs)} recordings, {workers} process(es), {threads} numerical thread(s)/process",flush=True)
    jobs.sort(key=lambda job:(-job[1].stat().st_size,job[0],str(job[1])))
    tasks=[(str(root),str(raw),pp,bounds,aliases,person,str(path)) for person,path in jobs]
    timings=[];started=time.perf_counter()
    for result in record_results(_prepare_recording,tasks,workers,threads):
        timings.append(result)
        print(f"prepare progress {len(timings)}/{len(tasks)}: {result['recording_key']} {result['status']} {result['seconds']:.2f}s",flush=True)
    write_json(root/"prepare_timing.json",dict(workers=workers,worker_threads=threads,wall_seconds=time.perf_counter()-started,recordings=timings))
    # A partial preparation must not silently remove already prepared participants.
    all_sessions = {}; all_records = {}
    for receipt in sorted((root / "people").glob("*/*.json")):
        obj = read_json(receipt)
        all_records[obj["recording"]["key"]] = obj["recording"]
        for s in obj["sessions"]: all_sessions[s["key"]] = s
    manifest = dict(schema=CACHE_SCHEMA, unit="recording", build_id=digest(build), sessions=list(all_sessions.values()), recordings=list(all_records.values()),
                    source_revision=SOURCE_REVISION, complete=True)
    write_json(root / "manifest.json", manifest)
    write_json(root/"prepare_status.json",dict(state="complete",source_revision=SOURCE_REVISION,recordings=len(all_records)))
    return manifest


def _prepare_recording(task):
    root,raw,pp,bounds,aliases,person,recording_file=task
    root=Path(root);raw=Path(raw);recording_file=Path(recording_file)
    started=time.perf_counter()
    files = [recording_file]
    rid = recording_file.stem if identity(recording_file.parent.name) == person else recording_file.parent.name + "__" + recording_file.stem
    rkey = person + "/" + rid
    stats = [dict(path=str(p.resolve()), size=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns) for p in files]
    receipt = root / "people" / person / (rid + ".json")
    if receipt.exists():
        old = read_json(receipt)
        if old["source_stats"] != stats: raise ValueError(f"Source changed for {person}; use a new cache_root")
        if all(all((root / s["path"] / filename).exists() for filename in ("manifest.json","data.npy","ticks.npy","observed_mask.npy","imu_support.npy","continuous_segment.npy")) for s in old["sessions"]):
            print(f"cache hit {rkey}", flush=True)
            return dict(recording_key=rkey,status="hit",seconds=time.perf_counter()-started)
    own = []; parse_seconds=0.; callback_seconds=0.; resample_seconds=0.; write_seconds=0.
    with tempfile.TemporaryDirectory(prefix="spool-", dir=root) as tmp:
        spools = []; clocks = []; malformed = []
        for i, path in enumerate(files):
            stage_started=time.perf_counter()
            chunks, times, bad = spool_txt(path, Path(tmp) / str(i), pp)
            parse_seconds+=time.perf_counter()-stage_started
            spools.append(dict(file=str(path), chunks=chunks)); clocks.extend(times)
            malformed.extend(dict(file=str(path), **r) for r in bad)
            print(f"parsed {rkey}", flush=True)
        native = [b for b in bounds if aliases.get(identity(b.get("participant_key", b.get("person"))), identity(b.get("participant_key", b.get("person")))) == person and (not b.get("recording_id") or b["recording_id"] == rid)]
        if bounds and not native: raise ValueError(f"session_manifest has no entries for {person}")
        intervals = [(max(int(b["start_ms"]), min(clocks)), min(int(b["end_ms"]), max(clocks)+1000), str(b.get("session_id", f"{j:05d}"))) for j, b in enumerate(native) if clocks and int(b["start_ms"])<=max(clocks) and int(b["end_ms"])>min(clocks)] if native else [(a, b, f"{j:05d}") for j, (a, b) in enumerate(infer_sessions(clocks, pp))]
        ordered = sorted(intervals)
        if any(a >= b for a, b, _ in ordered) or any(ordered[i][0] < ordered[i - 1][1] for i in range(1, len(ordered))):
            raise ValueError("Session intervals must be positive and non-overlapping")
        if len({sid for _, _, sid in intervals}) != len(intervals): raise ValueError("Duplicate session_id")
        for lo, hi, sid in intervals:
            if not sid or any(c in sid for c in '/\\') or sid in (".", ".."): raise ValueError("Invalid session_id")
            stage_started=time.perf_counter()
            acc, aa = load_callbacks(spools, "acc", lo, hi, pp); gyro, ga = load_callbacks(spools, "gyro", lo, hi, pp)
            callback_seconds+=time.perf_counter()-stage_started;stage_started=time.perf_counter()
            arrays = build_device_imu(acc, gyro, lo, hi, pp)
            resample_seconds+=time.perf_counter()-stage_started;stage_started=time.perf_counter()
            relative = Path("recordings") / person / rid / sid; dest = root / relative; dest.mkdir(parents=True, exist_ok=True)
            for name, value in arrays.items():
                with (dest / (name + ".npy.tmp")).open("wb") as f: np.save(f, value, allow_pickle=False)
                (dest / (name + ".npy.tmp")).replace(dest / (name + ".npy"))
            entry = dict(key=f"{rkey}/{sid}", recording_key=rkey, recording_id=rid, participant_key=person, session_id=sid, start_ms=lo, end_ms=hi, path=relative.as_posix())
            meta = dict(**entry, schema=CACHE_SCHEMA, coordinate_space="raw_device", gravity_removed=False,
                        hand_reflected=False, sample_rate_hz=15, channels=["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"],
                        acc_unit="g", gyro_unit="deg/s", source_stats=stats, acc_audit=aa, gyro_audit=ga,
                        session_source="explicit_manifest" if native else "imu_clock_gap", samples=len(arrays["ticks"]),
                        supported_samples=int(arrays["imu_support"].sum()), preprocessing=pp,source_revision=SOURCE_REVISION)
            write_json(dest / "manifest.json", meta); own.append(entry)
            write_seconds+=time.perf_counter()-stage_started
            for audit in (aa,ga):
                if audit["quarantined_callback_groups"]:
                    print(f"quarantined {rkey} {audit['modality']}: {audit['quarantined_callback_groups']} callbacks, "
                          f"{audit['quarantined_rows_removed']} rows; details: {dest / 'manifest.json'}",flush=True)
            print(f"cached {entry['key']}: {meta['supported_samples']}/{meta['samples']} supported", flush=True)
        recording = dict(key=rkey, recording_id=rid, participant_key=person, source_file=str(recording_file.resolve()),
                         source_relative=recording_file.relative_to(raw).as_posix(), start_ms=min(clocks) if clocks else None,
                         end_ms=max(clocks)+1000 if clocks else None, session_keys=[s["key"] for s in own])
        write_json(receipt, dict(source_stats=stats, sessions=own, recording=recording, malformed_rows=malformed,source_revision=SOURCE_REVISION))
    return dict(recording_key=rkey,status="built",seconds=time.perf_counter()-started,source_bytes=sum(v["size"] for v in stats),parse_seconds=parse_seconds,callback_seconds=callback_seconds,resample_seconds=resample_seconds,write_seconds=write_seconds)
