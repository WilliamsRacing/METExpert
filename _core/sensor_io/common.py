from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def read_cache_manifest(root):
    root=Path(root);path=root/"manifest.json";status=root/"prepare_status.json"
    state=read_json(status) if status.exists() else {}
    if not path.exists() or state.get("state") in ("preparing","incomplete"):
        reason=state.get("error","prepare has not finished successfully")
        raise RuntimeError(f"IMU cache is incomplete: {reason}. Run prepare successfully before audit/split/train. "
                           "Keep existing completed recording receipts; do not create an empty manifest.json.")
    return read_json(path)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def identity(value):
    key = "".join(str(value or "").split()).upper()
    if not key or any(c in key for c in '/\\\0,:') or key in (".", ".."):
        raise ValueError(f"Invalid identity: {value!r}")
    return key


def load_config(path):
    path = Path(path).resolve()
    c = read_json(path)
    base = path.parent.parent if path.parent.name == "config" else path.parent
    for key in ("input_root", "annotation_path", "cache_root", "split_file", "run_dir", "identity_map", "session_manifest"):
        if c.get(key):
            p = Path(c[key]).expanduser()
            c[key] = str((base / p).resolve() if not p.is_absolute() else p.resolve())
    for key in ("cache_root","run_dir","split_file"):
        if not c.get(key):continue
        out=Path(c[key]);protected=[]
        if c.get("input_root"):protected.append(Path(c["input_root"]))
        if c.get("annotation_path"):
            annotation=Path(c["annotation_path"])
            if annotation.is_dir():protected.append(annotation)
            elif out==annotation:raise ValueError(f"{key} cannot overwrite annotation file")
        if any(raw==out or raw in out.parents for raw in protected):
            raise ValueError(f"{key} must not write into raw inputs or annotation directory")
        if any(part.lower().startswith("fast_release_v2") or part.lower()=="fast-release-v2" for part in out.parts):
            raise ValueError("Teacher outputs must not be placed inside a V2 directory")
    return c


def jsonl(path):
    with Path(path).open(encoding="utf-8-sig") as f:
        for i, line in enumerate(f, 1):
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"{path}:{i}: expected JSON object")
                yield row


def write_jsonl(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    temp.replace(path)
