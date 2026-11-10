"""Fingerprints, worker-local read-only handles and non-destructive quotas."""
from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import shutil
import uuid

import h5py


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True,
                                        allow_nan=False), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


class ReadHandles:
    """Never share live HDF5 handles across fork/spawn workers; cap the LRU."""
    def __init__(self, limit=4, *, chunk_cache_bytes=2 * 1024**2):
        self.limit = max(1, int(limit))
        self.chunk_cache_bytes = int(chunk_cache_bytes)
        self.pid = os.getpid()
        self.handles = OrderedDict()
        self.datasets = {}

    def get(self, path):
        if self.pid != os.getpid():
            self.close()
            self.pid = os.getpid()
        key = str(Path(path).resolve())
        if key in self.handles:
            self.handles.move_to_end(key)
            return self.handles[key]
        handle = h5py.File(key, "r", rdcc_nbytes=self.chunk_cache_bytes)
        self.handles[key] = handle
        while len(self.handles) > self.limit:
            expired, old_handle = self.handles.popitem(last=False)
            self.datasets.pop(expired, None)
            old_handle.close()
        return handle

    def dataset(self, handle, name):
        """Keep Dataset IDs (and their decoded-chunk caches) until file eviction."""
        key = str(handle.filename)
        cached = self.datasets.setdefault(key, {})
        if name not in cached:
            cached[name] = handle[name]
        return cached[name]

    def close(self):
        self.datasets.clear()
        for handle in self.handles.values():
            handle.close()
        self.handles.clear()

    def __getstate__(self):
        return {"limit": self.limit, "chunk_cache_bytes": self.chunk_cache_bytes}

    def __setstate__(self, state):
        self.__init__(state["limit"], chunk_cache_bytes=state.get("chunk_cache_bytes", 2 * 1024**2))

    def __del__(self):
        self.close()


class QuotaExceeded(RuntimeError):
    pass


class Quota:
    """Decimal GB budgets shared by views, staging, metadata and native.

    Reservations use uncompressed upper bounds, so quota refusal can be
    conservative. Existing artifacts are never deleted to make room.
    """
    DEFAULT_GB = {"native": 16, "imu": 4, "ppg": 3, "metadata": .5,
                  "staging": 1, "experts": .5, "total": 25}

    def __init__(self, native_root, view_root, limits_gb=None):
        self.native_root, self.root = Path(native_root), Path(view_root)
        if set(limits_gb or {}) - set(self.DEFAULT_GB):
            raise ValueError("Unknown quota category")
        self.limits = {key: int(value * 1e9) for key, value in
                       (self.DEFAULT_GB | (limits_gb or {})).items()}
        if any(value < 0 for value in self.limits.values()):
            raise ValueError("Quota limits must be nonnegative")
        self._sizes = None

    @staticmethod
    def size(path):
        return sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) if path.exists() else 0

    def usage(self, *, refresh=False):
        if self._sizes is not None and not refresh:
            return self._sizes.copy()
        sizes = {key: self.size(self.root / key) for key in ("imu", "ppg", "experts", "staging")}
        sizes["native"] = self.size(self.native_root / "native")
        sizes["metadata"] = (self.size(self.root) - sum(sizes[k] for k in ("imu", "ppg", "experts", "staging"))
                             + self.size(self.native_root) - sizes["native"])
        self._sizes = sizes
        return sizes.copy()

    def account_metadata(self, old_bytes, new_bytes):
        if self._sizes is None: self.usage()
        self._sizes["metadata"] += int(new_bytes) - int(old_bytes)

    def check(self, category, reserve=0, *, staging=0):
        sizes = self.usage()
        for key, used in sizes.items():
            needed = used + (reserve if key == category else 0) + (staging if key == "staging" else 0)
            if needed > self.limits[key]:
                raise QuotaExceeded(f"{key} quota exceeded ({needed} > {self.limits[key]}); existing artifacts retained")
        if sum(sizes.values()) + reserve + staging > self.limits["total"]:
            raise QuotaExceeded("total quota exceeded; existing artifacts retained")
        anchor = self.root if self.root.exists() else self.root.parent
        if shutil.disk_usage(anchor).free < reserve + staging + 1024**2:
            raise QuotaExceeded("Insufficient disk space; existing artifacts retained")

    @contextmanager
    def artifact(self, category, name, reserve):
        self.check(category, reserve, staging=reserve)
        directory = self.root / category
        directory.mkdir(parents=True, exist_ok=True)
        staging_dir = self.root / "staging"
        staging_dir.mkdir(parents=True, exist_ok=True)
        # Exclusive lock also prevents concurrent reservations bypassing quotas.
        lock = self.root / ".view-write.lock"
        try:
            descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as exc:
            raise RuntimeError(f"View writer already active: {lock}") from exc
        temporary = staging_dir / (uuid.uuid4().hex + ".h5")
        try:
            self.check(category, reserve, staging=reserve)
            yield temporary
            actual = temporary.stat().st_size
            self.check(category, actual)
            destination = directory / name
            if destination.exists():
                raise FileExistsError(f"Refusing to replace existing immutable view: {destination}")
            os.replace(temporary, destination)
            self._sizes[category] += actual
        finally:
            os.close(descriptor)
            lock.unlink(missing_ok=True)
            temporary.unlink(missing_ok=True)
