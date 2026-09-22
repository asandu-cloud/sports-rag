"""Disposable statistical intermediates; prices and decisions are NEVER cached.

Only persisted Match Read generation opts in. A changed fixture/context,
local data, model/config/code revision, or a six-hour TTL forces recomputation.
Remote/unknown Chroma revisions and unserialisable contexts bypass reuse.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import time

from .prediction_metrics import count

ROOT = Path(__file__).resolve().parents[3]
_active = ContextVar("statistical_intermediates", default=None)


def _encode(value):
    if is_dataclass(value):
        return _encode(asdict(value))
    if isinstance(value, tuple):
        return {"__tuple__": [_encode(item) for item in value]}
    if isinstance(value, list):
        return [_encode(item) for item in value]
    if isinstance(value, dict):
        return {key: _encode(item) for key, item in value.items()}
    return value


def _decode(value):
    if isinstance(value, dict):
        if set(value) == {"__tuple__"}:
            return tuple(_decode(item) for item in value["__tuple__"])
        return {key: _decode(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_decode(item) for item in value]
    return value


def _json(value):
    return json.dumps(_encode(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


def data_revision():
    from .team_resolution import _profile_data_revision
    chroma = _profile_data_revision()
    if chroma is None:
        return None
    # stat fingerprints detect updates/replacements without reading vector
    # contents or large feature files. Include non-Chroma projection inputs.
    files = set((ROOT / "Scripts" / "rag_ingest").rglob("*.py"))
    files.update((ROOT / "Index").glob("*.json"))
    files.update((ROOT / "Index" / "ml_models").rglob("*"))
    files.update((ROOT / "Output").rglob("*.json"))
    for directory in ("Config", "config", "configs"):
        files.update((ROOT / directory).rglob("*"))
    revision = []
    for path in sorted(files):
        if path.is_file():
            stat = path.stat()
            revision.append((str(path), stat.st_ino, stat.st_mtime_ns, stat.st_size))
    # Environment values are hashed, never persisted/logged as plaintext.
    return hashlib.sha256(_json((chroma, revision, sorted(os.environ.items()),
                                datetime.now(timezone.utc).date().isoformat())).encode()).hexdigest()


class ProjectionStore:
    def __init__(self, path=None):
        self.path = Path(path) if path is not None else ROOT / "Index/cache/match_read_projections.sqlite3"

    @contextmanager
    def connection(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=0.2)
        try:
            connection.execute("CREATE TABLE IF NOT EXISTS projections (key TEXT PRIMARY KEY, revision TEXT, saved REAL, payload TEXT)")
            yield connection
            connection.commit()
        finally:
            connection.close()

    def get(self, key, revision):
        with self.connection() as connection:
            row = connection.execute("SELECT payload FROM projections WHERE key=? AND revision=? AND saved>=?",
                                     (key, revision, time.time() - 21600)).fetchone()
        return _decode(json.loads(row[0])) if row else {}

    def put(self, key, revision, values):
        payload = _json(values)
        with self.connection() as connection:
            connection.execute("INSERT OR REPLACE INTO projections VALUES (?,?,?,?)", (key, revision, time.time(), payload))
            connection.execute("DELETE FROM projections WHERE saved<?", (time.time() - 21600,))


@contextmanager
def statistical_reuse(identity, *, enabled=False, store=None, revision_provider=data_revision):
    if not enabled:
        yield
        return
    state = {"values": {}, "valid": True, "dirty": False}
    revision = key = None
    store = store or ProjectionStore()
    try:
        revision = revision_provider()
        key = hashlib.sha256(_json(identity).encode()).hexdigest()
        if revision is not None:
            values = store.get(key, revision)
            if isinstance(values, dict):
                state["values"] = values
    except (OSError, ValueError, TypeError, sqlite3.Error):
        revision = None
        count("statistical_cache.bypasses")
    token = _active.set(state)
    try:
        yield
        if revision is not None and state["valid"] and state["dirty"]:
            try:
                if revision_provider() == revision:
                    store.put(key, revision, state["values"])
            except (OSError, ValueError, TypeError, sqlite3.Error):
                count("statistical_cache.write_failures")
    finally:
        _active.reset(token)


def statistic(name, function, *args, **kwargs):
    state = _active.get()
    if state is not None and name in state["values"]:
        count("statistical_cache.hits")
        return deepcopy(state["values"][name])
    count("statistical_cache.misses")
    try:
        value = function(*args, **kwargs)
    except BaseException:
        invalidate()
        raise
    if state is not None:
        state["values"][name] = deepcopy(value)
        state["dirty"] = True
    return value


def invalidate():
    state = _active.get()
    if state is not None:
        state["valid"] = False
