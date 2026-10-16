"""SQLite document store for the engine's metadata lookups (rework A2a).

The prediction engine filters knowledge-base documents by metadata only
(document type, league, team, season, fixture); it never searches by meaning.
This store answers those ``collection.get(where=..., limit=..., include=...)``
calls from SQLite, so predictions need neither the 18 GB Chroma folder nor
OpenAI embeddings. Results come back in insertion order, as Chroma returns
them, which matters when a query hits its ``limit``.

Supported filters: implicit or explicit ``$and``, ``$or``; per field a value
or ``$eq``, ``$ne``, ``$in``, ``$nin``, ``$gt``, ``$gte``, ``$lt``, ``$lte``.
Anything else raises instead of silently matching.

Selected by ``DOCUMENT_STORE=sqlite`` (see ``Scripts/rag_ingest/core/team_resolution.py``).
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import sqlite3
import threading

logger = logging.getLogger(__name__)
DEFAULT_PATH = Path(__file__).resolve().parents[3] / "Index/documents.sqlite3"
NAME = "football_top5"
INDEXED = ("doc_type", "league", "team", "season", "fixture")
OPERATORS = {"$eq": "=", "$ne": "!=", "$gt": ">", "$gte": ">=", "$lt": "<", "$lte": "<="}


def _schema():
    generated = ",\n".join(f"    {c} TEXT GENERATED ALWAYS AS (json_extract(metadata, '$.{c}')) VIRTUAL"
                           for c in INDEXED)
    return (f"CREATE TABLE IF NOT EXISTS documents (\n    seq INTEGER PRIMARY KEY AUTOINCREMENT,\n"
            f"    id TEXT NOT NULL UNIQUE,\n    document TEXT,\n    metadata TEXT NOT NULL,\n{generated});\n"
            "CREATE INDEX IF NOT EXISTS documents_lookup ON documents(doc_type, league, team);\n"
            "CREATE INDEX IF NOT EXISTS documents_fixture ON documents(doc_type, league, season, fixture);\n")


def _field(key, params):
    if key in INDEXED:
        return key
    params.append(f'$."{key}"')
    return "json_extract(metadata, ?)"


def _condition(key, value, params):
    column = _field(key, params)
    if not isinstance(value, dict):
        params.append(value)
        return f"{column} = ?"
    if len(value) != 1:
        raise ValueError(f"one operator per field expected: {value!r}")
    (op, operand), = value.items()
    if op in OPERATORS:
        params.append(operand)
        return f"{column} {OPERATORS[op]} ?"
    if op in ("$in", "$nin"):
        operand = list(operand)
        if not operand:
            return "0" if op == "$in" else "1"
        params.extend(operand)
        return f"{column} {'IN' if op == '$in' else 'NOT IN'} ({', '.join('?' * len(operand))})"
    raise ValueError(f"unsupported filter operator {op!r}")


def _where(where, params):
    if not where:
        return "1"
    parts = []
    for key, value in where.items():
        if key in ("$and", "$or"):
            joined = f" {key[1:].upper()} ".join(f"({_where(item, params)})" for item in value)
            parts.append(f"({joined})" if value else ("1" if key == "$and" else "0"))
        elif key.startswith("$"):
            raise ValueError(f"unsupported filter operator {key!r}")
        else:
            parts.append(_condition(key, value, params))
    return " AND ".join(parts)


class DocumentStore:
    """Chroma-compatible ``get``/``upsert``/``count`` over SQLite."""

    name = NAME

    def __init__(self, path=DEFAULT_PATH):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        with self._connect() as db:
            db.executescript(_schema())

    def _connect(self):
        db = getattr(self._local, "db", None)
        if db is None:
            db = sqlite3.connect(self.path, timeout=60)
            db.execute("PRAGMA journal_mode=WAL")
            self._local.db = db
        return db

    def get(self, ids=None, where=None, limit=None, offset=None, include=("metadatas", "documents"), **_):
        params = []
        clause = _where(where, params)
        if ids is not None:
            ids = list(ids)
            clause = f"({clause}) AND id IN ({', '.join('?' * len(ids))})" if ids else "0"
            params.extend(ids)
        sql = f"SELECT id, document, metadata FROM documents WHERE {clause} ORDER BY seq"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(int(limit))
            if offset:
                sql += " OFFSET ?"
                params.append(int(offset))
        elif offset:
            sql += " LIMIT -1 OFFSET ?"
            params.append(int(offset))
        rows = self._connect().execute(sql, params).fetchall()
        include = list(include or [])
        return {"ids": [r[0] for r in rows],
                "documents": [r[1] for r in rows] if "documents" in include else None,
                "metadatas": [json.loads(r[2]) for r in rows] if "metadatas" in include else None,
                "embeddings": None, "included": include}

    def upsert(self, ids, documents=None, metadatas=None, **_):
        documents = documents or [None] * len(ids)
        metadatas = metadatas or [{}] * len(ids)
        db = self._connect()
        with db:
            db.executemany(
                "INSERT INTO documents (id, document, metadata) VALUES (?, ?, ?) "
                "ON CONFLICT(id) DO UPDATE SET document = excluded.document, metadata = excluded.metadata",
                [(i, d, json.dumps(m, ensure_ascii=False)) for i, d, m in zip(ids, documents, metadatas)])
        return len(ids)

    def delete(self, ids):
        ids = list(ids)
        db = self._connect()
        with db:
            db.executemany("DELETE FROM documents WHERE id = ?", [(i,) for i in ids])
        return len(ids)

    def count(self):
        return self._connect().execute("SELECT count(*) FROM documents").fetchone()[0]


def export_from_chroma(collection, store, *, batch=5000):
    """Copy every document (text and metadata, no embeddings) from a Chroma collection, keeping its order."""
    total, offset = collection.count(), 0
    while offset < total:
        page = collection.get(limit=batch, offset=offset, include=["documents", "metadatas"])
        if not page["ids"]:
            break
        store.upsert(ids=page["ids"], documents=page["documents"], metadatas=page["metadatas"])
        offset += len(page["ids"])
    return offset


IMMUTABLE_DOC_TYPES = {"team_fixture", "player_fixture"}  # as in rag_ingest/01_embed_and_upsert.py


def load_normalized_documents(store, index_dir, *, season=None, batch=1000):
    """Load ``normalized_*.json`` into the store with the incremental embed's rules.

    Completed-fixture documents already present are left unchanged; profiles
    (whose season-to-date figures move every gameweek) are rewritten; within a
    file the first copy of an id wins.
    """
    pattern = f"normalized_*_{season}_*.json" if season is not None else "normalized_*.json"
    paths = sorted(Path(index_dir).glob(pattern))
    existing = set(store.get(include=[])["ids"])
    counts = {"files": len(paths), "written": 0, "skipped": 0}
    for path in paths:
        seen, buffer = set(), []
        for doc in json.loads(path.read_text()):
            if doc["id"] in seen:
                continue
            seen.add(doc["id"])
            if doc["id"] in existing and (doc.get("metadata") or {}).get("doc_type") in IMMUTABLE_DOC_TYPES:
                counts["skipped"] += 1
                continue
            buffer.append(doc)
            if len(buffer) >= batch:
                counts["written"] += _write(store, buffer)
                buffer = []
        counts["written"] += _write(store, buffer)
    return counts


def _write(store, docs):
    if not docs:
        return 0
    return store.upsert(ids=[d["id"] for d in docs], documents=[d.get("text") for d in docs],
                        metadatas=[d.get("metadata") or {} for d in docs])


class DocumentStoreBackend:
    """``VectorBackend`` writing knowledge-base documents into the SQLite store (no embeddings)."""

    name = "sqlite"

    def __init__(self, store=None):
        self.store = store or DocumentStore(os.environ.get("DOCUMENT_STORE_PATH") or DEFAULT_PATH)

    def upsert(self, docs):
        from .base import VectorUpsertResult
        return VectorUpsertResult(upserted=_write(self.store, list(docs)))

    def delete(self, doc_ids):
        return self.store.delete(doc_ids)

    def query(self, text, *, limit=5, where=None):
        raise NotImplementedError("The document store serves metadata lookups, not semantic search.")


class MirroredBackend:
    """Write to the document store (required), then to Chroma on a best-effort basis.

    The prediction engine reads the store, so an OpenAI or Chroma failure must
    not fail the refresh: it is logged and the store stays current. Chroma only
    serves the semantic-search features (Whale chat, rag_cli_v2) until A7.
    """

    name = "sqlite+chroma"

    def __init__(self, primary, secondary):
        self.primary, self.secondary = primary, secondary

    def upsert(self, docs):
        docs = list(docs)
        result = self.primary.upsert(docs)
        try:
            self.secondary.upsert(docs)
        except Exception as exc:  # best effort by design
            logger.warning("Chroma mirror skipped %d documents: %s", len(docs), exc)
        return result

    def delete(self, doc_ids):
        doc_ids = list(doc_ids)
        removed = self.primary.delete(doc_ids)
        try:
            self.secondary.delete(doc_ids)
        except Exception as exc:
            logger.warning("Chroma mirror delete skipped: %s", exc)
        return removed

    def query(self, text, *, limit=5, where=None):
        return self.secondary.query(text, limit=limit, where=where)
