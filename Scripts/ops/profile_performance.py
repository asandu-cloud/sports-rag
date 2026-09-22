"""Freeze local profile inputs and verify retrieval parity without live providers.

The capture uses a read-only SQLite transaction over Chroma's metadata (no
vectors or embedding calls). Replay uses the actual profile code with an
in-memory implementation of collection.get. Its counts measure retrieval
work; its elapsed time is NOT a benchmark of the production Chroma backend.
"""

import argparse
from contextlib import closing
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import socket
import sqlite3
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Scripts" / "rag_ingest"))
from core import team_resolution as resolver


class FrozenCollection:
    def __init__(self, records):
        self.records = records
        self.calls = 0

    @staticmethod
    def matches(meta, where):
        if "$and" in where:
            return all(FrozenCollection.matches(meta, part) for part in where["$and"])
        if "$or" in where:
            return any(FrozenCollection.matches(meta, part) for part in where["$or"])
        for key, value in where.items():
            if isinstance(value, dict):
                if set(value) - {"$in", "$eq"}:
                    raise ValueError(f"Unsupported frozen collection filter: {value}")
                if "$in" in value and meta.get(key) not in value["$in"]:
                    return False
                if "$eq" in value and meta.get(key) != value["$eq"]:
                    return False
            elif meta.get(key) != value:
                return False
        return True

    def get(self, *, where, include, limit=None, offset=0):
        self.calls += 1
        rows = [r for r in self.records if self.matches(r["meta"], where)]
        rows = rows[offset:offset + limit if limit is not None else None]
        result = {"ids": [r["id"] for r in rows]}
        if "documents" in include:
            result["documents"] = [r["text"] for r in rows]
        if "metadatas" in include:
            result["metadatas"] = [dict(r["meta"]) for r in rows]
        return result


def reset_caches():
    for name, value in vars(resolver).items():
        if name.endswith("_cache") and isinstance(value, dict):
            value.clear()


def replay(payload):
    reset_caches()
    collection = FrozenCollection(payload["records"])
    results = []
    with patch.object(resolver, "get_collection_handle", return_value=collection), \
         patch.object(socket.socket, "connect", side_effect=AssertionError("Offline replay attempted network access")):
        for case in payload["cases"]:
            before = collection.calls
            profile, audit = resolver.get_prediction_profile_context(
                case["team"], case["league"], target_date=case["date"],
            )
            recent = resolver.get_team_recent_stats(case["team"], case["league"], target_date=case["date"])
            results.append({"case": case, "profile": profile, "audit": audit, "recent": recent,
                            "collection_get_calls": collection.calls - before})
    return {"results": results, "collection_get_calls": collection.calls}


def capture_records(database, collection, leagues):
    marks = ",".join("?" for _ in leagues)
    sql = f"""
        SELECT e.id, e.embedding_id, m.key, m.string_value, m.int_value, m.float_value, m.bool_value
        FROM embeddings e JOIN embedding_metadata m ON m.id=e.id
        WHERE e.segment_id IN (
            SELECT s.id FROM segments s JOIN collections c ON c.id=s.collection
            WHERE c.name=? AND s.scope='METADATA'
        ) AND e.id IN (
            SELECT d.id FROM embedding_metadata d JOIN embedding_metadata l ON l.id=d.id
            WHERE d.key='doc_type' AND d.string_value IN ('team_profile','team_fixture')
              AND l.key='league' AND l.string_value IN ({marks})
        ) ORDER BY e.id, m.key
    """
    rows = {}
    with closing(sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)) as connection:
        connection.execute("BEGIN")
        for internal_id, doc_id, key, string, integer, real, boolean in connection.execute(sql, (collection, *leagues)):
            row = rows.setdefault(internal_id, {"id": doc_id, "meta": {}, "text": ""})
            value = next((v for v in (string, integer, real) if v is not None), None)
            if boolean is not None:
                value = bool(boolean)
            if key == "chroma:document":
                row["text"] = value or ""
            else:
                row["meta"][key] = value
    if not rows:
        raise ValueError("No profile inputs found for this collection/league selection")
    return list(rows.values())


def write_json(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    capture = sub.add_parser("capture")
    capture.add_argument("--directory", type=Path, required=True)
    capture.add_argument("--league", nargs="+", required=True)
    capture.add_argument("--case", action="append", required=True, help="LEAGUE:TEAM:YYYY-MM-DD")
    capture.add_argument("--database", type=Path, default=ROOT / "Index/chroma/chroma.sqlite3")
    capture.add_argument("--collection", default="football_top5")
    verify = sub.add_parser("verify")
    verify.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    directory = args.directory.resolve()
    if args.command == "capture":
        directory.mkdir(parents=True, exist_ok=False)
        cases = []
        for value in args.case:
            league, team, target_date = value.split(":", 2)
            cases.append({"league": league, "team": team, "date": target_date})
        payload = {"cases": cases, "records": capture_records(args.database, args.collection, args.league)}
        write_json(directory / "inputs.json", payload)
        (directory / "team_resolution.py").write_bytes(Path(resolver.__file__).read_bytes())
        baseline = replay(payload)
        write_json(directory / "baseline.json", baseline)
        write_json(directory / "manifest.json", {
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "boundary": "frozen_metadata_to_prediction_profile_and_recent_stats",
            "limitations": "Does not benchmark live Chroma or replay all market projections/providers.",
            "files": {name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
                      for name in ("inputs.json", "baseline.json", "team_resolution.py")},
        })
        print(json.dumps({"directory": str(directory), "cases": len(cases),
                          "records": len(payload["records"]), "collection_get_calls": baseline["collection_get_calls"]}))
    else:
        manifest = json.loads((directory / "manifest.json").read_text())
        for name, digest in manifest["files"].items():
            if hashlib.sha256((directory / name).read_bytes()).hexdigest() != digest:
                raise ValueError(f"Baseline checksum mismatch: {name}")
        baseline = json.loads((directory / "baseline.json").read_text())
        actual = replay(json.loads((directory / "inputs.json").read_text()))
        for before, after in zip(baseline["results"], actual["results"], strict=True):
            for key in ("case", "profile", "audit", "recent"):
                if before[key] != after[key]:
                    raise AssertionError(f"Parity failed for {before['case']}: {key}")
        print(json.dumps({"parity": "passed", "cases": len(actual["results"]),
                          "before_get_calls": baseline["collection_get_calls"],
                          "after_get_calls": actual["collection_get_calls"]}))


if __name__ == "__main__":
    main()
