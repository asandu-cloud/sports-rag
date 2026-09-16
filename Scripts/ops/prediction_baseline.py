"""Preserve and verify local prediction baselines without altering production.

This standalone tool copies source/data, takes SQLite online backups, and
replays the pure Match Read compiler from its archived source. It never runs
training, fetches providers, initializes Chroma, or publishes recommendations.
"""

from __future__ import annotations

import argparse
from contextlib import closing
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import shutil
import socket
import sqlite3
import subprocess
import sys
import tarfile
import time
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "prediction-baseline.v1"
COMPILER = "Scripts/data_platform/services/match_read_compiler.py"
ENV_NAMES = (
    "PREDICTION_RELEASE_MODE", "PLATFORM_ENABLED", "MATCH_READ_WORKER_MODE",
    "MATCH_READ_WORKER_LEAGUES", "MATCH_READ_OUTLOOK_HOURS",
    "MATCH_READ_FINAL_WINDOW_MINUTES", "MATCH_READ_REFRESH_MINUTES",
    "MATCH_READ_EARLY_REFRESH_MINUTES", "MATCH_READ_MATCHDAY_TIMEZONE",
)
LIMITATIONS = [
    "Compiler replay starts from recorded canonical market results: it does not "
    "recompute profiles, projections, probabilities, or the original market selectors.",
    "Historical Match Reads lack complete raw profile/quote candidate snapshots. "
    "Today's exported data must not be passed off as their historical inputs.",
    "Chroma vectors/index and the separate live-betting database are excluded. "
    "Output exports are preserved but are not asserted to equal Chroma's contents.",
    "Each SQLite backup is transactionally consistent independently. File exports "
    "and separate databases are not one cross-store transaction; capture times are recorded.",
    "Allowlisted configuration describes this shell/project .env, not an attestation "
    "of the environment already loaded by an existing bot or scheduler process.",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def identity(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def write_json(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), *args], text=True,
        env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"},
    ).strip()


def source_paths(root: Path) -> list[Path]:
    paths = []
    for directory in ("Scripts", "config"):
        for p in (root / directory).rglob("*"):
            if "__pycache__" in p.parts or p.is_symlink() or not p.is_file():
                continue
            if p.suffix in {".py", ".yaml", ".yml", ".js", ".css", ".html"}:
                paths.append(p)
            elif "fixtures" in p.parts and "tests" in p.parts and p.suffix == ".json":
                paths.append(p)
    for name in ("requirements.txt", "pytest.ini", "docs/ml-prediction-improvement-plan.md",
                 "docs/prediction-baseline.md"):
        p = root / name
        if p.is_file():
            paths.append(p)
    return sorted(set(paths))


def data_paths(root: Path) -> list[Path]:
    paths = [p for p in (root / "Output").rglob("*")
             if p.is_file() and not p.is_symlink() and p.suffix in {".json", ".jsonl", ".csv"}]
    paths.extend(p for p in (root / "Index").glob("*.json") if p.is_file())
    for directory in ("ml_models", "raw_archive", "shadow_runs"):
        paths.extend(p for p in (root / "Index" / directory).rglob("*")
                     if p.is_file() and not p.is_symlink()
                     and p.suffix in {".json", ".jsonl", ".joblib", ".gz"})
    return sorted(set(paths))


def archive(root: Path, paths: list[Path], destination: Path) -> list[dict]:
    """Hash the exact bytes archived; fail if a source changes during capture."""
    entries = []
    with tarfile.open(destination, "w:gz", compresslevel=1) as bundle:
        for p in paths:
            if p.is_symlink() or not p.resolve().is_relative_to(root.resolve()):
                raise ValueError(f"Archive source escapes repository: {p}")
            before = p.stat()
            content = p.read_bytes()
            name = p.relative_to(root).as_posix()
            info = tarfile.TarInfo(name)
            info.size = len(content)
            info.mode = 0o600
            info.mtime = before.st_mtime
            bundle.addfile(info, io.BytesIO(content))
            checksum = hashlib.sha256(content).hexdigest()
            if digest(p) != checksum:
                raise RuntimeError(f"Source changed during capture: {name}; retry when refresh is idle")
            entries.append({"path": name, "sha256": checksum, "bytes": len(content),
                            "source_mtime_ns": before.st_mtime_ns})
    return entries


def sqlite_backup(source: Path, target: Path) -> dict:
    if target.exists():
        raise FileExistsError(target)
    started = utc_now()
    deadline = time.monotonic() + 180

    def progress(_status: int, _remaining: int, _total: int) -> None:
        if time.monotonic() > deadline:
            raise TimeoutError("SQLite backup exceeded three minutes; retry when ingestion is idle")

    with closing(sqlite3.connect(source.resolve().as_uri() + "?mode=ro", uri=True)) as src:
        with closing(sqlite3.connect(target)) as dst:
            src.backup(dst, pages=512, progress=progress, sleep=0.05)
            if dst.execute("PRAGMA quick_check").fetchone() != ("ok",):
                raise RuntimeError(f"SQLite backup validation failed: {target.name}")
    return {"started_at": started, "completed_at": utc_now(), "consistency": "sqlite_online_backup"}


def load_compiler(source: bytes):
    """Load only the stdlib-only compiler; never import application startup."""
    name = "_preserved_match_read_compiler"
    spec = importlib.util.spec_from_loader(name, loader=None)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    exec(compile(source, COMPILER, "exec"), module.__dict__)
    return module.compile_match_read


def decision_view(draft) -> dict:
    return {
        "status": draft.status,
        "selections": [
            {"role": s["role"], "result_index": s["result_index"],
             "decision": draft.canonical_results[s["result_index"]]["decision"]}
            for s in draft.selections
        ],
        "alternative_candidates": draft.game_script.get("alternative_candidates", []),
        "tags": draft.game_script.get("tags", []),
    }


def capture_replay(db: Path, compiler_source: bytes) -> tuple[list[dict], dict]:
    compiler = load_compiler(compiler_source)
    cases, excluded, historical_differences = [], [], []
    with closing(sqlite3.connect(db.resolve().as_uri() + "?mode=ro", uri=True)) as conn:
        rows = conn.execute("SELECT id, read_json FROM match_reads ORDER BY id").fetchall()
        for read_id, raw in rows:
            payload = json.loads(raw)
            try:
                draft = compiler(payload["canonical_results"], stage=payload["stage"])
            except (ValueError, KeyError, TypeError) as exc:
                excluded.append({"read_id": read_id, "reason": str(exc)})
                continue
            expected = decision_view(draft)
            original = [{k: s[k] for k in ("role", "result_index")}
                        for s in payload.get("selections", [])]
            if payload["status"] != draft.status or original != list(draft.selections):
                historical_differences.append(read_id)
            cases.append({"read_id": read_id, "recorded_at": payload.get("evaluated_at"),
                          "stage": payload["stage"], "canonical_results": payload["canonical_results"],
                          "historical_status": payload["status"], "historical_selections": original,
                          "expected_baseline": expected})
        coverage = [dict(zip(("league", "season", "fixtures", "latest_completed_kickoff",
                             "latest_fetch"), row)) for row in conn.execute(
            "SELECT c.code,s.year,count(*),"
            "max(CASE WHEN f.status IN ('FT','AET','PEN') THEN f.kickoff_utc END),"
            "max(f.last_fetched_at) FROM fixtures f JOIN competitions c ON c.id=f.competition_id "
            "JOIN seasons s ON s.id=f.season_id GROUP BY c.code,s.year ORDER BY c.code,s.year"
        )]
        public = conn.execute(
            "SELECT count(*),count(outcome),count(clv) FROM predictions "
            "WHERE source LIKE 'canonical_published:%'"
        ).fetchone()
    return cases, {"coverage": coverage, "compiler_replay_cases": len(cases),
                   "excluded_replays": excluded, "historical_selection_differences": historical_differences,
                   "published_predictions": dict(zip(("total", "settled", "with_clv"), public))}


def configuration(root: Path) -> dict:
    # Never capture the .env file or the entire process environment.
    try:
        from dotenv import dotenv_values
        file_values = dotenv_values(root / ".env")
    except ImportError:
        file_values = {}
    values = {}
    for name in ENV_NAMES:
        value = os.environ.get(name, file_values.get(name))
        if value is not None:
            # These settings are simple flags, integers, league lists, or timezones.
            if not re.fullmatch(r"[A-Za-z0-9_, /+.-]{0,256}", value):
                raise ValueError(f"Unexpected value for allowlisted setting {name}; refusing capture")
            values[name] = value
    return {"explicit_values": values, "unset_values_use_archived_code_defaults": True}


def create(root: Path, label: str) -> Path:
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,79}", label):
        raise ValueError("Label must be 1–80 lowercase letters, numbers, underscores or hyphens")
    sources, data = source_paths(root), data_paths(root)
    databases = [root / "Index" / name for name in ("platform.db", "predictions.db")]
    if not all(p.is_file() for p in databases):
        raise FileNotFoundError("Both platform.db and predictions.db must exist")
    required = sum(p.stat().st_size for p in sources + data + databases) + 256 * 1024**2
    if shutil.disk_usage(root).free < required:
        raise OSError(f"Baseline needs at least {required / 1024**3:.2f} GiB free")
    target = root / "Index" / "prediction_baselines" / label
    target.parent.mkdir(parents=True, exist_ok=True)
    target.mkdir(mode=0o700)  # Refuse to overwrite any existing baseline.
    (target / "databases").mkdir()
    started = utc_now()
    initial_git = {"revision": git(root, "rev-parse", "HEAD"),
                   "status": git(root, "status", "--porcelain=v1", "--untracked-files=all")}
    print("Archiving current source, including uncommitted files...", flush=True)
    source_entries = archive(root, sources, target / "source.tar.gz")
    print("Archiving Output exports and prediction artifacts...", flush=True)
    data_entries = archive(root, data, target / "data.tar.gz")
    backups = {}
    for db in databases:
        print(f"Taking consistent backup of {db.name}...", flush=True)
        backups[db.name] = sqlite_backup(db, target / "databases" / db.name)
    # Catch concurrent edits/refreshes spanning the capture, not just one file read.
    for entry in source_entries + data_entries:
        if digest(root / entry["path"]) != entry["sha256"]:
            raise RuntimeError(f"Source changed while baseline was built: {entry['path']}")
    with tarfile.open(target / "source.tar.gz") as bundle:
        compiler_source = bundle.extractfile(COMPILER).read()
    print("Capturing offline compiler replay cases and database coverage...", flush=True)
    with patch.object(socket.socket, "connect", side_effect=RuntimeError("Baseline capture is offline")):
        cases, summary = capture_replay(target / "databases/platform.db", compiler_source)
    write_json(target / "replay.json", {"boundary": "canonical_results_to_match_read", "cases": cases})
    write_json(target / "data-summary.json", summary)
    source_map = {e["path"]: e["sha256"] for e in source_entries}
    data_map = {e["path"]: e["sha256"] for e in data_entries}
    components = {
        "features": identity({p: h for p, h in source_map.items()
                              if "feature" in p.lower() or p.endswith(("team_resolution.py", "ml_edge.py"))}),
        "fitted_models": identity({p: h for p, h in data_map.items() if p.startswith("Index/ml_models/")}),
        "probability_policy": identity({p: h for p, h in source_map.items()
                                        if p.endswith(("prob_models.py", "prediction_guardrails.py"))}),
        "selection_policy": identity({p: h for p, h in source_map.items()
                                      if p.endswith(("weights.py", "line_selection.py", "match_read_compiler.py",
                                                     "market_service.py", "release_control.py", "projections.py"))}),
    }
    settings = configuration(root)
    artifacts = {p.relative_to(target).as_posix(): {"sha256": digest(p), "bytes": p.stat().st_size}
                 for p in sorted(target.rglob("*")) if p.is_file()}
    manifest = {
        "schema_version": SCHEMA, "label": label, "started_at": started, "completed_at": utc_now(),
        "git": initial_git, "source_files": source_entries, "data_files": data_entries,
        "databases": backups, "artifacts": artifacts, "configuration": settings,
        "python": sys.version, "dependencies": sorted(
            ({"name": d.metadata["Name"], "version": d.version} for d in importlib.metadata.distributions()),
            key=lambda d: (d["name"] or "", d["version"])),
        "component_versions": components,
        "prediction_system_version": "baseline-" + identity(
            {"components": components, "source": source_map, "configuration": settings})[:24],
        "effective_model": {"mode": "recorded_artifacts_plus_archived_policy",
                            "r2": json.loads((root / "Index/ml_models/model_r2.json").read_text()),
                            "calibration": "fixed_distribution_and_confidence_rules_no_fitted_calibrator"},
        "forecast_cohorts": ["preliminary", "final_pre_match", "confirmed_lineups"],
        "limitations": LIMITATIONS,
    }
    write_json(target / "manifest.json", manifest)
    # Written last: partial failures have no completion marker.
    write_json(target / "COMPLETE.json", {"manifest_sha256": digest(target / "manifest.json")})
    return target


def verify(target: Path, *, replay: bool = True) -> dict:
    marker = json.loads((target / "COMPLETE.json").read_text())
    if digest(target / "manifest.json") != marker["manifest_sha256"]:
        raise ValueError("Baseline manifest checksum mismatch")
    manifest = json.loads((target / "manifest.json").read_text())
    if manifest["schema_version"] != SCHEMA:
        raise ValueError("Unsupported baseline schema")
    for name, expected in manifest["artifacts"].items():
        p = target / name
        if p.is_symlink() or not p.resolve().is_relative_to(target.resolve()):
            raise ValueError(f"Unsafe baseline artifact path: {name}")
        if digest(p) != expected["sha256"]:
            raise ValueError(f"Artifact checksum mismatch: {name}")
    count = 0
    if replay:
        with tarfile.open(target / "source.tar.gz") as bundle:
            compiler_source = bundle.extractfile(COMPILER).read()
        with patch.object(socket.socket, "connect", side_effect=RuntimeError("Replay is offline")):
            compiler = load_compiler(compiler_source)
            for case in json.loads((target / "replay.json").read_text())["cases"]:
                result = decision_view(compiler(case["canonical_results"], stage=case["stage"]))
                if result != case["expected_baseline"]:
                    raise AssertionError(f"Compiler replay differs for read {case['read_id']}")
                count += 1
    return {"label": manifest["label"], "verified_artifacts": len(manifest["artifacts"]),
            "compiler_cases_replayed": count, "prediction_system_version": manifest["prediction_system_version"],
            "boundary": "canonical_results_to_match_read", "full_projection_replay": False}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    capture = sub.add_parser("create", help="Create a new local baseline; refuses overwrite")
    capture.add_argument("--label", required=True)
    check = sub.add_parser("verify", help="Verify hashes and offline compiler replay using frozen source")
    check.add_argument("--baseline", type=Path, required=True)
    args = parser.parse_args()
    target = create(ROOT, args.label) if args.command == "create" else args.baseline.resolve()
    print(json.dumps(verify(target), indent=2))
    print(f"Baseline: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
