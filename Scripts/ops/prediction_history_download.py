"""Collect older history into isolated staging; default is a network-free plan.

This wraps the existing bootstrap CLI, never imports into the live database.
Resume is per competition-season; an interrupted season is fetched again.
"""
from __future__ import annotations

import argparse
from contextlib import closing
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DIRECTORY = Path("Index/history_staging/eight-seasons-2026-09-25")
PAIRS = tuple((code, year) for code in (
    "EPL", "LaLiga", "SerieA", "Bundesliga", "Ligue1", "Championship",
    "SuperLig", "Eredivisie", "PrimeiraLiga", "BelgianProLeague", "UCL", "UEL",
) for year in (2019, 2020, 2021, 2022)) + (("UECL", 2021), ("UECL", 2022))
SOURCE_FILES = (
    "Scripts/ops/prediction_history_download.py", "Scripts/data_platform/cli.py",
    "Scripts/data_platform/config.py", "Scripts/data_platform/sync/pipeline.py",
    "Scripts/data_platform/sync/upserts.py", "Scripts/data_platform/sync/apifootball.py",
    "Scripts/data_platform/registry/competitions.yaml", "Scripts/football_http.py",
)


def _write(path, value):
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_hashes(root):
    return {name: _hash(root / name) for name in SOURCE_FILES}


def staging_directory(root, directory):
    root = root.resolve()
    directory = (root / directory).resolve()
    parent = (root / "Index/history_staging").resolve()
    if not parent.is_relative_to(root / "Index") or directory == parent or not directory.is_relative_to(parent):
        raise ValueError("Staging must be a dedicated child of Index/history_staging")
    live = root / "Index/platform.db"
    for name in ("platform.db", "raw_archive", "manifest.json", "collection.lock"):
        path = directory / name
        if path.is_symlink() or not path.resolve().is_relative_to(directory):
            raise ValueError("Staging paths must not redirect outside their directory")
        if path.is_file() and (path.stat().st_nlink > 1 or (live.exists() and path.samefile(live))):
            raise ValueError("Staging files must not alias existing database/source files")
    return directory


def existing_live_pairs(database):
    """Do not recollect a requested season already represented in live history."""
    with closing(sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)) as db:
        db.execute("PRAGMA query_only=ON")
        return {(code, year): count for code, year, count in db.execute(
            "SELECT c.code,s.year,count(*) FROM fixtures f "
            "JOIN competitions c ON c.id=f.competition_id JOIN seasons s ON s.id=f.season_id "
            "GROUP BY c.code,s.year") if (code, year) in PAIRS}


def plan(root, directory):
    directory = staging_directory(root, directory)
    existing = existing_live_pairs(root / "Index/platform.db")
    return {
        "schema": "isolated-history-collection.v1",
        "purpose": "Older FT fixtures and team statistics for offline Phase 3 review; no live import or promotion",
        "database": str(directory / "platform.db"),
        "raw_archive_root": str(directory / "raw_archive"),
        "pairs": [list(pair) for pair in PAIRS if pair not in existing],
        "existing_live_pairs_excluded": [dict(competition=c, season=y, fixtures=n)
                                          for (c, y), n in sorted(existing.items())],
        "source_sha256": _source_hashes(root),
        "python": sys.version,
        "policy": "team-statistics-only; retain provider nulls; archive raw responses; validate before canonical import",
        "limitations": "Successful collection is not certified coverage or regulation-target eligibility; retries refetch a whole season",
    }


def child_environment(directory):
    env = os.environ.copy()
    env.update(DATABASE_URL=f"sqlite:///{directory / 'platform.db'}",
               RAW_ARCHIVE_ROOT=str(directory / "raw_archive"), RAW_ARCHIVE_BACKEND="local")
    env.pop("BETTING_REFRESH_LOCK_FD", None)
    # Collection is isolated and must not send operational messages.
    env["REFRESH_ALERT_WEBHOOK_URL"] = ""
    return env


def verify_result(database, pair, summary):
    if not isinstance(summary, list) or len(summary) != 1:
        raise ValueError("Expected exactly one bootstrap result")
    result = summary[0]
    code, year = pair
    if (result.get("competition"), result.get("season")) != (code, year):
        raise ValueError("Bootstrap result scope mismatch")
    for key in ("run_id", "fixtures_seen"):
        if type(result.get(key)) is not int or result[key] <= 0:
            raise ValueError("Empty/invalid bootstrap result: " + key)
    if result.get("error_count") != 0 or result.get("errors") != []:
        raise ValueError("Bootstrap reported fixture errors; collection stopped")
    with closing(sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)) as db:
        db.execute("PRAGMA query_only=ON")
        db.execute("BEGIN")
        row = db.execute("SELECT run_kind,scope,status,stats FROM sync_runs WHERE id=?",
                         (result["run_id"],)).fetchone()
        if row is None or row[:3] != ("bootstrap", f"bootstrap:{code}:{year}", "completed"):
            raise ValueError("Bootstrap result has no matching completed staging run")
        stats = json.loads(row[3])
        for key in ("fixtures_seen", "fixture_detail_calls", "team_stats_rows", "player_stats_rows", "error_count", "errors"):
            if key not in result or stats.get(key) != result[key]:
                raise ValueError("Staging run/result mismatch: " + key)
        if stats["player_stats_rows"] != 0:
            raise ValueError("Unexpected player download")
        count, team_rows, paired = db.execute(
            "SELECT count(*),coalesce(sum(n),0),coalesce(sum(n=2),0) FROM "
            "(SELECT f.id,count(t.id) n FROM fixtures f JOIN competitions c ON c.id=f.competition_id "
            "JOIN seasons s ON s.id=f.season_id LEFT JOIN fixture_team_stats t ON t.fixture_id=f.id "
            "WHERE c.code=? AND s.year=? GROUP BY f.id)", (code, year)).fetchone()
        if count != result["fixtures_seen"]:
            raise ValueError("Staging fixture coverage differs from the saved run")
    return {"fixtures": count, "team_rows": team_rows, "fixtures_with_two_team_rows": paired,
            "fixtures_without_two_team_rows": count - paired,
            "statistical_completeness": "unverified", "regulation_target_eligibility": "unverified"}


def _saved_success(directory, pair):
    code, year = pair
    marker = directory / f"complete-{code}-{year}.json"
    if not marker.exists():
        return False
    if marker.is_symlink():
        raise ValueError("Saved completion must not be a symlink")
    complete = json.loads(marker.read_text())
    name = complete["attempt"]
    if Path(name).name != name:
        raise ValueError("Invalid saved attempt path")
    attempt_path = directory / name
    if attempt_path.is_symlink() or _hash(attempt_path) != complete["sha256"]:
        raise ValueError("Saved attempt checksum/path mismatch")
    attempt = json.loads(attempt_path.read_text())
    if attempt["returncode"] != 0:
        raise ValueError("Saved completion points to a failed process")
    coverage = verify_result(directory / "platform.db", pair, json.loads(attempt["stdout"]))
    if coverage != complete["coverage"]:
        raise ValueError("Staging coverage changed since collection completed")
    return True


def run(root, directory, *, execute=False, runner=None):
    root = root.resolve()
    prepared = plan(root, directory)
    if not execute:
        print(json.dumps({"dry_run": True, **prepared}, indent=2))
        return prepared
    if not prepared["pairs"]:
        print("Every requested season is already represented in the live database; no downloads started.")
        return prepared
    previous_mask = os.umask(0o077)
    try:
        return _collect(root, prepared, runner)
    finally:
        os.umask(previous_mask)


def _collect(root, prepared, runner):
    directory = Path(prepared["database"]).parent
    runner = runner or subprocess.run
    directory.mkdir(parents=True, exist_ok=True)
    manifest = directory / "manifest.json"
    if manifest.exists():
        if json.loads(manifest.read_text()) != prepared:
            raise ValueError("Frozen collection manifest differs; use a new staging directory after review")
    elif any(directory.iterdir()):
        raise ValueError("Refusing an existing staging directory without a collection manifest")
    else:
        _write(manifest, prepared)
    env = child_environment(directory)
    with (directory / "collection.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        def invoke(arguments, label):
            command = [sys.executable, "-m", "Scripts.data_platform.cli", *arguments]
            result = runner(command, cwd=root, env=env, stdout=subprocess.PIPE, text=True, check=False)
            path = directory / f"attempt-{label}-{uuid4().hex}.json"
            _write(path, {"command": command, "returncode": result.returncode, "stdout": result.stdout})
            if result.returncode:
                raise ValueError(f"Collection subprocess failed; retained report: {path}")
            return path, result.stdout
        invoke(["init-db"], "init")
        for pair in map(tuple, prepared["pairs"]):
            code, year = pair
            if _saved_success(directory, pair):
                print(f"Already collected and verified: {code}:{year}", flush=True)
                continue
            print(f"Collecting {code}:{year} into staging; no players", flush=True)
            path, output = invoke(["bootstrap", "--competition", code, "--season", str(year),
                                   "--no-players", "--archive"], f"{code}-{year}")
            coverage = verify_result(directory / "platform.db", pair, json.loads(output))
            _write(directory / f"complete-{code}-{year}.json",
                   {"attempt": path.name, "sha256": _hash(path), "coverage": coverage})
            print(f"Collected {code}:{year}: {json.dumps(coverage)}", flush=True)
    print(f"Collection finished in {directory}. Review coverage and raw evidence before canonical import.")
    return prepared


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=DEFAULT_DIRECTORY)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true", help="Perform the provider downloads into staging")
    mode.add_argument("--dry-run", action="store_true", help="Print the plan without writes or provider calls (default)")
    args = parser.parse_args(argv)
    try:
        run(ROOT, args.directory, execute=args.execute)
    except (ValueError, OSError, KeyError, sqlite3.Error, json.JSONDecodeError) as exc:
        print(f"History collection stopped: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
