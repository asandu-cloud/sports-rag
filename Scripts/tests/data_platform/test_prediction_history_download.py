"""Offline checks for staging isolation and trustworthy collection completion."""
import json
from pathlib import Path
import sqlite3
from types import SimpleNamespace

import pytest

from Scripts.ops import prediction_history_download as download


def make_db(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as db:
        db.executescript("""
            CREATE TABLE IF NOT EXISTS competitions (id INTEGER PRIMARY KEY,code TEXT);
            CREATE TABLE IF NOT EXISTS seasons (id INTEGER PRIMARY KEY,year INTEGER);
            CREATE TABLE IF NOT EXISTS fixtures (id INTEGER PRIMARY KEY,competition_id INTEGER,season_id INTEGER);
            CREATE TABLE IF NOT EXISTS fixture_team_stats (id INTEGER PRIMARY KEY,fixture_id INTEGER);
            CREATE TABLE IF NOT EXISTS sync_runs (id INTEGER PRIMARY KEY,run_kind TEXT,scope TEXT,status TEXT,stats TEXT);
        """)


def completed_pair(database, *, errors=0, rows=2):
    make_db(database)
    stats = dict(fixtures_seen=1, fixture_detail_calls=1, team_stats_rows=rows,
                 player_stats_rows=0, error_count=errors,
                 errors=[] if not errors else [{"fixture_id": 1, "stage": "team_stats"}])
    with sqlite3.connect(database) as db:
        db.execute("INSERT OR REPLACE INTO competitions VALUES(1,'EPL')")
        db.execute("INSERT OR REPLACE INTO seasons VALUES(1,2019)")
        db.execute("INSERT OR REPLACE INTO fixtures VALUES(1,1,1)")
        for n in range(rows):
            db.execute("INSERT OR REPLACE INTO fixture_team_stats VALUES(?,1)", (n + 1,))
        db.execute("INSERT OR REPLACE INTO sync_runs VALUES(1,'bootstrap','bootstrap:EPL:2019','completed',?)",
                   (json.dumps(stats),))
    return [dict(competition="EPL", season=2019, run_id=1, **stats)]


@pytest.fixture
def root(tmp_path, monkeypatch):
    make_db(tmp_path / "Index/platform.db")
    monkeypatch.setattr(download, "_source_hashes", lambda root: {"test-source": "frozen"})
    return tmp_path


def test_exact_scope_has_fifty_unique_pairs():
    assert len(download.PAIRS) == len(set(download.PAIRS)) == 50
    assert {year for code, year in download.PAIRS if code == "UECL"} == {2021, 2022}
    assert all(2019 <= year <= 2022 for _, year in download.PAIRS)
    assert len({code for code, _ in download.PAIRS}) == 13


def test_default_plan_is_network_free_and_has_no_writes(root):
    before = (root / "Index/platform.db").read_bytes()
    def forbidden(*args, **kwargs):
        raise AssertionError("Dry run launched a subprocess")
    result = download.run(root, download.DEFAULT_DIRECTORY, runner=forbidden)
    assert len(result["pairs"]) == 50
    assert not (root / "Index/history_staging").exists()
    assert (root / "Index/platform.db").read_bytes() == before


def test_existing_live_seasons_are_not_requested(root):
    completed_pair(root / "Index/platform.db")
    result = download.plan(root, download.DEFAULT_DIRECTORY)
    assert ["EPL", 2019] not in result["pairs"]
    assert result["existing_live_pairs_excluded"] == [dict(competition="EPL", season=2019, fixtures=1)]


def test_no_requested_pairs_does_not_initialize_staging(root, monkeypatch):
    monkeypatch.setattr(download, "PAIRS", (("EPL", 2019),))
    completed_pair(root / "Index/platform.db")
    def forbidden(*args, **kwargs):
        raise AssertionError("No-op collection launched a subprocess")
    download.run(root, download.DEFAULT_DIRECTORY, execute=True, runner=forbidden)
    assert not (root / "Index/history_staging").exists()


def test_subprocess_routes_database_archives_and_gate_to_staging(root, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "sqlite:////live.db")
    monkeypatch.setenv("RAW_ARCHIVE_ROOT", "/live/archive")
    monkeypatch.setenv("RAW_ARCHIVE_BACKEND", "s3")
    monkeypatch.setenv("BETTING_REFRESH_LOCK_FD", "42")
    monkeypatch.setenv("REFRESH_ALERT_WEBHOOK_URL", "private-destination")
    directory = download.staging_directory(root, download.DEFAULT_DIRECTORY)
    env = download.child_environment(directory)
    assert env["DATABASE_URL"] == f"sqlite:///{directory / 'platform.db'}"
    assert env["RAW_ARCHIVE_ROOT"] == str(directory / "raw_archive")
    assert env["RAW_ARCHIVE_BACKEND"] == "local"
    assert "BETTING_REFRESH_LOCK_FD" not in env
    assert env["REFRESH_ALERT_WEBHOOK_URL"] == ""


@pytest.mark.parametrize("directory", [Path("Index"), Path("Index/platform.db"), Path("Output/new"), Path("Index/history_staging")])
def test_rejects_nonstaging_destinations(root, directory):
    with pytest.raises(ValueError):
        download.staging_directory(root, directory)


def test_rejects_database_symlink_to_live(root):
    directory = root / download.DEFAULT_DIRECTORY
    directory.mkdir(parents=True)
    (directory / "platform.db").symlink_to(root / "Index/platform.db")
    with pytest.raises(ValueError):
        download.staging_directory(root, directory)


def test_empty_statistics_are_reported_as_unverified_coverage(root):
    database = root / "empty-stats.db"
    summary = completed_pair(database, rows=0)
    result = download.verify_result(database, ("EPL", 2019), summary)
    assert result["fixtures_without_two_team_rows"] == 1
    assert result["statistical_completeness"] == "unverified"


@pytest.mark.parametrize("mutation", ["errors", "zero_fixtures", "scope", "db_run", "db_stats"])
def test_does_not_trust_exit_zero_or_unmatched_run(root, mutation):
    database = root / "attempt.db"
    summary = completed_pair(database)
    if mutation == "errors": summary[0]["error_count"] = 1
    if mutation == "zero_fixtures": summary[0]["fixtures_seen"] = 0
    if mutation == "scope": summary[0]["season"] = 2020
    if mutation in ("db_run", "db_stats"):
        with sqlite3.connect(database) as db:
            if mutation == "db_run": db.execute("UPDATE sync_runs SET status='running'")
            else: db.execute("UPDATE sync_runs SET stats='{}'")
    with pytest.raises(ValueError):
        download.verify_result(database, ("EPL", 2019), summary)


def test_execute_resume_and_failed_attempt_preservation(root, monkeypatch):
    monkeypatch.setattr(download, "PAIRS", (("EPL", 2019),))
    calls = []
    failed = True
    def runner(command, **kwargs):
        calls.append(command)
        database = Path(kwargs["env"]["DATABASE_URL"].removeprefix("sqlite:///"))
        assert database != root / "Index/platform.db"
        if command[-1] == "init-db":
            make_db(database)
            return SimpleNamespace(returncode=0, stdout="Schema upgraded")
        assert command[-2:] == ["--no-players", "--archive"]
        return SimpleNamespace(returncode=0, stdout=json.dumps(completed_pair(database, errors=int(failed))))
    with pytest.raises(ValueError, match="fixture errors"):
        download.run(root, download.DEFAULT_DIRECTORY, execute=True, runner=runner)
    directory = root / download.DEFAULT_DIRECTORY
    attempts = {p.name: p.read_bytes() for p in directory.glob("attempt-*.json")}
    assert (directory / "manifest.json").stat().st_mode & 0o777 == 0o600
    assert directory.stat().st_mode & 0o777 == 0o700
    assert not list(directory.glob("complete-*.json"))
    failed = False
    download.run(root, download.DEFAULT_DIRECTORY, execute=True, runner=runner)
    assert all((directory / name).read_bytes() == content for name, content in attempts.items())
    assert len(list(directory.glob("complete-*.json"))) == 1
    calls.clear()
    download.run(root, download.DEFAULT_DIRECTORY, execute=True, runner=runner)
    assert all(command[-1] == "init-db" for command in calls)
    marker = json.loads((directory / "complete-EPL-2019.json").read_text())
    (directory / marker["attempt"]).write_text("{}")
    with pytest.raises(ValueError, match="checksum"):
        download.run(root, download.DEFAULT_DIRECTORY, execute=True, runner=runner)


def test_changed_code_requires_new_manifest_and_unowned_directory_is_rejected(root, monkeypatch):
    directory = root / download.DEFAULT_DIRECTORY
    directory.mkdir(parents=True)
    (directory / "platform.db").write_text("existing work")
    with pytest.raises(ValueError, match="without a collection manifest"):
        download.run(root, download.DEFAULT_DIRECTORY, execute=True)
    prepared = download.plan(root, download.DEFAULT_DIRECTORY)
    prepared["source_sha256"] = {"test-source": "old"}
    (directory / "manifest.json").write_text(json.dumps(prepared))
    with pytest.raises(ValueError, match="manifest differs"):
        download.run(root, download.DEFAULT_DIRECTORY, execute=True)
