"""Baseline capture must preserve data, reject overwrite, and replay frozen code."""

import json
from pathlib import Path
import sqlite3

import pytest

from ops import prediction_baseline as baseline


def _root(tmp_path, monkeypatch):
    root = tmp_path / "project"
    (root / "Index/ml_models").mkdir(parents=True)
    (root / "Index/ml_models/model_r2.json").write_text('{"goals": 0.0181}')
    (root / "Output").mkdir()
    (root / "Output/team.json").write_text('[{"team": "Example", "goals": 2}]')
    compiler = root / baseline.COMPILER
    compiler.parent.mkdir(parents=True)
    compiler.write_bytes((baseline.ROOT / baseline.COMPILER).read_bytes())
    (root / ".env").write_text("API_FOOTBALL_KEY=do-not-archive\nPREDICTION_RELEASE_MODE=live\n")
    monkeypatch.delenv("PREDICTION_RELEASE_MODE", raising=False)
    monkeypatch.setattr(baseline, "git", lambda *_args: "test-revision")
    result = {
        "fixture": {"event_id": "42", "league": "EPL", "home_team": "Home", "away_team": "Away"},
        "market": {"key": "totals", "group": "goals"},
        "projection": {"value": 3.0},
        "decision": {"status": "recommended", "confidence": "medium", "value_edge": 0.08,
                     "expected_value": 0.16, "model_probability": 0.58,
                     "quote": {"side": "over", "line": 2.5, "odds": 2.0, "market_key": "totals"}},
        "provenance": {"generated_at": "2026-09-15T10:00:00Z"},
    }
    payload = {"canonical_results": [result], "stage": "pre_match", "status": "recommended",
               "selections": [{"result_index": 0, "role": "core"}]}
    with sqlite3.connect(root / "Index/platform.db") as db:
        db.executescript("""
            CREATE TABLE match_reads (id INTEGER PRIMARY KEY, read_json TEXT);
            CREATE TABLE fixtures (competition_id INT, season_id INT, status TEXT,
                                   kickoff_utc TEXT, last_fetched_at TEXT);
            CREATE TABLE competitions (id INT, code TEXT);
            CREATE TABLE seasons (id INT, year INT);
            CREATE TABLE predictions (source TEXT, outcome TEXT, clv REAL);
        """)
        db.execute("INSERT INTO match_reads VALUES (1, ?)", (json.dumps(payload),))
    with sqlite3.connect(root / "Index/predictions.db") as db:
        db.execute("CREATE TABLE example (value INT)")
    return root


def test_capture_preserves_uncommitted_source_and_replays_without_working_tree(tmp_path, monkeypatch):
    root = _root(tmp_path, monkeypatch)
    before = baseline.digest(root / "Index/platform.db")
    target = baseline.create(root, "baseline-test")
    (root / baseline.COMPILER).write_text("raise AssertionError('must use archived compiler')")
    report = baseline.verify(target)
    assert report["compiler_cases_replayed"] == 1
    assert report["full_projection_replay"] is False
    assert baseline.digest(root / "Index/platform.db") == before
    manifest = json.loads((target / "manifest.json").read_text())
    assert manifest["configuration"]["explicit_values"]["PREDICTION_RELEASE_MODE"] == "live"
    assert "do-not-archive" not in (target / "manifest.json").read_text()
    assert ".env" not in {f["path"] for f in manifest["source_files"]}


def test_repeated_capture_cannot_overwrite_baseline(tmp_path, monkeypatch):
    root = _root(tmp_path, monkeypatch)
    target = baseline.create(root, "baseline-test")
    before = baseline.digest(target / "manifest.json")
    with pytest.raises(FileExistsError):
        baseline.create(root, "baseline-test")
    assert baseline.digest(target / "manifest.json") == before


def test_tampering_is_detected_before_loading_archived_code(tmp_path, monkeypatch):
    root = _root(tmp_path, monkeypatch)
    target = baseline.create(root, "baseline-test")
    (target / "source.tar.gz").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="Artifact checksum mismatch"):
        baseline.verify(target)


def test_online_backup_includes_committed_wal_rows_without_changing_source(tmp_path):
    source, target = tmp_path / "source.db", tmp_path / "backup.db"
    db = sqlite3.connect(source)
    try:
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("CREATE TABLE values_table (value INT)")
        db.execute("INSERT INTO values_table VALUES (42)")
        db.commit()
        baseline.sqlite_backup(source, target)
        with sqlite3.connect(target) as copied:
            assert copied.execute("SELECT value FROM values_table").fetchall() == [(42,)]
        assert db.execute("SELECT value FROM values_table").fetchall() == [(42,)]
    finally:
        db.close()


def test_capture_failure_does_not_mark_partial_archive_complete(tmp_path, monkeypatch):
    root = _root(tmp_path, monkeypatch)

    def fail(*_args):
        raise RuntimeError("simulated backup failure")

    monkeypatch.setattr(baseline, "sqlite_backup", fail)
    with pytest.raises(RuntimeError, match="simulated backup failure"):
        baseline.create(root, "partial")
    assert not (root / "Index/prediction_baselines/partial/COMPLETE.json").exists()
