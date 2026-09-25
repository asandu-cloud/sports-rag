from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3

import pytest

from Scripts.data_platform.features.historical_recovery import apply_recovery_plan
from Scripts.data_platform.features.history_diagnostic import snapshot
from Scripts.data_platform.models import Competition
from Scripts.ops.prediction_card_repair import build_plan, apply_updates
from Scripts.ops.prediction_history import digest, write
from Scripts.tests.data_platform.test_historical_recovery import plan as history_plan


def evidence(settings, session_factory, tmp_path, *, reds=(0, 0)):
    database = Path(settings.database_url.removeprefix("sqlite:///"))
    with session_factory() as session:
        session.add(Competition(code="EPL", name="Premier League", api_football_id=39))
        session.flush()
        apply_recovery_plan(session, history_plan())
    target = tmp_path / "evidence"
    target.mkdir()
    request = {"endpoint": "/fixtures/statistics", "params": {"fixture": 123}}
    write(target / "local-snapshot.json", snapshot(database, [2025]))
    write(target / "local-audit.json", {})
    write(target / "sample.json", {})
    write(target / "request-plan.json", {"max_attempts": 120, "requests": [request]})
    write(target / "PREPARED.json", {p.name: digest(p) for p in target.glob("*.json")})
    provider = target / "provider"
    provider.mkdir()
    write(provider / "001.request.json", {"request": request})
    write(provider / "001.body", {"parameters": {"fixture": "123"}, "errors": [], "response": [
        {"team": {"id": tid}, "statistics": [{"type": "Yellow Cards", "value": 1},
                                              {"type": "Red Cards", "value": red}]}
        for tid, red in zip((1, 2), reds)]})
    write(provider / "001.receipt.json", {"success": True, "completed_at": datetime.now(timezone.utc).isoformat(),
                                          "body_sha256": digest(provider / "001.body")})
    return database, target


def connect(database):
    db = sqlite3.connect(database)
    db.row_factory = sqlite3.Row
    return db


def test_explicit_zero_repair_is_audited_and_idempotent(settings, session_factory, tmp_path):
    database, saved = evidence(settings, session_factory, tmp_path)
    prepared = build_plan(database, saved)
    assert len(prepared["updates"]) == 2
    with connect(database) as db:
        before = list(db.execute("SELECT updated_at FROM fixture_team_stats"))
        first = apply_updates(db, prepared)
        assert len(first["updated_fields"]) == 2
        second = apply_updates(db, prepared)
        assert len(second["already_applied"]) == 2
        assert second["updated_fields"] == []
        for row, old in zip(db.execute("SELECT * FROM fixture_team_stats"), before):
            assert row["red_cards"] == 0
            assert row["updated_at"] >= old[0]
            source = json.loads(row["stats_json"])
            assert source["red_cards"] == 0
            assert len(source["_card_stat_repairs"]) == 1
            assert source["_card_stat_repairs"][0]["body_sha256"]
    assert build_plan(database, saved)["updates"] == []


def test_null_provider_values_cannot_repair_missing_reds(settings, session_factory, tmp_path):
    database, saved = evidence(settings, session_factory, tmp_path, reds=(None, None))
    assert build_plan(database, saved)["updates"] == []


def test_conflict_midway_rolls_back_all_repairs(settings, session_factory, tmp_path):
    database, saved = evidence(settings, session_factory, tmp_path)
    prepared = build_plan(database, saved)
    with connect(database) as db:
        db.execute("UPDATE fixture_team_stats SET red_cards=2 WHERE is_home=0")
    with pytest.raises(ValueError, match="changed after review"):
        with connect(database) as db:
            apply_updates(db, prepared)
    with connect(database) as db:
        assert db.execute("SELECT red_cards FROM fixture_team_stats WHERE is_home=1").fetchone()[0] is None
        assert db.execute("SELECT red_cards FROM fixture_team_stats WHERE is_home=0").fetchone()[0] == 2


@pytest.mark.parametrize("mutation", ["plan", "body", "fixture", "sql_field", "duplicate"])
def test_tampered_or_stale_repair_cannot_write(settings, session_factory, tmp_path, mutation):
    database, saved = evidence(settings, session_factory, tmp_path)
    prepared = build_plan(database, saved)
    if mutation == "plan": prepared["updates"][0]["value"] = 4
    if mutation == "body": (saved / "provider/001.body").write_text("{}")
    if mutation == "fixture":
        with connect(database) as db:
            db.execute("UPDATE fixtures SET home_goals=5")
    if mutation == "sql_field": prepared["updates"][0]["field"] = "red_cards=9 --"
    if mutation == "duplicate": prepared["updates"].append(deepcopy(prepared["updates"][0]))
    with pytest.raises(ValueError):
        with connect(database) as db:
            apply_updates(db, prepared)
    with connect(database) as db:
        assert all(row[0] is None for row in db.execute("SELECT red_cards FROM fixture_team_stats"))
