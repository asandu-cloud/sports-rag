from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
from unittest.mock import Mock

import pytest
from sqlalchemy import select, func

from Scripts.data_platform.features.historical_recovery import build_recovery_plan, apply_recovery_plan
from Scripts.data_platform.models import Competition, Fixture, FixtureTeamStats, Team
from Scripts.ops import prediction_history as history_ops


def inputs():
    raw = {"fixture": {"id": 123, "date": "2025-08-15T19:00:00+00:00", "status": {"short": "FT"}},
           "league": {"id": 39, "season": 2025},
           "teams": {"home": {"id": 1, "name": "One"}, "away": {"id": 2, "name": "Two"}},
           "goals": {"home": 0, "away": 2}}
    rows = [{"fixture_id": 123, "team": name, "home_team": "One", "away_team": "Two",
             "fixture_date_utc": raw["fixture"]["date"], "home_goals": 0, "away_goals": 2,
             "Corner Kicks": 5, "Shots on Goal": 2, "Red Cards": 0, "Yellow Cards": 1,
             "Ball Possession": 0.5} for name in ("One", "Two")]
    return {"EPL": [raw]}, {"EPL": rows}


def plan(metadata=None, legacy=None):
    original, rows = inputs()
    return build_recovery_plan(original if metadata is None else metadata, rows if legacy is None else legacy,
                               season=2025, sources={"EPL": {"sha256": "test"}},
                               as_of=datetime(2026, 9, 22, tzinfo=timezone.utc))


def test_null_filled_zeros_are_unknown_but_scores_keep_true_zero():
    result = plan()
    entry = result["entries"][0]
    assert entry["stats"][0]["red_cards"] is None
    assert entry["stats"][0]["yellow_cards"] == 1
    assert entry["stats"][0]["possession"] == 0.5
    assert entry["api_row"]["goals"]["home"] == 0


@pytest.mark.parametrize("mutation", ["name", "score", "date", "duplicate", "side"])
def test_bad_legacy_pairs_keep_only_verified_metadata(mutation):
    meta, rows = inputs()
    if mutation == "name": rows["EPL"][0]["team"] = "Similar but different"
    if mutation == "score": rows["EPL"][0]["home_goals"] = 3
    if mutation == "date": rows["EPL"][0]["fixture_date_utc"] = "2025-08-16T19:00:00+00:00"
    if mutation == "duplicate": rows["EPL"].append(deepcopy(rows["EPL"][0]))
    if mutation == "side": rows["EPL"][0]["home_team"] = "Two"
    result = plan(meta, rows)
    assert result["entries"][0]["stats"] == [{}, {}]
    assert result["quarantined"][0]["scope"] == "statistics_only"


@pytest.mark.parametrize("mutation", ["league", "season", "status", "id", "future", "duplicate"])
def test_invalid_fixture_identity_is_not_importable(mutation):
    meta, rows = inputs()
    row = meta["EPL"][0]
    if mutation == "league": row["league"]["id"] = 140
    if mutation == "season": row["league"]["season"] = 2024
    if mutation == "status": row["fixture"]["status"]["short"] = "AET"
    if mutation == "id": row["teams"]["away"]["id"] = 1
    if mutation == "future": row["fixture"]["date"] = "2027-01-01T12:00:00+00:00"
    if mutation == "duplicate": meta["EPL"].append(deepcopy(row))
    assert plan(meta, rows)["entries"] == []


def test_absent_legacy_stats_are_not_invented():
    result = plan(legacy={})
    assert result["entries"][0]["stats"] == [{}, {}]
    assert result["coverage"]["EPL"]["metadata_only"] == 1


def test_invalid_numeric_stats_stay_unknown():
    meta, rows = inputs()
    rows["EPL"][0].update({"Yellow Cards": float("nan"), "Corner Kicks": -1,
                           "Red Cards": 0.5, "Ball Possession": 50, "Shots on Goal": True})
    stats = plan(meta, rows)["entries"][0]["stats"][0]
    assert all(stats[k] is None for k in ("yellow_cards", "red_cards", "corners", "possession", "shots_on"))


def test_import_is_additive_and_idempotent_without_renaming_current_team(session_factory):
    with session_factory() as session:
        session.add(Competition(code="EPL", name="Premier League", api_football_id=39))
        session.add(Team(api_football_id=1, name="Current team name"))
    with session_factory() as session:
        result = apply_recovery_plan(session, plan())
        assert result["counts"] == {"fixtures_inserted": 1, "team_rows_inserted": 2}
    with session_factory() as session:
        fixture = session.scalar(select(Fixture))
        assert fixture.home_goals == 0
        assert fixture.payload_digest is None  # Not falsely marked fully bootstrapped.
        assert session.scalar(select(Team).where(Team.api_football_id == 1)).name == "Current team name"
        assert session.scalar(select(FixtureTeamStats)).red_cards is None
        before = fixture.last_fetched_at
        result = apply_recovery_plan(session, plan())
        assert result["counts"] == {"existing_fixtures_preserved": 1}
        assert fixture.last_fetched_at == before
        assert session.scalar(select(func.count()).select_from(FixtureTeamStats)) == 2


def test_existing_conflicting_fixture_is_quarantined_without_overwrite(session_factory):
    with session_factory() as session:
        session.add(Competition(code="EPL", name="Premier League", api_football_id=39))
    with session_factory() as session:
        apply_recovery_plan(session, plan())
        fixture = session.scalar(select(Fixture))
        fixture.home_team_id, fixture.away_team_id = fixture.away_team_id, fixture.home_team_id
    with session_factory() as session:
        result = apply_recovery_plan(session, plan())
        assert len(result["quarantined"]) == 1
        assert "fixtures_inserted" not in result["counts"]


def test_preparation_frozen_evidence_roundtrip_and_tamper(tmp_path):
    meta, legacy = inputs()
    source = tmp_path / "Output/Prem_teams/team_fixture_stats_2025.json"
    source.parent.mkdir(parents=True)
    source.write_text(json.dumps(legacy["EPL"]))
    client = Mock()
    client.fixtures.return_value = meta["EPL"]
    target = history_ops.prepare(tmp_path, "recovery", 2025, ["EPL"], 1, client)
    result = history_ops.verified_plan(target)
    assert len(result["entries"]) == 1
    client.fixtures.assert_called_once_with(league=39, season=2025, status="FT")
    client.fixture_statistics.assert_not_called()
    with pytest.raises(FileExistsError):
        history_ops.prepare(tmp_path, "recovery", 2025, ["EPL"], 1, client)
    (target / "legacy-EPL.json").write_text("[]")
    with pytest.raises(ValueError, match="checksum"):
        history_ops.verified_plan(target)


def test_request_budget_fails_before_calls_or_files(tmp_path):
    client = Mock()
    with pytest.raises(ValueError, match="budget"):
        history_ops.prepare(tmp_path, "recovery", 2025, ["EPL", "UCL"], 1, client)
    client.fixtures.assert_not_called()
    assert not list(tmp_path.iterdir())


def test_card_audit_never_repairs_provider_null_from_legacy_zero(settings, session_factory):
    from Scripts.data_platform.storage.archive import PayloadArchiver, LocalDiskStorage
    path = Path(settings.database_url.removeprefix("sqlite:///"))
    raw = [{"team": {"id": tid}, "statistics": [{"type": "Yellow Cards", "value": 1},
                                               {"type": "Red Cards", "value": None}]} for tid in (1, 2)]
    with session_factory() as session:
        session.add(Competition(code="EPL", name="Premier League", api_football_id=39))
        session.flush()
        apply_recovery_plan(session, plan())
        PayloadArchiver(session, storage=LocalDiskStorage(path.parent / "raw_archive")).archive_json(
            provider="api_football", endpoint="/fixtures/statistics", params={"fixture": 123}, payload=raw)
    report = history_ops.audit_cards(path)
    assert report["supported_repair_candidates"] == []
    assert report["invalid_archives"] == []
    assert report["cohorts"]["EPL:2025"]["red_cards_archive_also_unknown"] == 2
    assert report["writes_performed"] is False


def test_numeric_card_archive_is_reported_not_blindly_applied(settings, session_factory):
    from Scripts.data_platform.storage.archive import PayloadArchiver, LocalDiskStorage
    path = Path(settings.database_url.removeprefix("sqlite:///"))
    raw = [{"team": {"id": tid}, "statistics": [{"type": "Red Cards", "value": 0}]} for tid in (1, 2)]
    with session_factory() as session:
        session.add(Competition(code="EPL", name="Premier League", api_football_id=39))
        session.flush()
        apply_recovery_plan(session, plan())
        PayloadArchiver(session, storage=LocalDiskStorage(path.parent / "raw_archive")).archive_json(
            provider="api_football", endpoint="/fixtures/statistics", params={"fixture": 123}, payload=raw)
    report = history_ops.audit_cards(path)
    assert len(report["supported_repair_candidates"]) == 2
    assert all(r["value"] == 0 for r in report["supported_repair_candidates"])
    with session_factory() as session:
        assert all(r.red_cards is None for r in session.scalars(select(FixtureTeamStats)))


def test_recovered_history_uses_current_observation_time_and_retains_unknowns(settings, session_factory):
    from Scripts.data_platform.features.model_dataset import load_canonical_inputs
    from Scripts.rag_ingest.core.model_features import capture_snapshot, FeaturePolicy
    path = Path(settings.database_url.removeprefix("sqlite:///"))
    with session_factory() as session:
        session.add(Competition(code="EPL", name="Premier League", api_football_id=39))
        session.flush()
        apply_recovery_plan(session, plan())
    fixtures, history, _ = load_canonical_inputs(path)
    assert history[0]["home"]["goals"] == 0
    assert history[0]["home"]["cards"] is None
    assert history[0]["home"]["corners"] == 5
    assert datetime.fromisoformat(history[0]["observed_at"]).year >= 2026
    target = {**fixtures[0], "fixture_id": 999, "kickoff": "2025-09-01T19:00:00+00:00"}
    snapshot = capture_snapshot(target, history, as_of=target["kickoff"], competitions={"EPL": "domestic_league"},
                                availability="observed", policy=FeaturePolicy())
    assert snapshot["history"] == []  # Not falsely backdated to the historical match.
