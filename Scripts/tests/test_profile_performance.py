"""Parity, query budget and invalidation checks for the profile retrieval path."""

import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rag_ingest"))
from core import market_service as markets
from core import prediction_metrics as metrics
from core import team_resolution as resolver
from Scripts.ops.profile_performance import FrozenCollection

local_data_revision = resolver._profile_data_revision


@pytest.fixture(autouse=True)
def isolated_profiles(monkeypatch):
    resolver.clear_profile_caches()
    monkeypatch.setattr(resolver, "_profile_data_revision", lambda: "test-revision")
    monkeypatch.setattr(resolver, "_cache_revision", "test-revision")
    yield
    resolver.clear_profile_caches()


def row(team="Arsenal", *, fixture="Arsenal vs Chelsea", day="2026-08-20", season="2026/27", cards=2):
    return {"id": f"{team}:{season}:{day}:{fixture}", "text": "", "meta": {
        "doc_type": "team_fixture", "league": "EPL", "season": season, "fixture": fixture,
        "fixture_date": day, "team": team, "opponent": "Chelsea" if team == "Arsenal" else "Arsenal",
        "home_away": "home" if team == "Arsenal" else "away", "final_score": "2-1",
        "cards_per_90_team": cards, "yellow_cards": cards, "red_cards": 0,
        "fouls_per_90_team": 10, "corners_for": 5, "sot_for": 4,
    }}


def test_batched_profiles_and_recent_form_equal_individual_lookup_results(monkeypatch):
    records = []
    for n in range(30):
        for team in ("Arsenal", "Chelsea"):
            records.append(row(team, fixture=f"match-{n}", day=f"2026-04-{n + 1:02}", season="2025/26", cards=n % 4))
    records += [row(), row("Chelsea"), row(day="2026-09-20", cards=999)]
    collection = FrozenCollection(records)
    monkeypatch.setattr(resolver, "get_collection_handle", lambda **kw: collection)

    def evaluate():
        return (resolver.get_team_profile_context("Arsenal", "EPL", "2026-09-19"),
                resolver.get_team_recent_stats("Arsenal", "EPL", target_date="2026-09-19"))

    batch = resolver._prefetch_opponent_fixture_meta
    monkeypatch.setattr(resolver, "_prefetch_opponent_fixture_meta", lambda *a, **kw: None)
    expected = copy.deepcopy(evaluate())
    old_calls = collection.calls
    resolver.clear_profile_caches()
    collection.calls = 0
    monkeypatch.setattr(resolver, "_prefetch_opponent_fixture_meta", batch)
    with metrics.collect_metrics() as report:
        actual = evaluate()
    assert actual == expected
    assert actual[0][1]["current_season_matches"] == 1
    assert actual[0][1]["prior_season_matches"] == 30
    assert collection.calls < old_calls / 2
    assert report.counts["opponent_lookup.cache_hits"] >= 31


def test_batch_paginates_and_keeps_exact_league_date_season_and_missing_values(monkeypatch):
    original = row()
    wanted = row("Chelsea", cards=None)
    records = [row(f"Noise-{i}") for i in range(260)]
    records += [row("Chelsea", season="2025/26", cards=888),
                row("Chelsea", day="2026-08-19", cards=777), wanted,
                row("Chelsea", cards=555)]  # First exact duplicate wins, as before.
    wrong_league = row("Chelsea", cards=999)
    wrong_league["meta"]["league"] = "UCL"
    records.insert(0, wrong_league)
    collection = FrozenCollection(records)
    monkeypatch.setattr(resolver, "get_collection_handle", lambda **kw: collection)
    resolver._prefetch_opponent_fixture_meta("EPL", [original["meta"]])
    assert collection.calls == 2
    assert resolver._get_team_fixture_meta("Chelsea", "EPL", original["meta"]["fixture"],
                                          "2026-08-20", "2026/27") == wanted["meta"]
    assert collection.calls == 2


def test_failed_batch_does_not_negative_cache_an_unread_opponent(monkeypatch):
    collection = FrozenCollection([row(), row("Chelsea")])
    original_get = collection.get

    def get(**kwargs):
        if "offset" in kwargs:
            raise RuntimeError("Backend batch failure")
        return original_get(**kwargs)

    monkeypatch.setattr(collection, "get", get)
    monkeypatch.setattr(resolver, "get_collection_handle", lambda **kw: collection)
    resolver._prefetch_opponent_fixture_meta("EPL", [row()["meta"]])
    assert not resolver._team_fixture_meta_cache
    assert resolver._get_team_fixture_meta("Chelsea", "EPL", "Arsenal vs Chelsea",
                                          "2026-08-20", "2026/27")["team"] == "Chelsea"


def test_revision_change_invalidates_profiles_rows_and_opponent_cache(monkeypatch):
    revision = ["first"]
    collection = FrozenCollection([row(), row("Chelsea", cards=1)])
    monkeypatch.setattr(resolver, "_profile_data_revision", lambda: revision[0])
    monkeypatch.setattr(resolver, "get_collection_handle", lambda **kw: collection)
    before = resolver.get_team_profile_context("Arsenal", "EPL", "2026-09-19")
    first_calls = collection.calls
    assert resolver.get_team_profile_context("Arsenal", "EPL", "2026-09-19") == before
    assert collection.calls == first_calls
    collection.records[1]["meta"]["cards_per_90_team"] = 4
    revision[0] = "second"
    after = resolver.get_team_profile_context("Arsenal", "EPL", "2026-09-19")
    assert after[0]["opp_cards_induced_pm"] == 4
    assert before[0]["opp_cards_induced_pm"] == 1
    assert collection.calls > first_calls


def test_remote_backend_reuses_only_within_one_evaluation_boundary(monkeypatch):
    monkeypatch.setattr(resolver, "_profile_data_revision", lambda: None)
    collection = FrozenCollection([row(), row("Chelsea")])
    monkeypatch.setattr(resolver, "get_collection_handle", lambda **kw: collection)

    @resolver.profile_cache_boundary
    def evaluate():
        resolver.get_team_profile_context("Arsenal", "EPL", "2026-09-19")
        calls = collection.calls
        resolver.get_team_profile_context("Arsenal", "EPL", "2026-09-19")
        assert collection.calls == calls

    evaluate()
    calls = collection.calls
    evaluate()
    assert collection.calls > calls


def test_local_revision_tracks_wal_and_database_replacement(monkeypatch, tmp_path):
    # The production detector must notice WAL commits, not only checkpointed
    # main-file writes. No Chroma client or production database is opened.
    monkeypatch.setattr(resolver, "CHROMA_DIR", str(tmp_path))
    monkeypatch.setattr(resolver, "env_first", lambda *a, default=None: default)
    database = tmp_path / "chroma.sqlite3"
    database.write_bytes(b"snapshot one")
    before = local_data_revision()
    wal = tmp_path / "chroma.sqlite3-wal"
    wal.write_bytes(b"new committed metadata")
    after_wal = local_data_revision()
    assert after_wal != before
    replacement = tmp_path / "replacement"
    replacement.write_bytes(b"snapshot two")
    replacement.replace(database)
    assert local_data_revision() != after_wal


def test_event_quality_is_computed_once_and_market_audits_are_independent(monkeypatch):
    from Scripts.tests.test_market_service import _event
    quality = {"profiles": {"home": {"current_season_matches": 8}}, "lineup": {"state": "expected"}}
    calls = []
    monkeypatch.setattr(markets, "_input_quality", lambda **kw: calls.append(kw) or copy.deepcopy(quality))
    monkeypatch.setattr(markets, "projected_total_goals", lambda *a, **kw: (3.2, 3.0, 3.4))
    monkeypatch.setattr(markets, "projected_total_corners", lambda *a, **kw: (9.5, 9.0, 10.0))
    monkeypatch.setattr(markets, "_total_variance", lambda h, a, l, market, *rest: (None, {"source": market}))
    monkeypatch.setattr(markets, "_apply_quality_guardrails", lambda decision, *a, **kw: decision)
    options = {"fixture_date": "2026-09-19", "generated_at": "2026-09-19T10:00:00Z"}
    expected = [markets.evaluate_market(_event(), "EPL", name, **options).to_dict() for name in ("goals", "corners")]
    assert len(calls) == 2
    calls.clear()
    actual = markets.evaluate_event(_event(), "EPL", markets=("goals", "corners"), **options)
    assert len(calls) == 1
    assert [r.to_dict() for r in actual] == expected
    actual[0].context["data_quality"]["profiles"]["home"]["current_season_matches"] = 0
    assert actual[1].context["data_quality"]["profiles"]["home"]["current_season_matches"] == 8
    assert quality["profiles"]["home"]["current_season_matches"] == 8


def test_empty_or_invalid_markets_do_not_load_quality(monkeypatch):
    monkeypatch.setattr(markets, "_input_quality", lambda **kw: pytest.fail("Unexpected quality lookup"))
    assert markets.evaluate_event({}, "EPL", markets=[]) == []
    with pytest.raises(ValueError):
        markets.evaluate_event({}, "EPL", markets=["unknown"])
