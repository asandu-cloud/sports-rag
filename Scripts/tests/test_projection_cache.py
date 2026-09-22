"""Price refreshes reuse statistics, not selections, odds audits or provenance."""
from copy import deepcopy
import json
from pathlib import Path
import sqlite3
import sys
from unittest.mock import Mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rag_ingest"))

from core import market_service as markets
from core import projection_cache as cache


def test_all_market_outputs_match_cold_evaluation_after_prices_change(tmp_path, monkeypatch):
    scenario = json.loads((Path(__file__).parent / "fixtures/pipeline_v1/priced_standard/input.json").read_text())
    overrides = {
        **scenario["projection_overrides"],
        "projected_total_corners": (10.3, 10.1, 10.5),
        "projected_total_cards": (4.7, 4.5, 4.9, None),
        "projected_total_sot": (9.2, 9.0, 9.4),
        "projected_goal_difference": (0.8, 0.7, 0.9),
    }
    projections = []
    for name, values in overrides.items():
        function = Mock(return_value=tuple(values))
        monkeypatch.setattr(markets, name, function)
        projections.append(function)
    monkeypatch.setattr(markets, "_profile_quality", Mock(return_value={"status": "available", "home": {}, "away": {}}))
    monkeypatch.setattr(markets, "_total_variance", Mock(return_value=(None, {"source": "synthetic"})))
    monkeypatch.setattr(markets, "_apply_quality_guardrails", lambda decision, *_, **__: decision)
    store = cache.ProjectionStore(tmp_path / "cache.sqlite")
    monkeypatch.setattr(markets, "statistical_reuse", lambda identity, enabled: cache.statistical_reuse(
        identity, enabled=enabled, store=store, revision_provider=lambda: "frozen"))
    event = deepcopy(scenario["event"])
    kwargs = dict(fixture_date=scenario["fixture_date"], generated_at=scenario["generated_at"])
    first = markets.evaluate_event(event, "EPL", reuse_statistics=True, **kwargs)
    for bookmaker in event["bookmakers"]:
        for market in bookmaker["markets"]:
            for outcome in market["outcomes"]:
                outcome["price"] += 0.4
    warm = markets.evaluate_event(event, "EPL", reuse_statistics=True, **kwargs)
    assert all(function.call_count == 1 for function in projections)
    assert markets._profile_quality.call_count == 1
    cold = markets.evaluate_event(event, "EPL", **kwargs)
    assert warm == cold
    assert first[0].provenance.input_snapshot_id != warm[0].provenance.input_snapshot_id
    assert first[0].decision != warm[0].decision


@pytest.mark.parametrize("change", ["revision", "lineup", "fixture", "date", "stage", "model"])
def test_changed_inputs_invalidate_reuse(tmp_path, change):
    store = cache.ProjectionStore(tmp_path / "cache.sqlite")
    revision = ["revision-1"]
    identity = {"fixture": "123", "date": "2026-09-21", "lineup": ["player-1"], "stage": "pre_match", "model": "v1"}
    function = Mock(return_value=(3.0, None, {"value": 2.0}))
    for _ in range(2):
        with cache.statistical_reuse(identity, enabled=True, store=store, revision_provider=lambda: revision[0]):
            assert cache.statistic("goals", function) == (3.0, None, {"value": 2.0})
    assert function.call_count == 1
    if change == "revision":
        revision[0] = "revision-2"
    else:
        identity[change] = "new value"
    with cache.statistical_reuse(identity, enabled=True, store=store, revision_provider=lambda: revision[0]):
        cache.statistic("goals", function)
    assert function.call_count == 2


def test_expiry_corruption_and_unknown_revision_fall_back(tmp_path, monkeypatch):
    store = cache.ProjectionStore(tmp_path / "cache.sqlite")
    clock = [100000.0]
    monkeypatch.setattr(cache.time, "time", lambda: clock[0])
    function = Mock(return_value=3.0)
    def evaluate(revision="fixed"):
        with cache.statistical_reuse("fixture", enabled=True, store=store, revision_provider=lambda: revision):
            return cache.statistic("goals", function)
    assert evaluate() == 3.0
    clock[0] += 20000
    evaluate()
    assert function.call_count == 1
    clock[0] += 2000
    evaluate()
    assert function.call_count == 2  # cache hits did not extend the six-hour TTL
    with sqlite3.connect(store.path) as db:
        db.execute("UPDATE projections SET payload='invalid JSON'")
    assert evaluate() == 3.0
    evaluate(None)
    evaluate(None)
    assert function.call_count == 5


def test_interruption_or_revision_change_does_not_store_partial_work(tmp_path):
    store = cache.ProjectionStore(tmp_path / "cache.sqlite")
    function = Mock(return_value=3.0)
    with pytest.raises(KeyboardInterrupt):
        with cache.statistical_reuse("fixture", enabled=True, store=store, revision_provider=lambda: "r1"):
            cache.statistic("goals", function)
            raise KeyboardInterrupt()
    with store.connection() as db:
        assert db.execute("SELECT count(*) FROM projections").fetchone()[0] == 0
    versions = iter(["r1", "r2"])
    with cache.statistical_reuse("fixture", enabled=True, store=store, revision_provider=lambda: next(versions)):
        cache.statistic("goals", function)
    with store.connection() as db:
        assert db.execute("SELECT count(*) FROM projections").fetchone()[0] == 0


def test_cache_payload_never_contains_odds_or_decisions(tmp_path):
    store = cache.ProjectionStore(tmp_path / "cache.sqlite")
    with cache.statistical_reuse("fixture", enabled=True, store=store, revision_provider=lambda: "r1"):
        cache.statistic("goals", lambda: (3.1, 2.9, 3.3))
    with store.connection() as db:
        payload = json.loads(db.execute("SELECT payload FROM projections").fetchone()[0])
    assert set(payload) == {"goals"}
