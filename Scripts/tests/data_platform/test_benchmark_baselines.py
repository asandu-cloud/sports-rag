"""Numerical production-core parity and temporal isolation for comparators."""
from __future__ import annotations

import ast
import builtins
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import math
from pathlib import Path
import random
import socket

import pytest

from Scripts.data_platform.features.benchmarks import baselines as b
from Scripts.rag_ingest.core import model_features as core

ROOT = Path(__file__).resolve().parents[3]
COMPETITIONS = {"EPL": "domestic_league", "LaLiga": "domestic_league", "UCL": "continental_cup"}
BOUNDARY = "2024-01-01T00:00:00Z"


@pytest.fixture
def scoring():
    return b.baseline_metadata(ROOT)["scoring"]


@pytest.fixture(autouse=True)
def no_live_dependencies(monkeypatch):
    original = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if (name.startswith(("chromadb", "requests", "httpx"))
                or name in {"rag_cli_v2", "core.projections", "Scripts.rag_ingest.core.projections",
                            "core.team_resolution", "Scripts.rag_ingest.core.team_resolution"}):
            raise AssertionError(f"Live dependency requested: {name}")
        return original(name, *args, **kwargs)

    def blocked(*args, **kwargs):
        raise AssertionError("Comparator attempted network access")

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)


def fixture(fid, day=1, *, competition="EPL", season=2023, home=1, away=2, **updates):
    kickoff = datetime(season, 8, 1, 12, tzinfo=timezone.utc) + timedelta(days=day)
    row = {"fixture_id": fid, "kickoff": kickoff.isoformat(), "competition": competition,
           "season": season, "home_team_id": home, "away_team_id": away, "status": "FT",
           "observed_at": (kickoff + timedelta(hours=4)).isoformat(),
           "home": {"goals": 2.0, "corners": 6.0, "sot": 5.0, "xg": 1.7},
           "away": {"goals": 1.0, "corners": 4.0, "sot": 3.0, "xg": .9}}
    row.update(updates)
    return row


def request(target, history, *, availability="assumed_final"):
    # Only the fixture identity/context goes into the sidecar request; its own
    # result is deliberately absent, exactly as in frozen feature rows.
    identity = {k: v for k, v in target.items() if k not in {"home", "away"}}
    snapshot = core.capture_snapshot(identity, history, as_of=identity["kickoff"],
                                     competitions=COMPETITIONS, availability=availability)
    return {"fixture": identity, "as_of": snapshot["as_of"], "snapshot_id": snapshot["snapshot_id"],
            "feature_contract_id": "fixture-test-contract", "availability": availability,
            "partition": "development", "market_eligibility": {m: {"eligible": True} for m in b.MARKETS}}


def production_reference(scoring, profiles, recent):
    """Extract the real functions without executing production module imports.

    Provider, Chroma, ML and context resolution remain forbidden. Wrappers
    execute only with explicit in-memory profile/recent stubs and ML weight0.
    This catches arithmetic drift without testing a duplicate reference formula.
    """
    names = {"safe_float", "_venue_blend", "_stat_blend", "_trend_adjustment",
             "_blend_total_with_divergence", "projected_goals", "projected_corners", "projected_sot",
             "projected_total_goals", "projected_total_corners", "projected_total_sot"}
    tree = ast.parse((ROOT / "Scripts/rag_ingest/core/projections.py").read_text())
    definitions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in definitions} == names
    frozen = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
                             *definitions], type_ignores=[])
    ast.fix_missing_locations(frozen)
    namespace = {"math": math, "SCORING_WEIGHTS": scoring,
                 "_profile_as_of": lambda team, *args: profiles[team],
                 "_recent_stats": lambda team, *args, **kwargs: recent[team],
                 "_resolve_league_context": lambda *args, **kwargs: None,
                 "_ml_blend_weight": lambda stat: 0.0}
    exec(compile(frozen, "frozen-production-core-for-parity", "exec"), namespace)
    return namespace


@pytest.mark.parametrize("market", b.MARKETS)
def test_core_numerical_parity_over_missing_xg_venue_recent_and_divergence_cases(scoring, market):
    rng = random.Random(921)
    profile_fields = ("goals_for_pm", "goals_against_pm", "goals_home_pm", "goals_away_pm", "xg_home_pm", "xg_away_pm",
                      "corners_pm", "corners_against_pm", "corners_home_pm", "corners_away_pm",
                      "sot_for_pm", "sot_against_pm", "sot_home_pm", "sot_away_pm")
    recent_fields = ("xg_for_avg", "goals_slope", "corners_for_avg", "corners_against_avg", "corners_for_slope",
                     "sot_for_avg", "sot_against_avg", "sot_for_slope")
    for i in range(80):
        profiles = {side: {k: rng.choice((None, 0.0, rng.uniform(.1, 12))) for k in profile_fields}
                    for side in ("home", "away")}
        recent = {side: {k: rng.choice((None, 0.0, rng.uniform(.1, 12))) for k in recent_fields}
                  for side in ("home", "away")}
        scoring["recency"]["trend_weight"] = .03 if i % 2 else 0
        scoring["projection"]["home_advantage"] = .2 if i % 3 else 0
        reference = production_reference(scoring, profiles, recent)
        expected = reference[f"projected_total_{market}"]("home", "away", "EPL")
        actual = b.statistical_projection(profiles["home"], profiles["away"], recent["home"], recent["away"],
                                          market, scoring=scoring)
        for value, expected_value in zip((actual["value"], actual["season_total"], actual["recent_total"]), expected):
            if expected_value is None:
                assert value is None
            else:
                assert value == pytest.approx(expected_value, abs=1e-12)


def test_metadata_freezes_actual_config_sources_and_limitations(scoring):
    metadata = b.baseline_metadata(ROOT)
    assert metadata["statistical_version"] == "statistical-core-reconstruction.v1"
    assert metadata["scoring_sha256"] == core.digest(scoring)
    assert scoring["projection"]["xg_blend"] == .5  # Production docstring still says60%.
    assert metadata["ml_weight"] == 0
    assert len(metadata["source_sha256"]) == 6
    assert all(len(v) == 64 for v in metadata["source_sha256"].values())
    assert "knockout_context" in metadata["omitted_context"]


def test_six_recent_exponential_weights_preserve_missing_positions_and_xg_not_goals(scoring):
    history = [fixture(i, i) for i in range(1, 8)]
    for i, row in enumerate(history):
        row["home"]["corners"] = float(i)
        row["home"]["xg"] = float(i + 1)
        row["home"]["goals"] = 100.0
    history[-2]["home"]["corners"] = None
    snapshot = core.capture_snapshot(fixture(100, 10), history, as_of=fixture(100, 10)["kickoff"],
                                     competitions=COMPETITIONS, availability="assumed_final")
    inputs = b.reconstruct_snapshot(snapshot, scoring=scoring)
    recent = inputs["recent"]["home"]
    alpha = scoring["recency"]["alpha"]
    vals = [6.0, None, 4.0, 3.0, 2.0, 1.0]
    expected = sum(alpha ** i * v for i, v in enumerate(vals) if v is not None) / sum(
        alpha ** i for i, v in enumerate(vals) if v is not None)
    assert recent["corners_for_avg"] == pytest.approx(expected)
    assert recent["n"] == 6
    assert recent["xg_for_avg"] < 8
    assert inputs["evidence"]["home"]["recent_sources"]["EPL"] == [7, 6, 5, 4, 3, 2]


def test_current_prior_shrinkage_ignores_older_history_and_stops_after_eight_matches(scoring):
    old = fixture(1, season=2021)
    prior = fixture(2, season=2022)
    prior["home"]["goals"] = 8
    current = fixture(3, 2)
    current["home"]["goals"] = 1
    target = fixture(100, 40)
    snapshot = core.capture_snapshot(target, [old, prior, current], as_of=target["kickoff"],
                                     competitions=COMPETITIONS, availability="assumed_final")
    inputs = b.reconstruct_snapshot(snapshot, scoring=scoring)
    assert inputs["profiles"]["home"]["goals_for_pm"] == pytest.approx(1 / 9 + 8 * 8 / 9)
    assert inputs["evidence"]["home"]["primary"]["source_fixture_ids"] == [2, 3]
    for i in range(4, 11):
        row = fixture(i, i)
        row["home"]["goals"] = 1
        snapshot["history"].append(row)
    inputs = b.reconstruct_snapshot(snapshot, scoring=scoring)
    assert inputs["profiles"]["home"]["goals_for_pm"] == 1
    assert inputs["evidence"]["home"]["primary"]["prior_weight"] == 0


def test_european_profiles_and_recent_use_distinct_production_blending_gates(scoring):
    domestic = [fixture(i, i) for i in range(1, 7)]
    european = fixture(7, 7, competition="UCL")
    european["home"]["corners"] = 16
    target = fixture(100, 20, competition="UCL")
    snapshot = core.capture_snapshot(target, domestic + [european], as_of=target["kickoff"],
                                     competitions=COMPETITIONS, availability="assumed_final")
    inputs = b.reconstruct_snapshot(snapshot, scoring=scoring)
    assert inputs["evidence"]["home"]["mode"] == "domestic_anchor"
    assert inputs["profiles"]["home"]["corners_pm"] == 6
    assert inputs["recent"]["home"]["corners_for_avg"] == pytest.approx(.8 * 6 + .2 * 16)
    assert inputs["recent"]["home"]["n"] == 7
    for i in (8, 9):
        row = deepcopy(european)
        row["fixture_id"] = i
        snapshot["history"].append(row)
    inputs = b.reconstruct_snapshot(snapshot, scoring=scoring)
    assert inputs["evidence"]["home"]["mode"] == "domestic_continental_blend"
    assert inputs["profiles"]["home"]["corners_pm"] == pytest.approx(.8 * 6 + .2 * 16)


def test_ambiguous_or_previous_domestic_membership_never_resolved_by_name(scoring):
    target = fixture(100, 20, competition="UCL")
    history = [fixture(1, 1), fixture(2, 2, competition="LaLiga"), fixture(3, 3, competition="UCL")]
    snapshot = core.capture_snapshot(target, history, as_of=target["kickoff"], competitions=COMPETITIONS,
                                     availability="assumed_final")
    assert b.reconstruct_snapshot(snapshot, scoring=scoring)["evidence"]["home"]["mode"] == "ambiguous_domestic"
    snapshot["history"] = [fixture(1, season=2022), fixture(3, 3, competition="UCL")]
    assert b.reconstruct_snapshot(snapshot, scoring=scoring)["evidence"]["home"]["mode"] == "unknown_current_domestic"


def test_average_is_paired_league_venue_history_with_explicit_pooled_fallback(scoring):
    history = [fixture(i, i) for i in range(1, 6)]
    unrelated = fixture(6, 6, competition="LaLiga", home=10, away=11)
    unrelated["home"]["goals"] = 10
    history.append(unrelated)
    target = fixture(100, 20)
    row = request(target, history)
    rows = b.build_baseline_rows([row], history, COMPETITIONS, confirmation_start=BOUNDARY, scoring=scoring)
    goals = next(r for r in rows if r["market"] == "goals")
    assert goals["league_average"] == 3
    assert goals["evidence"]["league_average"]["scope"] == "competition"
    history[0]["home"]["goals"] = None
    row = request(target, history)
    rows = b.build_baseline_rows([row], history, COMPETITIONS, confirmation_start=BOUNDARY, scoring=scoring)
    goals = next(r for r in rows if r["market"] == "goals")
    assert goals["league_average"] == pytest.approx((4 * 3 + 11) / 5)
    assert goals["evidence"]["league_average"]["complete_fixtures"] == 5
    assert goals["evidence"]["league_average"]["scope"] == "pooled"


def test_future_results_and_exact_completion_boundary_cannot_change_earlier_baselines(scoring):
    history = [fixture(i, i) for i in range(1, 7)]
    target = fixture(100, 20)
    row = request(target, history)
    expected = b.build_baseline_rows([row], history, COMPETITIONS, confirmation_start=BOUNDARY, scoring=scoring)
    exact = fixture(200, 20)
    exact["kickoff"] = (core.utc(target["kickoff"]) - timedelta(hours=3)).isoformat()
    future = fixture(201, 25)
    locked = fixture(202, 1, season=2024)
    locked["home"] = object()  # Must be excluded before statistics are read.
    actual = b.build_baseline_rows([row], history + [exact, future, locked], COMPETITIONS,
                                   confirmation_start=BOUNDARY, scoring=scoring)
    assert actual == expected
    assert core.digest(actual) == core.digest(expected)


def test_missing_statistical_and_league_inputs_are_unavailable_but_zero_is_real(scoring):
    target = fixture(100, 20)
    empty = request(target, [])
    rows = b.build_baseline_rows([empty], [], COMPETITIONS, confirmation_start=BOUNDARY, scoring=scoring)
    assert all(r["statistical"] is None and r["league_average"] is None for r in rows)
    assert all(len(r["evidence"]["unavailable_reasons"]) == 2 for r in rows)
    prior = fixture(1)
    for side in ("home", "away"):
        prior[side] = {m: 0.0 for m in (*b.MARKETS, "xg")}
    rows = b.build_baseline_rows([request(target, [prior])], [prior], COMPETITIONS,
                                 confirmation_start=BOUNDARY, scoring=scoring)
    assert all(r["statistical"] == 0 and r["league_average"] == 0 for r in rows)


def test_builder_is_order_deterministic_and_does_not_mutate_inputs(scoring):
    history = [fixture(i, i) for i in range(1, 7)]
    requests = [request(fixture(100, 20), history), request(fixture(101, 21), history)]
    before = deepcopy((history, requests, scoring))
    first = b.build_baseline_rows(requests, history, COMPETITIONS, confirmation_start=BOUNDARY, scoring=scoring)
    second = b.build_baseline_rows(requests[::-1], history[::-1], COMPETITIONS, confirmation_start=BOUNDARY, scoring=scoring)
    assert first == second
    assert core.digest(first) == core.digest(second)
    assert (history, requests, scoring) == before
    assert len(first) == 6


def test_snapshot_hash_mismatch_fails_before_any_comparator_output(scoring):
    history = [fixture(1)]
    row = request(fixture(100, 20), history)
    history[0]["home"]["goals"] = 50
    with pytest.raises(ValueError, match="snapshot replay mismatch"):
        b.build_baseline_rows([row], history, COMPETITIONS, confirmation_start=BOUNDARY, scoring=scoring)


@pytest.mark.parametrize("partition", ["phase3_confirmation", "calibration", "final_system_test", "prospective_reserve"])
def test_heldout_partition_rejected_before_reading_history(scoring, partition):
    row = request(fixture(100, 20), [])
    row["partition"] = partition

    class ForbiddenHistory:
        def __iter__(self):
            raise AssertionError("History was read before rejecting held-out request")

    with pytest.raises(ValueError, match="held-out"):
        b.build_baseline_rows([row], ForbiddenHistory(), COMPETITIONS, confirmation_start=BOUNDARY, scoring=scoring)


def test_post_boundary_fixture_cannot_be_smuggled_as_development(scoring):
    row = request(fixture(100, season=2024), [])
    with pytest.raises(ValueError, match="held-out"):
        b.build_baseline_rows([row], [], COMPETITIONS, confirmation_start=BOUNDARY, scoring=scoring)


def test_market_eligibility_is_not_inferred_from_target_values(scoring):
    target = fixture(100, 20)
    row = request(target, [fixture(1)])
    row["market_eligibility"]["corners"] = {"eligible": False, "reasons": ["unverified_period"]}
    row["market_eligibility"]["cards"] = {"eligible": True}
    rows = b.build_baseline_rows([row], [fixture(1)], COMPETITIONS, confirmation_start=BOUNDARY, scoring=scoring)
    assert [r["market"] for r in rows] == ["goals", "sot"]


def test_mixed_availability_and_duplicate_forecast_identities_fail(scoring):
    row = request(fixture(100, 20), [])
    other = request(fixture(101, 20), [], availability="observed")
    with pytest.raises(ValueError, match="mix availability"):
        b.build_baseline_rows([row, other], [], COMPETITIONS, confirmation_start=BOUNDARY, scoring=scoring)
    with pytest.raises(ValueError, match="Duplicate baseline"):
        b.build_baseline_rows([row, row], [], COMPETITIONS, confirmation_start=BOUNDARY, scoring=scoring)


def test_observed_availability_honors_actual_later_observation(scoring):
    history = [fixture(1)]
    target = fixture(100, 20)
    history[0]["observed_at"] = (core.utc(target["kickoff"]) + timedelta(minutes=1)).isoformat()
    row = request(target, history, availability="observed")
    rows = b.build_baseline_rows([row], history, COMPETITIONS, confirmation_start=BOUNDARY, scoring=scoring)
    assert all(r["league_average"] is None and r["statistical"] is None for r in rows)
