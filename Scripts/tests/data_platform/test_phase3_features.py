from copy import deepcopy
from datetime import datetime, timedelta, timezone

import pytest

from Scripts.data_platform.features.phase3_features import (
    HistoryIndex, apply_support, build_reference, feature_contract,
)
from Scripts.rag_ingest.core import model_features as core

COMPETITIONS = {"EPL": "domestic_league", "UCL": "continental_cup"}


def match(fid, day, *, season=2025, competition="EPL", goals=1):
    dt = datetime(2025, 8, 1, tzinfo=timezone.utc) + timedelta(days=day)
    stats = {"goals": goals, "corners": 0, "sot": 2, "cards": None}
    return {"fixture_id": fid, "competition": competition, "season": season,
            "home_team_id": 1, "away_team_id": 2, "kickoff": dt.isoformat(),
            "observed_at": (dt + timedelta(hours=5)).isoformat(),
            "status": "FT", "home": dict(stats), "away": dict(stats)}


def test_index_is_exact_reference_and_future_perturbations_do_not_change_features():
    history = [match(i, i) for i in range(1, 20)]
    target = match(999, 12)
    indexed = build_reference(target, HistoryIndex(history), COMPETITIONS)
    original = build_reference(target, history, COMPETITIONS)
    assert indexed == original
    changed = deepcopy(history)
    for row in changed[11:]:  # target/simultaneous/future results
        row["home"]["goals"] = 50
    changed.append(deepcopy(target))
    assert build_reference(target, HistoryIndex(changed), COMPETITIONS) == original
    assert build_reference(target, HistoryIndex(list(reversed(history))), COMPETITIONS) == original


def test_inactive_prior_observations_cannot_meet_support_gate():
    old = [match(i, -365 + i, season=2024) for i in range(1, 12)]
    current = [match(100 + i, i, goals=1 if i < 3 else None) for i in range(8)]
    _, features, row = build_reference(match(999, 20), old + current, COMPETITIONS)
    assert features["profile_audit"]["home"]["primary"]["prior_weight"] == 0
    assert row["support"]["goals"]["home"]["count"] == 3
    decisions = {market: {"eligible": True, "reasons": []} for market in ("goals", "corners", "sot")}
    checked = apply_support(decisions, row["support"])
    assert checked["goals"]["reasons"] == ["insufficient_home_history", "insufficient_away_history"]
    assert checked["corners"]["eligible"]


def test_prior_support_counts_actual_known_values_and_explicit_zero():
    old = [match(i, -365 + i, season=2024, goals=0 if i < 6 else None) for i in range(1, 9)]
    _, _, row = build_reference(match(999, 20), old, COMPETITIONS)
    assert row["support"]["goals"]["home"]["count"] == 5
    assert row["support"]["corners"]["home"]["count"] == 8


def test_card_mask_removes_elo_referee_rates_and_their_missing_flags():
    _, features, row = build_reference(match(999, 20), [match(1, 1)], COMPETITIONS)
    contract = feature_contract(features["names"])
    assert len(features["names"]) > len(contract["names"])
    assert all("card" not in name for name in contract["names"])
    assert "referee_fouls_pm" in contract["names"]
    assert "home_elo_goals" in contract["names"]
    values = dict(zip(contract["names"], row["values"]))
    assert values["home_corners_pm"] == 0
    assert values["home_corners_pm__missing"] == 0
    assert values["home_xg_pm"] is None
    assert values["home_xg_pm__missing"] == 1
    assert row["feature_contract_id"] == core.digest(contract)


def test_duplicate_history_fails_instead_of_reweighting_team():
    row = match(1, 1)
    with pytest.raises(ValueError, match="Duplicate"):
        HistoryIndex([row, row])


def test_index_normalizes_timezone_offsets_before_bisection():
    history = [match(i, 1) for i in range(1, 4)]
    for row, kickoff in zip(history, ["2025-08-02T01:00:00Z", "2025-08-02T10:00:00Z", "2025-08-02T11:00:00+08:00"]):
        row["kickoff"] = kickoff
    target = match(999, 1)
    target["kickoff"] = "2025-08-02T07:00:00Z"
    assert build_reference(target, HistoryIndex(history), COMPETITIONS) == build_reference(target, history, COMPETITIONS)


def test_continental_support_includes_only_nonzero_used_components():
    domestic = [match(i, i) for i in range(1, 4)]
    european = [match(100 + i, i, competition="UCL") for i in range(1, 4)]
    _, f, row = build_reference(match(999, 20, competition="UCL"), domestic + european, COMPETITIONS)
    assert f["profile_audit"]["home"]["mode"] == "domestic_continental_blend"
    assert row["support"]["goals"]["home"]["count"] == 6
    _, f, row = build_reference(match(999, 20, competition="UCL"), domestic + european[:2], COMPETITIONS)
    assert f["profile_audit"]["home"]["mode"] == "domestic_anchor"
    assert row["support"]["goals"]["home"]["count"] == 3
