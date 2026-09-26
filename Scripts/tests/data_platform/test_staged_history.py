from copy import deepcopy
from datetime import datetime, timezone

import pytest

from Scripts.data_platform.features.staged_history import normalize_team_statistics, regulation_issues


AS_OF = datetime(2026, 9, 26, tzinfo=timezone.utc)


def fixture():
    return {
        "fixture": {"id": 728649, "date": "2021-07-29T17:00:00+00:00",
                    "status": {"short": "FT", "elapsed": 90}},
        "teams": {"home": {"id": 10}, "away": {"id": 20}},
        "goals": {"home": 0, "away": 2},
        "score": {"fulltime": {"home": 0, "away": 2},
                  "extratime": {"home": None, "away": None},
                  "penalty": {"home": None, "away": None}},
    }


def blocks(entries=None):
    return [{"team": {"id": 10}, "statistics": [] if entries is None else entries},
            {"team": {"id": 20}, "statistics": []}]


def test_valid_regulation_fixture_is_unchanged_and_zero_score_is_valid():
    row = fixture()
    original = deepcopy(row)
    assert regulation_issues(row, as_of=AS_OF) == []
    assert row == original


def test_real_728649_period_conflict_is_not_silently_repaired():
    row = fixture()
    row["goals"]["away"] = 3
    assert regulation_issues(row, as_of=AS_OF) == ["goals_fulltime_mismatch"]
    assert row["goals"]["away"] == 3


@pytest.mark.parametrize("period", ["extratime", "penalty"])
@pytest.mark.parametrize("value", [0, 1])
def test_any_explicit_other_period_evidence_is_excluded(period, value):
    row = fixture()
    row["score"][period]["home"] = value
    assert f"nonnull_{period}_score" in regulation_issues(row, as_of=AS_OF)


@pytest.mark.parametrize("mutation,reason", [
    (lambda r: r["fixture"].update(id=True), "invalid_fixture_id"),
    (lambda r: r["teams"]["home"].update(id="10"), "invalid_fixture_team_ids"),
    (lambda r: r["teams"]["away"].update(id=10), "invalid_fixture_team_ids"),
    (lambda r: r["fixture"]["status"].update(short="AET"), "not_regulation_finished"),
    (lambda r: r["fixture"]["status"].update(elapsed=120), "invalid_or_extra_time_elapsed"),
    (lambda r: r["fixture"]["status"].update(elapsed=-1), "invalid_or_extra_time_elapsed"),
    (lambda r: r["fixture"].update(date="2021-01-01T12:00:00"), "invalid_or_future_kickoff"),
    (lambda r: r["fixture"].update(date="2027-01-01T12:00:00Z"), "invalid_or_future_kickoff"),
    (lambda r: r["fixture"].update(date=None), "invalid_or_future_kickoff"),
    (lambda r: r["goals"].update(home=True), "invalid_final_goals"),
    (lambda r: r["goals"].update(home=float("inf")), "invalid_final_goals"),
    (lambda r: r["goals"].update(home=-1), "invalid_final_goals"),
    (lambda r: r["goals"].update(home=0.5), "invalid_final_goals"),
    (lambda r: r["score"].pop("fulltime"), "missing_or_invalid_fulltime_score"),
    (lambda r: r["score"].update(extratime="invalid"), "invalid_extratime_score"),
])
def test_bad_regulation_evidence_has_explicit_reasons(mutation, reason):
    row = fixture()
    mutation(row)
    assert reason in regulation_issues(row, as_of=AS_OF)


def test_bad_payload_and_naive_replay_cutoff_are_not_accepted():
    assert regulation_issues(None, as_of=AS_OF) == ["invalid_fixture_payload"]
    with pytest.raises(ValueError, match="timezone-aware"):
        regulation_issues(fixture(), as_of=datetime(2026, 9, 26))


def test_statistics_preserve_zero_null_and_absent_types_without_mutation():
    raw = blocks([{"type": "Red Cards", "value": 0},
                  {"type": "Yellow Cards", "value": None},
                  {"type": "Corner Kicks", "value": "0"},
                  {"type": "Ball Possession", "value": "62%"},
                  {"type": "expected_goals", "value": "1.27"},
                  {"type": "goals_prevented", "value": -0.5}])
    original = deepcopy(raw)
    result = normalize_team_statistics(fixture(), raw)
    assert result == {10: {"red_cards": 0, "yellow_cards": None, "corners": 0,
                           "possession": 0.62, "expected_goals": 1.27, "goals_prevented": -0.5}, 20: {}}
    assert "shots_on" not in result[10]
    assert raw == original
    assert normalize_team_statistics(fixture(), []) == {}


@pytest.mark.parametrize("value", [True, "nan", float("inf"), "bad", [], -1, 0.5, "0%"])
def test_malformed_count_is_rejected_not_converted_to_null(value):
    with pytest.raises(ValueError, match="Invalid numeric statistic"):
        normalize_team_statistics(fixture(), blocks([{"type": "Red Cards", "value": value}]))


@pytest.mark.parametrize("key,value", [("Ball Possession", "101%"), ("Passes %", -0.1),
                                        ("expected_goals", -0.1), ("goals_prevented", "nan")])
def test_continuous_statistic_validation(key, value):
    with pytest.raises(ValueError, match="Invalid numeric statistic"):
        normalize_team_statistics(fixture(), blocks([{"type": key, "value": value}]))


@pytest.mark.parametrize("mutation", [
    lambda b: b.pop(),
    lambda b: b.append(deepcopy(b[0])),
    lambda b: b[1]["team"].update(id=10),
    lambda b: b[1]["team"].update(id=999),
    lambda b: b[1]["team"].update(id="20"),
    lambda b: b[1]["team"].update(id=False),
])
def test_exact_two_team_identity_required(mutation):
    raw = blocks()
    mutation(raw)
    with pytest.raises(ValueError, match="exactly the two fixture teams"):
        normalize_team_statistics(fixture(), raw)


@pytest.mark.parametrize("entries", [None, {}, [{"type": "Red Cards"}], [{"type": ""}],
                                      [{"type": "Red Cards", "value": 0}, {"type": "Red Cards", "value": None}]])
def test_statistics_structure_and_duplicate_types_rejected(entries):
    raw = blocks()
    raw[0]["statistics"] = entries
    with pytest.raises(ValueError):
        normalize_team_statistics(fixture(), raw)


@pytest.mark.parametrize("part,total", [("Shots on Goal", "Total Shots"), ("Shots off Goal", "Total Shots"),
                                       ("Blocked Shots", "Total Shots"), ("Passes accurate", "Total passes")])
def test_inconsistent_subtotals_rejected(part, total):
    with pytest.raises(ValueError, match="exceeds"):
        normalize_team_statistics(fixture(), blocks([{"type": part, "value": 3}, {"type": total, "value": 2}]))


def test_provider_block_order_does_not_assign_home_away_by_position():
    raw = blocks([{"type": "Corner Kicks", "value": 7}])
    assert normalize_team_statistics(fixture(), raw[::-1]) == {20: {}, 10: {"corners": 7}}
