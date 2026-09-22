from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
import socket

import pytest

from Scripts.rag_ingest.core.model_features import (
    FeaturePolicy, build_features, capture_snapshot, digest, inference_features, training_features,
)


COMPETITIONS = {"EPL": "domestic_league", "Championship": "domestic_league", "UCL": "continental_cup"}


def match(fid=1, *, kickoff="2026-08-20T12:00:00+00:00", season=2026, league="EPL", home=1, away=2, goals=2, cards=0):
    return {"fixture_id": fid, "competition": league, "season": season, "home_team_id": home, "away_team_id": away,
            "kickoff": kickoff, "status": "FT", "referee": "Official", "referee_observed_at": "2026-08-01T00:00:00Z",
            "observed_at": (datetime.fromisoformat(kickoff) + timedelta(hours=4)).isoformat(),
            "home": {"goals": goals, "corners": 0, "cards": cards, "sot": 4, "fouls": 9},
            "away": {"goals": 0, "corners": 4, "cards": 1, "sot": 2, "fouls": 8}}


def target(**kwargs):
    return match(999, kickoff="2026-09-10T12:00:00+00:00", **kwargs)


def features(history, fixture=None, **kwargs):
    fixture = fixture or target()
    return build_features(capture_snapshot(fixture, history, as_of=fixture["kickoff"], competitions=COMPETITIONS, **kwargs))


def values(result):
    return dict(zip(result["names"], result["values"]))


def test_target_and_future_results_cannot_change_features_or_snapshot():
    fixture = target()
    prior = match()
    baseline = features([prior, fixture], fixture)
    modified = deepcopy(fixture)
    modified["home"]["goals"] = 40
    later = match(2, kickoff="2026-10-20T12:00:00+00:00", goals=50)
    assert features([later, modified, prior], modified) == baseline
    assert values(baseline)["home_elo_goals"] == 1516
    assert values(baseline)["referee_cards_pm"] == 1


def test_training_and_inference_replay_identical_without_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Feature replay must not access network")
    monkeypatch.setattr(socket, "create_connection", forbidden)
    snapshot = capture_snapshot(target(), [match()], as_of=target()["kickoff"], competitions=COMPETITIONS)
    frozen = json.loads(json.dumps(snapshot))
    assert training_features(snapshot) == inference_features(frozen)
    assert frozen == snapshot


def test_missing_is_not_zero_and_production_rates_survive():
    missing = match(cards=None)
    absent = values(features([missing]))
    zero = values(features([match(cards=0)]))
    assert absent["home_cards_per_90_team"] is None
    assert absent["home_cards_per_90_team__missing"] == 1
    assert zero["home_cards_per_90_team"] == 0
    assert zero["home_cards_per_90_team__missing"] == 0
    assert zero["home_goals_for_pm"] == 2
    assert zero["home_sot_for_pm"] == 4
    assert zero["away_corners_pm"] == 4


def test_provider_season_does_not_reset_at_new_year():
    history = [match(kickoff="2025-12-20T12:00:00+00:00", season=2025)]
    fixture = match(999, kickoff="2026-01-10T12:00:00+00:00", season=2025)
    result = values(features(history, fixture))
    assert result["home_current_matches"] == 1
    assert result["home_prior_matches"] == 0


def test_prior_blend_and_season_elo_regression():
    history = [match(kickoff="2026-05-20T12:00:00+00:00", season=2025, goals=10), match(2, goals=2)]
    result = values(features(history))
    assert result["home_prior_matches"] == 1
    assert result["home_goals_for_pm"] == pytest.approx((2 + 8 * 10) / 9)
    prior_only = values(features(history[:1]))
    assert prior_only["home_elo_goals"] == 1512  # 1500 + .75 * 16
    assert prior_only["home_goals_for_pm"] == 10


def test_prior_is_removed_after_eight_current_matches_and_third_season_is_ignored():
    history = [match(kickoff="2024-05-20T12:00:00+00:00", season=2023, goals=99),
               match(2, kickoff="2026-05-20T12:00:00+00:00", season=2025, goals=10)]
    history += [match(i + 10, kickoff=f"2026-08-{10+i:02d}T12:00:00+00:00", goals=2) for i in range(8)]
    result = values(features(history))
    assert result["home_prior_weight"] == 0
    assert result["home_goals_for_pm"] == 2
    assert 1 not in features(history)["provenance"]["source_fixture_ids"]


def test_promoted_team_does_not_inherit_unadjusted_other_league_prior():
    old = match(kickoff="2026-05-20T12:00:00+00:00", season=2025, league="Championship", goals=10)
    result = values(features([old]))
    assert result["home_goals_for_pm"] is None
    assert result["home_elo_goals"] == 1500
    assert result["home_prior_matches"] == 0


def test_european_domestic_anchor_and_three_match_blend():
    domestic = match(goals=4)
    european = [match(i + 2, kickoff=f"2026-08-{21+i:02d}T12:00:00+00:00", league="UCL", goals=1) for i in range(3)]
    early = features([domestic, european[0]], target(league="UCL"))
    mature = features([domestic, *european], target(league="UCL"))
    assert early["profiles"]["home"]["goals_for_pm"] == 4
    assert early["profile_audit"]["home"]["mode"] == "domestic_anchor"
    assert mature["profiles"]["home"]["goals_for_pm"] == pytest.approx(.8 * 4 + .2)
    assert values(mature)["home_current_matches"] == 1
    assert values(mature)["home_continental_matches"] == 3


def test_ambiguous_domestic_membership_is_not_guessed():
    result = features([match(), match(2, league="Championship")], target(league="UCL"))
    assert result["profile_audit"]["home"]["mode"] == "ambiguous_domestic"
    assert result["profiles"]["home"]["goals_for_pm"] is None


def test_european_anchor_cannot_infer_current_membership_from_last_season():
    old = match(kickoff="2026-05-20T12:00:00+00:00", season=2025, league="Championship", goals=10)
    result = features([old], target(league="UCL"))
    assert result["profile_audit"]["home"]["mode"] == "unknown_current_domestic"
    assert result["profiles"]["home"]["goals_for_pm"] is None


def test_actual_rest_cross_competition_and_recent_windows():
    history = [match(i + 1, kickoff=f"2026-09-{i+1:02d}T12:00:00+00:00", goals=i) for i in range(6)]
    history.append(match(20, kickoff="2026-09-08T18:00:00+00:00", league="UCL", goals=99))
    result = values(features(history))
    assert result["home_goals_last_5"] == 3  # 1,2,3,4,5; not UCL's 99
    assert result["home_rest_days"] == 1.75


def test_observation_availability_and_completion_cutoffs():
    backfilled = match()
    backfilled["observed_at"] = "2026-09-15T00:00:00Z"
    assert values(features([backfilled]))["home_current_matches"] == 0
    retrospective = features([backfilled], availability="assumed_final")
    assert values(retrospective)["home_current_matches"] == 1
    assert retrospective["provenance"]["availability"] == "assumed_final"
    playing = match(kickoff="2026-09-10T10:00:00+00:00")
    assert values(features([playing], availability="assumed_final"))["home_current_matches"] == 0


def test_referee_assignment_after_cutoff_is_not_used():
    fixture = target()
    fixture["referee_observed_at"] = "2026-09-11T00:00:00Z"
    assert values(features([match()], fixture))["referee_cards_pm"] is None


@pytest.mark.parametrize("change", [{"fixture_id": None}, {"season": None}, {"kickoff": "2026-09-10"}, {"competition": "Unknown"}])
def test_ambiguous_identity_and_undated_rows_are_rejected(change):
    fixture = target()
    fixture.update(change)
    with pytest.raises(ValueError):
        features([match()], fixture)


def test_duplicate_history_and_tampered_snapshot_are_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        features([match(), match()])
    snapshot = capture_snapshot(target(), [match()], as_of=target()["kickoff"], competitions=COMPETITIONS)
    snapshot["history"][0]["home"]["goals"] = 40
    with pytest.raises(ValueError, match="digest"):
        build_features(snapshot)
    snapshot["history"][0]["kickoff"] = "2027-01-01T00:00:00+00:00"
    snapshot["snapshot_id"] = digest({k: v for k, v in snapshot.items() if k != "snapshot_id"})
    with pytest.raises(ValueError, match="ineligible"):
        build_features(snapshot)


def test_candidate_fit_and_prediction_share_schema_and_missingness():
    import numpy as np
    from Scripts.rag_ingest.core.candidate_features import candidate_matrix, candidate_predict

    snapshot = capture_snapshot(target(), [match()], as_of=target()["kickoff"], competitions=COMPETITIONS)
    matrix, contract = candidate_matrix([snapshot])
    assert np.isnan(matrix).any()  # No means learned from the entire dataset.

    class Candidate:
        def predict(self, x):
            np.testing.assert_equal(x, matrix)
            return [2.7]

    assert candidate_predict(Candidate(), snapshot, feature_contract=contract) == 2.7
    wrong = {**contract, "policy_id": "wrong"}
    with pytest.raises(ValueError, match="schema/policy"):
        candidate_predict(Candidate(), snapshot, feature_contract=wrong)


def test_candidate_rejects_invalid_predictions():
    from types import SimpleNamespace
    from Scripts.rag_ingest.core.candidate_features import candidate_matrix, candidate_predict

    snapshot = capture_snapshot(target(), [], as_of=target()["kickoff"], competitions=COMPETITIONS)
    _, contract = candidate_matrix([snapshot])
    for result in ([float("nan")], [], [1, 2]):
        with pytest.raises(ValueError, match="invalid prediction"):
            candidate_predict(SimpleNamespace(predict=lambda _x: result), snapshot, feature_contract=contract)
