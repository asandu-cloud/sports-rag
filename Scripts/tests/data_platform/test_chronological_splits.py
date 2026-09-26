from copy import deepcopy
from datetime import datetime, timedelta, timezone

import pytest

from Scripts.data_platform.features.chronological_splits import (
    PARTITIONS, SPLIT_VERSION, assert_development_access, assign_partitions,
    choose_boundaries, label_available_at, training_rows,
)


def row(fixture_id=1, kickoff="2019-08-01T15:00:00Z", *, market="goals", nested=True, **changes):
    fixture = {"fixture_id": fixture_id, "kickoff": kickoff, "competition": "EPL", "season": 2019,
               "status": "FT"}
    value = {"fixture": fixture} if nested else dict(fixture)
    value.update(availability="assumed_final", market_eligibility={market: {"eligible": True}})
    value.update(changes)
    return value


def boundaries():
    return choose_boundaries([row(), row(999, "2026-09-26T15:00:00Z")])


def test_coverage_drives_explicit_twelve_six_twelve_calendar_reserves():
    result = boundaries()
    assert result["version"] == SPLIT_VERSION
    assert result["final_system_end"] == "2026-07-01T00:00:00+00:00"
    assert result["final_system_start"] == "2025-07-01T00:00:00+00:00"
    assert result["calibration_start"] == "2025-01-01T00:00:00+00:00"
    assert result["phase3_confirmation_start"] == "2024-01-01T00:00:00+00:00"
    assert result["development_start"] == "2022-01-01T00:00:00+00:00"
    assert len(result["development_folds"]) == 4
    assert result["development_folds"][2]["has_two_earlier_inner_blocks"] is True


def test_future_ineligible_or_uncompleted_records_do_not_move_endpoints():
    rows = [row(), row(999, "2026-09-26T15:00:00Z")]
    expected = choose_boundaries(rows)
    ineligible = row(1000, "2031-08-01T15:00:00Z", market_eligibility={})
    uncompleted = row(1001, "2032-08-01T15:00:00Z")
    uncompleted["fixture"]["status"] = "NS"
    actual = choose_boundaries(rows + [ineligible, uncompleted])
    assert actual == expected


def test_july_anchor_does_not_extend_beyond_latest_eligible_date():
    result = choose_boundaries([row(), row(999, "2026-06-30T23:59:59Z")])
    assert result["final_system_end"] == "2025-07-01T00:00:00+00:00"


@pytest.mark.parametrize("key,partition", [
    ("initial_training_start", "initial_training"), ("development_start", "development"),
    ("phase3_confirmation_start", "phase3_confirmation"), ("calibration_start", "calibration"),
    ("final_system_start", "final_system_test"), ("final_system_end", "prospective_reserve"),
])
def test_exact_boundaries_start_new_partition_with_half_open_intervals(key, partition):
    split = boundaries()
    actual = assign_partitions([row(7, split[key])], split)
    assert actual["partition_by_fixture"] == {"7": partition}


def test_fixture_versions_markets_and_simultaneous_matches_never_cross_partitions():
    rows = [row(8, "2024-01-01T00:00:00Z", revision="a"),
            row(8, "2024-01-01T02:00:00+02:00", market="sot", revision="b"),
            row(9, "2024-01-01T00:00:00Z", market="corners")]
    result = assign_partitions(rows, boundaries())
    assert result["partition_by_fixture"] == {"8": "phase3_confirmation", "9": "phase3_confirmation"}
    assert result["memberships"][0]["eligible_markets"] == ["goals", "sot"]
    assert result["memberships"][0]["row_count"] == 2
    assert not result["quarantined"]


def test_kickoff_revision_conflict_quarantines_all_versions_even_within_same_partition():
    rows = [row(8, "2024-02-01T15:00:00Z"), row(8, "2024-02-02T15:00:00Z")]
    result = assign_partitions(rows, boundaries())
    assert not result["memberships"]
    assert result["quarantined"][0]["reasons"] == ["conflicting_kickoff_revisions"]
    assert not training_rows(rows, "2025-01-01T00:00:00Z", availability="assumed_final")["rows"]


def test_conflicting_competition_or_season_is_not_silently_selected():
    rows = [row(), row()]
    rows[1]["fixture"].update(competition="UCL", season=2020)
    result = assign_partitions(rows, boundaries())
    assert result["quarantined"][0]["reasons"] == ["conflicting_fixture_identity"]


@pytest.mark.parametrize("side", ["home_team_id", "away_team_id"])
def test_conflicting_team_ids_quarantine_all_versions(side):
    examples = [row(), row()]
    examples[0]["fixture"].update(home_team_id=10, away_team_id=20)
    examples[1]["fixture"].update(home_team_id=10, away_team_id=20)
    examples[1]["fixture"][side] = 30
    result = assign_partitions(examples, boundaries())
    assert result["memberships"] == []
    assert result["quarantined"][0]["reasons"] == ["conflicting_fixture_team_identities"]


def test_reordering_and_repeated_runs_preserve_memberships_hashes_and_source():
    rows = [row(), row(3, "2025-01-01T15:00:00Z"), row(2, "2024-01-01T15:00:00Z"),
            row(2, "2024-01-01T15:00:00Z", market="sot")]
    original = deepcopy(rows)
    split = boundaries()
    result = assign_partitions(rows, split)
    assert result == assign_partitions(list(reversed(rows)), split)
    assert result == assign_partitions(rows, split)
    assert rows == original


def test_split_contract_never_reads_target_numbers_or_changes_membership_for_them():
    class TargetsForbidden(dict):
        def __getitem__(self, key):
            if key in {"targets", "target", "y"}:
                raise AssertionError("target accessed")
            return super().__getitem__(key)

        def get(self, key, default=None):
            if key in {"targets", "target", "y"}:
                raise AssertionError("target accessed")
            return super().get(key, default)

    original = row()
    poisoned = TargetsForbidden(dict(original, target=999999, targets={"goals": 999999}, y=-5))
    assert assign_partitions([original], boundaries()) == assign_partitions([poisoned], boundaries())
    assert training_rows([poisoned], "2020-01-01T00:00:00Z", availability="assumed_final")["rows"] == [poisoned]


def test_no_eligible_or_too_short_coverage_is_explicitly_insufficient():
    assert choose_boundaries([])["reasons"] == ["no_eligible_completed_fixtures"]
    only = row(1, "2026-08-01T15:00:00Z")
    result = choose_boundaries([only])
    assert result["status"] == "insufficient"
    assert "insufficient_initial_training_and_development_history" in result["reasons"]
    assert result["development_folds"] == []
    assert assign_partitions([only], result)["partition_by_fixture"]["1"] == "prospective_reserve"


def test_empty_eligibility_can_be_exported_without_fabricating_boundaries():
    examples = [row(market_eligibility={})]
    result = assign_partitions(examples, choose_boundaries(examples))
    assert result["status"] == "insufficient"
    assert result["reasons"] == ["no_eligible_completed_fixtures"]
    assert result["boundaries"] == {} and result["memberships"] == []
    assert result["unassigned_fixture_ids"] == [1]


def test_coverage_is_unique_per_fixture_and_has_empty_market_failures():
    rows = [row(1, "2024-01-01T15:00:00Z"), row(1, "2024-01-01T15:00:00Z", revision=2)]
    result = assign_partitions(rows, boundaries())
    total = next(s for s in result["coverage"]["slices"] if s["partition"] == "phase3_confirmation"
                 and s["market"] == "goals" and s["competition"] == "all")
    assert total["unique_fixtures"] == 1
    assert total["minimum_sample_gate"] is False
    empty = next(s for s in result["coverage"]["slices"] if s["partition"] == "phase3_confirmation"
                 and s["market"] == "sot" and s["competition"] == "all")
    assert empty["unique_fixtures"] == 0 and empty["minimum_sample_gate"] is False


def test_confirmation_sample_gate_requires_calendar_spread_not_only_row_count():
    examples = [row(i + 1, "2024-01-01T15:00:00Z") for i in range(1000)]
    def confirmation(rows):
        return next(item for item in assign_partitions(rows, boundaries())["coverage"]["slices"]
                    if item["partition"] == "phase3_confirmation" and item["market"] == "goals"
                    and item["competition"] == "all")
    assert confirmation(examples)["minimum_sample_gate"] is False
    for i, example in enumerate(examples):
        example["fixture"]["kickoff"] = (datetime(2024, 1, 1, 15, tzinfo=timezone.utc)
                                           + timedelta(days=i % 140)).isoformat()
    assert confirmation(examples)["minimum_sample_gate"] is True


def test_missing_league_market_partitions_and_present_seasons_have_explicit_zero_slices():
    examples = [row(1, "2024-01-01T15:00:00Z"), row(2)]
    examples[1]["fixture"]["competition"] = "UECL"
    slices = assign_partitions(examples, boundaries())["coverage"]["slices"]
    for competition in ("EPL", "UECL"):
        empty = next(item for item in slices if item["partition"] == "phase3_confirmation"
                     and item["competition"] == competition and item["market"] == "sot" and item["season"] == "all")
        assert empty["unique_fixtures"] == 0 and empty["minimum_slice_sample_gate"] is False
    season = next(item for item in slices if item["partition"] == "phase3_confirmation"
                  and item["competition"] == "EPL" and item["market"] == "sot" and item["season"] == "2019")
    assert season["unique_fixtures"] == 0 and season["minimum_slice_sample_gate"] is False


def test_fold_reports_count_only_strictly_earlier_declared_label_availability():
    rows = [row(1, "2021-12-31T20:00:00Z"), row(2, "2021-12-31T21:00:00Z"),
            row(3, "2022-01-01T12:00:00Z")]
    result = assign_partitions(rows, boundaries())
    counts = result["development_folds"][0]["markets"]["goals"]
    assert counts["training_unique_fixtures"] == 1
    assert counts["validation_unique_fixtures"] == 1
    assert counts["minimum_validation_count"] is False
    assert counts["weighted_ess_gate"] == "not_evaluated_until_recipe_fitting"


def test_mixed_availability_is_not_used_to_pass_development_support_gates():
    examples = [row(), row(2, availability="observed", observed_at="2019-08-02T00:00:00Z")]
    result = assign_partitions(examples, boundaries())
    counts = result["development_folds"][0]["markets"]["goals"]
    assert counts["training_unique_fixtures"] == 0
    assert counts["availability_issue"] == "mixed_availability_cohorts"


def test_assumed_label_time_is_strict_and_does_not_use_later_fetch_as_historical_vintage():
    example = row(observed_at="2026-09-26T00:00:00Z", label_available_at="2019-08-01T18:00:00Z")
    cutoff = datetime(2019, 8, 1, 18, tzinfo=timezone.utc)
    at = training_rows([example], cutoff, availability="assumed_final")
    after = training_rows([example], cutoff + timedelta(microseconds=1), availability="assumed_final")
    assert not at["rows"]
    assert at["excluded"][0]["reasons"] == ["label_available_after_or_at_fit"]
    assert after["rows"] == [example]


def test_observed_mode_requires_latest_actual_time_and_conservative_completion():
    example = row(availability="observed", observed_at="2019-08-01T17:00:00Z")
    assert label_available_at(example, availability="observed").hour == 18
    example["fixture"]["observed_at"] = "2026-09-26T00:00:00Z"
    assert label_available_at(example, availability="observed").year == 2026
    assert not training_rows([example], "2020-01-01T00:00:00Z", availability="observed")["rows"]


@pytest.mark.parametrize("changes,mode,reason", [
    ({"availability": "observed"}, "assumed_final", "availability_mode_mismatch"),
    ({"availability": "assumed_final"}, "observed", "availability_mode_mismatch"),
    ({"availability": "observed"}, "observed", "missing_observed_label_time"),
    ({"label_available_at": "2019-08-01T17:59:59Z"}, "assumed_final", "label_availability_contract_mismatch"),
])
def test_vintage_mixing_missing_observation_and_stale_contract_fail_closed(changes, mode, reason):
    result = training_rows([row(**changes)], "2027-01-01T00:00:00Z", availability=mode)
    assert not result["rows"]
    assert result["excluded"][0]["reasons"] == [reason]


def test_training_market_filter_retains_missing_values_and_real_zero_untouched():
    example = row(targets={"goals": 0, "corners": None})
    result = training_rows([example], "2020-01-01T00:00:00Z", availability="assumed_final", market="goals")
    assert result["rows"][0]["targets"] == {"goals": 0, "corners": None}
    assert not training_rows([example], "2020-01-01T00:00:00Z", availability="assumed_final", market="corners")["rows"]


def test_training_membership_hash_is_order_independent_for_same_fixture_markets():
    examples = [row(market="goals"), row(market="sot")]
    a = training_rows(examples, "2020-01-01T00:00:00Z", availability="assumed_final")
    b = training_rows(examples[::-1], "2020-01-01T00:00:00Z", availability="assumed_final")
    assert a == b


@pytest.mark.parametrize("partition", [*PARTITIONS[2:], "unknown", "", None])
def test_development_access_rejects_every_held_out_or_unknown_store(partition):
    with pytest.raises(PermissionError):
        assert_development_access(partition)


@pytest.mark.parametrize("partition", PARTITIONS[:2])
def test_development_access_allows_only_training_and_development(partition):
    assert_development_access(partition)


def test_flat_metadata_is_supported_and_invalid_identifiers_cannot_alias_integers():
    examples = [row(nested=False), row(True, nested=False), None, {"fixture": "bad"}]
    result = training_rows(examples, "2020-01-01T00:00:00Z", availability="assumed_final")
    assert result["rows"] == [examples[0]]
    assert len(result["excluded"]) == 3


def test_naive_timestamps_and_stale_split_versions_are_rejected():
    with pytest.raises(ValueError, match="timezone-aware"):
        training_rows([row()], datetime(2020, 1, 1), availability="assumed_final")
    invalid = row(kickoff="2019-08-01T15:00:00")
    result = assign_partitions([invalid], boundaries())
    assert result["quarantined"][0]["reasons"] == ["invalid_kickoff"]
    stale = dict(boundaries(), version="older-version")
    with pytest.raises(ValueError, match="version"):
        assign_partitions([row()], stale)


def test_boundary_hash_and_twelve_six_twelve_duration_are_verified():
    tampered = dict(boundaries(), calibration_start="2025-02-01T00:00:00+00:00")
    with pytest.raises(ValueError, match="hash mismatch"):
        assign_partitions([row()], tampered)
    tampered.pop("sha256")
    with pytest.raises(ValueError, match="twelve/six/twelve"):
        assign_partitions([row()], tampered)


def test_development_fold_cannot_extend_into_confirmation():
    tampered = deepcopy(boundaries())
    tampered.pop("sha256")
    tampered["development_folds"][-1]["validation_end"] = "2024-07-01T00:00:00+00:00"
    with pytest.raises(ValueError, match="development fold"):
        assign_partitions([row()], tampered)
