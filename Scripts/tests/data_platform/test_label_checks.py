"""A5.3 checks (D44): a label passes only if its high picks keep more value at the close than its low picks."""
from datetime import datetime, timedelta, timezone

import pytest

from data_platform import label_checks as lc

START = datetime(2026, 10, 9, 15, tzinfo=timezone.utc)


def pick(fid, day, *, spread=0.01, value=0.0, y=1.0, p=0.5, market="goals", spix=0.0, books=5):
    return {"fixture_id": str(fid), "stage": "pre_match", "market": market,
            "kickoff": (START + timedelta(days=day)).isoformat(), "odds": 2.0, "p": p, "y": y, "w": 1.0,
            "book_count": books, "book_spread": spread, "spix_value": spix, "value_at_close": value}


def test_holdout_is_the_last_week():
    picks = [pick(day, day) for day in range(15)]
    train, holdout = lc.split_by_week(picks)
    assert [p["fixture_id"] for p in holdout] == [str(d) for d in range(8, 15)]
    assert [p["fixture_id"] for p in train] == [str(d) for d in range(8)]


def test_the_cutoff_keeps_a_fifth_of_picks_in_each_label():
    tied = [pick(i, 0, spread=0.01, value=0.05) for i in range(85)] + \
           [pick(100 + i, 0, spread=0.05 + i / 1000, value=-0.05) for i in range(15)]
    assert lc.choose_spread_cutoff(tied) is None  # every candidate leaves only 15% low
    spread_out = [pick(i, 0, spread=i / 1000, value=0.05 if i < 50 else -0.05) for i in range(100)]
    cutoff = lc.choose_spread_cutoff(spread_out)
    high, low = lc.split_labels(spread_out, cutoff)
    assert min(len(high), len(low)) >= 20 and cutoff == pytest.approx(0.0495, abs=0.002)


def test_a_clearly_better_high_label_passes_and_an_equal_one_does_not():
    high = [pick(i, 0, value=0.05, y=1.0 if i % 2 else 0.0) for i in range(60)]
    low = [pick(100 + i, 0, value=-0.02, y=1.0 if i % 2 else 0.0) for i in range(60)]
    assert lc.passes(lc.compare(high, low, resamples=300))
    same = [pick(200 + i, 0, value=0.05, y=1.0 if i % 2 else 0.0) for i in range(60)]
    assert not lc.passes(lc.compare(high, same, resamples=300))


def test_the_spix_check_needs_enough_picks_on_both_sides():
    agreeing = [pick(i, 0, market="corners", spix=0.05, value=0.08) for i in range(10)]
    others = [pick(100 + i, 0, market="corners", spix=-0.05, value=-0.02) for i in range(60)]
    check = lc.spix_check(agreeing + others, resamples=200)["corners"]
    assert (check["agree"], check["disagree"], check["passes"]) == (10, 60, False)


def test_pick_rows_takes_the_market_policy_and_carries_the_evidence():
    row = {"fixture_id": "7", "stage": "pre_match", "market": "goals", "kickoff": "2026-10-10T15:00:00Z",
           "p_ref": 0.5, "p_spix": 0.45, "best_odds": 2.1, "best_book": "A", "selected": False, "y": 1.0, "w": 1.0,
           "book_count": 5, "book_spread": 0.01, "p_close": 0.52}
    no_value = {**row, "fixture_id": "8", "best_odds": 2.0}
    [only] = lc.pick_rows([row, no_value])
    assert (only["fixture_id"], only["p"], only["book_spread"]) == ("7", pytest.approx(0.5), 0.01)
    assert only["spix_value"] == pytest.approx(0.45 * 2.1 - 1)
    assert only["value_at_close"] == pytest.approx(0.52 * 2.1 - 1)
