"""Late-goal study (D103): another goal after 75', its splits, the fair-price clock and the in-play first look."""
import json
import math

import pytest

from Scripts.ops import late_goal_study as lg

HOME, AWAY = 10, 20


def goal(minute, team=HOME, extra=None, detail="Normal Goal"):
    return {"time": {"elapsed": minute, "extra": extra}, "team": {"id": team}, "type": "Goal", "detail": detail,
            "player": {"id": 1}}


def red(minute, team=HOME, player=7):
    return {"time": {"elapsed": minute, "extra": None}, "team": {"id": team}, "type": "Card", "detail": "Red Card",
            "player": {"id": player}}


def fixture(home_goals, away_goals, **extra):
    return {"fixture_id": "1", "league": "EPL", "season": 2023, "home_id": HOME, "away_id": AWAY,
            "home_goals": home_goals, "away_goals": away_goals, **extra}


def match(goals=(), reds=(), season=2023, **extra):
    """A built match: goals and reds as (minute, side)."""
    return {"fixture_id": "1", "league": "EPL", "season": season, "goals": list(goals), "reds": list(reds), **extra}


# --- minutes ---------------------------------------------------------------------------------------------------

def test_the_shown_minute_keeps_first_half_stoppage_in_the_first_half():
    assert lg.shown_minute(45, 3) == 45
    assert lg.shown_minute(90, 4) == 94
    assert lg.shown_minute(76, None) == 76


def test_extra_time_and_missing_minutes_have_no_shown_minute():
    assert lg.shown_minute(105, None) is None
    assert lg.shown_minute(120, 3) is None
    assert lg.shown_minute(None, None) is None
    assert lg.shown_minute(-5, None) is None


# --- one match from its events -----------------------------------------------------------------------------------

def test_a_goal_counts_for_the_team_the_event_lists_own_goals_included():
    events = [goal(12), goal(40, AWAY, detail="Own Goal"), goal(80, AWAY, detail="Penalty")]
    built = lg.build_match(events, fixture(1, 2))
    assert built["goals"] == [(12, "home"), (40, "away"), (80, "away")]


def test_missed_penalties_are_not_goals():
    built = lg.build_match([goal(30), goal(60, detail="Missed Penalty")], fixture(1, 0))
    assert built["goals"] == [(30, "home")]


def test_a_match_whose_goals_do_not_add_up_to_its_score_is_left_out():
    assert lg.build_match([goal(30)], fixture(2, 0)) is None
    assert lg.build_match([goal(None)], fixture(1, 0)) is None  # a goal with no minute cannot be placed


def test_a_goalless_match_with_no_events_is_kept():
    assert lg.build_match([], fixture(0, 0))["goals"] == []


def test_a_red_card_cancelled_by_var_is_dropped():
    cancelled = {"time": {"elapsed": 52, "extra": None}, "team": {"id": HOME}, "type": "Var",
                 "detail": "Red card cancelled", "player": {"id": 7}}
    built = lg.build_match([red(50), cancelled, red(70, AWAY, player=9)], fixture(0, 0))
    assert built["reds"] == [(70, "away")]


# --- the state at a minute and what came after -------------------------------------------------------------------

def test_a_goal_shown_at_75_is_before_the_bet_and_one_at_76_after_it():
    m = match(goals=[(75, "home"), (76, "away")])
    assert lg.score_at(m, 75) == (1, 0)
    assert lg.goal_after(m, 75) is True
    assert lg.goal_after(match(goals=[(75, "home")]), 75) is False


def test_stoppage_time_goals_count_as_after():
    assert lg.goal_after(match(goals=[(93, "home")]), 85) is True


def test_a_red_card_before_75_includes_minute_75():
    assert lg.red_before(match(reds=[(75, "home")]), 75) is True
    assert lg.red_before(match(reds=[(76, "home")]), 75) is False


def test_groups():
    assert [lg.gap_group(1, 1), lg.gap_group(2, 1), lg.gap_group(0, 2)] == ["level", "one apart", "two or more"]
    assert [lg.goals_group(0), lg.goals_group(3), lg.goals_group(6)] == ["0", "3", "4+"]
    assert [lg.expectation_band(2.0), lg.expectation_band(2.5), lg.expectation_band(2.9),
            lg.expectation_band(3.4)] == ["under 2.3", "2.3 to 2.7", "2.7 to 3.1", "3.1 or more"]


# --- rates ---------------------------------------------------------------------------------------------------------

def test_wilson_interval():
    low, high = lg.wilson(50, 100)
    assert low == pytest.approx(0.4038, abs=1e-3) and high == pytest.approx(0.5962, abs=1e-3)
    assert lg.wilson(0, 0) == (None, None)


def test_split_counts_matches_with_a_later_goal_per_group():
    matches = [match(goals=[(30, "home"), (80, "home")]), match(goals=[(30, "home")]), match(goals=[(85, "away")])]
    rows = {r["group"]: r for r in lg.split(matches, 75, lambda m, h, a: lg.gap_group(h, a))}
    assert rows["one apart"]["matches"] == 2 and rows["one apart"]["with_goal"] == 1
    assert rows["level"]["matches"] == 1 and rows["level"]["rate"] == 1.0


# --- the fair-price clock ----------------------------------------------------------------------------------------

def test_intensity_is_goals_per_match_at_each_minute_from_the_start():
    matches = [match(goals=[(76, "home"), (93, "home")]), match(goals=[(10, "home"), (76, "away")])]
    intensity = lg.minute_intensity(matches, start=75)
    assert intensity[76] == 1.0 and intensity[93] == 0.5 and 10 not in intensity


def test_the_price_drifts_out_as_the_remaining_goals_run_down():
    intensity = {m: 0.03 for m in range(76, 96)}  # 20 equal minutes, 0.6 goals left at 75'
    clock = lg.price_clock(intensity, 1.60, targets=(1.80, 3.00))
    lam = -math.log(1 - 1 / 1.60)
    first, second = clock["targets"]
    assert clock["lambda"] == pytest.approx(lam)
    # a fair 1.80 needs 0.811 goals left, which is lam * (95 - t) / 20, so the first whole minute is 79
    assert first["minute"] == 79 and second["minute"] == 87
    assert first["fires"] == pytest.approx(math.exp(-lam * 4 / 20))   # no goal from 75' to 79'
    assert second["fires"] == pytest.approx(math.exp(-lam * 12 / 20))


def test_a_match_already_at_the_target_fires_at_the_start():
    clock = lg.price_clock({m: 0.03 for m in range(76, 96)}, 1.90, targets=(1.80,))
    assert clock["targets"][0]["minute"] == 75 and clock["targets"][0]["fires"] == 1.0


# --- the pre-match expectation -----------------------------------------------------------------------------------

def test_the_expected_total_from_an_over_2_5_price():
    p_over = lg.devig_over(1.80, 2.10)
    assert p_over == pytest.approx((1 / 1.80) / (1 / 1.80 + 1 / 2.10))
    lam = lg.lambda_from_over25(p_over)
    assert 1 - math.exp(-lam) * (1 + lam + lam * lam / 2) == pytest.approx(p_over, abs=1e-6)


def test_calibration_compares_predicted_and_happened_by_fifths():
    matches = [match(goals=[(80, "home")] if i % 2 else [], expectation=2.0 + i / 100) for i in range(100)]
    rows = lg.calibration(matches, 75, share=0.25, groups=5)
    assert len(rows) == 5 and sum(r["matches"] for r in rows) == 100
    assert rows[0]["predicted"] < rows[-1]["predicted"]
    assert rows[0]["happened"] == 0.5


# --- the live engine's timing table ------------------------------------------------------------------------------

def test_timing_shares_put_stoppage_time_in_the_half_it_belongs_to():
    matches = [match(goals=[(45, "home"), (93, "home"), (10, "away"), (76, "away")])]
    rows = {r["window"]: r for r in lg.timing_shares(matches)}
    assert rows["31'–45' and stoppage"]["goals"] == 1
    assert rows["76'–90' and stoppage"]["goals"] == 2
    assert rows["76'–90' and stoppage"]["share"] == 0.5
    assert rows["76'–90' and stoppage"]["engine"] == 0.22


# --- the in-play first look --------------------------------------------------------------------------------------

def odds(lines, bookmakers=1, name="API-Football Live"):
    return {"bookmakers_count": bookmakers, "totals": {"goals": [
        {"line": line, "offers": [{"bookmaker": name, "over_odds": price, "under_odds": 1.5,
                                   "fair_over_prob": 0.4}]} for line, price in lines]}}


def test_the_one_more_goal_price_is_the_line_half_a_goal_above_the_score():
    assert lg.one_more_goal(odds([(2.5, 2.10), (3.5, 7.0)]), total=2) == (2.10, 0.4)


def test_a_snapshot_whose_lines_do_not_fit_the_score_is_skipped():
    assert lg.one_more_goal(odds([(2.5, 1.36), (3.5, 2.75)]), total=1) is None  # the score was stale
    assert lg.one_more_goal(odds([(1.5, 1.20), (2.5, 2.10)]), total=2) is None  # a settled line still listed


def test_the_pre_match_fallback_is_skipped():
    assert lg.one_more_goal(odds([(2.5, 1.77)], bookmakers=13, name="Bet365"), total=2) is None


def test_an_inplay_path_records_the_price_at_75_and_when_it_reached_each_target():
    snaps = [{"minute": 74.0, "total": 1, "price": 1.70, "fair_prob": 0.55},
             {"minute": 78.0, "total": 1, "price": 1.85, "fair_prob": 0.52},
             {"minute": 86.0, "total": 1, "price": 3.10, "fair_prob": 0.30}]
    path = lg.inplay_path(snaps, final_total=2, goal_minutes=[89])
    assert path["price_75"] == 1.70 and path["qualifies"] is True
    assert path["reached"] == {"1.8": 78.0, "3.0": 86.0}
    assert path["goal_after_75"] is True


def test_an_inplay_path_stops_at_the_first_goal_after_75():
    snaps = [{"minute": 75.0, "total": 1, "price": 1.60, "fair_prob": 0.6},
             {"minute": 84.0, "total": 2, "price": 1.90, "fair_prob": 0.5}]  # 1.90 only after the goal
    path = lg.inplay_path(snaps, final_total=2, goal_minutes=[80])
    assert path["reached"] == {"1.8": None, "3.0": None}


# --- step 3's error bars and step 2's cost -----------------------------------------------------------------------

def test_the_roi_interval_narrows_with_the_square_root_of_the_bets():
    wide, narrow = lg.roi_half_width(1.80, 1 / 1.80, 100), lg.roi_half_width(1.80, 1 / 1.80, 400)
    assert wide == pytest.approx(1.96 * 1.80 * math.sqrt((1 / 1.80) * (1 - 1 / 1.80) / 100))
    assert narrow == pytest.approx(wide / 2)


def test_api_use_by_day_takes_the_lowest_remaining_count_each_day():
    rows = [("2026-11-07T09:00:00+00:00", json.dumps({"x-ratelimit-requests-limit": "7500",
                                                       "x-ratelimit-requests-remaining": "7000"})),
            ("2026-11-07T21:00:00+00:00", json.dumps({"x-ratelimit-requests-limit": "7500",
                                                       "x-ratelimit-requests-remaining": "3100"})),
            ("2026-11-08T10:00:00+00:00", json.dumps({"x-ratelimit-requests-limit": "75000",
                                                       "x-ratelimit-requests-remaining": "70000"}))]
    assert lg.api_use_by_day(rows) == {"2026-11-07": {"used": 4400, "limit": 7500, "last_seen": "21:00"},
                                       "2026-11-08": {"used": 5000, "limit": 75000, "last_seen": "10:00"}}


def test_the_read_expectation_is_its_goals_total_projection():
    read = {"data": {"canonical_results": [
        {"market": {"group": "corners", "key": "totals"}, "projection": {"value": 9.8}},
        {"market": {"group": "goals", "key": "totals"}, "projection": {"value": 2.84}}]}}
    assert lg.goals_projection(read) == 2.84
    assert lg.goals_projection({"data": {"canonical_results": []}}) is None


def test_the_read_expectation_is_the_market_anchored_total_where_the_card_shows_it():
    goals = {"market": {"group": "goals", "key": "totals"}, "projection": {"value": 1.39},
             "context": {"market_anchor": {"goal_figures": {"total_goals": 2.83}}}}
    assert lg.goals_projection({"data": {"canonical_results": [goals]}}) == 2.83


def test_the_clock_table_also_reads_the_chance_after_a_quiet_spell():
    ms = [match(goals=[(78, "home")]), match(goals=[(85, "home")]), match()]
    rows = {r["minute"]: r for r in lg.clock_table(ms, (75, 80))}
    assert rows[75]["chance_after"] == pytest.approx(2 / 3)
    assert rows[80]["chance_after"] == pytest.approx(1 / 3)  # only the 85' goal is after 80'
    assert rows[80]["quiet_since_75"] == 2 and rows[80]["chance_after_if_quiet"] == 0.5
