"""A4.1: calibration dataset from the decision audits of match reads."""
import json

import pytest

from data_platform import calibration_data as cd


def option(market_key, side=None, point=None, p=0.5, team=None, book="Book"):
    o = {"market_key": market_key, "point": point, "_model_prob": p, "bookmaker": book}
    if side is not None:
        o["side"] = side
    if team is not None:
        o["team"] = team
    return {"option": o}


def read(fid="7", evaluated="2026-10-10T12:00:00Z", kickoff="2026-10-10T15:00:00Z", results=None):
    return {"fixture_api_id": fid, "stage": "pre_match", "evaluated_at": evaluated, "kickoff": kickoff,
            "read_json": json.dumps({"canonical_results": results or []})}


def result(group, candidates, mean=None):
    return {"market": {"group": group}, "projection": {"value": mean},
            "fixture": {"home_team": "Lyon", "away_team": "Lens", "kickoff": "2026-10-10T15:00:00Z"},
            "context": {"decision_audit": {"candidates": candidates}}}


def test_the_latest_read_before_kickoff_is_used_per_fixture_and_stage():
    early = read(evaluated="2026-10-10T09:00:00Z", results=[result("btts", [option("btts", "Yes", p=0.4)])])
    late = read(evaluated="2026-10-10T14:00:00Z", results=[result("btts", [option("btts", "Yes", p=0.6)])])
    after = read(evaluated="2026-10-10T15:30:00Z", results=[result("btts", [option("btts", "Yes", p=0.9)])])
    chosen = cd.latest_reads([early, late, after])
    assert len(chosen) == 1 and chosen[0]["evaluated_at"] == late["evaluated_at"]


def test_selections_are_unique_per_line_and_side_across_bookmakers():
    r = read(results=[result("goals", [option("totals", "over", 2.5, 0.55), option("totals", "over", 2.5, 0.55, book="Other"),
                                       option("totals", "under", 2.5, 0.45), option("totals", "over", 3.5, 0.3)], mean=2.7)])
    picks = cd.selections(r)
    assert sorted((s["side"], s["point"]) for s in picks) == [("over", 2.5), ("over", 3.5), ("under", 2.5)]
    assert all(s["mean"] == 2.7 and s["market"] == "goals" for s in picks)


@pytest.mark.parametrize("selection,expected", [
    ({"market": "goals", "market_key": "totals", "side": "over", "point": 2.5}, (1.0, 1.0)),       # 3 goals
    ({"market": "goals", "market_key": "totals", "side": "under", "point": 3.25}, (1.0, 0.5)),     # half win
    ({"market": "goals", "market_key": "totals", "side": "over", "point": 3.0}, (None, 0.0)),      # push
    ({"market": "spreads", "market_key": "spreads", "side": "Lyon", "point": -1.0}, (None, 0.0)),  # 2-1: push
    ({"market": "spreads", "market_key": "spreads", "side": "Lens", "point": 0.75}, (0.0, 0.5)),   # half loss
    ({"market": "btts", "market_key": "btts", "side": "Yes", "point": None}, (1.0, 1.0)),
    ({"market": "moneyline", "market_key": "h2h", "side": "away", "point": None}, (0.0, 1.0)),
    ({"market": "corners", "market_key": "totals_corners_over_under", "side": "under", "point": 9.5}, (0.0, 1.0)),
    ({"market": "cards", "market_key": "totals_cards_over_under", "side": "over", "point": 3.5}, (1.0, 1.0)),
])
def test_grading_matches_settlement_including_half_results(selection, expected):
    outcome = {"home_goals": 2, "away_goals": 1, "corners": 11, "cards": 5}
    selection = dict(selection, home="Lyon", away="Lens")
    assert cd.grade(selection, outcome) == expected


def test_unknown_team_or_missing_statistic_is_not_graded():
    outcome = {"home_goals": 2, "away_goals": 1, "corners": None, "cards": 5}
    assert cd.grade({"market": "spreads", "market_key": "spreads", "side": "Nice", "point": 0.5, "home": "Lyon", "away": "Lens"}, outcome) is None
    assert cd.grade({"market": "corners", "market_key": "totals_corners_over_under", "side": "over", "point": 9.5,
                     "home": "Lyon", "away": "Lens"}, outcome) is None


def test_database_timestamps_without_a_timezone_are_read_as_utc():
    stored = read(evaluated="2026-10-10 14:00:00.000000", kickoff="2026-10-10T15:00:00Z",
                  results=[result("btts", [option("btts", "Yes", p=0.6)])])
    assert len(cd.latest_reads([stored])) == 1


def test_a_selection_carries_the_pick_flag_variance_and_a_vig_free_market_price():
    picked = option("totals", "over", 2.5, 0.55)
    picked["selector_selected"] = True
    picked["option"].update({"_implied_prob": 0.5, "_vig_adjusted_pair": True})
    one_sided = option("totals", "over", 3.5, 0.3)
    one_sided["option"].update({"_implied_prob": 0.35, "_vig_adjusted_pair": False})
    r = read(results=[result("goals", [option("totals", "over", 2.5, 0.55), picked, one_sided], mean=2.7)])
    r["read_json"] = r["read_json"].replace('"value": 2.7', '"value": 2.7, "variance": 3.4')
    by_line = {s["point"]: s for s in cd.selections(r)}
    assert by_line[2.5]["selected"] and by_line[2.5]["variance"] == 3.4
    assert by_line[2.5]["market_prob"] == 0.5  # from the later bookmaker with a two-sided price
    assert by_line[3.5]["market_prob"] is None and not by_line[3.5]["selected"]
