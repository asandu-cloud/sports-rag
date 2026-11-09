"""Pick evidence (D82): past matches priced like tonight, what each pick would have done in them, the verdict and the
fixed wording. Pure: no database."""
from datetime import date, timedelta

import pytest

from data_platform import pick_evidence as pe

H, A, OTHER = 40, 52, 99


def match(day, home_id, away_id, hg, ag, *, hp=2.0, dp=3.4, ap=3.6, corners=10, league="EPL", home=None, away=None):
    return {"date": (date(2025, 1, 1) + timedelta(days=day)).isoformat(), "season": "2024/25", "competition": league,
            "home_id": home_id, "away_id": away_id, "home": home or f"T{home_id}", "away": away or f"T{away_id}",
            "home_goals": hg, "away_goals": ag, "corners": corners,
            "home_price": hp, "draw_price": dp, "away_price": ap}


def pick(market, side, line=None, chance=0.6, price=1.5, label="pick"):
    return {"market": market, "side": side, "line": line, "chance": chance, "typical_price": price, "label": label}


# ---- choosing the groups -------------------------------------------------------------------------------------------

def test_team_group_takes_the_matches_whose_own_win_odds_are_closest_to_tonights():
    history = [match(1, H, OTHER, 1, 0, hp=1.60), match(2, OTHER, H, 0, 1, ap=1.70), match(3, H, OTHER, 1, 0, hp=3.00),
               match(4, OTHER, OTHER + 1, 1, 0, hp=1.60)]
    rows = pe.team_group(history, H, 1.62, "2026-01-01", n=2)
    assert [(r["date"], r["venue"], r["team_price"]) for r in rows] == [("2025-01-03", "A", 1.70),
                                                                        ("2025-01-02", "H", 1.60)]


def test_ties_in_price_go_to_the_newer_match_and_only_matches_before_tonight_count():
    history = [match(1, H, OTHER, 1, 0, hp=1.60), match(5, H, OTHER, 1, 0, hp=1.60), match(9, H, OTHER, 1, 0, hp=1.60)]
    rows = pe.team_group(history, H, 1.60, "2025-01-10", n=1)
    assert [r["date"] for r in rows] == ["2025-01-06"]


def test_league_group_uses_both_prices_and_only_tonights_league():
    history = [match(1, 1, 2, 0, 0, hp=1.50, ap=6.0), match(2, 3, 4, 0, 0, hp=1.55, ap=5.5),
               match(3, 5, 6, 0, 0, hp=1.55, ap=5.5, league="LaLiga"), match(4, 7, 8, 0, 0, hp=3.0, ap=2.4)]
    rows = pe.league_group(history, "EPL", 1.55, 5.6, "2026-01-01", n=2)
    assert [r["date"] for r in rows] == ["2025-01-03", "2025-01-02"]


# ---- what a pick would have done ------------------------------------------------------------------------------------

@pytest.mark.parametrize("view,venue,expected", [("home", "H", (2, 1)), ("home", "A", (1, 2)),
                                                 ("away", "H", (1, 2)), ("away", "A", (2, 1)),
                                                 ("league", None, (2, 1))])
def test_roles_read_the_score_from_tonights_home_and_away_sides(view, venue, expected):
    row = {**match(1, H, OTHER, 2, 1), "venue": venue}
    assert pe.roles(row, view) == expected


def test_won_for_every_market():
    row = {**match(1, H, OTHER, 2, 1, corners=11), "venue": "H"}
    assert pe.won(pick("goals", "over", 2.5), row, view="league")
    assert not pe.won(pick("goals", "under", 2.5), row, view="league")
    assert pe.won(pick("corners", "over", 10.5), row, view="league")
    assert pe.won(pick("btts", "yes"), row, view="league")
    assert pe.won(pick("moneyline", "home"), row, view="home")
    assert not pe.won(pick("moneyline", "draw"), row, view="home")
    assert not pe.won(pick("spreads", "home", -1.5), row, view="home")  # won by one only
    assert pe.won(pick("spreads", "away", 1.5), row, view="home")       # the other side lost by one


def test_a_negative_handicap_reads_won_by_two_or_more_from_every_view():
    p = pick("spreads", "home", -1.5)
    assert pe.means(p, view="home", home="Arsenal", away="Fulham") == "Arsenal won by 2 or more"
    assert pe.means(p, view="away", home="Arsenal", away="Fulham") == "Fulham lost by 2 or more"
    assert pe.means(p, view="league", home="Arsenal", away="Fulham") == "the home side won by 2 or more"


@pytest.mark.parametrize("p,view,expected", [
    (pick("spreads", "away", 2.5), "away", "Forest did not lose by more than 2"),
    (pick("spreads", "away", 2.5), "home", "City did not win by more than 2"),
    (pick("spreads", "away", 0.5), "away", "Forest did not lose"),
    (pick("spreads", "away", 0.5), "home", "City did not win"),
    (pick("spreads", "home", -0.5), "away", "Forest lost"),
    (pick("moneyline", "home"), "home", "City won"),
    (pick("moneyline", "home"), "away", "Forest lost"),
    (pick("moneyline", "away"), "league", "the away side won"),
    (pick("moneyline", "draw"), "home", "the match was drawn"),
    (pick("goals", "over", 2.5), "league", "the match had 3 or more goals"),
    (pick("corners", "under", 11.5), "home", "the match had 11 or fewer corners"),
    (pick("btts", "yes"), "home", "both teams scored"),
    (pick("btts", "no"), "home", "at least one team failed to score"),
])
def test_means_says_what_winning_was_in_plain_words(p, view, expected):
    assert pe.means(p, view=view, home="City", away="Forest") == expected


# ---- the verdict ---------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("rates,expected", [
    ([0.7, 0.6, 0.8], "All three groups agree."),
    ([0.4, 0.3, 0.2], "All three groups lean against it."),
    ([0.4, 0.3, 0.7], "Two of the three groups lean against it."),
    ([0.7, 0.6, 0.55], "Two of the three groups agree."),
    ([0.7, 0.4, 0.55], "The groups are split."),
    ([0.7, 0.55, 0.5], "Only one group clearly agrees."),
    ([0.55, 0.5, 0.58], "The groups don't settle it."),
    ([0.7, 0.6], "Both groups agree."),
    ([0.4, 0.3], "Both groups lean against it."),
    ([0.7, 0.4], "The groups are split."),
    ([0.7, 0.55], "Only one group clearly agrees."),
])
def test_verdict_is_a_rule_on_the_counts(rates, expected):
    assert pe.verdict(rates) == expected


# ---- the words -----------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("chances,goals,expected", [
    ({"home": 0.65, "draw": 0.2, "away": 0.15}, {"home": 2.1, "away": 0.6}, "Liverpool are clear favourites, with "
                                                                              "most goals expected from Liverpool."),
    ({"home": 0.30, "draw": 0.25, "away": 0.45}, {"home": 1.2, "away": 1.5}, "Palace are favourites, with goals "
                                                                              "expected at both ends."),
    ({"home": 0.40, "draw": 0.28, "away": 0.32}, {"home": 1.6, "away": 1.5}, "Liverpool are slight favourites, with "
                                                                              "plenty of goals expected."),
    ({"home": 0.36, "draw": 0.30, "away": 0.34}, {"home": 1.0, "away": 0.9}, "Too close to call, with few goals "
                                                                              "expected."),
    ({"home": 0.65, "draw": 0.2, "away": 0.15}, None, "Liverpool are clear favourites."),
])
def test_lead_names_the_favourite_and_the_goals(chances, goals, expected):
    assert pe.lead("Liverpool", "Palace", chances, goals) == expected


def _groups(totals_by_key, flags=None):
    out = []
    for key, totals in totals_by_key.items():
        rows = [{**match(i, H, OTHER, t, 0, corners=t), "venue": "H"} for i, t in enumerate(totals)]
        out.append({"key": key, "rows": rows})
    return out


def test_detail_for_a_total_gives_the_groups_averages_and_the_safer_line():
    p = pick("corners", "under", 11.5, chance=0.66, label="Under 11.5 corners")
    options = [p, pick("corners", "under", 12.5, chance=0.75, price=1.17, label="Under 12.5 corners"),
               pick("corners", "under", 13.5, chance=0.84, price=1.08, label="Under 13.5 corners"),
               pick("corners", "over", 9.5, chance=0.55, price=1.70, label="Over 9.5 corners")]
    groups = _groups({"home": [9, 10], "away": [10, 11], "league": [8, 9]})
    assert pe.detail(p, groups, options, "City", "Forest") == (
        "Similar matches averaged 8.5 to 10.5 corners. Under 12.5 corners is safer but pays only 1.17.")


def test_detail_for_a_handicap_gives_the_picked_teams_record():
    p = pick("spreads", "away", 2.5, label="Forest +2.5")
    groups = [{"key": "away", "rows": [{**match(1, OTHER, A, 1, 1), "venue": "A"}, {**match(2, A, OTHER, 0, 2),
                                                                                     "venue": "H"},
                                       {**match(3, A, OTHER, 3, 1), "venue": "H"}]}]
    assert pe.detail(p, groups, [p], "City", "Forest") == "Forest won 1, drew 1 and lost 1 of their 3 matches at " \
                                                          "similar odds."


def test_detail_for_both_teams_to_score_gives_each_teams_scoring():
    p = pick("btts", "yes", label="Both teams to score")
    groups = [{"key": "home", "rows": [{**match(1, H, OTHER, 1, 1), "venue": "H"}, {**match(2, H, OTHER, 0, 1),
                                                                                    "venue": "H"}]},
              {"key": "away", "rows": [{**match(1, A, OTHER, 2, 0), "venue": "H"}]}]
    assert pe.detail(p, groups, [p], "City", "Forest") == "City scored in 1 of their 2, Forest in 1 of theirs."


def test_detail_never_runs_over_25_words():
    p = pick("moneyline", "home", label="Borussia Mönchengladbach to win")
    rows = [{**match(i, H, OTHER, 1, 0), "venue": "H"} for i in range(12)]
    text = pe.detail(p, [{"key": "home", "rows": rows}], [p], " ".join(["Name"] * 15), "X")
    assert len(text.split()) <= 25


# ---- the whole card ------------------------------------------------------------------------------------------------

def block(most_likely, *, home_price=1.62, draw_price=4.0, away_price=5.5, extra_options=()):
    moneyline = [{**pick("moneyline", side, None, price=price), "label": side}
                 for side, price in (("home", home_price), ("draw", draw_price), ("away", away_price))]
    markets = [{"market": "moneyline", "options": moneyline}]
    for option in list(most_likely) + list(extra_options):
        if option["market"] != "moneyline":
            found = next((m for m in markets if m["market"] == option["market"]), None)
            if found is None:
                markets.append({"market": option["market"], "options": [option]})
            elif option not in found["options"]:
                found["options"].append(option)
    return {"version": "recommendations.v1", "most_likely": list(most_likely), "markets": markets}


FIXTURE = {"event_id": "1", "league": "EPL", "home_team": "City", "away_team": "Forest",
           "kickoff": "2026-11-21T15:00:00+00:00"}
FIGURES = {"result_chances": {"home": 0.62, "draw": 0.22, "away": 0.16}, "goals_expected": {"home": 1.9, "away": 0.8},
           "corners_expected": 10.4}


def season(n_home=12, n_away=12, n_league=40):
    history = []
    for i in range(n_home):
        history.append(match(i, H, OTHER, 1, 0, hp=1.6 + i / 100, corners=9, league="Championship"))
    for i in range(n_away):
        history.append(match(i, OTHER, A, 1, 1, ap=5.5 + i / 10, corners=12, league="Championship"))
    for i in range(n_league):
        history.append(match(100 + i, 1000 + i, 2000 + i, 1, 0, hp=1.6, ap=5.5, corners=8 + i % 6))
    return history


def test_build_gives_the_story_the_shared_groups_and_each_picks_evidence():
    over = pick("goals", "over", 1.5, chance=0.7, price=1.33, label="Over 1.5 goals")
    built = pe.build(block([over]), FIXTURE, history=season(), team_ids=(H, A), figures=FIGURES)
    assert built["story"] == {"lead": "City are clear favourites, with most goals expected from City.",
                              "result_chances": FIGURES["result_chances"], "goals_expected": FIGURES["goals_expected"],
                              "corners_expected": 10.4}
    evidence = built["evidence"]
    assert evidence["version"] == "pick-evidence.v1" and evidence["competition"] == "EPL"
    assert evidence["tonight"] == {"home": 1.62, "draw": 4.0, "away": 5.5}
    assert [(g["key"], g["team"], len(g["matches"])) for g in evidence["groups"]] == [
        ("home", "City", 12), ("away", "Forest", 12), ("league", None, 40)]
    assert evidence["groups"][0]["price_from"] == 1.6 and evidence["groups"][0]["price_to"] == 1.71
    assert set(evidence["groups"][0]["matches"][0]) >= {"date", "season", "competition", "home", "away", "home_price",
                                                        "draw_price", "away_price", "home_goals", "away_goals",
                                                        "corners", "venue"}
    one = built["picks"][0]
    assert [(g["key"], g["won"], g["of"]) for g in one["groups"]] == [("home", 0, 12), ("away", 12, 12),
                                                                      ("league", 0, 40)]
    assert one["groups"][1]["results"] == [True] * 12
    assert one["verdict"] == "Two of the three groups lean against it."
    assert one["groups"][0]["means"] == "the match had 2 or more goals"


def test_a_short_group_is_left_out_and_two_groups_are_needed():
    over = pick("goals", "over", 1.5, label="Over 1.5 goals")
    built = pe.build(block([over]), FIXTURE, history=season(n_away=7), team_ids=(H, A), figures=FIGURES)
    assert [g["key"] for g in built["evidence"]["groups"]] == ["home", "league"]
    assert built["picks"][0]["verdict"] == "Both groups lean against it."
    built = pe.build(block([over]), FIXTURE, history=season(n_league=19), team_ids=(H, A), figures=FIGURES)
    assert [g["key"] for g in built["evidence"]["groups"]] == ["home", "away"]
    built = pe.build(block([over]), FIXTURE, history=season(n_home=7, n_away=7), team_ids=(H, A), figures=FIGURES)
    assert built["evidence"] is None and built["picks"] == {}


def test_a_european_cup_card_gets_the_story_but_no_evidence():
    over = pick("goals", "over", 1.5, label="Over 1.5 goals")
    built = pe.build(block([over]), {**FIXTURE, "league": "UCL"}, history=season(), team_ids=(H, A), figures=FIGURES)
    assert built["story"]["lead"] and built["evidence"] is None and built["picks"] == {}


def test_no_evidence_without_tonights_prices_or_the_teams():
    over = pick("goals", "over", 1.5, label="Over 1.5 goals")
    unpriced = block([over], home_price=None)
    assert pe.build(unpriced, FIXTURE, history=season(), team_ids=(H, A), figures=FIGURES)["evidence"] is None
    assert pe.build(block([over]), FIXTURE, history=season(), team_ids=None, figures=FIGURES)["evidence"] is None


def test_no_story_without_the_result_chances():
    built = pe.build(block([]), FIXTURE, history=season(), team_ids=(H, A),
                     figures={**FIGURES, "result_chances": None})
    assert built["story"] is None
