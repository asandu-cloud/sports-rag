"""Pick notes (D75): each pick gets one line for it and one against, from its own reason, the market's chance and how
often it would have won in each team's recent matches. GPT-5.6 may word them; a note with a figure that is not in
its facts, or a promise, is refused and the fixed wording is used."""
import pytest

from Scripts.data_platform import pick_notes as pn


def match(goals_for, goals_against, corners_for=None, corners_against=None):
    return {"goals_for": goals_for, "goals_against": goals_against,
            "corners_for": corners_for, "corners_against": corners_against}


ARSENAL = [match(3, 1, 7, 2), match(2, 0, 6, 3), match(0, 0, 4, 4), match(1, 2, 3, 5), match(4, 0, 9, 1)]
BRIGHTON = [match(0, 2, 2, 6), match(1, 1, 5, 5), match(2, 1, 6, 3), match(0, 3, 1, 8), match(1, 0, 4, 4)]


def option(market, side, line=None, label="Option", chance=0.6, **extra):
    return {"market": market, "side": side, "line": line, "label": label, "chance": chance, **extra}


def test_an_option_is_counted_from_each_team_in_its_role_tonight():
    arsenal_to_win = option("moneyline", "home")
    assert pn.team_form(arsenal_to_win, "home", ARSENAL)["last"] == {"played": 5, "landed": 3}  # Arsenal won 3
    assert pn.team_form(arsenal_to_win, "away", BRIGHTON)["last"] == {"played": 5, "landed": 2}  # Brighton lost 2
    brighton_plus_half = option("spreads", "away", 0.5)
    assert pn.team_form(brighton_plus_half, "away", BRIGHTON)["last"]["landed"] == 3  # Brighton avoided defeat 3 times


def test_goal_totals_and_corners_count_the_whole_match_and_skip_a_match_without_corners():
    assert pn.team_form(option("goals", "over", 2.5), "home", ARSENAL)["last"]["landed"] == 3
    assert pn.team_form(option("btts", "yes"), "away", BRIGHTON)["last"]["landed"] == 2
    corners = [*ARSENAL[:2], match(1, 1), *ARSENAL[2:]]  # a match with no corners count
    form = pn.team_form(option("corners", "over", 8.5), "home", corners)
    assert form["last"] == {"played": 5, "landed": 3} and form["season"]["played"] == 5


def test_last_five_and_the_season_are_counted_newest_first():
    form = pn.team_form(option("goals", "over", 2.5), "home", ARSENAL + [match(5, 0), match(3, 3)])
    assert form == {"last": {"played": 5, "landed": 3}, "season": {"played": 7, "landed": 5}}


@pytest.mark.parametrize("line", [2.0, 2.25, 2.75])
def test_a_whole_or_quarter_line_gets_no_form(line):
    assert pn.team_form(option("goals", "over", line), "home", ARSENAL) is None


def test_too_few_matches_give_no_form():
    assert pn.team_form(option("goals", "over", 2.5), "home", ARSENAL[:2]) is None


def block():
    return {"most_likely": [option("goals", "over", 2.5, "Over 2.5 goals", 0.72)],
            "value": [option("moneyline", "home", None, "Arsenal to win", 0.55, worth_it_at=1.86)]}


FIXTURE = {"home_team": "Arsenal", "away_team": "Brighton", "competition": "Premier League"}


def facts():
    return pn.match_facts(block(), FIXTURE, {"home": ARSENAL, "away": BRIGHTON})


def test_facts_give_each_pick_its_reason_chance_and_each_teams_form():
    over, win = facts()["picks"]
    assert over["id"] == "most_likely-0" and over["label"] == "Over 2.5 goals"
    assert over["reason"] == "The likeliest option in its market that still pays at least 1.30."
    assert (over["chance_percent"], over["loses_about_in_10"]) == (72, 3)
    assert over["form"] == [
        {"team": "Arsenal", "about": "match", "counted": "had 3 or more goals", "opposite": "had 2 goals or fewer",
         "last": {"played": 5, "landed": 3}, "season": {"played": 5, "landed": 3}},
        {"team": "Brighton", "about": "match", "counted": "had 3 or more goals", "opposite": "had 2 goals or fewer",
         "last": {"played": 5, "landed": 2}, "season": {"played": 5, "landed": 2}}]
    assert win["id"] == "value-0" and win["worth_it_at"] == 1.86
    assert win["reason"] == "The best price found is above what the market's chance says it is worth."
    assert facts()["competition"] == "Premier League"


def test_the_fixed_note_gives_the_best_form_for_and_the_weakest_form_against():
    over = facts()["picks"][0]
    note = pn.fixed_note(over, "Premier League")
    assert note["for"] == "3 of Arsenal's last 5 Premier League matches had 3 or more goals."
    assert note["against"] == "3 of Brighton's last 5 Premier League matches had 2 goals or fewer."


def test_with_no_form_against_it_the_fixed_note_says_how_often_it_still_loses():
    pick = {**facts()["picks"][0], "form": [{"team": "Arsenal", "about": "match", "counted": "had 3 or more goals",
                                              "opposite": "had 2 goals or fewer", "last": {"played": 5, "landed": 4},
                                              "season": {"played": 9, "landed": 6}}]}
    assert pn.fixed_note(pick, "Premier League")["against"] == "It still loses about 3 times in 10."
    assert pn.fixed_note({**pick, "form": []}, "Premier League")["for"] == (
        "The likeliest option in its market that still pays at least 1.30.")


def test_a_value_pick_without_form_for_it_names_the_price_it_is_worth():
    win = facts()["picks"][1]
    assert pn.fixed_note(win, "Premier League")["for"] == "Arsenal won 3 of their last 5 Premier League matches."
    assert pn.fixed_note({**win, "form": []}, "Premier League")["for"] == (
        "The best price found is above the 1.86 it is worth at the market's 55%.")


@pytest.mark.parametrize("name, short", [
    ("UEFA Champions League", "Champions League"),
    ("UEFA Europa League", "Europa League"),
    ("UEFA Europa Conference League", "Conference League"),
    ("Premier League", "Premier League"),
    (None, None),
])
def test_competitions_take_the_short_names_the_site_uses(name, short):
    assert pn.match_facts(block(), {**FIXTURE, "competition": name}, {})["competition"] == short


@pytest.mark.parametrize("market, side, line, role, about, counted", [
    ("goals", "over", 2.5, "home", "match", "had 3 or more goals"),
    ("goals", "under", 2.5, "away", "match", "had 2 goals or fewer"),
    ("corners", "over", 8.5, "home", "match", "had 9 or more corners"),
    ("corners", "under", 10.5, "away", "match", "had 10 corners or fewer"),
    ("btts", "yes", None, "home", "match", "saw both teams score"),
    ("btts", "no", None, "away", "match", "had a team fail to score"),
    ("moneyline", "home", None, "home", "team", "won"),
    ("moneyline", "home", None, "away", "team", "lost"),
    ("moneyline", "draw", None, "away", "team", "drew"),
    ("spreads", "home", -0.5, "home", "team", "won"),
    ("spreads", "home", -1.5, "away", "team", "lost by 2 or more"),
    ("spreads", "away", 0.5, "away", "team", "avoided defeat"),
    ("spreads", "away", 0.5, "home", "team", "failed to win"),
    ("spreads", "away", 1.5, "away", "team", "avoided losing by 2 or more"),
    ("spreads", "away", 2.5, "home", "team", "failed to win by 3 or more"),
    ("spreads", "home", -2.5, "home", "team", "won by 3 or more"),
])
def test_what_is_counted_is_said_from_each_teams_side(market, side, line, role, about, counted):
    assert pn.counted(option(market, side, line), role) == (about, counted)


@pytest.mark.parametrize("market, side, line, role, opposite", [
    ("goals", "over", 2.5, "home", "had 2 goals or fewer"),
    ("corners", "under", 10.5, "away", "had 11 or more corners"),
    ("btts", "yes", None, "home", "had a team fail to score"),
    ("moneyline", "home", None, "away", "avoided defeat"),
    ("moneyline", "away", None, "away", "failed to win"),
    ("moneyline", "draw", None, "home", "did not draw"),
    ("spreads", "away", 1.5, "home", "won by 2 or more"),
    ("spreads", "home", -1.5, "home", "failed to win by 2 or more"),
])
def test_the_opposite_is_said_from_each_teams_side_too(market, side, line, role, opposite):
    assert pn.opposite_counted(option(market, side, line), role) == opposite


def test_team_results_read_as_the_team_did_them():
    pick = pn.match_facts({"most_likely": [option("moneyline", "home", None, "Arsenal to win", 0.6)]}, FIXTURE,
                          {"home": ARSENAL, "away": BRIGHTON})["picks"][0]
    note = pn.fixed_note(pick, "Premier League")
    assert note["for"] == "Arsenal won 3 of their last 5 Premier League matches."
    assert note["against"] == "Brighton avoided defeat in 3 of their last 5 Premier League matches."
    two_draws = [match(1, 0), match(1, 1), match(2, 0), match(0, 0), match(3, 0)]
    pick = pn.match_facts({"most_likely": [option("spreads", "away", 0.5, "Brighton +0.5", 0.6)]}, FIXTURE,
                          {"home": two_draws, "away": BRIGHTON})["picks"][0]
    assert pn.fixed_note(pick, "Premier League")["against"] == "Arsenal won 3 of their last 5 Premier League matches."
    pick["form"][0]["last"]["landed"] = 0
    assert pn.fixed_note(pick, None)["against"] == "Arsenal won 5 of their last 5 matches."


def test_a_name_ending_in_s_takes_an_apostrophe_alone():
    rangers = pn.match_facts(block(), {**FIXTURE, "away_team": "Rangers"}, {"home": ARSENAL, "away": BRIGHTON})
    assert pn.fixed_note(rangers["picks"][0], "Premiership")["against"] == (
        "3 of Rangers' last 5 Premiership matches had 2 goals or fewer.")


def gpt_note(**overrides):
    return {"id": "most_likely-0", "for": "Arsenal's last 5 Premier League games saw 3 go over, and the market "
            "puts it at 72%.", "against": "Brighton's last 5 went over only 2 times.", **overrides}


def test_a_note_using_only_its_own_figures_is_kept():
    assert pn.valid_note(gpt_note(), facts()["picks"][0], facts())


@pytest.mark.parametrize("text", [
    "Arsenal scored in 4 of their last 5.",           # 4 is not one of this pick's figures
    "A lock: Arsenal went over in 3 of 5.",           # tipster word
    "This is guaranteed to land at 72%.",             # a promise
    "Our model rates it at 72%.",                     # the chance is the market's, not a model's
    "Over landed in three of Arsenal's last 5.",      # numbers in words cannot be checked
    "Back it now: 3 of Arsenal's last 5 went over.",  # an instruction to bet
    " ".join(["word"] * 40),                          # too long
    "",
])
def test_a_note_with_a_figure_not_in_its_facts_or_a_promise_is_refused(text):
    assert not pn.valid_note(gpt_note(**{"for": text}), facts()["picks"][0], facts())


@pytest.mark.parametrize("text", [
    "Arsenal had 3 or more goals in 3 of their last 5.",  # a match count said as one team's goals
    "Brighton saw both teams score in 2 of 5.",
])
def test_a_match_count_said_as_one_teams_own_is_refused(text):
    assert not pn.valid_note(gpt_note(**{"against": text}), facts()["picks"][0], facts())


def test_a_match_count_said_as_the_teams_matches_is_kept():
    assert pn.valid_note(gpt_note(**{"against": "3 of Brighton's last 5 matches had 2 goals or fewer."}),
                         facts()["picks"][0], facts())


def test_numbers_in_team_and_competition_names_are_allowed():
    named = pn.match_facts(block(), {**FIXTURE, "home_team": "Schalke 04"}, {"home": ARSENAL, "away": BRIGHTON})
    pick = named["picks"][0]
    assert pn.valid_note(gpt_note(**{"for": "Schalke 04 went over in 3 of their last 5."}), pick, named)


def test_gpt_wording_is_used_where_valid_and_the_fixed_wording_elsewhere():
    calls = []

    def generator(match_facts):
        calls.append(match_facts)
        return {"notes": [gpt_note(), {"id": "value-0", "for": "A sure thing.", "against": "None."}]}

    notes = pn.notes_for_match(facts(), generator=generator)
    assert notes["version"] == pn.VERSION and len(calls) == 1
    assert notes["notes"]["most_likely-0"] == {**{k: gpt_note()[k] for k in ("for", "against")}, "source": "gpt"}
    assert notes["notes"]["value-0"]["source"] == "fixed"
    assert notes["notes"]["value-0"] == {**pn.fixed_note(facts()["picks"][1], "Premier League"), "source": "fixed"}


def test_an_unreachable_writer_leaves_every_pick_with_the_fixed_wording():
    def broken(_):
        raise RuntimeError("down")

    notes = pn.notes_for_match(facts(), generator=broken)
    assert {n["source"] for n in notes["notes"].values()} == {"fixed"}


def test_notes_with_the_same_facts_are_reused_without_asking_again():
    first = pn.notes_for_match(facts(), generator=lambda _: {"notes": [gpt_note()]})

    def forbid(_):
        raise AssertionError("must not ask again")

    assert pn.notes_for_match(facts(), generator=forbid, cached=first) == first
    changed = pn.match_facts({**block(), "most_likely": [option("goals", "over", 2.5, "Over 2.5 goals", 0.70)]},
                             FIXTURE, {"home": ARSENAL, "away": BRIGHTON})
    assert pn.notes_for_match(changed, generator=lambda _: {"notes": []}, cached=first)["fact_key"] != first["fact_key"]
