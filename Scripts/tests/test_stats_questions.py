"""The stats questions prototype's question types (tracker D92): tested queries on the stats file, never the model."""
import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from Scripts.ops.stats_questions import questions, store

START = datetime(2026, 8, 16, 14, tzinfo=timezone.utc)
TEAMS = {1: ("Brentford", "EPL"), 2: ("Tottenham", "EPL"), 3: ("Arsenal", "EPL"), 4: ("Manchester United", "EPL"),
         5: ("Manchester City", "EPL"), 6: ("Bayern München", "Bundesliga")}


def _match(conn, fid, home, away, score, corners, *, comp="EPL", season=2026, day=0, cards=(2, 1), ref="ref-taylor"):
    kickoff = (START + timedelta(days=day)).isoformat()
    conn.execute("INSERT INTO matches VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                 (fid, comp, season, kickoff, home, away, score[0], score[1], ref))
    for team, opp, is_home, i in ((home, away, 1, 0), (away, home, 0, 1)):
        row = {"fixture_id": fid, "team_id": team, "opponent_id": opp, "is_home": is_home, "competition": comp,
               "season": season, "kickoff": kickoff, "goals_for": score[i], "goals_against": score[1 - i],
               "corners_for": corners[i], "corners_against": corners[1 - i], "cards_for": cards[i],
               "cards_against": cards[1 - i], "xg_for": 1.0 + i, "xg_against": 2.0 - i}
        conn.execute(f"INSERT INTO team_matches ({', '.join(row)}) VALUES ({', '.join('?' * len(row))})",
                     tuple(row.values()))


@pytest.fixture()
def conn(tmp_path):
    path = tmp_path / "stats.db"
    db = sqlite3.connect(path)
    db.executescript(store.SCHEMA)
    db.executemany("INSERT INTO competitions VALUES (?, ?, ?, ?)", [
        ("EPL", "Premier League", "domestic_league", "England"), ("UCL", "UEFA Champions League", "continental_cup",
                                                                   None),
        ("Bundesliga", "Bundesliga", "domestic_league", "Germany")])
    db.executemany("INSERT INTO teams VALUES (?, ?, NULL, ?)", [(i, n, lg) for i, (n, lg) in TEAMS.items()])
    db.executemany("INSERT INTO players VALUES (?, ?, ?, ?)", [
        (10, "Bukayo Saka", 3, "F"), (11, "Gabriel Magalhães", 3, "D"), (12, "Gabriel Jesus", 3, "F")])
    db.execute("INSERT INTO referees VALUES ('ref-taylor', 'Anthony Taylor'), ('ref-other', 'Other Ref')")
    # Brentford away in the league this season: match corners 12, 8, 11, 10, 9, 13 (Brentford's own 4, 3, 5, 4, 2, 6).
    for i, (own, opp) in enumerate(((4, 8), (3, 5), (5, 6), (4, 6), (2, 7), (6, 7))):
        _match(db, 100 + i, 2 + i % 4, 1, (1, i % 3), (opp, own), day=14 * i + 7)
    for i in range(2):  # Brentford at home, and an older away match that "this season" must leave out
        _match(db, 200 + i, 1, 2, (2, 0), (9, 1), day=14 * i)
    _match(db, 300, 3, 1, (3, 0), (10, 0), season=2025, day=-300, ref="ref-other")
    db.execute("INSERT INTO build VALUES ('data_up_to', '2026-11-08T20:30:00+00:00')")
    for fid, minutes, sot, started in ((100, 90, 2, 1), (101, 90, 0, 1), (102, 30, 3, 0), (103, 90, 2, 1)):
        db.execute("INSERT INTO player_matches (fixture_id, player_id, team_id, competition, season, kickoff, "
                   "started, position, minutes, shots_on_target) VALUES (?, 10, 3, 'EPL', 2026, ?, ?, 'F', ?, ?)",
                   (fid, (START + timedelta(days=fid)).isoformat(), started, minutes, sot))
    db.commit()
    db.close()
    connection = store.connect(path)
    yield connection
    connection.close()


def ask(conn, qtype, **params):
    return questions.answer(conn, qtype, {**questions.blank(qtype), **params})


def test_a_line_record_counts_the_matches_over_the_line_and_says_what_it_counted(conn):
    card = ask(conn, "line_record", team="brentford", stat="corners", side="match", line=9.5, venue="away")
    assert card["scope"] == "Brentford away · Premier League 2026/27 · 6 matches"
    assert "4 of 6" in card["headline"] and card["note"] is None
    assert len(card["matches"]["rows"]) == 6 and card["data_up_to"] == "Sun 8 Nov 2026"


def test_for_and_against_are_the_teams_own_and_its_opponents(conn):
    own = ask(conn, "team_average", team="Brentford", stat="corners", side="for", venue="away")
    conceded = ask(conn, "team_average", team="Brentford", stat="corners", side="against", venue="away")
    assert "4.0 corners won" in own["headline"] and "6.5 corners conceded" in conceded["headline"]


def test_a_team_average_compares_with_the_same_venue_across_the_league(conn):
    card = ask(conn, "team_average", team="Brentford", stat="corners", side="for", venue="away")
    away_sides = [4, 3, 5, 4, 2, 6, 1, 1]  # every away side's own corners this season: Brentford's six, Tottenham's two
    assert f"Premier League average for away sides: {sum(away_sides) / 8:.1f}." in card["detail"]


def test_names_resolve_by_nickname_initial_or_id_and_ask_when_several_fit(conn):
    assert ask(conn, "recent_form", team="Spurs")["scope"].startswith("Tottenham")
    which = ask(conn, "recent_form", team="Manchester")
    assert which["type"] == "which" and {o["label"] for o in which["options"]} == {"Manchester United",
                                                                                   "Manchester City"}
    gabriel = ask(conn, "player", player="Gabriel", stat="shots_on_target")
    assert gabriel["type"] == "which" and len(gabriel["options"]) == 2
    assert ask(conn, "player", player="B. Saka", stat="shots_on_target")["scope"].startswith("Bukayo Saka")
    assert ask(conn, "player", player="#10", stat="shots_on_target")["scope"].startswith("Bukayo Saka")
    assert ask(conn, "recent_form", team="Sputrs Hotpsur")["type"] == "not_found"


def test_recent_form_counts_results_from_the_teams_side(conn):
    card = ask(conn, "recent_form", team="Brentford", last=3)
    assert card["scope"] == "Brentford · All competitions · last 3 matches"
    # Newest first, Brentford away each time: won 2-1, drew 1-1, lost 0-1.
    assert "1 win, 1 draw, 1 defeat" in card["headline"]
    assert [row[4] for row in card["matches"]["rows"]] == ["W", "D", "L"]


def test_head_to_head_counts_wins_for_each_side_over_every_season(conn):
    card = ask(conn, "head_to_head", team="Arsenal", opponent="Brentford")
    assert "Arsenal 1 win, Brentford 1 win, 1 draw" in card["headline"]  # 3-0 last season, 1-1 and 1-2 this one
    assert card["scope"] == "Arsenal v Brentford · All competitions since 2019/20 · 3 meetings"
    assert card["note"] == "Small sample: only 3 matches."


def test_a_referees_cards_are_both_teams_cards_per_match(conn):
    card = ask(conn, "referee", referee="taylor", stat="cards", line=2.5)
    assert card["scope"] == "Anthony Taylor · All competitions · 2026/27 · 8 matches"
    assert "3.0 cards (yellow and red) a match" in card["headline"]
    assert any("Over 2.5 in 8 of 8" in line for line in card["detail"])


def test_a_player_threshold_counts_matches_and_starts_only_drops_the_sub_appearance(conn):
    card = ask(conn, "player", player="Saka", stat="shots_on_target", at_least=2)
    assert "2+ shots on target in 3 of 4 matches" in card["headline"]
    starts = ask(conn, "player", player="Saka", stat="shots_on_target", at_least=2, starts_only=True)
    assert "in 2 of 3 matches" in starts["headline"]


def test_leaders_rank_teams_with_enough_matches(conn):
    card = ask(conn, "leaders", kind="team", stat="corners", side="for", competition="EPL")
    assert card["table"]["rows"][0][1] == "Brentford" and card["note"] == "Teams with at least 3 matches."
    fewest = ask(conn, "leaders", kind="team", stat="corners", side="for", competition="EPL", order="fewest")
    assert fewest["table"]["rows"][0][1] != "Brentford"


def test_choices_outside_the_fixed_lists_are_refused(conn):
    assert ask(conn, "team_average", team="Brentford", stat="tackles")["type"] == "invalid"
    assert ask(conn, "line_record", team="Brentford", stat="possession", line=50.5)["type"] == "invalid"
    assert ask(conn, "leaders", kind="team", stat="corners")["type"] == "invalid"  # leaders need a competition
    assert ask(conn, "drop_table", team="x")["type"] == "invalid"


def test_a_whole_name_beats_a_prefix_and_a_much_busier_namesake_is_chosen(conn, tmp_path):
    path = tmp_path / "stats.db"
    db = sqlite3.connect(path)
    db.executemany("INSERT INTO players VALUES (?, ?, ?, ?)", [(13, "Tatsuhiro Sakamoto", 1, "F"),
                                                              (14, "Mathis Saka", 6, "M")])
    db.commit(); db.close()
    card = ask(conn, "player", player="Saka", stat="shots_on_target")
    assert card["type"] == "player" and card["scope"].startswith("Bukayo Saka (Arsenal)")


def test_a_match_total_line_compares_with_every_league_match_once(conn):
    card = ask(conn, "line_record", team="Brentford", stat="corners", side="match", line=9.5, venue="away")
    assert "Premier League: over 9.5 in 6 of 8 matches." in card["detail"]  # 12, 11, 10, 13 and both 10s


def test_a_referee_without_matches_this_season_says_which_season_it_shows(conn):
    card = ask(conn, "referee", referee="Other Ref", stat="cards")
    assert card["scope"] == "Other Ref · All competitions · 2025/26 · 1 match"
    assert card["detail"][0] == "Other Ref has no matches in 2026/27 in Spix's data, so this is 2025/26."


def test_a_choice_missing_its_stat_or_name_is_refused_not_raised(conn):
    for qtype, params in (("team_average", {"team": "Arsenal"}), ("line_record", {"stat": "corners", "line": 9.5}),
                          ("player", {"stat": "goals"}), ("leaders", {"stat": "corners", "competition": "EPL"})):
        card = questions.answer(conn, qtype, params)
        assert card["type"] == "invalid" and "missing" in card["headline"], (qtype, card["headline"])
    assert questions.answer(conn, "team_average", {"team": "Arsenal"})["headline"] == "The stat is missing."


def test_a_single_count_is_worded_in_the_singular(conn, tmp_path):
    db = sqlite3.connect(tmp_path / "stats.db")
    db.execute("UPDATE player_matches SET goals = CASE fixture_id WHEN 103 THEN 1 ELSE 0 END")
    db.commit(); db.close()
    assert ask(conn, "player", player="Saka", stat="goals")["headline"].startswith(
        "Bukayo Saka: 1 goal in 4 matches")  # the owner's "Madueke: 1 goals in 7 matches", 10 Nov
    assert ask(conn, "player", player="Saka", stat="shots_on_target", last=1)["headline"].startswith(
        "Bukayo Saka: 2 shots on target in 1 match,")
