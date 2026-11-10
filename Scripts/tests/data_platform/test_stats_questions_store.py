"""The stats questions prototype's stats file (tracker D92): built from the platform database, never asked directly."""
import sqlite3
from contextlib import nullcontext
from datetime import datetime, timedelta, timezone

import pytest

from data_platform.models import (Competition, Fixture, FixturePlayerStats, FixtureTeamStats, Player, Season,
                                  Team)


@pytest.fixture()
def platform(session_factory):
    with session_factory() as s:
        league = Competition(code="EPL", name="Premier League", api_football_id=39, competition_type="domestic_league",
                             country="England")
        cup = Competition(code="UCL", name="UEFA Champions League", api_football_id=2,
                          competition_type="continental_cup")
        s.add_all([league, cup]); s.flush()
        epl, ucl = (Season(competition_id=league.id, year=2026, label="2026/27"),
                    Season(competition_id=cup.id, year=2026, label="2026/27"))
        alpha, beta = Team(name="Alpha", api_football_id=1, country="England"), Team(name="Beta", api_football_id=2)
        saka, sub = Player(name="B. Saka", api_football_id=10), Player(name="Unused Sub", api_football_id=11)
        s.add_all([epl, ucl, alpha, beta, saka, sub]); s.flush()
        start = datetime(2026, 8, 15, 15, tzinfo=timezone.utc)
        played = [  # season, home, away, score, status, referee
            (epl, alpha, beta, (2, 1), "FT", "A. Taylor"),
            (epl, beta, alpha, (0, 0), "FT", "Anthony Taylor, England"),
            (ucl, alpha, beta, (1, 1), "FT", "New Ref"),
            (epl, alpha, beta, (None, None), "NS", None),
        ]
        for i, (season, home, away, (hg, ag), status, referee) in enumerate(played):
            f = Fixture(api_football_id=100 + i, competition_id=season.competition_id, season_id=season.id,
                        home_team_id=home.id, away_team_id=away.id, kickoff_utc=start + timedelta(days=7 * i),
                        status=status, home_goals=hg, away_goals=ag, round="Regular Season - 1"
                        if season is epl else "League Stage - 1", referee=referee)
            s.add(f); s.flush()
            if status != "FT":
                continue
            for team, opp, is_home in ((home, away, True), (away, home, False)):
                s.add(FixtureTeamStats(fixture_id=f.id, team_id=team.id, opponent_team_id=opp.id, is_home=is_home,
                                       goals=hg if is_home else ag, corners=7 if is_home else 3,
                                       shots_total=14 if is_home else 6, shots_on=5 if is_home else 2,
                                       expected_goals=1.6 if is_home else 0.4, possession=.62 if is_home else .38,
                                       yellow_cards=2 if is_home else 1, red_cards=1 if is_home else 0,
                                       fouls_committed=11 if is_home else 9))
            s.add(FixturePlayerStats(fixture_id=f.id, player_id=saka.id, team_id=alpha.id, minutes=90,
                                     substitute=False, position="F", goals=1, assists=0, shots_total=3, shots_on=2,
                                     yellow_cards=0, red_cards=0, fouls_committed=1, fouls_drawn=2, rating=7.4))
            s.add(FixturePlayerStats(fixture_id=f.id, player_id=sub.id, team_id=alpha.id, minutes=0,
                                     substitute=True, position="M"))
        return {"alpha": alpha.id, "beta": beta.id, "saka": saka.id}


def _build(tmp_path, session_factory):
    from Scripts.ops.stats_questions import store
    target = tmp_path / "stats.db"
    target.write_text("yesterday's file")
    report = store.build(target, session_factory=session_factory, gate=nullcontext)
    return store, target, report


def test_the_build_keeps_finished_matches_with_each_side_and_its_opponent_side_by_side(tmp_path, session_factory,
                                                                                       platform):
    store, target, report = _build(tmp_path, session_factory)
    assert report["matches"] == 3 and report["team_matches"] == 6 and report["player_matches"] == 3
    conn = store.connect(target)
    beta_away = dict(conn.execute(
        "SELECT * FROM team_matches WHERE team_id = ? AND is_home = 0 AND competition = 'EPL'",
        (platform["beta"],)).fetchone())
    assert (beta_away["goals_for"], beta_away["goals_against"]) == (1, 2)
    assert (beta_away["corners_for"], beta_away["corners_against"]) == (3, 7)
    assert (beta_away["cards_for"], beta_away["cards_against"]) == (1, 3)  # yellows plus reds, as Research counts
    assert beta_away["possession_for"] == pytest.approx(38.0) and beta_away["season"] == 2026
    assert beta_away["opponent_id"] == platform["alpha"]


def test_referees_share_the_research_areas_keys_and_unknown_names_keep_their_own(tmp_path, session_factory, platform):
    store, target, _ = _build(tmp_path, session_factory)
    conn = store.connect(target)
    keys = [r[0] for r in conn.execute("SELECT referee_key FROM matches ORDER BY kickoff")]
    assert keys[0] == keys[1] and keys[2] != keys[0]  # "A. Taylor" and "Anthony Taylor, England" are one referee
    names = dict(conn.execute("SELECT referee_key, name FROM referees").fetchall())
    assert names[keys[0]] == "Anthony Taylor" and names[keys[2]] == "New Ref"


def test_teams_get_their_domestic_league_and_players_their_latest_club(tmp_path, session_factory, platform):
    store, target, _ = _build(tmp_path, session_factory)
    conn = store.connect(target)
    assert conn.execute("SELECT league FROM teams WHERE id = ?", (platform["alpha"],)).fetchone()[0] == "EPL"
    player = dict(conn.execute("SELECT * FROM players WHERE id = ?", (platform["saka"],)).fetchone())
    assert player["name"] == "B. Saka" and player["team_id"] == platform["alpha"]
    assert conn.execute("SELECT count(*) FROM player_matches WHERE minutes = 0").fetchone()[0] == 0
    started = [r[0] for r in conn.execute("SELECT DISTINCT started FROM player_matches")]
    assert started == [1]


def test_the_new_file_replaces_the_old_one_whole_and_opens_read_only(tmp_path, session_factory, platform):
    store, target, report = _build(tmp_path, session_factory)
    assert not (tmp_path / "stats.db.tmp").exists()
    conn = store.connect(target)
    assert conn.execute("SELECT value FROM build WHERE key = 'data_up_to'").fetchone()[0].startswith("2026-08-29")
    with pytest.raises(sqlite3.OperationalError):
        conn.execute("DELETE FROM matches")
