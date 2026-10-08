"""Research landing data: weekend matches and leaders, on a private database."""
from datetime import datetime, timedelta, timezone

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from data_platform.models import (Competition, Fixture, FixturePlayerStats, FixtureTeamStats, Player,
                                  ResearchPeerStat, ResearchProfileBuild, ResearchRefereeAlias, Season, Team)
from data_platform.repositories.research import ResearchRepository
from data_platform.services.research_home import (ResearchHomeService, best_scorer, standings,
                                                  team_leader_rows)

NOW = datetime(2026, 10, 8, 12, tzinfo=timezone.utc)


def test_standings_order_points_goal_difference_goals_then_name():
    rows = [dict(home_team_id=1, away_team_id=2, home_goals=2, away_goals=0, home_name="A", away_name="B"),
            dict(home_team_id=3, away_team_id=4, home_goals=1, away_goals=0, home_name="C", away_name="D"),
            dict(home_team_id=2, away_team_id=4, home_goals=1, away_goals=1, home_name="B", away_name="D"),
            dict(home_team_id=1, away_team_id=3, home_goals=None, away_goals=None, home_name="A", away_name="C")]
    table = standings(rows)
    assert [t for t, _ in sorted(table.items(), key=lambda kv: kv[1]["position"])] == [1, 3, 4, 2]
    assert table[1] == {"played": 1, "won": 1, "drawn": 0, "lost": 0, "goals_for": 2, "goals_against": 0,
                        "points": 3, "position": 1, "teams": 4}
    assert table[4]["points"] == table[2]["points"] == 1  # D ahead on goal difference


def test_best_scorer_breaks_ties_on_target_then_fewer_appearances():
    pool = [dict(name="Zed", goals=3, shots_on_target=5, appearances=5),
            dict(name="Amy", goals=3, shots_on_target=7, appearances=6),
            dict(name="Bob", goals=3, shots_on_target=7, appearances=4)]
    assert best_scorer(pool)["name"] == "Bob"


def test_team_leaders_need_minimum_matches_and_sort_conceded_ascending():
    def row(team, gf, ga, corners):
        return dict(team_id=team, home_team_id=team, away_team_id=99, home_goals=gf, away_goals=ga, status="FT",
                    corners=corners, shots_on=3, opponent_corners=2, opponent_shots_on=2, yellow_cards=1, red_cards=None)
    rows = [row(1, 2, 0, 5) for _ in range(3)] + [row(2, 0, 3, None) for _ in range(3)] + [row(3, 9, 0, 9)] * 2
    boards = team_leader_rows(rows)
    assert [e["team_id"] for e in boards["goals_for"]["rows"]] == [1, 2]  # team 3 has two matches only
    assert [e["team_id"] for e in boards["goals_against"]["rows"]] == [1, 2]
    assert boards["corners"]["rows"][1]["value"] == 0  # observed match: missing count is zero
    assert boards["cards"]["rows"][0]["value"] == 1


@pytest.fixture()
def home(session_factory):
    with session_factory() as s:
        epl = Competition(code="EPL", name="Premier League", api_football_id=39, competition_type="domestic_league")
        ucl = Competition(code="UCL", name="Champions League", api_football_id=2, competition_type="continental_cup")
        s.add_all([epl, ucl]); s.flush()
        season = Season(competition_id=epl.id, year=2026, label="2026/27")
        cup_season = Season(competition_id=ucl.id, year=2026, label="2026/27")
        teams = [Team(name=n, api_football_id=100 + i, short_code=n[:3].upper()) for i, n in enumerate(("Alpha", "Beta", "Gamma", "Delta"))]
        players = [Player(name=n, api_football_id=500 + i) for i, n in enumerate(("Striker", "Winger", "Cup hero", "Hacker"))]
        s.add_all([season, cup_season, *teams, *players]); s.flush()
        a, b, c, d = (t.id for t in teams)
        played = [(a, b, 3, 0), (c, d, 1, 1), (a, c, 2, 1), (b, d, 0, 2)]
        for i, (h, w, hg, ag) in enumerate(played):
            f = Fixture(api_football_id=1000 + i, competition_id=epl.id, season_id=season.id, home_team_id=h, away_team_id=w,
                        kickoff_utc=NOW - timedelta(days=20 - i), status="FT", home_goals=hg, away_goals=ag,
                        round="Regular Season - %d" % (i + 1), referee="M. Oliver")
            s.add(f); s.flush()
            for tid, opp, home_side in ((h, w, True), (w, h, False)):
                s.add(FixtureTeamStats(fixture_id=f.id, team_id=tid, opponent_team_id=opp, is_home=home_side,
                                       corners=6 if home_side else 2, shots_on=4, yellow_cards=2, red_cards=None))
            if h == a:
                s.add(FixturePlayerStats(fixture_id=f.id, player_id=players[0].id, team_id=a, minutes=90, goals=hg, shots_on=3))
                s.add(FixturePlayerStats(fixture_id=f.id, player_id=players[1].id, team_id=a, minutes=90, goals=None, assists=2))
            s.add(FixturePlayerStats(fixture_id=f.id, player_id=players[3].id, team_id=w, minutes=90, yellow_cards=1, red_cards=1))
        # A cup goal does not outrank league goals for the club's top scorer.
        cup = Fixture(api_football_id=1100, competition_id=ucl.id, season_id=cup_season.id, home_team_id=a, away_team_id=c,
                      kickoff_utc=NOW - timedelta(days=3), status="FT", home_goals=5, away_goals=0, round="League Stage - 1")
        s.add(cup); s.flush()
        s.add(FixturePlayerStats(fixture_id=cup.id, player_id=players[2].id, team_id=a, minutes=90, goals=5))
        upcoming = [(a, d, 1, "M. Oliver"), (b, c, 2, None), (c, a, 9, "M. Oliver"), (d, b, 3, "Unknown Ref")]
        for i, (h, w, days, ref) in enumerate(upcoming):
            s.add(Fixture(api_football_id=2000 + i, competition_id=epl.id, season_id=season.id, home_team_id=h, away_team_id=w,
                          kickoff_utc=NOW + timedelta(days=days), status="NS", round="Regular Season - 9", referee=ref))
        s.add(ResearchRefereeAlias(raw_name="M. Oliver", referee_key="m-oliver", name="Michael Oliver"))
        built = NOW - timedelta(days=1)
        for year, value in ((2024, 3.0), (2025, 4.5)):
            scope = f"referee:{year}"
            s.add(ResearchProfileBuild(scope_key=scope, season_year=year, build_id="profiles-v3:x", built_at=built))
            s.flush()
            s.add(ResearchPeerStat(scope_key=scope, subject_type="referee", subject_key="m-oliver", season_year=year,
                                   peer_group="all", metric="cards", value=value, percentile=80, rank=3, peer_count=40,
                                   sample_matches=20, build_id="profiles-v3:x", built_at=built))
        ids = {"a": a, "b": b, "c": c, "d": d, "striker": players[0].id, "hacker": players[3].id}
    return ResearchHomeService(ResearchRepository(session_factory), clock=lambda: NOW, cache_seconds=0), ids


def test_players_weekend_window_top_scorers_and_leaders(home):
    service, ids = home
    result = service.home("players", league="EPL")
    fixtures = result["weekend"]["fixtures"]
    assert [f["home"]["id"] for f in fixtures] == [ids["a"], ids["b"], ids["d"]]  # day 9 is outside the week
    first = fixtures[0]
    assert first["home"]["top_scorer"]["name"] == "Striker"  # league goals beat the cup hat-trick
    assert first["home"]["top_scorer"]["goals"] == 5
    assert first["home"]["top_scorer"]["photo_url"].endswith("/500.png")
    assert first["away"]["top_scorer"] is None  # Delta has no scorer yet
    boards = result["leaders"]["boards"]
    assert result["leaders"]["season"] == {"year": 2026, "label": "2026/27"}
    assert [r["name"] for r in boards["goals"]["rows"]] == ["Striker"]
    assert boards["assists"]["rows"][0]["value"] == 4  # two home matches, two assists each
    assert boards["cards"]["rows"][0]["value"] == 4  # yellow plus red in two Delta matches
    assert boards["goals"]["rows"][0]["scope"] == "EPL-2026"
    assert result["data_basis"]["latest_finished_kickoff"].startswith("2026-09-21")


def test_teams_weekend_standings_and_per_match_leaders(home):
    service, ids = home
    result = service.home("teams", league="EPL")
    first = result["weekend"]["fixtures"][0]
    assert first["home"]["standing"]["position"] == 1 and first["home"]["standing"]["points"] == 6
    assert first["away"]["standing"]["points"] == 4
    assert result["leaders"]["boards"]["goals_for"]["rows"] == []  # nobody has three matches yet


def test_referees_weekend_only_named_officials_with_latest_rate(home):
    service, ids = home
    result = service.home("referees", league="EPL")
    fixtures = result["weekend"]["fixtures"]
    assert [f["referee"]["name"] for f in fixtures] == ["Michael Oliver", "Unknown Ref"]
    assert fixtures[0]["referee"]["cards"]["season_label"] == "2025/26"
    assert fixtures[0]["referee"]["cards"]["cards_per_match"] == 4.5
    assert fixtures[1]["referee"]["referee_key"] is None and fixtures[1]["referee"]["cards"] is None
    assert result["leaders"]["boards"] == {}  # fewer than ten ranked officials


def test_league_filter_groups_and_router(home):
    service, _ = home
    assert service.home("players")["league"] == {"key": "top5", "label": "Top five leagues", "codes": ["EPL"]}
    assert service.home("players", league="europe")["league"]["codes"] == ["UCL"]
    with pytest.raises(ValueError):
        service.home("players", league="Nowhere")
    from web_app.routers import research as router_module
    app = FastAPI(); app.include_router(router_module.router)
    original = router_module._get_home_service
    router_module._get_home_service = lambda: service
    try:
        client = TestClient(app)
        assert client.get("/api/research/home", params={"type": "teams", "league": "EPL"}).status_code == 200
        assert client.get("/api/research/home", params={"type": "teams", "league": "Nowhere"}).status_code == 422
        assert client.get("/api/research/home", params={"type": "coaches"}).status_code == 422
    finally:
        router_module._get_home_service = original
