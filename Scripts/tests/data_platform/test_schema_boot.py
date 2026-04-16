"""Schema boot + basic round-trips.

These are the ground-truth migration tests: the tables exist and the
ORM can insert/select canonical rows.
"""

from __future__ import annotations

from datetime import date, datetime, timezone

from data_platform.models import (
    AppUser,
    Competition,
    Fixture,
    FixturePlayerStats,
    FixtureTeamStats,
    Player,
    Season,
    SyncRun,
    SyncWatermark,
    Team,
    TeamFeatureSnapshot,
)


def test_tables_exist_and_basic_crud(session_factory):
    with session_factory() as s:
        c = Competition(code="EPL", name="Premier League", api_football_id=39,
                        competition_type="domestic_league")
        s.add(c)
        s.flush()
        season = Season(competition_id=c.id, year=2025, label="2025/26", is_current=True)
        s.add(season)
        s.flush()
        home = Team(api_football_id=100, name="Home FC")
        away = Team(api_football_id=200, name="Away United")
        s.add_all([home, away])
        s.flush()
        fx = Fixture(
            api_football_id=999001,
            competition_id=c.id, season_id=season.id,
            home_team_id=home.id, away_team_id=away.id,
            kickoff_utc=datetime(2025, 8, 10, 14, 0, tzinfo=timezone.utc),
            status="FT", home_goals=2, away_goals=1,
        )
        s.add(fx)
        s.flush()
        s.add(FixtureTeamStats(
            fixture_id=fx.id, team_id=home.id, opponent_team_id=away.id,
            is_home=True, goals=2, shots_total=12, shots_on=6, corners=5,
            possession=0.55, yellow_cards=1,
        ))
        player = Player(api_football_id=500, name="Some Player", position="F")
        s.add(player)
        s.flush()
        s.add(FixturePlayerStats(
            fixture_id=fx.id, player_id=player.id, team_id=home.id,
            minutes=90, goals=1, assists=0, shots_total=3, shots_on=2,
        ))
        s.add(SyncRun(run_kind="bootstrap", scope="EPL:2025",
                      started_at=datetime.now(timezone.utc), status="completed",
                      stats={"fixtures_seen": 1}))
        s.add(SyncWatermark(scope="fixtures:EPL:2025",
                            last_successful_at=datetime.now(timezone.utc),
                            last_cursor="2025-08-10"))
        s.add(TeamFeatureSnapshot(
            team_id=home.id, season_id=season.id, as_of_date=date(2025, 8, 10),
            matches_played=1, goals_for_pm=2.0, corners_pm=5.0,
            cards_per_90_team=0.5, control_index=0.6,
        ))
        s.add(AppUser(discord_id="dev-1", discord_username="devuser",
                      tier="free", status="active"))

    # Second session: ensure the rows actually landed.
    with session_factory() as s:
        assert s.query(Competition).count() == 1
        assert s.query(Season).first().label == "2025/26"
        assert s.query(Fixture).first().home_goals == 2
        assert s.query(FixtureTeamStats).first().goals == 2
        assert s.query(FixturePlayerStats).first().minutes == 90
        snap = s.query(TeamFeatureSnapshot).first()
        assert snap is not None
        assert float(snap.goals_for_pm) == 2.0
        assert float(snap.control_index) == 0.6
