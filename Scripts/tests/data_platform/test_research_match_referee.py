"""One match's research view (D110, the overhaul's B2): the referee, head to head and the data basis."""
from datetime import datetime, timedelta, timezone

import pytest

from data_platform.models import (Competition, Fixture, FixtureTeamStats, ResearchRefereeAlias, Season, Team)
from data_platform.repositories.research import ResearchRepository
from data_platform.services.research_match import ResearchMatchService
from data_platform.services.research_profiles import referee_aliases

NOW = datetime(2026, 11, 16, 12, 0, tzinfo=timezone.utc)
OLIVER = "Michael Oliver"


def service(session_factory):
    return ResearchMatchService(ResearchRepository(session_factory), load_read=lambda event_id: None,
                                now=lambda: NOW, outlook_hours=96)


@pytest.fixture()
def meetings(session_factory):
    """Chelsea v Leeds on 21 Nov with Michael Oliver; his matches under two spellings, one before this season;
    other EPL matches for the league's average; seven meetings since 2023 and one before."""
    with session_factory() as s:
        epl = Competition(code="EPL", name="Premier League", api_football_id=39)
        ucl = Competition(code="UCL", name="UEFA Champions League", api_football_id=2,
                          competition_type="continental_cup")
        s.add_all([epl, ucl]); s.flush()
        seasons = {(code, year): Season(competition_id=comp.id, year=year, label=f"{year}/{str(year + 1)[-2:]}")
                   for code, comp in (("EPL", epl), ("UCL", ucl)) for year in range(2022, 2027)}
        teams = {n: Team(name=n, api_football_id=800 + i, short_code=n[:3].upper())
                 for i, n in enumerate(("Chelsea", "Leeds", "Arsenal", "Spurs"))}
        s.add_all([*seasons.values(), *teams.values()]); s.flush()
        api = iter(range(4000, 4100))

        def play(code, year, when, home, away, score=(1, 1), cards=None, corners=(5, 4), referee=None, status="FT"):
            f = Fixture(api_football_id=next(api), competition_id=(epl if code == "EPL" else ucl).id,
                        season_id=seasons[(code, year)].id, home_team_id=teams[home].id,
                        away_team_id=teams[away].id, kickoff_utc=when, status=status, referee=referee,
                        home_goals=score[0] if status == "FT" else None,
                        away_goals=score[1] if status == "FT" else None, round="Regular Season - 1")
            s.add(f); s.flush()
            if cards is not None:
                for name, other, yellow, corner, home_side in ((home, away, cards[0], corners[0], True),
                                                               (away, home, cards[1], corners[1], False)):
                    s.add(FixtureTeamStats(fixture_id=f.id, team_id=teams[name].id, opponent_team_id=teams[other].id,
                                           is_home=home_side, yellow_cards=yellow, red_cards=0, corners=corner,
                                           shots_on=3))
            return f

        day = lambda m, d: datetime(2026, m, d, 15, tzinfo=timezone.utc)  # noqa: E731
        play("EPL", 2026, day(9, 1), "Arsenal", "Spurs", cards=(2, 2), referee=OLIVER)
        play("EPL", 2026, day(9, 15), "Spurs", "Arsenal", cards=(3, 2), referee=OLIVER)
        play("UCL", 2026, day(10, 1), "Arsenal", "Chelsea", cards=(3, 3), referee=OLIVER)
        play("EPL", 2026, day(10, 20), "Spurs", "Leeds", cards=(1, 2), referee="M. Oliver")  # not in the table
        play("EPL", 2026, day(11, 1), "Leeds", "Arsenal", cards=(0, 1), referee="Anthony Taylor")
        play("EPL", 2025, datetime(2026, 3, 1, 15, tzinfo=timezone.utc), "Arsenal", "Spurs", cards=(5, 5),
             referee=OLIVER)
        for year, month, home, away in ((2022, 10, "Chelsea", "Leeds"), (2023, 9, "Leeds", "Chelsea"),
                                        (2024, 1, "Chelsea", "Leeds"), (2024, 9, "Leeds", "Chelsea"),
                                        (2025, 2, "Chelsea", "Leeds"), (2025, 9, "Leeds", "Chelsea"),
                                        (2026, 1, "Chelsea", "Leeds"), (2026, 9, "Leeds", "Chelsea")):
            season = year if month >= 7 else year - 1
            play("EPL", season, datetime(year, month, 10, 15, tzinfo=timezone.utc), home, away, score=(2, 1),
                 cards=(2, 3), corners=(6, 4))
        match = play("EPL", 2026, day(11, 21), "Chelsea", "Leeds", referee=OLIVER, status="NS")
        key = referee_aliases([OLIVER])[0]["referee_key"]
        s.add(ResearchRefereeAlias(raw_name=OLIVER, name=OLIVER, referee_key=key))
        no_referee = play("EPL", 2026, day(11, 22), "Arsenal", "Spurs", status="NS")
        ids = {"match": str(match.api_football_id), "no_referee": str(no_referee.api_football_id), "key": key}
    return session_factory, ids


def test_referee_this_season_cards_league_average_and_last_ten(meetings):
    session_factory, ids = meetings
    referee = service(session_factory).match(ids["match"])["referee"]
    assert referee["name"] == OLIVER and referee["referee_key"] == ids["key"]
    assert referee["matches"] == 4  # both spellings, every competition, this season only
    assert referee["cards_per_match"] == 4.5  # (4 + 5 + 6 + 3) / 4
    assert referee["league"] == {"code": "EPL", "name": "Premier League"}
    # Every finished EPL 2026/27 match before kickoff with both rows: 4, 5, 3, 1, and the 5 a meeting
    assert referee["league_matches"] == 5 and referee["league_cards_per_match"] == 3.6
    last = referee["last_matches"]
    assert [m["date"][:10] for m in last] == ["2026-03-01", "2026-09-01", "2026-09-15", "2026-10-01", "2026-10-20"]
    assert last[-1] == {"date": last[-1]["date"], "competition": {"code": "EPL", "name": "Premier League"},
                        "home": "Spurs", "away": "Leeds", "cards": 3}


def test_referee_null_until_named(meetings):
    session_factory, ids = meetings
    assert service(session_factory).match(ids["no_referee"])["referee"] is None


def test_h2h_is_up_to_six_meetings_since_2023_newest_first(meetings):
    session_factory, ids = meetings
    h2h = service(session_factory).match(ids["match"])["h2h"]
    assert [m["date"][:7] for m in h2h] == ["2026-09", "2026-01", "2025-09", "2025-02", "2024-09", "2024-01"]
    assert h2h[0] == {"event_id": h2h[0]["event_id"], "date": h2h[0]["date"],
                      "competition": {"code": "EPL", "name": "Premier League"},
                      "home": {"id": h2h[0]["home"]["id"], "name": "Leeds"},
                      "away": {"id": h2h[0]["away"]["id"], "name": "Chelsea"},
                      "score": {"home": 2, "away": 1}, "goals": 3, "cards": 5, "corners": 10}


def test_data_basis_is_the_latest_finished_kickoff_used(meetings):
    session_factory, ids = meetings
    basis = service(session_factory).match(ids["match"])["data_basis"]
    assert basis["latest_finished_kickoff"] == "2026-11-01T15:00:00+00:00"  # Leeds v Arsenal, Leeds' last match
    assert basis["generated_at"] == NOW.isoformat()
