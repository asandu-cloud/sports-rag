"""One match's research view (D110, the overhaul's B2): each team's last 10, booked, shooters and what it allows."""
from datetime import datetime, timedelta, timezone

import pytest

from data_platform.models import Competition, Fixture, Season, Team
from data_platform.repositories.research import ResearchRepository
from data_platform.services.research_match import ResearchMatchService

NOW = datetime(2026, 8, 1, 12, 0, tzinfo=timezone.utc)


def service(session_factory):
    return ResearchMatchService(ResearchRepository(session_factory), load_read=lambda event_id: None,
                                now=lambda: NOW, outlook_hours=96)


SQUAD = (("Sanchez", "G"), ("James", "D"), ("Colwill", "D"), ("Cucurella", "D"), ("Caicedo", "M"), ("Enzo", "M"),
         ("Palmer", "M"), ("Neto", "F"), ("Jackson", "F"), ("Nkunku", "F"), ("George", "F"))
YELLOWS = {"Cucurella": {1, 4, 7}, "Caicedo": {0, 3, 6, 9}, "James": {2, 8}, "Enzo": {3, 10}, "Palmer": {11},
           "Colwill": {5}}


@pytest.fixture()
def history(session_factory):
    """Chelsea's 12 finished matches in two competitions (an opponent's statistics row missing in the 11th), Leeds'
    3, Chelsea's squad, last season's cards, and Chelsea v Leeds still to play."""
    from data_platform.models import FixturePlayerStats, FixtureTeamStats, Player
    start = datetime(2026, 8, 1, 15, tzinfo=timezone.utc)
    with session_factory() as s:
        epl = Competition(code="EPL", name="Premier League", api_football_id=39)
        ucl = Competition(code="UCL", name="UEFA Champions League", api_football_id=2,
                          competition_type="continental_cup")
        s.add_all([epl, ucl]); s.flush()
        seasons = {key: Season(competition_id=comp.id, year=year, label=f"{year}/{str(year + 1)[-2:]}")
                   for key, comp, year in (("EPL25", epl, 2025), ("EPL", epl, 2026), ("UCL", ucl, 2026))}
        teams = {n: Team(name=n, api_football_id=500 + i, short_code=n[:3].upper())
                 for i, n in enumerate(("Chelsea", "Leeds", "Arsenal", "Spurs", "Everton", "Napoli"))}
        players = {n: Player(name=n, api_football_id=700 + i) for i, (n, _) in enumerate(SQUAD)}
        s.add_all([*seasons.values(), *teams.values(), *players.values()]); s.flush()
        comps = {"EPL25": epl, "EPL": epl, "UCL": ucl}

        def play(api_id, key, day, home, away, score, stats=None, status="FT"):
            rnd = "League Stage - 1" if key == "UCL" else "Regular Season - 1"
            f = Fixture(api_football_id=api_id, competition_id=comps[key].id, season_id=seasons[key].id,
                        home_team_id=teams[home].id, away_team_id=teams[away].id, round=rnd, status=status,
                        kickoff_utc=start + timedelta(days=day), home_goals=score[0] if score else None,
                        away_goals=score[1] if score else None)
            s.add(f); s.flush()
            for name, values in (stats or {}).items():
                other = away if name == home else home
                s.add(FixtureTeamStats(fixture_id=f.id, team_id=teams[name].id, opponent_team_id=teams[other].id,
                                       is_home=name == home, **values))
            return f

        rivals = ("Arsenal", "Spurs", "Everton", "Napoli")
        for i in range(12):
            key, rival = ("UCL", "Napoli") if i % 4 == 3 else ("EPL", rivals[i % 4])
            home, away = ("Chelsea", rival) if i % 2 == 0 else (rival, "Chelsea")
            stats = {"Chelsea": dict(yellow_cards=i % 3, red_cards=1 if i == 5 else 0, corners=5 + i % 4, shots_on=4)}
            if i != 10:  # the 11th match lacks the opponent's row: both sides' statistics are unknown
                stats[rival] = dict(yellow_cards=2, red_cards=0, corners=3, shots_on=i % 5)
            f = play(3000 + i, key, i * 4, home, away, (2, 1) if home == "Chelsea" else (0, 1), stats)
            for n, position in SQUAD:
                if n == "Cucurella" and i == 0:
                    continue
                starts = (n not in ("Jackson", "Nkunku", "George") or (n == "Jackson" and i <= 8)
                          or (n == "Nkunku" and i >= 9))
                s.add(FixturePlayerStats(fixture_id=f.id, player_id=players[n].id, team_id=teams["Chelsea"].id,
                                         position=position, minutes=90 if starts else 15, substitute=not starts,
                                         shots_on=(i + len(n)) % 3, yellow_cards=int(i in YELLOWS.get(n, ())),
                                         red_cards=int(n == "Cucurella" and i == 5)))
        old = play(2900, "EPL25", -300, "Chelsea", "Arsenal", (1, 1),
                   {"Chelsea": dict(shots_on=3), "Arsenal": dict(shots_on=3)})
        s.add(FixturePlayerStats(fixture_id=old.id, player_id=players["Palmer"].id, team_id=teams["Chelsea"].id,
                                 position="M", minutes=90, substitute=False, yellow_cards=1, red_cards=1))
        for j, day in enumerate((6, 18, 30)):
            play(3100 + j, "EPL", day, "Leeds", "Everton", (1, 1),
                 {"Leeds": dict(shots_on=2, yellow_cards=1, corners=4), "Everton": dict(shots_on=6, corners=6)})
        play(3200, "EPL", 60, "Chelsea", "Leeds", None, status="NS")
    return session_factory


def test_last_matches_are_ten_finished_in_every_competition_newest_last(history):
    teams = service(history).match("3200")["teams"]
    chelsea, leeds = teams["home"], teams["away"]
    assert chelsea["team"]["name"] == "Chelsea"
    last = chelsea["last_matches"]
    assert [m["event_id"] for m in last] == [str(3000 + i) for i in range(2, 12)]
    assert {m["competition"]["code"] for m in last} == {"EPL", "UCL"}
    third = last[1]  # i = 3, at Napoli in the Champions League
    assert third["venue"] == "away" and third["opponent"]["name"] == "Napoli"
    assert third["score"] == {"home": 0, "away": 1}
    assert (third["goals_for"], third["goals_against"]) == (1, 0)
    assert (third["cards_for"], third["cards_against"]) == (0, 2)
    assert (third["corners_for"], third["corners_against"], third["sot_for"], third["sot_against"]) == (8, 3, 4, 3)
    unknown = last[8]  # i = 10: a missing row leaves every statistic unknown, never 0
    assert unknown["cards_for"] is None and unknown["sot_against"] is None and unknown["goals_for"] == 2
    assert len(leeds["last_matches"]) == 3  # fewer than 10: never padded


def test_booked_is_the_top_five_by_cards_this_season(history):
    booked = service(history).match("3200")["teams"]["home"]["booked"]
    assert [p["name"] for p in booked] == ["Cucurella", "Caicedo", "Enzo", "James", "Colwill"]
    assert booked[0] == {"player_id": booked[0]["player_id"], "name": "Cucurella", "yellow": 3, "red": 1,
                         "appearances": 11}
    assert service(history).match("3200")["teams"]["away"]["booked"] == []  # Leeds: no player rows


def test_shooters_are_up_to_eight_with_most_starts_in_the_last_five(history):
    shooters = service(history).match("3200")["teams"]["home"]["shooters"]
    assert len(shooters) == 8
    names = [p["name"] for p in shooters]
    assert "Sanchez" not in names  # no goalkeepers
    assert names[-1] == "Nkunku" and "Jackson" not in names  # 3 starts in the last 5 against 2
    assert shooters[-1]["starts_last_5"] == 3 and shooters[0]["starts_last_5"] == 5
    palmer = next(p for p in shooters if p["name"] == "Palmer")
    assert palmer["position"] == "M"
    assert [a["date"][:10] for a in palmer["last_10"]] == [
        (datetime(2026, 8, 1) + timedelta(days=i * 4)).date().isoformat() for i in range(2, 12)]
    assert palmer["last_10"][-1] == {"date": palmer["last_10"][-1]["date"], "opponent": "Napoli", "minutes": 90,
                                     "sot": (11 + 6) % 3, "started": True}
    assert "price_sot_1plus" not in palmer


def test_allowed_is_sot_conceded_a_match_and_its_league_rank(history):
    teams = service(history).match("3200")["teams"]
    assert teams["home"]["allowed"] == {"competition": {"code": "EPL", "name": "Premier League"},
                                        "per_match": 1.9, "matches": 8, "rank": 1, "peer_count": 5}
    assert teams["away"]["allowed"]["per_match"] == 6.0 and teams["away"]["allowed"]["rank"] == 5
