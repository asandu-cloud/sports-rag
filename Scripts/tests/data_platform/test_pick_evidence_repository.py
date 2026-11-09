"""Pick evidence (D82) reads this season's finished league matches, their corners and their final pre-kickoff 1X2
prices from the platform database. Read only."""
from datetime import datetime, timezone


def _book(name, home, draw, away, bet_id=1):
    return {"id": hash(name) % 1000, "name": name, "bets": [{"id": bet_id, "name": "Match Winner", "values": [
        {"value": "Home", "odd": str(home)}, {"value": "Draw", "odd": str(draw)}, {"value": "Away", "odd": str(away)}]}]}


def _payload(*books):
    return {"get": "odds", "response": [{"fixture": {"id": 1}, "bookmakers": list(books)}]}


def _seed(session_factory):
    from data_platform.models import Competition, Fixture, FixtureTeamStats, OddsSnapshot, Season, Team
    at = lambda day: datetime(2026, 10, day, 15, tzinfo=timezone.utc)
    with session_factory() as session:
        a, b, c = (Team(api_football_id=n, name=name) for n, name in ((40, "Liverpool"), (52, "Crystal Palace"),
                                                                       (65, "Nottingham Forest")))
        session.add_all([a, b, c])
        league = Competition(code="EPL", name="Premier League", api_football_id=39)
        cup = Competition(code="UCL", name="Champions League", api_football_id=2)
        session.add_all([league, cup]); session.flush()
        now, last, cup_now = (Season(competition_id=league.id, year=2026, label="2026/27"),
                              Season(competition_id=league.id, year=2025, label="2025/26"),
                              Season(competition_id=cup.id, year=2026, label="2026/27"))
        session.add_all([now, last, cup_now]); session.flush()

        def fixture(api_id, home, away, day, status="FT", goals=(1, 0), competition=league, season=now):
            row = Fixture(api_football_id=api_id, competition_id=competition.id, season_id=season.id,
                          home_team_id=home.id, away_team_id=away.id, kickoff_utc=at(day), status=status,
                          home_goals=goals[0], away_goals=goals[1])
            session.add(row); session.flush()
            return row

        played = fixture(1001, a, b, 10, goals=(2, 1))
        fixture(1002, b, c, 11, goals=(0, 0))                          # no corners
        fixture(1003, a, c, 30, status="NS", goals=(None, None))        # not played yet
        fixture(1004, a, c, 5, competition=cup, season=cup_now)         # a cup match
        fixture(1005, a, b, 12, season=last)                            # last season
        session.add_all([
            FixtureTeamStats(fixture_id=played.id, team_id=a.id, opponent_team_id=b.id, is_home=True, corners=7),
            FixtureTeamStats(fixture_id=played.id, team_id=b.id, opponent_team_id=a.id, is_home=False, corners=3),
        ])
        snap = lambda event, captured, payload: OddsSnapshot(
            event_id=event, league="EPL", kind="final_prematch", captured_at=captured, odds_json=payload)
        session.add_all([
            snap("1001", datetime(2026, 10, 10, 13, tzinfo=timezone.utc),
                 _payload(_book("A", 9.0, 9.0, 9.0))),                               # an older snapshot
            snap("1001", datetime(2026, 10, 10, 14, tzinfo=timezone.utc),
                 _payload(_book("A", 1.50, 4.20, 6.00), _book("B", 1.55, 4.00, 5.50), _book("C", 1.52, 4.10, 5.80))),
            snap("1002", datetime(2026, 10, 11, 14, tzinfo=timezone.utc),
                 _payload(_book("A", 2.10, 3.30, 3.60))),                            # one book only
            OddsSnapshot(event_id="1003", league="EPL", kind="closing", captured_at=at(30),
                         odds_json=_payload(_book("A", 1.5, 4, 6), _book("B", 1.5, 4, 6))),  # not a final snapshot
        ])


def test_finished_league_matches_of_seasons_after_the_history_with_corners_when_both_teams_have_them(
        settings, engine, session_factory):
    from data_platform.repositories.pick_evidence import PickEvidenceRepository

    _seed(session_factory)
    rows = PickEvidenceRepository(session_factory).finished(after_season=2025, competitions=("EPL",))
    assert sorted(rows, key=lambda r: r["event_id"]) == [
        {"event_id": "1001", "date": "2026-10-10", "season": "2026/27", "competition": "EPL", "home_id": 40,
         "away_id": 52, "home": "Liverpool", "away": "Crystal Palace", "home_goals": 2, "away_goals": 1,
         "corners": 10},
        {"event_id": "1002", "date": "2026-10-11", "season": "2026/27", "competition": "EPL", "home_id": 52,
         "away_id": 65, "home": "Crystal Palace", "away": "Nottingham Forest", "home_goals": 0, "away_goals": 0,
         "corners": None}]


def test_prices_are_the_medians_of_the_latest_final_snapshot_with_two_books_or_more(
        settings, engine, session_factory):
    from data_platform.repositories.pick_evidence import PickEvidenceRepository

    _seed(session_factory)
    prices = PickEvidenceRepository(session_factory).prices(["1001", "1002", "1003", "9999"])
    assert prices == {"1001": {"home": 1.52, "draw": 4.10, "away": 5.80}}


def test_teams_and_names_by_api_football_id(settings, engine, session_factory):
    from data_platform.repositories.pick_evidence import PickEvidenceRepository

    _seed(session_factory)
    repository = PickEvidenceRepository(session_factory)
    assert repository.teams("1003") == (40, 65)
    assert repository.teams("424242") is None and repository.teams("not-a-number") is None
    assert repository.names([40, 65, 777]) == {40: "Liverpool", 65: "Nottingham Forest"}
