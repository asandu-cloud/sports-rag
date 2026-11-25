"""D136: since August 2026 API-Football's games.substitute says False for most players who came on, so who started
comes from the match's substitution events."""
from datetime import date

from sqlalchemy import select

from data_platform.models import FixturePlayerStats, Player
from data_platform.sync.upserts import (mark_substitutes, upsert_competition, upsert_fixture_events,
                                        upsert_fixture_from_api_row, upsert_fixture_player_stats, upsert_season)
from Scripts.tests.data_platform.test_sync_upserts import SPEC, _api_fixture_row, _stats_response

# api id: (minutes, the provider's broken flag)
PLAYERS = {9001: (90, False), 9004: (70, False), 9002: (20, False), 9003: (None, False), 9005: (90, True)}


def players_response():
    return [{"team": {"id": 100, "name": "Home FC"}, "players": [
        {"player": {"id": pid, "name": f"P{pid}"},
         "statistics": [{"games": {"minutes": minutes, "position": "M", "substitute": flag}}]}
        for pid, (minutes, flag) in PLAYERS.items()]}]


def events_response():
    """9004 off, 9002 on at 70 minutes (API-Football: ``player`` goes off, ``assist`` comes on), and a card."""
    return [{"time": {"elapsed": 12, "extra": None}, "team": {"id": 100}, "player": {"id": 9001, "name": "x"},
             "assist": {"id": None}, "type": "Card", "detail": "Yellow Card", "comments": None},
            {"time": {"elapsed": 70, "extra": None}, "team": {"id": 100}, "player": {"id": 9004, "name": "x"},
             "assist": {"id": 9002, "name": "y"}, "type": "subst", "detail": "Substitution 1", "comments": None}]


def flags(session, fixture):
    rows = session.execute(select(Player.api_football_id, FixturePlayerStats.substitute, FixturePlayerStats.stats_json)
                           .join(Player, Player.id == FixturePlayerStats.player_id)
                           .where(FixturePlayerStats.fixture_id == fixture.id)).all()
    assert all(r.stats_json["substitute"] == r.substitute for r in rows)
    return {r.api_football_id: r.substitute for r in rows}


def a_match(session):
    comp = upsert_competition(session, SPEC)
    season = upsert_season(session, competition=comp, year=2026)
    session.flush()
    fixture, _ = upsert_fixture_from_api_row(session, api_row=_api_fixture_row(), competition=comp, season=season)
    upsert_fixture_player_stats(session, fixture=fixture, players_response=players_response())
    return fixture


def test_who_came_on_or_never_played_is_a_substitute_and_every_other_player_started(session_factory):
    with session_factory() as s:
        fixture = a_match(s)
        assert mark_substitutes(s, fixture=fixture) == 0  # no events stored: the provider's flag stays
        assert flags(s, fixture)[9002] is False
        upsert_fixture_events(s, fixture=fixture, events_response=events_response())
        assert mark_substitutes(s, fixture=fixture) == 3
        assert flags(s, fixture) == {9001: False, 9004: False, 9002: True, 9003: True, 9005: False}
        assert mark_substitutes(s, fixture=fixture) == 0
        # The same provider reply again changes nothing: the digest is the provider's, so the mark stays
        assert upsert_fixture_player_stats(s, fixture=fixture, players_response=players_response()) == 0
        assert flags(s, fixture)[9002] is True


def test_the_refresh_marks_substitutes_after_saving_a_matchs_events(settings, engine, session_factory):
    from data_platform.sync.pipeline import incremental_refresh

    class Client:
        def fixtures(self, **kwargs):
            return [_api_fixture_row()]

        def fixture_statistics(self, fixture):
            return _stats_response()

        def fixture_players(self, fixture):
            return players_response()

        def fixture_events(self, fixture):
            return events_response()

    with session_factory() as session:
        result = incremental_refresh(session, client=Client(), codes=["EPL"], years=[2025], short_transactions=True)
        assert result["error_count"] == 0
        fixture = session.scalars(select(FixturePlayerStats.fixture_id)).first()
    with session_factory() as session:
        from data_platform.models import Fixture
        assert flags(session, session.get(Fixture, fixture))[9002] is True


def test_remark_fixes_stored_rows_from_stored_events_and_a_dry_run_writes_nothing(settings, engine, session_factory):
    from data_platform.sync.substitutes import remark_substitutes
    with session_factory() as s:
        fixture = a_match(s)
        upsert_fixture_events(s, fixture=fixture, events_response=events_response())
        s.commit()
        fixture_id = fixture.id
    since = date(2025, 1, 1)  # the shared fixture row kicks off on 10 Aug 2025
    assert remark_substitutes(session_factory, since=since, dry_run=True) == {
        "fixtures_with_events": 1, "fixtures_changed": 1, "rows_changed": 3, "dry_run": True}
    with session_factory() as s:
        from data_platform.models import Fixture
        assert flags(s, s.get(Fixture, fixture_id))[9002] is False
    assert remark_substitutes(session_factory, since=since)["rows_changed"] == 3
    with session_factory() as s:
        from data_platform.models import Fixture
        assert flags(s, s.get(Fixture, fixture_id))[9002] is True
    assert remark_substitutes(session_factory, since=since)["rows_changed"] == 0
    assert remark_substitutes(session_factory, since=date(2027, 1, 1))["fixtures_with_events"] == 0


def test_a_substitution_without_a_known_incoming_player_takes_the_one_whose_minutes_fit(session_factory):
    """API-Football sometimes leaves ``assist`` empty, or names a player the match's rows lack: the player who came
    on is then the one whose minutes fit the substitution, within five minutes, never the one who went off."""
    nameless = {"time": {"elapsed": 70, "extra": None}, "team": {"id": 100}, "player": {"id": 9004, "name": "x"},
                "assist": {"id": None}, "type": "subst", "detail": "Substitution 1", "comments": None}
    with session_factory() as s:
        fixture = a_match(s)
        upsert_fixture_events(s, fixture=fixture, events_response=[nameless])
        mark_substitutes(s, fixture=fixture)
        assert flags(s, fixture) == {9001: False, 9004: False, 9002: True, 9003: True, 9005: False}
        late = {**nameless, "time": {"elapsed": 30, "extra": None}, "assist": {"id": 123456}}  # 60 left: nobody fits
        upsert_fixture_events(s, fixture=fixture, events_response=[late])
        mark_substitutes(s, fixture=fixture)
        assert flags(s, fixture)[9002] is False
