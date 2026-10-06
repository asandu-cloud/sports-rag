"""Schedule visibility must not create predictions or reveal betting/outcome data."""
from datetime import datetime, timezone

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import event, text


def _seed(session_factory):
    from data_platform.models import Competition, Fixture, Season, Team
    with session_factory() as session:
        home = Team(api_football_id=101, name='Home', logo_url='https://example.invalid/home.png')
        away = Team(api_football_id=102, name='Away')
        session.add_all([home, away])
        session.flush()
        for code, api_id in [('EPL', 39), ('LaLiga', 140), ('Unsupported', 999)]:
            competition = Competition(code=code, name=code, api_football_id=api_id)
            session.add(competition); session.flush()
            season = Season(competition_id=competition.id, year=2026, label='2026/27')
            session.add(season); session.flush()
            for day, hour, status in [(10, 11, 'NS'), (10, 17, 'NS'), (10, 19, 'PST'), (11, 12, 'NS')]:
                session.add(Fixture(api_football_id=api_id * 10000 + day * 100 + hour,
                    competition_id=competition.id, season_id=season.id,
                    home_team_id=home.id, away_team_id=away.id,
                    kickoff_utc=datetime(2026, 10, day, hour, tzinfo=timezone.utc), status=status,
                    home_goals=99, away_goals=99))  # Sentinel outcomes: must never be selected.


def _client(monkeypatch, session_factory):
    from data_platform.repositories.fixture_schedule import FixtureScheduleRepository
    from web_app.routers import match_reads
    monkeypatch.setattr(match_reads, '_utc_now', lambda: datetime(2026, 10, 6, 12, tzinfo=timezone.utc))
    monkeypatch.setattr(match_reads, '_get_fixture_schedule_repository',
                        lambda: FixtureScheduleRepository(session_factory))
    def forbid():
        raise AssertionError('Schedule requests must not consult or create Match Reads')
    monkeypatch.setattr(match_reads, '_get_match_read_service', forbid)
    app = FastAPI(); app.include_router(match_reads.router)
    return TestClient(app)


def test_schedule_includes_unanalysed_late_fixtures_without_writes_or_outcomes(
    settings, engine, session_factory, monkeypatch,
):
    _seed(session_factory)
    statements = []
    with session_factory() as session:
        query_engine = session.get_bind()
    def capture(connection, cursor, statement, parameters, context, executemany):
        statements.append(statement)
    event.listen(query_engine, 'before_cursor_execute', capture)
    try:
        response = _client(monkeypatch, session_factory).get('/api/match-reads/schedule/2026-10-10')
    finally:
        event.remove(query_engine, 'before_cursor_execute', capture)
    assert response.status_code == 200
    body = response.json()
    assert body['schema_version'] == 'fixture-schedule.v1'
    assert body['count'] == 6
    assert {r['fixture']['league'] for r in body['fixtures']} == {'EPL', 'LaLiga'}
    assert {r['status'] for r in body['fixtures']} == {'NS', 'PST'}
    assert any('T17:00:00+00:00' in r['fixture']['kickoff'] for r in body['fixtures'])
    assert all(set(row) == {'fixture', 'status', 'visuals'} for row in body['fixtures'])
    assert all(set(row['fixture']) == {'event_id', 'league', 'home_team', 'away_team', 'kickoff'} for row in body['fixtures'])
    assert statements and all(s.lstrip().upper().startswith('SELECT') for s in statements)
    assert all('home_goals' not in s and 'away_goals' not in s for s in statements)
    with session_factory() as session:
        assert session.execute(text('select count(*) from match_reads')).scalar_one() == 0
        assert session.execute(text('select count(*) from match_read_deliveries')).scalar_one() == 0


def test_schedule_filter_and_empty_day(settings, engine, session_factory, monkeypatch):
    _seed(session_factory)
    client = _client(monkeypatch, session_factory)
    body = client.get('/api/match-reads/schedule/2026-10-10?league=EPL').json()
    assert body['count'] == 3
    assert all(r['fixture']['league'] == 'EPL' for r in body['fixtures'])
    assert client.get('/api/match-reads/schedule/2026-10-12').json()['fixtures'] == []


@pytest.mark.parametrize('path', [
    '2025-10-10', '2026-10-14', '2026-10-10?league=Unsupported', 'not-a-date',
])
def test_schedule_rejects_out_of_scope_requests_before_reading_store(monkeypatch, path):
    from web_app.routers import match_reads
    monkeypatch.setattr(match_reads, '_utc_now', lambda: datetime(2026, 10, 6, tzinfo=timezone.utc))
    def forbid():
        raise AssertionError('Invalid requests must not open the store')
    monkeypatch.setattr(match_reads, '_get_fixture_schedule_repository', forbid)
    app = FastAPI(); app.include_router(match_reads.router)
    assert TestClient(app).get('/api/match-reads/schedule/' + path).status_code == 400


def test_schedule_failure_is_explicit_not_an_empty_slate(monkeypatch):
    from web_app.routers import match_reads
    monkeypatch.setattr(match_reads, '_utc_now', lambda: datetime(2026, 10, 6, tzinfo=timezone.utc))
    def unavailable():
        raise RuntimeError('Test database unavailable')
    monkeypatch.setattr(match_reads, '_get_fixture_schedule_repository', unavailable)
    app = FastAPI(); app.include_router(match_reads.router)
    response = TestClient(app).get('/api/match-reads/schedule/2026-10-10')
    assert response.status_code == 503
    assert 'temporarily unavailable' in response.json()['detail']
