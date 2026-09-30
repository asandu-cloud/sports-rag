"""Offline integration checks for the staged player collector and its quota guard."""
from datetime import datetime, timezone
import hashlib
import json
import sqlite3

import pytest
import requests
from Scripts.ops import player_history_download as job

TEST_SEASONS = (2021, 2022, 2023, 2024, 2025)


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError('No real provider requests in tests')
    monkeypatch.setattr(requests.sessions.Session, 'request', blocked)


def fixture(fid=42, season=2021, code='EPL', status='FT'):
    return {'platform_fixture_id': fid + 1000, 'fixture_id': fid,
            'competition_id': 1, 'code': code, 'competition_name': 'Premier League',
            'league_id': 39, 'country': 'England', 'competition_type': 'domestic_league',
            'season_id': season, 'season': season, 'kickoff_utc': f'{season}-08-01 15:00:00',
            'status': status, 'round': 'Regular Season - 1', 'referee': 'Test Referee',
            'home_id': 10, 'home_team_id': 100, 'home_name': 'Home',
            'away_id': 20, 'away_team_id': 200, 'away_name': 'Away'}


def payload(fid=42):
    return {'get': 'fixtures/players', 'parameters': {'fixture': str(fid)}, 'errors': [],
            'results': 2, 'paging': {'current': 1, 'total': 1}, 'response': [
                {'team': {'id': tid, 'name': name}, 'players': [
                    {'player': {'id': tid * 100 + i, 'name': f'Player {tid}-{i}'}, 'statistics': [{
                        'games': {'minutes': 0 if i == 0 else 1 if i == 1 else 90,
                                  'position': 'M', 'rating': '7.1', 'captain': False, 'substitute': False},
                        'cards': {'yellow': None if i == 2 else 0, 'red': None if i == 3 else 0},
                        'shots': {'total': 2, 'on': 1}, 'passes': {'total': 20, 'key': 3, 'accuracy': 15},
                        'dribbles': {'attempts': 4, 'success': 2, 'past': 1},
                        'goals': {'total': 0, 'assists': None, 'saves': 1, 'conceded': 2},
                        'penalty': {'won': None, 'commited': 0, 'scored': 0, 'missed': 0, 'saved': 0},
                    }]} for i in range(11)]}
                for tid, name in ((100, 'Home'), (200, 'Away'))]}


def lineup_payload(fid=42):
    blocks = []
    for tid in (100, 200):
        players = [{'player': {'id': tid * 100 + i, 'name': f'Player {tid}-{i}',
                               'number': i + 1, 'pos': 'M', 'grid': '2:1'}} for i in range(15)]
        blocks.append({'team': {'id': tid}, 'coach': {'id': tid + 1, 'name': 'Coach'},
                       'formation': '4-3-3', 'startXI': players[:11], 'substitutes': players[11:]})
    return {'get': 'fixtures/lineups', 'parameters': {'fixture': str(fid)}, 'errors': [],
            'results': 2, 'paging': {'current': 1, 'total': 1}, 'response': blocks}


def events_payload(fid=42):
    events = [
        {'time': {'elapsed': 45, 'extra': 2}, 'team': {'id': 100},
         'player': {'id': 10001}, 'assist': {'id': None}, 'type': 'Card', 'detail': 'Yellow Card', 'comments': None},
        {'time': {'elapsed': 67, 'extra': None}, 'team': {'id': 100},
         'player': {'id': 10001}, 'assist': {'id': None}, 'type': 'Card', 'detail': 'Yellow-Red Card', 'comments': None},
        {'time': {'elapsed': 70, 'extra': None}, 'team': {'id': 200},
         'player': {'id': 20002}, 'assist': {'id': 20011}, 'type': 'subst', 'detail': 'Substitution 1', 'comments': None},
    ]
    return {'get': 'fixtures/events', 'parameters': {'fixture': str(fid)}, 'errors': [],
            'results': len(events), 'paging': {'current': 1, 'total': 1}, 'response': events}


class Response:
    def __init__(self, body=None, status=200, headers=None):
        self.body, self.status_code, self.headers = body or payload(), status, headers or {}
        self.closed = False

    def json(self):
        return self.body

    def close(self):
        self.closed = True


class Transport:
    def __init__(self, responses=()):
        self.responses = iter(responses)
        self.calls = []
        self.headers = {}
        self.closed = False

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        response = next(self.responses)
        if isinstance(response, BaseException):
            raise response
        return response

    def close(self):
        self.closed = True


@pytest.fixture()
def source(tmp_path):
    root = tmp_path / 'project'
    directory = root / 'Index/history_staging/source-fixture'
    directory.mkdir(parents=True)
    fixtures = [fixture(42, 2021), fixture(43, 2022), fixture(44, 2025),
                fixture(45, 2026), fixture(46, 2021, status='NS')]
    job.initialize(directory, fixtures, {'test': True})
    database = root / 'Index/platform.db'
    (directory / 'platform.db').replace(database)
    return root, database


def test_read_only_plan_filters_seasons_status_and_has_no_outcome_columns(source):
    root, database = source
    before = hashlib.sha256(database.read_bytes()).hexdigest()
    rows = job.fixture_inventory(database, TEST_SEASONS)
    assert [f['fixture_id'] for f in rows] == [42, 43, 44]
    assert not {'home_goals', 'away_goals', 'minutes', 'yellow_cards'} & rows[0].keys()
    out = root / 'Index/history_staging/new'
    report = job.run(root, database, TEST_SEASONS, out)
    assert report['fixture_count'] == 3
    assert not out.exists()
    assert hashlib.sha256(database.read_bytes()).hexdigest() == before


def test_download_is_linked_archived_unfiltered_and_resumes_without_http(source):
    root, database = source
    directory = root / 'Index/history_staging/download'
    before = hashlib.sha256(database.read_bytes()).hexdigest()
    transport = Transport([Response(payload(42)), Response(payload(43))])
    first = job.run(root, database, TEST_SEASONS, directory, execute=True,
                    key='test', transport=transport, max_fixtures=2)
    assert first['states'] == {'pending': 1, 'ready': 2}
    assert first['linked_player_rows'] == 44
    assert first['foreign_key_errors'] == 0
    assert len(transport.calls) == 2
    assert all(call[0].endswith('/fixtures/players') for call in transport.calls)
    second_transport = Transport([Response(payload(44))])
    second = job.run(root, database, TEST_SEASONS, directory, execute=True,
                    key='test', transport=second_transport)
    assert second['states'] == {'ready': 3}
    assert len(second_transport.calls) == 1
    assert second_transport.calls[0][1]['params'] == {'fixture': 44}
    with sqlite3.connect(directory / 'platform.db') as db:
        values = db.execute('''SELECT f.api_football_id,t.api_football_id,p.api_football_id,x.minutes,
                     x.yellow_cards,x.stats_json FROM fixture_player_stats x
                     JOIN fixtures f ON f.id=x.fixture_id JOIN teams t ON t.id=x.team_id
                     JOIN players p ON p.id=x.player_id WHERE f.api_football_id=42 ORDER BY p.api_football_id''').fetchall()
        assert len(values) == 22
        assert values[0][:4] == (42, 100, 10000, 0)
        assert values[1][3] == 1  # Relevance filtering must never destroy source evidence.
        assert values[2][4] is None
        raw = json.loads(values[0][5])
        assert raw['provider_statistics']['passes']['key'] == 3
        assert raw['provider_statistics']['dribbles']['success'] == 2
        assert raw['source_fixture_id'] == 42
        archive_id = raw['source_archive_id']
        assert db.execute('SELECT count(*) FROM fixture_team_stats').fetchone()[0] == 0
    assert job.read_archive(directory, archive_id) == payload(42)
    assert hashlib.sha256(database.read_bytes()).hexdigest() == before


@pytest.mark.parametrize('mutation', [
    lambda p: p['parameters'].update(fixture='99'),
    lambda p: p['response'][0]['team'].update(id=999),
    lambda p: p['response'][0]['players'].pop(),
    lambda p: p['response'][0]['players'][1]['player'].update(id=10000),
    lambda p: p['response'][0]['players'][0]['player'].update(id=True),
    lambda p: p['response'][0]['players'][0]['statistics'].append({}),
    lambda p: p['response'][0]['players'][0]['statistics'][0]['games'].update(minutes=-1),
    lambda p: p['response'][0]['players'][0]['statistics'][0]['cards'].update(red=0.5),
    lambda p: p['response'][0]['players'][0]['statistics'][0]['games'].update(rating='nan'),
])
def test_invalid_sources_are_archived_but_never_normalized(source, mutation):
    root, database = source
    p = payload()
    mutation(p)
    directory = root / 'Index/history_staging/invalid'
    result = job.run(root, database, TEST_SEASONS, directory, execute=True, key='test',
                     transport=Transport([Response(p)]), max_fixtures=1)
    assert result['states']['invalid'] == 1
    assert result['linked_player_rows'] == 0
    assert job.read_archive(directory, 1) == p


def test_empty_is_not_success_or_zero_and_retry_is_explicit(source):
    root, database = source
    p = payload()
    p.update(results=0, response=[])
    directory = root / 'Index/history_staging/empty'
    result = job.run(root, database, TEST_SEASONS, directory, execute=True, key='test',
                     transport=Transport([Response(p)]), max_fixtures=1)
    assert result['states']['unavailable'] == 1
    assert result['linked_player_rows'] == 0
    second = Transport([Response(payload(43))])
    job.run(root, database, TEST_SEASONS, directory, execute=True, key='test',
            transport=second, max_fixtures=1)
    assert second.calls[0][1]['params']['fixture'] == 43
    third = Transport([Response(payload(42))])
    result = job.run(root, database, TEST_SEASONS, directory, execute=True, key='test',
                     transport=third, max_fixtures=1, retry_unavailable=True)
    assert third.calls[0][1]['params']['fixture'] == 42
    assert result['states']['ready'] == 2


def test_interruption_after_archive_resumes_locally(source, monkeypatch):
    root, database = source
    directory = root / 'Index/history_staging/interrupted'
    normalize = job.normalize_response
    def interrupted(*args):
        raise KeyboardInterrupt()
    monkeypatch.setattr(job, 'normalize_response', interrupted)
    result = job.run(root, database, TEST_SEASONS, directory, execute=True, key='test',
                     transport=Transport([Response()]))
    assert result['states']['downloaded'] == 1
    monkeypatch.setattr(job, 'normalize_response', normalize)
    no_http = Transport()
    result = job.run(root, database, TEST_SEASONS, directory, execute=True, key='test',
                     transport=no_http, max_fixtures=1)
    assert result['states']['ready'] == 1
    assert not no_http.calls


def budget_database(tmp_path):
    path = tmp_path / 'budget.db'
    with sqlite3.connect(path) as db:
        db.execute('CREATE TABLE player_collection_budget(day TEXT PRIMARY KEY, attempts INTEGER NOT NULL)')
    return path


def test_attempt_budget_counts_retries_and_survives_restarts(tmp_path):
    database = budget_database(tmp_path)
    utc = lambda: datetime(2026, 9, 30, tzinfo=timezone.utc)
    transport = Transport([requests.Timeout(), Response(status=503), Response()])
    session = job.BudgetSession(database, 'test', daily_budget=2, session=transport,
                                sleep=lambda _: None, utc=utc)
    with pytest.raises(job.BudgetPause):
        job.get_json(job.BASE_URL + job.ENDPOINT, params={'fixture': 42}, session=session, sleep=lambda _: None)
    assert len(transport.calls) == 2
    restarted = job.BudgetSession(database, 'test', daily_budget=2, session=Transport(), utc=utc)
    with pytest.raises(job.BudgetPause):
        restarted.get(job.BASE_URL + job.ENDPOINT)
    next_day = job.BudgetSession(database, 'test', daily_budget=2, session=Transport([Response()]),
                                 utc=lambda: datetime(2026, 10, 1, tzinfo=timezone.utc))
    assert next_day.get(job.BASE_URL + job.ENDPOINT).status_code == 200


def test_remaining_provider_quota_stops_before_next_request(tmp_path):
    transport = Transport([Response(headers={'x-ratelimit-requests-remaining': '5000'})])
    session = job.BudgetSession(budget_database(tmp_path), 'test', session=transport, reserve=5000)
    session.get(job.BASE_URL + job.ENDPOINT)
    with pytest.raises(job.BudgetPause):
        session.get(job.BASE_URL + job.ENDPOINT)
    assert len(transport.calls) == 1


def test_rate_limit_headers_and_429_slow_requests(tmp_path):
    sleeps = []
    transport = Transport([Response(status=429, headers={'X-Ratelimit-Remaining': '0', 'Retry-After': '90'}), Response()])
    session = job.BudgetSession(budget_database(tmp_path), 'test', session=transport,
                                clock=lambda: 0, sleep=sleeps.append)
    session.get(job.BASE_URL + job.ENDPOINT)
    session.get(job.BASE_URL + job.ENDPOINT)
    assert sleeps == [90]


def test_lower_provider_minute_limit_is_respected(tmp_path):
    sleeps = []
    transport = Transport([Response(headers={'X-Ratelimit-Limit': '30'}), Response()])
    session = job.BudgetSession(budget_database(tmp_path), 'test', session=transport,
                                rpm=120, clock=lambda: 0, sleep=sleeps.append)
    session.get(job.BASE_URL + job.ENDPOINT)
    session.get(job.BASE_URL + job.ENDPOINT)
    assert sleeps == [2]


@pytest.mark.parametrize('accuracy', ['85%', -1, True, 'nan', 21, 'unknown'])
def test_ambiguous_pass_accuracy_preserved_raw_but_not_misrepresented(source, accuracy):
    root, database = source
    p = payload()
    p['response'][0]['players'][0]['statistics'][0]['passes']['accuracy'] = accuracy
    directory = root / 'Index/history_staging/passes'
    result = job.run(root, database, TEST_SEASONS, directory, execute=True, key='test',
                     transport=Transport([Response(p)]), max_fixtures=1)
    assert result['states']['ready'] == 1
    with sqlite3.connect(directory / 'platform.db') as db:
        row = db.execute('SELECT passes_accurate,pass_accuracy,stats_json FROM fixture_player_stats ORDER BY id LIMIT 1').fetchone()
    assert row[:2] == (None, None)
    assert json.loads(row[2])['provider_statistics']['passes']['accuracy'] == accuracy


def test_budget_pause_leaves_unfetched_fixtures_pending(source):
    root, database = source
    directory = root / 'Index/history_staging/quota'
    directory.mkdir()
    fixtures = job.fixture_inventory(database, TEST_SEASONS)
    job.initialize(directory, fixtures, job.plan(database, TEST_SEASONS, fixtures))
    transport = Transport([Response()])
    session = job.BudgetSession(directory / 'platform.db', 'test', daily_budget=1,
                                session=transport, sleep=lambda _: None)
    result = job.collect(directory, fixtures, key='test', daily_budget=1, reserve=0, rpm=120, transport=session)
    assert result['states'] == {'pending': 2, 'ready': 1}
    assert sum(result['http_attempts_by_utc_day'].values()) == 1
    assert 'budget' in result['stopped']


def test_pagination_or_api_errors_cannot_become_success(source):
    root, database = source
    p = payload()
    p['paging']['total'] = 2
    result = job.run(root, database, TEST_SEASONS, root / 'Index/history_staging/pagination',
                     execute=True, key='test', transport=Transport([Response(p)]))
    assert result['states'] == {'pending': 3}
    assert result['linked_player_rows'] == 0


def test_auth_failure_stops_whole_run_and_keeps_pending(source):
    root, database = source
    transport = Transport([Response(status=401)])
    result = job.run(root, database, TEST_SEASONS, root / 'Index/history_staging/auth',
                     execute=True, key='test', transport=transport)
    assert result['states'] == {'pending': 3}
    assert 'failure' in result['stopped']
    assert len(transport.calls) == 1


def test_scope_change_and_checksum_corruption_fail_closed(source):
    root, database = source
    directory = root / 'Index/history_staging/frozen'
    job.run(root, database, TEST_SEASONS, directory, execute=True, key='test',
            transport=Transport([Response()]), max_fixtures=1)
    with pytest.raises(ValueError, match='scope/identity'):
        job.run(root, database, (2022, 2023, 2024, 2025, 2026), directory,
                execute=True, key='test', transport=Transport())
    archive = next((directory / 'raw_archive').rglob('*.gz'))
    archive.write_bytes(job.gzip.compress(b'{}'))
    with pytest.raises(ValueError, match='checksum'):
        job.read_archive(directory, 1)


def test_directory_cannot_alias_live_database(source):
    root, database = source
    directory = root / 'Index/history_staging/alias'
    directory.mkdir()
    (directory / 'platform.db').symlink_to(database)
    with pytest.raises(ValueError, match='alias'):
        job.run(root, database, TEST_SEASONS, directory, execute=True, key='test')
    with pytest.raises(ValueError, match='child'):
        job.run(root, database, TEST_SEASONS, root / 'Index', execute=True, key='test')


def test_concurrent_collector_is_rejected(source):
    root, database = source
    directory = root / 'Index/history_staging/busy'
    directory.mkdir()
    with (directory / 'collection.lock').open('w') as lock:
        job.fcntl.flock(lock, job.fcntl.LOCK_EX | job.fcntl.LOCK_NB)
        with pytest.raises(ValueError, match='already running'):
            job.run(root, database, TEST_SEASONS, directory, execute=True, key='test', transport=Transport())


def test_three_endpoints_include_current_season_and_preserve_exact_links(source):
    root, database = source
    before = hashlib.sha256(database.read_bytes()).hexdigest()
    directory = root / 'Index/history_staging/all-current'
    transport = Transport([Response(payload(45)), Response(lineup_payload(45)), Response(events_payload(45))])
    result = job.run(root, database, (2026,), directory, endpoints=job.ENDPOINTS,
                     execute=True, key='test', transport=transport)
    assert [url for url, _ in transport.calls] == [job.BASE_URL + e for e in job.ENDPOINTS]
    assert all(kwargs['params'] == {'fixture': 45} for _, kwargs in transport.calls)
    assert result['states'] == {'ready': 3}
    assert result['linked_player_rows'] == 22
    assert result['linked_lineup_rows'] == 30
    assert result['linked_event_rows'] == 3
    assert result['foreign_key_errors'] == 0
    with sqlite3.connect(directory / 'platform.db') as db:
        assert db.execute('SELECT count(*) FROM players').fetchone()[0] == 30
        assert db.execute("SELECT count(*) FROM fixture_lineup_entries WHERE role='bench'").fetchone()[0] == 8
        row = db.execute('''SELECT f.api_football_id,t.api_football_id,p.api_football_id,l.raw_json
            FROM fixture_lineup_entries l JOIN fixtures f ON f.id=l.fixture_id
            JOIN teams t ON t.id=l.team_id JOIN players p ON p.id=l.player_id
            WHERE p.api_football_id=20011''').fetchone()
        assert row[:3] == (45, 200, 20011)
        assert json.loads(row[3])['player']['grid'] == '2:1'
        events = db.execute('''SELECT f.api_football_id,t.api_football_id,e.player_api_id,e.assist_api_id,
            e.elapsed,e.extra,e.detail FROM fixture_events e JOIN fixtures f ON f.id=e.fixture_id
            JOIN teams t ON t.id=e.team_id ORDER BY e.event_index''').fetchall()
        assert events[0] == (45, 100, 10001, None, 45, 2, 'Yellow Card')
        assert events[1][-1] == 'Yellow-Red Card'
        assert events[2][:4] == (45, 200, 20002, 20011)
    assert hashlib.sha256(database.read_bytes()).hexdigest() == before
    rerun = Transport()
    job.run(root, database, (2026,), directory, endpoints=job.ENDPOINTS,
            execute=True, key='test', transport=rerun)
    assert not rerun.calls


def test_one_shared_budget_across_endpoints_and_resume_only_remaining_endpoint(source):
    root, database = source
    directory = root / 'Index/history_staging/shared-budget'
    directory.mkdir()
    fixtures = job.fixture_inventory(database, (2026,))
    manifest = job.plan(database, (2026,), fixtures, job.ENDPOINTS)
    job.initialize(directory, fixtures, manifest)
    network = Transport([Response(payload(45)), Response(lineup_payload(45))])
    budget = job.BudgetSession(directory / 'platform.db', 'test', daily_budget=2,
                               session=network, sleep=lambda _: None)
    result = job.run(root, database, (2026,), directory, endpoints=job.ENDPOINTS,
                     execute=True, key='test', transport=budget)
    assert result['states'] == {'pending': 1, 'ready': 2}
    assert result['states_by_endpoint']['/fixtures/events'] == {'pending': 1}
    assert sum(result['http_attempts_by_utc_day'].values()) == 2
    later = Transport([Response(events_payload(45))])
    result = job.run(root, database, (2026,), directory, endpoints=job.ENDPOINTS,
                     execute=True, key='test', transport=later)
    assert result['states'] == {'ready': 3}
    assert [url for url, _ in later.calls] == [job.BASE_URL + '/fixtures/events']


@pytest.mark.parametrize('endpoint,builder,mutation', [
    ('/fixtures/lineups', lineup_payload, lambda p: p['parameters'].update(fixture=999)),
    ('/fixtures/lineups', lineup_payload, lambda p: p['response'][0]['team'].update(id=999)),
    ('/fixtures/lineups', lineup_payload, lambda p: p['response'][0]['startXI'].pop()),
    ('/fixtures/lineups', lineup_payload, lambda p: p['response'][0]['substitutes'][0]['player'].update(id=10000)),
    ('/fixtures/lineups', lineup_payload, lambda p: p['response'][0].update(formation={})),
    ('/fixtures/events', events_payload, lambda p: p['parameters'].update(fixture=999)),
    ('/fixtures/events', events_payload, lambda p: p['response'][0]['team'].update(id=999)),
    ('/fixtures/events', events_payload, lambda p: p['response'][0]['time'].update(elapsed=-1)),
    ('/fixtures/events', events_payload, lambda p: p['response'][0]['player'].update(id=True)),
    ('/fixtures/events', events_payload, lambda p: p['response'][0].update(detail={})),
])
def test_context_mismatches_are_archived_without_normalized_rows(source, endpoint, builder, mutation):
    root, database = source
    directory = root / 'Index/history_staging/invalid-context'
    p = builder(45)
    mutation(p)
    result = job.run(root, database, (2026,), directory, endpoints=(endpoint,),
                     execute=True, key='test', transport=Transport([Response(p)]))
    assert result['states'] == {'invalid': 1}
    assert result['linked_lineup_rows'] == result['linked_event_rows'] == 0
    assert job.read_archive(directory, 1) == p


@pytest.mark.parametrize('endpoint,builder,expected', [
    ('/fixtures/lineups', lineup_payload, 'unavailable'),
    ('/fixtures/events', events_payload, 'empty_unverified'),
])
def test_empty_context_is_not_zero_evidence_and_does_not_trigger_player_calls(source, endpoint, builder, expected):
    root, database = source
    directory = root / 'Index/history_staging/empty-context'
    p = builder(45)
    p.update(results=0, response=[])
    result = job.run(root, database, (2026,), directory, endpoints=(endpoint,),
                     execute=True, key='test', transport=Transport([Response(p)]))
    assert result['states'] == {expected: 1}
    no_http = Transport()
    result = job.run(root, database, (2026,), directory, endpoints=(endpoint,),
                     execute=True, key='test', transport=no_http)
    assert result['states'] == {expected: 1}
    assert not no_http.calls


def test_unknown_event_actors_and_duplicates_are_retained_as_partial(source):
    root, database = source
    directory = root / 'Index/history_staging/partial-events'
    p = events_payload(45)
    p['response'][0]['player']['id'] = None
    p['response'].append(dict(p['response'][0]))
    p['results'] = 4
    result = job.run(root, database, (2026,), directory, endpoints=('/fixtures/events',),
                     execute=True, key='test', transport=Transport([Response(p)]))
    assert result['states'] == {'partial': 1}
    assert result['linked_event_rows'] == 4
    with sqlite3.connect(directory / 'platform.db') as db:
        assert db.execute('SELECT player_api_id FROM fixture_events WHERE event_index=0').fetchone()[0] is None


def test_scope_extension_adds_current_season_and_endpoints_without_replacing_players(source):
    root, database = source
    directory = root / 'Index/history_staging/extended'
    job.run(root, database, TEST_SEASONS, directory, execute=True, key='test',
            transport=Transport([Response()]), max_fixtures=1)
    first_archive = next((directory / 'raw_archive').rglob('*.gz'))
    before = first_archive.read_bytes()
    with sqlite3.connect(directory / 'platform.db') as db:
        db.execute("INSERT INTO player_collection_budget VALUES ('2026-09-30',7)")
    transport = Transport([Response(lineup_payload()), Response(events_payload())])
    result = job.run(root, database, job.DEFAULT_SEASONS, directory, endpoints=job.ENDPOINTS,
                     extend_scope=True, execute=True, key='test', transport=transport, max_fixtures=1)
    assert result['states'] == {'ready': 3, 'pending': 9}
    assert [url for url, _ in transport.calls] == [job.BASE_URL + e for e in job.ENDPOINTS[1:]]
    assert first_archive.read_bytes() == before
    assert result['http_attempts_by_utc_day']['2026-09-30'] == 7
    assert len(list((directory / 'scope-history').glob('*.json'))) == 1
    assert job.saved_scope(directory)[0]['fixture_count'] == 4


def test_v1_collection_upgrade_preserves_archives_progress_and_budget(source):
    root, database = source
    directory = root / 'Index/history_staging/v1'
    job.run(root, database, TEST_SEASONS, directory, execute=True, key='test',
            transport=Transport([Response()]), max_fixtures=1)
    old = json.loads((directory / 'manifest.json').read_text())
    old.update(schema='fixture-player-history-collection.v1', endpoint=job.ENDPOINT,
               source_sha256={'old_script': 'retained-for-provenance'})
    old.pop('endpoints')
    job.write_json(directory / 'manifest.json', old)
    with sqlite3.connect(directory / 'platform.db') as db:
        for table in ('fixture_endpoint_collection', 'player_collection_scope',
                      'fixture_events', 'fixture_lineup_entries', 'fixture_lineup_teams'):
            db.execute('DROP TABLE ' + table)
        db.execute("INSERT INTO player_collection_budget VALUES ('2026-09-30',5)")
    transport = Transport([Response(lineup_payload()), Response(events_payload())])
    result = job.run(root, database, job.DEFAULT_SEASONS, directory, endpoints=job.ENDPOINTS,
                     extend_scope=True, execute=True, key='test', transport=transport, max_fixtures=1)
    assert result['states'] == {'ready': 3, 'pending': 9}
    assert result['http_attempts_by_utc_day']['2026-09-30'] == 5
    assert all(not url.endswith('/players') for url, _ in transport.calls)
    history = json.loads(next((directory / 'scope-history').glob('*.json')).read_text())
    assert history['manifest'] == old


def test_extension_rejects_reidentified_fixtures_and_scope_shrink(source):
    root, database = source
    directory = root / 'Index/history_staging/identity-conflict'
    job.run(root, database, TEST_SEASONS, directory, execute=True, key='test',
            transport=Transport([Response()]), max_fixtures=1)
    with pytest.raises(ValueError, match='cannot remove'):
        job.run(root, database, (2025,), directory, extend_scope=True,
                execute=True, key='test', transport=Transport())
    with sqlite3.connect(database) as db:
        db.execute("UPDATE fixtures SET round='Quarter-finals' WHERE api_football_id=42")
    with pytest.raises(ValueError, match='metadata changed'):
        job.run(root, database, job.DEFAULT_SEASONS, directory, extend_scope=True,
                execute=True, key='test', transport=Transport())


def test_partial_initialization_resumes_from_preparation_journal(source, monkeypatch):
    root, database = source
    directory = root / 'Index/history_staging/preparing'
    initialize = job.initialize
    def interrupted(*args):
        raise OSError('simulated disk interruption')
    monkeypatch.setattr(job, 'initialize', interrupted)
    with pytest.raises(OSError, match='interruption'):
        job.run(root, database, (2026,), directory, endpoints=job.ENDPOINTS,
                execute=True, key='test', transport=Transport())
    assert (directory / 'preparing-scope.json').exists()
    monkeypatch.setattr(job, 'initialize', initialize)
    result = job.run(root, database, (2026,), directory, endpoints=job.ENDPOINTS,
                     execute=True, key='test', transport=Transport([
                         Response(payload(45)), Response(lineup_payload(45)), Response(events_payload(45))]))
    assert result['states'] == {'ready': 3}
    assert not (directory / 'preparing-scope.json').exists()


def test_cli_defaults_to_six_seasons_and_three_endpoints(monkeypatch):
    captured = {}
    def run(*args, **kwargs):
        captured.update(seasons=args[2], **kwargs)
        return {'dry_run': True}
    monkeypatch.setattr(job, 'run', run)
    assert job.main(['--dry-run']) == 0
    assert captured['seasons'] == (2021, 2022, 2023, 2024, 2025, 2026)
    assert captured['endpoints'] == job.ENDPOINTS
