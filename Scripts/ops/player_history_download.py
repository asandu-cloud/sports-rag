"""Resumable player, lineup and event collection into a fixture-linked staging DB.

Default invocation is a read-only plan. --execute makes provider requests and
writes staging only. It never imports into the publishing database or builds
profiles. Provider seasons are explicit (2021 means 2021/22).
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from collections import Counter
from contextlib import closing
from datetime import datetime, timezone
import fcntl
import gzip
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sqlite3
import sys
import time
from urllib.parse import unquote, urlparse

import requests

from Scripts.football_http import ApiFootballResponseError, get_json, validate_envelope, validate_stat_blocks

ROOT = Path(__file__).resolve().parents[2]
LEAGUES = ('EPL', 'LaLiga', 'SerieA', 'Bundesliga', 'Ligue1', 'Championship',
           'SuperLig', 'Eredivisie', 'PrimeiraLiga', 'BelgianProLeague', 'UCL', 'UEL', 'UECL')
DEFAULT_SEASONS = (2021, 2022, 2023, 2024, 2025, 2026)
DEFAULT_RPM = 420
MAX_RPM = 420  # Ultra's seven-per-second limit, with evenly spaced starts.
DEFAULT_DAILY_BUDGET = 75000
DEFAULT_RESERVE = 0
SCHEMA = 'fixture-player-history-collection.v2'
ENDPOINT = '/fixtures/players'
ENDPOINTS = (ENDPOINT, '/fixtures/lineups', '/fixtures/events')
TERMINAL_STATES = ('ready', 'partial', 'invalid', 'unavailable', 'empty_unverified')
BASE_URL = 'https://v3.football.api-sports.io'
SOURCES = ('Scripts/ops/player_history_download.py', 'Scripts/football_http.py',
           'Scripts/data_platform/sync/upserts.py', 'Scripts/data_platform/storage/archive.py',
           'Scripts/data_platform/models/core.py', 'Scripts/data_platform/models/fixtures.py',
           'Scripts/data_platform/models/sync.py')


def now():
    return datetime.now(timezone.utc)


def encoded(value):
    return json.dumps(value, sort_keys=True, allow_nan=False, default=str).encode()


def digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def write_bytes(path, data):
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('wb') as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    parent_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)


def write_json(path, value):
    write_bytes(path, encoded(value) + b'\n')


def readonly(path):
    db = sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True, timeout=5)
    db.row_factory = sqlite3.Row
    db.execute('PRAGMA query_only=ON')
    return db


def fixture_inventory(database, seasons):
    """Read identity metadata, never outcomes from protected evaluation periods."""
    with closing(readonly(database)) as db:
        placeholders = ','.join('?' for _ in seasons)
        rows = db.execute(f'''SELECT f.id AS platform_fixture_id,
            f.api_football_id AS fixture_id, c.id AS competition_id, c.code,
            c.name AS competition_name, c.api_football_id AS league_id,
            c.country, c.competition_type, s.id AS season_id, s.year AS season,
            f.kickoff_utc, f.status, f.round, f.referee,
            h.id AS home_id, h.api_football_id AS home_team_id, h.name AS home_name,
            a.id AS away_id, a.api_football_id AS away_team_id, a.name AS away_name
            FROM fixtures f JOIN competitions c ON c.id=f.competition_id
            JOIN seasons s ON s.id=f.season_id JOIN teams h ON h.id=f.home_team_id
            JOIN teams a ON a.id=f.away_team_id
            WHERE s.year IN ({placeholders}) AND f.status IN ('FT','AET','PEN')
            ORDER BY s.year,f.kickoff_utc,f.api_football_id''', tuple(seasons)).fetchall()
    fixtures = [dict(r) for r in rows if r['code'] in LEAGUES]
    if not fixtures:
        raise ValueError('No completed fixtures in the requested seasons')
    if len({f['fixture_id'] for f in fixtures}) != len(fixtures):
        raise ValueError('Duplicate provider fixture identities')
    return fixtures


def safe_directory(root, directory):
    root = root.resolve()
    directory = (root / directory).absolute()
    parent = root / 'Index/history_staging'
    if directory == parent or not directory.is_relative_to(parent):
        raise ValueError('Collection directory must be a child of Index/history_staging')
    if directory.resolve() == parent.resolve() or not directory.resolve().is_relative_to(parent.resolve()):
        raise ValueError('Resolved collection path must remain a child of Index/history_staging')
    # Forbid symlinks anywhere on the staging path, including future writes.
    for p in (directory, *directory.parents):
        if p == root:
            break
        if p.is_symlink():
            raise ValueError('Staging path must not contain symlinks')
    if directory.exists():
        for p in directory.rglob('*'):
            if p.is_symlink() or (p.is_file() and p.stat().st_nlink > 1):
                raise ValueError('Staging files must not alias existing data')
    return directory.resolve()


def plan(database, seasons, fixtures, endpoints=(ENDPOINT,)):
    return {'schema': SCHEMA, 'source_database': str(database.resolve()),
            'seasons': list(seasons), 'competitions': list(LEAGUES),
            'endpoints': list(endpoints), 'fixture_count': len(fixtures),
            'fixture_identity_sha256': digest(fixtures),
            'by_league_season': dict(sorted(Counter(f"{f['code']}:{f['season']}" for f in fixtures).items())),
            'base_request_estimate': len(fixtures) * len(endpoints),
            'source_sha256': {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in SOURCES},
            'dependencies': {name: importlib.metadata.version(name) for name in ('requests', 'SQLAlchemy')},
            'policy': 'full unfiltered responses; nulls preserved; staging only; no profile builds',
            'evaluation': 'collection is not permission to inspect protected test outcomes'}


def positive_id(value):
    return type(value) is int and value > 0


def validate_identity(fixture, payload, endpoint):
    validate_envelope(payload, endpoint)
    if payload.get('get') not in (endpoint.lstrip('/'), endpoint):
        raise ValueError('Unexpected response endpoint')
    parameters = payload.get('parameters')
    if not isinstance(parameters, dict) or str(parameters.get('fixture')) != str(fixture['fixture_id']):
        raise ValueError('Response fixture identity mismatch')


def validate_players(fixture, payload):
    """Source structure/identity validation, separate from card eligibility."""
    validate_identity(fixture, payload, ENDPOINT)
    validate_stat_blocks(payload, ENDPOINT, params={'fixture': fixture['fixture_id']})
    blocks = payload['response']
    if not blocks:
        return 'unavailable'
    if {b['team']['id'] for b in blocks} != {fixture['home_team_id'], fixture['away_team_id']}:
        raise ValueError('Response team identities do not match the saved fixture')
    seen = set()
    for block in blocks:
        if not positive_id(block['team']['id']) or len(block['players']) < 11:
            raise ValueError('Incomplete team roster')
        if block['team'].get('name') is not None and not isinstance(block['team']['name'], str):
            raise ValueError('Invalid team name')
        for row in block['players']:
            player = row.get('player') or {}
            pid = player.get('id')
            if not positive_id(pid) or pid in seen:
                raise ValueError('Missing or duplicate player identity')
            if player.get('name') is not None and not isinstance(player['name'], str):
                raise ValueError('Invalid player name')
            seen.add(pid)
            stats = row.get('statistics')
            if not isinstance(stats, list) or len(stats) != 1 or not isinstance(stats[0], dict):
                raise ValueError('Ambiguous player statistics blocks')
            for key in ('games', 'shots', 'goals', 'passes', 'tackles', 'duels', 'dribbles', 'fouls', 'cards', 'penalty'):
                value = stats[0].get(key)
                if value is not None and not isinstance(value, dict):
                    raise ValueError('Malformed player statistics section')
            position = (stats[0].get('games') or {}).get('position')
            if position is not None and not isinstance(position, str):
                raise ValueError('Invalid player position')
            for section, fields in (
                ('games', ('minutes', 'number')), ('shots', ('total', 'on')),
                ('goals', ('total', 'assists', 'conceded', 'saves')),
                ('passes', ('total', 'key')), ('tackles', ('total', 'blocks', 'interceptions')),
                ('duels', ('total', 'won')), ('dribbles', ('attempts', 'success', 'past')),
                ('fouls', ('drawn', 'committed')), ('cards', ('yellow', 'red')),
                ('penalty', ('won', 'commited', 'scored', 'missed', 'saved')),
            ):
                for field in fields:
                    value = (stats[0].get(section) or {}).get(field)
                    if value is not None and (type(value) is not int or value < 0):
                        raise ValueError(f'Invalid player count: {section}.{field}')
            for field in ('captain', 'substitute'):
                value = (stats[0].get('games') or {}).get(field)
                if value is not None and type(value) is not bool:
                    raise ValueError(f'Invalid games.{field}')
            rating = (stats[0].get('games') or {}).get('rating')
            if rating is not None:
                if isinstance(rating, bool) or not 0 <= float(rating) <= 10:
                    raise ValueError('Invalid games.rating')
    return 'ready'


def validate_lineups(fixture, payload):
    validate_identity(fixture, payload, '/fixtures/lineups')
    blocks = payload['response']
    if not blocks:
        return 'unavailable'
    if (len(blocks) != 2 or {b['team']['id'] for b in blocks}
            != {fixture['home_team_id'], fixture['away_team_id']}):
        raise ValueError('Lineup team identities do not match the saved fixture')
    seen = set()
    for block in blocks:
        if not positive_id(block['team']['id']):
            raise ValueError('Invalid lineup team identity')
        if block.get('formation') is not None and not isinstance(block['formation'], str):
            raise ValueError('Invalid lineup formation')
        starters, substitutes = block.get('startXI'), block.get('substitutes')
        if not isinstance(starters, list) or len(starters) != 11 or not isinstance(substitutes, list):
            raise ValueError('Incomplete starting XI or missing bench list')
        for row in starters + substitutes:
            pid = row['player']['id']
            if not positive_id(pid) or pid in seen:
                raise ValueError('Missing or duplicate lineup player identity')
            if row['player'].get('name') is not None and not isinstance(row['player']['name'], str):
                raise ValueError('Invalid lineup player name')
            seen.add(pid)
    return 'ready'


def validate_events(fixture, payload):
    validate_identity(fixture, payload, '/fixtures/events')
    if not payload['response']:
        # A genuine event-free match and absent provider coverage are not yet distinguishable.
        return 'empty_unverified'
    partial, seen = False, set()
    for event in payload['response']:
        team = event.get('team') or {}
        tid = team.get('id')
        if tid is not None and (not positive_id(tid) or tid not in (fixture['home_team_id'], fixture['away_team_id'])):
            raise ValueError('Event team identity does not match the saved fixture')
        for role in ('player', 'assist'):
            pid = (event.get(role) or {}).get('id')
            if pid is not None and not positive_id(pid):
                raise ValueError('Invalid event participant identity')
        timing = event.get('time') or {}
        for key in ('elapsed', 'extra'):
            value = timing.get(key)
            if value is not None and type(value) is not int:
                raise ValueError('Invalid event time')
            # Observed provider sentinel -5 has no established minute meaning.
            # Retain the event as partial evidence, never turn it into minute 0.
            partial |= value is not None and value < 0
        if not isinstance(event.get('type'), str) or not event['type']:
            raise ValueError('Missing event type')
        if event.get('detail') is not None and not isinstance(event['detail'], str):
            raise ValueError('Invalid event detail')
        partial |= tid is None or timing.get('elapsed') is None
        if event['type'] in ('Goal', 'Card', 'subst'):
            partial |= (event.get('player') or {}).get('id') is None
        if event['type'] == 'subst':
            partial |= (event.get('assist') or {}).get('id') is None
        fingerprint = digest(event)
        partial |= fingerprint in seen
        seen.add(fingerprint)
    return 'partial' if partial else 'ready'


def normalization_payload(payload):
    """Avoid interpreting a percentage as an accurate-pass count.

    Keep the original value in the archive/provider_statistics. The legacy
    generic upsert accepts percentages and would otherwise truncate to zero.
    """
    blocks = deepcopy(payload['response'])
    for block in blocks:
        for player in block['players']:
            stats = player['statistics'][0]
            passes = stats.get('passes') or {}
            accuracy = passes.get('accuracy')
            if accuracy is not None:
                try:
                    valid = (not isinstance(accuracy, bool) and
                             float(accuracy).is_integer() and float(accuracy) >= 0)
                except (TypeError, ValueError, OverflowError):
                    valid = False
                if not valid or (passes.get('total') is not None and float(accuracy) > passes['total']):
                    passes['accuracy'] = None
    return blocks


def initialize(directory, fixtures, manifest):
    """Use the existing canonical schema and exact saved catalogue/fixture IDs."""
    from sqlalchemy import create_engine
    from Scripts.data_platform.models import Base
    engine = create_engine(f"sqlite:///{directory / 'platform.db'}")
    Base.metadata.create_all(engine)
    engine.dispose()
    with sqlite3.connect(directory / 'platform.db') as db:
        db.execute('PRAGMA foreign_keys=ON')
        db.executescript('''
            CREATE TABLE IF NOT EXISTS player_collection (
              fixture_id INTEGER PRIMARY KEY REFERENCES fixtures(id), state TEXT NOT NULL,
              archive_id INTEGER, error TEXT, updated_at TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS player_collection_budget (
              day TEXT PRIMARY KEY, attempts INTEGER NOT NULL);
            CREATE TABLE IF NOT EXISTS fixture_endpoint_collection (
              fixture_id INTEGER NOT NULL REFERENCES fixtures(id), endpoint TEXT NOT NULL,
              state TEXT NOT NULL, archive_id INTEGER REFERENCES raw_payload_archive(id),
              error TEXT, updated_at TEXT NOT NULL, PRIMARY KEY(fixture_id,endpoint));
            CREATE TABLE IF NOT EXISTS fixture_lineup_teams (
              fixture_id INTEGER NOT NULL REFERENCES fixtures(id),
              team_id INTEGER NOT NULL REFERENCES teams(id), formation TEXT,
              coach_json TEXT, raw_json TEXT NOT NULL,
              archive_id INTEGER NOT NULL REFERENCES raw_payload_archive(id),
              PRIMARY KEY(fixture_id,team_id));
            CREATE TABLE IF NOT EXISTS fixture_lineup_entries (
              fixture_id INTEGER NOT NULL REFERENCES fixtures(id),
              team_id INTEGER NOT NULL REFERENCES teams(id),
              player_id INTEGER NOT NULL REFERENCES players(id), role TEXT NOT NULL,
              raw_json TEXT NOT NULL, archive_id INTEGER NOT NULL REFERENCES raw_payload_archive(id),
              PRIMARY KEY(fixture_id,player_id));
            CREATE TABLE IF NOT EXISTS fixture_events (
              fixture_id INTEGER NOT NULL REFERENCES fixtures(id),
              archive_id INTEGER NOT NULL REFERENCES raw_payload_archive(id), event_index INTEGER NOT NULL,
              team_id INTEGER REFERENCES teams(id), player_api_id INTEGER, assist_api_id INTEGER,
              elapsed INTEGER, extra INTEGER, event_type TEXT NOT NULL, detail TEXT,
              raw_json TEXT NOT NULL, PRIMARY KEY(fixture_id,archive_id,event_index));
            CREATE INDEX IF NOT EXISTS ix_fixture_events_player ON fixture_events(player_api_id);
            CREATE TABLE IF NOT EXISTS player_collection_scope (
              id INTEGER PRIMARY KEY CHECK(id=1), manifest TEXT NOT NULL, fixtures TEXT NOT NULL);
        ''')
        # v1 progress and its request ledger remain intact. This is a staging-only migration.
        db.execute('''INSERT OR IGNORE INTO fixture_endpoint_collection
            SELECT fixture_id,?,state,archive_id,error,updated_at FROM player_collection''', (ENDPOINT,))
        timestamp = now().isoformat()
        for f in fixtures:
            db.execute('''INSERT OR IGNORE INTO competitions
                (id,code,name,country,api_football_id,competition_type,created_at,updated_at)
                VALUES (?,?,?,?,?,?,?,?)''', (f['competition_id'], f['code'], f['competition_name'],
                f['country'], f['league_id'], f['competition_type'], timestamp, timestamp))
            db.execute('''INSERT OR IGNORE INTO seasons
                (id,competition_id,year,label,is_current,created_at,updated_at) VALUES (?,?,?,?,0,?,?)''',
                (f['season_id'], f['competition_id'], f['season'],
                 f"{f['season']}/{str(f['season'] + 1)[-2:]}", timestamp, timestamp))
            for side in ('home', 'away'):
                db.execute('''INSERT OR IGNORE INTO teams
                    (id,api_football_id,name,created_at,updated_at) VALUES (?,?,?,?,?)''',
                    (f[side + '_id'], f[side + '_team_id'], f[side + '_name'], timestamp, timestamp))
            db.execute('''INSERT OR IGNORE INTO fixtures
                (id,api_football_id,competition_id,season_id,home_team_id,away_team_id,
                 kickoff_utc,status,round,referee,created_at,updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)''',
                (f['platform_fixture_id'], f['fixture_id'], f['competition_id'], f['season_id'],
                 f['home_id'], f['away_id'], f['kickoff_utc'], f['status'], f['round'], f['referee'], timestamp, timestamp))
            if ENDPOINT in manifest.get('endpoints', (ENDPOINT,)):
                db.execute('INSERT OR IGNORE INTO player_collection VALUES (?,\'pending\',NULL,NULL,?)',
                           (f['platform_fixture_id'], timestamp))
            for endpoint in manifest.get('endpoints', (ENDPOINT,)):
                db.execute('INSERT OR IGNORE INTO fixture_endpoint_collection VALUES (?,?,\'pending\',NULL,NULL,?)',
                           (f['platform_fixture_id'], endpoint, timestamp))
        db.execute('INSERT OR REPLACE INTO player_collection_scope VALUES (1,?,?)',
                   (encoded(manifest).decode(), encoded(fixtures).decode()))
    # SQLite is authoritative if an interruption occurs between these file writes.
    write_json(directory / 'fixture-identities.json', fixtures)
    write_json(directory / 'manifest.json', manifest)


class BudgetPause(RuntimeError):
    pass


class BudgetSession:
    """Count every HTTP attempt (including shared-client retries) durably."""
    def __init__(self, database, key, *, daily_budget=DEFAULT_DAILY_BUDGET, reserve=DEFAULT_RESERVE, rpm=DEFAULT_RPM,
                 session=None, sleep=time.sleep, clock=time.monotonic, utc=now):
        if not 1 <= rpm <= MAX_RPM:
            raise ValueError(f'Request rate must be between 1 and {MAX_RPM} per minute')
        self.database, self.daily_budget, self.reserve, self.rpm = database, daily_budget, reserve, rpm
        self.session = session or requests.Session()
        self.session.headers.update({'x-apisports-key': key})
        self.sleep, self.clock, self.utc = sleep, clock, utc
        self.next_at = 0.0
        self.provider_remaining = None
        self.provider_rpm = None

    def request_interval(self):
        interval = 60 / self.rpm
        if self.provider_rpm is not None:
            # The provider also meters whole requests per second (Ultra: 7).
            interval = max(interval, 60 / self.provider_rpm,
                           1 / max(1, self.provider_rpm // 60))
        return interval

    def get(self, url, **kwargs):
        if url not in {BASE_URL + endpoint for endpoint in ENDPOINTS}:
            raise ValueError('Collector cannot request endpoints outside the approved three')
        if self.provider_remaining is not None and self.provider_remaining <= self.reserve:
            raise BudgetPause('Provider remaining daily allowance reached the reserved margin')
        delay = self.next_at - self.clock()
        if delay > 0:
            self.sleep(delay)
        day = self.utc().date().isoformat()
        with sqlite3.connect(self.database) as db:
            db.execute('BEGIN IMMEDIATE')
            used = db.execute('SELECT attempts FROM player_collection_budget WHERE day=?', (day,)).fetchone()
            if used and used[0] >= self.daily_budget:
                raise BudgetPause('This collection reached its UTC daily request budget')
            db.execute('''INSERT INTO player_collection_budget VALUES (?,1)
                ON CONFLICT(day) DO UPDATE SET attempts=attempts+1''', (day,))
        started_at = self.clock()
        self.next_at = started_at + self.request_interval()
        response = self.session.get(url, **kwargs)
        headers = {k.lower(): v for k, v in response.headers.items()}
        remaining = headers.get('x-ratelimit-requests-remaining')
        if remaining is not None and str(remaining).isdigit():
            self.provider_remaining = int(remaining)
        per_minute = headers.get('x-ratelimit-remaining')
        limit = headers.get('x-ratelimit-limit')
        if limit is not None and str(limit).isdigit() and int(limit) > 0:
            self.provider_rpm = int(limit)
        # Space request starts; response latency already consumes this interval.
        # Retain the last advertised cap if a later response omits the header.
        self.next_at = max(self.next_at, started_at + self.request_interval())
        if per_minute is not None and str(per_minute).isdigit() and int(per_minute) == 0:
            self.next_at = max(self.next_at, self.clock() + 60)
        if response.status_code == 429:
            retry = headers.get('retry-after', '60')
            delay = min(300, max(60, int(retry))) if str(retry).isdigit() else 60
            self.next_at = max(self.next_at, self.clock() + delay)
        return response

    def close(self):
        self.session.close()


def update_state(session, fixture, endpoint, state, archive_id, error=None):
    from sqlalchemy import text
    params = {'s': state, 'a': archive_id, 'e': error, 't': now().isoformat(),
              'f': fixture['platform_fixture_id'], 'endpoint': endpoint}
    session.execute(text('''UPDATE fixture_endpoint_collection SET state=:s,archive_id=:a,
        error=:e,updated_at=:t WHERE fixture_id=:f AND endpoint=:endpoint'''), params)
    if endpoint == ENDPOINT:
        session.execute(text('''UPDATE player_collection SET state=:s,archive_id=:a,
            error=:e,updated_at=:t WHERE fixture_id=:f'''), params)


def archive_response(directory, fixture, payload, endpoint=ENDPOINT, fetched_at=None):
    from sqlalchemy import create_engine, text
    from sqlalchemy.orm import Session
    from Scripts.data_platform.storage.archive import LocalDiskStorage, PayloadArchiver
    from Scripts.data_platform.models import RawPayloadArchive
    class DurableStorage(LocalDiskStorage):
        def put(self, key, data, *, content_type='application/octet-stream'):
            path = self._path(key)
            path.parent.mkdir(parents=True, exist_ok=True)
            write_bytes(path, data)
            return path.resolve().as_uri()
    engine = create_engine(f"sqlite:///{directory / 'platform.db'}")
    try:
        with Session(engine) as session, session.begin():
            result = PayloadArchiver(session, DurableStorage(directory / 'raw_archive')).archive_json(
                provider='api_football', endpoint=endpoint, params={'fixture': fixture['fixture_id']}, payload=payload)
            if fetched_at is not None and result.was_new:
                captured = datetime.fromisoformat(fetched_at)
                if captured.tzinfo is None:
                    raise ValueError('Response capture timestamp must include its timezone')
                session.get(RawPayloadArchive, result.archive_id).fetched_at = captured
            update_state(session, fixture, endpoint, 'downloaded', result.archive_id)
        return result.archive_id
    finally:
        engine.dispose()


def read_archive(directory, archive_id, *, fixture=None, endpoint=None):
    with closing(readonly(directory / 'platform.db')) as db:
        row = db.execute('SELECT storage_uri,payload_digest,provider,endpoint,params FROM raw_payload_archive WHERE id=?', (archive_id,)).fetchone()
    if row is None:
        raise ValueError('Missing archived response metadata')
    if endpoint is not None and (row['provider'] != 'api_football' or row['endpoint'] != endpoint):
        raise ValueError('Archive endpoint mismatch')
    if fixture is not None and str(json.loads(row['params']).get('fixture')) != str(fixture['fixture_id']):
        raise ValueError('Archive fixture mismatch')
    url = urlparse(row['storage_uri'])
    path = Path(unquote(url.path))
    if url.scheme != 'file' or url.netloc or not path.resolve().is_relative_to(directory / 'raw_archive'):
        raise ValueError('Archive reference escaped the collection directory')
    raw = gzip.decompress(path.read_bytes())
    if hashlib.sha256(raw).hexdigest() != row['payload_digest']:
        raise ValueError('Archived response checksum mismatch')
    return json.loads(raw)


def normalize_context(session, fixture, payload, endpoint, archive_id):
    """Store original lineup/event fields; do not infer minutes, zeros or card totals."""
    from sqlalchemy import text
    from Scripts.data_platform.sync.upserts import upsert_player
    fid = fixture['platform_fixture_id']
    team_ids = {fixture['home_team_id']: fixture['home_id'], fixture['away_team_id']: fixture['away_id']}
    if endpoint == '/fixtures/lineups':
        for block in payload['response']:
            tid = team_ids[block['team']['id']]
            session.execute(text('''INSERT OR REPLACE INTO fixture_lineup_teams
                VALUES (:f,:t,:formation,:coach,:raw,:a)'''),
                {'f': fid, 't': tid, 'formation': block.get('formation'),
                 'coach': encoded(block.get('coach')).decode(), 'raw': encoded(block).decode(), 'a': archive_id})
            for field, role in (('startXI', 'starter'), ('substitutes', 'bench')):
                for row in block[field]:
                    player = row['player']
                    stored = upsert_player(session, api_id=player['id'],
                                           name=player.get('name') or f"Player {player['id']}")
                    session.execute(text('''INSERT OR REPLACE INTO fixture_lineup_entries
                        VALUES (:f,:t,:p,:role,:raw,:a)'''),
                        {'f': fid, 't': tid, 'p': stored.id, 'role': role,
                         'raw': encoded(row).decode(), 'a': archive_id})
    else:
        for index, event in enumerate(payload['response']):
            session.execute(text('''INSERT OR REPLACE INTO fixture_events
                VALUES (:f,:a,:i,:t,:p,:assist,:elapsed,:extra,:type,:detail,:raw)'''),
                {'f': fid, 'a': archive_id, 'i': index, 't': team_ids.get((event.get('team') or {}).get('id')),
                 'p': (event.get('player') or {}).get('id'), 'assist': (event.get('assist') or {}).get('id'),
                 'elapsed': usable_event_time((event.get('time') or {}).get('elapsed')),
                 'extra': usable_event_time((event.get('time') or {}).get('extra')),
                 'type': event['type'], 'detail': event.get('detail'), 'raw': encoded(event).decode()})


def usable_event_time(value):
    """Unknown/negative source timing stays unknown; raw_json retains the value."""
    return value if type(value) is int and value >= 0 else None


def normalize_response(directory, fixture, archive_id, endpoint=ENDPOINT):
    from sqlalchemy import create_engine, select, text, delete
    from sqlalchemy.orm import Session
    from Scripts.data_platform.models import Fixture, FixturePlayerStats, Player
    from Scripts.data_platform.sync.upserts import upsert_fixture_player_stats
    payload = read_archive(directory, archive_id, fixture=fixture, endpoint=endpoint)
    try:
        validator = {ENDPOINT: validate_players, '/fixtures/lineups': validate_lineups,
                     '/fixtures/events': validate_events}[endpoint]
        state = validator(fixture, payload)
        error = None
    except (ValueError, TypeError, KeyError, AttributeError, ApiFootballResponseError) as exc:
        state, error = 'invalid', str(exc)
    engine = create_engine(f"sqlite:///{directory / 'platform.db'}")
    try:
        with Session(engine) as session, session.begin():
            if state in ('ready', 'partial'):
                saved = session.get(Fixture, fixture['platform_fixture_id'])
                if (saved is None or saved.api_football_id != fixture['fixture_id'] or
                        saved.home_team_id != fixture['home_id'] or saved.away_team_id != fixture['away_id']):
                    raise ValueError('Staging fixture identity mismatch')
            if state == 'ready' and endpoint == ENDPOINT:
                # Derived staging rows can be rebuilt from their original response.
                # Deleting only this fixture's derived rows avoids stale columns/roster entries.
                session.execute(delete(FixturePlayerStats).where(FixturePlayerStats.fixture_id == saved.id))
                upsert_fixture_player_stats(session, fixture=saved, players_response=normalization_payload(payload))
                session.flush()
                raw = {p['player']['id']: p['statistics'][0] for b in payload['response'] for p in b['players']}
                for row, pid in session.execute(select(FixturePlayerStats, Player.api_football_id).join(
                        Player, Player.id == FixturePlayerStats.player_id).where(FixturePlayerStats.fixture_id == saved.id)):
                    # Retain every provider field, including fields without a dedicated SQL column.
                    row.stats_json = {**(row.stats_json or {}), 'provider_statistics': raw[pid],
                                      'source_archive_id': archive_id, 'source_fixture_id': fixture['fixture_id']}
            elif state in ('ready', 'partial'):
                tables = ('fixture_lineup_entries', 'fixture_lineup_teams') if endpoint == '/fixtures/lineups' else ('fixture_events',)
                for table in tables:
                    session.execute(text(f'DELETE FROM {table} WHERE fixture_id=:f'), {'f': saved.id})
                normalize_context(session, fixture, payload, endpoint, archive_id)
            update_state(session, fixture, endpoint, state, archive_id, error)
        return state
    finally:
        engine.dispose()


def summary(directory, *, stopped=None):
    with closing(readonly(directory / 'platform.db')) as db:
        states = dict(db.execute('SELECT state,count(*) FROM fixture_endpoint_collection GROUP BY state').fetchall())
        endpoints = {}
        for endpoint, state, count in db.execute('''SELECT endpoint,state,count(*)
                FROM fixture_endpoint_collection GROUP BY endpoint,state ORDER BY endpoint,state'''):
            endpoints.setdefault(endpoint, {})[state] = count
        attempts = dict(db.execute('SELECT day,attempts FROM player_collection_budget ORDER BY day').fetchall())
        rows = db.execute('SELECT count(*) FROM fixture_player_stats').fetchone()[0]
        lineup_rows = db.execute('SELECT count(*) FROM fixture_lineup_entries').fetchone()[0]
        event_rows = db.execute('SELECT count(*) FROM fixture_events').fetchone()[0]
        foreign_key_errors = len(db.execute('PRAGMA foreign_key_check').fetchall())
    value = {'schema': SCHEMA, 'states': states, 'states_by_endpoint': endpoints, 'http_attempts_by_utc_day': attempts,
             'linked_player_rows': rows, 'foreign_key_errors': foreign_key_errors,
             'linked_lineup_rows': lineup_rows, 'linked_event_rows': event_rows,
             'stopped': stopped, 'canonical_imported': False,
             'card_target_eligibility': 'not_assessed', 'updated_at': now().isoformat()}
    write_json(directory / 'summary.json', value)
    print(json.dumps(value, indent=2), flush=True)
    return value


def verify_saved(directory, fixtures, endpoints):
    """Verify the frozen identities and every checkpoint archive without HTTP."""
    by_id = {f['platform_fixture_id']: f for f in fixtures}
    with closing(readonly(directory / 'platform.db')) as db:
        if db.execute('PRAGMA quick_check').fetchone()[0] != 'ok' or db.execute('PRAGMA foreign_key_check').fetchone():
            raise ValueError('Staging database integrity check failed')
        rows = [dict(r) for r in db.execute('SELECT * FROM fixture_endpoint_collection')]
        actual = {(r['fixture_id'], r['endpoint']) for r in rows}
        expected = {(f, e) for f in by_id for e in endpoints}
        if actual != expected:
            raise ValueError('Endpoint checkpoints do not match the frozen fixture scope')
        for r in db.execute('''SELECT f.id AS platform_fixture_id,f.api_football_id AS fixture_id,
                f.home_team_id AS home_id,f.away_team_id AS away_id,f.season_id,f.competition_id,
                f.kickoff_utc,f.status,f.round,f.referee,c.code,c.api_football_id AS league_id,
                s.year AS season,h.api_football_id AS home_team_id,a.api_football_id AS away_team_id
                FROM fixtures f JOIN competitions c ON c.id=f.competition_id
                JOIN seasons s ON s.id=f.season_id JOIN teams h ON h.id=f.home_team_id
                JOIN teams a ON a.id=f.away_team_id'''):
            f = by_id.get(r['platform_fixture_id'])
            if f is None or any(f[k] != r[k] for k in r.keys()):
                raise ValueError('Staging fixture catalogue differs from its saved identity')
    checked = 0
    for row in rows:
        if row['state'] not in (*TERMINAL_STATES, 'pending', 'downloaded'):
            raise ValueError('Unknown endpoint checkpoint state')
        if row['state'] == 'pending':
            if row['archive_id'] is not None:
                raise ValueError('Pending checkpoint unexpectedly refers to an archive')
            continue
        if row['archive_id'] is None:
            raise ValueError('Completed/downloaded checkpoint has no source archive')
        read_archive(directory, row['archive_id'], fixture=by_id[row['fixture_id']], endpoint=row['endpoint'])
        checked += 1
        if checked % 1000 == 0:
            print(f'Archive verification: {checked} saved responses checked', flush=True)
    print(f'Archive verification: {checked} saved responses verified', flush=True)
    return checked


def pending_response(directory, fixture, endpoint, payload=None):
    """A durable response spool recovers interruption before the DB archive commit."""
    path = directory / 'pending-responses' / f"{fixture['fixture_id']}-{endpoint.rsplit('/', 1)[-1]}.json"
    if payload is not None:
        path.parent.mkdir(exist_ok=True)
        write_json(path, {'fixture_id': fixture['fixture_id'], 'endpoint': endpoint,
                         'fetched_at': now().isoformat(), 'payload': payload, 'sha256': digest(payload)})
    if not path.exists():
        return path, None
    saved = json.loads(path.read_text())
    if (saved['fixture_id'] != fixture['fixture_id'] or saved['endpoint'] != endpoint
            or saved['sha256'] != digest(saved['payload'])):
        raise ValueError('Pending response identity or checksum mismatch')
    return path, saved


def reprocess_saved(directory, fixtures, endpoints):
    """Rebuild derived rows from archives, never from another paid request."""
    by_id = {f['platform_fixture_id']: f for f in fixtures}
    with closing(readonly(directory / 'platform.db')) as db:
        rows = [dict(r) for r in db.execute('SELECT * FROM fixture_endpoint_collection WHERE archive_id IS NOT NULL')]
    for i, row in enumerate(rows, 1):
        if row['endpoint'] in endpoints:
            normalize_response(directory, by_id[row['fixture_id']], row['archive_id'], row['endpoint'])
        if i % 1000 == 0:
            print(f'Local replay: {i} saved responses processed', flush=True)
    return summary(directory, stopped=None)


def collect(directory, fixtures, *, key, daily_budget, reserve, rpm, max_fixtures=None,
            retry_unavailable=False, transport=None, endpoints=(ENDPOINT,)):
    network = transport or BudgetSession(directory / 'platform.db', key, daily_budget=daily_budget, reserve=reserve, rpm=rpm)
    stopped, processed, consecutive_gaps = None, 0, {e: 0 for e in endpoints}
    try:
        with closing(readonly(directory / 'platform.db')) as db:
            states = {(r['fixture_id'], r['endpoint']): dict(r)
                      for r in db.execute('SELECT * FROM fixture_endpoint_collection')}
        for f in fixtures:
            due = [(e, states[f['platform_fixture_id'], e]) for e in endpoints
                   if states[f['platform_fixture_id'], e]['state'] not in TERMINAL_STATES or
                   (retry_unavailable and states[f['platform_fixture_id'], e]['state'] in ('unavailable', 'empty_unverified'))]
            if not due:
                continue
            if max_fixtures is not None and processed >= max_fixtures:
                stopped = 'Requested fixture limit reached; rerun to resume'
                break
            for endpoint, saved in due:
                print(f"[{processed + 1}] {f['code']}:{f['season']} fixture={f['fixture_id']} {endpoint}", flush=True)
                archive_id = saved['archive_id']
                spool, response = pending_response(directory, f, endpoint)
                if saved['state'] != 'downloaded':
                    if response is None:
                        payload = get_json(BASE_URL + endpoint, params={'fixture': f['fixture_id']}, session=network)
                        spool, response = pending_response(directory, f, endpoint, payload)
                    archive_id = archive_response(directory, f, response['payload'], endpoint, response['fetched_at'])
                state = normalize_response(directory, f, archive_id, endpoint)
                if spool.exists():
                    spool.unlink()
                consecutive_gaps[endpoint] = consecutive_gaps[endpoint] + 1 if state == 'invalid' else 0
                if consecutive_gaps[endpoint] >= 10:
                    raise BudgetPause(f'Ten consecutive invalid responses from {endpoint}; inspect source structure before resuming')
            processed += 1
            if processed % 100 == 0:
                print(f'Checkpoint: {processed} fixtures handled in this invocation', flush=True)
    except BudgetPause as exc:
        stopped = str(exc)
    except KeyboardInterrupt:
        stopped = 'Interrupted; rerun the same command to resume'
    except (ApiFootballResponseError, requests.RequestException):
        # Do not print request objects, headers, keys or provider error bodies.
        stopped = 'Provider/network failure; check API access and quota, then rerun to resume'
    finally:
        network.close()
    return summary(directory, stopped=stopped)


def saved_scope(directory):
    database = directory / 'platform.db'
    if database.exists():
        with closing(readonly(database)) as db:
            if db.execute("SELECT name FROM sqlite_master WHERE name='player_collection_scope'").fetchone():
                row = db.execute('SELECT manifest,fixtures FROM player_collection_scope WHERE id=1').fetchone()
                if row:
                    return json.loads(row['manifest']), json.loads(row['fixtures'])
    # Interrupted first initialization can safely replay its frozen metadata.
    preparing = directory / 'preparing-scope.json'
    if preparing.exists():
        value = json.loads(preparing.read_text())
        return value['manifest'], value['fixtures']
    if (directory / 'manifest.json').exists():
        if not database.exists():
            raise ValueError('Saved collection database is missing; restore it before resuming, rather than redownloading')
        return (json.loads((directory / 'manifest.json').read_text()),
                json.loads((directory / 'fixture-identities.json').read_text()))
    return None


def prepare_scope(database, seasons, directory, endpoints, *, extend_scope=False, allow_local_replay=False):
    previous = saved_scope(directory)
    if previous is None:
        if any(p.name != 'collection.lock' for p in directory.iterdir()):
            raise ValueError('Unrecognized staging directory; preserve it and choose a new directory')
        fixtures = fixture_inventory(database, seasons)
        return plan(database, seasons, fixtures, endpoints), fixtures, None
    old, fixtures = previous
    old_endpoints = tuple(old.get('endpoints', (old.get('endpoint', ENDPOINT),)))
    if (old.get('schema') not in (SCHEMA, 'fixture-player-history-collection.v1')
            or old.get('source_database') != str(database.resolve())
            or old.get('fixture_identity_sha256') != digest(fixtures)
            or old.get('fixture_count') != len(fixtures)):
        raise ValueError('Collection scope/identity checksum mismatch')
    expected = plan(database, tuple(old['seasons']), fixtures, old_endpoints)
    if old['schema'] == SCHEMA and old != expected:
        identity_fields = set(expected) - {'source_sha256', 'dependencies'}
        if (not allow_local_replay or any(old.get(k) != expected[k] for k in identity_fields)):
            raise ValueError('Collection implementation changed; use --reprocess-saved for offline replay after review')
    changed = tuple(old['seasons']) != tuple(seasons) or old_endpoints != tuple(endpoints)
    if changed and not extend_scope:
        raise ValueError('Collection scope/identity changed; add --extend-scope for an additive extension')
    if not set(old['seasons']).issubset(seasons) or not set(old_endpoints).issubset(endpoints):
        raise ValueError('Scope extension cannot remove existing seasons or endpoints')
    if extend_scope:
        current = fixture_inventory(database, seasons)
        by_id = {f['fixture_id']: f for f in current}
        if any(by_id.get(f['fixture_id']) != f for f in fixtures):
            raise ValueError('Saved fixture metadata changed; reconcile identities before extending')
        fixtures = current
    return plan(database, seasons, fixtures, endpoints), fixtures, old


def run(root, database, seasons, directory, *, execute=False, daily_budget=DEFAULT_DAILY_BUDGET, reserve=DEFAULT_RESERVE,
        rpm=DEFAULT_RPM, max_fixtures=None, retry_unavailable=False, key=None, transport=None,
        endpoints=(ENDPOINT,), extend_scope=False, replay=False, verify_only=False):
    if (daily_budget < 1 or reserve < 0 or not 1 <= rpm <= MAX_RPM
            or (max_fixtures is not None and max_fixtures < 1)):
        raise ValueError('Invalid request budget, rate or fixture limit')
    if not seasons or len(seasons) > 8 or tuple(seasons) != tuple(range(seasons[0], seasons[-1] + 1)):
        raise ValueError('Specify one to eight consecutive provider seasons')
    if not endpoints or len(set(endpoints)) != len(endpoints) or not set(endpoints).issubset(ENDPOINTS):
        raise ValueError('Choose unique approved fixture endpoints')
    directory = safe_directory(root, directory)
    if replay and verify_only:
        raise ValueError('Choose replay or verification')
    if (replay or verify_only) and (extend_scope or retry_unavailable):
        raise ValueError('Offline archive operations cannot extend scope or retry HTTP requests')
    if not execute and not replay and not verify_only:
        fixtures = fixture_inventory(database, seasons)
        result = {**plan(database, seasons, fixtures, endpoints), 'dry_run': True, 'directory': str(directory),
                  'daily_budget': daily_budget, 'reserve': reserve, 'requests_per_minute': rpm}
        print(json.dumps(result, indent=2))
        return result
    if (replay or verify_only) and saved_scope(directory) is None:
        raise ValueError('No saved collection to verify/replay')
    if key is None and not replay and not verify_only:
        from dotenv import load_dotenv
        load_dotenv(root / '.env', override=False)
        key = os.getenv('API_FOOTBALL_KEY') or os.getenv('API-FOOTBALL-KEY')
    if not key and not replay and not verify_only:
        raise ValueError('Set API_FOOTBALL_KEY or API-FOOTBALL-KEY in the project .env')
    previous_mask = os.umask(0o077)
    try:
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / 'collection.lock').open('a+') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise ValueError('This collection is already running') from None
            replay_marker = directory / 'REPLAY_REQUIRED.json'
            if replay_marker.exists() and not replay:
                raise ValueError('Local replay was interrupted; complete --reprocess-saved before downloading')
            manifest, fixtures, previous = prepare_scope(database, seasons, directory, endpoints,
                extend_scope=extend_scope, allow_local_replay=replay)
            # Check an existing committed ledger BEFORE initialize can insert new
            # pending rows. Lost progress must not silently become another API job.
            preverified = False
            if (directory / 'platform.db').exists():
                with closing(readonly(directory / 'platform.db')) as check_db:
                    has_scope = check_db.execute("SELECT name FROM sqlite_master WHERE name='player_collection_scope'").fetchone()
                    committed = check_db.execute('SELECT fixtures,manifest FROM player_collection_scope WHERE id=1').fetchone() if has_scope else None
                if committed:
                    committed_manifest = json.loads(committed['manifest'])
                    verify_saved(directory, json.loads(committed['fixtures']), committed_manifest['endpoints'])
                    preverified = True
            if previous is not None and previous != manifest:
                history = directory / 'scope-history'
                history.mkdir(exist_ok=True)
                write_json(history / (digest(previous) + '.json'), {'manifest': previous, 'fixtures': saved_scope(directory)[1]})
            if replay:
                write_json(replay_marker, {'schema': SCHEMA, 'source_sha256': manifest['source_sha256'],
                                          'started_at': now().isoformat()})
            write_json(directory / 'preparing-scope.json', {'manifest': manifest, 'fixtures': fixtures})
            initialize(directory, fixtures, manifest)
            (directory / 'preparing-scope.json').unlink()
            if not preverified or previous != manifest:
                verify_saved(directory, fixtures, endpoints)
            if replay:
                result = reprocess_saved(directory, fixtures, endpoints)
                replay_marker.unlink()
                return result
            if verify_only:
                return summary(directory)
            print(f"Collecting {len(fixtures)} saved fixtures, {len(endpoints)} endpoints into {directory}; canonical database remains unchanged", flush=True)
            return collect(directory, fixtures, key=key, daily_budget=daily_budget, reserve=reserve, rpm=rpm,
                           max_fixtures=max_fixtures, retry_unavailable=retry_unavailable, transport=transport, endpoints=endpoints)
    finally:
        os.umask(previous_mask)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seasons', nargs='+', type=int, default=DEFAULT_SEASONS)
    parser.add_argument('--endpoints', nargs='+', choices=['players', 'lineups', 'events'], default=['players', 'lineups', 'events'])
    parser.add_argument('--directory', type=Path)
    parser.add_argument('--daily-budget', type=int, default=DEFAULT_DAILY_BUDGET)
    parser.add_argument('--reserve', type=int, default=DEFAULT_RESERVE, help='Stop at this provider-reported remaining daily quota')
    parser.add_argument('--rpm', type=int, default=DEFAULT_RPM,
                        help=f'Maximum evenly paced requests/minute (1–{MAX_RPM}; default: {DEFAULT_RPM})')
    parser.add_argument('--max-fixtures', type=int, help='Optional bounded pilot; rerun without this flag for the remainder')
    parser.add_argument('--retry-unavailable', action='store_true', help='Retry previously empty responses; invalid responses require review')
    parser.add_argument('--extend-scope', action='store_true', help='Append newly completed fixtures/seasons/endpoints to this staging collection')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--execute', action='store_true')
    mode.add_argument('--dry-run', action='store_true', help='Read-only plan; default')
    mode.add_argument('--verify-saved', action='store_true', help='Verify saved archive hashes and identities without API requests')
    mode.add_argument('--reprocess-saved', action='store_true', help='Rebuild staging rows from saved responses, without API requests')
    args = parser.parse_args(argv)
    directory = args.directory or Path(f'Index/history_staging/player-stats-{args.seasons[0]}-{args.seasons[-1]}')
    try:
        result = run(ROOT, ROOT / 'Index/platform.db', tuple(args.seasons), directory,
                     execute=args.execute, daily_budget=args.daily_budget, reserve=args.reserve,
                     rpm=args.rpm, max_fixtures=args.max_fixtures, retry_unavailable=args.retry_unavailable,
                     endpoints=tuple('/fixtures/' + e for e in args.endpoints), extend_scope=args.extend_scope,
                     replay=args.reprocess_saved, verify_only=args.verify_saved)
    except (ValueError, OSError, sqlite3.Error) as exc:
        print(f'Player collection stopped: {exc}', file=sys.stderr)
        return 1
    if args.execute or args.reprocess_saved or args.verify_saved:
        states = result['states']
        if result['stopped'] or states.get('pending') or states.get('downloaded'):
            return 2
        if any(states.get(s) for s in ('invalid', 'unavailable', 'partial', 'empty_unverified')) or result['foreign_key_errors']:
            return 3
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
