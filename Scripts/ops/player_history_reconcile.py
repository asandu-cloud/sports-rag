"""Offline player-history preparation, additive import and development card labels.

No provider requests, embedding, profile builds, backtests or model activation.
prepare produces a new sealed copy; apply uses the normal refresh writer gate.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import closing
import fcntl
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
from urllib.parse import unquote, urlparse

from Scripts.ops import player_history_download as source
from Scripts.ops.prediction_baseline import sqlite_backup
from Scripts.data_platform.features import player_history_reconciliation as research
from Scripts.data_platform.features import phase4_cards as cards

ROOT = Path(__file__).resolve().parents[2]
SOURCES = ('Scripts/ops/player_history_reconcile.py',
           'Scripts/data_platform/features/player_history_reconciliation.py',
           *source.SOURCES,
           'Scripts/data_platform/features/phase4_card_policy.py',
           'Scripts/data_platform/features/phase4_cards.py',
           'Scripts/data_platform/features/market_eligibility.py',
           'Scripts/data_platform/participation_cards.py')


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def write(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write('\n')


def new_output(path):
    path = Path(path).absolute()
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError('Output must not use symlinks')
    path.mkdir(parents=True, exist_ok=False, mode=0o700)
    return path.resolve()


def prepare(collection, output):
    collection = Path(collection).resolve()
    with (collection / 'collection.lock').open('r') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest, fixtures = source.saved_scope(collection)
        if tuple(manifest['endpoints']) != source.ENDPOINTS:
            raise ValueError('All three endpoints required for this preparation')
        output = new_output(output)
        write(output / 'rules.json', research.RULES)  # Freeze before processing data.
        write(output / 'collection-manifest.json', manifest)
        write(output / 'fixture-identities.json', fixtures)
        write(output / 'source-hashes.json', {name: sha(ROOT / name) for name in SOURCES})
        for name in SOURCES:
            dest = output / 'source' / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, dest)
        original_hash = sha(collection / 'platform.db')
        sqlite_backup(collection / 'platform.db', output / 'platform.db')
        with closing(source.readonly(output / 'platform.db')) as db:
            archives = {r['id']: dict(r) for r in db.execute('SELECT * FROM raw_payload_archive')}
            progress = [dict(r) for r in db.execute('SELECT * FROM fixture_endpoint_collection ORDER BY fixture_id,endpoint')]
        expected = {(f['platform_fixture_id'], ep) for f in fixtures for ep in source.ENDPOINTS}
        if ({(r['fixture_id'], r['endpoint']) for r in progress} != expected
                or any(r['state'] not in source.TERMINAL_STATES or r['archive_id'] is None for r in progress)):
            raise ValueError('Collection has incomplete or mismatched checkpoints')
        fixture_map = {f['platform_fixture_id']: f for f in fixtures}
        coverage, reasons, changes = Counter(), Counter(), Counter()
        archive_index, repairs = [], []
        with sqlite3.connect(output / 'platform.db') as db, (output / 'source-review.jsonl').open('x') as report:
            for i, checkpoint in enumerate(progress, 1):
                f = fixture_map[checkpoint['fixture_id']]
                a = archives[checkpoint['archive_id']]
                payload = source.read_archive(collection, a['id'], fixture=f, endpoint=checkpoint['endpoint'])
                validated = research.classify(f, payload, checkpoint['endpoint'])
                raw_path = Path(unquote(urlparse(a['storage_uri']).path))
                relative = Path('raw_archive') / a['endpoint'].strip('/') / (a['payload_digest'] + '.json.gz')
                dest = output / relative
                dest.parent.mkdir(parents=True, exist_ok=True)
                if not dest.exists():
                    shutil.copyfile(raw_path, dest)
                if sha(raw_path) != sha(dest):
                    raise ValueError('Copied archive differs from source')
                db.execute('UPDATE raw_payload_archive SET storage_uri=? WHERE id=?', (dest.as_uri(), a['id']))
                archive_index.append({'id': a['id'], 'path': str(relative), 'sha256': sha(dest),
                                      'payload_sha256': a['payload_digest']})
                record = {'fixture_id': f['fixture_id'], 'competition': f['code'], 'season': f['season'],
                          'endpoint': checkpoint['endpoint'], 'archive_id': a['id'],
                          'original_state': checkpoint['state'], **validated}
                report.write(json.dumps(record, sort_keys=True) + '\n')
                coverage[(f['code'], f['season'], checkpoint['endpoint'], validated['state'])] += 1
                reasons.update(validated['reasons'])
                if checkpoint['state'] != validated['state']:
                    if checkpoint['state'] != 'invalid' or validated['state'] != 'partial' or a['endpoint'] != '/fixtures/events':
                        raise ValueError('Unexpected repair; needs an explicit reviewed adapter')
                    repairs.append((f, a['id'], a['endpoint']))
                    changes[checkpoint['state'] + '->' + validated['state']] += 1
                if i % 5000 == 0:
                    print(f'Verified and copied {i} responses', flush=True)
        for i, (f, aid, ep) in enumerate(repairs, 1):
            if source.normalize_response(output, f, aid, ep) != 'partial':
                raise ValueError('Repaired timeline did not retain its uncertainty')
            if i % 100 == 0:
                print(f'Retained {i} uncertain timelines in prepared copy', flush=True)
        with closing(source.readonly(output / 'platform.db')) as db:
            if db.execute('PRAGMA foreign_key_check').fetchone():
                raise ValueError('Prepared foreign-key violation')
        if sha(collection / 'platform.db') != original_hash:
            raise ValueError('Original collection changed during preparation')
        write(output / 'archives.json', archive_index)
        report = {'version': research.VERSION, 'source_database_sha256': original_hash,
                  'fixture_count': len(fixtures), 'verified_responses': len(progress),
                  'repairs': dict(changes), 'reasons': dict(reasons),
                  'coverage': [{'competition': c, 'season': s, 'endpoint': e, 'state': state, 'count': n}
                               for (c, s, e, state), n in sorted(coverage.items())],
                  'provider_requests': 0, 'original_collection_changed': False}
        write(output / 'report.json', report)
        write(output / 'PREPARED.json', {p.name: sha(p) for p in output.iterdir() if p.is_file()})
        return report


def verify(package, *, archives=True):
    package = Path(package).resolve()
    manifest = json.loads((package / 'PREPARED.json').read_text())
    for name, digest in manifest.items():
        p = package / name
        if Path(name).name != name or p.is_symlink() or sha(p) != digest:
            raise ValueError('Prepared artifact checksum mismatch: ' + name)
    if json.loads((package / 'rules.json').read_text()) != research.RULES:
        raise ValueError('Prepared research rules differ from running code')
    hashes = json.loads((package / 'source-hashes.json').read_text())
    if any(sha(ROOT / name) != expected for name, expected in hashes.items()):
        raise ValueError('Code changed since preparation; use the preserved code or a new preparation')
    if archives:
        for a in json.loads((package / 'archives.json').read_text()):
            p = package / a['path']
            if not p.resolve().is_relative_to(package / 'raw_archive') or p.is_symlink() or sha(p) != a['sha256']:
                raise ValueError('Prepared archive checksum/path mismatch')
    return source.digest(manifest)


SCHEMA = (
    '''CREATE TABLE IF NOT EXISTS player_history_batches (
      batch_id TEXT PRIMARY KEY, version TEXT NOT NULL, prepared_path TEXT NOT NULL,
      imported_at TEXT NOT NULL, report_json TEXT NOT NULL)''',
    '''CREATE TABLE IF NOT EXISTS player_history_responses (
      batch_id TEXT NOT NULL REFERENCES player_history_batches(batch_id),
      fixture_id INTEGER NOT NULL REFERENCES fixtures(id), endpoint TEXT NOT NULL,
      archive_id INTEGER NOT NULL REFERENCES raw_payload_archive(id), state TEXT NOT NULL,
      error TEXT, PRIMARY KEY(batch_id,fixture_id,endpoint))''',
    '''CREATE TABLE IF NOT EXISTS player_history_observations (
      batch_id TEXT NOT NULL REFERENCES player_history_batches(batch_id),
      fixture_id INTEGER NOT NULL REFERENCES fixtures(id), team_id INTEGER NOT NULL REFERENCES teams(id),
      player_id INTEGER NOT NULL REFERENCES players(id), archive_id INTEGER NOT NULL REFERENCES raw_payload_archive(id),
      stats_json TEXT NOT NULL, existing_row_id INTEGER REFERENCES fixture_player_stats(id),
      differences_json TEXT NOT NULL, PRIMARY KEY(batch_id,fixture_id,player_id))''',
    '''CREATE TABLE IF NOT EXISTS player_history_lineups (
      batch_id TEXT NOT NULL REFERENCES player_history_batches(batch_id),
      fixture_id INTEGER NOT NULL REFERENCES fixtures(id), team_id INTEGER NOT NULL REFERENCES teams(id),
      player_id INTEGER NOT NULL REFERENCES players(id), archive_id INTEGER NOT NULL REFERENCES raw_payload_archive(id),
      role TEXT NOT NULL, raw_json TEXT NOT NULL, PRIMARY KEY(batch_id,fixture_id,player_id))''',
    '''CREATE TABLE IF NOT EXISTS player_history_lineup_teams (
      batch_id TEXT NOT NULL REFERENCES player_history_batches(batch_id),
      fixture_id INTEGER NOT NULL REFERENCES fixtures(id), team_id INTEGER NOT NULL REFERENCES teams(id),
      archive_id INTEGER NOT NULL REFERENCES raw_payload_archive(id), formation TEXT, coach_json TEXT, raw_json TEXT NOT NULL,
      PRIMARY KEY(batch_id,fixture_id,team_id))''',
    '''CREATE TABLE IF NOT EXISTS player_history_events (
      batch_id TEXT NOT NULL REFERENCES player_history_batches(batch_id),
      fixture_id INTEGER NOT NULL REFERENCES fixtures(id), archive_id INTEGER NOT NULL REFERENCES raw_payload_archive(id),
      event_index INTEGER NOT NULL, team_id INTEGER REFERENCES teams(id), player_api_id INTEGER, assist_api_id INTEGER,
      elapsed INTEGER, extra INTEGER, event_type TEXT NOT NULL, detail TEXT, raw_json TEXT NOT NULL,
      PRIMARY KEY(batch_id,fixture_id,archive_id,event_index))''',
)


def import_transaction(database, package, batch, archive_root, *, fault=None):
    """Add observations atomically; every pre-existing canonical row is preserved.

    Caller holds the writer gate and has taken a verified backup. Used identically
    on a disposable rehearsal database before canonical application.
    """
    timestamp = source.now().isoformat()
    with closing(sqlite3.connect(database, uri=True, timeout=5)) as db:
        db.row_factory = sqlite3.Row
        db.execute('PRAGMA foreign_keys=ON')
        db.execute('ATTACH DATABASE ? AS incoming', ((package / 'platform.db').as_uri() + '?mode=ro',))
        db.execute('BEGIN IMMEDIATE')
        try:
            for statement in SCHEMA:
                db.execute(statement)
            previous = db.execute('SELECT report_json FROM player_history_batches WHERE batch_id=?', (batch,)).fetchone()
            if previous:
                db.rollback()
                return {**json.loads(previous[0]), 'already_imported': True}
            db.execute('INSERT INTO player_history_batches VALUES (?,?,?,?,?)',
                       (batch, research.VERSION, str(package), timestamp, '{}'))
            db.execute('''CREATE TEMP TABLE fixture_map AS SELECT s.id AS sid,t.id AS cid
                FROM incoming.fixtures s JOIN fixtures t ON t.api_football_id=s.api_football_id
                JOIN incoming.teams sh ON sh.id=s.home_team_id JOIN teams th ON th.id=t.home_team_id
                JOIN incoming.teams sa ON sa.id=s.away_team_id JOIN teams ta ON ta.id=t.away_team_id
                JOIN incoming.competitions sc ON sc.id=s.competition_id JOIN competitions tc ON tc.id=t.competition_id
                JOIN incoming.seasons ss ON ss.id=s.season_id JOIN seasons ts ON ts.id=t.season_id
                WHERE sh.api_football_id=th.api_football_id AND sa.api_football_id=ta.api_football_id
                  AND sc.api_football_id=tc.api_football_id AND sc.code=tc.code AND ss.year=ts.year
                  AND s.status=t.status AND s.round IS t.round AND julianday(s.kickoff_utc)=julianday(t.kickoff_utc)''')
            if db.execute('SELECT count(*) FROM fixture_map').fetchone()[0] != db.execute('SELECT count(*) FROM incoming.fixtures').fetchone()[0]:
                raise ValueError('Canonical fixture identity/status/round changed; review before import')
            db.execute('CREATE UNIQUE INDEX temp.ix_fmap ON fixture_map(sid)')
            db.execute('CREATE TEMP TABLE team_map AS SELECT s.id sid,t.id cid FROM incoming.teams s JOIN teams t ON t.api_football_id=s.api_football_id')
            db.execute('CREATE UNIQUE INDEX temp.ix_tmap ON team_map(sid)')
            if db.execute('SELECT count(*) FROM team_map').fetchone()[0] != db.execute('SELECT count(*) FROM incoming.teams').fetchone()[0]:
                raise ValueError('Canonical team identity missing')
            added_players = db.execute('''INSERT INTO players (api_football_id,name,position,nationality,birth_date,height_cm,weight_kg,created_at,updated_at)
                SELECT api_football_id,name,position,nationality,birth_date,height_cm,weight_kg,created_at,updated_at
                FROM incoming.players s WHERE NOT EXISTS (SELECT 1 FROM players t WHERE t.api_football_id=s.api_football_id)''').rowcount
            db.execute('CREATE TEMP TABLE player_map AS SELECT s.id sid,t.id cid FROM incoming.players s JOIN players t ON t.api_football_id=s.api_football_id')
            db.execute('CREATE UNIQUE INDEX temp.ix_pmap ON player_map(sid)')
            archive_map, added_archives = {}, 0
            known = {(r['provider'], r['endpoint'], r['params_digest'], r['payload_digest']): r['id']
                     for r in db.execute('SELECT * FROM raw_payload_archive')}
            archive_cols = [r[1] for r in db.execute('PRAGMA main.table_info(raw_payload_archive)') if r[1] != 'id']
            for a in db.execute('SELECT * FROM incoming.raw_payload_archive').fetchall():
                key = tuple(a[k] for k in ('provider', 'endpoint', 'params_digest', 'payload_digest'))
                if key not in known:
                    oldpath = Path(unquote(urlparse(a['storage_uri']).path))
                    rel = oldpath.relative_to(package / 'raw_archive')
                    dest = archive_root / rel
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    if not dest.exists():
                        shutil.copyfile(oldpath, dest)
                    if sha(dest) != sha(oldpath):
                        raise ValueError('Canonical archive copy mismatch')
                    values = dict(a)
                    values.update(storage_uri=dest.resolve().as_uri(), sync_run_id=None)
                    cur = db.execute('INSERT INTO raw_payload_archive (' + ','.join(archive_cols) + ') VALUES (' + ','.join('?' for _ in archive_cols) + ')',
                                     [values[k] for k in archive_cols])
                    known[key] = cur.lastrowid
                    added_archives += 1
                archive_map[a['id']] = known[key]
            db.execute('CREATE TEMP TABLE archive_map(sid INTEGER PRIMARY KEY,cid INTEGER NOT NULL)')
            db.executemany('INSERT INTO archive_map VALUES (?,?)', archive_map.items())
            db.execute('''INSERT INTO player_history_responses SELECT ?,f.cid,c.endpoint,a.cid,c.state,c.error
                FROM incoming.fixture_endpoint_collection c JOIN fixture_map f ON f.sid=c.fixture_id
                JOIN archive_map a ON a.sid=c.archive_id''', (batch,))
            compare_cols = ('team_id','position','minutes','rating','captain','substitute','goals','assists','shots_total','shots_on',
                            'passes_total','passes_accurate','pass_accuracy','tackles','interceptions','duels_won','duels_total',
                            'yellow_cards','red_cards','fouls_committed','fouls_drawn')
            # Read only differences' column names into the review, not reserved
            # outcome values. All incoming values remain versioned source evidence.
            different = "json_array(" + ','.join(
                f"CASE WHEN {'t.cid' if c == 'team_id' else 's.' + c} IS NOT old.{c} THEN '{c}' END" for c in compare_cols) + ')'
            observation_sql = f'''INSERT INTO player_history_observations
                SELECT ?,f.cid,t.cid,p.cid,a.cid,
                  json_set(s.stats_json,'$.source_archive_id',a.cid),old.id,
                  CASE WHEN old.id IS NULL THEN '[]' ELSE
                    (SELECT json_group_array(value) FROM json_each({different}) WHERE value IS NOT NULL) END
                FROM incoming.fixture_player_stats s
                JOIN fixture_map f ON f.sid=s.fixture_id JOIN team_map t ON t.sid=s.team_id
                JOIN player_map p ON p.sid=s.player_id
                JOIN archive_map a ON a.sid=json_extract(s.stats_json,'$.source_archive_id')
                LEFT JOIN fixture_player_stats old ON old.fixture_id=f.cid AND old.player_id=p.cid'''
            db.execute(observation_sql, (batch,))
            expected = db.execute('SELECT count(*) FROM incoming.fixture_player_stats').fetchone()[0]
            if db.execute('SELECT count(*) FROM player_history_observations WHERE batch_id=?', (batch,)).fetchone()[0] != expected:
                raise ValueError('Lost player provenance join')
            cols = [r[1] for r in db.execute('PRAGMA main.table_info(fixture_player_stats)') if r[1] != 'id']
            replacements = {'fixture_id':'f.cid','player_id':'p.cid','team_id':'t.cid',
                            'stats_json':"json_set(s.stats_json,'$.source_archive_id',a.cid)"}
            selected = ','.join(replacements.get(c, 's.' + c) for c in cols)
            added_rows = db.execute(f'''INSERT INTO fixture_player_stats ({','.join(cols)}) SELECT {selected}
                FROM incoming.fixture_player_stats s JOIN fixture_map f ON f.sid=s.fixture_id
                JOIN team_map t ON t.sid=s.team_id JOIN player_map p ON p.sid=s.player_id
                JOIN archive_map a ON a.sid=json_extract(s.stats_json,'$.source_archive_id')
                WHERE NOT EXISTS (SELECT 1 FROM fixture_player_stats old WHERE old.fixture_id=f.cid AND old.player_id=p.cid)''').rowcount
            db.execute('''INSERT INTO player_history_lineups SELECT ?,f.cid,t.cid,p.cid,a.cid,s.role,s.raw_json
                FROM incoming.fixture_lineup_entries s JOIN fixture_map f ON f.sid=s.fixture_id
                JOIN team_map t ON t.sid=s.team_id JOIN player_map p ON p.sid=s.player_id JOIN archive_map a ON a.sid=s.archive_id''', (batch,))
            db.execute('''INSERT INTO player_history_lineup_teams SELECT ?,f.cid,t.cid,a.cid,s.formation,s.coach_json,s.raw_json
                FROM incoming.fixture_lineup_teams s JOIN fixture_map f ON f.sid=s.fixture_id
                JOIN team_map t ON t.sid=s.team_id JOIN archive_map a ON a.sid=s.archive_id''', (batch,))
            db.execute('''INSERT INTO player_history_events SELECT ?,f.cid,a.cid,s.event_index,t.cid,s.player_api_id,s.assist_api_id,
                s.elapsed,s.extra,s.event_type,s.detail,s.raw_json FROM incoming.fixture_events s
                JOIN fixture_map f ON f.sid=s.fixture_id JOIN archive_map a ON a.sid=s.archive_id
                LEFT JOIN team_map t ON t.sid=s.team_id''', (batch,))
            if fault:
                fault(db)
            if db.execute('PRAGMA main.foreign_key_check').fetchone():
                raise ValueError('Import foreign-key validation failed')
            report = {'batch_id': batch, 'players_added': added_players, 'player_rows_added': added_rows,
                      'incoming_player_observations': expected, 'archives_added': added_archives,
                      'existing_differing_player_rows_retained': db.execute("SELECT count(*) FROM player_history_observations WHERE batch_id=? AND differences_json!='[]'", (batch,)).fetchone()[0],
                      'foreign_key_errors': 0, 'old_rows_overwritten': 0, 'provider_requests': 0}
            db.execute('UPDATE player_history_batches SET report_json=? WHERE batch_id=?', (json.dumps(report), batch))
            db.commit()
            return report
        except BaseException:
            db.rollback()
            raise


def preserved_rows(database, backup):
    """Check every old row, including publications/profiles/results, byte-for-value."""
    with closing(sqlite3.connect(database, uri=True)) as db:
        db.execute('ATTACH DATABASE ? AS prior', (Path(backup).resolve().as_uri() + '?mode=ro',))
        names = [r[0] for r in db.execute("SELECT name FROM prior.sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")]
        for name in names:
            quoted = '"' + name.replace('"','""') + '"'
            if db.execute(f'SELECT * FROM prior.{quoted} EXCEPT SELECT * FROM main.{quoted} LIMIT 1').fetchone():
                raise ValueError('Pre-existing row changed in ' + name)
    return len(names)


def apply(package, database, output, *, rehearsal=False):
    package, database = Path(package).resolve(), Path(database).resolve()
    batch = verify(package)
    output = new_output(output)
    from contextlib import nullcontext
    from Scripts.data_platform.services.refresh_coordination import data_access, RefreshBusy
    if not rehearsal:
        from Scripts.data_platform.config import SETTINGS
        from sqlalchemy.engine import make_url
        configured = make_url(SETTINGS.database_url)
        if configured.get_backend_name() != 'sqlite' or Path(configured.database).resolve() != database:
            raise ValueError('Writer gate and target database must match')
    gate = nullcontext() if rehearsal else data_access(writer=True, wait_seconds=1)
    try:
        with gate:
            # An interrupted refresh is not made healthy by a history import.
            if not rehearsal and Path(str(database) + '.refresh-incomplete').exists():
                raise RefreshBusy('Existing incomplete refresh must finish before canonical import')
            backup = output / 'platform-before.db'
            sqlite_backup(database, backup)
            target = output / 'platform-rehearsal.db' if rehearsal else database
            if rehearsal:
                shutil.copyfile(backup, target)
            write(output / 'before.json', {'database': str(database), 'backup': str(backup),
                                           'backup_sha256': sha(backup), 'batch': batch, 'rehearsal': rehearsal})
            archive_root = (output / 'raw_archive') if rehearsal else database.parent / 'raw_archive' / ('player-history-' + batch[:16])
            report = import_transaction(target, package, batch, archive_root)
            report['preserved_tables_verified'] = preserved_rows(target, backup)
            report.update(canonical_imported=not rehearsal, rehearsal=rehearsal)
            write(output / 'result.json', report)
            return report
    except RefreshBusy as exc:
        report = {'canonical_imported': False, 'status': 'refresh_gate_blocked', 'reason': str(exc),
                  'prepared': str(package), 'not_automatically_queued': True}
        write(output / 'blocked.json', report)
        return report


def reconstruct(package, canonical, output):
    package = Path(package).resolve()
    batch = verify(package)
    output = new_output(output)
    write(output / 'rules.json', research.RULES)
    fixtures = json.loads((package / 'fixture-identities.json').read_text())
    permitted = [f for f in fixtures if cards.permitted(f['kickoff_utc'])]
    # Filter on kickoff in SQL before reading any player counts/response bodies.
    old = defaultdict(list)
    with closing(source.readonly(Path(canonical))) as db:
        for r in db.execute('''SELECT f.api_football_id fixture_id,t.api_football_id team_id,p.api_football_id player_id,
            x.minutes,x.yellow_cards,x.red_cards FROM fixture_player_stats x JOIN fixtures f ON f.id=x.fixture_id
            JOIN teams t ON t.id=x.team_id JOIN players p ON p.id=x.player_id WHERE f.kickoff_utc<'2023-12-31 21:00:00' '''):
            old[r['fixture_id']].append(dict(r))
    with closing(source.readonly(package / 'platform.db')) as db:
        references = {(r['fixture_id'], r['endpoint']): dict(r) for r in db.execute('''SELECT c.fixture_id,c.endpoint,c.archive_id,
            a.payload_digest,a.fetched_at FROM fixture_endpoint_collection c JOIN raw_payload_archive a ON a.id=c.archive_id''')}
    counts, reasons, coverage = Counter(), Counter(), defaultdict(Counter)
    with (output / 'card-targets.jsonl').open('x') as handle:
        for i, f in enumerate(permitted, 1):
            scoped = research.policy.scope(research.fixture_metadata(f))
            payloads, refs = {}, {}
            if scoped['eligible']:
                for ep in source.ENDPOINTS:
                    ref = references[f['platform_fixture_id'], ep]
                    refs[ep] = ref
                    payloads[ep] = source.read_archive(package, ref['archive_id'], fixture=f, endpoint=ep)
            r = research.reconstruct(f, payloads, refs, existing_rows=old[f['fixture_id']])
            handle.write(json.dumps(r, sort_keys=True) + '\n')
            label = 'qualified' if r['eligible'] else 'pending_or_excluded'
            counts[label] += 1
            reasons.update(r['exclusions'])
            coverage[f"{f['code']}:{f['season']}"][label] += 1
            if i % 2000 == 0:
                print(f'Reconstructed {i} permitted fixtures', flush=True)
    report = {'version': research.VERSION, 'prepared_batch': batch, 'counts': dict(counts),
              'exclusions': dict(reasons), 'coverage': {k:dict(v) for k,v in sorted(coverage.items())},
              'reserved_fixtures_not_decoded_for_targets': len(fixtures)-len(permitted),
              'canonical_comparison_source': str(Path(canonical).resolve()),
              'profile_builds': False, 'backtests_run': False, 'weights_optimized': False, 'production_policy_changed': False}
    write(output / 'report.json', report)
    write(output / 'COMPLETE.json', {p.name:sha(p) for p in output.iterdir() if p.is_file()})
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare','rehearse','apply','reconstruct','verify'))
    parser.add_argument('--collection', type=Path, default=ROOT / 'Index/history_staging/player-stats-2021-2026')
    parser.add_argument('--package', type=Path)
    parser.add_argument('--database', type=Path, default=ROOT / 'Index/platform.db')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    if args.action == 'prepare':
        result = prepare(args.collection, args.output)
    elif args.action == 'verify':
        result = {'prepared_batch': verify(args.package)}
    elif args.action == 'reconstruct':
        result = reconstruct(args.package, args.database, args.output)
    else:
        result = apply(args.package, args.database, args.output, rehearsal=args.action=='rehearse')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
