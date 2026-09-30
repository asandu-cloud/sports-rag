"""Bounded Phase 4 research: read-only card audit and development dispersion fit.

Run with python -B -m Scripts.ops.phase4_development --output NEW_DIRECTORY.
Never fetches data, imports publisher services or changes model artifacts.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import closing
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sqlite3
import sys
from urllib.parse import unquote, urlparse

from Scripts.data_platform.features import phase4_cards as cards
from Scripts.data_platform.features import count_calibration as calibration
from Scripts.data_platform.features.benchmarks.isolation import offline_guard

ROOT = Path(__file__).resolve().parents[2]
DATASET = Path('Index/prediction_experiments/phase3-dataset-2026-09-26')
FORECASTS = Path('Research/prior-transition-2026-09-28/evaluation')
PROTOCOL = Path('docs/phase4-card-calibration-protocol-2026-09-28.md')
LEAGUES = ('EPL', 'LaLiga', 'SerieA', 'Bundesliga', 'Ligue1', 'Championship', 'SuperLig',
           'Eredivisie', 'PrimeiraLiga', 'BelgianProLeague', 'UCL', 'UEL', 'UECL')
CUTOFF = '2023-12-31T21:00:00+00:00'


def sha(body):
    return hashlib.sha256(body).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n')


def write_rows(path, rows):
    with path.open('w') as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True, allow_nan=False) + '\n')


def verified(directory, name):
    complete = json.loads((directory / 'COMPLETE.json').read_bytes())
    body = (directory / name).read_bytes()
    if complete.get(name) != sha(body):
        raise ValueError('Input checksum mismatch: ' + str(directory / name))
    return body


def snapshot(database):
    """Outcome SQL joins enforce cutoff BEFORE values leave SQLite."""
    with closing(sqlite3.connect(database.resolve().as_uri() + '?mode=ro', uri=True)) as db:
        db.row_factory = sqlite3.Row
        db.execute('PRAGMA query_only=ON')
        db.execute('BEGIN')
        fixtures = [dict(r) for r in db.execute('''SELECT f.id AS database_id,
            f.api_football_id AS fixture_id,c.code AS competition,s.year AS season,
            f.kickoff_utc AS kickoff,f.status,h.api_football_id AS home_team_id,
            a.api_football_id AS away_team_id,f.updated_at
            FROM fixtures f JOIN competitions c ON c.id=f.competition_id
            JOIN seasons s ON s.id=f.season_id JOIN teams h ON h.id=f.home_team_id
            JOIN teams a ON a.id=f.away_team_id
            WHERE julianday(f.kickoff_utc)<julianday(?) AND f.status IN ('FT','AET','PEN')
            ORDER BY f.kickoff_utc,f.api_football_id''', (CUTOFF,)) if r['competition'] in LEAGUES]
        def stats(table, columns, extra=''):
            return [dict(r) for r in db.execute(f'''SELECT f.api_football_id AS fixture_id,
                t.api_football_id AS team_id,{columns},x.raw_payload_digest,x.updated_at
                FROM {table} x JOIN fixtures f ON f.id=x.fixture_id
                JOIN teams t ON t.id=x.team_id {extra}
                WHERE julianday(f.kickoff_utc)<julianday(?) AND f.status IN ('FT','AET','PEN')
                ORDER BY f.api_football_id,t.api_football_id,x.id''', (CUTOFF,))]
        players = stats('fixture_player_stats', 'p.api_football_id AS player_id,x.minutes,x.yellow_cards,x.red_cards',
                        'JOIN players p ON p.id=x.player_id')
        teams = stats('fixture_team_stats', 'x.yellow_cards,x.red_cards')
        # Archive references have no sporting outcomes. Later payloads are never opened.
        archives = [dict(r) for r in db.execute("SELECT * FROM raw_payload_archive WHERE endpoint IN ('/fixtures/players','/fixtures/events') ORDER BY id")]
        inventory = [dict(r) for r in db.execute('''SELECT a.endpoint,
            CASE WHEN f.id IS NULL THEN 'unmatched' WHEN julianday(f.kickoff_utc)<julianday(?)
            THEN 'permitted' ELSE 'reserved' END AS period,COUNT(*) AS archives
            FROM raw_payload_archive a LEFT JOIN fixtures f
            ON f.api_football_id=CAST(json_extract(a.params,'$.fixture') AS INTEGER)
            WHERE a.endpoint IN ('/fixtures/players','/fixtures/events') GROUP BY 1,2''', (CUTOFF,))]
    ids = {f['fixture_id'] for f in fixtures}
    return {'fixtures': fixtures, 'players': [r for r in players if r['fixture_id'] in ids],
            'teams': [r for r in teams if r['fixture_id'] in ids], 'archives': archives,
            'archive_inventory': inventory}


def read_players(root, archive, fixture):
    if not cards.permitted(fixture['kickoff']):
        raise ValueError('reserved_archive')
    params = json.loads(archive['params'])
    if (archive['provider'] != 'api_football' or archive['endpoint'] != '/fixtures/players'
            or archive['storage_backend'] != 'local' or params != {'fixture': fixture['fixture_id']}):
        raise ValueError('archive_scope_mismatch')
    if sha(json.dumps(params, sort_keys=True, separators=(',', ':')).encode()) != archive['params_digest']:
        raise ValueError('archive_params_checksum_mismatch')
    uri = urlparse(archive['storage_uri'])
    path = Path(unquote(uri.path))
    if (uri.scheme != 'file' or uri.netloc not in ('', 'localhost') or path.is_symlink()
            or not path.resolve().is_relative_to((root / 'Index/raw_archive').resolve())):
        raise ValueError('archive_path_invalid')
    body = path.read_bytes()
    payload = gzip.decompress(body) if path.suffix == '.gz' else body
    if sha(payload) != archive['payload_digest']:
        raise ValueError('archive_payload_checksum_mismatch')
    result = json.loads(payload)
    if (result.get('errors') or str(result.get('parameters', {}).get('fixture')) != str(fixture['fixture_id'])
            or result.get('paging', {}).get('total', 1) != 1 or not isinstance(result.get('response'), list)):
        raise ValueError('player_response_envelope_invalid')
    return result['response'], {'archive_id': archive['id'], 'payload_sha256': sha(payload),
                                'file_sha256': sha(body), 'observed_at': archive['fetched_at']}


def audit(root, data, certified):
    players, teams, archives = defaultdict(list), defaultdict(list), defaultdict(list)
    for row in data['players']:
        players[row['fixture_id']].append(row)
    for row in data['teams']:
        teams[row['fixture_id']].append(row)
    for row in data['archives']:
        if row['endpoint'] == '/fixtures/players':
            archives[json.loads(row['params']).get('fixture')].append(row)
    results, requests = [], []
    for fixture in data['fixtures']:
        fid = fixture['fixture_id']
        raw, reference, error = None, None, None
        if archives[fid]:
            archive = max(archives[fid], key=lambda a: (a['fetched_at'], a['id']))
            try:
                raw, reference = read_players(root, archive, fixture)
            except (ValueError, OSError, EOFError) as exc:
                error = 'raw_evidence_error:' + str(exc)
        row = cards.qualify(fixture, players[fid], certified.get(fid), raw_players=raw,
                            raw_reference=reference, raw_error=error)
        pair = teams[fid]
        expected_ids = {fixture['home_team_id'], fixture['away_team_id']}
        valid_pair = len(pair) == 2 and {r['team_id'] for r in pair} == expected_ids
        def valid_count(v):
            return type(v) in (int, float) and v >= 0 and float(v).is_integer()
        complete = valid_pair and all(valid_count(r[k]) for r in pair for k in ('yellow_cards', 'red_cards'))
        red_known = valid_pair and all(valid_count(r['red_cards']) for r in pair)
        row.update(team_legacy_total=sum(r['yellow_cards']+r['red_cards'] for r in pair) if complete else None,
                   team_red_category=('positive' if sum(r['red_cards'] for r in pair) else 'zero') if red_known else 'unknown')
        results.append(row)
        # Bounded proposal: only fixtures where normalized player evidence already
        # exists. Not the many thousands without players, and not an API job.
        if players[fid] and not row['eligible']:
            endpoints = []
            if any(r.startswith('regulation_source_') for r in row['exclusions']):
                endpoints.append({'endpoint': '/fixtures', 'params': {'id': fid}})
            if row['exclusions']:
                endpoints.append({'endpoint': '/fixtures/players', 'params': {'fixture': fid}})
            requests.append({'fixture_id': fid, 'competition': fixture['competition'],
                             'kickoff': fixture['kickoff'], 'exclusions': row['exclusions'], 'requests': endpoints})
    def summarize(rows):
        return {'fixtures': len(rows), 'with_player_rows': sum(r['player_row_count'] > 0 for r in rows),
                'normalized_arithmetic_available': sum(r['normalized_candidate_total'] is not None for r in rows),
                'qualified': sum(r['eligible'] for r in rows),
                'exclusions': dict(sorted(Counter(x for r in rows for x in r['exclusions']).items())),
                'normalized_pending': dict(sorted(Counter(r['normalized_pending_reason'] for r in rows
                                                         if r['normalized_pending_reason']).items())),
                'legacy_label_red_categories': dict(Counter(r['team_red_category'] for r in rows if r['team_legacy_total'] is not None)),
                'all_fixture_red_categories': dict(Counter(r['team_red_category'] for r in rows)),
                'normalized_total_categories': dict(Counter('zero' if r['normalized_candidate_total'] == 0 else 'positive'
                                                           for r in rows if r['normalized_candidate_total'] is not None))}
    groups = defaultdict(list)
    for row in results:
        groups[f"{row['competition']}:{row['season']}"].append(row)
    fields = {}
    for source in ('players', 'teams'):
        fields[source] = {key: dict(Counter('null' if r[key] is None else 'zero' if r[key] == 0 else 'positive' if r[key] > 0 else 'invalid'
                                          for r in data[source])) for key in ('yellow_cards', 'red_cards')}
    return results, {'overall': summarize(results), 'league_season': {k: summarize(v) for k, v in sorted(groups.items())},
                     'player_cohort': summarize([r for r in results if r['player_row_count']]),
                     'normalized_field_counts_not_raw_verified': fields, 'archive_inventory': data['archive_inventory'],
                     'target_contract': cards.CONTRACT, 'qualification_scope': 'frozen regulation evidence and complete original player response',
                     'collection_proposal_fixtures': len(requests),
                     'collection_proposal_requests': sum(len(r['requests']) for r in requests)}, requests


def forecast_rows(body):
    unique = {}
    for line in body.splitlines():
        row = json.loads(line)
        calibration.partition(row)
        key = row['fixture_id'], row['market']
        value = {k: row[k] for k in ('fixture_id', 'market', 'kickoff', 'league', 'season', 'stage',
                                    'missing_fallback', 'week', 'target', 'reference', 'snapshot_id')}
        if key in unique and unique[key] != value:
            raise ValueError('Conflicting reference forecast versions')
        unique[key] = value
    # Consistent fixture metadata across markets; grouping cannot drift.
    fixtures = {}
    for row in unique.values():
        identity = row['kickoff'], row['snapshot_id']
        if row['fixture_id'] in fixtures and fixtures[row['fixture_id']] != identity:
            raise ValueError('Fixture versions disagree')
        fixtures[row['fixture_id']] = identity
    return sorted(unique.values(), key=lambda r: (r['kickoff'], r['fixture_id'], r['market']))


def run(root, output):
    root, output = root.resolve(), output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    write(output / 'RUNNING.json', {'started_at': datetime.now(timezone.utc).isoformat(), 'status': 'incomplete_until_COMPLETE'})
    (output / 'protocol.md').write_bytes((root / PROTOCOL).read_bytes())
    sources = ['Scripts/ops/phase4_development.py', 'Scripts/data_platform/features/phase4_cards.py',
               'Scripts/data_platform/features/count_calibration.py', 'Scripts/data_platform/participation_cards.py',
               'Scripts/data_platform/settlement.py', 'Scripts/data_platform/settlement_policy.py',
               'Scripts/data_platform/features/benchmarks/isolation.py']
    for name in sources:
        dest = output / 'source' / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes((root / name).read_bytes())
    write(output / 'source-hashes.json', {s: sha((root / s).read_bytes()) for s in sources})
    write(output / 'dependencies.json', {name: importlib.metadata.version(name) for name in ('numpy', 'scipy')})
    history_body = verified(root / DATASET, 'audit/inputs.json')
    certified, skipped = cards.period_index(history_body.decode())
    data = snapshot(root / 'Index/platform.db')
    write(output / 'local-development-snapshot.json', data)
    results, report, requests = audit(root, data, certified)
    write_rows(output / 'card-targets.jsonl', results)
    write(output / 'card-qualification.json', report)
    write(output / 'collection-proposal-NOT-APPROVED.json', {'approved': False, 'automatic_execution': False, 'fixtures': requests})
    body = verified(root / FORECASTS, 'predictions.jsonl')
    rows = forecast_rows(body)
    write(output / 'inputs.json', {'history_sha256': sha(history_body), 'reference_forecasts_sha256': sha(body),
                                 'certified_development_periods': len(certified), 'reserved_history_objects_not_decoded': skipped,
                                 'protocol_sha256': sha((root / PROTOCOL).read_bytes()),
                                 'database_mode': 'read_only_consistent_transaction',
                                 'refresh_incomplete_marker_present': Path(str(root / 'Index/platform.db') + '.refresh-incomplete').exists(),
                                 'publication_enabled': False, 'promotion_allowed': False, 'availability': 'assumed_final',
                                 'python': sys.version})
    # Numerical work cannot open databases, network or protected datasets.
    with offline_guard(root=root, output=output):
        fits, metrics, predictions = {}, {}, []
        for market in calibration.LINES:
            fitting = [r for r in rows if r['market'] == market and calibration.partition(r) == 'fit']
            evaluation = [r for r in rows if r['market'] == market and calibration.partition(r) == 'evaluation']
            if not calibration.support(fitting, fitting=True)['sufficient'] or not calibration.support(evaluation)['sufficient']:
                metrics[market] = {'status': 'insufficient_cohort', 'fitting': calibration.support(fitting, fitting=True),
                                   'evaluation': calibration.support(evaluation)}
                continue
            fits[market] = calibration.fit(fitting)
            metrics[market], output_rows = calibration.evaluate(evaluation, fits[market], market)
            predictions.extend(output_rows)
        write(output / 'dispersion-fits.json', fits)
        write(output / 'dispersion-metrics.json', metrics)
        write_rows(output / 'dispersion-predictions.jsonl', predictions)
        write_rows(output / 'reference-development-forecasts.jsonl', rows)
    (output / 'RUNNING.json').unlink()
    complete = {str(p.relative_to(output)): sha(p.read_bytes()) for p in sorted(output.rglob('*')) if p.is_file()}
    write(output / 'COMPLETE.json', complete)
    return report['overall'], {k: {'alpha': fits[k]['alpha'], 'nll_delta_week_bootstrap95': metrics[k]['nll_delta_week_bootstrap95'],
                                  'encouraging_development_result': metrics[k]['encouraging_development_result']}
                               for k in fits}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.root, args.output), indent=2))


if __name__ == '__main__':
    main()
