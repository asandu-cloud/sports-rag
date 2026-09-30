"""Offline card reconstruction; no API acquisition or weight optimisation.

python -B -m Scripts.ops.phase4_card_reconstruction --output NEW_DIRECTORY
python -B -m Scripts.ops.phase4_card_reconstruction --verify EXISTING_DIRECTORY
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
from contextlib import closing
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sqlite3
import sys
from urllib.parse import unquote, urlparse
import yaml

from Scripts.data_platform.features import phase4_card_reconstruction as model
from Scripts.data_platform.features import phase4_cards as cards
from Scripts.data_platform.features.benchmarks.artifacts import verify_complete
from Scripts.data_platform.features.benchmarks.isolation import offline_guard
from Scripts.ops import phase4_development as old

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = Path('docs/phase4-card-reconstruction-protocol-2026-09-30.md')
SOURCES = (
    'Scripts/ops/phase4_card_reconstruction.py',
    'Scripts/data_platform/features/phase4_card_reconstruction.py',
    'Scripts/data_platform/features/phase4_cards.py', 'Scripts/ops/phase4_development.py',
    'Scripts/data_platform/participation_cards.py', 'Scripts/data_platform/settlement.py',
    'Scripts/data_platform/settlement_policy.py',
    'Scripts/data_platform/features/benchmarks/confirmation_data.py',
    'Scripts/data_platform/features/benchmarks/isolation.py',
    'Scripts/rag_ingest/core/weights.py', 'Scripts/rag_ingest/core/projections.py',
    'Scripts/rag_ingest/referee_data.py',
    'Scripts/data_platform/registry/competitions.yaml',
)
LEGACY = {
    'EPL': ('Prem_output/player_fixture_stats_2023.json', 'Premier_League/prem_player_stats_per_gw.py'),
    'LaLiga': ('LaLiga_Output/LaLiga_player_fixture_stats_2023.json', 'La_Liga/laliga_player_stats_per_gw.py'),
    'SerieA': ('SeriaA_Output/SeriaA_player_fixture_stats_2023.json', 'Seria_A/SeriaA_player_stats.py'),
    'Bundesliga': ('Bundesliga_Output/Bundesliga_player_fixture_stats_2023.json', 'Bundesliga/bundesliga_stats_for_players.py'),
    'Ligue1': ('Ligue1_Output/Ligue1_player_fixture_stats_2023.json', 'Ligue_1/Ligue_1_Stats_for_players.py'),
    'UCL': ('UCL_output/player_fixture_stats_2023.json', 'Champions_League/UCL_player_stats.py'),
    'UEL': ('UEL_output/player_fixture_stats_2023.json', 'Europa_League/UEL_player_stats.py'),
    'UECL': ('UECL_output/player_fixture_stats_2023.json', 'Conference_League/UECL_player_stats.py'),
}


def local_guard(output):
    """Allow bounded read-only SQLite, deny network, secrets and external writes."""
    def audit(event, args):
        if event.startswith('socket.') or event in ('subprocess.Popen', 'os.system'):
            raise RuntimeError('Local card reconstruction forbids ' + event)
        if event == 'sqlite3.connect' and (not str(args[0]).startswith('file:') or '?mode=ro' not in str(args[0])):
            raise RuntimeError('Only read-only database URIs are permitted')
        if event == 'open' and not isinstance(args[0], int):
            path = Path(args[0]).resolve()
            if path.name == '.env':
                raise RuntimeError('Secrets are not inputs')
            mode, flags = args[1:3]
            writing = (isinstance(mode, str) and any(c in mode for c in 'wax+')) or (
                isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC))
            if writing and not path.is_relative_to(output):
                raise RuntimeError('Write outside card experiment: ' + str(path))
    sys.addaudithook(audit)


def snapshot(database):
    with closing(sqlite3.connect(database.resolve().as_uri() + '?mode=ro', uri=True, timeout=3)) as db:
        db.row_factory = sqlite3.Row
        db.execute('PRAGMA query_only=ON')
        db.execute('BEGIN')
        # All-period identity metadata only. Never SELECT scores or statistics here.
        metadata = [dict(r) for r in db.execute('''SELECT f.id AS database_id,
            f.api_football_id AS fixture_id,c.code AS competition,s.year AS season,
            f.kickoff_utc AS kickoff,f.status,h.api_football_id AS home_team_id,
            a.api_football_id AS away_team_id,h.name AS home_name,a.name AS away_name,
            f.referee,f.updated_at FROM fixtures f JOIN competitions c ON c.id=f.competition_id
            JOIN seasons s ON s.id=f.season_id JOIN teams h ON h.id=f.home_team_id
            JOIN teams a ON a.id=f.away_team_id ORDER BY f.kickoff_utc,f.api_football_id''')
                    if r['competition'] in old.LEAGUES and r['kickoff']]
        fixtures = [f for f in metadata if cards.permitted(f['kickoff']) and f['status'] in ('FT', 'AET', 'PEN')]
        def stats(table, columns, extra=''):
            return [dict(r) for r in db.execute(f'''SELECT f.api_football_id AS fixture_id,
                t.api_football_id AS team_id,{columns},x.raw_payload_digest,x.updated_at
                FROM {table} x JOIN fixtures f ON f.id=x.fixture_id
                JOIN teams t ON t.id=x.team_id {extra}
                WHERE julianday(f.kickoff_utc)<julianday(?) AND f.status IN ('FT','AET','PEN')
                ORDER BY f.api_football_id,t.api_football_id,x.id''', (old.CUTOFF,))]
        players = stats('fixture_player_stats', 'p.api_football_id AS player_id,x.minutes,x.yellow_cards,x.red_cards',
                        'JOIN players p ON p.id=x.player_id')
        teams = stats('fixture_team_stats', 'x.yellow_cards,x.red_cards')
        archives = [dict(r) for r in db.execute('SELECT * FROM raw_payload_archive ORDER BY id')]
        db.rollback()
    ids = {f['fixture_id'] for f in fixtures}
    by_id = {f['fixture_id']: f for f in metadata}
    inventory = Counter()
    for a in archives:
        if a['endpoint'] in ('/fixtures/players', '/fixtures/events'):
            fid = json.loads(a['params']).get('fixture')
            fixture = by_id.get(int(fid)) if str(fid).isdigit() else None
            period = 'unmatched' if fixture is None else 'permitted' if cards.permitted(fixture['kickoff']) else 'reserved'
            inventory[(a['endpoint'], period)] += 1
    return {'fixtures': fixtures, 'players': [r for r in players if r['fixture_id'] in ids],
            'teams': [r for r in teams if r['fixture_id'] in ids], 'archives': archives,
            'archive_inventory': [{'endpoint': e, 'period': p, 'archives': n} for (e, p), n in sorted(inventory.items())],
            'identity_metadata': metadata}


def literal_weights(root):
    def assignments(path):
        tree = ast.parse(path.read_text())
        out = {}
        for node in tree.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                try:
                    out[node.targets[0].id] = ast.literal_eval(node.value)
                except (ValueError, TypeError):
                    pass
        return out
    w = assignments(root / 'Scripts/rag_ingest/core/weights.py')['SCORING_WEIGHTS']
    r = assignments(root / 'Scripts/rag_ingest/referee_data.py')
    return {'own': w['projection_cards']['own'], 'opponent': w['projection_cards']['opp'],
            'venue': w['projection_cards']['venue_blend'], 'season': w['projection']['blend_cards_season'],
            'recent': w['projection']['blend_cards_recent'], 'recency_alpha': w['recency']['alpha'],
            'referee_modifier': w['referee']['modifier_weight'], 'referee_anchor': w['referee']['anchor_weight'],
            'referee_minimum': r['MIN_SAMPLE'], 'referee_anchor_minimum': w['referee']['min_sample_size'],
            'referee_full_confidence': r['FULL_CONFIDENCE']}


def fixture_aliases(text, competition):
    """Original provider names/IDs/dates only; never decode score/goals fields."""
    if text[model._white(text, 0)] == '{':
        envelope = {k: (a, b) for k, a, b in model._members(text)}
        a, b = envelope['response']
        text = text[a:b]
    rows = []
    for start, _ in model.array_spans(text):
        spans = {k: (a, b) for k, a, b in model._members(text, start)}
        fixture = model.fields(text, spans['fixture'][0], {'id', 'date', 'status', 'referee'})
        league = model.fields(text, spans['league'][0], {'season'})
        team_spans = {k: (a, b) for k, a, b in model._members(text, spans['teams'][0])}
        teams = {side: model.fields(text, team_spans[side][0], {'id', 'name'}) for side in ('home', 'away')}
        rows.append({'fixture_id': fixture['id'], 'competition': competition, 'season': league['season'],
                     'kickoff': fixture['date'], 'status': fixture['status']['short'],
                     'home_team_id': teams['home']['id'], 'away_team_id': teams['away']['id'],
                     'home_name': teams['home']['name'], 'away_name': teams['away']['name'],
                     'referee': fixture.get('referee')})
    return rows


def local_inventory(root, data):
    registered = set()
    for a in data['archives']:
        if a['storage_backend'] == 'local':
            uri = urlparse(a['storage_uri'])
            registered.add(str(Path(unquote(uri.path)).resolve()))
    raw = sorted(p for p in (root / 'Index/raw_archive').rglob('*') if p.is_file())
    unregistered = [str(p.relative_to(root)) for p in raw if str(p.resolve()) not in registered]
    staged = sorted(p for p in (root / 'Index/history_staging').rglob('*') if p.is_file())
    source_candidates = [str(p.relative_to(root)) for p in staged
                         if 'players' in p.parts or 'events' in p.parts]
    all_legacy = sorted(str(p.relative_to(root)) for p in (root / 'Output').glob('*/*player_fixture_stats_*.json'))
    backups = []
    for path in sorted((root / 'Index').glob('platform*.db')):
        if path.name == 'platform.db':
            continue
        try:
            with closing(sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True, timeout=3)) as db:
                counts = [list(r) for r in db.execute('''SELECT s.year,c.code,COUNT(DISTINCT x.fixture_id),COUNT(*)
                    FROM fixture_player_stats x JOIN fixtures f ON f.id=x.fixture_id
                    JOIN seasons s ON s.id=f.season_id JOIN competitions c ON c.id=f.competition_id
                    WHERE julianday(f.kickoff_utc)<julianday(?) GROUP BY s.year,c.code''', (old.CUTOFF,))]
            backups.append({'path': str(path.relative_to(root)), 'permitted_player_metadata': counts,
                            'outcomes_decoded': False})
        except sqlite3.DatabaseError as exc:
            backups.append({'path': str(path.relative_to(root)), 'inventory_error': str(exc)})
    return {'registered_archive_rows': len(data['archives']), 'raw_archive_files': len(raw),
            'unregistered_raw_paths': unregistered, 'staging_player_event_paths': source_candidates,
            'legacy_player_exports': all_legacy, 'backup_database_metadata': backups,
            'scope': 'Index/raw_archive, Index/history_staging, Output fixture-player exports, top-level Index platform backups',
            'limitation': 'Not an exhaustive search of every duplicate baseline/research backup or external disk; no opaque unassigned payloads decoded.'}


def run(root, output):
    root, output = root.resolve(), output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    local_guard(output)
    old.write(output / 'RUNNING.json', {'started_at': datetime.now(timezone.utc).isoformat()})
    (output / 'protocol.md').write_bytes((root / PROTOCOL).read_bytes())
    weights = literal_weights(root)
    old.write(output / 'fixed-weights.json', weights)
    sources = [*SOURCES, *('Scripts/' + item[1] for item in LEGACY.values())]
    for source in sources:
        dest = output / 'source' / source
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes((root / source).read_bytes())
    old.write(output / 'source-hashes.json', {p: old.sha((root / p).read_bytes()) for p in sources})
    old.write(output / 'dependencies.json', {'python': sys.version, **{
        p: importlib.metadata.version(p) for p in ('numpy', 'scipy', 'scikit-learn', 'sqlalchemy', 'pyyaml')}})
    print('Frozen protocol and fixed coefficients; reading local evidence.', flush=True)
    body = old.verified(root / old.DATASET, 'audit/inputs.json')
    certified, skipped = cards.period_index(body.decode())
    data = snapshot(root / 'Index/platform.db')
    old.write(output / 'canonical-development-inputs.json', data)
    inventory = local_inventory(root, data)
    old.write(output / 'local-source-inventory.json', inventory)
    aliases, alias_hashes, alias_conflicts = [], {}, []
    by_id = {f['fixture_id']: f for f in data['identity_metadata']}
    directory = root / 'Index/prediction_experiments/local-history-identities-2023-2026-09-25'
    for competition in LEGACY:
        filename = f'fixtures-{competition}.json'
        raw = old.verified(directory, filename)
        alias_hashes[str((directory / filename).relative_to(root))] = old.sha(raw)
        for f in fixture_aliases(raw.decode(), competition):
            existing = by_id.get(f['fixture_id'])
            if existing and (any(f[k] != existing[k] for k in ('competition', 'season', 'home_team_id', 'away_team_id', 'status'))
                             or cards.utc(f['kickoff']) != cards.utc(existing['kickoff'])):
                alias_conflicts.append({'fixture_id': f['fixture_id'], 'reason': 'provider_identity_conflicts_with_platform'})
            elif existing:
                aliases.append(f)
    old.write(output / 'provider-identity-aliases.json', aliases)
    old.write(output / 'provider-identity-conflicts.json', alias_conflicts)
    registry = yaml.safe_load((root / 'Scripts/data_platform/registry/competitions.yaml').read_text())
    team_aliases = {c: v.get('providers', {}).get('api_football', {}).get('team_aliases', {})
                    for c, v in registry['competitions'].items()}
    old.write(output / 'registry-team-aliases.json', team_aliases)
    identities = model.identity_map(data['identity_metadata'], aliases, team_aliases)
    legacy, excluded, legacy_reports = [], [], {}
    for competition, (relative, collector) in LEGACY.items():
        path = root / 'Output' / relative
        text = path.read_text()
        admitted, rejected, counts = model.legacy_rows(text, competition=competition, season=2023,
                                                      identities=identities, source=str(path.relative_to(root)))
        legacy.extend(admitted)
        excluded.extend(rejected)
        legacy_reports[competition] = {'source': str(path.relative_to(root)), 'sha256': old.sha(text.encode()), **counts,
                                       'collector': 'Scripts/' + collector,
                                       'current_collector_discards_minutes_le_one': True,
                                       'original_collector_version_not_certified': True}
    old.write(output / 'legacy-coverage.json', legacy_reports)
    old.write_rows(output / 'legacy-admitted-player-rows.jsonl', legacy)
    old.write_rows(output / 'legacy-excluded-row-identities.jsonl', excluded)
    print(f'Resolved {len(legacy)} permitted legacy player rows; later rows stayed opaque.', flush=True)
    canonical, previous_report, _ = old.audit(root, data, certified)
    old.write_rows(output / 'canonical-targets.jsonl', canonical)
    reconstructed = model.reconcile(canonical, data['players'], legacy)
    coverage = model.coverage(reconstructed)
    old.write_rows(output / 'card-targets.jsonl', reconstructed)
    old.write(output / 'coverage.json', coverage)
    old.write(output / 'canonical-evidence-report.json', previous_report)
    missing = []
    for f in reconstructed:
        if f['eligible']:
            continue
        needed = [{'evidence': 'complete_original_player_response', 'endpoint': '/fixtures/players',
                   'params': {'fixture': f['fixture_id']}}]
        if any(x.startswith('regulation_source_') for x in f['exclusions']):
            needed.append({'evidence': 'regulation_result_and_exact_identity', 'endpoint': '/fixtures',
                           'params': {'id': f['fixture_id']}})
        missing.append({k: f[k] for k in ('fixture_id', 'competition', 'season', 'kickoff', 'exclusions')} |
                       {'missing_evidence': needed, 'collection_authorized': False})
    old.write_rows(output / 'missing-evidence-NOT-A-COLLECTION-PLAN.jsonl', missing)
    old.write(output / 'input-manifest.json', {'frozen_history_sha256': old.sha(body),
        'reserved_history_objects_not_decoded': skipped, 'provider_identity_hashes': alias_hashes,
        'legacy_source_hashes': {c: r['sha256'] for c, r in legacy_reports.items()},
        'database': 'read_only_consistent_transaction', 'api_calls': 0, 'production_changes': False,
        'availability': 'assumed_final', 'cutoff': old.CUTOFF,
        'refresh_incomplete_marker_present': (root / 'Index/platform.db.refresh-incomplete').exists()})
    with offline_guard(root=root, output=output):
        result, predictions, profiles = model.backtest(reconstructed, weights)
        old.write(output / 'backtest.json', result)
        old.write_rows(output / 'predictions.jsonl', predictions)
        old.write_rows(output / 'dated-profiles.jsonl', profiles)
        # Replay is performed against the saved admitted evidence, not live state.
        replay = model.reconcile(canonical, data['players'], legacy)
        if replay != reconstructed or model.backtest(replay, weights) != (result, predictions, profiles):
            raise ValueError('Non-deterministic card reconstruction/backtest')
        old.write(output / 'replay.json', {'identical': True, 'targets_sha256': model.digest(reconstructed),
                                         'backtest_sha256': model.digest(result), 'predictions_sha256': model.digest(predictions)})
    (output / 'RUNNING.json').unlink()
    old.write(output / 'COMPLETE.json', {str(p.relative_to(output)): old.sha(p.read_bytes())
                                        for p in sorted(output.rglob('*')) if p.is_file()})
    verify_complete(output)
    print(json.dumps({'coverage': coverage['overall'], 'backtest': result, 'output': str(output)}, indent=2))


def replay(directory, output):
    directory, output = directory.resolve(), output.resolve()
    verify_complete(directory)
    output.mkdir(parents=True, exist_ok=False)
    local_guard(output)
    with offline_guard(root=ROOT, output=output):
        data = json.loads((directory / 'canonical-development-inputs.json').read_text())
        read_rows = lambda n: [json.loads(x) for x in (directory / n).read_text().splitlines()]
        rows = model.reconcile(read_rows('canonical-targets.jsonl'), data['players'],
                               read_rows('legacy-admitted-player-rows.jsonl'))
        if rows != read_rows('card-targets.jsonl'):
            raise ValueError('Target replay mismatch')
        report, predictions, profiles = model.backtest(rows, json.loads((directory / 'fixed-weights.json').read_text()))
        if (report != json.loads((directory / 'backtest.json').read_text())
                or predictions != read_rows('predictions.jsonl') or profiles != read_rows('dated-profiles.jsonl')):
            raise ValueError('Backtest replay mismatch')
        old.write(output / 'replay.json', {'matches_saved_artifact': True, 'fixture_targets': len(rows),
            'backtest_status': report['status'], 'scored_fixtures': report['scored_fixtures'],
            'source_complete_sha256': old.sha((directory / 'COMPLETE.json').read_bytes()),
            'targets_sha256': model.digest(rows), 'backtest_sha256': model.digest(report)})
        old.write(output / 'COMPLETE.json', {'replay.json': old.sha((output / 'replay.json').read_bytes())})
    print((output / 'replay.json').read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path)
    action = parser.add_mutually_exclusive_group()
    action.add_argument('--verify', type=Path)
    action.add_argument('--replay', type=Path)
    args = parser.parse_args()
    if args.verify:
        if args.output:
            parser.error('--verify does not write an output')
        print(json.dumps({'verified_files': len(verify_complete(args.verify))}))
    elif args.replay and args.output:
        replay(args.replay, args.output)
    elif args.output and not args.replay:
        run(args.root, args.output)
    else:
        parser.error('Use --output NEW_DIRECTORY, --replay EXISTING --output NEW_DIRECTORY, or --verify EXISTING')


if __name__ == '__main__':
    main()
