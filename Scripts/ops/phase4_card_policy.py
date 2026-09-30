"""Apply the two-minute, league-phase amendment to sealed local card evidence.

Reads fresh identity/round/archive-presence metadata only. Outcomes come solely
from the existing permitted development artifact. No API or publication writes.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import closing
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import sqlite3

from Scripts.data_platform.features import phase4_cards as cards
from Scripts.data_platform.features import phase4_card_policy as policy
from Scripts.data_platform.features import phase4_card_reconstruction as model
from Scripts.data_platform.features.benchmarks.artifacts import verify_complete
from Scripts.data_platform.features.benchmarks.isolation import offline_guard
from Scripts.ops import phase4_development as old
from Scripts.ops.phase4_card_reconstruction import local_guard

ROOT = Path(__file__).resolve().parents[2]
INPUT = Path('Research/phase4-card-reconstruction-2026-09-30/reconstruction')
PROTOCOL = Path('docs/phase4-card-policy-v2-protocol-2026-09-30.md')
SOURCES = (
    'Scripts/ops/phase4_card_policy.py', 'Scripts/ops/phase4_card_reconstruction.py',
    'Scripts/ops/phase4_development.py',
    'Scripts/data_platform/features/phase4_card_policy.py',
    'Scripts/data_platform/features/phase4_card_reconstruction.py',
    'Scripts/data_platform/features/phase4_cards.py',
    'Scripts/data_platform/features/market_eligibility.py',
    'Scripts/data_platform/participation_cards.py', 'Scripts/data_platform/settlement.py',
    'Scripts/data_platform/settlement_policy.py',
    'Scripts/data_platform/features/benchmarks/isolation.py',
)


def metadata_snapshot(path):
    """No minutes, card fields, scores, goals or player response bodies are read."""
    with closing(sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True, timeout=3)) as db:
        db.row_factory = sqlite3.Row
        db.execute('PRAGMA query_only=ON')
        db.execute('BEGIN')
        fixtures = [dict(r) for r in db.execute('''SELECT f.api_football_id AS fixture_id,
            c.code AS competition,s.year AS season,f.kickoff_utc AS kickoff,f.status,f.round,
            h.api_football_id AS home_team_id,a.api_football_id AS away_team_id,
            h.name AS home_name,a.name AS away_name,f.referee
            FROM fixtures f JOIN competitions c ON c.id=f.competition_id
            JOIN seasons s ON s.id=f.season_id JOIN teams h ON h.id=f.home_team_id
            JOIN teams a ON a.id=f.away_team_id ORDER BY f.kickoff_utc,f.api_football_id''')
                    if r['competition'] in old.LEAGUES]
        counts = {r['fixture_id']: r['rows'] for r in db.execute('''SELECT f.api_football_id AS fixture_id,COUNT(*) AS rows
            FROM fixture_player_stats x JOIN fixtures f ON f.id=x.fixture_id GROUP BY f.api_football_id''')}
        archives = [dict(r) for r in db.execute('''SELECT id,provider,endpoint,params,storage_uri,payload_digest,fetched_at
            FROM raw_payload_archive WHERE endpoint='/fixtures/players' ORDER BY id''')]
        db.rollback()
    for f in fixtures:
        f['normalized_player_rows'] = counts.get(f['fixture_id'], 0)
    return {'fixtures': fixtures, 'player_archive_metadata': archives, 'outcomes_decoded': False}


def inventory(metadata, legacy):
    archives, legacy_counts = defaultdict(list), Counter(r['fixture_id'] for r in legacy)
    for a in metadata['player_archive_metadata']:
        params = json.loads(a['params'])
        if a['provider'] == 'api_football' and str(params.get('fixture', '')).isdigit():
            archives[int(params['fixture'])].append(a['id'])
    rows, by_season, by_league_season, exclusions = [], defaultdict(Counter), defaultdict(Counter), Counter()
    for f in metadata['fixtures']:
        scoped = policy.scope(f)
        if not scoped['eligible']:
            exclusions[scoped['group'] + ':' + str(f['status'])] += 1
            continue
        archive_ids = archives[f['fixture_id']]
        row = {**f, 'round_group': scoped['group'], 'original_response_archive_ids': archive_ids,
               'archive_contents_verified': False, 'missing_archived_player_response': not archive_ids,
               'legacy_rows_in_permitted_artifact': legacy_counts[f['fixture_id']],
               'required_endpoint': '/fixtures/players', 'required_parameters': {'fixture': f['fixture_id']},
               'collection_authorized': False,
               'outcome_inspection': 'permitted_pre2024' if f['kickoff'] and cards.permitted(f['kickoff']) else 'metadata_only'}
        rows.append(row)
        for group in (by_season[str(f['season'])], by_league_season[f"{f['competition']}:{f['season']}"]):
            group['completed_in_scope_fixtures'] += 1
            group['with_normalized_players'] += bool(f['normalized_player_rows'])
            group['with_archived_player_response'] += bool(archive_ids)
            group['missing_archived_player_response'] += not archive_ids
            group['with_permitted_legacy_rows'] += bool(legacy_counts[f['fixture_id']])
    return rows, {'season': {k: dict(v) for k, v in sorted(by_season.items())},
                  'league_season': {k: dict(v) for k, v in sorted(by_league_season.items())},
                  'excluded_scope_or_status': dict(sorted(exclusions.items())),
                  'qualification_of_present_responses': 'not_assessed_by_metadata_inventory',
                  'collection_authorized': False}


def reconstruct(data, legacy, original_targets, metadata, reference):
    by_id = {f['fixture_id']: f for f in metadata['fixtures']}
    players = defaultdict(list)
    for r in data['players']:
        players[r['fixture_id']].append(r)
    old_targets = {f['fixture_id']: f for f in original_targets}
    fixtures, scope_counts = [], Counter()
    for previous in data['fixtures']:
        current = by_id.get(previous['fixture_id'])
        if current is None:
            raise ValueError('Saved development fixture no longer has metadata')
        fields = ('fixture_id', 'competition', 'season', 'status', 'home_team_id', 'away_team_id')
        if any(current[k] != previous[k] for k in fields) or cards.utc(current['kickoff']) != cards.utc(previous['kickoff']):
            raise ValueError('Fixture identity changed; explicit source reconciliation required')
        f = {**previous, 'round': current['round']}
        identity = {**current, 'metadata_reference': reference}
        # The approved input artifact has no qualified archived player responses;
        # do not silently ignore one if the input changes in a later acquisition.
        if old_targets[f['fixture_id']]['raw_reference'] is not None:
            raise ValueError('New raw player evidence needs a dedicated verified-response adapter')
        qualified = policy.qualify(f, players[f['fixture_id']], identity)
        qualified.update(team_red_category=old_targets[f['fixture_id']]['team_red_category'],
                         team_legacy_total=old_targets[f['fixture_id']]['team_legacy_total'])
        scope_counts[qualified['round_scope']['group']] += 1
        fixtures.append(qualified)
    combined = model.reconcile(fixtures, data['players'], legacy, minimum_recorded_minutes=2)
    in_scope = [r for r in combined if r['round_scope']['eligible']]
    return combined, in_scope, dict(sorted(scope_counts.items()))


def run(root, source, output):
    root, source, output = root.resolve(), source.resolve(), output.resolve()
    verify_complete(source)
    output.mkdir(parents=True, exist_ok=False)
    local_guard(output)
    old.write(output / 'RUNNING.json', {'started_at': datetime.now(timezone.utc).isoformat()})
    (output / 'protocol.md').write_bytes((root / PROTOCOL).read_bytes())
    old.write(output / 'policy.json', policy.POLICY)
    for name in SOURCES:
        dest = output / 'source' / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes((root / name).read_bytes())
    old.write(output / 'source-hashes.json', {n: old.sha((root / n).read_bytes()) for n in SOURCES})
    old.write(output / 'dependencies.json', {p: importlib.metadata.version(p) for p in ('numpy', 'scipy')})
    metadata = metadata_snapshot(root / 'Index/platform.db')
    old.write(output / 'identity-and-presence-metadata.json', metadata)
    reference = {'file': 'identity-and-presence-metadata.json', 'sha256': model.digest(metadata),
                 'origin': 'read_only_canonical_fixture_metadata', 'outcomes_decoded': False}
    # Retain exactly the admitted pre-2024 source rows for standalone replay.
    names = ('canonical-development-inputs.json', 'legacy-admitted-player-rows.jsonl', 'canonical-targets.jsonl',
             'card-targets.jsonl', 'fixed-weights.json')
    (output / 'inputs').mkdir()
    for name in names:
        (output / 'inputs' / name).write_bytes(old.verified(source, name))
    with offline_guard(root=root, output=output):
        data = json.loads((output / 'inputs/canonical-development-inputs.json').read_text())
        read_rows = lambda n: [json.loads(x) for x in (output / 'inputs' / n).read_text().splitlines()]
        legacy = read_rows('legacy-admitted-player-rows.jsonl')
        original = read_rows('canonical-targets.jsonl')
        all_rows, scoped, scope_counts = reconstruct(data, legacy, original, metadata, reference)
        inventory_rows, gaps = inventory(metadata, legacy)
        old.write_rows(output / 'card-targets-all-development.jsonl', all_rows)
        old.write_rows(output / 'card-targets-in-scope.jsonl', scoped)
        old.write(output / 'coverage.json', model.coverage(scoped))
        old.write(output / 'scope-counts.json', scope_counts)
        old.write_rows(output / 'player-response-inventory-NOT-A-DOWNLOAD-PLAN.jsonl', inventory_rows)
        old.write(output / 'player-response-gaps.json', gaps)
        before = {r['fixture_id']: r for r in read_rows('card-targets.jsonl')}
        comparable = [(r, before[r['fixture_id']]) for r in scoped
                      if r['local_provisional_total'] is not None and before[r['fixture_id']]['local_provisional_total'] is not None]
        effects = {'comparable_provisional_fixtures': len(comparable),
                   'changed_provisional_totals': sum(a['local_provisional_total'] != b['local_provisional_total'] for a, b in comparable),
                   'v1_only_arithmetic': sum(r['local_provisional_total'] is None and before[r['fixture_id']]['local_provisional_total'] is not None for r in scoped),
                   'v2_only_arithmetic': sum(r['local_provisional_total'] is not None and before[r['fixture_id']]['local_provisional_total'] is None for r in scoped),
                   'unresolved_periods_in_scope': sum(not r['period_evidence']['established'] for r in scoped),
                   'production_policy_activated': False}
        old.write(output / 'policy-effects.json', effects)
        weights = json.loads((output / 'inputs/fixed-weights.json').read_text())
        report, forecasts, profiles = model.backtest(scoped, weights, minimum_recorded_minutes=2)
        old.write(output / 'backtest.json', report)
        old.write_rows(output / 'predictions.jsonl', forecasts)
        old.write_rows(output / 'dated-profiles.jsonl', profiles)
        again = reconstruct(data, legacy, original, metadata, reference)
        if again != (all_rows, scoped, scope_counts):
            raise ValueError('Non-deterministic policy reconstruction')
        old.write(output / 'replay.json', {'identical': True, 'scoped_targets_sha256': model.digest(scoped),
                                          'metadata_sha256': model.digest(metadata), 'backtest_sha256': model.digest(report)})
        old.write(output / 'input-manifest.json', {'parent_artifact': str(source),
            'parent_complete_sha256': old.sha((source / 'COMPLETE.json').read_bytes()),
            'api_calls': 0, 'production_activation': False, 'weight_optimisation': False,
            'outcomes_source': 'sealed_pre2024_development_rows_only',
            'new_database_read': 'identity_round_and_presence_metadata_only'})
    (output / 'RUNNING.json').unlink()
    old.write(output / 'COMPLETE.json', {str(p.relative_to(output)): old.sha(p.read_bytes())
                                        for p in sorted(output.rglob('*')) if p.is_file()})
    verify_complete(output)
    print(json.dumps({'coverage': model.coverage(scoped)['overall'], 'policy_effects': effects,
                      'response_gaps_by_season': gaps['season'], 'backtest_status': report['status']}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--source', type=Path, default=ROOT / INPUT)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.root, args.source, args.output)


if __name__ == '__main__':
    main()
