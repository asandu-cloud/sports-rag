"""Offline fixed card comparison from sealed, source-qualified v2 targets.

Reuse the existing reference without fitting weights or changing production.
Archive references stay in the preparation batch's namespace, not platform.db's.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import re
import shutil
import sys

from Scripts.data_platform.features import phase4_card_reconstruction as model
from Scripts.data_platform.features import phase4_card_policy as policy
from Scripts.data_platform.features import phase4_cards as cards
from Scripts.data_platform.features.benchmarks.artifacts import (
    complete, read_json, sha, verify_complete, write_json,
)
from Scripts.data_platform.features.benchmarks.isolation import offline_guard

ROOT = Path(__file__).resolve().parents[2]
VERSION = 'phase4-fixed-card-backtest.v1'
TARGET_FILES = ('card-targets.jsonl', 'rules.json', 'report.json')
PROTOCOL_FILES = (
    'docs/phase4-card-reconstruction-protocol-2026-09-30.md',
    'docs/phase4-card-policy-v2-protocol-2026-09-30.md',
    'docs/phase4-card-fixed-backtest-2026-10-02.md',
)
FROZEN_FUNCTIONS = ('fixed_prediction', 'metrics', 'comparison', 'support', 'name', 'referee_key')


def function_hashes(source):
    functions = {node.name: node for node in ast.parse(source).body
                 if isinstance(node, ast.FunctionDef)}
    return {name: hashlib.sha256(ast.dump(functions[name], include_attributes=False).encode()).hexdigest()
            for name in FROZEN_FUNCTIONS}


def read_targets(path):
    """Inspect the cutoff before any decoder receives a row's outcome values."""
    rows = []
    with Path(path).open() as handle:
        for line in handle:
            meta = model.fields(line, 0, {'kickoff'})
            if not cards.permitted(meta['kickoff']):
                raise ValueError('Reserved card outcome access refused before decoding')
            rows.append(json.loads(line))
    return rows


def adapt_targets(rows, batch):
    """Validate saved qualification; never upgrade provisional or excluded rows."""
    if not isinstance(batch, str) or not re.fullmatch('[0-9a-f]{64}', batch):
        raise ValueError('Missing prepared source namespace')
    adapted = []
    for row in rows:
        if (any(type(row.get(k)) is not int or row[k] <= 0
                for k in ('fixture_id', 'home_team_id', 'away_team_id', 'season'))
                or row['home_team_id'] == row['away_team_id']):
            raise ValueError('Invalid provider fixture/team identity')
        if (row.get('policy') != policy.POLICY or row.get('availability') != 'assumed_final'
                or row.get('reconstruction_version') != 'player-history-reconciliation.v1'
                or type(row.get('eligible')) is not bool):
            raise ValueError('Unrecognized reconstructed target contract')
        if not row['eligible']:
            if not row.get('exclusions') or row.get('target') is not None or row.get('team_targets') is not None:
                raise ValueError('Excluded evidence exposes a numerical target')
            adapted.append(dict(row))
            continue
        refs = row.get('source_references', {})
        if set(refs) != {'/fixtures/players', '/fixtures/lineups', '/fixtures/events'}:
            raise ValueError('Incomplete qualified source references')
        for endpoint, ref in refs.items():
            if (ref.get('endpoint') != endpoint or type(ref.get('fixture_id')) is not int
                    or ref['fixture_id'] <= 0 or type(ref.get('archive_id')) is not int
                    or ref['archive_id'] <= 0 or not ref.get('fetched_at')
                    or not re.fullmatch('[0-9a-f]{64}', str(ref.get('payload_digest')))):
                raise ValueError('Invalid qualified source reference')
        if len({ref['fixture_id'] for ref in refs.values()}) != 1:
            raise ValueError('Source reference fixture mismatch')
        evidence = row['player_evidence']
        # Recheck arithmetic from the saved evidence. This normalized conversion
        # is NOT an original provider response or a new completeness certificate.
        players = [{**p, 'yellow_cards': p['yellow'], 'red_cards': p['red']}
                   for p in evidence['players']]
        parsed = model.evidence(row, players, minimum_recorded_minutes=2)
        canonical_evidence = lambda e: {**e, 'players': sorted(e['players'], key=lambda p: (p['team_id'], p['player_id']))}
        if (canonical_evidence(parsed) != canonical_evidence(evidence) or parsed['pending_reason'] is not None
                or parsed['totals'] != row['team_targets']):
            raise ValueError('Saved player evidence disagrees with qualified target')
        adapted.append({**row, 'contract': policy.CONTRACT, 'settlement_policy': policy.POLICY_VERSION,
                        'raw_reference': {'namespace': 'prepared_player_history', 'batch': batch,
                                          **refs['/fixtures/players']}})
    # Includes uniqueness, cutoff, round, total, period and v2 contract gates.
    model.qualified_rows(adapted, minimum_recorded_minutes=2)
    return adapted


def referee_audit(rows):
    names = defaultdict(Counter)
    candidates = defaultdict(set)
    for row in rows:
        key = model.referee_key(row.get('referee'))
        names[(row['competition'], key)][str(row.get('referee') or '')] += 1
        if key:
            candidates[(row['competition'], key.split()[-1])].add(key)
    return {
        'identity': 'exact_normalized_name_within_competition_no_alias_merging',
        'assignment_availability': 'original_pre_match_assignment_timestamp_unknown',
        'names': [{'competition': league, 'key': key, 'raw_spellings': dict(sorted(counts.items()))}
                  for (league, key), counts in sorted(names.items())],
        'possible_fragmentation': [
            {'competition': league, 'last_token': token, 'separate_keys': sorted(keys)}
            for (league, token), keys in sorted(candidates.items()) if len(keys) > 1],
        'fragmentation_caution': 'Shared final name tokens are review hints, not proof of shared identity.',
    }


def diagnostics(rows, predictions, snapshots, weights, report):
    by_id = {r['fixture_id']: r for r in predictions}
    inputs_by_id = {r['fixture_id']: r for r in snapshots}
    histories = defaultdict(list)
    for row in rows:
        if row['eligible']:
            histories[row['competition']].append(row)
    ledger, groups = [], defaultdict(list)
    for row in sorted(rows, key=lambda r: (cards.utc(r['kickoff']), r['fixture_id'])):
        year = cards.utc(row['kickoff']).year
        prediction = by_id.get(row['fixture_id'])
        item = {k: row[k] for k in ('fixture_id', 'competition', 'season', 'kickoff', 'eligible')}
        item.update(role='evaluation' if year == 2023 else 'support' if year == 2022 else 'history',
                    target_exclusions=row['exclusions'], scored=prediction is not None)
        snapshot = inputs_by_id.get(row['fixture_id'])
        if year == 2023 and row['eligible'] and snapshot is None and not report.get('reasons'):
            inputs = model.dated_inputs(row, histories[row['competition']], minimum_recorded_minutes=2)
            possible, reason = model.fixed_prediction(row, inputs, weights)
            if possible is not None or not reason:
                raise ValueError('A supported forecast disappeared from the fixed comparison')
            snapshot = {'fixture_id': row['fixture_id'], 'snapshot_id': model.digest(inputs), **inputs}
            snapshots.append(snapshot)
            item['forecast_exclusion'] = reason
        elif year == 2023 and row['eligible'] and snapshot is None:
            item['forecast_exclusion'] = 'comparison_support_gate'
        if snapshot is not None:
            item['snapshot_id'] = snapshot['snapshot_id']
            item['current_team_counts'] = {
                side: sum(r['season'] == row['season'] for r in snapshot['teams'][side])
                for side in ('home', 'away')}
            item['referee_matches'] = len(snapshot['referee'])
        ledger.append(item)
        groups[(row['competition'], year)].append(item)
    coverage = {}
    for (league, year), items in sorted(groups.items()):
        selected = [by_id[r['fixture_id']] for r in items if r['scored']]
        coverage[f'{league}:{year}'] = {
            'source_fixtures': len(items), 'qualified_targets': sum(r['eligible'] for r in items),
            'scored': model.support(selected),
            'target_exclusions': dict(sorted(Counter(x for r in items for x in r['target_exclusions']).items())),
            'forecast_exclusions': dict(sorted(Counter(r['forecast_exclusion'] for r in items
                                                       if r.get('forecast_exclusion')).items())),
        }
    known = [r for r in rows if r['eligible']]
    red = lambda r: any(p['minutes'] is not None and p['minutes'] >= 2 and p['red'] > 0
                        for p in r['player_evidence']['players'])
    target_counts = {
        'qualified': len(known), 'excluded': len(rows) - len(known),
        'zero_weighted_total': sum(r['target'] == 0 for r in known),
        'with_recorded_eligible_red': sum(red(r) for r in known),
        'without_recorded_eligible_red': sum(not red(r) for r in known),
    }
    return ledger, {'target_counts': target_counts, 'league_calendar_year': coverage,
                    'dated_input_snapshots': len(snapshots),
                    'scored_referee_changes': sum(r['team_only_mean'] != r['team_referee_mean'] for r in predictions)}, referee_audit(rows)


def compute(directory, progress=lambda _: None):
    rules = read_json(directory / 'inputs/rules.json')
    source_report = read_json(directory / 'inputs/report.json')
    if rules['card_policy'] != policy.POLICY or rules['development_end_exclusive'] != cards.END.isoformat():
        raise ValueError('Reconstructed source rules differ from the agreed contract')
    rows = adapt_targets(read_targets(directory / 'inputs/card-targets.jsonl'), source_report['prepared_batch'])
    actual = Counter('qualified' if r['eligible'] else 'pending_or_excluded' for r in rows)
    if dict(actual) != source_report['counts']:
        raise ValueError('Source coverage does not match the sealed targets')
    weights = read_json(directory / 'fixed-weights.json')
    reference = read_json(directory / 'reference.json')
    if (function_hashes(inspect.getsource(model)) != reference['function_hashes']
            or sha(directory / 'fixed-weights.json') != reference['weights_sha256']):
        raise ValueError('Frozen calculation or weights changed')
    progress(f"Validated {len(rows)} targets; {actual['qualified']} qualified. Building dated inputs and scoring the fixed reference.")
    report, predictions, snapshots = model.backtest(rows, weights, minimum_recorded_minutes=2)
    progress('Fixed comparison finished. Building excluded-forecast and referee coverage ledgers.')
    ledger, coverage, referees = diagnostics(rows, predictions, snapshots, weights, report)
    snapshots.sort(key=lambda r: (r['cutoff'], r['fixture_id']))
    return {'backtest.json': report, 'coverage.json': coverage, 'referee-identities.json': referees,
            'predictions.jsonl': predictions, 'dated-inputs.jsonl': snapshots, 'cohort-membership.jsonl': ledger}


def lines_bytes(rows):
    for row in rows:
        yield (json.dumps(row, sort_keys=True, allow_nan=False, separators=(',', ':')) + '\n').encode()


def source_files():
    files = {Path(__file__).resolve()}
    for module in list(sys.modules.values()):
        location = getattr(module, '__file__', None)
        if location:
            path = Path(location).resolve()
            if path.is_relative_to(ROOT / 'Scripts') and path.suffix == '.py':
                files.add(path)
    return sorted(files)


def replay(directory):
    directory = Path(directory).resolve()
    verify_complete(directory, required=('manifest.json', 'reference.json', 'backtest.json', 'predictions.jsonl',
                                         'dated-inputs.jsonl', 'cohort-membership.jsonl'))
    manifest = read_json(directory / 'manifest.json')
    if any(sha(ROOT / name) != expected for name, expected in manifest['source_hashes'].items()):
        raise ValueError('Replay needs the saved source version')
    with offline_guard(root=ROOT, output=directory):
        results = compute(directory, lambda message: print(message, flush=True))
        for name, value in results.items():
            if name.endswith('.jsonl'):
                result = hashlib.sha256()
                for line in lines_bytes(value):
                    result.update(line)
                equal = result.hexdigest() == sha(directory / name)
            else:
                equal = value == read_json(directory / name)
            if not equal:
                raise ValueError(f'Offline replay differs: {name}')
    return {'status': 'exact_offline_replay', 'files': sorted(results),
            'scored_fixtures': results['backtest.json']['scored_fixtures']}


def run(targets, frozen, output):
    targets, frozen, output = (Path(p).resolve() for p in (targets, frozen, output))
    if output.exists():
        raise FileExistsError(output)
    target_seal = verify_complete(targets, required=TARGET_FILES)
    frozen_name = 'source/Scripts/data_platform/features/phase4_card_reconstruction.py'
    verify_complete(frozen, required=('fixed-weights.json', frozen_name))
    formulas = function_hashes((frozen / frozen_name).read_text())
    if formulas != function_hashes(inspect.getsource(model)):
        raise ValueError('The fixed reference formulas have changed')
    # Resolve imports/dependency metadata before activating the offline guard.
    deps = {p: importlib.metadata.version(p) for p in ('numpy', 'scipy')}
    sources = source_files()
    output.mkdir(parents=True, exist_ok=False)
    with offline_guard(root=ROOT, output=output):
        (output / 'inputs').mkdir()
        for name in (*TARGET_FILES, 'COMPLETE.json'):
            shutil.copyfile(targets / name, output / 'inputs' / name)
        # Verify the actual copied bytes, not only the source before copying.
        verify_complete(output / 'inputs', required=TARGET_FILES)
        shutil.copyfile(frozen / 'fixed-weights.json', output / 'fixed-weights.json')
        write_json(output / 'reference.json', {
            'source': str(frozen), 'seal_sha256': sha(frozen / 'COMPLETE.json'),
            'weights_sha256': sha(frozen / 'fixed-weights.json'), 'function_hashes': formulas,
        })
        for source in sources + [ROOT / p for p in PROTOCOL_FILES]:
            target = output / 'source' / source.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        write_json(output / 'manifest.json', {
            'version': VERSION, 'targets': str(targets), 'target_seal': target_seal,
            'source_hashes': {str(p.relative_to(ROOT)): sha(p) for p in sources},
            'protocol_hashes': {p: sha(ROOT / p) for p in PROTOCOL_FILES},
            'dependencies': deps, 'python': sys.version, 'seed': model.SEED,
            'target_policy': policy.POLICY, 'availability': 'assumed_final',
            'api_calls': 0, 'database_access': False, 'production_changes': False,
            'weight_optimization': False, 'later_outcome_access': False,
        })
        results = compute(output, lambda message: print(message, flush=True))
        for name, value in results.items():
            if name.endswith('.jsonl'):
                with (output / name).open('xb') as handle:
                    for line in lines_bytes(value):
                        handle.write(line)
            else:
                write_json(output / name, value)
        complete(output)
    return results['backtest.json']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    create = commands.add_parser('run')
    create.add_argument('--targets', type=Path, required=True)
    create.add_argument('--frozen', type=Path, required=True)
    create.add_argument('--output', type=Path, required=True)
    verify = commands.add_parser('replay')
    verify.add_argument('--experiment', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'run':
        result = run(args.targets, args.frozen, args.output)
        print(json.dumps({k: result[k] for k in ('status', 'scored_fixtures', 'metrics')}, indent=2))
    else:
        print(json.dumps(replay(args.experiment), indent=2))


if __name__ == '__main__':
    main()
