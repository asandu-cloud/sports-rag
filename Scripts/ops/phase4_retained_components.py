"""Carry qualified corner/SoT components into an immutable research reference."""
import argparse
from collections import Counter
import gzip
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys

from Scripts.data_platform.features import phase4_retained as retained
from Scripts.data_platform.features.benchmarks.artifacts import complete, read_json, sha, verify_complete, write_json
from Scripts.data_platform.features.benchmarks.isolation import offline_guard

ROOT = Path(__file__).resolve().parents[2]
QUALIFICATION = ROOT / 'Research/phase4-qualification-2026-09-29'
DEVELOPMENT = ROOT / 'Research/phase4-finalists-2026-09-29'
BASELINE = ROOT / 'Research/phase4-baseline-2026-09-28/control'
SOURCES = (
    'Scripts/data_platform/features/phase4_retained.py',
    'Scripts/ops/phase4_retained_components.py',
    'Scripts/data_platform/features/phase4_candidates.py',
    'Scripts/data_platform/features/phase4_finalists.py',
    'Scripts/data_platform/features/count_calibration.py',
    'Scripts/data_platform/features/benchmarks/artifacts.py',
    'Scripts/data_platform/features/benchmarks/isolation.py',
)


def rows(path):
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt') as handle:
        for line in handle:
            yield json.loads(line)


def development_records(value):
    """No labels or scores are opened. These are reference forecasts, not a fit."""
    seen = set()
    for quarter in range(1, 5):
        key = f'2023-Q{quarter}'
        snapshots = {s['fixture']['fixture_id']: s for s in rows(DEVELOPMENT/'folds'/key/'snapshots.jsonl')}
        for r in rows(DEVELOPMENT/'runs'/key/'fixture-configurations.jsonl.gz'):
            if r['configuration'] != 'control':
                continue
            fid = r['fixture_id']
            if fid in seen:
                raise ValueError('Duplicate development fixture')
            seen.add(fid)
            snapshot = snapshots[fid]
            result = retained.retain(snapshot, r['rates'], value,
                                     baseline_id=value['provenance']['baseline_seal_sha256'])
            for field in ('goal_means', 'means'):
                if result[field] != r['rates'][field]:
                    raise ValueError('Retained component changed a control mean')
            yield {'fixture_id': fid, 'snapshot_id': snapshot['snapshot_id'], 'as_of': snapshot['as_of'],
                   'kickoff': snapshot['fixture']['kickoff'], 'competition': snapshot['fixture']['competition'],
                   'fold': key, 'rates': result, 'publication_enabled': False,
                   'goal_probability_source': 'unchanged_frozen_control', 'labels_included': False}


def qualification_parity(value):
    """Replay saved probabilities only; do not read targets or rerun scoring."""
    pending = {}
    counts = Counter()
    for r in rows(QUALIFICATION/'predictions/forecasts.jsonl.gz'):
        if not '2025-01-01' <= r['kickoff'] < '2025-07-01':
            raise ValueError('Unexpected saved qualification date')
        fid = r['fixture_id']
        if r['configuration'] == 'control':
            if fid in pending:
                raise ValueError('Duplicate control forecast')
            pending[fid] = r
            continue
        if r['configuration'] != 'supported_stack' or fid not in pending:
            raise ValueError('Missing paired control')
        control = pending.pop(fid)
        for market in retained.ALPHAS:
            if market not in control['probabilities']:
                continue
            rates = json.loads(json.dumps(control['rates']))
            mean = rates['means'][market]
            if r['active_scope']:
                alpha = value['components'][market]['alpha']
                rates['variances'][market] = mean + alpha * mean * mean
            actual = retained.count_distribution(rates, market)
            if actual != r['probabilities'][market]:
                raise ValueError('Saved qualified distribution changed')
            counts[market + (':active' if r['active_scope'] else ':control_fallback')] += 1
    if pending:
        raise ValueError('Unpaired saved controls')
    return dict(counts)


def verify_sources():
    for path in (QUALIFICATION, DEVELOPMENT, BASELINE):
        verify_complete(path)
    # The pure numerical helpers must remain the ones used for qualification.
    for name in ('phase4_candidates.py', 'phase4_finalists.py', 'count_calibration.py'):
        relative = 'Scripts/data_platform/features/' + name
        if sha(ROOT/relative) != sha(QUALIFICATION/'implementation'/relative):
            raise ValueError('Qualified numerical source changed: ' + name)


def prepare(output):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    # Capture existing publishing source/model hashes without touching live data.
    preserved = {p: sha(ROOT/p) for p in read_json(BASELINE/'source-hashes.json')
                 if (p.startswith('Scripts/') or p.startswith('Index/ml_models/') or p.startswith('requirements'))
                 and (ROOT/p).is_file()}
    output.mkdir(parents=True, exist_ok=False)
    with offline_guard(root=ROOT, output=output):
        verify_sources()
        evidence = output/'evidence'
        evidence.mkdir()
        for name in ('decision.json', 'reviewed-components.json', 'METHOD_LOCK.json', 'COMPLETE.json'):
            shutil.copyfile(QUALIFICATION/name, evidence/name)
        review = read_json(evidence/'reviewed-components.json')
        if review['qualification_decision_sha256'] != sha(evidence/'decision.json'):
            raise ValueError('Review/decision identity mismatch')
        provenance = {
            'qualification_path': str(QUALIFICATION.relative_to(ROOT)),
            'qualification_seal_sha256': sha(QUALIFICATION/'COMPLETE.json'),
            'qualification_decision_sha256': sha(evidence/'decision.json'),
            'qualification_method_lock_sha256': sha(evidence/'METHOD_LOCK.json'),
            'reviewed_components_sha256': sha(evidence/'reviewed-components.json'),
            'development_path': str(DEVELOPMENT.relative_to(ROOT)),
            'development_seal_sha256': sha(DEVELOPMENT/'COMPLETE.json'),
            'baseline_path': str(BASELINE.relative_to(ROOT)),
            'baseline_seal_sha256': sha(BASELINE/'COMPLETE.json'),
        }
        value = retained.bundle(read_json(evidence/'decision.json'), review, provenance)
        write_json(output/'bundle.json', value)
        source_hashes = {}
        for name in SOURCES:
            target = output/'source'/name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT/name, target)
            source_hashes[name] = sha(target)
        fixtures = 0
        components = Counter()
        limitations = Counter()
        with (output/'development-reference.jsonl').open('x') as stream:
            for record in development_records(value):
                fixtures += 1
                for market, component in record['rates']['component_ids'].items():
                    components[market + ':' + component] += 1
                limitations.update(record['rates']['retained_evidence']['support_limitations'])
                stream.write(retained.counts.canonical(record) + '\n')
        parity = qualification_parity(value)
        write_json(output/'verification.json', {
            'development_fixtures': fixtures, 'component_counts': dict(components),
            'development_support_limitations': dict(limitations),
            'saved_qualification_probability_parity': parity,
            'all_control_means_and_goal_rates_unchanged': True,
            'target_labels_decoded': False, 'fitting_or_optimization_performed': False,
            'qualification_rescored': False, 'new_qualification_claim': False,
            'final_system_test_opened': False, 'prospective_reserve_opened': False,
            'publication_enabled': False, 'api_calls': 0,
        })
        write_json(output/'manifest.json', {'version': retained.VERSION, 'bundle_id': value['id'],
                   'source_hashes': source_hashes, 'python': sys.version,
                   'dependencies': {p: importlib.metadata.version(p) for p in ('numpy', 'scipy')},
                   'provenance': provenance})
    after = {p: sha(ROOT/p) for p in preserved}
    if after != preserved:
        raise ValueError('Publishing source/model files changed during preparation')
    write_json(output/'production-preservation.json', {'unchanged': True, 'files': preserved})
    complete(output)
    return read_json(output/'verification.json')


def replay(output):
    output = Path(output).resolve()
    verify_complete(output)
    manifest = read_json(output/'manifest.json')
    if any(sha(ROOT/p) != h for p, h in manifest['source_hashes'].items()):
        raise ValueError('Retained research source changed')
    value = read_json(output/'bundle.json')
    retained.validate(value)
    with offline_guard(root=ROOT, output=output):
        verify_sources()
        expected = rows(output/'development-reference.jsonl')
        n = 0
        for actual in development_records(value):
            if actual != next(expected, None):
                raise ValueError('Retained development replay differs')
            n += 1
        if next(expected, None) is not None:
            raise ValueError('Unexpected extra retained forecast')
        if qualification_parity(value) != read_json(output/'verification.json')['saved_qualification_probability_parity']:
            raise ValueError('Qualification parity differs')
    return {'status': 'exact_replay', 'development_fixtures': n, 'target_labels_decoded': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'replay'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps((prepare if args.action == 'prepare' else replay)(args.output), indent=2))


if __name__ == '__main__':
    main()
