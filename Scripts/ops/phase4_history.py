"""Prepare, reproduce and seal development-only historical Phase 4 controls."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from Scripts.data_platform.features.benchmarks.data import DevelopmentDataset
from Scripts.data_platform.features.phase4_history import VERSION, development_history, encode, request_rows
from Scripts.ops.phase4_baseline import sha, verify, write

ROOT = Path(__file__).resolve().parents[2]
BASELINE = ROOT/'Research/phase4-baseline-2026-09-28/control'
DATASET = Path('Index/prediction_experiments/phase3-dataset-2026-09-26')
SOURCES = ('Scripts/ops/phase4_history.py', 'Scripts/ops/phase4_historical_worker.py',
           'Scripts/data_platform/features/phase4_history.py', 'Scripts/tests/test_phase4_history.py')


def prepare(output, baseline=BASELINE, scope='development'):
    baseline = baseline.resolve()
    verified = verify(baseline)
    workspace = baseline/'workspace'
    dataset = DevelopmentDataset(workspace/DATASET, source_root=workspace)
    complete = json.loads((workspace/DATASET/'COMPLETE.json').read_text())
    history_path = ROOT/DATASET/'audit/inputs.json'
    expected = complete['audit/inputs.json']
    if sha(history_path) != expected:
        raise ValueError('Mixed-history archive differs from frozen source')
    # The sole mixed archive is checked as bytes, then lexical timestamps decide
    # which objects may be decoded. No other field in a reserve row is parsed.
    history, skipped = development_history(history_path.read_text())
    if sha(history_path) != expected:
        raise ValueError('History changed while reading')
    requests, targets = request_rows(dataset, history, scope=scope)
    output.mkdir(parents=True, exist_ok=False)
    write(output/'INCOMPLETE.json', {'version': VERSION, 'status': 'preparing'})
    for name, rows in [('history', history), ('requests', requests), ('targets', targets)]:
        with (output/(name+'.jsonl')).open('x') as stream:
            for row in rows:
                stream.write(encode(row)+'\n')
    write(output/'snapshot-schema.json', {
        'version': VERSION, 'identity': 'SHA256 of canonical snapshot JSON before snapshot_id field',
        'availability': 'assumed_final', 'assumed_label_delay_hours': 3,
        'forecast_cutoff': 'kickoff; strictly earlier calendar days and available before cutoff',
        'scope': scope, 'forecast_years': sorted({int(r['fixture']['kickoff'][:4]) for r in requests}),
        'fixture_grouping': 'one immediately-before-kickoff forecast per exact fixture ID',
        'profiles': 'unchanged frozen current/prior/European arithmetic',
        'recent': 'unchanged frozen current-season six; .85 positional decay',
        'variance': 'unchanged frozen current-season recent eight plus season variance',
        'raw_history': 'history.jsonl keyed by fixture_id; snapshot history_evidence lists actual contributing IDs',
        'targets': 'separate targets.jsonl; regulation goals/corners/SoT; cards unqualified',
        'production_profiles': 'not read or rebuilt; metadata adapter uses exact IDs',
        'comparison_eligibility': 'retain frozen Phase3 reasons, then require an available control distribution',
        'price_policy': 'no historical price reconstruction; diagnostic lines only',
        'reserve_policy': '2024+ outcomes never decoded',
    })
    sources = {}
    for name in SOURCES:
        path = ROOT/name
        if path.exists():
            target = output/'implementation'/name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            sources[name] = sha(path)
    write(output/'manifest.json', {
        'version': VERSION, 'baseline': str(baseline), 'baseline_manifest_sha256': sha(baseline/'manifest.json'),
        'baseline_verified_entries': verified, 'protocol_sha256': sha(workspace/'docs/phase4-evaluation-protocol-2026-09-28.md'),
        'mixed_history_sha256': expected, 'history_rows': len(history), 'reserved_objects_skipped_before_decode': skipped,
        'scope': scope, 'requests': len(requests), 'dataset_hashes': dataset.artifact_hashes, 'source_hashes': sources,
        'publication_enabled': False, 'promotion_allowed': False, 'parameters_fitted': False,
        'availability': 'assumed_final', 'seed': None, 'randomness': 'none',
        'python': sys.version,
        'dependencies': sorted([{'name': d.metadata.get('Name', 'unknown'), 'version': d.version}
                                for d in importlib.metadata.distributions()], key=lambda d: d['name'].lower()),
    })
    write(output/'access-ledger.json', {'purpose': 'Step 1 historical inputs and unchanged control',
        'read': ['frozen baseline source/models/protocol/development allowlist', 'audit/inputs.json: checksum plus lexical pre-2024 decoder'],
        'outcomes_decoded': 'pre-2024 regulation history and development labels only',
        'not_accessed': ['canonical SQLite', 'Chroma', 'provider API', '2024+ outcome objects', 'historical quote stores']})
    write(output/'PREPARED.json', {str(p.relative_to(output)): sha(p)
        for p in sorted(output.rglob('*')) if p.is_file() and p.name != 'INCOMPLETE.json'})
    print(json.dumps({'prepared': str(output), 'requests': len(requests), 'history_rows': len(history), 'reserved_skipped': skipped}), flush=True)


def execute(output, baseline=BASELINE, limit=None):
    prepared_hashes = json.loads((output/'PREPARED.json').read_text())
    def verify_prepared():
        for name, expected in prepared_hashes.items():
            path = output/name
            if path.is_symlink() or not path.resolve().is_relative_to(output) or sha(path) != expected:
                raise ValueError('Prepared input changed: '+name)
    verify_prepared()
    worker = output/'implementation/Scripts/ops/phase4_historical_worker.py'
    command = [sys.executable, '-B', str(worker), '--source', str(baseline.resolve()/'workspace'),
               '--prepared', str(output), '--output', str(output/'reproduction')]
    if limit is not None:
        command += ['--limit', str(limit)]
    subprocess.run(command, cwd=ROOT, env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'}, check=True, timeout=1800)
    verify_prepared()


def seal(output, baseline=BASELINE):
    from Scripts.ops.phase4_history_audit import audit
    manifest = json.loads((output/'manifest.json').read_text())
    report = json.loads((output/'reproduction/report.json').read_text())
    if report['fixtures'] != manifest['requests'] or report['exact_serialized_replays'] != report['fixtures'] or report['io_violations']:
        raise ValueError('Incomplete reproduction; cannot seal')
    if sha(baseline/'manifest.json') != manifest['baseline_manifest_sha256']:
        raise ValueError('Baseline identity changed')
    verify(baseline)
    write(output/'integrity-audit.json', audit(output))
    validation = output/'validation'
    validation.mkdir(exist_ok=True)
    for name in ('Scripts/ops/phase4_history_audit.py', 'Scripts/ops/phase4_history.py', 'Scripts/tests/test_phase4_history.py'):
        shutil.copyfile(ROOT/name, validation/Path(name).name)
    write(output/'context-coverage.json', {'fixtures': report['fixtures'], 'contexts': report['context'],
        'meaning': 'common omissions for all later candidates; not full historical live-context replay'})
    (output/'INCOMPLETE.json').unlink()
    hashes = {str(p.relative_to(output)): sha(p) for p in sorted(output.rglob('*')) if p.is_file()}
    write(output/'COMPLETE.json', hashes)
    for path in output.rglob('*'):
        if path.is_file():
            path.chmod(0o444)
    print(json.dumps({'sealed': str(output), 'files': verify(output), 'fixtures': report['fixtures']}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'run', 'seal', 'verify'))
    parser.add_argument('output', type=Path)
    parser.add_argument('--baseline', type=Path, default=BASELINE)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--scope', choices=('development', 'earlier_fitting'), default='development')
    args = parser.parse_args()
    output = args.output.resolve()
    if args.action == 'prepare': prepare(output, args.baseline, args.scope)
    elif args.action == 'run': execute(output, args.baseline, args.limit)
    elif args.action == 'seal': seal(output, args.baseline)
    else: print(json.dumps({'verified_files': verify(output)}))


if __name__ == '__main__':
    main()
