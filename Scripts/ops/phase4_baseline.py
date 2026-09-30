"""Capture an isolated Phase 4 control and protocol without reading held-out labels.

Archive dirty working-tree source, active models and only the explicit development
allowlist. Compare fresh-process live-source and archived-source replay, then seal
with checksums. This command never opens the canonical database or runs a provider.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from Scripts.ops.prediction_baseline import source_paths, git
from Scripts.ops.phase4_baseline_replay import scenarios

ROOT = Path(__file__).resolve().parents[2]
VERSION = 'phase4-repaired-statistical-control.v1'
PROTOCOL = 'docs/phase4-evaluation-protocol-2026-09-28.md'
DATASET = 'Index/prediction_experiments/phase3-dataset-2026-09-26'
DEVELOPMENT = ('manifest.json','feature-schema.json','splits.json','lockbox-policy.json',
               'development/features.jsonl','development/labels.jsonl')
DOCS = (PROTOCOL, 'docs/phase4-prior-transition-results-2026-09-28.md',
        'docs/phase4-card-calibration-results-2026-09-28.md',
        'docs/phase4-card-calibration-protocol-2026-09-28.md',
        'docs/phase4-richer-lineup-effects-plan.md', 'docs/participation-settlement-policy.md')


def sha(path):
    with path.open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()


def write(path, value):
    with path.open('x') as f: json.dump(value,f,indent=2,sort_keys=True,allow_nan=False); f.write('\n')


def copy_file(root, destination, relative):
    relative = Path(relative)
    source = root / relative
    if source.is_symlink() or not source.resolve().is_relative_to(root.resolve()):
        raise ValueError('Unsafe source path')
    target = destination / relative
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists(): raise FileExistsError(target)
    before = sha(source)
    shutil.copyfile(source,target)
    if sha(source) != before or sha(target) != before:
        raise ValueError('Source changed during capture: '+str(relative))
    return before


def verify(directory):
    complete = json.loads((directory/'COMPLETE.json').read_text())
    # Only the root COMPLETE is excluded; nested source manifests remain entries.
    actual = {str(p.relative_to(directory)) for p in directory.rglob('*') if p.is_file() and p != directory/'COMPLETE.json'}
    if actual != set(complete): raise ValueError('Manifest membership mismatch')
    for name, expected in complete.items():
        path = directory/name
        if path.is_symlink() or not path.resolve().is_relative_to(directory.resolve()) or sha(path) != expected:
            raise ValueError('Baseline checksum mismatch: '+name)
    return len(complete)


def capture(root, directory):
    root, directory = root.resolve(), directory.resolve()
    directory.mkdir(parents=True,exist_ok=False)
    write(directory/'INCOMPLETE.json',{'version':VERSION,'started_at':datetime.now(timezone.utc).isoformat()})
    workspace = directory/'workspace'
    paths = {str(p.relative_to(root)) for p in source_paths(root)}
    paths.update(DOCS)
    for name in ('requirements-phase3.in','requirements-phase3.lock.txt'):
        if (root/name).is_file(): paths.add(name)
    paths.update(str(p.relative_to(root)) for p in (root/'Index/ml_models').iterdir() if p.is_file() and not p.is_symlink())
    source_hashes = {name:copy_file(root,workspace,name) for name in sorted(paths)}
    write(directory/'source-hashes.json',source_hashes)
    write(directory/'git-state.json',{'head':git(root,'rev-parse','HEAD'),'status':git(root,'status','--porcelain=v1'),
                                     'identity_basis':'actual_file_hashes_including_dirty_and_untracked_source'})
    write(directory/'dependencies.json',{'python':sys.version,'executable':sys.executable,
          'packages':sorted([{'name':d.metadata.get('Name','unknown'),'version':d.version} for d in importlib.metadata.distributions()],key=lambda d:d['name'].lower())})
    dataset_complete=json.loads((root/DATASET/'COMPLETE.json').read_text())
    data_hashes={}
    for name in DEVELOPMENT:
        relative=DATASET+'/'+name
        data_hashes[relative]=copy_file(root,workspace,relative)
        if data_hashes[relative] != dataset_complete[name]: raise ValueError('Development source checksum mismatch')
    # This manifest contains hashes, not reserved outcomes; the dataset reader
    # explicitly allows missing reserved files when reading development only.
    copy_file(root,workspace,DATASET+'/COMPLETE.json')
    write(directory/'data-boundary.json',{'copied_development':data_hashes,
          'never_copied':['audit/inputs.json','audit/evidence.jsonl','snapshot.db','confirmation','calibration','final_system_test','prospective_reserve'],
          'calibration_period':'2025-01-01/2025-07-01','final_system_test':'2025-07-01/2026-07-01',
          'historical_inputs_status':'development features/labels preserved; full-engine historical input reconstruction remains next work',
          'availability':'assumed_final'})
    write(directory/'replay-inputs.json',scenarios())
    env={'PATH':os.environ.get('PATH','/usr/bin:/bin'), 'PYTHONDONTWRITEBYTECODE':'1',
         'PREDICTION_RELEASE_MODE':'shadow','TZ':'UTC','OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'}
    # Fixed runner, explicit source roots, fresh process for every comparison.
    replay=workspace/'Scripts/ops/phase4_baseline_replay.py'
    for name, source in [('current-source.json',root),('archived-source.json',workspace),('archived-repeat.json',workspace)]:
        output=directory/name
        result=subprocess.run([sys.executable,'-B',str(replay),'--source-root',str(source),
                               '--inputs',str(directory/'replay-inputs.json'),'--output',str(output)],
                              cwd=directory,env=env,text=True,capture_output=True,timeout=120)
        if result.returncode:
            (directory/(name+'.failure.txt')).write_text(result.stdout+result.stderr)
            raise RuntimeError('Replay failed; see '+str(directory/(name+'.failure.txt')))
    hashes={name:sha(directory/name) for name in ('current-source.json','archived-source.json','archived-repeat.json')}
    if len(set(hashes.values())) != 1: raise ValueError('Live/frozen or repeated replay mismatch')
    if any(sha(root/name)!=value for name,value in source_hashes.items()): raise ValueError('Source changed before sealing')
    if any(sha(root/name)!=value for name,value in data_hashes.items()): raise ValueError('Development data changed before sealing')
    replay_result=json.loads((directory/'archived-source.json').read_text())
    write(directory/'replay-validation.json',{'exact_byte_match':True,'hashes':hashes,
          'scenarios':len(replay_result['scenarios']),
          'canonical_results':sum(len(x['results']) for x in replay_result['scenarios']),
          'ml_weights':replay_result['ml_weights'],'io_violations':replay_result['io_violations'],
          'historical_published_forecast_replay':False})
    manifest={'version':VERSION,'protocol_version':'phase4-evaluation.v1','protocol_sha256':sha(root/PROTOCOL),
          'frozen_at':datetime.now(timezone.utc).isoformat(),'publication_enabled':False,'promotion_allowed':False,
          'prior_policy':'installed_eight_match_cutoff; gradual_strength_16_is_separate_candidate',
          'cards':'preserved_unqualified_for_fitting','confidence':'existing_heuristic_not_calibrated',
          'scope':'fixed_code_configuration_control_with_synthetic_input_replay; not a historical probability evaluation',
          'baseline_identity':hashlib.sha256(json.dumps({'sources':source_hashes,'data':data_hashes,'protocol':sha(root/PROTOCOL)},sort_keys=True).encode()).hexdigest(),
          'source_files':len(source_hashes),'development_files':len(data_hashes),
          'production_source_unchanged':True,'model_artifacts_unchanged':True,
          'snapshot_or_live_database_read':False,'provider_requests':0,'reserved_outcomes_read':False}
    write(directory/'manifest.json',manifest)
    write(directory/'access-ledger.json',{'events':[{'purpose':'baseline_freeze',
          'outcome_access':'previously_permitted_development_files_only',
          'reserves_opened':[],'protocol_version':'phase4-evaluation.v1'}]})
    (directory/'INCOMPLETE.json').unlink()
    complete={str(p.relative_to(directory)):sha(p) for p in sorted(directory.rglob('*')) if p.is_file()}
    write(directory/'COMPLETE.json',complete)
    verify(directory)
    # Accidental edits fail at filesystem level; verification remains authoritative.
    for p in directory.rglob('*'):
        if p.is_file(): p.chmod(0o444)
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('capture');p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--output',type=Path,required=True)
    p=sub.add_parser('verify');p.add_argument('directory',type=Path)
    args=parser.parse_args()
    print(json.dumps(capture(args.root,args.output) if args.command=='capture' else {'verified_files':verify(args.directory.resolve())},indent=2))


if __name__ == '__main__': main()
