"""Capture candidate implementation and run a bounded, unscored historical smoke."""
from pathlib import Path
import argparse
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys

from Scripts.data_platform.features.phase4_candidates import registry,VERSION
from Scripts.ops.phase4_baseline import verify,sha,write

ROOT=Path(__file__).resolve().parents[2]
BASELINE=ROOT/'Research/phase4-baseline-2026-09-28/control'
INPUTS=ROOT/'Research/phase4-historical-control-2026-09-29'
PROTOCOL='docs/phase4-step2-candidate-protocol-v2-2026-09-29.md'


def capture(output):
    if output.exists(): raise FileExistsError(output)
    verify(BASELINE);verify(INPUTS)
    output.mkdir(parents=True)
    write(output/'INCOMPLETE.json',{'version':VERSION})
    chosen={}
    # Metadata/input-only strata; no target labels or candidate scores are read.
    for line in (INPUTS/'reproduction/snapshots.jsonl').open():
        r=json.loads(line);f=r['fixture'];league=f['competition']
        if league=='EPL':
            if any(meta.get('expected_goals') is not None for meta in r['profiles'].values()): key='EPL:with_xg'
            else: key='EPL:'+('early' if f['kickoff'][5:7]=='08' else 'other')
        elif league in ('UCL','UECL','BelgianProLeague','Championship'): key=league
        else: continue
        if key not in chosen: chosen[key]=r
    snapshots=sorted(chosen.values(),key=lambda r:(r['as_of'],r['fixture']['fixture_id']))
    ids={r['fixture']['fixture_id'] for r in snapshots}
    with (output/'snapshots.jsonl').open('x') as stream:
        for r in snapshots: stream.write(json.dumps(r,sort_keys=True,allow_nan=False)+'\n')
    with (output/'control.jsonl').open('x') as stream:
        for line in (INPUTS/'reproduction/control.jsonl').open():
            if json.loads(line)['fixture_id'] in ids: stream.write(line)
    with (output/'eligibility.jsonl').open('x') as stream:
        for line in (INPUTS/'reproduction/eligibility.jsonl').open():
            if json.loads(line)['fixture_id'] in ids: stream.write(line)
    shutil.copyfile(INPUTS/'history.jsonl',output/'history.jsonl')
    sources=[PROTOCOL,'Scripts/ops/phase4_candidates.py','Scripts/ops/phase4_candidate_worker.py',
        'Scripts/data_platform/features/phase4_candidates.py','Scripts/data_platform/features/phase4_candidate_adapter.py',
        'Scripts/data_platform/features/phase4_history.py','Scripts/tests/test_phase4_candidates.py']
    for name in sources:
        target=output/'implementation'/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/name,target)
    write(output/'registry.json',registry())
    write(output/'manifest.json',{'version':VERSION,'baseline_complete_sha256':sha(BASELINE/'COMPLETE.json'),
        'input_complete_sha256':sha(INPUTS/'COMPLETE.json'),'source_hashes':{n:sha(ROOT/n) for n in sources},
        'purpose':'unscored_implementation_smoke','fixture_ids':sorted(ids),'publication_enabled':False,
        'reserved_labels_accessed':False,'historical_fitting':False,'python':sys.version,
        'dependencies':{d.metadata.get('Name','unknown'):d.version for d in importlib.metadata.distributions()}})
    write(output/'PREPARED.json',{str(p.relative_to(output)):sha(p) for p in output.rglob('*') if p.is_file() and p.name!='INCOMPLETE.json'})
    return output


def smoke(output):
    for name,expected in json.loads((output/'PREPARED.json').read_text()).items():
        p=output/name
        if p.is_symlink() or not p.resolve().is_relative_to(output) or sha(p)!=expected: raise ValueError('Input checksum mismatch')
    subprocess.run([sys.executable,'-B',str(output/'implementation/Scripts/ops/phase4_candidate_worker.py'),
        '--source',str(BASELINE/'workspace'),'--prepared',str(output),'--output',str(output/'smoke')],
        cwd=ROOT,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'},check=True,timeout=1800)


def seal(output):
    report=json.loads((output/'smoke/report.json').read_text())
    if report['control_parity']!=report['fixtures'] or report['io_violations'] or report['invalid']:
        raise ValueError('Smoke failures prevent completion seal')
    verify(BASELINE);verify(INPUTS)
    for name,expected in json.loads((output/'PREPARED.json').read_text()).items():
        if sha(output/name)!=expected: raise ValueError('Input changed during execution')
    (output/'INCOMPLETE.json').unlink()
    write(output/'COMPLETE.json',{str(p.relative_to(output)):sha(p) for p in sorted(output.rglob('*')) if p.is_file()})
    for p in output.rglob('*'):
        if p.is_file():p.chmod(0o444)
    print(json.dumps({'verified_files':verify(output),'report':report}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('capture','smoke','seal','verify'))
    parser.add_argument('output',type=Path)
    args=parser.parse_args();output=args.output.resolve()
    if args.action=='capture':capture(output)
    elif args.action=='smoke':smoke(output)
    elif args.action=='seal':seal(output)
    else: print(json.dumps({'verified_files':verify(output)}))
