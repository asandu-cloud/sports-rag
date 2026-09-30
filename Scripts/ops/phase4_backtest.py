"""Prepare/run/select/report/seal the bounded Phase 4 development comparison."""
import argparse
from collections import defaultdict,Counter
from concurrent.futures import ThreadPoolExecutor
import gzip
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from Scripts.data_platform.features.phase4_candidates import registry,canonical
from Scripts.data_platform.features.phase4_backtest import quarter,select_settings,compare,VERSION
from Scripts.ops.phase4_baseline import sha,verify,write

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'Research/phase4-baseline-2026-09-28/control'
DATA=ROOT/'Research/phase4-historical-control-2026-09-29'
EARLY=ROOT/'Research/phase4-historical-control-fitting-2026-09-29'
CANDIDATES=ROOT/'Research/phase4-step2-candidates-2026-09-29'
PROTOCOL='docs/phase4-step2-backtest-execution-2026-09-29.md'
SOURCES=[PROTOCOL,'Scripts/ops/phase4_backtest.py','Scripts/ops/phase4_backtest_worker.py',
 'Scripts/ops/phase4_candidate_worker.py','Scripts/data_platform/features/phase4_backtest.py',
 'Scripts/data_platform/features/phase4_candidates.py','Scripts/data_platform/features/phase4_candidate_adapter.py',
 'Scripts/data_platform/features/phase4_history.py','Scripts/tests/test_phase4_backtest.py']


def prepare(output):
    if output.exists():raise FileExistsError(output)
    inputs={str(p.relative_to(ROOT)):{'verified':verify(p),'complete_sha256':sha(p/'COMPLETE.json')} for p in (BASE,DATA,EARLY,CANDIDATES)}
    output.mkdir(parents=True);write(output/'INCOMPLETE.json',{'version':VERSION})
    write(output/'access-ledger.json',{'purpose':'owner-authorized Step 2 forward development backtest',
        'outcomes':'pre2022 fitting; 2022 tuning; 2023 later development','reserved_outcomes_read':False,'inputs':inputs})
    for name in SOURCES:
        dest=output/'implementation'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,dest)
    shutil.copyfile(DATA/'history.jsonl',output/'history.jsonl')
    groups=defaultdict(set);streams={}
    try:
        for line in (DATA/'reproduction/snapshots.jsonl').open():
            r=json.loads(line);key=quarter(r['fixture']['kickoff']);fid=r['fixture']['fixture_id']
            if key not in streams:
                folder=output/'folds'/key;folder.mkdir(parents=True);streams[key]=(folder/'snapshots.jsonl').open('x')
            streams[key].write(line);groups[key].add(fid)
    finally:
        for stream in streams.values():stream.close()
    keybyid={fid:key for key,ids in groups.items() for fid in ids}
    assert len(keybyid)==sum(map(len,groups.values()))
    for filename,source in [('targets.jsonl',DATA/'targets.jsonl'),('eligibility.jsonl',DATA/'reproduction/eligibility.jsonl'),
                            ('control-rates.jsonl',DATA/'reproduction/control.jsonl')]:
        streams={key:(output/'folds'/key/filename).open('x') for key in groups}
        try:
            for line in source.open():
                r=json.loads(line)
                if filename=='control-rates.jsonl':
                    projections={x['market']['group']:x['projection'] for x in r['results']}
                    r={'fixture_id':r['fixture_id'],'means':{m:projections[m]['value'] for m in ('goals','corners','sot')},
                       'goal_means':r['goal_means'],'variances':{m:projections[m]['variance'] for m in ('corners','sot')}}
                streams[keybyid[r['fixture_id']]].write(canonical(r)+'\n')
        finally:
            for stream in streams.values():stream.close()
    ids=defaultdict(set);coverage=Counter();exclusions=defaultdict(Counter)
    for parent in (EARLY,DATA):
        for line in (parent/'reproduction/eligibility.jsonl').open():
            r=json.loads(line)
            for market in ('goals','corners','sot'):
                if r['markets'][market]['eligible']:ids[market].add(r['fixture_id'])
                if parent==DATA:
                    key=keybyid[r['fixture_id']]+':'+market
                    coverage[key]+=int(r['markets'][market]['eligible'])
                    exclusions[key].update(r['markets'][market]['reasons'])
    write(output/'fit-eligibility.json',{m:sorted(v) for m,v in ids.items()})
    write(output/'coverage.json',{'fixtures':{k:len(v) for k,v in groups.items()},'eligible':dict(coverage),'exclusions':dict(exclusions),
                                 'lineups':'unavailable: no qualified dated player evidence','cards':'unqualified target'})
    write(output/'registry.json',registry())
    write(output/'manifest.json',{'version':VERSION,'inputs':inputs,'sources':{n:sha(ROOT/n) for n in SOURCES},
        'folds':sorted(groups),'seed':20260928,'publication_enabled':False,'workers':4,
        'dependencies':{d.metadata.get('Name','unknown'):d.version for d in importlib.metadata.distributions()},'python':sys.version})
    write(output/'PREPARED.json',{str(p.relative_to(output)):sha(p) for p in sorted(output.rglob('*')) if p.is_file() and p.name!='INCOMPLETE.json'})


def check_prepared(output):
    for name,h in json.loads((output/'PREPARED.json').read_text()).items():
        if sha(output/name)!=h:raise ValueError('Changed prepared input: '+name)


def run_year(output,year):
    check_prepared(output)
    if year==2023:
        expected=json.loads((output/'SELECTION_LOCK.json').read_text())
        if sha(output/'selection.json')!=expected['selection_sha256']:raise ValueError('Selection lock changed')
    (output/'runs').mkdir(exist_ok=True)
    def run(key):
        dest=output/'runs'/key
        if dest.exists():raise FileExistsError(dest)
        with (output/'runs'/(key+'.log')).open('x') as log:
            subprocess.run([sys.executable,'-B',str(output/'implementation/Scripts/ops/phase4_backtest_worker.py'),
                '--source',str(BASE/'workspace'),'--prepared',str(output),'--fold',key],cwd=ROOT,
                env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','VECLIB_MAXIMUM_THREADS':'1'},
                stdout=log,stderr=subprocess.STDOUT,check=True)
        print(key+' completed',flush=True)
    with ThreadPoolExecutor(max_workers=4) as executor:list(executor.map(run,[f'{year}-Q{i}' for i in range(1,5)]))


def rows(output,year):
    for quarter_ in range(1,5):
        report=json.loads((output/'runs'/f'{year}-Q{quarter_}'/'report.json').read_text())
        if report['io_violations']:raise ValueError('Invalid worker IO')
        with gzip.open(output/'runs'/f'{year}-Q{quarter_}'/'forecasts.jsonl.gz','rt') as stream:
            for line in stream:yield json.loads(line)


def select(output):
    if (output/'selection.json').exists():raise FileExistsError('Selection already locked')
    slim=[]
    for r in rows(output,2022):
        if 'scores' in r:r['scores']={'nll':r['scores']['nll']}
        slim.append(r)
    result=select_settings(slim,registry())
    write(output/'selection.json',result)
    write(output/'SELECTION_LOCK.json',{'selection_sha256':sha(output/'selection.json'),
        'selected_before_2023_scoring':True,'sources':json.loads((output/'manifest.json').read_text())['sources']})
    (output/'selection.json').chmod(0o444);(output/'SELECTION_LOCK.json').chmod(0o444)
    print(json.dumps(result['selected'],indent=2))


def summarize(output):
    check_prepared(output)
    selected=json.loads((output/'selection.json').read_text())['selected'];results={};coverage={}
    for market in ('goals','corners','sot'):
        cohorts=defaultdict(list)
        for r in rows(output,2023):
            if r['market']==market:cohorts[r['candidate_id']].append(r)
        control=sorted(cohorts['control'],key=lambda r:r['fixture_id'])
        if any(r['status']=='invalid' for r in control):raise ValueError('Invalid control: comparison cannot proceed')
        for key,candidate in selected.items():
            if not key.startswith(market+':'):continue
            if candidate is None:results[key]={'status':'no_valid_2022_setting','passes':False};continue
            cohort=sorted(cohorts[candidate],key=lambda r:r['fixture_id'])
            bad=[r for r in cohort if r['status']=='invalid']
            coverage[key]={'control_n':len(control),'candidate_n':len(cohort),'invalid':len(bad),
                           'fallbacks':dict(Counter(x for r in cohort for x in r.get('fallbacks',[])))}
            if bad or [r['fixture_id'] for r in cohort]!=[r['fixture_id'] for r in control]:
                results[key]={'status':'invalid_or_incomplete_common_cohort','candidate_id':candidate,'passes':False,'failures':bad};continue
            result=compare(control,cohort,diagnostics=True);result.update(candidate_id=candidate,status='compared')
            results[key]=result
    write(output/'comparison.json',results);write(output/'comparison-coverage.json',coverage)
    write(output/'summary.json',{'version':VERSION,'phase':'Step 2 individual candidate development backtest',
        'passing_development_claims':[k for k,v in results.items() if v['passes']],
        'comparison_count':len(results),'reserved_outcomes_read':False,'production_changed':False,
        'lineup_comparison':'unavailable','combinations':'not yet selected or evaluated','calibration':'Step 3 pending'})
    print((output/'summary.json').read_text())


def seal(output):
    check_prepared(output)
    for p in (BASE,DATA,EARLY,CANDIDATES):verify(p)
    assert sha(output/'selection.json')==json.loads((output/'SELECTION_LOCK.json').read_text())['selection_sha256']
    if not (output/'summary.json').exists():raise ValueError('Missing comparison')
    (output/'INCOMPLETE.json').unlink()
    write(output/'COMPLETE.json',{str(p.relative_to(output)):sha(p) for p in sorted(output.rglob('*')) if p.is_file()})
    for p in output.rglob('*'):
        if p.is_file():p.chmod(0o444)
    print(json.dumps({'verified_files':verify(output)}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','run-2022','select','run-2023','summarize','seal','verify'))
    parser.add_argument('output',type=Path);args=parser.parse_args();out=args.output.resolve()
    if args.action=='prepare':prepare(out)
    elif args.action.startswith('run-'):run_year(out,int(args.action[-4:]))
    elif args.action=='select':select(out)
    elif args.action=='summarize':summarize(out)
    elif args.action=='seal':seal(out)
    else:print(verify(out))
