"""Review fixed combinations, lock research finalists, gate reserve access."""
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

from Scripts.ops.phase4_baseline import sha,verify,write
from Scripts.ops.phase4_backtest import check_prepared
from Scripts.data_platform.features import phase4_finalists as f,phase4_backtest as b,phase4_calibration as c
ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'Research/phase4-baseline-2026-09-28/control'
BACKTEST=ROOT/'Research/phase4-step2-backtest-2026-09-29'
CAL=ROOT/'Research/phase4-step3-calibration-extended-2026-09-29'
DATASET=ROOT/'Index/prediction_experiments/phase3-dataset-2026-09-26'
PROTOCOL='docs/phase4-finalist-review-protocol-2026-09-29.md'
SOURCES=[PROTOCOL,'Scripts/ops/phase4_finalists.py','Scripts/ops/phase4_finalist_worker.py','Scripts/ops/phase4_candidate_worker.py',
 'Scripts/data_platform/features/phase4_finalists.py','Scripts/data_platform/features/phase4_backtest.py',
 'Scripts/data_platform/features/phase4_calibration.py','Scripts/data_platform/features/phase4_candidates.py',
 'Scripts/data_platform/features/phase4_candidate_adapter.py','Scripts/data_platform/features/phase4_history.py',
 'Scripts/data_platform/features/count_calibration.py','Scripts/tests/test_phase4_finalists.py']


def prepare(output):
    if output.exists():raise FileExistsError(output)
    inputs={str(p.relative_to(ROOT)):{'entries':verify(p),'sha256':sha(p/'COMPLETE.json')} for p in (BASE,BACKTEST,CAL)}
    output.mkdir(parents=True);write(output/'INCOMPLETE.json',{'version':f.VERSION})
    for n in SOURCES:
        dest=output/'implementation'/n;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/n,dest)
    shutil.copyfile(BACKTEST/'history.jsonl',output/'history.jsonl')
    expected=json.loads((DATASET/'COMPLETE.json').read_text())['splits.json']
    if sha(DATASET/'splits.json')!=expected:raise ValueError('Split metadata changed')
    shutil.copyfile(DATASET/'splits.json',output/'split-metadata.json')
    for q in range(1,5):
        key=f'2023-Q{q}';folder=output/'folds'/key;folder.mkdir(parents=True);(folder/'bundles').mkdir()
        for n in ('snapshots.jsonl','targets.jsonl','eligibility.jsonl','control-rates.jsonl'):
            shutil.copyfile(BACKTEST/'folds'/key/n,folder/n)
        bids=set()
        with gzip.open(BACKTEST/'runs'/key/'forecasts.jsonl.gz','rt') as stream,(folder/'reference.jsonl').open('x') as target:
            for r in map(json.loads,stream):
                if r['candidate_id'] in ('control','strength_goals:10.0','dispersion_corners:0.025','dispersion_sot:0.01'):
                    if r['status']=='invalid':raise ValueError('Invalid reference component')
                    target.write(f.encode(r)+'\n')
                    if r['fit_id']:bids.add(r['fit_id'])
        for bid in bids:shutil.copyfile(BACKTEST/'runs'/key/(bid+'.bundle.json'),folder/'bundles'/(bid+'.json'))
    # Binary diagnostics are separate from the full count distributions.
    with gzip.open(CAL/'evaluation/binary-predictions.jsonl.gz','rt') as stream,gzip.open(output/'corner-diagnostic.jsonl.gz','wt') as target:
        for r in map(json.loads,stream):
            if r['family']=='sigmoid:challenger:corners':target.write(f.encode(r)+'\n')
    write(output/'access-ledger.json',{'purpose':'owner-authorized Steps 4 and 5','outcomes':'inspected development 2022/2023 only',
        'reserved_metadata':'splits.json only, no targets/features','qualification_support_preflight_before_label_access':True,
        'reserved_outcomes_read':False,'production_changes':False})
    write(output/'manifest.json',{'version':f.VERSION,'inputs':inputs,'sources':{n:sha(ROOT/n) for n in SOURCES},
        'split_metadata_sha256':expected,'seed':20260928,'configurations':['component_stack','supported_stack'],
        'dependencies':{d.metadata.get('Name','unknown'):d.version for d in importlib.metadata.distributions()},'publication_enabled':False})
    write(output/'PREPARED.json',{str(p.relative_to(output)):sha(p) for p in sorted(output.rglob('*')) if p.is_file() and p.name!='INCOMPLETE.json'})


def run(output):
    check_prepared(output);(output/'runs').mkdir(exist_ok=False)
    def execute(key):
        with (output/'runs'/(key+'.log')).open('x') as log:
            subprocess.run([sys.executable,'-B',str(output/'implementation/Scripts/ops/phase4_finalist_worker.py'),
                '--source',str(BASE/'workspace'),'--prepared',str(output),'--fold',key],cwd=ROOT,
                env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','VECLIB_MAXIMUM_THREADS':'1'},
                stdout=log,stderr=subprocess.STDOUT,check=True)
        print(key+' complete',flush=True)
    with ThreadPoolExecutor(max_workers=4) as executor:list(executor.map(execute,[f'2023-Q{i}' for i in range(1,5)]))


def records(output):
    for q in range(1,5):
        folder=output/'runs'/f'2023-Q{q}';report=json.loads((folder/'report.json').read_text())
        if report['io_violations'] or report['maximum_component_score_difference']>1e-10:raise ValueError('Invalid composition run')
        with gzip.open(folder/'forecasts.jsonl.gz','rt') as stream:
            yield from map(json.loads,stream)


def review(output):
    check_prepared(output);cohorts=defaultdict(list);seen=set();membership={}
    for r in records(output):
        key=(r['configuration'],r['market'],r['fixture_id'])
        if key in seen:raise ValueError('Duplicate composite forecast')
        seen.add(key);cohorts[(r['configuration'],r['market'])].append(r)
        if r['configuration']=='control':membership[(r['fixture_id'],r['market'])]=r
    comparisons={};coverage={};components={}
    for m in f.MARKETS:
        control=sorted(cohorts[('control',m)],key=lambda r:r['fixture_id'])
        for name in ('component_stack','supported_stack'):
            candidate=sorted(cohorts[(name,m)],key=lambda r:r['fixture_id'])
            comparisons[name+':'+m]=b.compare(control,candidate,diagnostics=True)
            active=[r for r in candidate if r['active_scope']];ref=[r for r in control if r['active_scope']]
            comparisons[name+':'+m+':active_scope']=b.compare(ref,active,diagnostics=True)
            coverage[name+':'+m]={'served':b.support(candidate),'active':b.support(active),
                'fallbacks':dict(Counter(x for r in candidate for x in r['fallbacks']))}
        result=comparisons['supported_stack:'+m+':active_scope']
        components[m]={'method':{'goals':'strength_goals:10.0','corners':'dispersion_corners:0.025','sot':'dispersion_sot:0.01'}[m] if result['passes'] else 'control',
            'development_pass':result['passes'],'active_support':result['control'],
            'leagues':list(f.LEAGUES),'minimum_current_matches_both_teams':8,
            'fallback':'unchanged_control_for_other_scope_or_unknown_sparse_goal_teams',
            'calibration':'identity','related_markets':'shared_goal_score_distribution' if m=='goals' else 'full_count_distribution_and_Asian_vectors'}
    raw=[];calibrated=[];frozen=[]
    with gzip.open(output/'corner-diagnostic.jsonl.gz','rt') as stream:
        for r in map(json.loads,stream):
            if not f.supported(r['league'],r['season_stage']):continue
            meta={k:r[k] for k in ('fixture_id','week','league','season','season_stage','forecast_stage','missingness')}
            calibrated.append({**meta,'scores':r['scores']});raw.append({**meta,'scores':c.binary_score(r['raw_probability'],r['scores']['y'])})
            p=membership[(r['fixture_id'],'corners')]['scores']['totals']['9.5']['binary']['p']
            frozen.append({**meta,'scores':c.binary_score(p,r['scores']['y'])})
    diagnostic={'against_raw_challenger':c.compare_binary(raw,calibrated),'against_frozen_control':c.compare_binary(frozen,calibrated)}
    write(output/'comparison.json',comparisons);write(output/'coverage.json',coverage);write(output/'corner-diagnostic-comparison.json',diagnostic)
    finalist={'version':f.VERSION,'kind':'research_recipe_not_reserve_fitted_or_deployable','components':components,
        'diagnostics':{'corners_over_9_5':{'method':'monotone_sigmoid','C':.01,'base':'dispersion_corners:0.025',
            'development_pass':all(v['passes'] for v in diagnostic.values()),'role':'binary_diagnostic_only_not_count_PMF'}},
        'temperature':1.,'rho':-.1,'cross_market_joint_probability':None,
        'data_availability':'assumed_final','historical_contexts':'same disclosed omissions as Step 1',
        'production_enabled':False,'later_qualification_complete':False,'final_system_test_opened':False,
        'development_fit_source':str(BACKTEST.relative_to(ROOT)),'calibration_fit_source':str(CAL.relative_to(ROOT)),
        'input_manifest_sha256':sha(output/'manifest.json'),'protocol_sha256':sha(output/'implementation'/PROTOCOL),
        'primary_claims_for_future_Holm':[m for m,v in components.items() if v['development_pass']]+(['corner_over9.5_binary'] if all(v['passes'] for v in diagnostic.values()) else [])}
    finalist['id']=f.digest(finalist);write(output/'finalists.json',finalist)
    lock={n:sha(output/n) for n in ('finalists.json','comparison.json','coverage.json','corner-diagnostic-comparison.json')}
    write(output/'FINALIST_LOCK.json',lock)
    for n in lock:(output/n).chmod(0o444)
    (output/'FINALIST_LOCK.json').chmod(0o444)
    print(json.dumps({'components':{m:v['development_pass'] for m,v in components.items()},'binary_diagnostic':finalist['diagnostics']}))


def preflight(output):
    check_prepared(output)
    for n,h in json.loads((output/'FINALIST_LOCK.json').read_text()).items():
        if sha(output/n)!=h:raise ValueError('Finalist lock changed')
    metadata=json.loads((output/'split-metadata.json').read_text())
    report=f.reserve_preflight(metadata['memberships'])
    report['finalist_lock_sha256']=sha(output/'FINALIST_LOCK.json');report['metadata_sha256']=sha(output/'split-metadata.json')
    write(output/'qualification-preflight.json',report)
    write(output/'summary.json',{'step4':'completed','step5':report['status'],'calibration_period_opened':False,
        'qualification_period_scored':False,'final_system_test_opened':False,'production_changed':False})
    print(json.dumps({'status':report['status'],'qualification':report['stages']['qualification']}))


def seal(output):
    check_prepared(output)
    for n,h in json.loads((output/'FINALIST_LOCK.json').read_text()).items():
        if sha(output/n)!=h:raise ValueError('Finalist lock changed')
    if not (output/'qualification-preflight.json').exists():raise ValueError('Missing qualification gate')
    for p in (BASE,BACKTEST,CAL):verify(p)
    (output/'INCOMPLETE.json').unlink()
    write(output/'COMPLETE.json',{str(p.relative_to(output)):sha(p) for p in sorted(output.rglob('*')) if p.is_file()})
    for p in output.rglob('*'):
        if p.is_file():p.chmod(0o444)
    print({'verified_files':verify(output)})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=('prepare','run','review','preflight','seal','verify'));parser.add_argument('output',type=Path)
    args=parser.parse_args();output=args.output.resolve()
    if args.action=='verify':print(verify(output))
    else:globals()[args.action](output)
