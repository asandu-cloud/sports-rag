"""Generate and seal prior-period challenger forecasts for calibration warmup."""
import argparse
from collections import Counter,defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import gzip
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from Scripts.ops.phase4_baseline import sha,verify,write
from Scripts.ops.phase4_backtest import BASE,EARLY,DATA,check_prepared
from Scripts.data_platform.features import phase4_candidates as cm

ROOT=Path(__file__).resolve().parents[2]
BACKTEST=ROOT/'Research/phase4-step2-backtest-2026-09-29'
PREVIOUS=ROOT/'Research/phase4-step3-calibration-2026-09-29'
VERSION='phase4-calibration-earlier-forecasts.v2'
PROTOCOL='docs/phase4-step3-earlier-forecasts-protocol-2026-09-29.md'
SOURCES=[PROTOCOL,'Scripts/ops/phase4_earlier_forecasts.py','Scripts/ops/phase4_backtest_worker.py',
 'Scripts/ops/phase4_candidate_worker.py','Scripts/data_platform/features/phase4_backtest.py',
 'Scripts/data_platform/features/phase4_candidates.py','Scripts/data_platform/features/phase4_candidate_adapter.py',
 'Scripts/data_platform/features/phase4_history.py','Scripts/data_platform/features/count_calibration.py',
 'Scripts/tests/test_phase4_calibration.py']


def earlier_quarter(value):
    t=cm.utc(value)
    if t.year not in range(2019,2022):raise ValueError('Not an earlier forecast date')
    return f'{t.year}-Q{(t.month-1)//3+1}'


def prepare(output):
    if output.exists():raise FileExistsError(output)
    inputs={str(p.relative_to(ROOT)):{'entries':verify(p),'sha256':sha(p/'COMPLETE.json')} for p in (BASE,EARLY,DATA,BACKTEST,PREVIOUS)}
    output.mkdir(parents=True);write(output/'INCOMPLETE.json',{'version':VERSION})
    for name in SOURCES:
        dest=output/'implementation'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,dest)
    with (output/'history.jsonl').open('x') as stream:
        for line in (DATA/'history.jsonl').open():
            if cm.utc(json.loads(line)['kickoff']).year<2022:stream.write(line)
    groups=defaultdict(list);keybyid={}
    for r in map(json.loads,(EARLY/'reproduction/snapshots.jsonl').open()):
        fid=r['fixture']['fixture_id'];key=earlier_quarter(r['fixture']['kickoff'])
        if fid in keybyid:raise ValueError('Duplicate fixture snapshot')
        keybyid[fid]=key;groups[key].append(r)
    for key,rows in groups.items():
        folder=output/'folds'/key;folder.mkdir(parents=True)
        (folder/'snapshots.jsonl').write_text(''.join(cm.canonical(r)+'\n' for r in rows))
    for filename,source in [('targets.jsonl',EARLY/'targets.jsonl'),('eligibility.jsonl',EARLY/'reproduction/eligibility.jsonl'),('control-rates.jsonl',EARLY/'reproduction/control.jsonl')]:
        streams={key:(output/'folds'/key/filename).open('x') for key in groups}
        try:
            for r in map(json.loads,source.open()):
                if filename=='control-rates.jsonl':
                    p={x['market']['group']:x['projection'] for x in r['results']}
                    r={'fixture_id':r['fixture_id'],'means':{m:p[m]['value'] for m in cm.MARKETS},
                       'goal_means':r['goal_means'],'variances':{m:p[m]['variance'] for m in ('corners','sot')}}
                streams[keybyid[r['fixture_id']]].write(cm.canonical(r)+'\n')
        finally:
            for stream in streams.values():stream.close()
    ids=defaultdict(list);coverage=Counter();exclusions=defaultdict(Counter)
    for r in map(json.loads,(EARLY/'reproduction/eligibility.jsonl').open()):
        for m in cm.MARKETS:
            key=keybyid[r['fixture_id']]+':'+m
            if r['markets'][m]['eligible']:ids[m].append(r['fixture_id']);coverage[key]+=1
            exclusions[key].update(r['markets'][m]['reasons'])
    write(output/'fit-eligibility.json',{m:sorted(v) for m,v in ids.items()})
    write(output/'coverage.json',{'eligible':dict(coverage),'exclusions':dict(exclusions),'fixtures':{k:len(v) for k,v in groups.items()}})
    write(output/'manifest.json',{'version':VERSION,'inputs':inputs,'folds':sorted(groups),'sources':{n:sha(ROOT/n) for n in SOURCES},
          'dependencies':{d.metadata.get('Name','unknown'):d.version for d in importlib.metadata.distributions()},'seed':20260928,
          'publication_enabled':False,'later_outcomes_read':False,'availability':'assumed_final'})
    write(output/'PREPARED.json',{str(p.relative_to(output)):sha(p) for p in sorted(output.rglob('*')) if p.is_file() and p.name!='INCOMPLETE.json'})
    print(json.dumps(dict(coverage)),flush=True)


def run(output):
    check_prepared(output);(output/'runs').mkdir(exist_ok=False)
    keys=json.loads((output/'manifest.json').read_text())['folds']
    def execute(key):
        with (output/'runs'/(key+'.log')).open('x') as log:
            subprocess.run([sys.executable,'-B',str(output/'implementation/Scripts/ops/phase4_backtest_worker.py'),
                '--source',str(BASE/'workspace'),'--prepared',str(output),'--fold',key,'--earlier-calibration'],cwd=ROOT,
                env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','VECLIB_MAXIMUM_THREADS':'1'},
                stdout=log,stderr=subprocess.STDOUT,check=True)
        print(key+' completed',flush=True)
    with ThreadPoolExecutor(max_workers=4) as executor:list(executor.map(execute,keys))


def audit(output):
    check_prepared(output)
    (output/'validation').mkdir(exist_ok=True)
    history={r['fixture_id']:r for r in map(json.loads,(output/'history.jsonl').open())}
    ids=json.loads((output/'fit-eligibility.json').read_text());counts=Counter();fallbacks=Counter();fit_errors=Counter()
    bundle_count=0;parity=0;maximum=0.;fit_replays=0;replayed=set()
    for folder in sorted((output/'runs').glob('*-Q*')):
        if not folder.is_dir():continue
        report=json.loads((folder/'report.json').read_text())
        assert not report['invalid'] and not report['io_violations'];parity+=report['eligible_controls_replayed']
        for entry in report['fits']:
            if 'error' in entry:fit_errors[entry['error']]+=1
        bundles={}
        for path in folder.glob('*.bundle.json'):
            b=json.loads(path.read_text());assert b['id']==cm.digest({k:v for k,v in b.items() if k!='id'})
            rows=cm.available_history(list(history.values()),b['cutoff'],competition=b['competition'],seasons={b['season'],b['season']-1},eligible_ids=set(ids['goals']))
            rows=[r for r in rows if all(r[s].get('goals') is not None for s in ('home','away'))]
            assert b['fixture_ids']==[r['fixture_id'] for r in rows]
            training=[{k:r[k] for k in ('fixture_id','competition','season','kickoff','home_team_id','away_team_id')}|{'target':[r['home']['goals'],r['away']['goals']]} for r in rows]
            assert b['training_sha256']==cm.digest(training)
            group=(b['cutoff'][:4],b['ridge'])
            if group not in replayed:
                fitted=cm.fit_strength(list(history.values()),eligible_ids=ids['goals'],competition=b['competition'],market='goals',cutoff=b['cutoff'],season=b['season'],ridge=b['ridge'])
                assert b==fitted;replayed.add(group);fit_replays+=1
            bundles[b['id']]=b;bundle_count+=1
        targets={r['fixture_id']:r for r in map(json.loads,(output/'folds'/folder.name/'targets.jsonl').open())}
        eligible={r['fixture_id']:r['markets'] for r in map(json.loads,(output/'folds'/folder.name/'eligibility.jsonl').open())}
        seen=set();actual=Counter()
        with gzip.open(folder/'forecasts.jsonl.gz','rt') as stream:
            for r in map(json.loads,stream):
                key=(r['fixture_id'],r['market'],r['candidate_id']);assert key not in seen;seen.add(key)
                assert eligible[r['fixture_id']][r['market']]['eligible'] and r['status']!='invalid'
                assert earlier_quarter(r['kickoff'])==folder.name
                counts[r['market']+':'+r['candidate_id']]+=1;actual[r['market']]+=1;fallbacks.update(r['fallbacks'])
                if r['fit_id']:
                    b=bundles[r['fit_id']];assert r['fixture_id'] not in b['fixture_ids'];assert cm.utc(b['cutoff'])<=cm.utc(r['kickoff'])
        for m,multiplier in [('goals',4),('corners',10),('sot',10)]:
            assert actual[m]==sum(e[m]['eligible'] for e in eligible.values())*multiplier
    result={'fitted_bundles':bundle_count,'independent_strength_refits':fit_replays,'exact_control_replays':parity,
            'record_counts':dict(counts),'fallbacks':dict(fallbacks),'unavailable_fits':dict(fit_errors),
            'chronology_and_membership_verified':True,'invalid_forecasts':0,'reserved_outcomes_read':False}
    write(output/'validation/earlier-audit.json',result);print(json.dumps(result),flush=True)


def seal(output):
    audit(output)
    for p in (BASE,EARLY,DATA,BACKTEST,PREVIOUS):verify(p)
    (output/'INCOMPLETE.json').unlink()
    write(output/'COMPLETE.json',{str(p.relative_to(output)):sha(p) for p in sorted(output.rglob('*')) if p.is_file()})
    for p in output.rglob('*'):
        if p.is_file():p.chmod(0o444)
    print({'verified_files':verify(output)})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=('prepare','run','audit','seal','verify'));parser.add_argument('output',type=Path)
    args=parser.parse_args();output=args.output.resolve()
    if args.action=='verify':print(verify(output))
    else:globals()[args.action](output)
