"""Prepare, execute and preserve the bounded Phase 4 calibration comparison."""
from collections import defaultdict,Counter
from datetime import datetime
import argparse
import gzip
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from Scripts.ops.phase4_baseline import sha,verify,write
from Scripts.data_platform.features.phase4_calibration import VERSION,encode
ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'Research/phase4-baseline-2026-09-28/control'
EARLY=ROOT/'Research/phase4-historical-control-fitting-2026-09-29'
BACKTEST=ROOT/'Research/phase4-step2-backtest-2026-09-29'
SOURCES=['Scripts/ops/phase4_calibration.py','Scripts/ops/phase4_calibration_worker.py',
 'Scripts/ops/phase4_candidate_worker.py','Scripts/data_platform/features/phase4_calibration.py',
 'Scripts/data_platform/features/phase4_backtest.py','Scripts/tests/test_phase4_calibration.py',
 'docs/phase4-step3-calibration-protocol-2026-09-29.md']


def metadata(snapshot):
    f=snapshot['fixture'];q=snapshot['profile_quality'];n=min(x.get('current_season_matches',0) for x in q.values())
    return {'fixture_id':f['fixture_id'],'kickoff':f['kickoff'],'as_of':snapshot['as_of'],'league':f['competition'],
            'week':'%d-W%02d'%datetime.fromisoformat(f['kickoff']).isocalendar()[:2],
            'year':int(f['kickoff'][:4]),'season':str(f['season']),'season_stage':'0-7' if n<8 else '8-15' if n<16 else '16+',
            'forecast_stage':snapshot['forecast_stage'],'missingness':'xg_partial_or_missing' if any(p.get('xg_home_pm') is None or p.get('xg_away_pm') is None for p in snapshot['profiles'].values()) else 'xg_both_venues',
            'snapshot_id':snapshot['snapshot_id'],'period':'regulation_time'}


def prepare(output,earlier_forecasts=None):
    if output.exists():raise FileExistsError(output)
    inputs={str(p.relative_to(ROOT)):{'entries':verify(p),'sha256':sha(p/'COMPLETE.json')} for p in (BASE,EARLY,BACKTEST)}
    if earlier_forecasts is not None:
        inputs[str(earlier_forecasts.relative_to(ROOT))]={'entries':verify(earlier_forecasts),'sha256':sha(earlier_forecasts/'COMPLETE.json')}
        if json.loads((earlier_forecasts/'manifest.json').read_text())['version']!='phase4-calibration-earlier-forecasts.v2':
            raise ValueError('Unqualified earlier forecast artifact')
    output.mkdir(parents=True);write(output/'INCOMPLETE.json',{'version':VERSION})
    write(output/'access-ledger.json',{'purpose':'owner-authorized Step 3 development',
        'inputs':inputs,'permitted_outcomes':'earlier fitting, 2022 tuning, inspected 2023 development only','reserve_access':False})
    for source in SOURCES:
        dest=output/'implementation'/source;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/source,dest)
    write(output/'warmup-policy.json',{'earlier_fitting':earlier_forecasts is not None,
        'protocol':'phase4-calibration-earlier-forecasts.v2' if earlier_forecasts is not None else VERSION})
    if earlier_forecasts is not None:
        name='docs/phase4-step3-earlier-forecasts-protocol-2026-09-29.md'
        shutil.copyfile(ROOT/name,output/'implementation'/name)
    shutil.copyfile(BASE/'workspace/Scripts/rag_ingest/prob_models.py',output/'frozen_prob_models.py')
    shutil.copyfile(BACKTEST/'selection.json',output/'step2-selection.json')
    counts=Counter()
    for scope in ('early','2022','2023'):
        rows={};targets={}
        folders=[EARLY/'reproduction'] if scope=='early' else [BACKTEST/'folds'/f'{scope}-Q{i}' for i in range(1,5)]
        for folder in folders:
            for r in map(json.loads,(folder/'snapshots.jsonl').open()):rows[r['fixture']['fixture_id']]=metadata(r)
            target_path=EARLY/'targets.jsonl' if scope=='early' else folder/'targets.jsonl'
            for r in map(json.loads,target_path.open()):targets[r['fixture_id']]=r
        records={}
        def add(fid,market,base_id,base):
            target=targets[fid];meta=rows[fid]
            for problem in (('goals','btts') if market=='goals' else (market,)):
                key=(fid,problem)
                if key not in records:
                    y=int(target['team_labels']['home']['goals']>0 and target['team_labels']['away']['goals']>0) if problem=='btts' else int(target['labels'][market]>{'goals':2.5,'corners':9.5,'sot':8.5}[market])
                    records[key]={**meta,'problem':problem,'available_at':target['label_available_at'],'y':y,
                        'team_target':[target['team_labels'][s]['goals'] for s in ('home','away')] if market=='goals' else None,
                        'bases':{},'availability':'assumed_final'}
                data={k:v for k,v in base.items() if k not in ('p_total','p_btts')}
                data['p']=base['p_btts'] if problem=='btts' else base['p_total']
                records[key]['bases'][base_id]=data
        if scope=='early':
            eligible={r['fixture_id']:r['markets'] for r in map(json.loads,(EARLY/'reproduction/eligibility.jsonl').open())}
            for r in map(json.loads,(EARLY/'reproduction/control.jsonl').open()):
                fid=r['fixture_id']
                projections={x['market']['group']:x['projection'] for x in r['results']}
                for market,centre in [('goals',2.5),('corners',9.5),('sot',8.5)]:
                    if not eligible[fid][market]['eligible']:continue
                    pmf=r['distributions'][market]['pmf']
                    add(fid,market,'control',{'p_total':sum(p for i,p in enumerate(pmf) if i>centre),
                        'p_btts':r['diagnostics'].get('btts',{}).get('yes'),
                        'mean':sum(r['goal_means']) if market=='goals' else projections[market]['value'],
                        'goal_means':r['goal_means'] if market=='goals' else None,
                        'primary_nll':None,'base_fit_id':None,'fallbacks':[]})
        else:
            for q in range(1,5):
                with gzip.open(BACKTEST/'runs'/f'{scope}-Q{q}'/'forecasts.jsonl.gz','rt') as stream:
                    for line in stream:
                        r=json.loads(line);market=r['market'];candidate=r['candidate_id']
                        family={'goals':'strength_goals','corners':'dispersion_corners','sot':'dispersion_sot'}[market]
                        if candidate!='control' and not candidate.startswith(family+':'):continue
                        if r['status']=='invalid':raise ValueError('Invalid base forecast')
                        scores=r['scores'];d=scores['distribution'];centre={'goals':'2.5','corners':'9.5','sot':'8.5'}[market]
                        add(r['fixture_id'],market,candidate,{'p_total':scores['totals'][centre]['binary']['p'],
                            'p_btts':scores['derived'].get('btts',{}).get('p'),
                            'mean':sum(d['goal_means']) if market=='goals' else d['mean'],
                            'goal_means':d.get('goal_means'),'primary_nll':scores['nll'],
                            'base_fit_id':r['fit_id'],'fallbacks':r['fallbacks']})
        if scope=='early' and earlier_forecasts is not None:
            add_earlier_bases(records,earlier_forecasts)
        with (output/(scope+'.jsonl')).open('x') as stream:
            for key in sorted(records,key=lambda k:(records[k]['kickoff'],k)):
                r=records[key];assert 'control' in r['bases'];stream.write(encode(r)+'\n');counts[scope+':'+r['problem']]+=1
    write(output/'coverage.json',dict(counts))
    write(output/'manifest.json',{'version':VERSION,'inputs':inputs,'source_hashes':{name:sha(ROOT/name) for name in SOURCES},
        'frozen_probability_sha256':sha(output/'frozen_prob_models.py'),'publication_enabled':False,
        'dependencies':{d.metadata.get('Name','unknown'):d.version for d in importlib.metadata.distributions()},'seed':20260928})
    write(output/'PREPARED.json',{str(p.relative_to(output)):sha(p) for p in sorted(output.rglob('*')) if p.is_file() and p.name!='INCOMPLETE.json'})
    print(dict(counts))


def add_earlier_bases(records,earlier_forecasts):
    seen=set()
    for folder in sorted((earlier_forecasts/'runs').glob('*-Q*')):
        if not folder.is_dir():continue
        report=json.loads((folder/'report.json').read_text())
        if report['invalid'] or report['io_violations']:raise ValueError('Invalid earlier forecast run')
        with gzip.open(folder/'forecasts.jsonl.gz','rt') as stream:
            for r in map(json.loads,stream):
                if r['status']=='invalid' or int(r['kickoff'][:4])>=2022:raise ValueError('Invalid earlier forecast')
                m=r['market'];bid=r['candidate_id'];scores=r['scores'];d=scores['distribution']
                for problem in (('goals','btts') if m=='goals' else (m,)):
                    key=(r['fixture_id'],problem);row=records[key];identity=(*key,bid)
                    if identity in seen or row['snapshot_id']!=r['snapshot_id']:raise ValueError('Earlier identity mismatch')
                    seen.add(identity)
                    p=scores['derived']['btts']['p'] if problem=='btts' else scores['totals'][{'goals':'2.5','corners':'9.5','sot':'8.5'}[m]]['binary']['p']
                    if bid=='control':
                        if abs(row['bases']['control']['p']-p)>1e-10:raise ValueError('Earlier control probability mismatch')
                        continue
                    row['bases'][bid]={'p':p,'mean':sum(d['goal_means']) if m=='goals' else d['mean'],
                        'goal_means':d.get('goal_means'),'primary_nll':scores['nll'],'base_fit_id':r['fit_id'],'fallbacks':r['fallbacks']}
    for row in records.values():
        if len(row['bases'])!=(4 if row['problem'] in ('goals','btts') else 10):raise ValueError('Incomplete earlier candidate slate')


def check(output):
    for name,expected in json.loads((output/'PREPARED.json').read_text()).items():
        if sha(output/name)!=expected:raise ValueError('Prepared input changed: '+name)


def run(output,stage):
    check(output)
    if stage=='evaluate':
        lock=json.loads((output/'tuning/LOCK.json').read_text())
        for name,expected in lock.items():
            if sha(output/'tuning'/name)!=expected:raise ValueError('Fitted bundle/selection lock changed')
    subprocess.run([sys.executable,'-B',str(output/'implementation/Scripts/ops/phase4_calibration_worker.py'),
        '--prepared',str(output),'--stage',stage],cwd=ROOT,
        env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','VECLIB_MAXIMUM_THREADS':'1'},check=True)


def seal(output):
    check(output)
    for parent in (BASE,EARLY,BACKTEST):verify(parent)
    for name,entry in json.loads((output/'manifest.json').read_text())['inputs'].items():
        parent=ROOT/name;verify(parent)
        if sha(parent/'COMPLETE.json')!=entry['sha256']:raise ValueError('Parent identity changed')
    for name,expected in json.loads((output/'tuning/LOCK.json').read_text()).items():
        if sha(output/'tuning'/name)!=expected:raise ValueError('Changed fitting output')
    for folder in ('tuning','evaluation'):
        r=json.loads((output/folder/'audit.json').read_text())
        if r['io_violations']:raise ValueError('IO audit failed')
    (output/'INCOMPLETE.json').unlink()
    write(output/'COMPLETE.json',{str(p.relative_to(output)):sha(p) for p in sorted(output.rglob('*')) if p.is_file()})
    for p in output.rglob('*'):
        if p.is_file():p.chmod(0o444)
    print({'verified_files':verify(output)})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=('prepare','tune','evaluate','seal','verify'));parser.add_argument('output',type=Path)
    parser.add_argument('--earlier-forecasts',type=Path)
    args=parser.parse_args();output=args.output.resolve()
    if args.action=='prepare':prepare(output,args.earlier_forecasts.resolve() if args.earlier_forecasts else None)
    elif args.action in ('tune','evaluate'):run(output,args.action)
    elif args.action=='seal':seal(output)
    else:print(verify(output))
