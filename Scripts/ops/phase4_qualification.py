"""One scheduled, research-only qualification of the frozen Phase 4 finalists."""
import argparse
from collections import Counter,defaultdict
from datetime import datetime,timezone,timedelta
import gzip
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys
import os
from types import SimpleNamespace

from Scripts.ops.phase4_baseline import sha,verify,write
from Scripts.data_platform.features import phase4_qualification as q,phase4_finalists as f,phase4_calibration as c
from Scripts.data_platform.features.benchmarks import confirmation_data as reader
from Scripts.data_platform.features.benchmarks.data import _decode,_validate_feature_row

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'Research/phase4-baseline-2026-09-28/control'
FINALISTS=ROOT/'Research/phase4-finalists-2026-09-29'
CAL=ROOT/'Research/phase4-step3-calibration-extended-2026-09-29'
DATASET=ROOT/'Index/prediction_experiments/phase3-dataset-2026-09-26'
PROTOCOL='docs/phase4-qualification-amendment-2026-09-29.md'
SOURCES=[PROTOCOL,'Scripts/ops/phase4_qualification.py','Scripts/ops/phase4_qualification_worker.py',
    'Scripts/ops/phase4_historical_worker.py','Scripts/ops/phase4_candidate_worker.py','Scripts/ops/phase4_baseline.py',
    *['Scripts/data_platform/features/'+n+'.py' for n in ('phase4_qualification','phase4_finalists','phase4_candidates',
       'phase4_candidate_adapter','phase4_history','phase4_cards','phase4_backtest','phase4_calibration','count_calibration')],
    'Scripts/data_platform/features/benchmarks/confirmation_data.py','Scripts/data_platform/features/benchmarks/data.py',
    'Scripts/tests/test_phase4_qualification.py']
DATA_FILES={'splits.json','manifest.json','feature-schema.json','audit/inputs.json',
            'lockbox/calibration/features.jsonl','lockbox/calibration/labels.jsonl'}


def now():return datetime.now(timezone.utc).isoformat()


def lock(output,name,paths):
    values={str(p.relative_to(output)):sha(p) for p in sorted(paths)}
    write(output/name,values)
    for relative in values:(output/relative).chmod(0o444)
    (output/name).chmod(0o444)


def check(output,name='PREPARED.json'):
    for relative,expected in json.loads((output/name).read_text()).items():
        p=output/relative
        if p.is_symlink() or not p.resolve().is_relative_to(output) or sha(p)!=expected:
            raise ValueError('Qualification artifact changed: '+relative)
    manifest=json.loads((output/'manifest.json').read_text())
    for relative,expected in manifest['sources'].items():
        if sha(ROOT/relative)!=expected:raise ValueError('Implementation changed after preparation: '+relative)


def dataset_read(output,name):
    if name not in DATA_FILES:raise ValueError('Forbidden qualification data store')
    if name=='audit/inputs.json' and not (output/'HISTORY_OPENED.json').exists():
        raise ValueError('Historical opening must be recorded')
    if name.startswith('lockbox/') and not (output/'QUALIFICATION_OPENED.json').exists():
        raise ValueError('Qualification opening must be recorded')
    p=DATASET/name
    if p.resolve()!=p:raise ValueError('Dataset symlink refused')
    expected=json.loads((output/'dataset-hashes.json').read_text())[name]
    body=p.read_bytes()
    if hashlib.sha256(body).hexdigest()!=expected:raise ValueError('Dataset checksum mismatch')
    return body


def jsonl(path,rows):
    with path.open('x') as stream:
        for row in rows:stream.write(f.encode(row)+'\n')


def prepare(output):
    if output.exists():raise FileExistsError(output)
    inputs={str(p.relative_to(ROOT)):{'entries':verify(p),'manifest_sha256':sha(p/'COMPLETE.json')} for p in (BASE,FINALISTS,CAL)}
    recipe=json.loads((FINALISTS/'finalists.json').read_text())
    if recipe['id']!=q.RECIPE_ID or recipe['id']!=f.digest({k:v for k,v in recipe.items() if k!='id'}):
        raise ValueError('Unexpected finalist recipe')
    output.mkdir(parents=True);write(output/'INCOMPLETE.json',{'version':q.VERSION,'started_at':now()})
    for relative in SOURCES:
        dest=output/'implementation'/relative;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/relative,dest)
    hashes=json.loads((DATASET/'COMPLETE.json').read_text())
    write(output/'dataset-hashes.json',{name:hashes[name] for name in sorted(DATA_FILES)})
    for name in ('splits.json','manifest.json','feature-schema.json'):
        (output/('dataset-'+name)).write_bytes(dataset_read(output,name))
    splits=json.loads((output/'dataset-splits.json').read_text())
    if splits['sha256']!=f.digest({k:v for k,v in splits.items() if k!='sha256'}) or splits['membership_sha256']!=f.digest(splits['memberships']):
        raise ValueError('Split identity changed')
    write(output/'preflight.json',q.preflight(splits['memberships']))
    shutil.copyfile(FINALISTS/'finalists.json',output/'finalist-recipe.json')
    shutil.copyfile(CAL/'tuning'/(q.CALIBRATOR_ID+'.json'),output/'corner-calibrator.json')
    calibration=json.loads((output/'corner-calibrator.json').read_text())
    if calibration['id']!=q.CALIBRATOR_ID or calibration['id']!=c.digest({k:v for k,v in calibration.items() if k!='id'}):
        raise ValueError('Wrong sealed calibrator')
    write(output/'manifest.json',{'version':q.VERSION,'sources':{n:sha(ROOT/n) for n in SOURCES},'input_archives':inputs,
        'dataset_complete_sha256':sha(DATASET/'COMPLETE.json'),'period':[q.START,q.END],'created_at':now(),
        'seed':20260928,'dependencies':{d.metadata.get('Name','unknown'):d.version for d in importlib.metadata.distributions()},
        'publication_enabled':False,'final_system_test_opened':False,'2025_outcomes_decoded':False})
    lock(output,'PREPARED.json',[p for p in output.rglob('*') if p.is_file() and p.name!='INCOMPLETE.json'])
    print('Prepared prospective protocol, code, method recipe, calibrator and metadata; no reserved outcomes opened.',flush=True)


def fit(output):
    check(output)
    write(output/'HISTORY_OPENED.json',{'at':now(),'version':q.VERSION,'purpose':'fixed-method season-2024 refit and dated state',
        'availability_before':q.START,'2024_reuse':'previously consumed Phase 3 confirmation; no new selection/evaluation',
        'prepared_lock_sha256':sha(output/'PREPARED.json'),'2025_outcomes_decoded':False})
    history,_,audit=reader.history_before(dataset_read(output,'audit/inputs.json').decode(),q.START)
    jsonl(output/'fitting-history.jsonl',history);write(output/'fitting-read-audit.json',audit)
    memberships=json.loads((output/'dataset-splits.json').read_text())['memberships']
    eligible={r['fixture_id'] for r in memberships if 'goals' in r['eligible_markets'] and q.cm.utc(r['kickoff'])<q.cm.utc(q.START)}
    bundles={league:q.fit_strength(history,eligible,league) for league in f.LEAGUES}
    for bundle in bundles.values():q.check_bundle(bundle)
    write(output/'strength-bundles.json',bundles)
    write(output/'fit-summary.json',{league:{k:b[k] for k in ('id','season','cutoff','fixture_count','effective_fixture_count','iterations')}
                                     for league,b in bundles.items()})
    lock(output,'METHOD_LOCK.json',[p for p in output.rglob('*') if p.is_file() and p.name!='INCOMPLETE.json'])
    print(json.dumps({'fixed_strength_fits':{lg:b['fixture_count'] for lg,b in bundles.items()},'calibrator_refitted':False,
                      '2025_outcomes_decoded':False,'method_lock_sha256':sha(output/'METHOD_LOCK.json')}),flush=True)


def open_qualification(output):
    check(output,'METHOD_LOCK.json')
    # The exclusive opening claim persists through any error or interrupted run.
    write(output/'QUALIFICATION_OPENED.json',{'at':now(),'version':q.VERSION,'period':[q.START,q.END],
        'method_lock_sha256':sha(output/'METHOD_LOCK.json'),'purpose':'single qualification, no fitting/selection',
        'protected_from':q.END,'opening_claim_permanent_even_if_operation_fails':True})
    history,_,audit=reader.history_before(dataset_read(output,'audit/inputs.json').decode(),q.END)
    data=SimpleNamespace(memberships={r['fixture_id']:r for r in json.loads((output/'dataset-splits.json').read_text())['memberships']},
        schema=json.loads((output/'dataset-feature-schema.json').read_text()),manifest=json.loads((output/'dataset-manifest.json').read_text()),
        boundaries={'initial_training_start':q.cm.utc('2019-08-30T00:00:00+00:00')})
    rows=[_decode(line) for line in dataset_read(output,'lockbox/calibration/features.jsonl').splitlines() if line.strip()]
    for row in rows:_validate_feature_row(data,row,partitions={'calibration'},start=q.cm.utc(q.START),end=q.cm.utc(q.END))
    if len({r['fixture']['fixture_id'] for r in rows})!=len(rows) or len({r['snapshot_id'] for r in rows})!=len(rows):
        raise ValueError('Duplicate fixture/snapshot')
    expected={fid for fid,r in data.memberships.items() if r['partition']=='calibration'}
    if {r['fixture']['fixture_id'] for r in rows}!=expected:raise ValueError('Qualification membership mismatch')
    body=dataset_read(output,'lockbox/calibration/labels.jsonl');reader.labels_by_fixture(body,rows)
    labels={r['fixture_id']:r for r in map(_decode,body.splitlines())}
    historical={r['fixture_id']:r for r in history};requests=[];targets=[]
    for row in sorted(rows,key=lambda r:(q.cm.utc(r['as_of']),r['fixture']['fixture_id'])):
        fixture=row['fixture'];fid=fixture['fixture_id'];certified=historical.get(fid);label=labels[fid]
        if certified is None and (any(v is not None for v in label['labels'].values()) or any(d['eligible'] for d in row['market_eligibility'].values())):
            raise ValueError('Uncertified qualification target')
        if certified:
            if any(certified[k]!=fixture[k] for k in ('competition','season','home_team_id','away_team_id','kickoff','status')):
                raise ValueError('Certified fixture identity mismatch')
            if any(certified[s][m]!=label['team_labels'][s][m] for s in ('home','away') for m in f.MARKETS):
                raise ValueError('Target differs from certified history')
        requests.append({'fixture':{k:fixture[k] for k in ('fixture_id','competition','season','home_team_id','away_team_id','kickoff')},
            'as_of':row['as_of'],'forecast_stage':row['forecast_stage'],'availability':'assumed_final',
            'source_snapshot_id':row['snapshot_id'],'source_eligibility':row['market_eligibility']})
        targets.append({**label,'label_available_at':row['label_available_at'],'actual_observed_at':row['actual_observed_at'],
                        'period':'regulation_time','availability':'assumed_final'})
    jsonl(output/'history.jsonl',history);jsonl(output/'requests.jsonl',requests);jsonl(output/'targets.jsonl',targets)
    write(output/'qualification-read-audit.json',{**audit,'qualification_features':len(rows),'qualification_labels':len(labels),
        'decoded_qualification_outcomes':True,'final_system_test_opened':False})
    lock(output,'INPUT_LOCK.json',[output/n for n in ('history.jsonl','requests.jsonl','targets.jsonl',
         'qualification-read-audit.json','QUALIFICATION_OPENED.json')])
    print(json.dumps({'requests':len(requests),'certified_history':len(history),'later_outcomes_decoded':False}),flush=True)


def worker(output,name,args,log):
    with (output/log).open('x') as stream:
        subprocess.run([sys.executable,'-B',str(output/'implementation/Scripts/ops'/name),
            '--source',str(BASE/'workspace'),'--prepared',str(output),*args],cwd=ROOT,
            env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','VECLIB_MAXIMUM_THREADS':'1'},
            stdout=stream,stderr=subprocess.STDOUT,check=True)


def reproduce(output):
    check(output,'METHOD_LOCK.json');check(output,'INPUT_LOCK.json')
    worker(output,'phase4_historical_worker.py',['--output',str(output/'reproduction'),'--qualification'],'reproduction.log')
    lock(output,'CONTROL_LOCK.json',[p for p in (output/'reproduction').iterdir() if p.is_file()])
    print('Historical qualification inputs and frozen controls reproduced.',flush=True)


def predict(output):
    for name in ('METHOD_LOCK.json','INPUT_LOCK.json','CONTROL_LOCK.json'):check(output,name)
    worker(output,'phase4_qualification_worker.py',['--action','predict'],'prediction.log')
    lock(output,'PREDICTION_LOCK.json',[p for p in (output/'predictions').iterdir() if p.is_file()]+[output/'CONTROL_LOCK.json'])
    print('All forecasts locked before aggregate qualification scoring.',flush=True)


def score(output):
    for name in ('METHOD_LOCK.json','INPUT_LOCK.json','CONTROL_LOCK.json','PREDICTION_LOCK.json'):check(output,name)
    write(output/'SCORING_STARTED.json',{'at':now(),'prediction_lock_sha256':sha(output/'PREDICTION_LOCK.json'),
        'protocol_sha256':sha(output/'implementation'/PROTOCOL),'repeat_tuning_allowed':False})
    worker(output,'phase4_qualification_worker.py',['--action','score'],'scoring.log')
    cohorts=defaultdict(list);binary=[];seen=set()
    with gzip.open(output/'scores/records.jsonl.gz','rt') as stream:
        for r in map(json.loads,stream):
            key=(r['configuration'],r['market'],r['fixture_id'])
            if key in seen:raise ValueError('Duplicate qualification score')
            seen.add(key)
            if r['market']=='corner_over9.5_binary':binary.append(r)
            else:cohorts[(r['configuration'],r['market'])].append(r)
    comparisons={};coverage={};claims={}
    for market in f.MARKETS:
        control=sorted(cohorts[('control',market)],key=lambda r:r['fixture_id'])
        candidate=sorted(cohorts[('supported_stack',market)],key=lambda r:r['fixture_id'])
        comparisons[market+':served']=q.compare(control,candidate)
        a,b=([r for r in rows if r['active_scope']] for rows in (control,candidate))
        comparisons[market+':active']=q.compare(a,b);claims[market]=comparisons[market+':active']
        coverage[market]={'served':q.support(control),'active':q.support(a),
            'fallbacks':dict(Counter(reason for r in candidate for reason in r['fallbacks'])),
            'active_league_contributions':dict(Counter(r['league'] for r in a))}
    binary.sort(key=lambda r:r['fixture_id']);raw=[{**r,'scores':r['raw_scores']} for r in binary]
    claims['corner_over9.5_binary']=q.compare(raw,binary,binary=True)
    controls={r['fixture_id']:r for r in cohorts[('control','corners')]}
    reference=[{**r,'scores':c.binary_score(controls[r['fixture_id']]['scores']['totals']['9.5']['binary']['p'],r['scores']['y'])} for r in binary]
    comparisons['corner_calibration:against_control']=q.compare(reference,binary,binary=True)
    comparisons['corner_calibration:against_raw']=claims['corner_over9.5_binary']
    corrected=q.holm({name:r['one_sided_p'] if r['control']['sufficient'] else 1. for name,r in claims.items()})
    decisions={name:{'qualified':bool(r['passes_before_holm'] and corrected[name]['passes']),
        'passes_before_holm':r['passes_before_holm'],'holm':corrected[name],
        'relative_change':r.get('relative_change'),'paired_week_delta95':r.get('paired_week_delta95'),
        'support':{k:r['control'][k] for k in ('n','weeks','sufficient')},
        'supported_slice_regressions':r.get('supported_slice_regressions',[]),
        'related_market_regressions':r.get('related_market_regressions',[])} for name,r in claims.items()}
    write(output/'comparison.json',comparisons);write(output/'coverage.json',coverage)
    write(output/'decision.json',{'version':q.VERSION,'claims':decisions,'period':[q.START,q.END],
        'production_enabled':False,'final_system_test_opened':False,'qualification_period_consumed':True})
    lock(output,'EVALUATION_LOCK.json',[output/n for n in ('comparison.json','coverage.json','decision.json','SCORING_STARTED.json')]
         +[p for p in (output/'scores').iterdir() if p.is_file()])
    print(json.dumps(decisions,indent=2),flush=True)


def audit(output):
    for name in ('METHOD_LOCK.json','INPUT_LOCK.json','CONTROL_LOCK.json','PREDICTION_LOCK.json','EVALUATION_LOCK.json'):check(output,name)
    history={r['fixture_id']:r for r in map(json.loads,(output/'history.jsonl').open())}
    checked=0
    for snapshot in map(json.loads,(output/'reproduction/snapshots.jsonl').open()):
        q.qualification_time(snapshot['as_of']);cutoff=q.cm.utc(snapshot['as_of'])
        if snapshot['snapshot_id']!=f.digest({k:v for k,v in snapshot.items() if k!='snapshot_id'}):raise ValueError('Snapshot identity changed')
        for leagues in snapshot['history_evidence'].values():
            for evidence in leagues.values():
                for key in ('current','prior','recent_six','recent_eight'):
                    for fid in evidence[key]['fixture_ids']:
                        t=q.cm.utc(history[fid]['kickoff'])
                        if t.date()>=cutoff.date() or t+timedelta(hours=3)>=cutoff:raise ValueError('Future/same-day input')
                        checked+=1
    fitting=[json.loads(l) for l in (output/'fitting-history.jsonl').open()]
    memberships=json.loads((output/'dataset-splits.json').read_text())['memberships']
    eligible={r['fixture_id'] for r in memberships if 'goals' in r['eligible_markets'] and q.cm.utc(r['kickoff'])<q.cm.utc(q.START)}
    bundles=json.loads((output/'strength-bundles.json').read_text())
    for league,bundle in bundles.items():
        if q.fit_strength(fitting,eligible,league)!=bundle:raise ValueError('Independent strength refit mismatch')
    counts=Counter();fallback_checks=0;seen={};probability_checks=0
    with gzip.open(output/'predictions/forecasts.jsonl.gz','rt') as stream:
        for row in map(json.loads,stream):
            fid=row['fixture_id'];key=(fid,row['configuration'])
            if key in seen:raise ValueError('Duplicate prediction')
            seen[key]=row
            for market,d in row['probabilities'].items():
                values=[p for rr in d['score_matrix'] for p in rr] if market=='goals' else d['pmf']
                if min(values)<0 or abs(sum(values)-1)>1e-10:raise ValueError('Invalid saved probability distribution')
                counts[row['configuration']+':'+market]+=1;probability_checks+=1
            if row['configuration']=='supported_stack' and not row['active_scope']:
                control=seen[(fid,'control')]
                if row['probabilities']!=control['probabilities'] or any(row['rates'][k]!=control['rates'][k] for k in ('means','variances','goal_means')):
                    raise ValueError('Unsupported scope changed the control')
                fallback_checks+=1
    with gzip.open(output/'scores/records.jsonl.gz','rt') as stream:
        for row in map(json.loads,stream):
            if row['market']=='corner_over9.5_binary':continue
            totals=row['scores']['totals'];over=[]
            for line,profile in sorted(totals.items(),key=lambda x:float(x[0])):
                if min(profile['p'])<0 or abs(sum(profile['p'])-1)>1e-10:raise ValueError('Invalid Asian outcome vector')
                if float(line)%1==.5:over.append(profile['binary']['p'])
            if any(a<b-1e-12 for a,b in zip(over,over[1:])):raise ValueError('Nonmonotone total lines')
    reproduction=json.loads((output/'reproduction/report.json').read_text())
    if reproduction['fixtures']!=reproduction['exact_serialized_replays'] or reproduction['io_violations']:raise ValueError('Incomplete control replay')
    for p in (BASE,FINALISTS,CAL):verify(p)
    write(output/'audit.json',{'version':q.VERSION,'historical_source_references_checked':checked,'strength_refits_exact':len(bundles),
        'probability_distributions_checked':probability_checks,'unsupported_scope_control_equal':fallback_checks,
        'counts':dict(counts),'control_replays':reproduction['exact_serialized_replays'],
        'frozen_archives_verified':True,'final_system_test_opened':False,'publication_enabled':False})
    print(json.dumps(json.loads((output/'audit.json').read_text())),flush=True)


def seal(output):
    for name in ('METHOD_LOCK.json','INPUT_LOCK.json','CONTROL_LOCK.json','PREDICTION_LOCK.json','EVALUATION_LOCK.json'):check(output,name)
    for name in ('audit.json','test-results.xml','review.md'):
        if not (output/name).is_file():raise ValueError('Missing final validation: '+name)
    (output/'INCOMPLETE.json').unlink()
    lock(output,'COMPLETE.json',[p for p in output.rglob('*') if p.is_file()])
    print({'verified_files':verify(output)},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','fit','open_qualification','reproduce','predict','score','audit','seal','verify'))
    parser.add_argument('output',type=Path);args=parser.parse_args();output=args.output.resolve()
    if args.action=='verify':print(verify(output))
    else:globals()[args.action](output)
