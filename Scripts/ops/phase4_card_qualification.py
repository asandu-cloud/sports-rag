"""One frozen card finalist on explicitly reused Jan--Jun 2025 evidence.

Earlier sealed readers retain their pre-2024 guards. Private function namespaces
inject this versioned period contract without changing any original module global.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import closing
from datetime import timedelta
import gzip
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import sqlite3
import sys
from types import FunctionType, SimpleNamespace

import numpy as np

from Scripts.ops import phase4_card_candidates as dev
from Scripts.data_platform.features import player_history_reconciliation as evidence
from Scripts.data_platform.features import phase4_card_policy as policy
from Scripts.data_platform.features.benchmarks.isolation import offline_guard

m=dev.m
ROOT=dev.ROOT
START=m.cards.utc('2025-01-01')
END=m.cards.utc('2025-07-01')
VERSION='phase4-card-reused-qualification.v1'
PROTOCOL='docs/phase4-card-reused-qualification-2026-10-02.md'


def permitted(kickoff):
    return m.cards.utc(kickoff)+timedelta(hours=3)<END


def scoped_function(fn,**overrides):
    """Dependency injection into a private namespace; original guards unchanged."""
    copy=FunctionType(fn.__code__,{**fn.__globals__,**overrides},fn.__name__,fn.__defaults__,fn.__closure__)
    copy.__kwdefaults__=fn.__kwdefaults__
    return copy


def scoped_cards():
    return SimpleNamespace(**{**vars(m.cards),'permitted':permitted,'END':END})


def qualified(rows):
    return scoped_function(m.fixed.qualified_rows,cards=scoped_cards())(rows,minimum_recorded_minutes=2)


def build_features(rows,context,weights):
    reference=SimpleNamespace(**{**vars(m.fixed),'qualified_rows':lambda rr,**kw:qualified(rr)})
    return scoped_function(m.build_features,fixed=reference)(rows,context,weights)


def read_payload(package,archive,ref,fixture,endpoint):
    # Mandatory metadata boundary before decompressing or decoding any response.
    if not permitted(fixture['kickoff_utc']):raise ValueError('Protected final-system outcome refused')
    path=package/archive['path']
    if not path.resolve().is_relative_to(package/'raw_archive') or path.is_symlink():raise ValueError('Unsafe archive path')
    if dev.sha(path)!=archive['sha256']:raise ValueError('Archive changed')
    raw=gzip.decompress(path.read_bytes())
    if hashlib.sha256(raw).hexdigest()!=ref['payload_digest']:raise ValueError('Payload changed')
    payload=json.loads(raw)
    evidence.collector.validate_identity(fixture,payload,endpoint)
    return payload


def check_development(development):
    dev.verify_complete(development)
    report=dev.read_json(development/'report.json')
    if not report['development_gates_passed']:raise ValueError('No eligible frozen finalist')
    recipe=dev.read_json(development/'METHOD_LOCK.json');cal=dev.read_json(development/'CALIBRATION_LOCK.json')
    if recipe['id']!=m.fixed.digest({k:v for k,v in recipe.items() if k!='id'}):raise ValueError('Bad method lock')
    if cal.get('id')!=m.fixed.digest({k:v for k,v in cal.items() if k!='id'}):raise ValueError('Bad calibration lock')
    for path,expected in dev.read_json(development/'manifest.json')['source_hashes'].items():
        if dev.sha(ROOT/path)!=expected:raise ValueError('Development implementation changed')
    return recipe,cal


def prepare(development,output):
    development,output=(Path(p).resolve() for p in (development,output))
    if output.exists():raise FileExistsError(output)
    recipe,cal=check_development(development)
    package=dev.PACKAGE
    seal=dev.read_json(package/'PREPARED.json')
    for name in ('archives.json','fixture-identities.json','platform.db'):
        if dev.sha(package/name)!=seal[name]:raise ValueError('Prepared source changed: '+name)
    database_hash=dev.sha(dev.DATABASE)
    prepared_hash=dev.sha(package/'PREPARED.json')
    if database_hash!=dev.read_json(dev.DATABASE.parent/'before.json')['backup_sha256']:raise ValueError('Comparison backup changed')
    metadata=dev.read_json(package/'fixture-identities.json')
    admitted=[f for f in metadata if not m.cards.permitted(f['kickoff_utc']) and permitted(f['kickoff_utc'])]
    output.mkdir(parents=True,exist_ok=False)
    # Permanent record BEFORE reading any additional 2024/2025 outcome field.
    dev.write_json(output/'ACCESS_OPENED.json',{'version':VERSION,'development_seal_sha256':dev.sha(development/'COMPLETE.json'),
                 'method_id':recipe['id'],'calibration_id':cal['id'],'owner_approved':'2026-10-02',
                 'qualification_start':START.isoformat(),'end_exclusive':END.isoformat(),
                 'history':'strictly earlier available 2024 context and previously approved development',
                 'reused_period':True,'2024_or_2025_parameter_fitting':False,
                 'additional_fixture_ids':[f['fixture_id'] for f in admitted],
                 'protected_final_system_and_prospective_payloads_decoded':False})
    old=defaultdict(list);fouls=defaultdict(dict)
    with closing(sqlite3.connect(dev.DATABASE.as_uri()+'?mode=ro',uri=True)) as db:
        db.row_factory=sqlite3.Row;db.execute('PRAGMA query_only=ON');db.execute('BEGIN')
        ids=[f['fixture_id'] for f in admitted]
        for start in range(0,len(ids),400):
            chunk=ids[start:start+400]; where=" WHERE julianday(f.kickoff_utc,'+3 hours')<julianday('2025-07-01') AND f.api_football_id IN ("+','.join('?' for _ in chunk)+')'
            for r in db.execute('''SELECT f.api_football_id fixture_id,t.api_football_id team_id,p.api_football_id player_id,
                 x.minutes,x.yellow_cards,x.red_cards FROM fixture_player_stats x JOIN fixtures f ON f.id=x.fixture_id
                 JOIN teams t ON t.id=x.team_id JOIN players p ON p.id=x.player_id'''+where,chunk):old[r['fixture_id']].append(dict(r))
            for r in db.execute('''SELECT f.api_football_id fixture_id,t.api_football_id team_id,x.fouls_committed,x.raw_payload_digest
                 FROM fixture_team_stats x JOIN fixtures f ON f.id=x.fixture_id JOIN teams t ON t.id=x.team_id'''+where,chunk):
                if r['team_id'] in fouls[r['fixture_id']]:raise ValueError('Duplicate foul source')
                fouls[r['fixture_id']][r['team_id']]=dict(r)
        db.rollback()
    with closing(sqlite3.connect((package/'platform.db').as_uri()+'?mode=ro',uri=True)) as db:
        db.row_factory=sqlite3.Row;db.execute('PRAGMA query_only=ON')
        refs={(r['fixture_id'],r['endpoint']):dict(r) for r in db.execute('''SELECT c.fixture_id,c.endpoint,c.archive_id,
             a.payload_digest,a.fetched_at FROM fixture_endpoint_collection c JOIN raw_payload_archive a ON a.id=c.archive_id''')}
    archives={a['id']:a for a in dev.read_json(package/'archives.json')}
    targets=dev.rows(development/'inputs/targets.jsonl')
    context=dev.read_json(development/'inputs/context.json')
    reconstruct=scoped_function(evidence.reconstruct,cards=scoped_cards())
    new=[];exclusions=Counter()
    for i,f in enumerate(admitted,1):
        payloads={};reference={}
        if policy.scope(evidence.fixture_metadata(f))['eligible']:
            for endpoint in evidence.collector.ENDPOINTS:
                ref=refs[f['platform_fixture_id'],endpoint];reference[endpoint]=ref
                payloads[endpoint]=read_payload(package,archives[ref['archive_id']],ref,f,endpoint)
        row=reconstruct(f,payloads,reference,existing_rows=old[f['fixture_id']])
        if row['eligible']:
            row.update(contract=policy.CONTRACT,settlement_policy=policy.POLICY_VERSION,
                       raw_reference={'namespace':'prepared_player_history','batch':dev.read_json(development/'inputs/manifest.json')['prepared_batch'],**reference['/fixtures/players']})
            item={'fouls':{},'foul_sources':{},'starters':{},'lineup_reference':reference['/fixtures/lineups']}
            for side in ('home','away'):
                source=fouls[f['fixture_id']].get(f[side+'_team_id'],{})
                value=source.get('fouls_committed')
                if value is not None and (type(value) not in (int,float) or value<0 or not float(value).is_integer()):raise ValueError('Invalid foul count')
                item['fouls'][side]=value;item['foul_sources'][side]=source
            for team in payloads['/fixtures/lineups']['response']:
                side='home' if team['team']['id']==f['home_team_id'] else 'away'
                item['starters'][side]=[p['player']['id'] for p in team['startXI']]
            context[str(f['fixture_id'])]=item
        exclusions.update(row['exclusions']);new.append(row)
        if i%2000==0:print(f'Reconstructed {i} additional permitted history/qualification fixtures.',flush=True)
    targets.extend(new);qualified(targets)
    if dev.sha(dev.DATABASE)!=database_hash or dev.sha(package/'PREPARED.json')!=prepared_hash:
        raise ValueError('Qualification source changed during preparation')
    with offline_guard(root=ROOT,output=output):
        dev.write_rows(output/'targets.jsonl',targets);dev.write_json(output/'context.json',context)
        for name in ('projections.py','weights.py','fixed-weights.json'):shutil.copyfile(development/'inputs'/name,output/name)
        for name in ('METHOD_LOCK.json','CALIBRATION_LOCK.json'):shutil.copyfile(development/name,output/name)
        shutil.copyfile(ROOT/PROTOCOL,output/'protocol.md')
        dev.write_json(output/'report.json',{'version':VERSION,'new_records':len(new),'new_qualified':sum(r['eligible'] for r in new),
                'exclusions':dict(exclusions),'qualification_qualified':sum(r['eligible'] and START<=m.cards.utc(r['kickoff'])<END for r in new),
                'source_database_sha256':database_hash,'prepared_seal_sha256':prepared_hash,
                'api_calls':0,'reused_period':True,'methods_refitted':False})
        dev.complete(output)
    return dev.read_json(output/'report.json')


def comparison(rows,candidate,control):
    result=m.compare(rows,candidate,control)
    if not rows:return result
    sufficient=lambda n,w,league=False:n>=(100 if league else 500) and w>=10
    for key in ('candidate','control'):
        s=result[key];s['sufficient']=sufficient(s['fixtures'],s['weeks'])
    for key,s in result['slices'].items():
        if key.startswith('competition:'):s['sufficient']=sufficient(s['fixtures'],s['weeks'],True)
    delta=result['nll_delta'];by_week=defaultdict(list)
    for r in rows:by_week[m.cards.utc(r['kickoff']).strftime('%G-W%V')].append(r['scores'][candidate]['nll']-r['scores'][control]['nll']-delta)
    groups=[by_week[k] for k in sorted(by_week)];sums=np.array([sum(g) for g in groups]);counts=np.array([len(g) for g in groups])
    draws=np.random.default_rng(m.SEED).integers(0,len(groups),(2000,len(groups)))
    p=(1+np.count_nonzero(sums[draws].sum(axis=1)/counts[draws].sum(axis=1)<=delta))/2001
    passed=result['candidate']['sufficient'] and result['relative_improvement']>=.005 and result['paired_week_bootstrap95'][1]<0 and p<=.05 and all(not s['sufficient'] or s['relative_regression']<=.02 for s in result['slices'].values())
    result.pop('development_gate_passed',None)
    result.update(reused_qualification_gate_passed=bool(passed),one_sided_p=float(p),
                  period_inspected_previously=True,qualification_period='2025-01-01/2025-07-01')
    return result


def calculate(output):
    source=output/'inputs';recipe=dev.read_json(source/'METHOD_LOCK.json');cal=dev.read_json(source/'CALIBRATION_LOCK.json')
    features=build_features(dev.rows(source/'targets.jsonl'),dev.read_json(source/'context.json'),dev.read_json(source/'fixed-weights.json'))
    evaluation=[f for f in features if START<=m.cards.utc(f['kickoff'])<END and f['baseline']]
    engine=m.load_engine((source/'projections.py').read_text(),(source/'weights.py').read_text())
    records=[]
    for f in evaluation:
        observed=[]
        for side in ('home','away'):
            current=[r['own'] for r in f['inputs']['teams'][side] if r['season']==f['season']]
            observed.append(.7*float(np.var(current,ddof=1))+.3*float(np.var(current[-8:],ddof=1)))
        records.append({**{k:f[k] for k in ('fixture_id','competition','season','kickoff','as_of','target','input_id','stage','foul_state')},
                        'means':{'fuller_control':m.full_mean(f,{'id':'fuller_control'},engine)['mean']},
                        'fixed_referee':f['baseline']['team_referee_mean'],'observed_variances':observed})
    by_id={f['fixture_id']:f for f in evaluation};predictions=[]
    print(f'Producing {len(records)} qualification forecasts with frozen parameters.',flush=True)
    for r in records:
        f=by_id[r['fixture_id']];mean=m.full_mean(f,recipe['spec'],engine)['mean'];control=r['means']['fuller_control']
        variance,details=dev.resolve_total_variance('cards',control,*r['observed_variances'])
        ca=max(0.,(variance-control)/control**2)
        predictions.append({**r,'variance_evidence':details,'method_id':recipe['id'],'calibration_id':cal['id'],
             'scores':{'fixed_referee':m.score(r['fixed_referee'],0.,r['target']),
                       'fuller_control':m.score(control,ca,r['target']),
                       'selected_raw':m.score(mean,recipe['alpha'],r['target']),
                       'selected_calibrated':m.score(mean,recipe['alpha'],r['target'],cal['a'],cal['b'])}})
    comparisons={c:comparison(predictions,'selected_calibrated',c) for c in ('fixed_referee','fuller_control','selected_raw')}
    passed=all(comparisons[c].get('reused_qualification_gate_passed',False) for c in ('fixed_referee','fuller_control'))
    report={'version':VERSION,'status':'reused_historical_check_complete','reused_period':True,
            'comparisons':comparisons,'qualification_gates_passed':passed,'production_qualified':False,
            'new_fitting':False,'method_id':recipe['id'],'calibration_id':cal['id'],
            'eligible_evaluation':len(evaluation),'source_qualified_evaluation':sum(START<=m.cards.utc(f['kickoff'])<END for f in features),
            'scope':dict(Counter(r['competition'] for r in predictions)),
            'protected_final_system_and_prospective_outcomes_accessed':False}
    return {'features.jsonl':evaluation,'predictions.jsonl':predictions,'report.json':report}


def run(inputs,output):
    inputs,output=(Path(p).resolve() for p in (inputs,output))
    if output.exists():raise FileExistsError(output)
    dev.verify_complete(inputs);output.mkdir(parents=True,exist_ok=False)
    sources=dev.bridge.source_files()
    dependencies={p:importlib.metadata.version(p) for p in ('numpy','scipy')}
    with offline_guard(root=ROOT,output=output):
        shutil.copytree(inputs,output/'inputs')
        for source in sources+[ROOT/PROTOCOL]:
            dest=output/'source'/source.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,dest)
        dev.write_json(output/'manifest.json',{'version':VERSION,'source_hashes':{str(p.relative_to(ROOT)):dev.sha(p) for p in sources},
                    'inputs_seal_sha256':dev.sha(inputs/'COMPLETE.json'),'protocol_sha256':dev.sha(ROOT/PROTOCOL),
                    'dependencies':dependencies,'python':sys.version,
                    'seed':m.SEED,'reused_period':True,'one_primary_family':'frozen_card_count_nll_against_both_controls'})
        result=calculate(output)
        for name,value in result.items():
            if name.endswith('.jsonl'):dev.write_rows(output/name,value)
            else:dev.write_json(output/name,value)
        dev.complete(output)
    return {'status':result['report.json']['status'],'qualification_gates_passed':result['report.json']['qualification_gates_passed'],
            'eligible_evaluation':result['report.json']['eligible_evaluation']}


def replay(output):
    output=Path(output).resolve();dev.verify_complete(output)
    for path,h in dev.read_json(output/'manifest.json')['source_hashes'].items():
        if dev.sha(ROOT/path)!=h:raise ValueError('Qualification source changed')
    with offline_guard(root=ROOT,output=output):
        result=calculate(output)
        for name,value in result.items():
            if name.endswith('.jsonl'):
                digest=hashlib.sha256()
                for line in dev.bridge.lines_bytes(value):digest.update(line)
                if digest.hexdigest()!=dev.sha(output/name):raise ValueError('Qualification replay mismatch: '+name)
            elif value!=dev.read_json(output/name):raise ValueError('Qualification replay mismatch: '+name)
    return {'status':'exact_offline_replay','files':sorted(result)}


def main():
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='action',required=True)
    a=s.add_parser('prepare');a.add_argument('--development',type=Path,required=True);a.add_argument('--output',type=Path,required=True)
    a=s.add_parser('run');a.add_argument('--inputs',type=Path,required=True);a.add_argument('--output',type=Path,required=True)
    a=s.add_parser('replay');a.add_argument('--output',type=Path,required=True)
    args=p.parse_args();result=prepare(args.development,args.output) if args.action=='prepare' else run(args.inputs,args.output) if args.action=='run' else replay(args.output)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
