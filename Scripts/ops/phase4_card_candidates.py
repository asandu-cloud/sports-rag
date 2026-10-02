"""Prepare sealed context and run bounded card improvements offline."""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import closing
import gzip
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import sqlite3
import sys

from Scripts.ops import phase4_card_backtest as bridge
from Scripts.data_platform.features import phase4_card_candidates as m
from Scripts.data_platform.features.benchmarks.artifacts import sha, verify_complete, read_json, write_json, complete
from Scripts.data_platform.features.benchmarks.isolation import offline_guard
from Scripts.rag_ingest.core.prediction_guardrails import resolve_total_variance

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = 'docs/phase4-card-improvements-protocol-2026-10-02.md'
FROZEN = ROOT/'Research/phase4-card-reconstruction-2026-09-30/reconstruction'
FIXED = ROOT/'Research/phase4-card-fixed-backtest-2026-10-02'
PACKAGE = ROOT/'Index/prediction_experiments/player-history-prepared-2026-10-02'
DATABASE = ROOT/'Index/prediction_experiments/player-history-import-rehearsal-2026-10-02/platform-before.db'


def write_rows(path, rows):
    with path.open('xb') as handle:
        for line in bridge.lines_bytes(rows):handle.write(line)


def rows(path):
    with path.open() as handle:return [json.loads(line) for line in handle]


def prepare(output):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError(output)
    verify_complete(FIXED)
    verify_complete(FROZEN)
    expected=read_json(DATABASE.parent/'before.json')['backup_sha256']
    if sha(DATABASE)!=expected:raise ValueError('Immutable foul source backup changed')
    seal=read_json(PACKAGE/'PREPARED.json')
    if sha(PACKAGE/'archives.json')!=seal['archives.json']:raise ValueError('Archive index changed')
    source_report=read_json(FIXED/'inputs/report.json')
    raw=bridge.read_targets(FIXED/'inputs/card-targets.jsonl')
    targets=bridge.adapt_targets(raw,source_report['prepared_batch'])
    admitted={r['fixture_id']:r for r in targets if r['eligible']}
    context={str(fid):{'fouls':{'home':None,'away':None},'foul_sources':{},'starters':{}}
             for fid in admitted}
    # Metadata JOIN resolves provider IDs. SQL filters the period and explicit
    # admitted identities BEFORE any outcome field enters the Python process.
    with closing(sqlite3.connect(DATABASE.as_uri()+'?mode=ro',uri=True)) as db:
        db.execute('PRAGMA query_only=ON'); db.execute('BEGIN')
        ids=list(admitted)
        for start in range(0,len(ids),400):
            chunk=ids[start:start+400]
            query='''SELECT f.api_football_id,t.api_football_id,x.fouls_committed,x.raw_payload_digest
                     FROM fixture_team_stats x JOIN fixtures f ON f.id=x.fixture_id
                     JOIN teams t ON t.id=x.team_id
                     WHERE julianday(f.kickoff_utc,'+3 hours')<julianday('2024-01-01')
                     AND f.api_football_id IN ('''+','.join('?' for _ in chunk)+')'
            for fid,tid,fouls,digest in db.execute(query,chunk):
                target=admitted[int(fid)]
                side=next((s for s in ('home','away') if int(tid)==target[s+'_team_id']),None)
                if side is None:raise ValueError('Foul team identity mismatch')
                if side in context[str(fid)]['foul_sources']:raise ValueError('Duplicate foul source')
                if fouls is not None and (type(fouls) not in (int,float) or fouls<0 or not float(fouls).is_integer()):
                    raise ValueError('Invalid foul count')
                context[str(fid)]['fouls'][side]=fouls
                context[str(fid)]['foul_sources'][side]={'payload_digest':digest,'database_sha256':expected}
        db.rollback()
    archives={a['id']:a for a in read_json(PACKAGE/'archives.json')}
    for fid,target in admitted.items():
        ref=target['source_references']['/fixtures/lineups']; archive=archives[ref['archive_id']]
        path=PACKAGE/archive['path']
        if not path.resolve().is_relative_to(PACKAGE/'raw_archive') or path.is_symlink():raise ValueError('Unsafe archive path')
        if sha(path)!=archive['sha256']:raise ValueError('Lineup archive changed')
        body=gzip.decompress(path.read_bytes())
        if hashlib.sha256(body).hexdigest()!=ref['payload_digest'] or archive['payload_sha256']!=ref['payload_digest']:
            raise ValueError('Lineup payload digest mismatch')
        payload=json.loads(body)
        if str(payload['parameters']['fixture'])!=str(fid) or payload['get']!='fixtures/lineups':
            raise ValueError('Lineup envelope identity mismatch')
        for team in payload['response']:
            side=next((s for s in ('home','away') if team['team']['id']==target[s+'_team_id']),None)
            if side is None:raise ValueError('Lineup team identity mismatch')
            pids=[p['player']['id'] for p in team['startXI']]
            if len(pids)!=11 or len(set(pids))!=11:raise ValueError('Incomplete certified starting XI')
            context[str(fid)]['starters'][side]=pids
        if set(context[str(fid)]['starters'])!={'home','away'}:raise ValueError('Missing lineup team')
        context[str(fid)]['lineup_reference']={'batch':source_report['prepared_batch'],**ref}
    if sha(DATABASE)!=expected:raise ValueError('Foul backup changed during read')
    output.mkdir(parents=True,exist_ok=False)
    with offline_guard(root=ROOT,output=output):
        write_rows(output/'targets.jsonl',targets)
        write_json(output/'context.json',context)
        for source,name in ((FROZEN/'source/Scripts/rag_ingest/core/projections.py','projections.py'),
                            (FROZEN/'source/Scripts/rag_ingest/core/weights.py','weights.py'),
                            (FROZEN/'fixed-weights.json','fixed-weights.json'),(ROOT/PROTOCOL,'protocol.md')):
            shutil.copyfile(source,output/name)
        write_json(output/'manifest.json',{'version':m.VERSION,'source_seals':{str(FIXED):sha(FIXED/'COMPLETE.json'),str(FROZEN):sha(FROZEN/'COMPLETE.json')},
                   'foul_database':str(DATABASE),'foul_database_sha256':expected,'prepared_batch':source_report['prepared_batch'],
                   'archive_index_sha256':seal['archives.json'],'availability':'assumed_final','qualified':len(admitted),
                   'complete_foul_pairs':sum(all(v is not None for v in c['fouls'].values()) for c in context.values()),
                   'original_lineup_announcement_timestamps':0,'lineups':'conditional_scenario_only','api_calls':0})
        complete(output)
    return read_json(output/'manifest.json')


def forecast_recipe(records, features, engine, recipe):
    return [{**r,'mean':m.full_mean(features[r['fixture_id']],recipe['spec'],engine)['mean'],
             'alpha':recipe['alpha'],'fit_cutoff':recipe['cutoff'],'fit_ids':recipe['training_ids'],'fit_id':recipe['id']}
            for r in records]


def calculate(directory, *, replay=False):
    source=directory/'inputs'
    engine=m.load_engine((source/'projections.py').read_text(),(source/'weights.py').read_text())
    print('Building dated card, foul, referee-residual and player inputs.',flush=True)
    features=m.build_features(rows(source/'targets.jsonl'),read_json(source/'context.json'),read_json(source/'fixed-weights.json'))
    by_id={f['fixture_id']:f for f in features}
    records=m.predict_registry(features,engine)
    # This assertion compares original forecasts before fitting any challenger.
    saved={r['fixture_id']:r for r in rows(directory/'fixed-predictions.jsonl')}
    for r in records:
        if m.cards.utc(r['kickoff']).year==2023:
            original=saved[r['fixture_id']]; f=by_id[r['fixture_id']]
            if (r['fixed_referee']!=original['team_referee_mean'] or r['fixed_team']!=original['team_only_mean']
                    or f['fixed_snapshot_id']!=original['snapshot_id']):raise ValueError('Fixed reference parity failed')
    print('Fixed parity passed. Selecting bounded recipes from earlier data only.',flush=True)
    recipe=m.select_recipe(records,by_id,engine,'2023-01-01T00:00:00+00:00')
    # Seal recipe before producing/scoring any 2023 candidate outputs.
    if not replay:write_json(directory/'METHOD_LOCK.json',recipe)
    elif recipe!=read_json(directory/'METHOD_LOCK.json'):raise ValueError('Recipe replay differs')
    nested=[]; folds=[]
    for start,end in [('2022-01-01','2022-04-01'),('2022-04-01','2022-07-01'),('2022-07-01','2022-10-01'),('2022-10-01','2023-01-01')]:
        try:fit=m.select_recipe(records,by_id,engine,start+'T00:00:00+00:00')
        except ValueError as exc:
            if str(exc)!='Insufficient chronological fitting support':raise
            folds.append({'start':start,'end':end,'status':'insufficient_earlier_support'}); continue
        rr=[r for r in records if m.cards.utc(start)<=m.cards.utc(r['kickoff'])<m.cards.utc(end)]
        pred=forecast_recipe(rr,by_id,engine,fit);nested.extend(pred)
        folds.append({'start':start,'end':end,'recipe':fit,'predictions':len(pred)})
    first_quarter=[r for r in records if m.cards.utc('2023-01-01')<=m.cards.utc(r['kickoff'])<m.cards.utc('2023-04-01')]
    nested.extend(forecast_recipe(first_quarter,by_id,engine,recipe))
    try:calibration=m.fit_calibration(nested)
    except ValueError as exc:
        if str(exc)!='Insufficient chronological fitting support':raise
        calibration={'kind':'identity','a':0.,'b':1.,'status':'insufficient_calibration_support',
                     'support':m.fixed.support(nested,fitting=True),'cutoff':'2023-04-01T00:00:00+00:00'}
    if not replay:write_json(directory/'CALIBRATION_LOCK.json',calibration)
    elif calibration!=read_json(directory/'CALIBRATION_LOCK.json'):raise ValueError('Calibration replay differs')
    evaluation=[r for r in records if m.cards.utc('2023-04-01')<=m.cards.utc(r['kickoff'])<m.cards.END]
    predictions=[]
    print(f'Scoring {len(evaluation)} later development fixtures; recipe and calibration locked.',flush=True)
    for r in forecast_recipe(evaluation,by_id,engine,recipe):
        mu=r['means']['fuller_control']
        variance,details=resolve_total_variance('cards',mu,*r['observed_variances'])
        observed_alpha=max(0.,(variance-mu)/mu**2)
        specs={'fixed_team':(r['fixed_team'],0.,0.,1.),'fixed_referee':(r['fixed_referee'],0.,0.,1.),
               'fuller_poisson':(mu,0.,0.,1.),'fuller_control':(mu,observed_alpha,0.,1.),
               'selected_mean_poisson':(r['mean'],0.,0.,1.),'selected_raw':(r['mean'],r['alpha'],0.,1.),
               'selected_calibrated':(r['mean'],r['alpha'],calibration['a'],calibration['b']),
               'lineup_scenario':(r['lineup_scenario_mean'],observed_alpha,0.,1.)}
        predictions.append({**{k:v for k,v in r.items() if k!='fit_ids'},'variance_evidence':details,
                            'scores':{k:m.score(*args[:2],r['target'],*args[2:]) for k,args in specs.items()}})
    comparisons={}
    for candidate,control in [('fuller_control','fixed_referee'),('fuller_poisson','fixed_referee'),
                              ('selected_mean_poisson','fuller_poisson'),('selected_raw','selected_mean_poisson'),
                              ('selected_calibrated','selected_raw'),('selected_calibrated','fixed_referee'),
                              ('selected_calibrated','fuller_control'),('lineup_scenario','fuller_control')]:
        comparisons[candidate+'__'+control]=m.compare(predictions,candidate,control)
    expanded=[]
    for f in features:
        if f['baseline'] or len(f['inputs']['source_fixture_ids'])<50 or not m.cards.utc('2023-04-01')<=m.cards.utc(f['kickoff'])<m.cards.END:continue
        pred=m.full_mean(f,{'id':'pool:16.0','pool':16.},engine)
        expanded.append({**{k:f[k] for k in ('fixture_id','competition','season','kickoff','target','stage','foul_state','input_id')},
                         'scores':{'pool16':m.score(pred['mean'],0.,f['target'])}})
    expanded_groups={league:[r for r in expanded if r['competition']==league] for league in sorted({r['competition'] for r in expanded})}
    passed=all(comparisons['selected_calibrated__'+c].get('development_gate_passed',False) for c in ('fixed_referee','fuller_control'))
    report={'version':m.VERSION,'status':'development_complete','recipe':recipe['spec'],'alpha':recipe['alpha'],
            'calibration':calibration,'fitting_support':recipe['support'],'evaluation_support':m.fixed.support(predictions),
            'comparisons':comparisons,'development_gates_passed':passed,'later_qualification':'not_run',
            'production_qualified':False,'fixed_reference_parity':len(saved),
            'expanded_coverage':{k:m.summary(v,'pool16') for k,v in expanded_groups.items()},
            'lineup_scenario_eligible_for_selection':False,'referee_aliases_merged':0,
            'limitations':['assumed_final','2023_previously_inspected','unknown_original_referee_assignment_time',
                           'lineup_scenario_not_timestamp_verified','aggression_and_league_regime_omitted',
                           'weighted_product_convention_not_bookmaker_equivalence']}
    # Store fitting memberships once per bundle, not thousands of duplicate lists.
    compact_nested=[{k:v for k,v in r.items() if k not in ('fit_ids','means')} for r in nested]
    bundles={f['recipe']['id']:f['recipe'] for f in folds if 'recipe' in f};bundles[recipe['id']]=recipe
    return {'features.jsonl':features,'candidate-means.jsonl':records,'nested-predictions.jsonl':compact_nested,
            'folds.json':folds,'fit-bundles.json':bundles,'predictions.jsonl':predictions,'expanded-predictions.jsonl':expanded,
            'report.json':report,'candidate-bundle.json':{'version':m.VERSION,'recipe':recipe,'calibration':calibration,
                  'target_policy':'spix-participation-settlement.v2','publication_enabled':False,
                  'qualification_required':True,'development_gates_passed':passed}}


def run(inputs,output):
    inputs,output=(Path(p).resolve() for p in (inputs,output))
    if output.exists():raise FileExistsError(output)
    verify_complete(inputs);verify_complete(FIXED)
    deps={p:importlib.metadata.version(p) for p in ('numpy','scipy')}
    sources=bridge.source_files()
    output.mkdir(parents=True,exist_ok=False)
    with offline_guard(root=ROOT,output=output):
        shutil.copytree(inputs,output/'inputs')
        shutil.copyfile(FIXED/'predictions.jsonl',output/'fixed-predictions.jsonl')
        for source in sources+[ROOT/PROTOCOL]:
            dest=output/'source'/source.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,dest)
        write_json(output/'manifest.json',{'version':m.VERSION,'source_hashes':{str(p.relative_to(ROOT)):sha(p) for p in sources},
                   'protocol_sha256':sha(ROOT/PROTOCOL),'input_seal_sha256':sha(inputs/'COMPLETE.json'),
                   'dependencies':deps,'seed':m.SEED,'python':sys.version,'registry':m.registry(),
                   'database_access':False,'api_calls':0,'production_changes':False})
        results=calculate(output)
        for name,value in results.items():
            if name.endswith('.jsonl'):write_rows(output/name,value)
            else:write_json(output/name,value)
        complete(output)
    return {'status':results['report.json']['status'],'development_gates_passed':results['report.json']['development_gates_passed'],
            'recipe':results['report.json']['recipe'],'alpha':results['report.json']['alpha'],
            'evaluation_support':results['report.json']['evaluation_support']}


def replay(output):
    output=Path(output).resolve(); verify_complete(output)
    manifest=read_json(output/'manifest.json')
    if any(sha(ROOT/p)!=h for p,h in manifest['source_hashes'].items()):raise ValueError('Source changed since run')
    with offline_guard(root=ROOT,output=output):
        result=calculate(output,replay=True)
        for name,value in result.items():
            if name.endswith('.jsonl'):
                h=hashlib.sha256()
                for line in bridge.lines_bytes(value):h.update(line)
                if h.hexdigest()!=sha(output/name):raise ValueError('Replay differs: '+name)
            elif value!=read_json(output/name):raise ValueError('Replay differs: '+name)
    return {'status':'exact_offline_replay','files':sorted(result)}


def main():
    parser=argparse.ArgumentParser(description=__doc__); sub=parser.add_subparsers(dest='action',required=True)
    p=sub.add_parser('prepare');p.add_argument('--output',type=Path,required=True)
    p=sub.add_parser('run');p.add_argument('--inputs',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p=sub.add_parser('replay');p.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    value=prepare(args.output) if args.action=='prepare' else run(args.inputs,args.output) if args.action=='run' else replay(args.output)
    print(json.dumps(value,indent=2))


if __name__=='__main__':main()
