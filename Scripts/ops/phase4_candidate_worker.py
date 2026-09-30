"""Isolated unscored candidate smoke execution against the archived control."""
import argparse
import json
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace


def guard(source, output, readable):
    violations=[]
    def audit(event,args):
        reason=None
        if event in ('socket.connect','socket.getaddrinfo','sqlite3.connect','subprocess.Popen','os.system'):
            reason=event
        if event=='open' and not isinstance(args[0],int):
            path=Path(os.fsdecode(args[0])).resolve();mode,flags=args[1:3]
            writing=(isinstance(mode,str) and any(c in mode for c in 'wax+')) or (
                isinstance(flags,int) and flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC))
            if writing and not path.is_relative_to(output): reason='outside_output:'+str(path)
            elif path.name=='.env' or path.suffix in ('.db','.sqlite','.sqlite3'): reason='live_store:'+str(path)
            elif any(p in path.parts for p in ('Index','Output','Research')):
                if path not in readable and not path.is_relative_to(source) and not path.is_relative_to(output):
                    reason='unapproved_research_read:'+str(path)
                elif 'Index' in path.parts and not path.is_relative_to(source/'Index/ml_models'):
                    reason='non_model_data:'+str(path)
        if reason:
            violations.append(reason);raise RuntimeError(reason)
    sys.addaudithook(audit)
    return violations


def load(name,path):
    module=ModuleType(name);sys.modules[name]=module
    exec(compile(path.read_bytes(),str(path),'exec'),module.__dict__)
    return module


def run(source,prepared,output):
    source,prepared,output=[p.resolve() for p in (source,prepared,output)]
    output.mkdir(exist_ok=False)
    history=[json.loads(line) for line in (prepared/'history.jsonl').open()]
    snapshots=[json.loads(line) for line in (prepared/'snapshots.jsonl').open()]
    controls={r['fixture_id']:r for r in map(json.loads,(prepared/'control.jsonl').open())}
    bundles=json.loads((prepared/'strength-bundles.json').read_text()) if (prepared/'strength-bundles.json').exists() else {}
    lineups=json.loads((prepared/'lineups.json').read_text()) if (prepared/'lineups.json').exists() else {}
    eligibility={r['fixture_id']:r['markets'] for r in map(json.loads,(prepared/'eligibility.jsonl').open())} if (prepared/'eligibility.jsonl').exists() else {}
    import dotenv
    dotenv.load_dotenv=lambda *a,**k:False;dotenv.dotenv_values=lambda *a,**k:{}
    os.environ['PREDICTION_RELEASE_MODE']='shadow';sys.dont_write_bytecode=True
    sys.path[:0]=[str(source),str(source/'Scripts'),str(source/'Scripts/rag_ingest')]
    implementation=Path(__file__).parents[2]
    paths={name:implementation/'Scripts/data_platform/features'/file for name,file in (
        ('phase4_candidate_math','phase4_candidates.py'),('phase4_candidate_adapter','phase4_candidate_adapter.py'),
        ('phase4_history_inputs','phase4_history.py'))}
    violations=guard(source,output,set(paths.values()))
    cm=load('phase4_candidate_math',paths['phase4_candidate_math'])
    adapter=load('phase4_candidate_adapter',paths['phase4_candidate_adapter'])
    historical=load('phase4_history_inputs',paths['phase4_history_inputs'])
    from core import projections,team_resolution,market_service,line_selection
    import prob_models
    for module in (projections,team_resolution,market_service,line_selection,prob_models):
        if not Path(module.__file__).resolve().is_relative_to(source): raise ValueError('Engine import escaped frozen source')
    if any(projections._ml_blend_weight(m) for m in (*cm.MARKETS,'cards')): raise ValueError('Nonzero ML control contribution')
    engine=SimpleNamespace(projections=projections,teams=team_resolution,markets=market_service,lines=line_selection,probability=prob_models)
    index=historical.HistoricalInputs(history);before_history=cm.digest(history)
    specs=cm.registry();counts={};failures=[];parity=0
    with (output/'forecasts.jsonl').open('x') as stream:
        for snapshot in snapshots:
            original=cm.canonical(snapshot)
            for spec in specs:
                try:
                    key=snapshot['fixture']['competition']+':'+spec['family'].removeprefix('strength_')+':'+str(spec['parameter'])
                    result=adapter.forecast(snapshot,index,spec,engine,strength_bundle=bundles.get(key),
                                           lineup=lineups.get(str(snapshot['fixture']['fixture_id'])))
                    if snapshot['fixture']['fixture_id'] in eligibility:
                        result['comparison_eligibility']=eligibility[snapshot['fixture']['fixture_id']]
                        result['id']=cm.digest({k:v for k,v in result.items() if k!='id'})
                    check(result)
                    if spec['family']=='control':
                        reference=controls[snapshot['fixture']['fixture_id']]
                        if result['projection_path_records']!=reference['results'] or result['score_distribution']!=reference['score_distribution']:
                            raise RuntimeError('Frozen control parity failure')
                        parity+=1
                    counts[result['status']]=counts.get(result['status'],0)+1
                except ValueError as exc:
                    result={'candidate':spec,'fixture_id':snapshot['fixture']['fixture_id'],'status':'invalid','reason':str(exc)}
                    failures.append(result)
                stream.write(cm.canonical(result)+'\n')
            repeated=adapter.forecast(snapshot,index,specs[0],engine)
            if repeated['projection_path_records']!=controls[snapshot['fixture']['fixture_id']]['results']:
                raise RuntimeError('Candidate patches leaked into subsequent control')
            if cm.canonical(snapshot)!=original: raise RuntimeError('Candidate mutated input snapshot')
            print('Unscored candidate smoke fixture '+str(snapshot['fixture']['fixture_id']),flush=True)
    if violations or before_history!=cm.digest(history): raise RuntimeError('Forbidden IO or history mutation')
    report={'version':cm.VERSION,'purpose':'implementation_smoke_not_model_comparison',
            'fixtures':len(snapshots),'registered_variants':len(specs),'control_parity':parity,
            'statuses':counts,'invalid':failures,'io_violations':violations,
            'historical_parameters_fitted':False,'scores_computed':False,'publication_enabled':False,
            'missing_lineup':'explicit control fallback','missing_strength_bundles':'fit interface tested synthetically; historical fitting not started'}
    (output/'report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')


def check(result):
    import math
    score=result['score_distribution']
    if score and (any(not math.isfinite(v) or v<0 for _,_,v in score) or abs(sum(v for _,_,v in score)-1)>1e-10):
        raise ValueError('Invalid joint probability')
    for market,d in result['distributions'].items():
        if any(not math.isfinite(v) or v<0 for v in d['pmf']) or abs(sum(d['pmf'])-1)>1e-10:
            raise ValueError('Invalid count probability')
        for outcomes in result['diagnostics'][market].values():
            for profile in outcomes.values():
                if min(profile.values())<0 or abs(sum(profile.values())-1)>1e-10:
                    raise ValueError('Invalid Asian settlement probabilities')
    if result['publication_enabled']: raise ValueError('Public candidate output refused')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('source','prepared','output'): parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();run(args.source,args.prepared,args.output)
