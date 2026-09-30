"""Execute fixed composite candidates on the saved 2023 development snapshots."""
import argparse
from collections import Counter
import gzip
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace


def run(source,prepared,key):
    if key not in ('2023-Q1','2023-Q2','2023-Q3','2023-Q4'):raise ValueError('Only inspected 2023 development')
    source=source.resolve();prepared=prepared.resolve();folder=prepared/'folds'/key;output=prepared/'runs'/key
    output.mkdir(parents=True,exist_ok=False)
    helper=Path(__file__).with_name('phase4_candidate_worker.py');ns={}
    exec(compile(helper.read_bytes(),str(helper),'exec'),ns);load=ns['load']
    import dotenv
    dotenv.load_dotenv=lambda *a,**k:False;dotenv.dotenv_values=lambda *a,**k:{}
    os.environ['PREDICTION_RELEASE_MODE']='shadow';sys.dont_write_bytecode=True
    sys.path[:0]=[str(source),str(source/'Scripts'),str(source/'Scripts/rag_ingest')]
    files=prepared/'implementation/Scripts/data_platform/features'
    paths={n:files/(f+'.py') for n,f in [('phase4_candidate_math','phase4_candidates'),('phase4_candidate_adapter','phase4_candidate_adapter'),
        ('phase4_history_inputs','phase4_history'),('phase4_comparison_math','phase4_backtest'),('phase4_finalist_math','phase4_finalists')]}
    readable=set(paths.values())|{prepared/'history.jsonl'}|set(folder.glob('*.jsonl'))|set((folder/'bundles').glob('*.json'))
    violations=ns['guard'](source,output,readable)
    cm=load('phase4_candidate_math',paths['phase4_candidate_math']);adapter=load('phase4_candidate_adapter',paths['phase4_candidate_adapter'])
    history=load('phase4_history_inputs',paths['phase4_history_inputs']);score=load('phase4_comparison_math',paths['phase4_comparison_math'])
    fm=load('phase4_finalist_math',paths['phase4_finalist_math'])
    from core import projections,team_resolution,market_service,line_selection
    import prob_models
    for module in (projections,team_resolution,market_service,line_selection,prob_models):
        if not Path(module.__file__).resolve().is_relative_to(source):raise ValueError('Non-frozen engine import')
    if any(projections._ml_blend_weight(m) for m in (*cm.MARKETS,'cards')):raise ValueError('Nonzero control ML weight')
    engine=SimpleNamespace(projections=projections,teams=team_resolution,markets=market_service,lines=line_selection,probability=prob_models)
    read=lambda name:[json.loads(l) for l in (folder/name).open()]
    snapshots=read('snapshots.jsonl');targets={r['fixture_id']:r for r in read('targets.jsonl')}
    eligibility={r['fixture_id']:r['markets'] for r in read('eligibility.jsonl')}
    controls={r['fixture_id']:r for r in read('control-rates.jsonl')}
    refs={(r['fixture_id'],r['market'],r['candidate_id']):r for r in read('reference.jsonl')}
    bundles={p.stem:json.loads(p.read_text()) for p in (folder/'bundles').glob('*.json')}
    index=history.HistoricalInputs([json.loads(l) for l in (prepared/'history.jsonl').open()])
    specs={s['id']:s for s in cm.registry()};counts=Counter();maximum=0.;parity=0
    def difference(a,b):
        if isinstance(a,dict):
            if a.keys()!=b.keys():raise ValueError('Score schema mismatch')
            return max((difference(a[k],b[k]) for k in a),default=0.)
        if isinstance(a,list):
            if len(a)!=len(b):raise ValueError('Score shape mismatch')
            return max((difference(x,y) for x,y in zip(a,b)),default=0.)
        if isinstance(a,(int,float)) and isinstance(b,(int,float)):return abs(a-b)
        if a!=b:raise ValueError('Score identity mismatch')
        return 0.
    with gzip.open(output/'forecasts.jsonl.gz','wt') as stream,gzip.open(output/'fixture-configurations.jsonl.gz','wt') as configs:
        for i,snapshot in enumerate(snapshots,1):
            fid=snapshot['fixture']['fixture_id'];allow=eligibility[fid]
            if not any(allow[m]['eligible'] for m in cm.MARKETS):continue
            control=adapter.forecast(snapshot,index,specs['control'],engine,rates_only=True)
            if any(control[k]!=controls[fid][k] for k in ('means','goal_means','variances')):raise ValueError('Control rate parity failed')
            goal_ref=refs.get((fid,'goals','strength_goals:10.0'));bid=goal_ref['fit_id'] if goal_ref else None
            bundle=bundles[bid] if bid else None
            strength=adapter.forecast(snapshot,index,specs['strength_goals:10.0'],engine,strength_bundle=bundle,rates_only=True)
            if parity==0:
                full=adapter.forecast(snapshot,index,specs['strength_goals:10.0'],engine,strength_bundle=bundle)
                if full['means']!=strength['means'] or full['goal_means']!=strength['goal_means']:raise ValueError('Full/rates parity failed')
            parity+=1
            for name in ('control','component_stack','supported_stack'):
                rates=control if name=='control' else fm.compose(snapshot,control,strength,name)
                config_id=fm.digest({'snapshot_id':snapshot['snapshot_id'],'configuration':name,'version':fm.VERSION})
                configs.write(fm.encode({'id':config_id,'fixture_id':fid,'snapshot_id':snapshot['snapshot_id'],
                    'configuration':name,'rates':rates,'goal_fit_id':fm.goal_fit_id(name,fm.scope(snapshot)['active'],bid),'cross_market_joint_probability':None})+'\n')
                for market in cm.MARKETS:
                    if not allow[market]['eligible']:continue
                    baseline=refs[(fid,market,'control')]
                    record={k:v for k,v in baseline.items() if k not in ('scores','family','candidate_id','fit_id','status','fallbacks')}
                    record.update(configuration=name,candidate_id=name,family=name,configuration_id=config_id,
                        active_scope=fm.scope(snapshot)['active'],fit_id=fm.goal_fit_id(name,fm.scope(snapshot)['active'],bid) if market=='goals' else None,
                        scores=score.score_forecast(rates,specs['control'],market,targets[fid],cm,prob_models),
                        fallbacks=rates.get('market_fallbacks',{}).get(market,rates['evidence']['fallbacks']),publication_enabled=False)
                    active=name=='component_stack' or (name=='supported_stack' and record['active_scope'])
                    component={'goals':'strength_goals:10.0','corners':'dispersion_corners:0.025','sot':'dispersion_sot:0.01'}[market] if active else 'control'
                    delta=difference(record['scores'],refs[(fid,market,component)]['scores']);maximum=max(maximum,delta)
                    if delta>1e-10:raise ValueError('Composed prediction differs from its saved component')
                    stream.write(fm.encode(record)+'\n');counts[name+':'+market]+=1
            if i%100==0:print(json.dumps({'fold':key,'processed':i,'fixtures':len(snapshots)}),flush=True)
    if violations:raise ValueError('Forbidden IO')
    report={'version':fm.VERSION,'fold':key,'control_replays':parity,'records':dict(counts),'maximum_component_score_difference':maximum,
        'io_violations':violations,'reserved_outcomes_read':False,'publication_enabled':False}
    (output/'report.json').write_text(fm.encode(report)+'\n');print(fm.encode(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','prepared'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--fold',required=True);a=p.parse_args();run(a.source,a.prepared,a.fold)
