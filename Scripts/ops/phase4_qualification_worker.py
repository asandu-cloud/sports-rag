"""Frozen-engine qualification forecasts and separately locked outcome scoring."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace


def run(source, prepared, action):
    if action not in ('predict','score'):raise ValueError('Unknown qualification operation')
    source, prepared = source.resolve(), prepared.resolve()
    # Validate locks before opening the operational inputs. Forecasting never
    # decodes targets; scoring requires a separately sealed prediction set.
    for lock in ('METHOD_LOCK.json','INPUT_LOCK.json') + (('PREDICTION_LOCK.json',) if action=='score' else ()):
        for name,expected in json.loads((prepared/lock).read_text()).items():
            p = prepared/name
            if not p.resolve().is_relative_to(prepared) or hashlib.sha256(p.read_bytes()).hexdigest()!=expected:
                raise ValueError('Qualification lock mismatch')
    output = prepared/('predictions' if action=='predict' else 'scores')
    output.mkdir(exist_ok=False)
    helper = Path(__file__).with_name('phase4_candidate_worker.py');ns={}
    exec(compile(helper.read_bytes(),str(helper),'exec'),ns);load=ns['load']
    import dotenv
    dotenv.load_dotenv=lambda *a,**k:False;dotenv.dotenv_values=lambda *a,**k:{}
    os.environ['PREDICTION_RELEASE_MODE']='shadow';sys.dont_write_bytecode=True
    sys.path[:0]=[str(source),str(source/'Scripts'),str(source/'Scripts/rag_ingest')]
    files=prepared/'implementation/Scripts/data_platform/features'
    paths={n:files/(f+'.py') for n,f in [
        ('phase4_candidate_math','phase4_candidates'),('phase4_candidate_adapter','phase4_candidate_adapter'),
        ('phase4_comparison_math','phase4_backtest'),('phase4_finalist_math','phase4_finalists'),
        ('phase4_calibration_math','phase4_calibration'),('phase4_qualification_math','phase4_qualification')]}
    allowed={'reproduction/snapshots.jsonl','reproduction/control.jsonl','reproduction/eligibility.jsonl',
             'strength-bundles.json','corner-calibrator.json'}
    if action=='score':allowed|={'targets.jsonl','predictions/forecasts.jsonl.gz'}
    violations=ns['guard'](source,output,set(paths.values())|{prepared/n for n in allowed})
    cm=load('phase4_candidate_math',paths['phase4_candidate_math'])
    adapter=load('phase4_candidate_adapter',paths['phase4_candidate_adapter'])
    scoring=load('phase4_comparison_math',paths['phase4_comparison_math'])
    fm=load('phase4_finalist_math',paths['phase4_finalist_math'])
    calibration=load('phase4_calibration_math',paths['phase4_calibration_math'])
    q=load('phase4_qualification_math',paths['phase4_qualification_math'])
    from core import projections,team_resolution,market_service,line_selection
    import prob_models
    for module in (projections,team_resolution,market_service,line_selection,prob_models):
        if not Path(module.__file__).resolve().is_relative_to(source):raise ValueError('Engine escaped archive')
    if any(projections._ml_blend_weight(m) for m in (*cm.MARKETS,'cards')):raise ValueError('Nonzero control ML weight')
    engine=SimpleNamespace(projections=projections,teams=team_resolution,markets=market_service,lines=line_selection,probability=prob_models)
    specs={s['id']:s for s in cm.registry()};counts=Counter();parity=0
    read=lambda n:[json.loads(line) for line in (prepared/n).open()]
    if action=='predict':
        snapshots=read('reproduction/snapshots.jsonl')
        controls={r['fixture_id']:r for r in read('reproduction/control.jsonl')}
        eligibility={r['fixture_id']:r for r in read('reproduction/eligibility.jsonl')}
        bundles=json.loads((prepared/'strength-bundles.json').read_text())
        calibrator=json.loads((prepared/'corner-calibrator.json').read_text())
        with gzip.open(output/'forecasts.jsonl.gz','wt') as stream:
            for i,snapshot in enumerate(snapshots,1):
                q.qualification_time(snapshot['as_of']);f=snapshot['fixture'];fid=f['fixture_id']
                allow=eligibility[fid]['markets']
                if not any(allow[m]['eligible'] for m in cm.MARKETS):continue
                scope=fm.scope(snapshot)
                # These two methods consume only saved aggregates. Reject any
                # unexpected fallback lookup instead of opening a live/history store.
                control=adapter.forecast(snapshot,None,specs['control'],engine,rates_only=True,end=cm.utc(q.END))
                expected=controls[fid]
                source_means={r['market']['group']:r['projection'] for r in expected['results']}
                if control['goal_means']!=expected['goal_means'] or any(
                    control['means'][m]!=source_means[m]['value'] for m in cm.MARKETS) or any(
                    control['variances'][m]!=source_means[m]['variance'] for m in ('corners','sot')):
                    raise ValueError('Frozen control numerical parity failed')
                parity+=1
                bundle=bundles.get(f['competition']) if scope['active'] else None
                if bundle:q.check_bundle(bundle)
                strength=adapter.forecast(snapshot,None,specs['strength_goals:10.0'],engine,
                    strength_bundle=bundle,rates_only=True,end=cm.utc(q.END)) if scope['active'] else control
                candidate=q.compose(snapshot,control,strength)
                missing='xg_partial_or_missing' if any(p.get('xg_home_pm') is None or p.get('xg_away_pm') is None
                    for p in snapshot['profiles'].values()) else 'xg_both_venues'
                meta={'fixture_id':fid,'snapshot_id':snapshot['snapshot_id'],'kickoff':f['kickoff'],
                    'week':q.week(f['kickoff']),'league':f['competition'],'season':str(f['season']),
                    'season_stage':scope['season_stage'],'forecast_stage':snapshot['forecast_stage'],
                    'missingness':missing,'active_scope':scope['active'],'publication_enabled':False}
                for name,rates in (('control',control),('supported_stack',candidate)):
                    probabilities={}
                    for market in cm.MARKETS:
                        if not allow[market]['eligible']:continue
                        if market=='goals':
                            probabilities[market]={'score_matrix':cm.goal_matrix(*rates['goal_means'],rho=-.1,frozen_probability=prob_models)}
                        else:
                            mu=rates['means'][market];alpha=max(0.,((rates['variances'][market] or mu)-mu)/mu**2)
                            probabilities[market]=cm.count_pmf(mu,alpha)
                        counts[name+':'+market]+=1
                    diagnostic=None
                    if name=='supported_stack' and scope['active'] and 'corners' in probabilities:
                        raw=sum(probabilities['corners']['pmf'][10:])
                        diagnostic={'raw_probability':raw,**q.calibrated_corner(calibrator,snapshot,raw),
                                    'role':'binary_diagnostic_only_not_count_PMF'}
                    stream.write(fm.encode({**meta,'configuration':name,'rates':rates,'probabilities':probabilities,
                        'goal_fit_id':bundle['id'] if name=='supported_stack' and bundle else None,
                        'corner_diagnostic':diagnostic,'cross_market_joint_probability':None})+'\n')
                if i%250==0:print(f'{i}/{len(snapshots)} qualification forecasts',flush=True)
    else:
        targets={r['fixture_id']:r for r in read('targets.jsonl')}
        with gzip.open(prepared/'predictions/forecasts.jsonl.gz','rt') as stream,gzip.open(output/'records.jsonl.gz','wt') as result:
            for forecast in map(json.loads,stream):
                q.qualification_time(forecast['kickoff']);target=targets[forecast['fixture_id']]
                meta={k:v for k,v in forecast.items() if k not in ('rates','probabilities','corner_diagnostic')}
                for market in forecast['probabilities']:
                    score=scoring.score_forecast(forecast['rates'],specs['control'],market,target,cm,prob_models)
                    result.write(fm.encode({**meta,'market':market,'scores':score,
                        'fallbacks':forecast['rates'].get('market_fallbacks',{}).get(market,[])})+'\n')
                    counts[forecast['configuration']+':'+market]+=1
                diagnostic=forecast['corner_diagnostic']
                if diagnostic:
                    y=target['labels']['corners']>9.5
                    result.write(fm.encode({**meta,'market':'corner_over9.5_binary',
                        'scores':calibration.binary_score(diagnostic['p'],y,diagnostic['logit']),
                        'raw_scores':calibration.binary_score(diagnostic['raw_probability'],y),
                        'calibrator_id':diagnostic['fit_id']})+'\n')
    if violations:raise ValueError('Forbidden worker IO')
    report={'version':q.VERSION,'action':action,'records':dict(counts),'control_parity':parity,
            'io_violations':violations,'target_store_decoded':action=='score','final_system_test_opened':False}
    (output/'report.json').write_text(fm.encode(report)+'\n');print(fm.encode(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','prepared'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--action',choices=('predict','score'),required=True)
    a=p.parse_args();run(a.source,a.prepared,a.action)
