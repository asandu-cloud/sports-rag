"""Isolated quarterly forecast/fitting worker, then development outcome scoring."""
import argparse
from collections import defaultdict
from datetime import datetime
from functools import lru_cache
import gzip
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace


def run(source,prepared,key,limit=None,earlier_calibration=False):
    if earlier_calibration and key not in {f'{y}-Q{q}' for y in range(2019,2022) for q in range(1,5)}:
        raise ValueError('Earlier calibration requires a 2019–2021 quarter')
    helper=Path(__file__).with_name('phase4_candidate_worker.py')
    namespace={'__name__':'captured_worker_helpers'}
    exec(compile(helper.read_bytes(),str(helper),'exec'),namespace)
    guard,load=namespace['guard'],namespace['load']
    source,prepared=source.resolve(),prepared.resolve()
    folder=prepared/'folds'/key;output=prepared/'runs'/key
    output.mkdir(parents=True,exist_ok=False)
    paths={name:prepared/'implementation/Scripts/data_platform/features'/file for name,file in (
        ('phase4_candidate_math','phase4_candidates.py'),('phase4_candidate_adapter','phase4_candidate_adapter.py'),
        ('phase4_history_inputs','phase4_history.py'),('phase4_comparison_math','phase4_backtest.py'))}
    import dotenv
    dotenv.load_dotenv=lambda *a,**k:False;dotenv.dotenv_values=lambda *a,**k:{}
    os.environ['PREDICTION_RELEASE_MODE']='shadow';sys.dont_write_bytecode=True
    sys.path[:0]=[str(source),str(source/'Scripts'),str(source/'Scripts/rag_ingest')]
    readable=set(paths.values())|{prepared/'history.jsonl',prepared/'fit-eligibility.json',prepared/'selection.json'}
    readable|=set(folder.glob('*.jsonl'))
    violations=guard(source,output,readable)
    cm=load('phase4_candidate_math',paths['phase4_candidate_math'])
    adapter=load('phase4_candidate_adapter',paths['phase4_candidate_adapter'])
    historical=load('phase4_history_inputs',paths['phase4_history_inputs'])
    scoring=load('phase4_comparison_math',paths['phase4_comparison_math'])
    from core import projections,team_resolution,market_service,line_selection
    import prob_models
    for module in (projections,team_resolution,market_service,line_selection,prob_models):
        if not Path(module.__file__).resolve().is_relative_to(source):raise ValueError('Non-frozen engine import')
    if any(projections._ml_blend_weight(m) for m in (*cm.MARKETS,'cards')):raise ValueError('Nonzero control ML weight')
    engine=SimpleNamespace(projections=projections,teams=team_resolution,markets=market_service,lines=line_selection,probability=prob_models)
    history=[json.loads(l) for l in (prepared/'history.jsonl').open()]
    snapshots=[json.loads(l) for l in (folder/'snapshots.jsonl').open()]
    if limit is not None:snapshots=snapshots[:limit]
    targets={r['fixture_id']:r for r in map(json.loads,(folder/'targets.jsonl').open())}
    eligibility={r['fixture_id']:r['markets'] for r in map(json.loads,(folder/'eligibility.jsonl').open())}
    controls={r['fixture_id']:r for r in map(json.loads,(folder/'control-rates.jsonl').open())}
    fit_ids=json.loads((prepared/'fit-eligibility.json').read_text())
    index=historical.HistoricalInputs(history)
    # Input-only caches; reset at each fixture. Cache outputs are never mutated.
    original_before=index.before
    index.before=lru_cache(maxsize=512)(original_before)
    original_league=adapter.league_summary
    byleague=defaultdict(dict)
    for r in history:byleague[r['competition']][r['fixture_id']]=r
    @lru_cache(maxsize=512)
    def league_cached(league,cutoff,field,current,prior,weight):
        return original_league(SimpleNamespace(history=byleague[league]),league,cutoff,field,
            {'current_rank':current,'prior_rank':prior,'prior_weight':weight})
    adapter.league_summary=lambda _index,league,cutoff,field,s:league_cached(league,cutoff,field,s['current_rank'],s['prior_rank'],s['prior_weight'])
    specs=[s for s in cm.registry() if s['family']!='lineup']
    if earlier_calibration:
        specs=[s for s in specs if s['family'] in ('control','strength_goals','dispersion_corners','dispersion_sot')]
    selected=None
    if key.startswith('2023'):
        selected=json.loads((prepared/'selection.json').read_text())['selected']
        chosen=set(selected.values())|{'control'}
        specs=[s for s in specs if s['id'] in chosen]
    cutoff=(f'{key[:4]}-{1+(int(key[-1])-1)*3:02d}-01T00:00:00+00:00' if earlier_calibration else scoring.quarter_cutoff(key))
    if earlier_calibration and any(int(r['kickoff'][:4])>=2022 for r in history):
        raise ValueError('Later history in earlier forecast experiment')
    if earlier_calibration and any(f"{s['fixture']['kickoff'][:4]}-Q{(int(s['fixture']['kickoff'][5:7])-1)//3+1}"!=key for s in snapshots):
        raise ValueError('Snapshot outside earlier quarter')
    bundles={};fit_records=[];failures=[];parity=0;total=0
    start=time.monotonic()
    with gzip.open(output/'forecasts.jsonl.gz','wt') as forecasts, gzip.open(output/'rate-evidence.jsonl.gz','wt') as evidencefile:
        for n,snapshot in enumerate(snapshots,1):
            f=snapshot['fixture'];fid=f['fixture_id'];allow=eligibility[fid]
            index.before.cache_clear();league_cached.cache_clear()
            if not any(allow[m]['eligible'] for m in cm.MARKETS):continue
            reference=adapter.forecast(snapshot,index,cm.registry()[0],engine,rates_only=True)
            expected=controls[fid]
            if reference['means']!=expected['means'] or reference['goal_means']!=expected['goal_means'] or reference['variances']!=expected['variances']:
                raise ValueError('Frozen numerical parity failed: '+str(fid))
            parity+=1
            # Verify complete candidate path against fast numerical path before
            # any scores: one fixture per fold, all active variants without fits.
            min_matches=min(q.get('current_season_matches',0) for q in snapshot['profile_quality'].values())
            stage='0-7' if min_matches<8 else '8-15' if min_matches<16 else '16+'
            missing='xg_partial_or_missing' if any(p.get('xg_home_pm') is None or p.get('xg_away_pm') is None for p in snapshot['profiles'].values()) else 'xg_both_venues'
            for spec in specs:
                markets=[m for m in spec['markets'] if allow[m]['eligible'] and
                    (selected is None or spec['family']=='control' or selected.get(m+':'+spec['family'])==spec['id'])]
                if not markets:continue
                family=spec['family'];bundle=None;error=None
                if family.startswith('strength_'):
                    m=family.removeprefix('strength_');bkey=(f['competition'],f['season'],m,spec['parameter'])
                    if bkey not in bundles:
                        try:
                            b=cm.fit_strength(history,eligible_ids=fit_ids[m],competition=f['competition'],market=m,
                                    cutoff=cutoff,season=f['season'],ridge=spec['parameter'])
                            bundles[bkey]=b
                            (output/(b['id']+'.bundle.json')).write_text(cm.canonical(b)+'\n')
                            fit_records.append({'key':list(bkey),'id':b['id'],'n':b['fixture_count'],'ess':b['effective_fixture_count'],'cutoff':cutoff})
                        except ValueError as exc:
                            bundles[bkey]={'error':str(exc)};fit_records.append({'key':list(bkey),'error':str(exc),'cutoff':cutoff})
                    entry=bundles[bkey]
                    if 'error' in entry:
                        if entry['error']!='Insufficient strength fitting cohort':error=entry['error']
                    else:bundle=entry
                try:
                    if error:raise ValueError(error)
                    if family=='control' or family.startswith('dispersion_') or family in ('goal_nb','goal_rho'):
                        rates=reference
                    else:rates=adapter.forecast(snapshot,index,spec,engine,strength_bundle=bundle,rates_only=True)
                    if parity==1:
                        full=adapter.forecast(snapshot,index,spec,engine,strength_bundle=bundle)
                        if full['means']!=rates['means'] or full['goal_means']!=rates['goal_means']:
                            raise ValueError('Fast/full mean mismatch')
                        for m in markets:
                            if m!='goals' and not family.startswith('dispersion_') and full['distributions'][m]['variance']!=rates['variances'][m]:
                                raise ValueError('Fast/full variance mismatch')
                    evidencefile.write(cm.canonical({'fixture_id':fid,'candidate_id':spec['id'],'rates':rates})+'\n')
                except ValueError as exc:error=str(exc)
                for market in markets:
                    row={'fixture_id':fid,'snapshot_id':snapshot['snapshot_id'],'candidate_id':spec['id'],'family':family,'market':market,
                         'year':int(key[:4]),'fold':key,'kickoff':f['kickoff'],'week':'%d-W%02d'%datetime.fromisoformat(f['kickoff']).isocalendar()[:2],
                         'league':f['competition'],'season':str(f['season']),'season_stage':stage,
                         'forecast_stage':snapshot['forecast_stage'],'missingness':missing,'fit_id':bundle['id'] if bundle else None}
                    try:
                        if error:raise ValueError(error)
                        row['scores']=scoring.score_forecast(rates,spec,market,targets[fid],cm,prob_models)
                        row['status']=rates['status'];row['fallbacks']=rates['evidence']['fallbacks']
                    except ValueError as exc:
                        row.update(status='invalid',reason=str(exc));failures.append({'fixture_id':fid,'candidate':spec['id'],'market':market,'reason':str(exc)})
                    forecasts.write(cm.canonical(row)+'\n');total+=1
            if n%50==0:print(json.dumps({'fold':key,'processed':n,'total_fixtures':len(snapshots),'records':total,'seconds':round(time.monotonic()-start,1)}),flush=True)
    if violations:raise ValueError('IO guard violation')
    report={'version':scoring.VERSION,'fold':key,'fixtures':len(snapshots),'eligible_controls_replayed':parity,
            'records':total,'fits':fit_records,'invalid':failures,'io_violations':violations,
            'seconds':time.monotonic()-start,'publication_enabled':False,'reserve_labels_read':False}
    (output/'report.json').write_text(cm.canonical(report)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('fits','invalid')}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for field in ('source','prepared'):parser.add_argument('--'+field,type=Path,required=True)
    parser.add_argument('--fold',required=True);parser.add_argument('--limit',type=int)
    parser.add_argument('--earlier-calibration',action='store_true')
    args=parser.parse_args();run(args.source,args.prepared,args.fold,args.limit,args.earlier_calibration)
