"""Candidate chronology, fitted strength recovery, evidence gates and frozen wiring."""
from copy import deepcopy
from datetime import timedelta
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from Scripts.data_platform.features import phase4_candidates as c

ROOT=Path(__file__).resolve().parents[2]
FROZEN=ROOT/'Research/phase4-baseline-2026-09-28/control/workspace'
spec=importlib.util.spec_from_file_location('test_frozen_probability',FROZEN/'Scripts/rag_ingest/prob_models.py')
prob=importlib.util.module_from_spec(spec);spec.loader.exec_module(prob)


def history():
    rng=np.random.default_rng(20260929);rows=[]
    attack=[.7,.2,-.3,-.6];defence=[-.4,.1,.2,.1]
    for i in range(210):
        home=i%4;away=(home+1+(i//4)%3)%4
        kickoff=c.utc('2021-08-01T15:00:00Z')+timedelta(days=i*2)
        row={'fixture_id':i+1,'competition':'EPL','season':2021 if kickoff.month<7 or kickoff.year==2021 else 2022,
             'kickoff':kickoff.isoformat(),'home_team_id':11+home,'away_team_id':11+away,
             'status':'FT','observed_at':'2026-09-25T00:00:00Z'}
        for side,team,opp in [('home',home,away),('away',away,home)]:
            rate=math.exp(.3+.15*(side=='home')+attack[team]+defence[opp])
            row[side]={'goals':int(rng.poisson(rate)),'xg':rate if side=='away' and i%4 else None,
                'corners':int(rng.poisson(4+team)),'sot':int(rng.poisson(3+team)),
                'shots':12.,'fouls':10.,'possession':50.,'cards':None}
        rows.append(row)
    return rows


def fit(rows=None,**kwargs):
    rows=history() if rows is None else rows
    args=dict(eligible_ids=[r['fixture_id'] for r in rows],competition='EPL',market='goals',
              cutoff='2022-09-01T00:00:00Z',season=2022,ridge=10.)
    args.update(kwargs)
    return c.fit_strength(rows,**args)


def fixture():
    return {'fixture_id':999999,'competition':'EPL','season':2022,'home_team_id':11,'away_team_id':14,
            'kickoff':'2022-09-01T15:00:00Z'}


def test_registry_unique_closed_and_one_component_variants():
    specs=c.registry()
    assert len({s['id'] for s in specs})==len(specs)
    assert {s['family'] for s in specs} >= {'control','league_pool','venue_pool','defensive_xg','goal_nb','goal_rho','lineup'}
    for s in specs: c.validate_spec(s)
    bad=deepcopy(specs[1]);bad['parameter']=999
    with pytest.raises(ValueError):c.validate_spec(bad)


def test_weighted_missingness_support_and_pooling():
    assert c.weighted_summary([None,0.,2.],[100.,1.,1.])=={'mean':1.,'ess':2.,'n':2}
    assert c.weighted_summary([None])['mean'] is None
    assert c.pool(4.,0.,2.,8.)==2.
    assert c.pool(4.,1000.,2.,8.)>3.9
    a=c.weighted_summary([1.,2.]);b=c.weighted_summary([1.]*8)
    assert c.mixed_support(a,b,.5)==pytest.approx(6.4)


def test_future_values_do_not_enter_fits_and_bundle_is_replayable():
    rows=history();baseline=fit(rows)
    changed=deepcopy(rows)
    for r in changed:
        if c.utc(r['kickoff'])+timedelta(hours=3)>=c.utc(baseline['cutoff']):
            r['home']['goals']=float('nan');r['away']['goals']=999
    repeated=fit(list(reversed(changed)))
    assert baseline==repeated
    assert baseline['teams']['11']['attack']>baseline['teams']['14']['attack']
    assert abs(sum(t['attack'] for t in baseline['teams'].values()))<1e-10
    restored=json.loads(c.canonical(baseline))
    assert c.predict_strength(restored,fixture(),fixture()['kickoff'])==c.predict_strength(baseline,fixture(),fixture()['kickoff'])
    assert c.predict_strength(baseline,fixture(),fixture()['kickoff'])['means'][0]>c.predict_strength(baseline,fixture(),fixture()['kickoff'])['means'][1]


def test_other_league_targets_cannot_affect_fitted_strengths():
    rows=history();other=deepcopy(rows)
    for r in other:r.update(competition='LaLiga',fixture_id=r['fixture_id']+1000)
    assert fit(rows)==fit(rows+other)


@pytest.mark.parametrize('change', ['future','league','season','tamper','unknown'])
def test_strength_inference_validates_identity_and_cutoff(change):
    bundle=fit();f=fixture();as_of=f['kickoff']
    if change=='future':as_of='2021-01-01T00:00:00Z'
    if change=='league':f['competition']='LaLiga'
    if change=='season':f['season']=2023
    if change=='tamper':bundle['intercept']+=1
    if change=='unknown':
        f['home_team_id']=100000
        assert c.predict_strength(bundle,f,as_of)['status']=='control_fallback'
    else:
        with pytest.raises(ValueError):c.predict_strength(bundle,f,as_of)


def test_fit_does_not_promote_nonconverged_or_insufficient_models():
    with pytest.raises(ValueError,match='converge'):fit(maxiter=0)
    with pytest.raises(ValueError,match='Insufficient'):fit(history()[:10])
    with pytest.raises(ValueError,match='Reserved'):fit(cutoff='2025-01-01T00:00:00Z')
    rows=history();rows[0]['home']['goals']=1.5
    with pytest.raises(ValueError,match='counts'):fit(rows)


@pytest.mark.parametrize('mean,alpha',[(0.,0.),(0.,.8),(2.,0.),(2.,.4),(12.,1.6)])
def test_count_probability_tails_and_moments(mean,alpha):
    d=c.count_pmf(mean,alpha)
    assert min(d['pmf'])>=0 and abs(sum(d['pmf'])-1)<1e-10
    assert sum(i*p for i,p in enumerate(d['pmf']))==pytest.approx(mean,abs=1e-8)
    assert sum((i-mean)**2*p for i,p in enumerate(d['pmf']))==pytest.approx(mean+alpha*mean*mean,abs=1e-5)


def test_goal_alternatives_are_coherent_and_invalid_rho_rejected():
    for alpha in (0.,.1,.8):
        matrix=np.array(c.goal_matrix(1.8,1.1,alpha=alpha,rho=0.,frozen_probability=prob))
        assert matrix.sum()==pytest.approx(1.,abs=1e-10)
        assert (matrix*np.arange(matrix.shape[0])[:,None]).sum()==pytest.approx(1.8,abs=1e-8)
        assert (matrix*np.arange(matrix.shape[1])[None,:]).sum()==pytest.approx(1.1,abs=1e-8)
    with pytest.raises(ValueError):c.goal_matrix(2.,1.,alpha=.1,rho=-.1,frozen_probability=prob)
    with pytest.raises(ValueError):c.goal_matrix(10.,10.,rho=.05,frozen_probability=prob)
    with pytest.raises(ValueError):c.count_pmf(500.,1.6)


def fitting_probabilities():
    rows=[]
    for i in range(500):
        time=c.utc('2022-01-01T12:00:00Z')+timedelta(days=i//2)
        rows.append({'fixture_id':i+1,'kickoff':time.isoformat(),'as_of':time.isoformat(),
            'label_available_at':(time+timedelta(hours=3)).isoformat(),'mean':2.5,'target':2,
            'goal_means':[1.4,1.1],'team_target':[1,1]})
    return rows


@pytest.mark.parametrize('family',['dispersion','goal_rho','goal_nb'])
def test_probability_fits_only_earlier_complete_common_rows(family):
    rows=fitting_probabilities()
    market='corners' if family=='dispersion' else 'goals'
    for r in rows:r.update(market=market,period='regulation_time')
    result=c.fit_probability_parameter(rows,cutoff='2022-12-01T00:00:00Z',family=family,market=market,frozen_probability=prob)
    assert result['n']==500 and len(result['fixture_ids'])==500
    name='dispersion_'+market if family=='dispersion' else family
    spec=next(s for s in c.registry() if s['family']==name and s['parameter']==result['parameter'])
    c.validate_probability_bundle(result,spec,'2023-01-01T00:00:00Z')
    with pytest.raises(ValueError,match='future'):c.validate_probability_bundle(result,spec,'2022-01-01T00:00:00Z')
    with pytest.raises(ValueError,match='identity'):
        c.validate_probability_bundle(dict(result,parameter=99.),spec,'2023-01-01T00:00:00Z')
    rows[-1]['label_available_at']='2023-01-01T00:00:00Z'
    with pytest.raises(ValueError,match='unavailable'):c.fit_probability_parameter(rows,cutoff='2022-12-01T00:00:00Z',family=family,market=market,frozen_probability=prob)
    rows[-1].update(period='extra_time',label_available_at=rows[0]['label_available_at'])
    with pytest.raises(ValueError,match='identity'):c.fit_probability_parameter(rows,cutoff='2022-12-01T00:00:00Z',family=family,market=market,frozen_probability=prob)


def lineup():
    evidence={'fixture_id':fixture()['fixture_id'],'minutes_kind':'expected','stage':'confirmed','source_id':'synthetic-lineup',
        'observed_at':'2022-09-01T14:00:00Z','rates_cutoff':'2022-08-01T00:00:00Z',
        'coefficient_cutoff':'2022-08-01T00:00:00Z','coefficient_id':'synthetic-coefficients',
        'coefficients':{'goals':.5,'sot':.5}}
    for side,team,offset in [('home',11,0),('away',14,100)]:
        evidence[side]={'team_id':team,'players':[{'player_id':offset+i+1,'reference_minutes':90. if i<11 else 0.,
            'forecast_minutes':90. if i>0 else 0.,'goals_per90':.6 if i==0 else .1,'sot_per90':1. if i==0 else .5}
            for i in range(12)]}
    return evidence


def test_lineup_is_incremental_and_missing_data_stays_unavailable():
    result=c.lineup_delta(lineup(),fixture(),fixture()['kickoff'],1.)
    assert result['delta']['home']['goals']==pytest.approx(-.25)
    assert c.lineup_delta(None,fixture(),fixture()['kickoff'],1.)['status']=='control_fallback'
    same=lineup()
    for side in ('home','away'):
        for p in same[side]['players']:p['forecast_minutes']=p['reference_minutes']
    assert c.lineup_delta(same,fixture(),fixture()['kickoff'],1.)['delta']['home']['goals']==0


@pytest.mark.parametrize('bad',['time','actual','minutes','team','duplicate'])
def test_lineup_rejects_unqualified_evidence(bad):
    data=lineup()
    if bad=='time':data['rates_cutoff']='2022-09-02T00:00:00Z'
    if bad=='actual':data['minutes_kind']='actual'
    if bad=='minutes':data['home']['players'][0]['forecast_minutes']=90
    if bad=='team':data['home']['team_id']=14
    if bad=='duplicate':data['home']['players'][1]['player_id']=1
    with pytest.raises(ValueError):c.lineup_delta(data,fixture(),fixture()['kickoff'],1.)


@pytest.fixture(scope='module')
def wired(tmp_path_factory):
    path=tmp_path_factory.mktemp('phase4-candidates');rows=history();f=fixture()
    request={'fixture':f,'as_of':f['kickoff'],'forecast_stage':'reconstructed_immediately_before_kickoff',
        'availability':'assumed_final','source_snapshot_id':'synthetic-candidate-fixture',
        'source_eligibility':{m:{'eligible':m!='cards','reasons':[] if m!='cards' else ['cards_target_not_qualified']}
                              for m in ('goals','corners','sot','cards')}}
    for name,records in [('history',rows),('requests',[request])]:
        (path/(name+'.jsonl')).write_text(''.join(c.canonical(r)+'\n' for r in records))
    env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'}
    process=subprocess.run([sys.executable,'-B',str(ROOT/'Scripts/ops/phase4_historical_worker.py'),
        '--source',str(FROZEN),'--prepared',str(path),'--output',str(path/'control')],capture_output=True,text=True,timeout=120,env=env)
    assert process.returncode==0,process.stderr
    for name in ('snapshots.jsonl','control.jsonl'):(path/name).write_bytes((path/'control'/name).read_bytes())
    bundles={}
    for market in c.MARKETS:
        for ridge in (1.,10.,100.):bundles['EPL:'+market+':'+str(ridge)]=fit(rows,market=market,ridge=ridge)
    (path/'strength-bundles.json').write_text(c.canonical(bundles))
    (path/'lineups.json').write_text(c.canonical({str(f['fixture_id']):lineup()}))
    process=subprocess.run([sys.executable,'-B',str(ROOT/'Scripts/ops/phase4_candidate_worker.py'),
        '--source',str(FROZEN),'--prepared',str(path),'--output',str(path/'candidates')],capture_output=True,text=True,timeout=180,env=env)
    assert process.returncode==0,process.stderr
    repeated=subprocess.run([sys.executable,'-B',str(ROOT/'Scripts/ops/phase4_candidate_worker.py'),
        '--source',str(FROZEN),'--prepared',str(path),'--output',str(path/'repeat')],capture_output=True,text=True,timeout=180,env=env)
    assert repeated.returncode==0,repeated.stderr
    assert (path/'repeat/forecasts.jsonl').read_bytes()==(path/'candidates/forecasts.jsonl').read_bytes()
    report=json.loads((path/'candidates/report.json').read_text())
    assert not report['invalid'],report['invalid']
    return {r['candidate']['id']:r for r in map(json.loads,(path/'candidates/forecasts.jsonl').open())},report


def test_all_variants_execute_and_control_patches_restore(wired):
    forecasts,report=wired
    assert len(forecasts)==len(c.registry()) and report['control_parity']==1
    assert report['io_violations']==[]
    for r in forecasts.values():
        assert not r['publication_enabled']
        assert r['id']==c.digest({k:v for k,v in r.items() if k!='id'})


def test_distributions_retain_means_and_full_asian_outcomes(wired):
    forecasts,_=wired;control=forecasts['control']
    for id,r in forecasts.items():
        family=r['candidate']['family']
        if family.startswith('dispersion_') or family in ('goal_rho','goal_nb'):
            assert r['means']==control['means'] and r['goal_means']==control['goal_means']
        for market in c.MARKETS:
            assert len(r['diagnostics'][market])==7
            for outcomes in r['diagnostics'].get(market,{}).values():
                over,under=outcomes['Over'],outcomes['Under']
                assert over['full_win']==pytest.approx(under['full_loss'])
                assert over['half_loss']==pytest.approx(under['half_win'])
                assert over['push']==pytest.approx(under['push'])


def test_fitted_strength_and_lineup_reach_shared_goal_distribution(wired):
    forecasts,_=wired;control=forecasts['control']
    for family in ('strength_goals:10.0','lineup:1.0'):
        r=forecasts[family]
        assert r['goal_means']!=control['goal_means']
        assert r['diagnostics']['btts_yes']!=control['diagnostics']['btts_yes']
        assert r['diagnostics']['btts_yes']==pytest.approx(sum(p for h,a,p in r['score_distribution'] if h and a))
        assert sum(r['diagnostics']['winner'].values())==pytest.approx(1.)
    assert forecasts['strength_corners:10.0']['means']['corners']!=control['means']['corners']
    assert forecasts['strength_sot:10.0']['means']['sot']!=control['means']['sot']


def test_mean_candidates_change_inputs_without_affecting_other_markets(wired):
    forecasts,_=wired;control=forecasts['control']
    assert forecasts['defensive_xg:0.5']['means']['goals']!=control['means']['goals']
    assert forecasts['defensive_xg:0.5']['means']['corners']==control['means']['corners']
    assert forecasts['calendar_recency:30.0']['recent']!=control['recent']
    assert forecasts['calendar_recency:30.0']['evidence']['recent_inputs']
    assert forecasts['recent_window:12']['recent']!=control['recent']
    assert forecasts['league_pool:8.0']['evidence']['field_support']
    assert forecasts['venue_pool:8.0']['evidence']['field_support']
    assert forecasts['xg_fallback']['evidence']['field_support']
    assert forecasts['xg_fallback']['means']['goals']!=control['means']['goals']
    prior=forecasts['gradual_prior:16.0']['projection_path_records'][0]['context']['data_quality']['profiles']['home']
    old=control['projection_path_records'][0]['context']['data_quality']['profiles']['home']
    assert old['current_season_matches']>=8 and old['prior_weight']==0 and prior['prior_weight']>0


def test_profile_support_uses_known_values_and_exact_unrounded_mixture():
    from Scripts.data_platform.features.phase4_history import HistoricalInputs
    from Scripts.data_platform.features.phase4_candidate_adapter import component_summary,league_summary
    rows=history();index=HistoricalInputs(rows)
    # Early-season effective profile: field support can differ from match count.
    cutoff='2022-07-20T12:00:00Z'
    earlier=index.before(11,'EPL',cutoff)
    n=sum(r['season']==2022 for r in earlier)
    audit={'current_season':'2022','prior_season':'2021','profile_mode':'current_plus_prior','prior_weight':round(8/(n+8),4)}
    actual=component_summary(index,11,'EPL',cutoff,'xg_home_pm',audit)
    assert actual['mean'] is None and actual['ess']==0
    actual=component_summary(index,11,'EPL',cutoff,'goals_for_pm',audit)
    assert actual['prior_weight']==8/(n+8)
    assert actual['ess']>0
    # A future league record cannot change a dated league prior.
    original=league_summary(index,'EPL',cutoff,'goals_for_pm',actual)
    changed=deepcopy(rows)
    for r in changed:
        if c.utc(r['kickoff'])>=c.utc(cutoff):r['home']['goals']=1000
    assert league_summary(HistoricalInputs(changed),'EPL',cutoff,'goals_for_pm',actual)==original


@pytest.mark.parametrize('operation',[
    "socket.getaddrinfo('localhost',80)","sqlite3.connect(':memory:')",
    "Path('/tmp/.env').read_text()","subprocess.run(['true'])",
    "Path('/tmp/spix-unapproved-candidate-write').write_text('x')",
    "Path('Index/platform.db').read_bytes()",
    "Path('Research/forbidden-labels.json').read_text()",
])
def test_candidate_worker_blocks_live_and_unapproved_io(tmp_path,operation):
    script=f'''from pathlib import Path
import socket,sqlite3,subprocess
from Scripts.ops.phase4_candidate_worker import guard
violations=guard(Path({str(FROZEN)!r}),Path({str(tmp_path)!r}),set())
try:
    {operation}
except RuntimeError:
    pass
else:
    raise AssertionError('Forbidden operation succeeded')
assert violations
'''
    run=subprocess.run([sys.executable,'-B','-c',script],cwd=ROOT,capture_output=True,text=True,timeout=30)
    assert run.returncode==0,run.stderr
