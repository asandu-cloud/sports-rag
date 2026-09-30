"""Composition isolation, coherent outputs and fail-closed qualification gates."""
from copy import deepcopy
from datetime import timedelta
import importlib.util
from pathlib import Path
import pytest
from Scripts.data_platform.features import phase4_finalists as f,phase4_candidates as cm,phase4_backtest as b


def inputs(league='EPL',n=8):
    snapshot={'fixture':{'fixture_id':1,'competition':league},'snapshot_id':'saved','as_of':'2023-02-01T12:00:00Z',
        'profile_quality':{'1':{'current_season_matches':n},'2':{'current_season_matches':9}}}
    control={'fixture_id':1,'input_snapshot_id':'saved','goal_means':[1.,1.],
        'means':{'goals':2.,'corners':10.,'sot':8.},'variances':{'corners':20.,'sot':16.},'evidence':{'fallbacks':[]}}
    strength=deepcopy(control);strength['goal_means']=[1.7,.8];strength['means']['goals']=2.5
    return snapshot,control,strength


def test_composition_retains_identity_and_does_not_mutate_inputs():
    snapshot,control,strength=inputs();saved=deepcopy((snapshot,control,strength))
    r=f.compose(snapshot,control,strength,'supported_stack')
    assert (snapshot,control,strength)==saved
    assert r['goal_means']==[1.7,.8] and r['means']['corners']==10.
    assert r['variances']=={'corners':12.5,'sot':8.64}
    assert r['joint_cross_market_probability'] is None and not r['publication_enabled']
    assert f.digest(r)==f.digest(f.compose(snapshot,control,strength,'supported_stack'))


@pytest.mark.parametrize('league,n',[('UECL',20),('EPL',7),('Unknown',20)])
def test_unsupported_scope_is_exact_control_fallback(league,n):
    snapshot,control,strength=inputs(league,n);r=f.compose(snapshot,control,strength,'supported_stack')
    for k in ('goal_means','means','variances'):assert r[k]==control[k]
    assert set(r['component_ids'].values())=={'control'}
    assert r['evidence']['fallbacks']==['unsupported_league_or_season_stage']


def test_missing_counts_remain_missing_and_zero_remains_observed():
    snapshot,control,strength=inputs();control['means'].update(corners=None,sot=0.)
    r=f.compose(snapshot,control,strength,'supported_stack')
    assert r['means']['corners'] is None and r['means']['sot']==0 and r['variances']['sot']==0


def test_component_fallback_and_fit_provenance_are_market_specific():
    snapshot,control,strength=inputs();strength['evidence']['fallbacks']=['unknown_or_sparse_team']
    r=f.compose(snapshot,control,strength,'supported_stack')
    assert r['candidate']['family']=='composite'
    assert r['market_fallbacks']=={'goals':['unknown_or_sparse_team'],'corners':[],'sot':[]}
    assert f.goal_fit_id('control',True,'fit') is None
    assert f.goal_fit_id('supported_stack',False,'fit') is None
    assert f.goal_fit_id('supported_stack',True,'fit')=='fit'


def test_future_scope_and_mismatched_snapshot_are_rejected():
    snapshot,control,strength=inputs();strength['input_snapshot_id']='different'
    with pytest.raises(ValueError,match='identity'):f.compose(snapshot,control,strength,'supported_stack')
    snapshot,control,strength=inputs();snapshot['as_of']='2025-01-01T00:00:00Z'
    with pytest.raises(ValueError,match='Development'):f.compose(snapshot,control,strength,'supported_stack')


def test_combination_keeps_goal_and_asian_probabilities_coherent():
    root=Path(__file__).resolve().parents[2];path=root/'Research/phase4-baseline-2026-09-28/control/workspace/Scripts/rag_ingest/prob_models.py'
    spec=importlib.util.spec_from_file_location('finalist_frozen_probability',path);prob=importlib.util.module_from_spec(spec);spec.loader.exec_module(prob)
    snapshot,control,strength=inputs();r=f.compose(snapshot,control,strength,'supported_stack')
    target={'labels':{'goals':3,'corners':11,'sot':9},'team_labels':{'home':{'goals':2},'away':{'goals':1}}}
    for m in f.MARKETS:
        score=b.score_forecast(r,cm.registry()[0],m,target,cm,prob)
        for value in score['totals'].values():assert sum(value['p'])==pytest.approx(1.)
        probabilities=[score['totals'][str(line)]['binary']['p'] for line in b.LINES[m] if line%1==.5]
        assert probabilities==sorted(probabilities,reverse=True)
        if m=='goals':
            assert sum(score['derived']['winner']['p'])==pytest.approx(1.)
            for value in score['derived']['handicaps'].values():assert sum(value['p'])==pytest.approx(1.)


def memberships(n=500,weeks=10):
    rows=[]
    for start in ('2025-01-06T12:00:00Z','2025-04-07T12:00:00Z'):
        for i in range(n):
            t=f.utc(start)+timedelta(weeks=i%weeks)
            rows.append({'fixture_id':len(rows)+1,'competition':f.LEAGUES[i%5],'kickoff':t.isoformat(),
                'partition':'calibration','eligible_markets':list(f.MARKETS),'completed':True,'season':2024,'row_count':1})
    return rows


def test_exact_sample_boundary_is_only_a_metadata_upper_bound():
    report=f.reserve_preflight(memberships())
    assert report['status']=='requires_separate_locked_input_preparation'
    assert report['stages']['qualification']['markets']['goals']['supported_leagues_upper_bound']=={'fixtures':500,'observed_iso_weeks':10}
    assert not report['label_access_permitted'] and not report['calibrators_fitted']
    f.require_qualification_support(report)


@pytest.mark.parametrize('n,weeks,reason',[(499,10,'fewer_than_500_eligible_fixtures'),(600,9,'fewer_than_10_observed_weeks')])
def test_insufficient_quarter_cannot_open_reserves(n,weeks,reason):
    report=f.reserve_preflight(memberships(n,weeks))
    assert reason in report['stages']['qualification']['markets']['goals']['reasons']
    with pytest.raises(ValueError,match='blocked'):f.require_qualification_support(report)


def test_duplicate_and_outcome_contaminated_metadata_fail():
    rows=memberships()
    with pytest.raises(ValueError,match='Duplicate'):f.reserve_preflight(rows+[rows[0]])
    rows[0]['labels']={'goals':2}
    with pytest.raises(ValueError,match='Outcome'):f.reserve_preflight(rows)


def test_final_system_memberships_never_count_toward_qualification():
    rows=memberships(499,9);expected=f.reserve_preflight(rows)
    for i in range(1000):rows.append({'fixture_id':99999+i,'competition':'EPL','kickoff':'2025-07-01T12:00:00Z',
        'partition':'final_system_test','eligible_markets':list(f.MARKETS)})
    assert f.reserve_preflight(rows)==expected


def test_worker_rejects_qualification_dates_before_creating_output(tmp_path):
    from Scripts.ops.phase4_finalist_worker import run
    with pytest.raises(ValueError,match='2023'):run(tmp_path,tmp_path,'2025-Q2')
    assert not list(tmp_path.iterdir())
