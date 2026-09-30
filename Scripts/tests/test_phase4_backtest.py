"""Tests for chronological selection, exact scores and isolated fast execution."""
import importlib.util
from pathlib import Path
from copy import deepcopy
import math
import numpy as np
import pytest
from Scripts.data_platform.features import phase4_backtest as b,phase4_candidates as c
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('backtest_frozen_probability',ROOT/'Research/phase4-baseline-2026-09-28/control/workspace/Scripts/rag_ingest/prob_models.py')
p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)


def test_quarters_are_fixed_and_reserves_rejected():
    assert b.quarter('2022-04-01T00:00:00Z')=='2022-Q2'
    assert b.quarter_cutoff('2023-Q4')=='2023-10-01T00:00:00+00:00'
    for value in ('2024-01-01T00:00:00Z','2025-01-01T00:00:00Z','2022-01-01'):
        with pytest.raises(ValueError):b.quarter(value)


@pytest.mark.parametrize('line,actual,kind',[(2.25,2,3),(2.75,3,1),(2,2,2),(2.5,3,0),(2.5,2,4)])
def test_asian_actuals(line,actual,kind):
    assert b.outcome_class(actual,line)==kind
    pmf={0:.1,1:.2,2:.3,3:.3,4:.1}
    got=b.profile(list(pmf),list(pmf.values()),line)
    expected=p.asian_total_profile_from_counts(pmf,line,'Over')
    assert got==pytest.approx([expected[k] for k in b.OUTCOMES])


@pytest.mark.parametrize('market,family,parameter',[('goals','goal_rho',-.1),('goals','goal_nb',.2),('corners','dispersion_corners',.2),('sot','control',None)])
def test_score_distributions_and_exact_observed_likelihood(market,family,parameter):
    rates={'means':{'goals':2.5,'corners':10.,'sot':8.},'goal_means':[1.5,1.], 'variances':{'corners':15.,'sot':12.}}
    target={'labels':{'goals':3,'corners':12,'sot':7},'team_labels':{'home':{'goals':2},'away':{'goals':1}}}
    spec=next(s for s in c.registry() if s['family']==family and s['parameter']==parameter)
    result=b.score_forecast(rates,spec,market,target,c,p)
    assert math.isfinite(result['nll']) and result['rps']>=0
    for line,v in result['totals'].items():assert sum(v['p'])==pytest.approx(1,abs=1e-10)
    if family=='goal_rho':assert result['nll']==pytest.approx(-math.log(p.dixon_coles_scoreline_prob(2,1,1.5,1.,rho=-.1)))
    if market=='sot':assert result['nll']==pytest.approx(-math.log(p._count_pmf(7,8.,12.)))
    if market=='goals':
        assert sum(result['derived']['winner']['p'])==pytest.approx(1)
        assert len(result['derived']['handicaps'])==9


def selection_rows():
    rows=[]
    for i in range(600):
        for name,loss in [('control',3.),('goal_rho:-0.1',2.9),('goal_rho:0.0',2.8)]:
            rows.append({'fixture_id':i,'year':2022,'market':'goals','candidate_id':name,'week':str(i%30),
                         'status':'computed','scores':{'nll':loss}})
    return rows


def test_selection_is_2022_only_and_invalid_cannot_win_by_attrition():
    rows=selection_rows();result=b.select_settings(rows,c.registry())
    assert result['selected']['goals:goal_rho']=='goal_rho:0.0'
    bad=deepcopy(rows);bad[0]['year']=2023
    with pytest.raises(ValueError):b.select_settings(bad,c.registry())
    bad=deepcopy(rows);bad[2]['status']='invalid';bad[2].pop('scores')
    assert b.select_settings(bad,c.registry())['selected']['goals:goal_rho']=='goal_rho:-0.1'
    assert b.select_settings(rows[:-1],c.registry())['selected']['goals:goal_rho']=='goal_rho:-0.1'


def test_bootstrap_pairs_whole_weeks_and_repeats():
    rows=[{'week':str(i//10)} for i in range(300)]
    assert b.week_interval(rows,[-.1]*300)==pytest.approx([-.1,-.1])
    x=b.week_interval(rows,np.arange(300)/1000)
    assert x==b.week_interval(rows,np.arange(300)/1000)
    with pytest.raises(ValueError):b.compare([{'fixture_id':1}],[{'fixture_id':2}])
