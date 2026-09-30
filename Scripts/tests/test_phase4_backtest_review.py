"""Acceptance-gate checks independent of the scored candidate implementation."""
from copy import deepcopy
from Scripts.data_platform.features.phase4_backtest import compare


def sample(n=800):
    rows=[]
    for i in range(n):
        scores={'nll':3.,'absolute_error':1.,'error':0.,'rps':.5,'coverage80':.8,'coverage95':.95,
                'squared_error':1.,'target':i%3,'interval80':[0,4],'interval95':[0,6],
                'totals':{},'derived':{}}
        rows.append({'fixture_id':i,'week':str(i%40),'league':'A' if i<400 else 'B',
                     'season':'2022','season_stage':'16+','forecast_stage':'pre-kickoff','missingness':'missing',
                     'scores':scores})
    return rows


def test_primary_improvement_and_uncertainty_pass_with_support():
    control=sample();candidate=deepcopy(control)
    for row in candidate:row['scores']['nll']=2.9
    result=compare(control,candidate)
    assert result['passes']
    assert result['paired_week_delta95'][1]<0


def test_pooled_gain_does_not_hide_supported_league_regression():
    control=sample();candidate=deepcopy(control)
    for row in candidate:row['scores']['nll']=3.09 if row['league']=='A' else 2.7
    result=compare(control,candidate)
    assert result['relative_change']<-.005
    assert not result['passes'] and 'league:A' in result['supported_slice_regressions']


def test_small_gain_or_insufficient_support_does_not_pass():
    for n,loss in [(800,2.999),(100,2.5)]:
        control=sample(n);candidate=deepcopy(control)
        for row in candidate:row['scores']['nll']=loss
        assert not compare(control,candidate)['passes']
