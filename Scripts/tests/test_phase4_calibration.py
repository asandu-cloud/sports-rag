"""Leakage, monotonicity, replay and coherent probability calibration tests."""
from copy import deepcopy
from datetime import timedelta
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
from scipy.stats import poisson
from Scripts.data_platform.features import phase4_calibration as c

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('calibration_frozen_probability',ROOT/'Research/phase4-baseline-2026-09-28/control/workspace/Scripts/rag_ingest/prob_models.py')
p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)


def rows():
    rng=np.random.default_rng(20260928);records=[]
    for i in range(800):
        date=c.utc('2019-01-01T12:00:00Z')+timedelta(days=i);prob=float(rng.uniform(.05,.95));mean=float(rng.uniform(1,5))
        y=int(rng.random()<1/(1+math.exp(-(.2+.45*c.logit(prob)+.3*(mean-3)))))
        records.append({'fixture_id':i+1,'problem':'goals','period':'regulation_time','kickoff':date.isoformat(),'as_of':date.isoformat(),
            'available_at':(date+timedelta(hours=3)).isoformat(),'week':str(date.isocalendar()[:2]),'league':'A' if i%2 else 'B','y':y,
            'bases':{'control':{'p':prob,'mean':mean,'goal_means':[mean*.6,mean*.4]}}})
    return records


def fit(data=None,**kw):
    options=dict(cutoff='2022-01-01T00:00:00Z',problem='goals',kind='sigmoid',C=1.);options.update(kw)
    return c.fit_binary(rows() if data is None else data,**options)


def test_deterministic_fitting_json_replay_and_nonnegative_slope():
    data=rows();a=fit(data);b=fit(list(reversed(data)));assert a==b
    row=deepcopy(data[0]);row['as_of']=row['kickoff']='2022-01-03T12:00:00Z'
    assert c.predict_binary(a,row)==c.predict_binary(json.loads(json.dumps(a)),row)
    assert a['coefficients'][0]>=0


def test_preprocessing_is_training_only_and_future_rows_cannot_enter():
    data=rows();future=deepcopy(data[0]);future['fixture_id']=9999;future['kickoff']=future['as_of']='2022-02-01T00:00:00Z';future['available_at']='2022-02-01T03:00:00Z'
    future['bases']['control']['mean']=1000000.
    with pytest.raises(ValueError,match='unavailable'):fit(data+[future],kind='comparator')
    cutoff='2022-01-01T00:00:00Z'
    assert fit(c.preceding(data+[future],cutoff),kind='comparator')==fit(data,kind='comparator')
    model=fit(data,kind='comparator');expected=np.mean([c.numeric(r,'comparator','control') for r in data],axis=0)
    assert model['means']==pytest.approx(expected)
    future['league']='Unknown';assert c.predict_binary(model,future)['unknown_league']


def test_sigmoid_is_monotone_even_when_raw_signal_is_reversed():
    data=rows()
    for r in data:r['y']=int(r['bases']['control']['p']<.5)
    model=fit(data);assert model['coefficients'][0]>=0
    prediction=[]
    for probability in (0.,.001,.1,.5,.9,.999,1.):
        r=deepcopy(data[0]);r['as_of']=r['kickoff']='2022-01-01T00:00:00Z';r['bases']['control']['p']=probability
        prediction.append(c.predict_binary(model,r)['p'])
    assert prediction==sorted(prediction)


@pytest.mark.parametrize('case',['period','duplicate','class','support','nonfinite','future_asof'])
def test_bad_training_evidence_is_rejected(case):
    data=rows()
    if case=='period':data[0]['period']='extra_time'
    elif case=='duplicate':data.append(data[0])
    elif case=='class':
        for r in data:r['y']=1
    elif case=='support':data=data[:400]
    elif case=='nonfinite':data[0]['bases']['control']['p']=float('nan')
    elif case=='future_asof':data[0]['as_of']='2023-01-01T00:00:00Z'
    with pytest.raises(ValueError):fit(data)


def test_future_tampered_and_wrong_target_bundles_rejected():
    model=fit();r=rows()[0]
    with pytest.raises(ValueError):c.predict_binary(model,r)
    r['as_of']='2022-02-01T00:00:00Z';bad=deepcopy(model);bad['intercept']+=1
    with pytest.raises(ValueError):c.predict_binary(bad,r)
    r['problem']='sot'
    with pytest.raises(ValueError):c.predict_binary(model,r)
    r['problem']='goals';r['as_of']='2025-02-01T00:00:00Z'
    with pytest.raises(ValueError):c.predict_binary(model,r)


def test_nonconvergence_is_not_an_executable_bundle():
    with pytest.raises(ValueError,match='converge'):fit(maxiter=0)


def test_nested_base_selection_cannot_use_later_quarter_losses():
    data=rows()
    for i,r in enumerate(data):
        date=c.utc('2022-01-01T12:00:00Z')+timedelta(days=i//3)
        r['kickoff']=r['as_of']=date.isoformat();r['available_at']=(date+timedelta(hours=3)).isoformat();r['week']=str(date.isocalendar()[:2])
        r['bases'].update({'strength_goals:1.0':{'primary_nll':3.},'strength_goals:10.0':{'primary_nll':2.9},'strength_goals:100.0':{'primary_nll':3.1}})
    cutoff='2022-07-01T00:00:00Z';initial=c.choose_base(data,'goals',cutoff)
    for r in data:
        if c.utc(r['available_at'])>=c.utc(cutoff):r['bases']['strength_goals:1.0']['primary_nll']=-1e9
    assert c.choose_base(data,'goals',cutoff)==initial
    with pytest.raises(ValueError,match='Insufficient'):c.choose_base(data,'goals','2022-04-01T00:00:00Z')


@pytest.mark.parametrize('T',c.TEMPERATURES)
def test_temperature_tail_and_coherence_against_large_independent_grid(T):
    home,away=5.,3.;d=c.powered_goals(home,away,T,p);assert d['tail_bound']<=1e-10
    full=np.outer(poisson(home).pmf(np.arange(100)),poisson(away).pmf(np.arange(100)))
    for i in (0,1):
        for j in (0,1):full[i,j]*=p._tau(i,j,home,away,-.1)
    full=full**(1/T);full/=full.sum();n=d['extent']+1
    assert abs(1-float(full[:n,:n].sum()))<=d['tail_bound']+1e-14
    assert d['matrix']==pytest.approx(full[:n,:n],abs=1e-10)
    s=c.score_joint(d,[2,1]);assert math.isfinite(s['nll'])
    assert sum(s['derived']['winner']['p'])==pytest.approx(1.)
    for line,result in s['totals'].items():assert sum(result['p'])==pytest.approx(1.)
    for result in s['derived']['handicaps'].values():assert sum(result['p'])==pytest.approx(1.)
    pp=[s['totals'][str(line)]['binary']['p'] for line in (1.5,2.5,3.5)]
    assert pp==sorted(pp,reverse=True)
    if T==1.:
        assert s['nll']==pytest.approx(-math.log(p.dixon_coles_scoreline_prob(2,1,home,away)),abs=1e-10)


def test_temperature_expands_support_after_flattening_and_rejects_negative_cells():
    a=c.powered_goals(5.,3.,1.,p);b=c.powered_goals(5.,3.,1.5,p)
    assert b['extent']>a['extent']
    with pytest.raises(ValueError):c.powered_goals(20.,20.,1.,p)
    with pytest.raises(ValueError):c.powered_goals(1.,1.,2.,p)


def test_tuning_io_guard_denies_2023_shard(tmp_path):
    source=tmp_path/'implementation';source.mkdir();output=tmp_path/'tuning';output.mkdir()
    # Use a Research path to exercise the same archive boundary as real tuning.
    forbidden=ROOT/'Research/phase4-step3-calibration-2026-09-29/2023.jsonl'
    script=f'''from pathlib import Path
from Scripts.ops.phase4_candidate_worker import guard
v=guard(Path({str(source)!r}),Path({str(output)!r}),set())
try:Path({str(forbidden)!r}).read_text()
except RuntimeError:pass
else:raise AssertionError('Unapproved archive was readable')
assert v
'''
    result=subprocess.run([sys.executable,'-B','-c',script],cwd=ROOT,capture_output=True,text=True)
    assert result.returncode==0,result.stderr


def earlier_challengers():
    data=rows()
    for r in data:
        r['bases'].update({'strength_goals:1.0':{'primary_nll':3.},'strength_goals:10.0':{'primary_nll':2.9},'strength_goals:100.0':{'primary_nll':3.1}})
    return data


def test_earlier_warmup_is_explicit_deterministic_and_never_selects_final_setting():
    data=earlier_challengers();cutoff='2022-01-01T00:00:00Z'
    with pytest.raises(ValueError,match='Insufficient'):c.choose_base(data,'goals',cutoff)
    selected=c.choose_base(data,'goals',cutoff,earlier_fitting=True)
    assert selected['base_id']=='strength_goals:10.0' and len(selected['fixture_ids'])==800
    assert c.choose_base(list(reversed(data)),'goals',cutoff,earlier_fitting=True)==selected
    with pytest.raises(ValueError,match='nested 2022'):c.choose_base(data,'goals','2023-01-01T00:00:00Z',earlier_fitting=True)


def test_earlier_base_choices_ignore_future_labels_and_features():
    data=earlier_challengers();future=deepcopy(data[0]);future['fixture_id']=9999
    future['as_of']=future['kickoff']='2022-01-01T00:00:00Z';future['available_at']='2022-01-01T03:00:00Z'
    future['bases']['strength_goals:1.0']['primary_nll']=-1e9;future['y']=1-future['y']
    cutoff='2022-01-01T00:00:00Z'
    assert c.choose_base(data+[future],'goals',cutoff,earlier_fitting=True)==c.choose_base(data,'goals',cutoff,earlier_fitting=True)
    future['available_at']=cutoff
    assert c.choose_base(data+[future],'goals',cutoff,earlier_fitting=True)==c.choose_base(data,'goals',cutoff,earlier_fitting=True)


@pytest.mark.parametrize('case',['missing','extra','nonfinite'])
def test_earlier_selection_rejects_incomplete_or_invalid_candidate_slates(case):
    data=earlier_challengers()
    if case=='missing':del data[-1]['bases']['strength_goals:1.0']
    elif case=='extra':data[-1]['bases']['strength_goals:1000.0']={'primary_nll':1.}
    else:data[-1]['bases']['strength_goals:1.0']['primary_nll']=float('nan')
    with pytest.raises(ValueError):c.choose_base(data,'goals','2022-01-01T00:00:00Z',earlier_fitting=True)


def test_earlier_forecast_scope_rejects_later_folds_before_io(tmp_path):
    from Scripts.ops.phase4_backtest_worker import run
    from Scripts.ops.phase4_earlier_forecasts import earlier_quarter
    assert earlier_quarter('2021-12-31T23:59:00Z')=='2021-Q4'
    with pytest.raises(ValueError):earlier_quarter('2022-01-01T00:00:00Z')
    with pytest.raises(ValueError,match='2019–2021'):run(tmp_path,tmp_path,'2022-Q1',earlier_calibration=True)
    assert not list(tmp_path.iterdir())
