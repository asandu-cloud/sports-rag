"""Chronology, archived-function parity, fit isolation and coherent probabilities."""
from copy import deepcopy
from datetime import timedelta
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import poisson

from Scripts.data_platform.features import phase4_card_candidates as m
from Scripts.ops import phase4_card_candidates as cli
from Scripts.ops import phase4_card_backtest as bridge
from Scripts.tests.test_phase4_card_backtest import target, BATCH
from Scripts.tests.test_phase4_card_reconstruction import weights
from Scripts.rag_ingest.prob_models import asian_total_settlement_outcome


def engine():
    root=Path(__file__).resolve().parents[2]
    source=root/'Research/phase4-card-reconstruction-2026-09-30/reconstruction/source/Scripts/rag_ingest/core'
    return m.load_engine((source/'projections.py').read_text(),(source/'weights.py').read_text())


def data(n=20):
    start=m.cards.utc('2023-01-01T12:00:00Z')
    rr=[target(i+1,(start+timedelta(days=i*4)).isoformat()) for i in range(n)]
    rr=bridge.adapt_targets(rr,BATCH)
    context={str(r['fixture_id']):{'fouls':{'home':10.,'away':12.},
             'starters':{s:[int(p['player_id']) for p in r['player_evidence']['players'] if p['team_id']==str(r[s+'_team_id'])]
                         for s in ('home','away')}} for r in rr}
    return rr,context


def test_exact_fixed_inputs_and_means_survive_enrichment():
    rr,context=data(); features=m.build_features(rr,context,weights())
    for r,f in zip(rr,features):
        inputs=m.fixed.dated_inputs(r,rr,minimum_recorded_minutes=2)
        pred,reason=m.fixed.fixed_prediction(r,inputs,weights())
        assert f['fixed_snapshot_id']==m.fixed.digest(inputs)
        assert f['baseline']==pred and f['baseline_exclusion']==reason
        assert f['inputs']['source_fixture_ids']==inputs['source_fixture_ids']
    assert m.build_features(list(reversed(rr)),context,weights())==features


def test_future_targets_fouls_players_and_lineups_cannot_change_earlier_features():
    rr,context=data(); original=m.build_features(rr,context,weights())
    rr2=deepcopy(rr); cc=deepcopy(context)
    for r in rr2[12:]:
        r.update(target=22,team_targets={'home':11,'away':11})
        for p in r['player_evidence']['players']:p.update(minutes=120,yellow=1,weighted_cards=1)
        cc[str(r['fixture_id'])]['fouls']={'home':99,'away':99}
        cc[str(r['fixture_id'])]['starters']={'home':list(range(11)),'away':list(range(11,22))}
    changed=m.build_features(rr2,cc,weights())
    assert original[:12]==changed[:12]
    # The changed match's own target is absent from its input hash; lineup is
    # intentionally a conditional scenario so is tested independently above.
    own=deepcopy(rr);own[12].update(target=22,team_targets={'home':11,'away':11})
    own_features=m.build_features(own,context,weights())
    assert own_features[12]['input_id']==original[12]['input_id']


def test_current_lineup_outcomes_not_used_and_sparse_player_history_is_explicit():
    rr,context=data(); features=m.build_features(rr,context,weights())
    for f in features[1:]:
        for side in ('home','away'):
            for player in f['lineup_scenario'][side]['players']:
                assert f['fixture_id'] not in player['source_fixture_ids']
                assert set(player['source_fixture_ids'])<=set(f['inputs']['source_fixture_ids'])
                assert 0<player['expected_minutes']<=90
    assert features[0]['lineup_scenario'] is None


def test_residual_referee_uses_earlier_team_only_forecasts_not_current_target():
    rr,context=data(); features=m.build_features(rr,context,weights())
    f=features[-1]; residual=f['referee_residual']
    earlier={r['fixture_id']:r for r in features[:-1] if r['baseline']}
    assert residual['expected']==sum(earlier[i]['baseline']['team_only_mean'] for i in residual['fixture_ids'])
    assert f['fixture_id'] not in residual['fixture_ids']
    assert residual['n']==len(residual['fixture_ids'])


def test_archived_card_function_executes_known_foul_and_induced_components():
    rr,context=data();f=m.build_features(rr,context,weights())[-1];e=engine()
    result=m.full_mean(f,{'id':'fuller_control'},e)
    # All history targets are exactly 2, all means and referee anchors are 2.
    # The inherited separate induced component has weight .12 and total 2.
    assert result['mean']==pytest.approx(2.)
    assert result['effective_sample_sizes']['home']==19
    changed=deepcopy(f);changed['referee']['cards_per_foul']=.3
    assert m.full_mean(changed,{'id':'fuller_control'},e)['mean']>2
    assert m.full_mean(changed,{'id':'no_fouls','fouls':False},e)['mean']==pytest.approx(2)
    assert e['SCORING_WEIGHTS']['projection_cards']['foul_card_blend']==.15


def test_missing_fouls_remain_distinct_from_zero():
    rr,context=data();missing=deepcopy(context);zero=deepcopy(context)
    for c in missing.values():c['fouls']={'home':None,'away':None}
    for c in zero.values():c['fouls']={'home':0.,'away':0.}
    fm=m.build_features(rr,missing,weights())[-1];fz=m.build_features(rr,zero,weights())[-1]
    assert m.profile_inputs(fm,{})[0]['home']['fouls_per_90_team'] is None
    assert m.profile_inputs(fz,{})[0]['home']['fouls_per_90_team']==0
    assert fm['input_id']!=fz['input_id']


@pytest.mark.parametrize('alpha',[0.,.025,.2,1.6])
@pytest.mark.parametrize('a,b',[(0.,1.),(.5,.5),(-.5,2.)])
def test_distribution_calibration_is_normalized_monotone_and_asian_coherent(alpha,a,b):
    s=m.score(4.,alpha,5,a,b);p=np.array(s['pmf'])
    assert p.min()>=0 and p.sum()==pytest.approx(1,abs=1e-12)
    assert s['tail_bound']<=1e-10
    assert s['diagnostics']['3.5']['p']>=s['diagnostics']['4.5']['p']>=s['diagnostics']['5.5']['p']
    names=['full_win','half_win','push','half_loss','full_loss']
    for line,profiles in s['asian'].items():
        assert sum(profiles['over'])==pytest.approx(1)
        assert profiles['under']==list(reversed(profiles['over']))
        expected=[0.]*5
        for y,prob in enumerate(p):expected[names.index(asian_total_settlement_outcome(y,float(line),'Over'))]+=prob
        assert profiles['over']==pytest.approx(expected)
    assert float(m.transform_cdf(0,a,b))==0 and float(m.transform_cdf(1,a,b))==1


def test_identity_distribution_matches_exact_poisson_likelihood():
    s=m.score(3.,0.,9)
    assert s['nll']==pytest.approx(-poisson(3).logpmf(9),abs=1e-12)
    assert s['mean']==pytest.approx(3,abs=1e-8)
    with pytest.raises(ValueError):m.transform_cdf(.5,0,-1)


def fitting_rows():
    rng=np.random.default_rng(73);start=m.cards.utc('2022-01-01T12:00:00Z')
    result=[]
    for i in range(700):
        kickoff=start+timedelta(days=i//3)
        result.append({'fixture_id':i+1,'kickoff':kickoff.isoformat(),'as_of':kickoff.replace(hour=0).isoformat(),
                       'available_at':(kickoff+timedelta(hours=3)).isoformat(),'competition':'EPL',
                       'target':int(rng.poisson(3.)), 'mean':4.,'alpha':0.,'fit_cutoff':'2021-12-31T00:00:00Z','fit_ids':[],
                       'means':{s['id']:4. for s in m.registry()}})
    return result


def test_calibration_and_selection_are_isolated_from_later_outcomes(monkeypatch):
    rr=fitting_rows(); cutoff='2022-09-01T00:00:00Z'
    later=deepcopy(rr[-1]);later.update(fixture_id=9999,kickoff='2023-05-01T12:00:00Z',available_at='2023-05-01T15:00:00Z',target=9999)
    one=m.fit_calibration(rr,cutoff);two=m.fit_calibration([*reversed(rr),later],cutoff)
    assert one==two and one['a']>0 # lower counts increase the CDF at a fixed threshold
    monkeypatch.setattr(m,'full_mean',lambda f,s,e:{'mean':4.})
    features={r['fixture_id']:{} for r in rr}
    assert m.select_recipe(rr,features,{},cutoff)==m.select_recipe([*reversed(rr),later],features,{},cutoff)
    assert m.select_recipe(rr,features,{},cutoff)['spec']=={'id':'fuller_control'}


def test_calibration_rejects_in_sample_predictions_and_insufficient_support():
    rr=fitting_rows();rr[0]['fit_ids']=[rr[0]['fixture_id']]
    with pytest.raises(ValueError,match='held-out'):m.fit_calibration(rr)
    with pytest.raises(ValueError,match='support'):m.fit_calibration(rr[:100])
    with pytest.raises(ValueError,match='Reserved'):m.training(rr,'2025-01-01')
    bad=deepcopy(rr);bad[0]['available_at']=bad[0]['kickoff']
    with pytest.raises(ValueError,match='availability'):m.training(bad,'2023-01-01')


def test_existing_output_never_overwritten(tmp_path):
    with pytest.raises(FileExistsError):cli.run(tmp_path,tmp_path)
    with pytest.raises(FileExistsError):cli.prepare(tmp_path)
