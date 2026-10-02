"""Temporal and numerical contracts for the player research path."""
from copy import deepcopy
from datetime import datetime,timedelta,timezone
import math

import numpy as np
import pytest

from Scripts.data_platform.features import phase4_player_context as p
from Scripts.ops import phase4_assessment as run


def fixture(fid,day,home=10,away=20):
    return dict(fixture_id=fid,competition='EPL',season=2022,kickoff=day+'T15:00:00Z',home_team_id=home,away_team_id=away)


def record(fid,day,*,missing=False):
    f=fixture(fid,day); players={}; lineups={}
    for side,offset in (('home',100),('away',200)):
        pp=[]
        for i in range(12):
            pp.append({'player_id':offset+i,'minutes':90 if i<10 else 45,'goals':None if missing else (1 if i==fid%12 else 0),
                       'sot':None if missing else (2 if i==fid%12 else 0),'position':'F','starter':i<11})
        players[side]=pp; lineups[side]={'starters':[offset+i for i in range(11)],'bench':[offset+11],
                                      'positions':{str(offset+i):'F' for i in range(12)}}
    r={'fixture':f,'players':players,'lineups':lineups,'reasons':[]}; r['id']=p.cm.digest(r); return r


@pytest.fixture
def history():
    return [record(i+1,f'2022-09-{2+i*3:02d}') for i in range(8)]


def test_minutes_cap_and_budget():
    r=p.allocate({i:100. if i<2 else 10. for i in range(14)})
    assert sum(r.values())==pytest.approx(990) and max(r.values())<=90
    with pytest.raises(ValueError):p.allocate({i:90. for i in range(10)})


def test_future_and_own_performance_cannot_change_any_stage(history):
    target=record(99,'2022-09-28'); future=record(100,'2022-10-01'); f=target['fixture']; cutoff='2022-09-28T00:00:00Z'
    a=p.History(history+[target,future]); changed=deepcopy([target,future])
    for r in changed:
        for ps in r['players'].values():
            for player in ps:player.update(goals=900,sot=900,minutes=1)
    b=p.History(history+changed)
    for stage in p.STAGES:
        assert a.build(f,cutoff,stage)==b.build(f,cutoff,stage)


def test_expected_stage_does_not_read_target_xi(history):
    target=record(99,'2022-09-28'); other=deepcopy(target);other['lineups']=None
    f=target['fixture']; cutoff='2022-09-28T00:00:00Z'
    assert p.History(history+[target]).build(f,cutoff,'expected_players')==p.History(history+[other]).build(f,cutoff,'expected_players')
    assert p.History(history+[other]).build(f,cutoff,'conditional_actual_xi')['status']=='control_fallback'


def test_all_history_earlier_and_minute_allocations_conserved(history):
    f=fixture(99,'2022-09-28'); feature=p.History(history).build(f,'2022-09-28T00:00:00Z','expected_players')
    assert feature['status']=='available'
    for side in ('home','away'):
        for kind in ('reference_minutes','forecast_minutes'):
            assert sum(v[kind] for v in feature['teams'][side]['players'])==pytest.approx(990)
            assert all(0<=v[kind]<=90+1e-8 for v in feature['teams'][side]['players'])
        assert set(feature['teams'][side]['team_fixture_ids'])==set(range(1,9))


def test_unknown_player_shrinks_to_dated_prior(history):
    h=p.History(history);pr=h.priors('EPL',2022,'2022-09-28T00:00:00Z')
    r=h.player_rate('EPL',2022,'2022-09-28T00:00:00Z',9999,'F',pr)
    assert r['goals_known_minutes']==0 and r['fixture_ids']==[]
    assert r['goals_per90']==pytest.approx(pr['rates']['F:goals'],abs=1e-14)


def test_missing_counts_not_observed_zero(history):
    a=p.History(history); b=deepcopy(history)
    for r in b:
        for player in r['players']['home']:
            if player['player_id']==100:player['sot']=None
    b=p.History(b);cut='2022-09-28T00:00:00Z'
    x=a.player_rate('EPL',2022,cut,100,'F',a.priors('EPL',2022,cut))
    y=b.player_rate('EPL',2022,cut,100,'F',b.priors('EPL',2022,cut))
    assert x['sot_known_minutes']>0 and y['sot_known_minutes']==0


def test_same_composition_zero_effect(history):
    for r in history:
        for ps in r['players'].values():
            for v in ps:v['goals']=1;v['sot']=2
    feature=p.History(history).build(fixture(99,'2022-09-28'),'2022-09-28T00:00:00Z','expected_players')
    assert p.vector(feature,'goals')==pytest.approx([0.,0.],abs=1e-12)


def training(beta=.8):
    result=[]; start=datetime(2022,1,1,tzinfo=timezone.utc)
    for i in range(520):
        date=start+timedelta(days=i%300); x=.2 if i%2 else -.2; base=8.
        result.append({'fixture_id':i,'kickoff':date.isoformat(),'week':date.strftime('%G-W%V'),'available':True,
                       'base':[base],'x':[x,x],'target':[float(round(base*math.exp(beta*x)))]})
    return result


def test_fit_learns_effect_and_keeps_zero_valid():
    a=p.fit(training(),'sot','expected_players')
    assert a['status']=='fitted' and a['coefficient']>0 and a['training_objective']<a['zero_objective']
    b=p.fit(training(-.8),'sot','expected_players');assert b['coefficient']==0
    assert a==p.fit(training(),'sot','expected_players')


def test_late_training_and_public_activation_refused(history):
    r=training();r[0]['kickoff']='2023-01-01T00:00:00Z'
    with pytest.raises(ValueError,match='future_training'):p.fit(r,'sot','expected_players')
    b=p.fit(training(),'sot','expected_players');f=p.History(history).build(fixture(99,'2022-09-28'),'2022-09-28T00:00:00Z','expected_players')
    with pytest.raises(ValueError,match='not_publicly'):p.apply(b,f,[8.],public=True)
    with pytest.raises(ValueError,match='future_or_wrong'):p.apply(b,f,[8.])


def test_fixture_and_player_ids_cannot_be_swapped(history):
    target=record(99,'2022-09-28'); f=deepcopy(target['fixture']);f['home_team_id']=999
    feature=p.History(history+[target]).build(f,'2022-09-28T00:00:00Z','conditional_actual_xi')
    assert feature['status']=='control_fallback'


def test_normalize_rejects_incomplete_and_preserves_nulls():
    f=fixture(99,'2022-09-28')|{'home':{'goals':0,'sot':None},'away':{'goals':0,'sot':None}};xi=[];players=[]
    for tid,off in ((10,100),(20,200)):
        xi.append({'team':{'id':tid},'startXI':[{'player':{'id':off+i,'pos':'F'}} for i in range(11)],'substitutes':[]})
        players.append({'team':{'id':tid},'players':[{'player':{'id':off+i},'statistics':[{'games':{'minutes':90,'position':'F'},'goals':{'total':0},'shots':{'on':None}}]} for i in range(11)]})
    a=p.normalize(f,{'response':players},{'response':xi});assert a['players']['home'][0]['sot'] is None
    players[0]['players'][0]['statistics'][0]['games']['minutes']=None
    assert p.normalize(f,{'response':players},{'response':xi})['players'] is None


def test_certified_team_total_resolves_zeroes_but_positive_residual_never_does():
    f=fixture(99,'2022-09-28')|{'home':{'goals':1,'sot':2},'away':{'goals':0,'sot':0}}
    xi=[];players=[]
    for tid,off in ((10,100),(20,200)):
        xi.append({'team':{'id':tid},'startXI':[{'player':{'id':off+i,'pos':'F'}} for i in range(11)],'substitutes':[]})
        players.append({'team':{'id':tid},'players':[{'player':{'id':off+i},'statistics':[{'games':{'minutes':90,'position':'F'},
            'goals':{'total':1 if tid==10 and i==0 else None},'shots':{'on':2 if tid==10 and i==0 else None}}]} for i in range(11)]})
    a=p.normalize(f,{'response':players},{'response':xi})
    assert a['players']['home'][1]['goals']==0 and a['players']['home'][1]['reported_goals'] is None
    assert a['reconciliation']['home']['goals']['qualified']
    assert all(v['goals']==0 for v in a['players']['away'])
    f['home']['goals']=2
    b=p.normalize(f,{'response':players},{'response':xi})
    assert not b['reconciliation']['home']['goals']['qualified']
    assert all(v['goals'] is None for v in b['players']['home'])
    assert b['players']['home'][0]['reported_goals']==1 and b['players']['home'][0]['sot']==2


def test_simple_baseline_future_independence_and_no_zero_imputation():
    history={'EPL':[]}
    for i in range(25):
        h=fixture(i,f'2022-09-{1+i:02d}');h.update(home={'goals':0,'sot':None},away={'goals':2,'sot':None});history['EPL'].append(h)
    f=fixture(99,'2022-09-28')|{'as_of':'2022-09-28T00:00:00Z'}
    a=run.simple(f,'goals',history)
    history['EPL'].append(fixture(100,'2022-10-01')|{'home':{'goals':99},'away':{'goals':99}})
    assert a==run.simple(f,'goals',history) and a['league']==[0.,2.]
    with pytest.raises(ValueError,match='lacks_support|lacks support'):run.simple(f,'sot',history)


def test_market_missingness_does_not_disqualify_other_market(history):
    for r in history:
        for rows in r['players'].values():
            for item in rows:
                if item['player_id']%100<9:item['sot']=None
    f=p.History(history).build(fixture(99,'2022-09-28'),'2022-09-28T00:00:00Z','expected_players')
    assert p.available(f,'goals') and not p.available(f,'sot')
    assert p.vector(f,'sot')==[0.,0.]


def test_nonzero_goal_effect_reaches_coherent_derived_probabilities(history):
    f=p.History(history).build(fixture(99,'2023-01-05'),'2023-01-05T00:00:00Z','expected_players')
    assert p.available(f,'goals')
    f['teams']['home']['features']['goals']=.15;f['teams']['away']['features']['goals']=-.1
    f['id']=p.cm.digest({k:v for k,v in f.items() if k!='id'})
    bundle={'version':p.VERSION,'market':'goals','stage':'expected_players','cutoff':'2023-01-01T00:00:00Z','coefficient':.5}
    bundle['id']=p.cm.digest(bundle)
    mu=p.apply(bundle,f,[1.5,1.]);assert mu!=[1.5,1.]
    d={'means':np.array([[[1.5,1.]]]),'alpha':np.array([[0.]]),'targets':np.array([3.]),'team_targets':np.array([[2.,1.]])}
    a=run.forecasts([{'fixture_id':99}],d,'goals',[[1.5,1.]])[0]['scores']
    b=run.forecasts([{'fixture_id':99}],d,'goals',[mu])[0]['scores']
    assert a['derived']['btts']['p']!=b['derived']['btts']['p']
    assert sum(b['derived']['winner']['p'])==pytest.approx(1.,abs=1e-10)
    assert all(sum(v['p'])==pytest.approx(1.,abs=1e-10) for v in b['derived']['handicaps'].values())
