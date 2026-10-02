"""Explicit reused-period access, original-guard preservation and frozen scoring."""
from copy import deepcopy
from datetime import timedelta
from pathlib import Path
import json
import sqlite3

import pytest

from Scripts.ops import phase4_card_qualification as q
from Scripts.tests.test_phase4_card_backtest import target
from Scripts.tests.test_phase4_card_candidates import data,engine
from Scripts.tests.test_phase4_card_reconstruction import weights


def later_target(fid,kickoff):
    row=target(fid,kickoff)
    return {**row,'contract':q.policy.CONTRACT,'settlement_policy':q.policy.POLICY_VERSION,
            'raw_reference':{'synthetic_test':True}}


def test_new_cutoff_does_not_change_the_original_development_guard():
    row=later_target(1,'2025-06-01T12:00:00Z')
    assert q.qualified([row])==[row]
    with pytest.raises(ValueError,match='Reserved'):q.m.fixed.qualified_rows([row],minimum_recorded_minutes=2)
    with pytest.raises(ValueError,match='Reserved'):q.qualified([row|{'kickoff':'2025-07-01T00:00:00Z'}])
    assert q.m.cards.END==q.m.cards.utc('2024-01-01')
    assert q.evidence.cards.END==q.m.cards.END
    assert not q.permitted('2025-06-30T21:00:00Z')


def test_final_system_payload_refused_before_open_or_decode(tmp_path):
    with pytest.raises(ValueError,match='Protected final-system'):
        q.read_payload(tmp_path,{'path':'nonexistent'},{},{'kickoff_utc':'2025-08-01'},'/fixtures/players')


def test_reconstruct_cutoff_injection_is_local_and_fail_closed():
    reconstruct=q.scoped_function(q.evidence.reconstruct,cards=q.scoped_cards())
    fixture={'fixture_id':7,'code':'UCL','season':2024,'kickoff_utc':'2025-05-01T12:00:00Z',
             'status':'FT','round':'Final','home_team_id':10,'away_team_id':20}
    # Scoped knockout exits before reading any player evidence.
    result=reconstruct(fixture,{},{});assert result['eligible'] is False
    assert 'special_round_deferred' in result['exclusions']
    with pytest.raises(ValueError,match='Reserved'):q.evidence.reconstruct(fixture,{},{})
    with pytest.raises(ValueError,match='Reserved'):reconstruct(fixture|{'kickoff_utc':'2025-07-01'},{},{})


def test_qualification_builder_preserves_earlier_features_and_default_guard():
    rr,context=data(12)
    later=later_target(99,'2025-03-01T12:00:00Z')
    context['99']=deepcopy(context['1'])
    earlier=q.m.build_features(rr,context,weights())
    result=q.build_features([*rr,later],context,weights())
    assert result[:len(earlier)]==earlier
    with pytest.raises(ValueError,match='Reserved'):q.m.build_features([*rr,later],context,weights())


def test_actual_lineup_does_not_enter_selected_mean():
    rr,context=data();f=q.build_features(rr,context,weights())[-1]
    changed=deepcopy(f);changed['lineup_scenario']['home']['team_starter_cards_per_90']=1000
    for spec in ({'id':'pool:16.0','pool':16.},{'id':'fuller_control'}):
        assert q.m.full_mean(f,spec,engine())==q.m.full_mean(changed,spec,engine())


def test_reused_qualification_applies_ten_week_and_league_support_rules():
    # 600 fixtures over 12 weeks: enough for the declared qualification gate,
    # although not for the 20-week development gate.
    rows=[];a=q.m.score(3.,0.,3);b=q.m.score(6.,0.,3)
    for i in range(600):
        kickoff=q.START+timedelta(weeks=i%12)
        rows.append({'fixture_id':i,'kickoff':kickoff.isoformat(),'competition':'EPL',
                     'stage':'16+','foul_state':'available','target':3,'scores':{'candidate':a,'control':b}})
    result=q.comparison(rows,'candidate','control')
    assert result['reused_qualification_gate_passed']
    assert result['slices']['competition:EPL']['sufficient']
    assert result['candidate']['sufficient']
    assert not result['production_qualified']


def test_failed_development_prevents_qualification_access(tmp_path,monkeypatch):
    monkeypatch.setattr(q.dev,'verify_complete',lambda p:None)
    monkeypatch.setattr(q.dev,'read_json',lambda p:{'development_gates_passed':False})
    with pytest.raises(ValueError,match='No eligible'):q.prepare(tmp_path,tmp_path/'output')
    assert not (tmp_path/'output').exists()


def test_preparation_seals_after_readonly_database_phase_without_guard_violation(tmp_path,monkeypatch):
    package=tmp_path/'package';package.mkdir()
    database=tmp_path/'old.db'
    with sqlite3.connect(database) as db:
        db.executescript('''CREATE TABLE fixtures(id,api_football_id,kickoff_utc);
          CREATE TABLE fixture_player_stats(fixture_id,team_id,player_id,minutes,yellow_cards,red_cards);
          CREATE TABLE fixture_team_stats(fixture_id,team_id,fouls_committed,raw_payload_digest);
          CREATE TABLE teams(id,api_football_id); CREATE TABLE players(id,api_football_id);''')
    with sqlite3.connect(package/'platform.db') as db:
        db.executescript('''CREATE TABLE fixture_endpoint_collection(fixture_id,endpoint,archive_id);
          CREATE TABLE raw_payload_archive(id,payload_digest,fetched_at);''')
    fixture={'fixture_id':7,'platform_fixture_id':77,'code':'UCL','season':2024,'kickoff_utc':'2025-05-01T12:00:00Z',
             'status':'FT','round':'Final','home_team_id':10,'away_team_id':20}
    q.dev.write_json(package/'fixture-identities.json',[fixture]);q.dev.write_json(package/'archives.json',[])
    q.dev.write_json(package/'PREPARED.json',{p:q.dev.sha(package/p) for p in ('platform.db','fixture-identities.json','archives.json')})
    q.dev.write_json(tmp_path/'before.json',{'backup_sha256':q.dev.sha(database)})
    development=tmp_path/'development';(development/'inputs').mkdir(parents=True)
    for name in ('projections.py','weights.py','fixed-weights.json'):(development/'inputs'/name).write_text('preserved')
    (development/'inputs/targets.jsonl').write_text('');q.dev.write_json(development/'inputs/context.json',{})
    for name in ('METHOD_LOCK.json','CALIBRATION_LOCK.json'):q.dev.write_json(development/name,{'id':'test'})
    q.dev.complete(development)
    monkeypatch.setattr(q.dev,'DATABASE',database);monkeypatch.setattr(q.dev,'PACKAGE',package)
    monkeypatch.setattr(q,'check_development',lambda p:({'id':'method'},{'id':'calibration'}))
    output=tmp_path/'output';report=q.prepare(development,output)
    assert report['new_records']==1 and report['new_qualified']==0
    assert q.dev.verify_complete(output)
    assert json.loads((output/'ACCESS_OPENED.json').read_text())['reused_period'] is True
