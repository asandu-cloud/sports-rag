"""Evidence qualification and real SQLite import/rollback with synthetic sources."""
from copy import deepcopy
import json
from pathlib import Path
import sqlite3

import pytest
import requests

from Scripts.ops import player_history_download as download
from Scripts.ops import player_history_reconcile as ops
from Scripts.data_platform.features import player_history_reconciliation as research
from Scripts.tests.test_player_history_download import fixture, payload, lineup_payload, Response, Transport


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def fail(*a, **kw):
        raise AssertionError('No live requests allowed')
    monkeypatch.setattr(requests.sessions.Session, 'request', fail)


def evidence():
    p, l = payload(), lineup_payload()
    for team in p['response']:
        for row in team['players']:
            row['statistics'][0]['games']['minutes'] = 90
            row['statistics'][0]['cards'] = {'yellow': 0, 'red': 0}
    e = {'get': 'fixtures/events', 'parameters': {'fixture':'42'}, 'errors': [], 'results':1,
         'paging': {'current':1,'total':1}, 'response': [event('Goal','Normal Goal',10000)]}
    return dict(zip(download.ENDPOINTS, (p,l,e)))


def event(kind, detail, pid=10000, minute=45):
    return {'time': {'elapsed':minute,'extra':None}, 'team':{'id':100},
            'player':{'id':pid}, 'assist':{'id':None}, 'type':kind, 'detail':detail}


def target(p, **kwargs):
    p = deepcopy(p)
    for v in p.values():
        v['results'] = len(v['response'])
    return research.reconstruct(fixture(), p, {ep:{'hash':'test'} for ep in download.ENDPOINTS}, **kwargs)


def test_full_envelope_and_legacy_list_are_supported_but_wrong_fixture_rejected():
    p = evidence()[download.ENDPOINT]
    assert research.response_blocks(p,fixture(),download.ENDPOINT) == p['response']
    assert research.response_blocks(p['response'],fixture(),download.ENDPOINT) == p['response']
    p['parameters']['fixture'] = '123'
    with pytest.raises(ValueError, match='identity'):
        research.response_blocks(p,fixture(),download.ENDPOINT)


def test_negative_event_is_retained_partial_not_zero_or_fabricated_time():
    p = evidence()['/fixtures/events']
    p['response'][0]['time']['elapsed'] = -5
    assert research.classify(fixture(),p,'/fixtures/events')['state'] == 'partial'
    assert p['response'][0]['time']['elapsed'] == -5
    assert download.usable_event_time(-5) is None
    assert download.usable_event_time(0) == 0


@pytest.mark.parametrize('yellow,red,details,total', [
    (0,0,[],0),(1,0,['Yellow Card'],1),(0,1,['Red Card'],2),
    (1,1,['Yellow Card','Red Card'],3),
    (1,1,['Yellow Card','Yellow-Red Card'],3),
    (2,1,['Yellow Card','Yellow-Red Card'],3),
])
def test_weighted_cards_and_second_yellow_semantics(yellow,red,details,total):
    p = evidence()
    p[download.ENDPOINT]['response'][0]['players'][0]['statistics'][0]['cards'] = {'yellow':yellow,'red':red}
    p['/fixtures/events']['response'] += [event('Card', d, minute=50+i) for i,d in enumerate(details)]
    r = target(p)
    assert r['eligible'], r['exclusions']
    assert r['target'] == total


def test_null_red_not_filled_from_absence_of_event():
    p=evidence()
    p[download.ENDPOINT]['response'][0]['players'][0]['statistics'][0]['cards']['red']=None
    r=target(p)
    assert r['target'] is None
    assert 'missing_player_card_counts' in r['exclusions']


@pytest.mark.parametrize('kind', ['conflict','negative_time','missing_actor','duplicate'])
def test_uncertain_card_evidence_cannot_produce_target(kind):
    p=evidence()
    e=event('Card','Yellow Card')
    p['/fixtures/events']['response'].append(e)
    if kind!='conflict':
        p[download.ENDPOINT]['response'][0]['players'][0]['statistics'][0]['cards']['yellow']=1
    if kind=='negative_time': e['time']['elapsed']=-5
    if kind=='missing_actor': e['player']['id']=None
    if kind=='duplicate': p['/fixtures/events']['response'].append(deepcopy(e))
    r=target(p)
    assert not r['eligible'] and r['target'] is None


def test_one_minute_card_excluded_two_minute_card_counts():
    p=evidence()
    s=p[download.ENDPOINT]['response'][0]['players'][0]['statistics'][0]
    s['cards']['yellow']=1
    p['/fixtures/events']['response'].append(event('Card','Yellow Card'))
    s['games']['minutes']=1
    assert target(p)['target']==0
    s['games']['minutes']=2
    assert target(p)['target']==1


def test_missing_starting_participation_and_existing_conflicts_remain_pending():
    p=evidence()
    p[download.ENDPOINT]['response'][0]['players'][0]['statistics'][0]['games']['minutes']=None
    assert 'starting_player_minutes_unknown' in target(p)['exclusions']
    p=evidence()
    old=[{'player_id':10000,'team_id':100,'minutes':88,'yellow_cards':0,'red_cards':0}]
    assert 'existing_nonnull_minutes_conflict' in target(p,existing_rows=old)['exclusions']
    old[0]['minutes']=None
    assert target(p,existing_rows=old)['eligible']


def test_reserved_period_rejected_before_payload_decode():
    class Forbidden(dict):
        def __contains__(self,key):
            raise AssertionError('Reserved payload accessed')
    with pytest.raises(ValueError,match='Reserved'):
        research.reconstruct(fixture(season=2025),Forbidden(),{})


@pytest.fixture()
def prepared(tmp_path):
    root=tmp_path/'project'
    canonical=root/'Index'
    canonical.mkdir(parents=True)
    download.initialize(canonical,[fixture()],{'test':True})
    db=canonical/'platform.db'
    with sqlite3.connect(db) as d:
        d.execute("INSERT INTO players(id,api_football_id,name,created_at,updated_at) VALUES (1,10000,'Original name','old','old')")
        d.execute("INSERT INTO fixture_player_stats(id,fixture_id,team_id,player_id,minutes,yellow_cards,red_cards,stats_json,created_at,updated_at) VALUES(99,1042,10,1,88,0,0,'{}','old','old')")
        d.execute("UPDATE fixtures SET home_goals=2,away_goals=1")
        # Force archive IDs to differ between staging and canonical.
        d.execute("INSERT INTO raw_payload_archive(id,provider,endpoint,params_digest,payload_digest,storage_backend,storage_uri,fetched_at,created_at,updated_at) VALUES (1,'test','/old','x','x','local','file:///unused','old','old','old')")
    collection=root/'Index/history_staging/players'
    p=evidence()
    p['/fixtures/events']['response'][0]['time']['elapsed']=-5
    download.run(root,db,(2021,),collection,execute=True,key='test',endpoints=download.ENDPOINTS,
                 transport=Transport([Response(p[ep]) for ep in download.ENDPOINTS]))
    # Simulate the original collector's rejection, preserving the raw response.
    with sqlite3.connect(collection/'platform.db') as d:
        d.execute("UPDATE fixture_endpoint_collection SET state='invalid',error='Invalid event time' WHERE endpoint='/fixtures/events'")
        d.execute('DELETE FROM fixture_events')
    original=ops.sha(collection/'platform.db')
    package=root/'prepared'
    report=ops.prepare(collection,package)
    assert report['repairs']=={'invalid->partial':1}
    assert ops.sha(collection/'platform.db')==original
    return root,db,package


def test_import_remaps_ids_preserves_conflicts_and_is_idempotent(prepared):
    root,db,package=prepared
    backup=root/'before.db'
    ops.sqlite_backup(db,backup)
    batch=ops.verify(package)
    report=ops.import_transaction(db,package,batch,root/'archives')
    assert report['player_rows_added']==21
    assert report['existing_differing_player_rows_retained']==1
    assert ops.preserved_rows(db,backup)>0
    with sqlite3.connect(db) as d:
        assert d.execute('select minutes from fixture_player_stats where id=99').fetchone()==(88,)
        assert d.execute('select name from players where id=1').fetchone()==('Original name',)
        assert d.execute('select home_goals,away_goals from fixtures').fetchone()==(2,1)
        assert d.execute('select count(*) from player_history_observations').fetchone()==(22,)
        row=d.execute('select elapsed,raw_json from player_history_events').fetchone()
        assert row[0] is None and json.loads(row[1])['time']['elapsed']==-5
        assert d.execute("select count(*) from fixture_player_stats x join raw_payload_archive a on a.id=json_extract(x.stats_json,'$.source_archive_id') where a.endpoint='/fixtures/players'").fetchone()==(21,)
        assert not d.execute('pragma foreign_key_check').fetchall()
        counts=d.execute('select count(*) from raw_payload_archive').fetchone()
    assert ops.import_transaction(db,package,batch,root/'archives')['already_imported']
    with sqlite3.connect(db) as d:
        assert d.execute('select count(*) from raw_payload_archive').fetchone()==counts


def test_failed_import_rolls_back_data_and_schema(prepared):
    root,db,package=prepared
    backup=root/'before.db'
    ops.sqlite_backup(db,backup)
    def fail(_): raise RuntimeError('injected failure')
    with pytest.raises(RuntimeError,match='injected'):
        ops.import_transaction(db,package,ops.verify(package),root/'archives',fault=fail)
    assert ops.preserved_rows(db,backup)>0
    with sqlite3.connect(db) as d:
        assert d.execute('select count(*) from players').fetchone()==(1,)
        assert not d.execute("select name from sqlite_master where name like 'player_history_%'").fetchall()


def test_identity_conflict_blocks_import_and_archive_tampering_blocks_verification(prepared):
    root,db,package=prepared
    with sqlite3.connect(db) as d:
        d.execute("UPDATE fixtures SET round='Final'")
    with pytest.raises(ValueError,match='identity'):
        ops.import_transaction(db,package,ops.verify(package),root/'archives')
    next((package/'raw_archive').rglob('*.gz')).write_bytes(b'broken')
    with pytest.raises(ValueError,match='archive'):
        ops.verify(package)


def test_reconstruction_creates_new_artifact_without_import_or_backtest(prepared):
    root,db,package=prepared
    before=ops.sha(db)
    report=ops.reconstruct(package,db,root/'targets')
    assert report['counts']=={'pending_or_excluded':1}  # Existing 88 vs incoming 90 minutes.
    assert not report['backtests_run'] and not report['weights_optimized']
    assert ops.sha(db)==before
