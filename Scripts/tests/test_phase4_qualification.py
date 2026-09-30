"""Qualification access, chronological state, fixed fitting and multiplicity."""
from copy import deepcopy
from datetime import timedelta
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import pytest

from Scripts.data_platform.features import phase4_qualification as q,phase4_candidates as cm
from Scripts.data_platform.features import phase4_history as h,phase4_finalists as f,phase4_calibration as c
from Scripts.data_platform.features.benchmarks.confirmation_data import history_before,COMPETITIONS


def historical(fid=1,kickoff='2024-08-01T12:00:00Z'):
    return {'fixture_id':fid,'kickoff':kickoff,'season':2024,'competition':'EPL','home_team_id':11+fid%4,
        'away_team_id':11+(fid+1)%4,'status':'FT',
        'home':{'goals':fid%4,'corners':0.,'sot':2.,'shots':4.,'xg':None,'possession':50.,'fouls':8.},
        'away':{'goals':(fid+2)%3,'corners':None,'sot':3.,'shots':5.,'xg':1.,'possession':50.,'fouls':9.}}


def histories():
    return [historical(i+1,(cm.utc('2024-08-01T12:00:00Z')+timedelta(days=i)).isoformat()) for i in range(90)]


def test_bounded_mixed_archive_skips_later_payload_before_decoding():
    first=historical();qualification=historical(2,'2025-03-01T12:00:00Z')
    text='{"competitions":'+json.dumps(COMPETITIONS)+',"fixtures":[{"outcome":INVALID}],"history":['+json.dumps(first)+','+json.dumps(qualification)+',{"kickoff":"2025-07-01T12:00:00Z","outcome":INVALID}]}'
    rows,_,audit=history_before(text,q.START)
    assert rows==[first] and audit['opaque_later_history_rows']==2
    rows,_,audit=history_before(text,q.END)
    assert rows==[first,qualification] and audit['opaque_later_history_rows']==1
    assert not audit['fixtures_array_decoded'] and not audit['later_outcomes_decoded']


def test_qualification_fit_is_deterministic_and_rejects_future_payloads():
    rows=histories();ids={r['fixture_id'] for r in rows}
    a=q.fit_strength(rows,ids,'EPL');b=q.fit_strength(list(reversed(rows)),ids,'EPL')
    assert a==b and a['fixture_count']==90 and a['season']==2024 and a['cutoff']==q.START
    q.check_bundle(a)
    with pytest.raises(ValueError,match='Qualification label'):
        q.fit_strength(rows+[historical(999,'2025-01-01T12:00:00Z')],ids,'EPL')
    with pytest.raises(ValueError,match='scope'):q.fit_strength(rows,ids,'UCL')
    a['ridge']=1.
    with pytest.raises(ValueError,match='frozen'):q.check_bundle(a)


def test_original_development_fit_and_inference_still_reject_reserves():
    rows=histories();ids={r['fixture_id'] for r in rows}
    with pytest.raises(ValueError,match='Reserved'):
        cm.fit_strength(rows,eligible_ids=ids,competition='EPL',market='goals',cutoff=q.START,season=2024)
    bundle=q.fit_strength(rows,ids,'EPL')
    fixture={**historical(999),'kickoff':'2025-01-02T12:00:00Z'}
    with pytest.raises(ValueError,match='reserved'):cm.predict_strength(bundle,fixture,fixture['kickoff'])
    assert cm.predict_strength(bundle,fixture,fixture['kickoff'],end=cm.utc(q.END))['status']=='available'
    fixture['season']=2025
    with pytest.raises(ValueError,match='season mismatch'):cm.predict_strength(bundle,fixture,fixture['kickoff'],end=cm.utc(q.END))


def test_future_and_same_day_outcomes_cannot_change_earlier_qualification_state():
    earlier=historical(1,'2025-01-01T12:00:00Z');future=historical(2,'2025-01-10T10:00:00Z')
    cutoff='2025-01-10T20:00:00Z';end=cm.utc(q.END)
    a=h.HistoricalInputs([earlier,future],end=end)
    future['home'].update(goals=999.,xg=999.)
    b=h.HistoricalInputs([earlier,future,historical(3,'2025-02-01T12:00:00Z')],end=end)
    assert a.rows(12,'EPL',cutoff)==b.rows(12,'EPL',cutoff)
    assert a.evidence(12,'EPL',cutoff,2024)==b.evidence(12,'EPL',cutoff,2024)
    assert a.metas[('12','1')]['meta']['corners_for']==0.
    assert a.metas[('13','1')]['meta']['corners_for'] is None
    with pytest.raises(ValueError):h.HistoricalInputs([earlier])
    with pytest.raises(ValueError):h.HistoricalInputs([historical(99,q.END)],end=end)


def test_preflight_cannot_borrow_final_system_rows():
    rows=[]
    for i in range(500):
        rows.append({'fixture_id':i+1,'competition':'EPL','kickoff':(cm.utc(q.START)+timedelta(weeks=i%20)).isoformat(),
            'partition':'calibration','eligible_markets':list(f.MARKETS)})
    report=q.preflight(rows)
    assert report['supported_league_upper_bounds']['goals']['n']==500
    assert q.preflight(rows+[{'fixture_id':9999,'kickoff':q.END}])==report
    with pytest.raises(ValueError,match='Insufficient'):q.preflight(rows[1:]+[{'fixture_id':9999,'kickoff':q.END}])
    with pytest.raises(ValueError,match='duplicate'):q.preflight(rows+[rows[0]])
    rows[0]['target']=4
    with pytest.raises(ValueError,match='Outcomes'):q.preflight(rows)


@pytest.mark.parametrize('date',['2024-12-31T23:59:59Z',q.END,'2026-01-01T00:00:00Z'])
def test_qualification_boundaries_reject_other_periods(date):
    with pytest.raises(ValueError,match='interval'):q.qualification_time(date)


def test_holm_keeps_all_four_claims_and_prevents_unadjusted_pass():
    corrected=q.holm(dict(zip(q.CLAIMS,[.001,.02,.03,1.])))
    assert corrected['goals']['adjusted_p']==.004 and corrected['goals']['passes']
    assert corrected['corners']['adjusted_p']==pytest.approx(.06) and not corrected['corners']['passes']
    with pytest.raises(ValueError,match='four'):q.holm({'goals':.001})


def test_paired_week_resampling_is_repeatable_and_handles_null():
    rows=[{'week':str(i%20)} for i in range(600)]
    a=q.paired_inference(rows,[-.1]*len(rows))
    assert a==q.paired_inference(rows,[-.1]*len(rows))
    assert a['paired_week_delta95'][1]<0 and a['one_sided_p']==pytest.approx(1/2001)
    null=q.paired_inference(rows,[0.]*len(rows))
    assert null['one_sided_p']==1 and null['paired_week_delta95']==[0.,0.]


def test_fixed_development_calibrator_cannot_be_refitted_on_reserve_or_substituted():
    with pytest.raises(ValueError,match='scope'):c.validate_training([],q.START,'corners','dispersion_corners:0.025')
    with pytest.raises(ValueError,match='substitution'):
        q.calibrated_corner({'id':'a different fit'},{'as_of':q.START},.5)


def test_qualification_composition_rejects_outside_dates_and_preserves_scope():
    snapshot={'fixture':{'fixture_id':1,'competition':'UCL'},'snapshot_id':'saved','as_of':q.START,
        'profile_quality':{'1':{'current_season_matches':20},'2':{'current_season_matches':20}}}
    control={'fixture_id':1,'input_snapshot_id':'saved','goal_means':[1.,1.],
        'means':{'goals':2.,'corners':10.,'sot':8.},'variances':{'corners':20.,'sot':16.},'evidence':{'fallbacks':[]}}
    strength=deepcopy(control);strength['goal_means']=[5.,5.]
    r=q.compose(snapshot,control,strength)
    assert r['goal_means']==control['goal_means'] and r['variances']==control['variances']
    assert not r['publication_enabled'] and r['joint_cross_market_probability'] is None
    with pytest.raises(ValueError):f.compose(snapshot,control,strength,'supported_stack')
    snapshot['as_of']=q.END
    with pytest.raises(ValueError):q.compose(snapshot,control,strength)


def test_dataset_reader_refuses_final_test_before_any_file_access(tmp_path):
    from Scripts.ops.phase4_qualification import dataset_read
    with pytest.raises(ValueError,match='Forbidden'):dataset_read(tmp_path,'lockbox/final_system_test/labels.jsonl')
    with pytest.raises(ValueError,match='opening'):dataset_read(tmp_path,'lockbox/calibration/labels.jsonl')
    with pytest.raises(ValueError,match='opening'):dataset_read(tmp_path,'audit/inputs.json')
    assert not list(tmp_path.iterdir())


def test_guarded_prediction_worker_before_reserve_access(tmp_path):
    """Shift a known development snapshot as a synthetic fixture, never a backtest.

    Exercise the actual archived engine, later-date adapter and frozen sigmoid
    together before opening any real 2025 input. Targets are deliberately absent.
    """
    root=Path(__file__).resolve().parents[2]
    from Scripts.ops.phase4_qualification import SOURCES,CAL,BASE
    for name in SOURCES:
        dest=tmp_path/'implementation'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(root/name,dest)
    folder=root/'Research/phase4-step2-backtest-2026-09-29/folds/2023-Q1'
    with (folder/'snapshots.jsonl').open() as stream:
        original=next(r for r in map(json.loads,stream) if r['fixture']['competition']=='EPL')
    fid=original['fixture']['fixture_id'];snapshot=deepcopy(original)
    snapshot['as_of']='2025-02-01T12:00:00+00:00'
    snapshot['fixture'].update(kickoff=snapshot['as_of'],season=2024)
    snapshot['snapshot_id']=f.digest({k:v for k,v in snapshot.items() if k!='snapshot_id'})
    with (folder/'control-rates.jsonl').open() as stream:rates=next(r for r in map(json.loads,stream) if r['fixture_id']==fid)
    control={'fixture_id':fid,'goal_means':rates['goal_means'],
        'results':[{'market':{'group':m},'projection':{'value':rates['means'][m],
                    'variance':rates['variances'].get(m)}} for m in f.MARKETS]}
    reproduction=tmp_path/'reproduction';reproduction.mkdir()
    for name,row in [('snapshots',snapshot),('control',control),('eligibility',{'fixture_id':fid,'markets':{m:{'eligible':True} for m in f.MARKETS}})]:
        (reproduction/(name+'.jsonl')).write_text(f.encode(row)+'\n')
    rows=histories();bundle=q.fit_strength(rows,{r['fixture_id'] for r in rows},'EPL')
    (tmp_path/'strength-bundles.json').write_text(f.encode({'EPL':bundle}))
    shutil.copyfile(CAL/'tuning'/(q.CALIBRATOR_ID+'.json'),tmp_path/'corner-calibrator.json')
    for name in ('METHOD_LOCK.json','INPUT_LOCK.json'):(tmp_path/name).write_text('{}')
    proc=subprocess.run([sys.executable,'-B',str(tmp_path/'implementation/Scripts/ops/phase4_qualification_worker.py'),
        '--source',str(BASE/'workspace'),'--prepared',str(tmp_path),'--action','predict'],cwd=root,
        env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'},capture_output=True,text=True,timeout=90)
    assert proc.returncode==0,proc.stdout+'\n'+proc.stderr
    report=json.loads((tmp_path/'predictions/report.json').read_text())
    assert report['control_parity']==1 and not report['target_store_decoded'] and not report['io_violations']
    assert not (tmp_path/'targets.jsonl').exists()
