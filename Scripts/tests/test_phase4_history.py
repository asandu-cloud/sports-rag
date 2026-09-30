"""Research boundary, chronological reconstruction and frozen numerical parity."""
from copy import deepcopy
from datetime import timedelta
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from Scripts.data_platform.features import phase4_history as h
from Scripts.ops.phase4_historical_worker import check_probabilities
from Scripts.ops.phase4_history_audit import validate_history_membership

ROOT = Path(__file__).resolve().parents[2]
FROZEN = ROOT/'Research/phase4-baseline-2026-09-28/control/workspace'


def row(fid=1, kickoff='2022-08-01T15:00:00Z', season=2022, competition='EPL'):
    stats = dict(goals=1., corners=4., sot=3., shots=10., fouls=12., xg=1.1, possession=50., cards=None)
    return dict(fixture_id=fid, kickoff=kickoff, season=season, competition=competition,
                home_team_id=11, away_team_id=12, status='FT', observed_at='2026-09-25T00:00:00Z',
                home=deepcopy(stats), away=deepcopy(stats))


def test_reserved_object_is_not_json_decoded(monkeypatch):
    allowed = row()
    # Invalid JSON value in a reserved record proves lexical skip precedes decoding.
    text = '{"history":['+json.dumps(allowed)+',{"kickoff":"2025-01-01T00:00:00Z","secret":INVALID}]}'
    kept, skipped = h.development_history(text)
    assert kept == [allowed] and skipped == 1


@pytest.mark.parametrize('text', [
    '{"history":[{"kickoff":"2022-01-01Z","kickoff":"2025-01-01Z"}]}',
    '{"history":[', '{"history":[42]}', '{"history":[],"history":[]}'])
def test_ambiguous_or_truncated_archive_rejected(text):
    with pytest.raises(ValueError):
        h.development_history(text)


def test_duplicate_and_non_regulation_rejected():
    for rows in ([row(), row()], [dict(row(), status='AET')]):
        with pytest.raises(ValueError):
            h.development_history(json.dumps({'history': rows}))


def test_future_and_same_day_results_cannot_change_earlier_inputs():
    earlier = row(1, '2022-08-01T15:00:00Z')
    future = row(2, '2022-08-10T10:00:00Z')
    cutoff = '2022-08-10T20:00:00Z'
    before = h.HistoricalInputs([earlier, future])
    future['home'].update(goals=99., xg=99., corners=99.)
    after = h.HistoricalInputs([earlier, future, row(3, '2022-09-01T10:00:00Z')])
    assert before.rows(11, 'EPL', cutoff) == after.rows(11, 'EPL', cutoff)
    assert h.digest(before.evidence(11, 'EPL', cutoff, 2022)) == h.digest(after.evidence(11, 'EPL', cutoff, 2022))


def test_previous_day_result_still_requires_three_hour_availability():
    index = h.HistoricalInputs([row(1, '2022-08-01T23:00:00Z')])
    assert not index.rows(11, 'EPL', '2022-08-02T01:00:00Z')
    assert not index.rows(11, 'EPL', '2022-08-02T02:00:00Z')
    assert index.rows(11, 'EPL', '2022-08-02T02:00:01Z')


def test_nulls_zeros_exact_ids_and_prior_gap_are_preserved():
    zero, missing = row(1, '2020-08-01T15:00:00Z', 2020), row(2)
    zero['home']['corners'] = 0.
    missing['home']['corners'] = None
    index = h.HistoricalInputs([zero, missing])
    assert index.metas[('11', '1')]['meta']['corners_for'] == 0
    assert index.metas[('11', '2')]['meta']['corners_for'] is None
    assert not index.rows(111, 'EPL', '2023-01-01T00:00:00Z')
    evidence = index.evidence(11, 'EPL', '2023-01-01T00:00:00Z', 2022)
    assert evidence['prior_rank'] == 2020
    assert evidence['current']['field_counts']['corners_for'] == 0
    assert evidence['prior']['field_counts']['corners_for'] == 1


def test_domestic_membership_is_dated_not_future():
    earlier = row(1, '2022-08-01T15:00:00Z', competition='EPL')
    moved = row(2, '2023-08-01T15:00:00Z', 2023, 'LaLiga')
    index = h.HistoricalInputs([moved, earlier])
    assert index.domestic(11, '2022-09-01T00:00:00Z') == 'EPL'
    assert index.domestic(11, '2023-09-01T00:00:00Z') == 'LaLiga'


def test_input_order_does_not_change_membership_or_hashes():
    rows = [row(3), row(2), row(1)]
    a, b = h.HistoricalInputs(rows), h.HistoricalInputs(list(reversed(rows)))
    assert h.digest(a.evidence(11, 'EPL', '2022-08-03T00:00:00Z', 2022)) == h.digest(b.evidence(11, 'EPL', '2022-08-03T00:00:00Z', 2022))


def test_fitting_and_development_scopes_never_split_fixture_or_include_reserves():
    history, rows = [], []
    for year in (2021, 2022, 2023, 2024, 2025):
        f = row(year, f'{year}-08-01T15:00:00Z', year)
        if year < 2024:
            history.append(f)
        labels = {m: (None if m == 'cards' else f['home'][m]+f['away'][m]) for m in ('goals','corners','sot','cards')}
        rows.append(dict(fixture=f, as_of=f['kickoff'], snapshot_id=str(year), forecast_stage='reconstructed_immediately_before_kickoff',
            market_eligibility={m:{'eligible':m!='cards','reasons':[]} for m in labels}, labels=labels,
            team_labels={side:{m:f[side][m] for m in labels} for side in ('home','away')},
            label_available_at=(h.utc(f['kickoff'])+timedelta(hours=3)).isoformat(), actual_observed_at=f['observed_at']))
    dataset = SimpleNamespace(rows=rows)
    initial,_ = h.request_rows(dataset, history, scope='earlier_fitting')
    development,_ = h.request_rows(dataset, history)
    assert [r['fixture']['fixture_id'] for r in initial] == [2021]
    assert [r['fixture']['fixture_id'] for r in development] == [2022,2023]
    with pytest.raises(ValueError):
        h.request_rows(dataset, history, scope='calibration')


def test_final_audit_catches_future_or_wrong_identity_references():
    source = row()
    index = h.HistoricalInputs([source])
    snapshot = {'as_of':'2022-08-03T15:00:00Z','history_evidence':{
        '11':{'EPL':index.evidence(11,'EPL','2022-08-03T15:00:00Z',2022)}}}
    validate_history_membership(snapshot,{1:source})
    for changed in (dict(source,kickoff='2022-08-03T16:00:00Z'), dict(source,home_team_id=999),
                    dict(source,competition='UCL'), dict(source,season=2021)):
        with pytest.raises(ValueError):
            validate_history_membership(snapshot,{1:changed})


@pytest.mark.parametrize('values', [[.5,.6], [-.1,1.1], [float('nan'),1.], [], [float('inf')]])
def test_invalid_distributions_rejected(values):
    with pytest.raises(ValueError):
        check_probabilities(values)


def make_request(fid, kickoff, competition='EPL'):
    f = row(fid, kickoff, competition=competition)
    return {'fixture': {k:f[k] for k in ('fixture_id','competition','season','home_team_id','away_team_id','kickoff')},
            'as_of': kickoff, 'availability':'assumed_final', 'source_snapshot_id':'synthetic-'+str(fid),
            'forecast_stage':'reconstructed_immediately_before_kickoff',
            'source_eligibility': {m: {'eligible':m!='cards','reasons':[] if m!='cards' else ['cards_target_not_qualified']}
                                   for m in ('goals','corners','sot','cards')}}


def invoke_worker(tmp_path, history, requests, name):
    prepared = tmp_path/name
    prepared.mkdir()
    for file, rows in [('history',history),('requests',requests)]:
        (prepared/(file+'.jsonl')).write_text(''.join(h.encode(r)+'\n' for r in rows))
    run = subprocess.run([sys.executable,'-B',str(ROOT/'Scripts/ops/phase4_historical_worker.py'),
        '--source',str(FROZEN),'--prepared',str(prepared),'--output',str(prepared/'output')],
        capture_output=True, text=True, timeout=120,
        env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'})
    assert run.returncode == 0, run.stderr
    return prepared/'output'


@pytest.mark.skipif(not FROZEN.exists(), reason='local frozen control required')
def test_frozen_arithmetic_replay_future_invariance_and_asian_coherence(tmp_path):
    history = [row(i+1,(h.utc('2022-08-01T15:00:00Z')+timedelta(days=i*7)).isoformat()) for i in range(10)]
    history += [row(50+i, f'2021-09-{i+1:02d}T15:00:00Z', 2021) for i in range(10)]
    history += [row(80+i, f'2022-08-{i+1:02d}T15:00:00Z', competition='UCL') for i in range(3)]
    requests = [make_request(101,'2022-09-18T20:00:00Z'), make_request(102,'2022-09-25T20:00:00Z'),
                make_request(103,'2022-08-04T20:00:00Z','UCL'), make_request(104,'2022-07-01T20:00:00Z','UCL')]
    a = invoke_worker(tmp_path,history,requests,'first')
    changed = deepcopy(history)
    for r in changed:
        if h.utc(r['kickoff']) > h.utc('2022-09-25T20:00:00Z'):
            r['home'].update(goals=50.,corners=100.,sot=50.,xg=30.)
    b = invoke_worker(tmp_path,list(reversed(changed)),requests,'second')
    for file in ('snapshots.jsonl','control.jsonl','eligibility.jsonl'):
        assert (a/file).read_bytes() == (b/file).read_bytes()
    report = json.loads((a/'report.json').read_text())
    assert report['exact_serialized_replays'] == 4 and not report['io_violations']
    snapshots = [json.loads(line) for line in (a/'snapshots.jsonl').open()]
    assert snapshots[0]['profile_quality']['11']['prior_weight'] > 0
    assert snapshots[1]['profile_quality']['11']['prior_weight'] == 0
    assert snapshots[2]['profile_quality']['11']['weights'] == {'domestic':.8,'european':.2}
    assert snapshots[3]['recent']['11']['n'] == 0
    predictions = list(map(json.loads,(a/'control.jsonl').open()))
    # Independent arithmetic on a constant-rate fixture, including variance floors.
    known = predictions[1]
    assert known['goal_means'] == pytest.approx([1.015, 1.015])
    assert known['distributions']['corners']['mean'] == pytest.approx(8.)
    assert known['distributions']['corners']['variance'] == pytest.approx(12.)
    assert known['distributions']['sot']['mean'] == pytest.approx(6.)
    assert known['distributions']['sot']['variance'] == pytest.approx(9.6)
    assert known['distributions']['corners']['pmf'][0] == pytest.approx((2/3)**16)
    p00 = next(p for home,away,p in known['score_distribution'] if home == away == 0)
    assert p00 == pytest.approx(math.exp(-2.03)*(1+.1*1.015**2), abs=1e-10)
    assert known['diagnostics']['goals']['2.25']['Over']['half_loss'] == pytest.approx(known['distributions']['goals']['pmf'][2])
    for prediction in predictions:
        if not prediction['score_distribution']:
            continue
        check_probabilities(p for _,_,p in prediction['score_distribution'])
        d = prediction['diagnostics']
        check_probabilities(d['winner'].values()); check_probabilities(d['btts'].values())
        for market in ('goals','corners','sot'):
            totals = d[market]
            over = []
            for line, outcomes in sorted(totals.items(),key=lambda item:float(item[0])):
                o,u = outcomes['Over'],outcomes['Under']
                assert o['full_win'] == pytest.approx(u['full_loss'])
                assert o['half_win'] == pytest.approx(u['half_loss'])
                assert o['push'] == pytest.approx(u['push'])
                over.append(o['full_win']+.5*o['half_win'])
            assert over == sorted(over,reverse=True)
        for outcome in d['handicaps'].values():
            check_probabilities(outcome.values())


@pytest.mark.parametrize('operation', [
    "socket.getaddrinfo('localhost',80)", "sqlite3.connect(':memory:')", "Path('/tmp/forbidden.db').read_bytes()",
    "Path('/tmp/.env').read_bytes()", "subprocess.run(['true'])", "Path('/tmp/unapproved-spix-write').write_text('x')"])
def test_io_guard_rejects_forbidden_operations(tmp_path, operation):
    script = f'''from pathlib import Path
import socket, sqlite3, subprocess
from Scripts.ops.phase4_historical_worker import install_guard
violations=install_guard(Path({str(FROZEN)!r}),Path({str(tmp_path)!r}))
try:
    {operation}
except RuntimeError:
    pass
else:
    raise AssertionError('Forbidden operation succeeded')
assert violations
'''
    result = subprocess.run([sys.executable,'-B','-c',script],cwd=ROOT,capture_output=True,text=True,timeout=30)
    assert result.returncode == 0, result.stderr
