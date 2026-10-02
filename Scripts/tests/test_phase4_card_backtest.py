"""Sealed-source bridge, chronology, fixed reference and offline replay checks."""
from copy import deepcopy
import inspect
import json
from pathlib import Path
import socket
import sqlite3

import pytest

from Scripts.ops import phase4_card_backtest as cli
from Scripts.data_platform.features.benchmarks.artifacts import complete, write_json
from Scripts.data_platform.features.benchmarks.isolation import offline_guard
from Scripts.tests.test_phase4_card_reconstruction import weights


BATCH = 'a' * 64


def target(fid=1, kickoff='2023-09-01T12:00:00Z'):
    row = dict(fixture_id=fid, competition='EPL', season=int(kickoff[:4]), kickoff=kickoff,
               home_team_id=10, away_team_id=20, status='FT', round='Regular Season - 10',
               referee='A. Referee, England', availability='assumed_final',
               reconstruction_version='player-history-reconciliation.v1',
               policy=deepcopy(cli.policy.POLICY), eligible=True, exclusions=[])
    players = [dict(team_id=tid, player_id=tid * 100 + i, minutes=90, yellow_cards=int(i == 0), red_cards=0)
               for tid in (10, 20) for i in range(11)]
    evidence = cli.model.evidence(row, players, minimum_recorded_minutes=2)
    row.update(player_evidence=evidence, team_targets=evidence['totals'], target=2,
               source_references={
                   endpoint: dict(endpoint=endpoint, archive_id=i + 1, fixture_id=fid + 5000,
                                  fetched_at='2026-09-30T12:00:00Z', payload_digest=str(i) * 64)
                   for i, endpoint in enumerate(('/fixtures/players', '/fixtures/lineups', '/fixtures/events'))})
    return row


def test_adapter_preserves_explicit_zero_source_namespace_and_unresolved_rows():
    known = target()
    excluded = target(2) | {'eligible': False, 'target': None, 'team_targets': None,
                            'exclusions': ['missing_player_card_counts']}
    original = deepcopy([known, excluded])
    result = cli.adapt_targets(original, BATCH)
    assert original == [known, excluded]
    assert result[0]['raw_reference']['fixture_id'] == 5001
    assert result[0]['fixture_id'] == 1
    assert result[0]['raw_reference']['batch'] == BATCH
    assert result[0]['raw_reference']['namespace'] == 'prepared_player_history'
    assert result[0]['player_evidence']['players'][1]['red'] == 0
    assert cli.model.qualified_rows(result, minimum_recorded_minutes=2) == result[:1]
    assert result[1]['target'] is None


@pytest.mark.parametrize('damage', ['policy', 'null_red', 'weighted', 'total', 'reference', 'source_fixture',
                                  'excluded_total', 'duplicate', 'scope', 'identity', 'future'])
def test_inconsistent_or_unqualified_evidence_cannot_enter_reference(damage):
    row = target()
    rows = [row]
    if damage == 'policy':
        row['policy']['minimum_recorded_minutes'] = 1
    elif damage == 'null_red':
        row['player_evidence']['players'][0]['red'] = None
    elif damage == 'weighted':
        row['player_evidence']['players'][0]['weighted_cards'] = 3
    elif damage == 'total':
        row['target'] = 10
    elif damage == 'reference':
        del row['source_references']['/fixtures/events']
    elif damage == 'source_fixture':
        row['source_references']['/fixtures/events']['fixture_id'] = 999
    elif damage == 'excluded_total':
        row.update(eligible=False, exclusions=['pending'])
    elif damage == 'duplicate':
        rows.append(deepcopy(row))
    elif damage == 'scope':
        row.update(competition='UCL', round='Final')
    elif damage == 'identity':
        row['away_team_id'] = 10
    elif damage == 'future':
        row['kickoff'] = '2024-01-01T12:00:00Z'
    with pytest.raises(ValueError):
        cli.adapt_targets(rows, BATCH)


def test_evidence_order_does_not_change_qualification():
    row = target()
    row['player_evidence']['players'].reverse()
    assert cli.adapt_targets([row], BATCH)[0]['target'] == 2


def test_later_outcome_is_rejected_before_decoding(tmp_path, monkeypatch):
    path = tmp_path / 'targets.jsonl'
    path.write_text(json.dumps({'kickoff': '2025-08-01T12:00:00Z', 'target': 'PROTECTED_OUTCOME'}) + '\n')
    decode = json.loads
    def guarded(value, *args, **kwargs):
        assert 'PROTECTED_OUTCOME' not in value
        return decode(value, *args, **kwargs)
    monkeypatch.setattr(json, 'loads', guarded)
    with pytest.raises(ValueError, match='before decoding'):
        cli.read_targets(path)


def test_referee_names_are_never_merged_by_surname():
    rows = [target(), target(2) | {'referee': 'Alex Referee'}, target(3) | {'referee': None}]
    audit = cli.referee_audit(rows)
    assert len(audit['names']) == 3
    assert audit['possible_fragmentation'][0]['separate_keys'] == ['a. referee', 'alex referee']


def test_v2_future_and_same_day_results_do_not_change_earlier_inputs():
    past = [target(i + 1, f'2023-08-{i+1:02}T12:00:00Z') for i in range(8)]
    forecast = target(9)
    future = target(10, '2023-10-01T12:00:00Z')
    # Equality at the label cutoff is excluded, as is the forecast's own result.
    at_cutoff = target(11, '2023-08-31T21:00:00Z')
    same_day = target(12, '2023-09-01T01:00:00Z')
    rows = cli.adapt_targets([*past, forecast, future, at_cutoff, same_day], BATCH)
    first = cli.model.dated_inputs(forecast, rows, minimum_recorded_minutes=2)
    for row in rows[8:]:
        row.update(target=99, team_targets={'home': 50, 'away': 49})
    assert first == cli.model.dated_inputs(forecast, list(reversed(rows)), minimum_recorded_minutes=2)
    assert first['source_fixture_ids'] == list(range(1, 9))
    assert cli.model.fixed_prediction(forecast, first, weights())[1] is None


def test_european_group_scope_still_needs_eight_current_matches():
    past = [target(i + 1, f'2023-08-{i+1:02}T12:00:00Z') | {'competition': 'UCL', 'round': 'Group Stage - 1'}
            for i in range(6)]
    forecast = target(7) | {'competition': 'UCL', 'round': 'Group Stage - 6'}
    inputs = cli.model.dated_inputs(forecast, cli.adapt_targets(past, BATCH), minimum_recorded_minutes=2)
    assert cli.model.fixed_prediction(forecast, inputs, weights()) == (None, 'insufficient_current_team_history')


def test_excluded_forecasts_keep_their_dated_inputs_and_eight_match_boundary():
    rows = cli.adapt_targets([target(i + 1, f'2023-08-{i+1:02}T12:00:00Z') for i in range(9)], BATCH)
    forecast = rows[-1]
    inputs = cli.model.dated_inputs(forecast, rows, minimum_recorded_minutes=2)
    prediction, reason = cli.model.fixed_prediction(forecast, inputs, weights())
    assert reason is None
    snapshot_id = cli.model.digest(inputs)
    snapshots = [{'fixture_id': forecast['fixture_id'], 'snapshot_id': snapshot_id, **inputs}]
    predictions = [{**forecast, **prediction, 'snapshot_id': snapshot_id}]
    ledger, coverage, _ = cli.diagnostics(rows, predictions, snapshots, weights(), {})
    assert len(snapshots) == 9 and coverage['dated_input_snapshots'] == 9
    assert [r['scored'] for r in ledger] == [False] * 8 + [True]
    assert ledger[7]['current_team_counts'] == {'home': 7, 'away': 7}
    assert ledger[7]['forecast_exclusion'] == 'insufficient_current_team_history'
    assert ledger[8]['current_team_counts'] == {'home': 8, 'away': 8}
    assert snapshots[0]['snapshot_id'] == snapshot_id
    assert coverage['league_calendar_year']['EPL:2023']['scored']['sufficient'] is False


def make_artifacts(tmp_path):
    targets, frozen = tmp_path / 'targets', tmp_path / 'frozen'
    targets.mkdir()
    write_json(targets / 'rules.json', {'card_policy': cli.policy.POLICY,
                                      'development_end_exclusive': cli.cards.END.isoformat()})
    write_json(targets / 'report.json', {'prepared_batch': BATCH, 'counts': {'qualified': 1}})
    (targets / 'card-targets.jsonl').write_bytes(b''.join(cli.lines_bytes([target()])))
    complete(targets)
    source = frozen / 'source/Scripts/data_platform/features/phase4_card_reconstruction.py'
    source.parent.mkdir(parents=True)
    source.write_text(inspect.getsource(cli.model))
    write_json(frozen / 'fixed-weights.json', weights())
    complete(frozen)
    return targets, frozen


def test_sealed_run_and_replay_explicitly_block_insufficient_support(tmp_path):
    targets, frozen = make_artifacts(tmp_path)
    output = tmp_path / 'run'
    report = cli.run(targets, frozen, output)
    assert report['status'] == 'blocked_evidence_or_support' and report['metrics'] is None
    assert cli.replay(output)['status'] == 'exact_offline_replay'
    with pytest.raises(FileExistsError):
        cli.run(targets, frozen, output)
    (output / 'predictions.jsonl').write_text('tampered')
    with pytest.raises(ValueError, match='checksum'):
        cli.replay(output)


def test_tampered_inputs_and_changed_reference_are_rejected_before_output(tmp_path):
    targets, frozen = make_artifacts(tmp_path)
    original = (targets / 'card-targets.jsonl').read_bytes()
    (targets / 'card-targets.jsonl').write_text('bad')
    with pytest.raises(ValueError, match='checksum'):
        cli.run(targets, frozen, tmp_path / 'first')
    assert not (tmp_path / 'first').exists()
    (targets / 'card-targets.jsonl').write_bytes(original)
    source = frozen / 'source/Scripts/data_platform/features/phase4_card_reconstruction.py'
    source.write_text(source.read_text().replace('shift = min(.30,', 'shift = min(.40,'))
    (frozen / 'COMPLETE.json').unlink()
    complete(frozen)
    with pytest.raises(ValueError, match='formulas'):
        cli.run(targets, frozen, tmp_path / 'second')
    assert not (tmp_path / 'second').exists()


def test_run_guard_blocks_databases_network_and_external_writes(tmp_path):
    output = tmp_path / 'experiment'
    output.mkdir()
    with offline_guard(root=tmp_path, output=output):
        with pytest.raises(RuntimeError, match='sqlite3'):
            sqlite3.connect(':memory:')
        with pytest.raises(RuntimeError, match='socket'):
            socket.socket()
        with pytest.raises(RuntimeError, match='outside experiment'):
            (tmp_path / 'production-profile.json').write_text('no')
