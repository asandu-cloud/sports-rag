"""Independent no-fit budget, isolation and final-cutoff integration checks."""
from copy import deepcopy
import random
import socket
import sqlite3
from threading import Thread

import numpy as np
import pytest

from Scripts.data_platform.features.benchmarks import extension
from Scripts.data_platform.features.benchmarks.checkpoints import Checkpoints, RunBudgetReached
from Scripts.data_platform.features.benchmarks.isolation import offline_guard


def _strategy_rows():
    rows = []
    for method, delta in [('statistical', 0), ('selected_blend', 1), ('selected_ml', 2)]:
        for fid, target in [(1, 2), (2, 4)]:
            rows.append({'fixture_id': fid, 'snapshot_id': f'{fid:064x}', 'method': method,
                         'kickoff': f'2023-12-0{fid}T12:00:00+00:00',
                         'label_available_at': f'2023-12-0{fid}T15:00:00+00:00',
                         'target': target, 'prediction': target + delta})
    return rows


@pytest.mark.parametrize('available', ['2024-01-01T00:00:00+00:00', '2024-01-01T02:00:00+00:00'])
def test_final_strategy_excludes_unavailable_labels_before_scoring(available):
    cutoff = '2024-01-01T00:00:00+00:00'
    rows = _strategy_rows()
    for method, prediction in [('statistical', 10000.), ('selected_blend', 2.), ('selected_ml', 2.)]:
        rows.append({'fixture_id': 9, 'snapshot_id': '9' * 64, 'method': method,
                     'kickoff': '2023-12-31T23:00:00+00:00', 'label_available_at': available,
                     'target': 2., 'prediction': prediction})
    selected = extension.freeze_strategy(rows, cutoff=cutoff)
    assert selected['method'] == 'statistical'
    assert all(r['metrics']['n'] == 2 and r['late_label_exclusions'] == [9]
               for r in selected['comparisons'])
    changed = deepcopy(rows)
    for row in changed:
        if row['fixture_id'] == 9:
            row['target'] = 999999
            row['prediction'] = 0.
    assert extension.freeze_strategy(changed, cutoff=cutoff) == selected


def test_final_strategy_membership_hash_and_selection_are_order_invariant():
    rows = _strategy_rows()
    expected = extension.freeze_strategy(rows, cutoff='2024-01-01T00:00:00+00:00')
    random.Random(18).shuffle(rows)
    assert extension.freeze_strategy(rows, cutoff='2024-01-01T00:00:00+00:00') == expected


@pytest.mark.parametrize('changed', ['missing', 'target', 'snapshot'])
def test_final_strategy_rejects_inequivalent_comparison_cohorts(changed):
    rows = _strategy_rows()
    if changed == 'missing':
        rows.pop()
    elif changed == 'target':
        rows[-1]['target'] = 8
    else:
        rows[-1]['snapshot_id'] = 'f' * 64
    with pytest.raises(ValueError, match='cohorts, targets or snapshots differ'):
        extension.freeze_strategy(rows, cutoff='2024-01-01T00:00:00+00:00')


def test_final_strategy_rejects_duplicate_fixture_revision():
    rows = _strategy_rows()
    rows.append({**rows[0], 'snapshot_id': 'a' * 64})
    with pytest.raises(ValueError, match='unique fixture cohorts'):
        extension.freeze_strategy(rows, cutoff='2024-01-01T00:00:00+00:00')


def test_final_strategy_rejects_forecasts_from_reserved_period():
    rows = _strategy_rows()
    rows[-1]['kickoff'] = '2024-01-01T00:00:00+00:00'
    rows[-1]['label_available_at'] = '2024-01-01T03:00:00+00:00'
    with pytest.raises(ValueError, match='held-out forecast'):
        extension.freeze_strategy(rows, cutoff='2024-01-01T00:00:00+00:00')


def test_final_strategy_does_not_force_selection_without_available_labels():
    rows = _strategy_rows()
    for row in rows:
        row['label_available_at'] = '2024-01-01T00:00:00+00:00'
    with pytest.raises(ValueError, match='nonempty unique fixture cohorts'):
        extension.freeze_strategy(rows, cutoff='2024-01-01T00:00:00+00:00')


@pytest.mark.parametrize('baseline_error,expected', [(0., 'statistical'), (1., 'selected_blend')])
def test_final_strategy_ties_favour_statistical_then_blend(baseline_error, expected):
    rows = _strategy_rows()
    for row in rows:
        row['prediction'] = row['target'] + (baseline_error if row['method'] == 'statistical' else 0.)
    assert extension.freeze_strategy(rows, cutoff='2024-01-01T00:00:00+00:00')['method'] == expected


def _interrupted_attempts(path, count):
    """Represent already spent, interrupted attempts without creating any model."""
    for index in range(count):
        attempt = path / f"{index:064x}" / "attempt-0001"
        attempt.mkdir(parents=True)
        (attempt / "STARTED.json").write_text('{}\n')


@pytest.mark.parametrize("failure", [False, True])
def test_extension_hard_budget_counts_interrupted_attempts_across_restart(tmp_path, failure):
    assert extension.MAX_FITS == 78
    task_root = tmp_path / 'tasks'
    _interrupted_attempts(task_root, 77)
    attempts = Checkpoints(task_root, 'a' * 64, max_fits=extension.MAX_FITS)
    calls = []

    def simulated_component(attempt):
        calls.append(attempt)
        if failure:
            raise ValueError('synthetic interrupted estimator')
        return {'status': 'complete', 'predictions': np.asarray([1.]), 'raw_predictions': np.asarray([1.])}

    contract = {'kind': 'team_poisson', 'side': 'away', 'market': 'goals'}
    if failure:
        with pytest.raises(ValueError, match='synthetic interrupted'):
            attempts.get(contract, simulated_component)
    else:
        assert attempts.get(contract, simulated_component)['predictions'].tolist() == [1.]
    assert len(calls) == 1
    assert len(list(task_root.glob('*/attempt-*/STARTED.json'))) == 78
    resumed = Checkpoints(task_root, 'a' * 64, max_fits=extension.MAX_FITS)
    assert resumed.previous_fits == 78
    with pytest.raises(RunBudgetReached, match='fit budget'):
        resumed.get({**contract, 'side': 'home'}, simulated_component)
    assert len(calls) == 1
    if not failure:
        # Exhausting the cap does not prevent exact reuse of an already committed
        # estimator: the resume lookup happens before the next-fit budget check.
        restored = resumed.get(contract, lambda _: pytest.fail('Resume must not call fitting'))
        assert restored['predictions'].tolist() == [1.]
        assert resumed.created == 0 and resumed.reused == 1
    assert len(list(task_root.glob('*/attempt-*/STARTED.json'))) == 78


def test_extension_offline_guard_allows_declared_research_inputs_and_blocks_live_or_reserved_io(tmp_path):
    root = tmp_path / 'repo'
    output = root / 'Index/prediction_experiments/new-extension'
    output.mkdir(parents=True)
    allowed = root / 'Index/prediction_experiments/data/development/labels.jsonl'
    allowed.parent.mkdir(parents=True)
    allowed.write_text('synthetic-development-only\n')
    protected = [root / 'Index/prediction_experiments/data/lockboxes/confirmation.jsonl',
                 root / 'Index/prediction_experiments/data/audit/inputs.json',
                 root / 'Index/ml_models/goals_ridge.joblib', root / 'Index/platform.db',
                 root / '.env']
    for path in protected:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('synthetic-protected-fixture\n')
    with offline_guard(root=root, output=output, readable_files=[allowed]):
        assert allowed.read_text() == 'synthetic-development-only\n'
        (output / 'local-result.json').write_text('{}\n')
        for path in protected:
            with pytest.raises(RuntimeError, match='protected data read'):
                path.read_bytes()
        with pytest.raises(RuntimeError, match='write outside experiment'):
            allowed.write_text('overwrite')
        with pytest.raises(RuntimeError, match='sqlite3.connect'):
            sqlite3.connect(str(root / 'Index/platform.db'))
        with pytest.raises(RuntimeError, match='socket.'):
            socket.socket()
    assert allowed.read_text() == 'synthetic-development-only\n'
    assert all(path.read_text() == 'synthetic-protected-fixture\n' for path in protected)


def test_extension_offline_guard_remains_effective_in_worker_thread(tmp_path):
    root = tmp_path / 'repo'
    output = root / 'Index/prediction_experiments/new-extension'
    output.mkdir(parents=True)
    denied = root / 'Index/reserved.json'
    denied.write_text('{}\n')
    errors = []

    def attempt_read():
        try:
            denied.read_bytes()
        except RuntimeError as exc:
            errors.append(str(exc))

    with offline_guard(root=root, output=output):
        thread = Thread(target=attempt_read)
        thread.start()
        thread.join(timeout=3)
        assert not thread.is_alive()
    assert len(errors) == 1 and 'protected data read' in errors[0]
