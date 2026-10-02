"""Complete setup comparisons preserve chronology and the actual frozen control."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from Scripts.data_platform.features import phase4_weight_setups as w
from Scripts.ops import phase4_weight_comparison as run


@pytest.fixture(scope='module')
def sample():
    folder = run.STAT/'folds/2022-Q1'
    snapshots = list(run.rows(folder/'snapshots.jsonl'))
    s = next(s for s in snapshots if s['fixture']['competition'] == 'EPL' and w.scope.scope(s)['active'])
    historical = list(run.rows(run.STAT/'history.jsonl'))
    inputs = w.Inputs(historical)
    control = next(r for r in run.rows(folder/'control-rates.jsonl') if r['fixture_id'] == s['fixture']['fixture_id'])
    return s, historical, inputs.features(s), control


@pytest.mark.parametrize('market', w.MARKETS)
def test_registry_exact_budget_deterministic_and_reference_included(market):
    a, b = w.registry(market), w.registry(market)
    assert a == b and len(a) == 60
    assert len({s['id'] for s in a}) == 60
    assert a[0]['id'] == 'reference'
    assert any(s['changes'] > 1 for s in a)
    assert all(s['options']['recent'] <= .3 for s in a)


@pytest.mark.parametrize('market', ('goals', 'corners', 'sot'))
def test_real_archived_control_means_reproduce(sample, market):
    s, history, features, control = sample
    before = deepcopy(features)
    actual = w.stat_mean(features, w.registry(market)[0]['options'], market, run.engine())
    expected = control['goal_means'] if market == 'goals' else [control['means'][market]]
    assert actual == pytest.approx(expected, abs=1e-12)
    assert before == features


def test_both_options_take_effect_without_patching_publishing_source(sample):
    _, _, features, _ = sample
    default = w.registry('corners')[0]['options']
    engine = run.engine(); original = deepcopy(engine['SCORING_WEIGHTS'])
    a = w.stat_mean(features, default, 'corners', engine)
    b = w.stat_mean(features, default | {'own': .7}, 'corners', engine)
    c = w.stat_mean(features, default | {'own': .7, 'recent': 0.}, 'corners', engine)
    assert a != b and b != c
    assert engine['SCORING_WEIGHTS'] == original


def test_future_results_cannot_change_earlier_features(sample):
    snapshot, history, expected, _ = sample
    changed = deepcopy(history)
    for row in changed:
        if w.cm.utc(row['kickoff']).date() >= w.cm.utc(snapshot['as_of']).date():
            for side in ('home', 'away'):
                row[side] = {key: 999. for key in row[side]}
    actual = w.Inputs(changed).features(snapshot)
    assert actual == expected


def test_snapshot_history_cannot_smuggle_own_or_same_day_result(sample):
    snapshot, history, _, _ = sample
    snapshot = deepcopy(snapshot)
    fid = snapshot['fixture']['fixture_id']; team = str(snapshot['fixture']['home_team_id'])
    snapshot['history_evidence'][team]['EPL']['current']['fixture_ids'].insert(0, fid)
    with pytest.raises(ValueError, match='Current/future/same-day'):
        w.Inputs(history).features(snapshot)


def test_recent_missing_values_renormalize_and_zero_is_observed():
    result = w.weighted([None, 0., 4.], [10., 2., 2.])
    assert result['mean'] == 2. and result['n'] == 2 and result['ess'] == 2.
    assert w.weighted([None], [1.])['mean'] is None


def test_saved_recent_windows_have_only_admitted_history(sample):
    _, _, f, _ = sample
    for support in f['support'].values():
        for key, recent in support['recent'].items():
            assert recent['fixture_ids'] == support['current_ids'][:int(key.split(':')[0])]
            assert len(recent['fixture_ids']) == len(recent['weights'])
            assert all(0 < v <= 1 for v in recent['weights'])


def test_goal_primary_likelihood_matches_frozen_engine():
    probability = run.frozen_probability()
    means = np.array([[1.7, .9]] * 8)
    targets = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [2, 1], [4, 3], [0, 3], [3, 0]])
    losses = w.primary_loss(means, targets.sum(axis=1), team_targets=targets)
    expected = [-np.log(probability.dixon_coles_scoreline_prob(int(h), int(a), 1.7, .9, rho=-.1)) for h, a in targets]
    assert losses == pytest.approx(expected, abs=1e-12)


def test_card_default_matches_saved_fuller_control_and_target_not_an_input():
    f = next(f for f in run.rows(run.CARD/'features.jsonl') if f['baseline'] and f['kickoff'].startswith('2022'))
    old = next(r for r in run.rows(run.CARD/'candidate-means.jsonl') if r['fixture_id'] == f['fixture_id'])
    engine = run.engine(True); options = w.registry('cards')[0]['options']
    mean = w.card_mean(f, options, engine, {})
    assert mean == pytest.approx(old['means']['fuller_control'], abs=1e-12)
    changed = deepcopy(f); changed['target'] = 999
    assert w.card_mean(changed, options, engine, {}) == mean
    # The certified card archive uses SQL-style naive UTC timestamps; use its
    # existing UTC convention locally without relaxing statistical input guards.
    assert w.cards.cards.utc(f['kickoff']).tzinfo is not None
    changed_mean = w.card_mean(f, options | {'window': 12, 'half_life': 90., 'foul_blend': 0.}, engine, {})
    assert np.isfinite(changed_mean)


def tuning():
    specs = w.registry('goals')
    meta = [{'fixture_id': i, 'kickoff': f'2022-{1+i%9:02d}-01T12:00:00Z', 'week': str(i%26)} for i in range(520)]
    losses = np.ones((520, 60))*3.; losses[:, 0] = 2.
    return specs, losses, meta


def test_tuning_ties_retain_reference_and_later_rows_refused():
    specs, losses, meta = tuning()
    losses[:, 1] = 2.-1e-10
    assert w.choose(specs, losses, meta)['selected'] == 'reference'
    meta[-1]['kickoff'] = '2023-01-01T00:00:00Z'
    with pytest.raises(ValueError, match='2022'):
        w.choose(specs, losses, meta)


def test_incomplete_configuration_not_silently_ranked_on_smaller_sample():
    specs, losses, meta = tuning()
    losses[:, 1] = .1; losses[0, 1] = float('nan')
    result = w.choose(specs, losses, meta)
    assert result['selected'] == 'reference'
    assert specs[1]['id'] not in {s['id'] for s in result['ranking']}


def test_bootstrap_deterministic_and_holm_family_controls_all_markets():
    rows = [{'week': str(i%25)} for i in range(500)]
    a = w.paired(rows, np.full(500, -.1))
    assert a == w.paired(rows, np.full(500, -.1)) and a['interval95'][1] < 0
    corrected = w.holm({'goals': .01, 'corners': .02, 'sot': .3, 'cards': .8})
    assert corrected == {'goals': .04, 'corners': .06, 'sot': .6, 'cards': .8}


@pytest.mark.parametrize('market', w.MARKETS)
def test_npz_arrays_pass_strict_archived_diagnostic_interfaces(market):
    means = np.array([[[1.7, .9]]]) if market == 'goals' else np.array([[[7.2, 0.]]])
    data = {'means': means, 'alpha': np.array([[0. if market == 'goals' else .025]]),
            'targets': np.array([3.]), 'team_targets': np.array([[2., 1.]])}
    meta = [{'fixture_id': 1, 'league': 'EPL', 'season_stage': '16+', 'missingness': 'complete'}]
    result = run.detailed(meta, data, 0, market, run.frozen_probability())
    actual = result[0]['scores']['value'] if market == 'cards' else result[0]['scores']
    expected = w.primary_loss(means[:, 0, :], data['targets'], data['alpha'][:, 0],
                             data['team_targets'] if market == 'goals' else None)
    assert actual['nll'] == pytest.approx(expected[0], abs=1e-12)
    if market == 'cards':
        assert sum(actual['pmf']) == pytest.approx(1., abs=1e-10)
        for line in ('4.0', '4.25', '4.75', '5.0'):
            outcome = run.scoring.outcome_class(result[0]['target'], float(line))
            assert actual['asian'][line]['over'][outcome] > 0
    else: assert all(sum(line['p']) == pytest.approx(1., abs=1e-10) for line in actual['totals'].values())
