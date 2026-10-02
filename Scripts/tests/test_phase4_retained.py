"""Research carryover must preserve controls and cannot inherit new qualifications."""
from copy import deepcopy

import pytest
from scipy.stats import nbinom

from Scripts.data_platform.features import phase4_retained as r


def inputs(league='EPL', n=20):
    decision = {'production_enabled': False, 'final_system_test_opened': False,
                'period': ['2025-01-01', '2025-07-01'],
                'claims': {m: {'qualified': True} for m in r.ALPHAS}}
    review = {'publication_enabled': False, 'calibration': 'identity',
              'pooled_claim_scope': {'leagues': list(r.finalists.LEAGUES), 'minimum_current_matches_both_teams': 8},
              'unsupported_conditions': ['preserved_evidence_limit'],
              **{m: {'method': 'negative_binomial_fixed_control_mean', 'alpha': a} for m, a in r.ALPHAS.items()}}
    value = r.bundle(decision, review, {'baseline_seal_sha256': 'frozen'})
    snapshot = {'fixture': {'fixture_id': 7, 'competition': league, 'kickoff': '2023-02-01T12:00:00Z'},
                'as_of': '2023-02-01T12:00:00Z', 'snapshot_id': 'saved', 'forecast_stage': r.FORECAST_STAGE,
                'profile_quality': {str(i): {'current_season_matches': n} for i in (1, 2)},
                'profiles': {str(i): {'xg_home_pm': 1., 'xg_away_pm': 1.} for i in (1, 2)}}
    control = {'fixture_id': 7, 'input_snapshot_id': 'saved', 'candidate': r.counts.registry()[0],
               'means': {'goals': 3., 'corners': 10., 'sot': 8., 'cards': 4.},
               'goal_means': [2., 1.], 'variances': {'corners': 20., 'sot': 16., 'cards': 8.},
               'publication_enabled': False, 'evidence': {'fallbacks': []},
               'goal_score_matrix': [[.1, .2], [.3, .4]]}
    return snapshot, control, value


def apply(snapshot, control, value):
    return r.retain(snapshot, control, value, baseline_id='frozen')


def test_retains_only_qualified_variances_and_preserves_entire_goal_and_card_state():
    s, c, b = inputs()
    before = deepcopy((s, c, b))
    result = apply(s, c, b)
    assert (s, c, b) == before
    assert result['means'] == c['means']
    assert result['goal_means'] == c['goal_means']
    assert result['goal_score_matrix'] == c['goal_score_matrix']
    assert result['variances'] == {'corners': 12.5, 'sot': 8.64, 'cards': 8.}
    assert result['component_ids']['cards'] == result['component_ids']['goals'] == 'control'
    assert not result['publication_enabled'] and not result['production_qualified']
    assert result['joint_cross_market_probability'] is None
    assert result == apply(s, c, b)


@pytest.mark.parametrize('league,n', [('UECL', 25), ('EPL', 7), ('Championship', 30)])
def test_outside_pooled_scope_preserves_control_exactly(league, n):
    s, c, b = inputs(league, n)
    result = apply(s, c, b)
    assert result['variances'] == c['variances']
    assert result['means'] == c['means']
    for m in r.ALPHAS:
        assert result['component_ids'][m] == 'control'
        assert result['market_fallbacks'][m] == ['unsupported_league_or_season_stage']
        assert r.count_distribution(result, m) == r.count_distribution(c, m)


def test_original_pooled_stage_does_not_become_independent_slice_qualification():
    s, c, b = inputs(n=8)
    s['profiles']['1']['xg_home_pm'] = None
    result = apply(s, c, b)
    assert result['variances']['corners'] == 12.5
    assert set(result['retained_evidence']['support_limitations']) == {
        '8_to_15_match_stage_insufficient_standalone_support', 'partial_missing_xg_not_independently_qualified'}
    assert not result['production_qualified']
    assert 'control_for_unsupported_conditions' in b['required_public_fallback']


def test_new_forecast_stage_keeps_control():
    s, c, b = inputs()
    s['forecast_stage'] = 'confirmed_lineup_amendment'
    result = apply(s, c, b)
    assert result['variances'] == c['variances']
    assert 'forecast_stage_not_qualified' in result['market_fallbacks']['corners']


def test_missing_and_zero_stay_distinct():
    s, c, b = inputs()
    c['means'].update(corners=None, sot=0.)
    c['variances']['corners'] = None
    result = apply(s, c, b)
    assert r.count_distribution(result, 'corners') is None
    assert r.count_distribution(result, 'sot')['pmf'] == [1.]
    assert result['variances']['sot'] == 0.


@pytest.mark.parametrize('market', ['corners', 'sot'])
def test_preserved_nb_probabilities_match_independent_distribution(market):
    s, c, b = inputs()
    result = apply(s, c, b)
    distribution = r.count_distribution(result, market)
    mu, alpha = c['means'][market], r.ALPHAS[market]
    reference = nbinom(1 / alpha, 1 / (1 + alpha * mu))
    assert distribution['pmf'] == pytest.approx(reference.pmf(range(len(distribution['pmf']))), abs=1e-13)
    assert sum(distribution['pmf']) + distribution['omitted_mass'] == pytest.approx(1., abs=1e-12)
    assert distribution['omitted_mass'] <= 1e-12 * 1.01


@pytest.mark.parametrize('date', ['2024-01-01T00:00:00Z', '2025-06-01T12:00:00Z', '2025-07-01T12:00:00Z', '2026-08-01T12:00:00Z'])
def test_adapter_cannot_open_later_forecast_periods(date):
    s, c, b = inputs()
    s['as_of'] = s['fixture']['kickoff'] = date
    with pytest.raises(ValueError, match='pre-2024'):
        apply(s, c, b)


def test_new_weight_candidate_cannot_inherit_reference_qualification():
    s, c, b = inputs()
    c['candidate'] = next(v for v in r.counts.registry() if v['family'] == 'own_weight')
    c['means']['corners'] = 11.
    with pytest.raises(ValueError, match='unchanged frozen-control'):
        apply(s, c, b)
    with pytest.raises(ValueError, match='provenance'):
        r.retain(s, c, b, baseline_id='different_engine')


def test_mismatched_input_or_publishable_control_is_rejected():
    s, c, b = inputs()
    c['input_snapshot_id'] = 'another'
    with pytest.raises(ValueError, match='identity'):
        apply(s, c, b)
    c['input_snapshot_id'] = 'saved'
    c['publication_enabled'] = True
    with pytest.raises(ValueError, match='isolated'):
        apply(s, c, b)


@pytest.mark.parametrize('field,value', [('publication_enabled', True), ('production_qualified', True), ('calibration', 'sigmoid')])
def test_rehashed_promotion_or_calibrator_edits_are_rejected(field, value):
    s, c, b = inputs()
    b[field] = value
    b['id'] = r.counts.digest({k: v for k, v in b.items() if k != 'id'})
    with pytest.raises(ValueError, match='Invalid retained'):
        apply(s, c, b)


def test_unqualified_component_is_rejected_even_with_updated_identity():
    s, c, b = inputs()
    b['qualification']['sot']['qualified'] = False
    b['id'] = r.counts.digest({k: v for k, v in b.items() if k != 'id'})
    with pytest.raises(ValueError, match='Unqualified'):
        apply(s, c, b)
