"""Retained, research-only count components; no fitting or publishing imports."""
from copy import deepcopy

from . import phase4_candidates as counts
from . import phase4_finalists as finalists

VERSION = 'phase4-retained-count-components.v1'
ALPHAS = {'corners': 0.025, 'sot': 0.01}
FORECAST_STAGE = 'reconstructed_immediately_before_kickoff'


def bundle(decision, reviewed, provenance):
    """Separate accepted components from the old, mixed finalist recipe."""
    if (decision.get('production_enabled') is not False
            or decision.get('final_system_test_opened') is not False
            or reviewed.get('publication_enabled') is not False
            or reviewed.get('calibration') != 'identity'):
        raise ValueError('Unexpected qualification/review contract')
    for market, alpha in ALPHAS.items():
        if (decision['claims'][market]['qualified'] is not True
                or reviewed[market] != {'method': 'negative_binomial_fixed_control_mean', 'alpha': alpha}):
            raise ValueError('Component has not qualified with these parameters')
    expected_scope = {'leagues': list(finalists.LEAGUES), 'minimum_current_matches_both_teams': 8}
    if reviewed['pooled_claim_scope'] != expected_scope:
        raise ValueError('Qualification scope changed')
    result = {
        'version': VERSION,
        'role': 'retained_research_reference',
        'components': {m: deepcopy(reviewed[m]) for m in ALPHAS},
        'scope': expected_scope,
        'goals': 'unchanged_frozen_control_including_BTTS_winner_and_handicaps',
        'cards': 'no_component_carried_or_qualified_by_this_bundle',
        'calibration': 'identity',
        'mean_policy': 'unchanged_frozen_control_means',
        'variance_formula': 'mu + alpha * mu^2',
        'unsupported_conditions': deepcopy(reviewed['unsupported_conditions']),
        'qualification': {m: deepcopy(decision['claims'][m]) for m in ALPHAS},
        'qualification_period': deepcopy(decision['period']),
        'qualification_applies_to': 'original_pooled_comparison_not_every_slice_or_line',
        'changed_means_require_new_configuration_and_validation': True,
        'required_public_fallback': 'control_for_unsupported_conditions_pending_new_evidence',
        'data_availability': 'assumed_final',
        'development_end_exclusive': '2024-01-01T00:00:00Z',
        'publication_enabled': False,
        'production_qualified': False,
        'final_system_test_opened': False,
        'cross_market_joint_probability': None,
        'provenance': deepcopy(provenance),
    }
    return {**result, 'id': counts.digest(result)}


def validate(value):
    if (value.get('id') != counts.digest({k: v for k, v in value.items() if k != 'id'})
            or value.get('version') != VERSION
            or value.get('publication_enabled') is not False
            or value.get('production_qualified') is not False
            or value.get('final_system_test_opened') is not False
            or value.get('mean_policy') != 'unchanged_frozen_control_means'
            or value.get('calibration') != 'identity'
            or value.get('cross_market_joint_probability') is not None
            or value.get('development_end_exclusive') != '2024-01-01T00:00:00Z'
            or value.get('scope') != {'leagues': list(finalists.LEAGUES), 'minimum_current_matches_both_teams': 8}
            or value.get('components') != {m: {'method': 'negative_binomial_fixed_control_mean', 'alpha': a}
                                          for m, a in ALPHAS.items()}):
        raise ValueError('Invalid retained research bundle')
    if any(value.get('qualification', {}).get(m, {}).get('qualified') is not True for m in ALPHAS):
        raise ValueError('Unqualified retained component')


def retain(snapshot, control, value, *, baseline_id):
    """Apply retained dispersions to a frozen-control development rate record.

    The original >=8-match pooled research scope is reproduced, not silently
    redefined. Unsupported slices carry explicit limitations, not release rights.
    Modified-mean candidates must use a separate configuration and validation.
    """
    validate(value)
    if baseline_id != value['provenance']['baseline_seal_sha256']:
        raise ValueError('Wrong frozen control provenance')
    kickoff = finalists.utc(snapshot['fixture']['kickoff'])
    cutoff = finalists.utc(snapshot['as_of'])
    if cutoff > kickoff or max(cutoff, kickoff) >= finalists.utc(value['development_end_exclusive']):
        raise ValueError('Only pre-2024 development forecasts are permitted')
    if (control['fixture_id'] != snapshot['fixture']['fixture_id']
            or control['input_snapshot_id'] != snapshot['snapshot_id']):
        raise ValueError('Fixture/input identity mismatch')
    if control.get('candidate') != counts.registry()[0]:
        raise ValueError('Requires unchanged frozen-control rates, not a modified candidate')
    if control.get('publication_enabled') is not False:
        raise ValueError('Requires an isolated research record')
    scope = finalists.scope(snapshot)
    limits = []
    if scope['season_stage'] == '8-15':
        limits.append('8_to_15_match_stage_insufficient_standalone_support')
    profiles = snapshot.get('profiles', {})
    if len(profiles) != 2 or any(p.get('xg_home_pm') is None or p.get('xg_away_pm') is None
                                  for p in profiles.values()):
        limits.append('partial_missing_xg_not_independently_qualified')
    stage_supported = snapshot.get('forecast_stage') == FORECAST_STAGE
    if not stage_supported:
        limits.append('forecast_stage_not_qualified')
    r = deepcopy(control)
    r['configuration'] = VERSION
    r['research_bundle_id'] = value['id']
    r['frozen_control_rate_id'] = counts.digest(control)
    r['candidate'] = {'id': VERSION, 'family': 'retained_count_distributions', 'markets': list(ALPHAS)}
    r['component_ids'] = {m: 'control' for m in control['means']}
    r['market_fallbacks'] = {}
    for market, alpha in ALPHAS.items():
        mean = control['means'].get(market)
        reasons = []
        if not scope['active']:
            reasons.append('unsupported_league_or_season_stage')
        if not stage_supported:
            reasons.append('forecast_stage_not_qualified')
        if mean is None:
            reasons.append('projection_unavailable')
        elif not counts.finite_nonnegative(mean):
            raise ValueError('Invalid control mean')
        if not reasons:
            r['variances'][market] = mean + alpha * mean * mean
            r['component_ids'][market] = 'dispersion_' + market + ':' + str(alpha)
        r['market_fallbacks'][market] = reasons
    r['retained_evidence'] = {'scope': scope, 'support_limitations': limits,
                              'qualification': value['qualification_applies_to']}
    r['production_qualified'] = False
    r['publication_enabled'] = False
    r['joint_cross_market_probability'] = None
    return r


def count_distribution(rates, market):
    """Same PMF implementation as qualification; no binary recalibration."""
    if market not in ALPHAS:
        raise ValueError('Only retained count markets are supported')
    mean = rates['means'].get(market)
    if mean is None:
        return None
    variance = rates['variances'].get(market)
    if not counts.finite_nonnegative(mean) or (variance is not None and not counts.finite_nonnegative(variance)):
        raise ValueError('Invalid count rate')
    alpha = max(0., ((variance or mean) - mean) / mean ** 2) if mean > 0 else 0.
    return counts.count_pmf(mean, alpha)
