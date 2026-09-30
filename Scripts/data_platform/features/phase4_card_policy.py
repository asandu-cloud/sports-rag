"""Owner-approved two-minute card candidate, staged for joint engine promotion.

Production publications/settlements remain v1. Historical v1 artifacts stay valid.
Special playoffs, European qualifying and knockout fixtures are deferred.
"""
from copy import deepcopy

from . import phase4_cards as cards
from .market_eligibility import round_scope

MINIMUM_RECORDED_MINUTES = 2
CONTRACT, POLICY_VERSION = cards.MINUTE_POLICIES[MINIMUM_RECORDED_MINUTES]
ALLOWED_ROUND_GROUPS = frozenset({'domestic_regular', 'domestic_integral_split',
                                'european_group', 'european_league'})
POLICY = {
    'version': POLICY_VERSION, 'target_contract': CONTRACT, 'activation': 'research_only',
    'minimum_recorded_minutes': MINIMUM_RECORDED_MINUTES,
    'card_count': 'yellow_1_red_2_max_3_per_player',
    'card_eligibility': 'recorded_minutes_at_least_two', 'period': 'regulation_time',
    'basis': 'product_rules_estimate', 'null_policy': 'unknown_never_zero',
    'scope': sorted(ALLOWED_ROUND_GROUPS), 'production_promotion': 'joint_with_new_engine',
    'old_publications': 'retain_original_policy_and_grades',
}


def scope(fixture):
    result = round_scope(fixture['competition'], fixture['season'], fixture.get('round'))
    reasons = []
    if result['group'] not in ALLOWED_ROUND_GROUPS:
        reasons.append('unknown_round' if result['group'] == 'unknown' else 'special_round_deferred')
    if fixture.get('status') != 'FT':
        reasons.append('not_completed_regulation_fixture')
    return {**result, 'eligible': not reasons, 'exclusions': reasons}


def qualify(fixture, rows, identity_evidence, *, raw_players=None, raw_reference=None, raw_error=None):
    """Known league format plus matching saved identity establishes period scope.

    The supplied evidence is a saved metadata snapshot, not a fabricated raw
    provider response. It must match exact fixture/team/competition/season IDs,
    kickoff, round and completed status. Player-source completeness is separate.
    """
    round_result = scope(fixture)
    certified = identity_evidence
    if identity_evidence is not None and fixture.get('round') != identity_evidence.get('round'):
        # Prevent accidental acceptance of a different stage under the same ID.
        certified = None
    result = cards.qualify(fixture, rows, certified, raw_players=raw_players, raw_reference=raw_reference,
                           raw_error=raw_error, minimum_recorded_minutes=MINIMUM_RECORDED_MINUTES)
    exclusions = set(result['exclusions']) | set(round_result['exclusions'])
    if identity_evidence is not None and fixture.get('round') != identity_evidence.get('round'):
        exclusions.add('round_metadata_identity_conflict')
    result.update(policy=deepcopy(POLICY), round_scope=round_result,
                  period_evidence={'basis': 'saved_exact_fixture_identity_and_known_league_format',
                                   'source': (identity_evidence or {}).get('metadata_reference'),
                                   'established': round_result['eligible'] and
                                       not any(x.startswith('regulation_source_') or x == 'round_metadata_identity_conflict'
                                               for x in exclusions)},
                  minimum_recorded_minutes=MINIMUM_RECORDED_MINUTES,
                  eligible=not exclusions, exclusions=sorted(exclusions))
    if exclusions:
        result.update(target=None, team_targets=None)
    return result
