"""The user-approved product policy, snapshotted only on new publications.

This is a measurement convention, NOT a statement of any bookmaker's rules.
Never infer this policy for an old prediction that did not record it.
"""
from copy import deepcopy
from typing import Mapping


POLICY_VERSION = "spix-participation-settlement.v1"
POLICY_NOTE = (
    "Performance under Spix product rules, not guaranteed bookmaker settlement. "
    "Cards use players with recorded minutes > 0, not on-pitch eligibility at the time."
)
_POLICY = {
    "version": POLICY_VERSION,
    "basis": "product_rules_estimate",
    "period": "regulation_time",
    "void_statuses": ["ABD"],
    "card_market_key": "totals_cards_over_under",
    "card_count": "yellow_1_red_2_max_3_per_player",
    "card_eligibility": "recorded_minutes_greater_than_zero",
    "card_data": "api_football_FT_fixture_players",
    "evidence_reference": "docs/participation-settlement-policy.md",
}


def new_publication_policy():
    return deepcopy(_POLICY)


def product_policy(prediction):
    """Unknown/mutated versions are not silently treated as the current one."""
    policy = prediction.get("settlement_policy")
    return deepcopy(_POLICY) if isinstance(policy, Mapping) and policy == _POLICY else None


def uses_participation_cards(prediction):
    policy = product_policy(prediction)
    tracking = prediction.get("tracking")
    if not isinstance(tracking, Mapping):
        return False
    selection = tracking.get("selection")
    if not isinstance(selection, Mapping):
        return False
    return bool(policy and prediction.get("market") == "cards"
                and selection.get("market_key") == policy["card_market_key"]
                and selection.get("period") == policy["period"])


def policy_reporting(rows):
    """Keep the basis visible even when legacy and product-rule rows coexist."""
    from .outcomes import normalize_outcome
    settled = [p for p in rows if normalize_outcome(p.get("outcome"))]
    governed = [p for p in settled if product_policy(p)
                and (p.get("settlement") or {}).get("rule") == product_policy(p)]
    return {
        "product_rule_settlements": len(governed),
        "participation_card_settlements": sum(uses_participation_cards(p) for p in governed),
        "settlement_policy_note": POLICY_NOTE if governed else None,
    }
