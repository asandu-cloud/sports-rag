"""Explicit compatibility of normalized quotes with canonical regulation forecasts.

Missing period is accepted only for these established full-match keys. This is
an input contract, not certification of a bookmaker's card settlement rules.
"""
VERSION = "canonical-market-contract.v1"
KEYS = {
    "goals": {"totals", "alternate_totals", "totals_goals", "goals_over_under", "total_goals"},
    "corners": {"totals_corners", "totals_corners_over_under", "corners_over_under", "total_corners"},
    "cards": {"totals_cards", "totals_cards_over_under", "cards_over_under", "total_cards"},
    "sot": {"shots_on_target_over_under", "totals_shots_on_target", "total_shots_on_target", "sot"},
    "btts": {"btts", "both_teams_to_score"},
    "moneyline": {"h2h", "moneyline", "1x2"},
    "spreads": {"spreads", "alternate_spreads", "asian_handicap", "asian_handicap_goals"},
}
PERIODS = {"regulation_time", "regulation", "full_time", "full_match", "match", "90_minutes", "90"}
DEFINITIONS = {
    "goals": {"regulation_goals", "regulation_time"},
    "corners": {"regulation_corners", "regulation_time"},
    "sot": {"regulation_shots_on_target", "regulation_time"},
    "moneyline": {"regulation_goals", "regulation_time"},
    "btts": {"regulation_goals", "regulation_time"},
    "spreads": {"asian_handicap", "regulation_goals", "regulation_time"},
    # Card targets are not yet qualified against a declared weighted convention.
    "cards": set(),
}
def canonical_market(key):
    key = str(key or "").lower().strip()
    return next((group for group, keys in KEYS.items() if key in keys), None)

def incompatibility(market, expected=None):
    group = canonical_market(market.get("key"))
    if group is None or (expected is not None and group != expected):
        return "unsupported_market_definition"
    for field in ("period", "market_period"):
        period = market.get(field)
        if period is not None and str(period).lower().strip() not in PERIODS:
            return "incompatible_period"
    if market.get("includes_extra_time") not in (None, False):
        return "includes_extra_time"
    definition = market.get("settlement_definition")
    if definition is not None and str(definition).lower().strip() not in DEFINITIONS[group]:
        return "unqualified_settlement_definition"
    # Preserve provider names for review, and reject obvious conflicting labels.
    name = str(market.get("provider_bet_name") or "").lower()
    if any(token in name for token in ("first half", "second half", "1st half", "2nd half", "extra time", "penalties", "to qualify")):
        return "incompatible_provider_label"
    if group == "sot" and name and "target" not in name:
        return "incompatible_shot_definition"
    if group == "cards" and any(token in name for token in ("yellow", "red card", "booking point")):
        return "incompatible_card_definition"
    if group == "spreads" and ("european" in name or "3-way" in name or "3 way" in name):
        return "incompatible_handicap_type"
    return None

def compatible(market, expected=None):
    return incompatibility(market, expected) is None

def quote_metadata(event, market):
    return {
        "market_key": market.get("key"),
        "fixture_id": str(event.get("id") or event.get("fixture_id") or ""),
        "period": "regulation_time",
        "settlement_definition": market.get("settlement_definition"),
        "provider_bet_id": market.get("provider_bet_id"),
        "provider_bet_name": market.get("provider_bet_name"),
    }
