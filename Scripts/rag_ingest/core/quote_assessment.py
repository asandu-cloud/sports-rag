"""Canonical marginal assessments for existing specialist/unit-bet quotes.

Cache only for a single selection request; never cache across refreshes or users.
No joint probability, combined price, or joint expected value is inferred here.
"""
from contextvars import ContextVar
from functools import wraps
from .market_contract import canonical_market, compatible
from .market_service import evaluate_market_quote
from .team_resolution import profile_cache_boundary

_cache = ContextVar("quote_assessment_cache", default=None)
def assessment_scope(function):
    @wraps(function)
    def scoped(*args, **kwargs):
        if _cache.get() is not None:
            return function(*args, **kwargs)
        token = _cache.set({})
        try:
            return function(*args, **kwargs)
        finally:
            _cache.reset(token)
    return profile_cache_boundary(scoped)

def assess_leg(leg, league):
    event = getattr(leg, "event", None)
    key = (id(event), league, leg.market_key, leg.outcome, leg.point, leg.odds, leg.bookmaker)
    cache = _cache.get()
    if cache is not None and key in cache:
        return dict(cache[key])
    result = _assess(leg, league, event)
    if cache is not None:
        cache[key] = result
    return dict(result)

def _assess(leg, league, event):
    result = {"confidence": None, "model_prob": None, "projected": None,
              "side_agrees": False, "standalone_side": None, "eligible": False,
              "status": "unavailable", "warning": "Canonical quote assessment unavailable.",
              "assessment_source": "canonical-market-quote.v1"}
    group = canonical_market(leg.market_key)
    if group is None:
        result["warning"] = "This specialist market has no qualified canonical quote assessment."
        return result
    if not isinstance(event, dict) or str(event.get("id") or "") != str(leg.event_id):
        result["warning"] = "Original fixture odds snapshot is unavailable."
        return result
    cutoff = event.get("commence_time") or event.get("fixture_date") or event.get("kickoff")
    if not cutoff:
        result["warning"] = "Fixture cutoff is unavailable."
        return result
    markets = [market for book in event.get("bookmakers", []) if book.get("title") == leg.bookmaker
               for market in book.get("markets", []) if market.get("key") == leg.market_key]
    # Ambiguous duplicate definitions must not be reduced to one guessed contract.
    if len(markets) != 1 or not compatible(markets[0], group):
        result["warning"] = "Requested market definition is ambiguous or incompatible."
        return result
    quote = {"market_key": leg.market_key, "bookmaker": leg.bookmaker,
             "outcome": leg.outcome, "point": leg.point, "odds": leg.odds}
    contexts = {name: event[name] for name in ("league_ctx", "knockout_ctx", "lineup_ctx", "ref_mod") if name in event}
    assessed = evaluate_market_quote(event, league, group, quote, fixture_date=cutoff, **contexts)
    decision = assessed.decision
    result.update(confidence=decision.confidence, model_prob=decision.model_probability,
                  projected=assessed.projection.value, side_agrees=decision.is_recommended,
                  eligible=decision.is_recommended, status=decision.status.value,
                  warning=decision.reason, implied_prob=decision.implied_probability,
                  value_edge=decision.value_edge, expected_value=decision.expected_value,
                  probability_basis=decision.probability_basis,
                  probability_version=decision.probability_version,
                  settlement_profile=decision.settlement_profile,
                  input_snapshot_id=assessed.provenance.input_snapshot_id,
                  prediction_system_version=assessed.provenance.system_version)
    return result

def marginal_quality(leg, league):
    assessment = assess_leg(leg, league)
    if not assessment.get("eligible"):
        return 0.0
    # Ranking support for a single quote, never a combined or portfolio EV.
    return max(0.0, assessment.get("expected_value") or 0.0)


def quote_evidence(leg, league):
    assessment = assess_leg(leg, league)
    if assessment.get("model_prob") is None:
        return [assessment.get("warning") or "Canonical quote assessment unavailable."]
    basis = ("Asian price-comparison probability (not full-win probability)"
             if assessment.get("probability_basis") == "asian_equivalent_non_push" else "Outcome probability")
    lines = [f"{basis}: {assessment['model_prob']:.1%}."]
    if assessment.get("expected_value") is not None:
        lines.append(f"Expected return per unit stake at this quote: {assessment['expected_value']:+.1%}.")
    profile = assessment.get("settlement_profile")
    if profile:
        lines.append("Settlement probabilities: " + ", ".join(f"{key.replace('_', ' ')} {value:.1%}" for key, value in profile.items()) + ".")
    elif assessment.get("projected") is not None:
        lines.append(f"Canonical projection: {assessment['projected']:.3f}.")
    if assessment.get("warning"):
        lines.append(assessment["warning"])
    return lines
