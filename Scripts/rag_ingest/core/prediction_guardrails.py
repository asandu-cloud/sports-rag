"""Explicit, conservative guardrails for canonical pre-match recommendations.

These rules are deliberately separate from model projections and line
selection.  They explain *why* an otherwise positive-priced decision is held
back, and make every threshold available for later walk-forward calibration.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Tuple


GUARDRAIL_VERSION = "prediction-guardrails.v1"

# These are variance floors, not fitted model parameters.  They stop a missing
# or under-dispersed early profile from silently falling back to a Poisson
# distribution (variance == mean), which was the source of extreme tails.
TOTAL_VARIANCE_FLOORS = {
    "goals": {"multiplier": 1.25, "absolute": 2.0},
    "corners": {"multiplier": 1.50, "absolute": 8.0},
    "cards": {"multiplier": 1.60, "absolute": 4.0},
    "sot": {"multiplier": 1.60, "absolute": 7.0},
}

# Conservative launch rules.  They should be calibrated against chronological
# results before loosening; they are intentionally not optimised on a single
# matchday or on headline ROI.
QUALITY_GUARDRAILS = {
    "minimum_current_season_matches": 2,
    "low_confidence_until_current_matches": 6,
    "medium_confidence_until_current_matches": 10,
    "minimum_effective_sample_size": 8.0,
    "cards_require_profiled_referee": True,
}

_CONFIDENCE_RANK = {"low": 0, "medium": 1, "high": 2}


def resolve_total_variance(
    market: str,
    projection_total: Optional[float],
    home_variance: Optional[float],
    away_variance: Optional[float],
) -> Tuple[Optional[float], Dict[str, Any]]:
    """Return a match-total variance with a documented conservative floor."""
    if projection_total is None or projection_total <= 0:
        return None, {"status": "unavailable", "source": "unavailable"}

    config = TOTAL_VARIANCE_FLOORS.get(market, TOTAL_VARIANCE_FLOORS["goals"])
    observed_values = []
    for value in (home_variance, away_variance):
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if numeric > 0:
            observed_values.append(numeric)
    if len(observed_values) == 2:
        observed = sum(observed_values)
        observed_source = "two_team_profile_variance"
    elif len(observed_values) == 1:
        # A single side is not a match distribution.  Doubling it is a
        # deliberately cautious proxy until both profiles are complete.
        observed = 2.0 * observed_values[0]
        observed_source = "one_team_variance_doubled"
    else:
        observed = None
        observed_source = "missing_profile_variance"

    floor = max(float(config["absolute"]), float(projection_total) * float(config["multiplier"]))
    variance = max(observed or 0.0, floor)
    if observed is None:
        source = "conservative_fallback_floor"
    elif observed < floor:
        source = "conservative_floor_over_profile_variance"
    else:
        source = observed_source
    return variance, {
        "status": "available",
        "source": source,
        "profile_variance": observed,
        "variance_floor": floor,
        "variance": variance,
    }


def _number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _profile_values(data_quality: Mapping[str, Any]) -> Tuple[Mapping[str, Any], Mapping[str, Any]]:
    profiles = data_quality.get("profiles") or {}
    if not isinstance(profiles, Mapping):
        return {}, {}
    home = profiles.get("home") or {}
    away = profiles.get("away") or {}
    return (
        home if isinstance(home, Mapping) else {},
        away if isinstance(away, Mapping) else {},
    )


def assess_market_quality(
    data_quality: Mapping[str, Any],
    market: str,
    *,
    variance_source: Optional[str] = None,
    referee_source: Optional[str] = None,
) -> Dict[str, Any]:
    """Decide whether data quality permits a public recommendation.

    This does not recalculate odds or alter a model value.  It converts known
    input limitations into a transparent suppression/cap decision.
    """
    home, away = _profile_values(data_quality)
    profile_status = str((data_quality.get("profiles") or {}).get("status") or "unavailable")
    home_current = _number(home.get("current_season_matches"))
    away_current = _number(away.get("current_season_matches"))
    home_effective = _number(home.get("effective_sample_size"))
    away_effective = _number(away.get("effective_sample_size"))
    min_current = min(home_current, away_current)
    min_effective = min(home_effective, away_effective)
    reasons = []

    if profile_status != "available":
        reasons.append("Profile audit is unavailable for one or both teams.")
    for side, profile in (("home", home), ("away", away)):
        temporal_status = str(profile.get("temporal_status") or "")
        if data_quality.get("fixture_date") and temporal_status != "fixture_rows_strictly_before_target_date":
            reasons.append(f"{side.title()} profile is not time-safe for this fixture date.")
    if min_current < QUALITY_GUARDRAILS["minimum_current_season_matches"]:
        reasons.append(
            "Fewer than "
            f"{QUALITY_GUARDRAILS['minimum_current_season_matches']} completed current-season matches "
            "for one or both teams."
        )
    if min_effective < QUALITY_GUARDRAILS["minimum_effective_sample_size"]:
        reasons.append("Effective profile sample is below the minimum guardrail.")
    if market == "cards" and QUALITY_GUARDRAILS["cards_require_profiled_referee"]:
        if referee_source != "profile":
            reasons.append("Cards recommendations require a profiled referee assignment.")

    if reasons:
        return {
            "version": GUARDRAIL_VERSION,
            "status": "suppressed",
            "eligible": False,
            "confidence_cap": "low",
            "reasons": reasons,
            "minimum_current_season_matches": min_current,
            "minimum_effective_sample_size": min_effective,
            "variance_source": variance_source,
        }

    if min_current < QUALITY_GUARDRAILS["low_confidence_until_current_matches"]:
        cap = "low"
        status = "capped"
        cap_reason = "Early-season profile sample caps confidence at low."
    elif (
        min_current < QUALITY_GUARDRAILS["medium_confidence_until_current_matches"]
        or variance_source in {"conservative_fallback_floor", "conservative_floor_over_profile_variance"}
    ):
        cap = "medium"
        status = "capped"
        cap_reason = "Profile maturity or conservative variance floor caps confidence at medium."
    else:
        cap = None
        status = "eligible"
        cap_reason = None

    return {
        "version": GUARDRAIL_VERSION,
        "status": status,
        "eligible": True,
        "confidence_cap": cap,
        "reasons": [cap_reason] if cap_reason else [],
        "minimum_current_season_matches": min_current,
        "minimum_effective_sample_size": min_effective,
        "variance_source": variance_source,
    }


def cap_confidence(confidence: Optional[str], cap: Optional[str]) -> Optional[str]:
    if confidence is None or cap is None:
        return confidence
    if _CONFIDENCE_RANK.get(confidence, 0) > _CONFIDENCE_RANK.get(cap, 0):
        return cap
    return confidence
