"""Versioned, research-only target and historical-context eligibility.

Evidence decisions never inspect model errors. A known target is not permission
to score it: the exporter adds the separately declared pre-match support gate.
"""
from __future__ import annotations

import math
import re

VERSION = "phase3-eligibility.v1"
MARKETS = ("goals", "corners", "sot", "cards")
PRIMARY_MARKETS = ("goals", "corners", "sot")
DOMESTIC = frozenset({"EPL", "LaLiga", "SerieA", "Bundesliga", "Ligue1", "Championship",
                      "SuperLig", "Eredivisie", "PrimeiraLiga", "BelgianProLeague"})
CONTINENTAL = frozenset({"UCL", "UEL", "UECL"})
CONTRACT = {
    "version": VERSION,
    "period": "regulation_plus_stoppage",
    "availability": "assumed_final",
    "assumed_completion_hours": 3,
    "cards": "excluded_unqualified_target_semantics_and_selective_missingness",
    "minimum_contributing_team_observations": 5,
    "support_bands": ["0", "1-4", "5-9", "10+"],
    "unknown_round": "exclude_from_targets_and_history",
    "special_round_context": "verified_same_competition_only",
    "administrative_concern": "exclude_from_targets_and_history",
    "unverified_statistic": "unknown_never_zero",
    "legacy_zero": "unknown_without_independent_evidence",
    "evidence_policy": "checksum_verified_raw_or_verified_reconstruction",
    "scope_seasons": [2019, 2026],
}


def count(value):
    """Real zero survives; booleans, fractions, infinities and negatives do not."""
    if value is None or isinstance(value, bool):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return int(numeric) if math.isfinite(numeric) and numeric >= 0 and numeric.is_integer() else None


def round_scope(competition: str, season: int, original: str | None) -> dict:
    """Explicit provider-era aliases; domestic playoffs differ from split groups.

    Numbered Belgian relegation/split groups are integral league phases. Bare
    relegation rounds and promotion/qualification playoffs are context only.
    Ambiguous unrecognized names never acquire main-competition status.
    """
    result = {"original": original, "group": "unknown", "target": False, "context": False}
    if not isinstance(season, int) or not 2019 <= season <= 2026 or (competition == "UECL" and season < 2021):
        return result
    label = " ".join(str(original or "").strip().lower().split())
    base = re.sub(r" - [0-9]+$", "", label)
    if competition in DOMESTIC:
        if re.fullmatch(r"regular season - [0-9]+", label):
            result.update(group="domestic_regular", target=True, context=True)
        elif competition == "BelgianProLeague" and base in {
            "championship round", "championship group", "conference league group",
            "conference league play-off group", "relegation group",
        } and re.search(r" - [0-9]+$", label):
            result.update(group="domestic_integral_split", target=True, context=True)
        elif competition == "BelgianProLeague" and re.fullmatch(r"relegation round - [0-9]+", label):
            result.update(group="domestic_integral_split", target=True, context=True)
        else:
            special = {
                "Championship": {"promotion play-offs - semi-finals", "promotion play-offs - final", "final", "semi-finals"},
                "Bundesliga": {"relegation round", "final"},
                "Ligue1": {"relegation round", "final"},
                "PrimeiraLiga": {"relegation round", "final"},
                "SerieA": {"relegation decider"},
                "Eredivisie": {"relegation round", "1st round", "2nd round", "3rd round", "semi-finals",
                               "conference league play-offs - semi-finals", "conference league play-offs - final",
                               "europa league play-offs - semi-finals", "europa league play-offs - final"},
                "BelgianProLeague": {"relegation round", "conference league play-offs - final",
                                     "quarter-finals", "semi-finals", "final"},
            }
            if label in special.get(competition, set()):
                result.update(group="domestic_separate_playoff", context=True)
    elif competition in CONTINENTAL:
        if (base == "group stage" or re.fullmatch(r"group [a-l]", base)) and season <= 2023:
            result.update(group="european_group", target=True, context=True)
        elif base == "league stage" and season >= 2024:
            result.update(group="european_league", target=True, context=True)
        elif label in {"16th finals", "8th finals", "round of 32", "round of 16", "quarter-finals", "semi-finals", "final"}:
            result.update(group="european_knockout", target=True, context=True)
        elif label == "knockout round play-offs" and season >= (2024 if competition == "UCL" else 2021):
            result.update(group="european_knockout", target=True, context=True)
        elif label in {"1st qualifying round", "2nd qualifying round", "3rd qualifying round", "play-offs",
                       "playoff round", "preliminary round", "preliminary round 1", "preliminary round 2"}:
            result.update(group="european_qualifying", context=True)
    return result


def market_decisions(team_labels: dict, common_reasons=(), *, target_scope: bool = True,
                     round_group: str = "unknown", statistic_reasons: dict | None = None) -> dict:
    """Deterministic overlapping reasons, with no target imputation."""
    decisions = {}
    for market in MARKETS:
        reasons = list(common_reasons)
        if market == "cards":
            reasons.append("cards_target_not_qualified")
        if not target_scope:
            reasons.append("unknown_round" if round_group == "unknown" else "round_outside_primary_scope")
        for side in ("home", "away"):
            if count(team_labels.get(side, {}).get(market)) is None:
                reasons.append(f"missing_{side}_{market}")
            reasons.extend((statistic_reasons or {}).get(side, {}).get(market, []))
        reasons = list(dict.fromkeys(reasons))
        decisions[market] = {"eligible": not reasons, "reasons": reasons}
    return decisions
