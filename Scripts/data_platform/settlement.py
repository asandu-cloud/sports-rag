"""Pure, conservative grading of exact-market, regulation-time selections.

No provider calls or writes. Unknown evidence is pending, never an implied zero
or void. Bookmaker-specific policies require review; new publications may instead
record the explicitly approved, labelled product settlement convention.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import math
import re
from typing import Any, Mapping

from .settlement_policy import product_policy, uses_participation_cards


SETTLEMENT_VERSION = "regulation-settlement.v1"
MARKET_KEYS = {
    "goals": {"totals"}, "corners": {"totals_corners_over_under"},
    "sot": {"shots_on_target_over_under"},
    "cards": {"totals_cards_over_under", "totals_yellow_cards"},
    "btts": {"btts"}, "moneyline": {"h2h"}, "spreads": {"spreads"},
    "correct_score": {"correctscore", "correct_score"},
}


def provider_id(value: Any) -> str | None:
    text = str(value).strip() if value is not None else ""
    if not re.fullmatch(r"[0-9]{1,19}", text):
        return None
    number = int(text)
    return str(number) if 0 < number <= 2**63 - 1 else None


def count(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
        return number if math.isfinite(number) and number >= 0 and number.is_integer() else None
    except (ValueError, TypeError, OverflowError):
        return None


def utc_datetime(value: Any) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        # A timezone-free value does not establish the actual match instant.
        return parsed.astimezone(timezone.utc) if parsed.tzinfo else None
    except (ValueError, TypeError):
        return None


@dataclass(frozen=True)
class Grade:
    outcome: str | None = None
    actual_result: float | None = None
    pending_reason: str | None = None

    def to_dict(self):
        return asdict(self)


def asian_outcome(value: float, line: Any, *, over: bool = True) -> str | None:
    """Quarter lines are two half-stakes, including negative handicaps."""
    if isinstance(line, bool) or line is None:
        return None
    try:
        line = float(line)
        if not math.isfinite(line) or not math.isfinite(value):
            return None
        if not math.isclose(line * 4, round(line * 4), abs_tol=1e-8):
            return None
    except (TypeError, ValueError, OverflowError):
        return None
    lines = [line] if round(line * 4) % 2 == 0 else [line - 0.25, line + 0.25]
    margins = [(value - component) * (1 if over else -1) for component in lines]
    score = sum(1 if margin > 0 else -1 if margin < 0 else 0 for margin in margins) / len(lines)
    return {1: "hit", 0.5: "half_hit", 0: "push", -0.5: "half_miss", -1: "miss"}[score]


def parse_fixture_result(fixture: Mapping, statistics: list | None = None) -> dict:
    """Use provider team IDs, not fuzzy names; retain nulls and period limits."""
    info, teams = fixture.get("fixture") or {}, fixture.get("teams") or {}
    status = (info.get("status") or {}).get("short")
    result = {
        "fixture_id": provider_id(info.get("id")), "kickoff": info.get("date"),
        "status": status, "league_id": (fixture.get("league") or {}).get("id"),
        "home_team_id": provider_id((teams.get("home") or {}).get("id")),
        "away_team_id": provider_id((teams.get("away") or {}).get("id")),
        "home_team": (teams.get("home") or {}).get("name"),
        "away_team": (teams.get("away") or {}).get("name"),
    }
    fulltime = (fixture.get("score") or {}).get("fulltime") or {}
    goals = fixture.get("goals") or {}
    for side in ("home", "away"):
        value = count(fulltime.get(side))
        if status == "FT":
            final = count(goals.get(side))
            if value is not None and final is not None and value != final:
                result["score_error"] = "conflicting_fulltime_score"
            if value is None:
                value = final
        # For AET/PEN, NEVER substitute the overall or shootout score.
        result[f"goals_{side}"] = value
    result["statistics_period"] = "regulation_time" if status == "FT" else "unknown"
    ids = [result["home_team_id"], result["away_team_id"]]
    rows = statistics or []
    received = [provider_id((row.get("team") or {}).get("id")) for row in rows]
    valid = (None not in ids and ids[0] != ids[1] and len(rows) == 2
             and len(set(received)) == 2 and set(received) == set(ids))
    if not valid:
        result["statistics_error"] = "missing_statistics" if not rows else "statistics_team_identity_mismatch"
        return result
    fields = {"Corner Kicks": "corners", "Shots on Goal": "sot",
              "Yellow Cards": "yellow", "Red Cards": "red"}
    for row in rows:
        side = "home" if provider_id(row["team"]["id"]) == ids[0] else "away"
        stats = row.get("statistics") or []
        for label, field in fields.items():
            matches = [item.get("value") for item in stats if item.get("type") == label]
            result[f"{field}_{side}"] = count(matches[0]) if len(matches) == 1 else None
        yellow, red = result[f"yellow_{side}"], result[f"red_{side}"]
        result[f"cards_{side}"] = yellow + red if yellow is not None and red is not None else None
    return result


def selection_issue(prediction: Mapping) -> str | None:
    """Static identity checks, safe to run without spending provider requests."""
    fid = provider_id(prediction.get("fixture_id"))
    if fid is None:
        return "missing_fixture_id"
    selection = (prediction.get("tracking") or {}).get("selection") or {}
    if provider_id(selection.get("fixture_id")) != fid:
        return "missing_or_conflicting_selection_identity"
    market, side = prediction.get("market"), prediction.get("side")
    if (selection.get("market_group") != market or selection.get("side") != side
            or selection.get("line") != prediction.get("line")
            or selection.get("bookmaker") != prediction.get("bookmaker")):
        return "conflicting_selection_identity"
    key, period = selection.get("market_key"), selection.get("period")
    if period != "regulation_time":
        return "missing_or_unsupported_period"
    if key not in MARKET_KEYS.get(market, set()):
        return "missing_or_unsupported_market"
    if selection.get("participant") is not None:
        return "unsupported_participant_market"
    if not selection.get("bookmaker"):
        return "missing_bookmaker"
    return None


def grade_selection(prediction: Mapping, result: Mapping, *, rules: Mapping | None = None) -> Grade:
    """Grade exact identities; never retrofit a product policy onto old bets."""
    pending = lambda reason: Grade(pending_reason=reason)
    fid = provider_id(prediction.get("fixture_id"))
    if fid is None or fid != provider_id(result.get("fixture_id")):
        return pending("fixture_identity_mismatch")
    issue = selection_issue(prediction)
    if issue:
        return pending(issue)
    selection = prediction["tracking"]["selection"]
    market, side = prediction.get("market"), prediction.get("side")
    key, period = selection["market_key"], selection["period"]
    approved = product_policy(prediction)
    policy = dict(rules if rules is not None else approved or {})
    product = approved is not None and policy == approved
    reviewed = (policy.get("bookmaker") == selection["bookmaker"]
                and policy.get("market_key") == key and policy.get("period") == period
                and policy.get("version") and policy.get("evidence_reference"))
    status = result.get("status")
    if status in {"CANC", "ABD", "AWD", "WO", "PST"}:
        if product and status in policy["void_statuses"]:
            return Grade(outcome="void")
        # Postponed/abandoned does not universally mean refunded.
        if reviewed and status in policy.get("void_statuses", []) and status not in {"PST", "ABD"}:
            return Grade(outcome="void")
        return pending("bookmaker_status_rule_required")
    if status not in {"FT", "AET", "PEN"}:
        return pending("fixture_not_final")
    if market == "cards" and product and not uses_participation_cards(prediction):
        return pending("card_definition_unverified")
    if market in {"goals", "btts", "moneyline", "spreads", "correct_score"}:
        if result.get("score_error"):
            return pending(result["score_error"])
        home, away = count(result.get("goals_home")), count(result.get("goals_away"))
        if home is None or away is None:
            return pending("missing_regulation_score")
        if market == "btts":
            if side not in {"yes", "no"}:
                return pending("invalid_side")
            both = home > 0 and away > 0
            return Grade("hit" if both == (side == "yes") else "miss", float(both))
        if market == "moneyline":
            if side not in {"home", "away", "draw"}:
                return pending("invalid_side")
            winner = "home" if home > away else "away" if away > home else "draw"
            return Grade("hit" if side == winner else "miss", {"home": 1., "draw": .5, "away": 0.}[winner])
        if market == "correct_score":
            score = re.fullmatch(r"(\d+)-(\d+)", re.sub(r"\s+", "", str(prediction.get("pick", ""))))
            if score is None:
                return pending("invalid_score_selection")
            return Grade("hit" if (int(score[1]), int(score[2])) == (home, away) else "miss")
        if market == "spreads":
            if side not in {"home", "away"}:
                return pending("invalid_side")
            try:
                line = prediction.get("line")
                if isinstance(line, bool):
                    raise ValueError("boolean handicap")
                outcome = asian_outcome(home - away if side == "home" else away - home, -float(line))
            except (TypeError, ValueError, OverflowError):
                outcome = None
            return Grade(outcome, home - away) if outcome else pending("invalid_line")
        total = home + away
    else:
        if market == "cards" and product and uses_participation_cards(prediction):
            if status != "FT":
                return pending("regulation_player_statistics_unavailable")
            cards = result.get("participation_cards") or {}
            if cards.get("pending_reason"):
                return pending(cards["pending_reason"])
            if cards.get("period") != period or cards.get("source") != "api_football_fixture_players":
                return pending("missing_player_statistics")
            home, away = (count((cards.get("totals") or {}).get(side)) for side in ("home", "away"))
        else:
            if result.get("statistics_period") != period:
                return pending("regulation_statistics_unavailable")
            if result.get("statistics_error"):
                return pending(result["statistics_error"])
            field = market
            if market == "cards":
                expected = "yellow_only" if key == "totals_yellow_cards" else "yellow_plus_red"
                if not reviewed or policy.get("card_count") != expected or policy.get("provider_stats_compatible") is not True:
                    return pending("card_definition_unverified")
                field = "yellow" if expected == "yellow_only" else "cards"
            home, away = count(result.get(f"{field}_home")), count(result.get(f"{field}_away"))
        if home is None or away is None:
            return pending("missing_market_statistics")
        total = home + away
    if side not in {"over", "under"}:
        return pending("invalid_side")
    line = prediction.get("line")
    outcome = asian_outcome(total, line, over=side == "over")
    if outcome is None or float(line) < 0:
        return pending("invalid_line")
    return Grade(outcome, total)
