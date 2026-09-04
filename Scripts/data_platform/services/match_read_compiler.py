"""Compile canonical market results into one fixture-level Match Read draft.

This module is intentionally deterministic and presentation-neutral.  It does
not fetch odds, call a language model, send a Discord message, or create a
published recommendation.  Its only job is to make a conservative, auditable
fixture-level decision from the canonical results already produced by the
market service.

The compiler deliberately distinguishes *compatible individual selections*
from a same-game bet.  A selected goal, BTTS, or side angle may be useful in
the same Match Read, but the compiler never multiplies their probabilities,
invent a combined price, or emits a same-game package without a bookmaker
quote supplied elsewhere.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from typing import Any, Dict, Mapping, Optional, Sequence


MATCH_READ_COMPILER_VERSION = "match-read-compiler.v1"
MAX_ACTIONABLE_SELECTIONS = 3
MAX_ALTERNATIVE_CANDIDATES = 3

_CONFIDENCE_RANK = {"high": 3, "medium": 2, "low": 1}
# This only breaks exact ranking ties; it is not an assertion that one market
# is intrinsically better than another.
_MARKET_TIEBREAK_ORDER = {
    "goals": 0,
    "btts": 1,
    "moneyline": 2,
    "spreads": 3,
    "corners": 4,
    "cards": 5,
    "sot": 6,
}


class MatchReadCompileError(ValueError):
    """Raised when results cannot form one valid fixture-level draft."""


@dataclass(frozen=True)
class MatchReadDraft:
    """A validated, unpersisted Match Read.

    ``selections`` is ready for :class:`MatchReadService.create`; all other
    canonical recommendations remain in ``game_script['alternative_candidates']``
    for later product review.  Alternatives are intentionally not presented as
    official selections or published by this module.
    """

    canonical_results: tuple[Dict[str, Any], ...]
    status: str
    thesis: str
    selections: tuple[Dict[str, Any], ...]
    game_script: Dict[str, Any]
    stage: str = "pre_match"
    evaluated_at: Optional[str] = None

    def create_kwargs(self) -> Dict[str, Any]:
        """Return the explicit payload accepted by ``MatchReadService.create``."""
        return {
            "canonical_results": list(self.canonical_results),
            "status": self.status,
            "thesis": self.thesis,
            "selections": [dict(selection) for selection in self.selections],
            # Packages are intentionally absent until a delivery layer has a
            # real same-book quote to pass through the existing validator.
            "packages": [],
            "game_script": dict(self.game_script),
            "stage": self.stage,
            "evaluated_at": self.evaluated_at,
        }


@dataclass(frozen=True)
class _Candidate:
    result_index: int
    result: Mapping[str, Any]
    market_group: str
    market_key: str
    side: str
    line: Optional[float]
    odds: float
    confidence: str
    value_edge: float
    expected_value: float
    model_probability: float
    tags: frozenset[str]


def compile_match_read(
    canonical_results: Sequence[Any],
    *,
    stage: str = "pre_match",
) -> MatchReadDraft:
    """Turn all canonical results for one fixture into one Match Read draft.

    Selection policy is intentionally narrow until real matchday shadow output
    has been reviewed:

    * at most one result from each market group;
    * goal/BTTS directions must tell the same high-event or low-event story;
    * moneyline and spread sides must agree if both are selected;
    * unknown relationships are allowed only when they are not structurally
      contradictory, and are labelled as individual angles rather than a bet
      builder.

    The output is safe to persist but is not a public release event.  The
    caller chooses whether and when to deliver or publish it.
    """
    snapshots = tuple(_snapshot(result, field=f"canonical_results[{index}]")
                      for index, result in enumerate(canonical_results))
    if not snapshots:
        raise MatchReadCompileError("A Match Read requires at least one canonical result.")

    fixture = _fixture_from_results(snapshots)
    normalised_stage = str(stage or "").strip().lower()
    if normalised_stage not in {"pre_match", "confirmed_lineups"}:
        raise MatchReadCompileError("stage must be 'pre_match' or 'confirmed_lineups'.")

    recommended = [
        _candidate_from_result(index, result)
        for index, result in enumerate(snapshots)
        if _decision_status(result) == "recommended"
    ]
    ranked = sorted(recommended, key=_ranking_key)

    if not ranked:
        status = _non_recommended_status(snapshots)
        return MatchReadDraft(
            canonical_results=snapshots,
            status=status,
            thesis=_non_recommended_thesis(fixture, snapshots, status),
            selections=(),
            game_script=_non_recommended_game_script(snapshots, status),
            stage=normalised_stage,
            evaluated_at=_evaluated_at(snapshots),
        )

    selected = _select_compatible(ranked)
    # Every canonical recommended result is well-formed; this defensive check
    # documents the invariant rather than silently returning a recommended
    # Match Read without a required core selection.
    if not selected:
        raise MatchReadCompileError("No compatible core selection could be chosen.")

    selected_indices = {candidate.result_index for candidate in selected}
    alternatives = [
        _candidate_summary(candidate)
        for candidate in ranked
        if candidate.result_index not in selected_indices
    ][:MAX_ALTERNATIVE_CANDIDATES]
    selections = tuple(
        {
            "result_index": candidate.result_index,
            "role": "core" if position == 0 else "supporting",
        }
        for position, candidate in enumerate(selected)
    )
    tags = _story_tags(selected)
    return MatchReadDraft(
        canonical_results=snapshots,
        status="recommended",
        thesis=_recommended_thesis(fixture, selected, tags),
        selections=selections,
        game_script={
            "schema_version": "match-read-game-script.v1",
            "compiler_version": MATCH_READ_COMPILER_VERSION,
            "tags": tags,
            "selection_relationship": (
                "These are compatible individual selections from the same fixture. "
                "They are not a combined-bet recommendation; no combined price, "
                "probability, or expected value is implied."
            ),
            "selected_result_indexes": [candidate.result_index for candidate in selected],
            "alternative_candidates": alternatives,
            "market_summary": _market_summary(snapshots),
        },
        stage=normalised_stage,
        evaluated_at=_evaluated_at(snapshots),
    )


def persist_match_read(
    draft: MatchReadDraft,
    *,
    service: Any = None,
) -> Dict[str, Any]:
    """Persist a compiler draft through the existing immutable service.

    Kept as a tiny explicit boundary so callers can generate/review drafts
    without touching ``platform.db`` and tests can inject an isolated service.
    """
    if service is None:
        from .match_reads import MatchReadService

        service = MatchReadService()
    return service.create(**draft.create_kwargs())


def _snapshot(value: Any, *, field: str) -> Dict[str, Any]:
    to_dict = getattr(value, "to_dict", None)
    candidate = to_dict() if callable(to_dict) else value
    if not isinstance(candidate, Mapping):
        raise MatchReadCompileError(f"{field} must be a mapping or expose to_dict().")
    try:
        return json.loads(json.dumps(dict(candidate), sort_keys=True, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise MatchReadCompileError(f"{field} must be JSON serializable.") from exc


def _text(value: Any, field: str, *, required: bool = False) -> str:
    result = str(value or "").strip()
    if required and not result:
        raise MatchReadCompileError(f"{field} is required.")
    return result


def _number(value: Any, field: str, *, required: bool = False) -> Optional[float]:
    if value is None or value == "":
        if required:
            raise MatchReadCompileError(f"{field} is required.")
        return None
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise MatchReadCompileError(f"{field} must be numeric.") from exc
    if not math.isfinite(result):
        raise MatchReadCompileError(f"{field} must be finite.")
    return result


def _fixture_from_results(results: Sequence[Mapping[str, Any]]) -> Dict[str, str]:
    first = results[0].get("fixture")
    if not isinstance(first, Mapping):
        raise MatchReadCompileError("canonical_results[0].fixture is required.")
    fixture = {
        "event_id": _text(first.get("event_id"), "fixture.event_id", required=True),
        "league": _text(first.get("league"), "fixture.league", required=True),
        "home_team": _text(first.get("home_team"), "fixture.home_team", required=True),
        "away_team": _text(first.get("away_team"), "fixture.away_team", required=True),
    }
    for index, result in enumerate(results[1:], start=1):
        other = result.get("fixture")
        if not isinstance(other, Mapping):
            raise MatchReadCompileError(f"canonical_results[{index}].fixture is required.")
        for field, expected in fixture.items():
            actual = _text(other.get(field), f"canonical_results[{index}].fixture.{field}")
            if actual != expected:
                raise MatchReadCompileError(
                    f"canonical_results[{index}] belongs to a different fixture ({field})."
                )
    return fixture


def _decision_status(result: Mapping[str, Any]) -> str:
    decision = result.get("decision")
    if not isinstance(decision, Mapping):
        raise MatchReadCompileError("canonical result decision is required.")
    status = _text(decision.get("status"), "decision.status", required=True).lower()
    if status not in {"recommended", "no_bet", "unavailable"}:
        raise MatchReadCompileError(f"Unsupported canonical decision status: {status!r}.")
    return status


def _candidate_from_result(result_index: int, result: Mapping[str, Any]) -> _Candidate:
    market = result.get("market")
    decision = result.get("decision")
    if not isinstance(market, Mapping) or not isinstance(decision, Mapping):
        raise MatchReadCompileError("A selected canonical result requires market and decision data.")
    quote = decision.get("quote")
    if not isinstance(quote, Mapping):
        raise MatchReadCompileError("A recommended canonical result requires a quoted price.")

    market_group = _text(market.get("group"), "market.group", required=True).lower()
    market_key = _text(market.get("key"), "market.key", required=True).lower()
    side = _text(quote.get("side"), "decision.quote.side", required=True)
    odds = _number(quote.get("odds"), "decision.quote.odds", required=True)
    probability = _number(decision.get("model_probability"), "decision.model_probability", required=True)
    if odds is None or odds <= 1.0:
        raise MatchReadCompileError("A recommended canonical result requires odds above 1.0.")
    if probability is None or probability < 0.0 or probability > 1.0:
        raise MatchReadCompileError("model_probability must be between 0 and 1.")

    line = _number(quote.get("line"), "decision.quote.line")
    value_edge = _number(decision.get("value_edge"), "decision.value_edge") or 0.0
    expected_value = _number(decision.get("expected_value"), "decision.expected_value") or 0.0
    confidence = _text(decision.get("confidence"), "decision.confidence").lower() or "low"
    return _Candidate(
        result_index=result_index,
        result=result,
        market_group=market_group,
        market_key=market_key,
        side=side,
        line=line,
        odds=odds,
        confidence=confidence,
        value_edge=value_edge,
        expected_value=expected_value,
        model_probability=probability,
        tags=frozenset(_candidate_tags(result, market_group, market_key, side)),
    )


def _candidate_tags(
    result: Mapping[str, Any],
    market_group: str,
    market_key: str,
    side: str,
) -> set[str]:
    fixture = result.get("fixture") if isinstance(result.get("fixture"), Mapping) else {}
    side_lower = side.strip().lower()
    home = _text(fixture.get("home_team"), "fixture.home_team").lower()
    away = _text(fixture.get("away_team"), "fixture.away_team").lower()
    tags = {f"market:{market_group}"}

    if market_group == "goals":
        if side_lower.startswith("over"):
            tags.update({"goal_high", "open_game"})
        elif side_lower.startswith("under"):
            tags.update({"goal_low", "controlled_game"})
    elif market_group == "btts":
        if side_lower in {"yes", "y", "true"}:
            tags.update({"goal_high", "open_game"})
        elif side_lower in {"no", "n", "false"}:
            tags.update({"goal_low", "controlled_game"})
    elif market_group in {"moneyline", "spreads"}:
        if side_lower in {"home", home} or (home and side_lower.startswith(home)):
            tags.add("home_edge")
        elif side_lower in {"away", away} or (away and side_lower.startswith(away)):
            tags.add("away_edge")
        elif side_lower in {"draw", "tie", "x"}:
            tags.add("draw_edge")
    elif market_group in {"corners", "cards", "sot"}:
        if side_lower.startswith("over"):
            tags.add(f"high_{market_group}")
        elif side_lower.startswith("under"):
            tags.add(f"low_{market_group}")
    return tags


def _ranking_key(candidate: _Candidate) -> tuple[float, float, float, float, int, int]:
    """Strongest confidence/edge first with a deterministic final tie-break."""
    return (
        -_CONFIDENCE_RANK.get(candidate.confidence, 0),
        -candidate.value_edge,
        -candidate.expected_value,
        -candidate.model_probability,
        _MARKET_TIEBREAK_ORDER.get(candidate.market_group, 99),
        candidate.result_index,
    )


def _select_compatible(candidates: Sequence[_Candidate]) -> list[_Candidate]:
    selected: list[_Candidate] = []
    for candidate in candidates:
        if len(selected) >= MAX_ACTIONABLE_SELECTIONS:
            break
        if all(_compatible(candidate, existing) for existing in selected):
            selected.append(candidate)
    return selected


def _compatible(left: _Candidate, right: _Candidate) -> bool:
    """Return whether two individual fixture angles can share one thesis.

    This is a compatibility screen, not a correlation model.  It avoids
    internally conflicting market directions and duplicate market families;
    it does *not* make any claim that the legs can be multiplied into a valid
    same-game bet.
    """
    if left.market_group == right.market_group:
        return False
    tags = set(left.tags) | set(right.tags)
    if {"goal_high", "goal_low"}.issubset(tags):
        return False
    if {"home_edge", "away_edge"}.issubset(tags):
        return False
    if "draw_edge" in tags and ("home_edge" in tags or "away_edge" in tags):
        return False
    return True


def _story_tags(selected: Sequence[_Candidate]) -> list[str]:
    tags = set().union(*(candidate.tags for candidate in selected))
    # ``market:*`` supports audit/debugging but is not a user-facing game
    # state label.  Preserve a short stable set of actual thesis tags.
    ordered = [
        "open_game", "controlled_game", "home_edge", "away_edge", "draw_edge",
        "high_corners", "low_corners", "high_cards", "low_cards", "high_sot", "low_sot",
    ]
    return [tag for tag in ordered if tag in tags]


def _pick_label(candidate: _Candidate) -> str:
    line = f" {candidate.line:g}" if candidate.line is not None else ""
    return f"{candidate.side.title()}{line}"


def _candidate_summary(candidate: _Candidate) -> Dict[str, Any]:
    return {
        "result_index": candidate.result_index,
        "market": {"group": candidate.market_group, "key": candidate.market_key},
        "pick": _pick_label(candidate),
        "odds": candidate.odds,
        "bookmaker": _quote_bookmaker(candidate.result),
        "confidence": candidate.confidence,
        "value_edge": candidate.value_edge,
        "model_probability": candidate.model_probability,
        "excluded_from_actionable_set": True,
    }


def _quote_bookmaker(result: Mapping[str, Any]) -> Optional[str]:
    decision = result.get("decision") if isinstance(result.get("decision"), Mapping) else {}
    quote = decision.get("quote") if isinstance(decision.get("quote"), Mapping) else {}
    bookmaker = _text(quote.get("bookmaker"), "decision.quote.bookmaker")
    return bookmaker or None


def _recommended_thesis(
    fixture: Mapping[str, str],
    selected: Sequence[_Candidate],
    tags: Sequence[str],
) -> str:
    core = selected[0]
    fixture_label = f"{fixture['home_team']} vs {fixture['away_team']}"
    core_label = _pick_label(core)
    if "open_game" in tags:
        angle = "an open-game profile"
    elif "controlled_game" in tags:
        angle = "a controlled, lower-event profile"
    elif "home_edge" in tags:
        angle = f"an edge toward {fixture['home_team']}"
    elif "away_edge" in tags:
        angle = f"an edge toward {fixture['away_team']}"
    elif "draw_edge" in tags:
        angle = "a draw-oriented match profile"
    elif tags:
        angle = tags[0].replace("_", " ")
    else:
        angle = "the core market angle"
    if len(selected) == 1:
        tail = "It is the only actionable selection retained by the conservative compatibility screen."
    else:
        tail = (
            f"{len(selected) - 1} supporting individual angle"
            f"{'s' if len(selected) > 2 else ''} align with that read."
        )
    return f"{fixture_label}: {core_label} is the core selection for {angle}. {tail}"


def _non_recommended_status(results: Sequence[Mapping[str, Any]]) -> str:
    statuses = [_decision_status(result) for result in results]
    # A no-bet means the fixture was evaluated but no current price qualified;
    # it should win over unavailable if the two states coexist.
    return "no_bet" if "no_bet" in statuses else "unavailable"


def _non_recommended_thesis(
    fixture: Mapping[str, str],
    results: Sequence[Mapping[str, Any]],
    status: str,
) -> str:
    fixture_label = f"{fixture['home_team']} vs {fixture['away_team']}"
    if status == "no_bet":
        return (
            f"{fixture_label}: the fixture was assessed, but none of the current "
            "market prices cleared the model's release rules."
        )
    return (
        f"{fixture_label}: the fixture cannot be assessed safely yet because the "
        "required model inputs or current market prices are unavailable."
    )


def _reason_summary(results: Sequence[Mapping[str, Any]], status: str) -> list[str]:
    reasons: list[str] = []
    for result in results:
        if _decision_status(result) != status:
            continue
        decision = result.get("decision") if isinstance(result.get("decision"), Mapping) else {}
        reason = _text(decision.get("reason"), "decision.reason")
        if reason and reason not in reasons:
            reasons.append(reason)
    return reasons[:3]


def _non_recommended_game_script(
    results: Sequence[Mapping[str, Any]],
    status: str,
) -> Dict[str, Any]:
    return {
        "schema_version": "match-read-game-script.v1",
        "compiler_version": MATCH_READ_COMPILER_VERSION,
        "tags": [status],
        "selection_relationship": None,
        "selected_result_indexes": [],
        "alternative_candidates": [],
        "market_summary": _market_summary(results),
        "reasons": _reason_summary(results, status),
    }


def _market_summary(results: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    by_status: Dict[str, int] = {"recommended": 0, "no_bet": 0, "unavailable": 0}
    by_group: Dict[str, str] = {}
    for result in results:
        status = _decision_status(result)
        by_status[status] += 1
        market = result.get("market") if isinstance(result.get("market"), Mapping) else {}
        group = _text(market.get("group"), "market.group") or "unknown"
        by_group[group] = status
    return {
        "evaluated_market_count": len(results),
        "by_status": by_status,
        "by_group": by_group,
    }


def _evaluated_at(results: Sequence[Mapping[str, Any]]) -> Optional[str]:
    """Use the source evaluation timestamp when it is consistently supplied."""
    timestamps = []
    for result in results:
        provenance = result.get("provenance")
        if not isinstance(provenance, Mapping):
            continue
        value = _text(provenance.get("generated_at"), "provenance.generated_at")
        if value:
            timestamps.append(value)
    # Different market calls may cross a second boundary.  Keep the earliest
    # observed time rather than manufacturing an average; the raw leaf values
    # remain in the immutable snapshot.
    return min(timestamps) if timestamps else None
