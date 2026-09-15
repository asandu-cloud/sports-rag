"""Shared, read-only card shape for website and Discord Match Reads.

The database record is intentionally rich and archival.  Delivery surfaces
need a small common view of it so neither is tempted to parse legacy text or
re-run a model independently.  This module does not decide access, layout,
release timing, or whether alternatives are shown; those remain product-layer
choices.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence


MATCH_READ_CARD_SCHEMA_VERSION = "match-read-card.v2"


class MatchReadCardError(ValueError):
    """Raised when an incomplete persisted Match Read is adapted for a card."""


def build_match_read_card(read: Mapping[str, Any]) -> Dict[str, Any]:
    """Project one persisted Match Read into the common delivery contract.

    The result intentionally preserves the fact that individual selections
    are not a same-game package.  ``packages`` only carries a real quote from
    the persisted record; it never derives a combined price/probability.
    """
    source = dict(read or {})
    fixture = _mapping(source.get("fixture"), "fixture")
    script = _mapping_or_empty(source.get("game_script"))
    status = _required_text(source.get("status"), "status")
    timing = _mapping_or_empty(script.get("publication_timing"))
    selections = source.get("selections")
    if not isinstance(selections, Sequence) or isinstance(selections, (str, bytes)):
        raise MatchReadCardError("selections must be a sequence.")
    packages = source.get("packages")
    if not isinstance(packages, Sequence) or isinstance(packages, (str, bytes)):
        raise MatchReadCardError("packages must be a sequence.")

    return {
        "schema_version": MATCH_READ_CARD_SCHEMA_VERSION,
        "id": _required_int(source.get("id"), "id"),
        "fixture": {
            "event_id": _required_text(fixture.get("event_id"), "fixture.event_id"),
            "league": _required_text(fixture.get("league"), "fixture.league"),
            "home_team": _required_text(fixture.get("home_team"), "fixture.home_team"),
            "away_team": _required_text(fixture.get("away_team"), "fixture.away_team"),
            "kickoff": fixture.get("kickoff"),
        },
        "stage": _required_text(source.get("stage"), "stage"),
        "version": _required_int(source.get("version"), "version"),
        "status": status,
        "thesis": _required_text(source.get("thesis"), "thesis"),
        # The briefing is authored and persisted before delivery.  Neither
        # website nor Discord should call a model or infer this copy at view
        # time.  Older records remain readable with an empty briefing.
        "briefing": _briefing_card(script.get("briefing")),
        "visuals": _visuals_card(script.get("visuals")),
        "timing": {
            "is_preliminary": _optional_text(timing.get("state")) == "preliminary",
        },
        "game_script": {
            "tags": _string_list(script.get("tags")),
            "selection_relationship": _optional_text(script.get("selection_relationship")),
            # This is data for a product layer to opt into; it is not an
            # instruction to show each candidate as an official tracked bet.
            "alternative_candidates": _mapping_list(script.get("alternative_candidates")),
        },
        "selections": [_selection_card(item) for item in selections],
        "packages": [_package_card(item) for item in packages],
        "provenance": _mapping_or_empty(source.get("provenance")),
        "created_at": source.get("created_at"),
    }


def _selection_card(value: Any) -> Dict[str, Any]:
    selection = _mapping(value, "selection")
    data = _mapping_or_empty(selection.get("data"))
    decision = _mapping_or_empty(data.get("decision"))
    evidence = data.get("evidence")
    return {
        "position": _required_int(selection.get("position"), "selection.position"),
        "role": _required_text(selection.get("role"), "selection.role"),
        "market": _mapping(selection.get("market"), "selection.market"),
        "pick": _required_text(selection.get("pick"), "selection.pick"),
        "odds": _number(selection.get("quote_odds")),
        "bookmaker": _optional_text(selection.get("bookmaker")),
        "model_probability": _number(selection.get("model_probability")),
        "value_edge": _number(selection.get("value_edge")),
        "confidence": _optional_text(decision.get("confidence")),
        "published_recommendation_id": selection.get("published_recommendation_id"),
        "evidence": _mapping_list(evidence),
    }


def _package_card(value: Any) -> Dict[str, Any]:
    package = _mapping(value, "package")
    return {
        "position": _required_int(package.get("position"), "package.position"),
        "kind": _required_text(package.get("kind"), "package.kind"),
        "status": _required_text(package.get("status"), "package.status"),
        "label": _required_text(package.get("label"), "package.label"),
        "rationale": _optional_text(package.get("rationale")),
        "quoted_odds": _number(package.get("quoted_odds")),
        "bookmaker": _optional_text(package.get("bookmaker")),
        # The persistence contract intentionally leaves these null until a
        # correlation-aware model exists; do not add derived values here.
        "joint_model_probability": None,
        "expected_value": None,
        "selection_positions": list(package.get("selection_positions") or []),
    }


def _mapping(value: Any, field: str) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        raise MatchReadCardError(f"{field} must be an object.")
    return dict(value)


def _mapping_or_empty(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _mapping_list(value: Any) -> list[Dict[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)]


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return [str(item) for item in value if str(item).strip()]


def _briefing_card(value: Any) -> Dict[str, Any]:
    briefing = _mapping_or_empty(value)
    bullets = _string_list(briefing.get("bullets"))[:3]
    return {
        "summary": _optional_text(briefing.get("summary")),
        "bullets": bullets,
        "source": _optional_text(briefing.get("source")),
        "model": _optional_text(briefing.get("model")),
    }


def _visuals_card(value: Any) -> Dict[str, str | None]:
    visuals = _mapping_or_empty(value)
    return {
        "home_team_logo": _safe_image_url(visuals.get("home_team_logo")),
        "away_team_logo": _safe_image_url(visuals.get("away_team_logo")),
        "league_logo": _safe_image_url(visuals.get("league_logo")),
    }


def _safe_image_url(value: Any) -> str | None:
    url = _optional_text(value)
    return url if url and url.lower().startswith(("https://", "http://")) else None


def _required_text(value: Any, field: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise MatchReadCardError(f"{field} is required.")
    return text


def _optional_text(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _required_int(value: Any, field: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise MatchReadCardError(f"{field} must be an integer.") from exc


def _number(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
