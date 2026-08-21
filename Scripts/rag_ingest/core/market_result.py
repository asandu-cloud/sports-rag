"""Versioned, presentation-free contract for a single market decision.

This module deliberately contains no provider calls, model calculations, or
rendering.  It is the boundary between the canonical prediction pipeline and
all delivery surfaces.  Existing selector dictionaries can be adapted through
``decision_from_selector`` while callers are migrated incrementally.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple


MARKET_RESULT_SCHEMA_VERSION = "market_result.v1"


class MarketResultError(ValueError):
    """Raised when a market result does not satisfy the stable contract."""


class DecisionStatus(str, Enum):
    """The only terminal states a market decision may expose."""

    RECOMMENDED = "recommended"
    NO_BET = "no_bet"
    UNAVAILABLE = "unavailable"


def _text(value: Any, field_name: str, *, required: bool = False) -> Optional[str]:
    if value is None:
        if required:
            raise MarketResultError(f"{field_name} is required.")
        return None
    result = str(value).strip()
    if not result:
        if required:
            raise MarketResultError(f"{field_name} is required.")
        return None
    return result


def _number(value: Any, field_name: str, *, minimum: Optional[float] = None,
            maximum: Optional[float] = None) -> Optional[float]:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise MarketResultError(f"{field_name} must be numeric.") from exc
    if not math.isfinite(result):
        raise MarketResultError(f"{field_name} must be finite.")
    if minimum is not None and result < minimum:
        raise MarketResultError(f"{field_name} must be at least {minimum}.")
    if maximum is not None and result > maximum:
        raise MarketResultError(f"{field_name} must be at most {maximum}.")
    return result


def _json_mapping(value: Mapping[str, Any], field_name: str) -> Dict[str, Any]:
    result = {str(key): item for key, item in dict(value or {}).items()}
    try:
        json.dumps(result, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise MarketResultError(f"{field_name} must be JSON serializable.") from exc
    return result


def _optional_metric(option: Mapping[str, Any], *names: str) -> Optional[float]:
    for name in names:
        if name in option:
            return _number(option.get(name), name)
    return None


@dataclass(frozen=True)
class FixtureRef:
    """Stable identity for the fixture evaluated by a market result."""

    event_id: str
    league: str
    home_team: str
    away_team: str
    kickoff: Optional[str] = None

    def __post_init__(self) -> None:
        for name in ("event_id", "league", "home_team", "away_team"):
            value = _text(getattr(self, name), name, required=True)
            object.__setattr__(self, name, value)
        object.__setattr__(self, "kickoff", _text(self.kickoff, "kickoff"))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "league": self.league,
            "home_team": self.home_team,
            "away_team": self.away_team,
            "kickoff": self.kickoff,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FixtureRef":
        return cls(
            event_id=data.get("event_id") or data.get("id") or "",
            league=data.get("league") or "",
            home_team=data.get("home_team") or "",
            away_team=data.get("away_team") or "",
            kickoff=data.get("kickoff"),
        )


@dataclass(frozen=True)
class MarketRef:
    """Identity of the exact market evaluated, including team-specific markets."""

    key: str
    group: str
    unit: str
    participant: Optional[str] = None

    def __post_init__(self) -> None:
        for name in ("key", "group", "unit"):
            value = _text(getattr(self, name), name, required=True)
            object.__setattr__(self, name, value)
        object.__setattr__(self, "participant", _text(self.participant, "participant"))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "group": self.group,
            "unit": self.unit,
            "participant": self.participant,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MarketRef":
        return cls(
            key=data.get("key") or data.get("market_key") or "",
            group=data.get("group") or data.get("market_group") or "",
            unit=data.get("unit") or "count",
            participant=data.get("participant"),
        )


@dataclass(frozen=True)
class Projection:
    """Numeric model output before bookmaker selection is applied."""

    value: Optional[float]
    unit: str
    season_component: Optional[float] = None
    recent_component: Optional[float] = None
    variance: Optional[float] = None
    components: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", _number(self.value, "projection.value"))
        object.__setattr__(self, "unit", _text(self.unit, "projection.unit", required=True))
        object.__setattr__(self, "season_component", _number(
            self.season_component, "projection.season_component"
        ))
        object.__setattr__(self, "recent_component", _number(
            self.recent_component, "projection.recent_component"
        ))
        object.__setattr__(self, "variance", _number(self.variance, "projection.variance", minimum=0.0))
        object.__setattr__(self, "components", _json_mapping(self.components, "projection.components"))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "value": self.value,
            "unit": self.unit,
            "season_component": self.season_component,
            "recent_component": self.recent_component,
            "variance": self.variance,
            "components": dict(self.components),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Projection":
        return cls(
            value=data.get("value"),
            unit=data.get("unit") or "count",
            season_component=data.get("season_component"),
            recent_component=data.get("recent_component"),
            variance=data.get("variance"),
            components=data.get("components") or {},
        )


@dataclass(frozen=True)
class PriceQuote:
    """The exact offered price used when a decision is made."""

    side: str
    line: Optional[float] = None
    odds: Optional[float] = None
    bookmaker: Optional[str] = None
    market_key: Optional[str] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "side", _text(self.side, "price.side", required=True))
        object.__setattr__(self, "line", _number(self.line, "price.line"))
        odds = _number(self.odds, "price.odds", minimum=1.0)
        if odds is not None and odds <= 1.0:
            raise MarketResultError("price.odds must be greater than 1.0.")
        object.__setattr__(self, "odds", odds)
        object.__setattr__(self, "bookmaker", _text(self.bookmaker, "price.bookmaker"))
        object.__setattr__(self, "market_key", _text(self.market_key, "price.market_key"))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "side": self.side,
            "line": self.line,
            "odds": self.odds,
            "bookmaker": self.bookmaker,
            "market_key": self.market_key,
        }

    @classmethod
    def from_option(cls, option: Mapping[str, Any]) -> "PriceQuote":
        return cls(
            side=option.get("side") or option.get("outcome") or "",
            line=option.get("line", option.get("point")),
            odds=option.get("odds"),
            bookmaker=option.get("bookmaker"),
            market_key=option.get("market_key"),
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PriceQuote":
        return cls(
            side=data.get("side") or "",
            line=data.get("line", data.get("point")),
            odds=data.get("odds"),
            bookmaker=data.get("bookmaker"),
            market_key=data.get("market_key"),
        )


@dataclass(frozen=True)
class Decision:
    """Decision and price-aware metrics for one market."""

    status: DecisionStatus
    quote: Optional[PriceQuote] = None
    model_probability: Optional[float] = None
    implied_probability: Optional[float] = None
    value_edge: Optional[float] = None
    expected_value: Optional[float] = None
    confidence: Optional[str] = None
    reason: Optional[str] = None

    def __post_init__(self) -> None:
        status = self.status if isinstance(self.status, DecisionStatus) else DecisionStatus(str(self.status))
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "model_probability", _number(
            self.model_probability, "decision.model_probability", minimum=0.0, maximum=1.0
        ))
        object.__setattr__(self, "implied_probability", _number(
            self.implied_probability, "decision.implied_probability", minimum=0.0, maximum=1.0
        ))
        object.__setattr__(self, "value_edge", _number(self.value_edge, "decision.value_edge"))
        object.__setattr__(self, "expected_value", _number(self.expected_value, "decision.expected_value"))
        object.__setattr__(self, "confidence", _text(self.confidence, "decision.confidence"))
        object.__setattr__(self, "reason", _text(self.reason, "decision.reason"))

        if status is DecisionStatus.RECOMMENDED:
            if self.quote is None or self.quote.odds is None:
                raise MarketResultError("A recommended decision requires a priced quote.")
            if self.model_probability is None:
                raise MarketResultError("A recommended decision requires model_probability.")
            if self.reason is not None:
                raise MarketResultError("A recommended decision cannot carry a no-bet reason.")
        elif self.reason is None:
            raise MarketResultError(f"A {status.value} decision requires a reason.")

    @property
    def is_recommended(self) -> bool:
        return self.status is DecisionStatus.RECOMMENDED

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "quote": self.quote.to_dict() if self.quote else None,
            "model_probability": self.model_probability,
            "implied_probability": self.implied_probability,
            "value_edge": self.value_edge,
            "expected_value": self.expected_value,
            "confidence": self.confidence,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Decision":
        quote = data.get("quote")
        return cls(
            status=data.get("status") or DecisionStatus.UNAVAILABLE.value,
            quote=PriceQuote.from_dict(quote) if isinstance(quote, Mapping) else None,
            model_probability=data.get("model_probability"),
            implied_probability=data.get("implied_probability"),
            value_edge=data.get("value_edge"),
            expected_value=data.get("expected_value"),
            confidence=data.get("confidence"),
            reason=data.get("reason"),
        )


@dataclass(frozen=True)
class EvidenceItem:
    """Renderer-independent reason code with JSON-safe observed values."""

    code: str
    values: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "code", _text(self.code, "evidence.code", required=True))
        object.__setattr__(self, "values", _json_mapping(self.values, "evidence.values"))

    def to_dict(self) -> Dict[str, Any]:
        return {"code": self.code, "values": dict(self.values)}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EvidenceItem":
        return cls(code=data.get("code") or "", values=data.get("values") or {})


@dataclass(frozen=True)
class Provenance:
    """Information needed to reproduce or audit a decision later."""

    pipeline_version: str
    input_snapshot_id: str
    generated_at: str
    model_version: Optional[str] = None
    as_of: Optional[str] = None
    sources: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in ("pipeline_version", "input_snapshot_id", "generated_at"):
            value = _text(getattr(self, name), f"provenance.{name}", required=True)
            object.__setattr__(self, name, value)
        object.__setattr__(self, "model_version", _text(self.model_version, "provenance.model_version"))
        object.__setattr__(self, "as_of", _text(self.as_of, "provenance.as_of"))
        object.__setattr__(self, "sources", tuple(
            source for source in (_text(item, "provenance.sources") for item in self.sources) if source
        ))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pipeline_version": self.pipeline_version,
            "input_snapshot_id": self.input_snapshot_id,
            "generated_at": self.generated_at,
            "model_version": self.model_version,
            "as_of": self.as_of,
            "sources": list(self.sources),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Provenance":
        return cls(
            pipeline_version=data.get("pipeline_version") or "",
            input_snapshot_id=data.get("input_snapshot_id") or "",
            generated_at=data.get("generated_at") or "",
            model_version=data.get("model_version"),
            as_of=data.get("as_of"),
            sources=tuple(data.get("sources") or ()),
        )


@dataclass(frozen=True)
class MarketResult:
    """Stable v1 contract returned by the future canonical market service."""

    fixture: FixtureRef
    market: MarketRef
    projection: Projection
    decision: Decision
    provenance: Provenance
    evidence: Tuple[EvidenceItem, ...] = ()
    context: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = MARKET_RESULT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != MARKET_RESULT_SCHEMA_VERSION:
            raise MarketResultError(
                f"Unsupported market-result schema: {self.schema_version!r}."
            )
        if self.market.unit != self.projection.unit:
            raise MarketResultError("market.unit and projection.unit must match.")
        evidence = tuple(self.evidence)
        if not all(isinstance(item, EvidenceItem) for item in evidence):
            raise MarketResultError("evidence must contain EvidenceItem instances.")
        object.__setattr__(self, "evidence", evidence)
        object.__setattr__(self, "context", _json_mapping(self.context, "context"))

    @property
    def is_recommended(self) -> bool:
        return self.decision.is_recommended

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "fixture": self.fixture.to_dict(),
            "market": self.market.to_dict(),
            "projection": self.projection.to_dict(),
            "decision": self.decision.to_dict(),
            "provenance": self.provenance.to_dict(),
            "evidence": [item.to_dict() for item in self.evidence],
            "context": dict(self.context),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MarketResult":
        payload = dict(data or {})
        schema_version = payload.get("schema_version") or ""
        if schema_version != MARKET_RESULT_SCHEMA_VERSION:
            raise MarketResultError(f"Unsupported market-result schema: {schema_version!r}.")
        evidence = payload.get("evidence") or []
        if not isinstance(evidence, Iterable) or isinstance(evidence, (str, bytes, Mapping)):
            raise MarketResultError("evidence must be a list of evidence items.")
        evidence_items = []
        for item in evidence:
            if not isinstance(item, Mapping):
                raise MarketResultError("Each evidence item must be an object.")
            evidence_items.append(EvidenceItem.from_dict(item))
        return cls(
            fixture=FixtureRef.from_dict(payload.get("fixture") or {}),
            market=MarketRef.from_dict(payload.get("market") or {}),
            projection=Projection.from_dict(payload.get("projection") or {}),
            decision=Decision.from_dict(payload.get("decision") or {}),
            provenance=Provenance.from_dict(payload.get("provenance") or {}),
            evidence=tuple(evidence_items),
            context=payload.get("context") or {},
            schema_version=schema_version,
        )

    @classmethod
    def from_json(cls, value: str) -> "MarketResult":
        try:
            data = json.loads(value)
        except (TypeError, ValueError) as exc:
            raise MarketResultError("Market result JSON is invalid.") from exc
        if not isinstance(data, Mapping):
            raise MarketResultError("Market result JSON must contain an object.")
        return cls.from_dict(data)


def decision_from_selector(selector_result: Mapping[str, Any]) -> Decision:
    """Adapt a current ``core.line_selection`` response without rendering text.

    The adapter is deliberately pure.  It gives Phase 3 a safe compatibility
    bridge while old selector return dictionaries coexist with ``Decision``.
    Empty line lists mean the price is unavailable; a populated list with no
    eligible recommendation is a genuine no-bet decision.
    """

    result = dict(selector_result or {})
    selected = result.get("bet_recommendation") or result.get("recommended")
    if isinstance(selected, Mapping):
        quote = PriceQuote.from_option(selected)
        return Decision(
            status=DecisionStatus.RECOMMENDED,
            quote=quote,
            model_probability=_optional_metric(selected, "model_probability", "_model_prob"),
            implied_probability=_optional_metric(selected, "implied_probability", "_implied_prob"),
            value_edge=_optional_metric(selected, "value_edge", "_value_edge"),
            expected_value=_optional_metric(selected, "expected_value", "_ev"),
            confidence=_text(selected.get("confidence"), "selector.confidence"),
        )

    candidates = result.get("all_lines")
    if candidates is None:
        candidates = result.get("all_sides")
    has_candidates = bool(candidates)
    best_value = result.get("best_value")
    option = best_value if isinstance(best_value, Mapping) else None
    quote = PriceQuote.from_option(option) if option else None
    reason = _text(result.get("no_bet_reason"), "selector.no_bet_reason")
    if not reason:
        reason = "No eligible recommendation was returned by the line selector."
    return Decision(
        status=DecisionStatus.NO_BET if has_candidates else DecisionStatus.UNAVAILABLE,
        quote=quote,
        model_probability=_optional_metric(option, "model_probability", "_model_prob") if option else None,
        implied_probability=_optional_metric(option, "implied_probability", "_implied_prob") if option else None,
        value_edge=_optional_metric(option, "value_edge", "_value_edge") if option else None,
        expected_value=_optional_metric(option, "expected_value", "_ev") if option else None,
        confidence=_text(option.get("confidence"), "selector.confidence") if option else None,
        reason=reason,
    )
