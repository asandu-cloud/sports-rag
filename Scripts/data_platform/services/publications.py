"""Canonical boundary for recording recommendations that users can see."""

from __future__ import annotations

from datetime import date, datetime, timezone
import hashlib
import json
from typing import Any, Dict, Mapping, Optional

from ..repositories.publications import PublicationRepository


PUBLISHED_PREDICTION_SOURCE = "canonical_published"
VALID_PUBLICATION_SURFACES = frozenset({"discord", "website"})


class PublicationValidationError(ValueError):
    """Raised when a payload is not a priced canonical recommendation."""


def _as_mapping(value: Any) -> Dict[str, Any]:
    to_dict = getattr(value, "to_dict", None)
    payload = to_dict() if callable(to_dict) else value
    if not isinstance(payload, Mapping):
        raise PublicationValidationError("A publication must receive a MarketResult mapping.")
    try:
        # This both detaches the immutable snapshot from caller-owned objects
        # and rejects values that cannot be safely stored as audit evidence.
        return json.loads(json.dumps(dict(payload), sort_keys=True, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise PublicationValidationError("The canonical recommendation must be JSON serializable.") from exc


def _required_text(payload: Mapping[str, Any], field: str) -> str:
    value = str(payload.get(field) or "").strip()
    if not value:
        raise PublicationValidationError(f"{field} is required to publish a recommendation.")
    return value


def _optional_number(value: Any, field: str) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise PublicationValidationError(f"{field} must be numeric.") from exc


def _as_utc(value: Optional[Any]) -> datetime:
    if value is None:
        return datetime.now(timezone.utc)
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    text = str(value).strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise PublicationValidationError("published_at must be an ISO-8601 timestamp.") from exc
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _season_for(value: date) -> str:
    start_year = value.year if value.month >= 7 else value.year - 1
    return f"{start_year}/{str(start_year + 1)[-2:]}"


def _pick_label(quote: Mapping[str, Any]) -> str:
    side = _required_text(quote, "side")
    line = _optional_number(quote.get("line"), "quote.line")
    return f"{side.title()} {line:g}" if line is not None else side.title()


def _hash_key(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class PublicationService:
    """Persist only selections explicitly released through Discord or web."""

    def __init__(self, repo: Optional[PublicationRepository] = None):
        self._repo = repo or PublicationRepository()

    def publish(
        self,
        market_result: Any,
        *,
        surface: str,
        external_reference: Optional[str] = None,
        published_at: Optional[Any] = None,
        delivery_metadata: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Record one user-visible canonical recommendation and its delivery.

        This is deliberately not a generic ``log_prediction`` wrapper:
        research outputs, command responses, shadow candidates and unpriced
        no-bets cannot enter the official cohort through this method.
        """
        result = _as_mapping(market_result)
        normalised_surface = str(surface or "").strip().lower()
        if normalised_surface not in VALID_PUBLICATION_SURFACES:
            valid = ", ".join(sorted(VALID_PUBLICATION_SURFACES))
            raise PublicationValidationError(f"surface must be one of: {valid}.")

        fixture = result.get("fixture") if isinstance(result.get("fixture"), Mapping) else {}
        market = result.get("market") if isinstance(result.get("market"), Mapping) else {}
        projection = result.get("projection") if isinstance(result.get("projection"), Mapping) else {}
        decision = result.get("decision") if isinstance(result.get("decision"), Mapping) else {}
        provenance = result.get("provenance") if isinstance(result.get("provenance"), Mapping) else {}
        quote = decision.get("quote") if isinstance(decision.get("quote"), Mapping) else {}

        if decision.get("status") != "recommended":
            raise PublicationValidationError("Only a priced recommended decision can be published.")

        event_id = _required_text(fixture, "event_id")
        league = _required_text(fixture, "league")
        home_team = _required_text(fixture, "home_team")
        away_team = _required_text(fixture, "away_team")
        market_group = _required_text(market, "group")
        market_key = _required_text(market, "key")
        pipeline_version = _required_text(provenance, "pipeline_version")
        input_snapshot_id = _required_text(provenance, "input_snapshot_id")
        model_probability = _optional_number(decision.get("model_probability"), "decision.model_probability")
        odds = _optional_number(quote.get("odds"), "quote.odds")
        if model_probability is None or odds is None or odds <= 1.0:
            raise PublicationValidationError("A publication requires a model probability and decimal odds above 1.0.")

        release_time = _as_utc(published_at)
        recommendation_key = _hash_key({
            "event_id": event_id,
            "market_key": market_key,
            "market_group": market_group,
            "quote": dict(quote),
            "input_snapshot_id": input_snapshot_id,
            "pipeline_version": pipeline_version,
        })
        delivery_key = _hash_key({
            "recommendation_key": recommendation_key,
            "surface": normalised_surface,
            "external_reference": str(external_reference or "default"),
        })
        prediction_fields = {
            "fixture_api_id": event_id,
            "home_team": home_team,
            "away_team": away_team,
            "league": league,
            "kickoff": fixture.get("kickoff"),
            "market": market_group,
            "pick": _pick_label(quote),
            "side": _required_text(quote, "side"),
            "line": _optional_number(quote.get("line"), "quote.line"),
            "odds": odds,
            "bookmaker": str(quote.get("bookmaker") or "").strip() or None,
            "model_prob": model_probability,
            "implied_prob": _optional_number(decision.get("implied_probability"), "decision.implied_probability"),
            "value_edge": _optional_number(decision.get("value_edge"), "decision.value_edge"),
            "confidence": str(decision.get("confidence") or "").strip() or None,
            "projected_total": _optional_number(projection.get("value"), "projection.value"),
            "source": PUBLISHED_PREDICTION_SOURCE,
            "season": _season_for(release_time.date()),
            "prediction_date": release_time.date(),
            "extras": {
                "tracking_cohort": "published",
                "recommendation_key": recommendation_key,
                "market_result_schema": result.get("schema_version"),
            },
        }
        return self._repo.record(
            recommendation_key=recommendation_key,
            prediction_fields=prediction_fields,
            released_at=release_time,
            input_snapshot_id=input_snapshot_id,
            pipeline_version=pipeline_version,
            model_version=str(provenance.get("model_version") or "").strip() or None,
            decision_json=result,
            delivery_key=delivery_key,
            surface=normalised_surface,
            external_reference=str(external_reference) if external_reference is not None else None,
            delivered_at=release_time,
            delivery_metadata=delivery_metadata,
        )
