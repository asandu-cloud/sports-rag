"""Read-only, versioned tracking identities; never infer missing historical facts."""
from datetime import datetime, timezone
import hashlib
import json


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def utc_date(value):
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return (dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)).astimezone(timezone.utc).date().isoformat()
    except (TypeError, ValueError):
        return None


def tracking_identity(result):
    fixture = result.get("fixture") or {}
    market = result.get("market") or {}
    quote = (result.get("decision") or {}).get("quote") or {}
    provenance = result.get("provenance") or {}
    # The priced provider key takes precedence over a broad projection family.
    selection = {"fixture_id": fixture.get("event_id"), "league": fixture.get("league"),
                 "market_key": quote.get("market_key") or market.get("key"),
                 "market_group": market.get("group"), "participant": market.get("participant"),
                 "side": quote.get("side"), "line": quote.get("line"),
                 "bookmaker": quote.get("bookmaker"), "period": quote.get("period") or market.get("period")}
    missing = [key for key in ("fixture_id", "market_key", "side", "bookmaker", "period") if not selection[key]]
    return {"schema": "publication-tracking.v1", "selection": selection,
            "selection_key": digest(selection), "fixture_date": utc_date(fixture.get("kickoff")),
            "system_version": provenance.get("system_version"),
            "pipeline_version": provenance.get("pipeline_version"),
            "model_version": provenance.get("model_version"),
            "input_snapshot_id": provenance.get("input_snapshot_id"),
            # Do not label evaluation/publication time as bookmaker quote time.
            "quote_time": quote.get("quote_time"), "quote_captured_at": quote.get("captured_at"),
            "missing_identity_fields": missing,
            "card_definition_status": "requires_bookmaker_rule" if market.get("group") == "cards" else "not_card_market"}


def decision_identity(result):
    """Stable evaluation identity, excluding only wall-clock refresh fields."""
    value = json.loads(json.dumps(result, allow_nan=False))
    provenance = value.get("provenance") or {}
    provenance.pop("generated_at", None)
    return value


def source_cohort(source):
    if source == "unit_bets":
        return "unit_bet_legs"
    if source == "canonical_shadow":
        return "shadow"
    return "legacy_standalone"
