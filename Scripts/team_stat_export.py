"""Lossless missingness and source evidence for legacy team-stat exports.

The old exports replaced provider nulls with zero. Only new rows carrying
matching provider evidence may supply verified zeros to historical recovery.
"""
from __future__ import annotations

from datetime import datetime, timezone
import math


EVIDENCE_KEY = "_provider_statistics"
EVIDENCE_SCHEMA = "api-football-team-statistics.v1"


def statistic_number(value):
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value.strip().removesuffix("%")) if isinstance(value, str) else float(value)
        if isinstance(value, str) and value.strip().endswith("%"):
            result /= 100.0
        return result if math.isfinite(result) else None
    except (TypeError, ValueError, OverflowError):
        return None


def _provider_id(value):
    if type(value) is not int or value <= 0:
        raise ValueError("Missing or invalid provider identity in team statistics")
    return value


def export_statistics(block, fixture_id):
    """Preserve explicit zero, null and omitted fields; retain raw evidence."""
    fixture_id = _provider_id(fixture_id)
    team_id = _provider_id(block["team"]["id"])
    raw = {}
    for item in block["statistics"]:
        key = item.get("type")
        if not isinstance(key, str) or not key or key.startswith("_") or key in raw:
            raise ValueError("Missing or duplicate statistic type")
        raw[key] = item.get("value")
    return {
        **{key: statistic_number(value) for key, value in raw.items()},
        EVIDENCE_KEY: {
            "schema": EVIDENCE_SCHEMA, "fixture_id": fixture_id, "team_id": team_id,
            "observed_at": datetime.now(timezone.utc).isoformat(), "values": raw,
        },
    }


def verified_export_values(row, *, fixture_id, team_id):
    """Validate new export evidence; never certify zeros from an old row."""
    evidence = row.get(EVIDENCE_KEY)
    if evidence is None:
        return {}
    if not isinstance(evidence, dict) or evidence.get("schema") != EVIDENCE_SCHEMA:
        raise ValueError("Unsupported team-stat export evidence")
    if (_provider_id(evidence.get("fixture_id")) != fixture_id
            or _provider_id(evidence.get("team_id")) != team_id):
        raise ValueError("Team-stat export evidence identity mismatch")
    observed = datetime.fromisoformat(evidence["observed_at"])
    if observed.tzinfo is None:
        raise ValueError("Team-stat export observation must include a timezone")
    raw = evidence.get("values")
    if not isinstance(raw, dict):
        raise ValueError("Missing provider statistic evidence")
    normalized = {key: statistic_number(value) for key, value in raw.items()}
    for key, value in normalized.items():
        if key not in row or statistic_number(row[key]) != value:
            raise ValueError("Team-stat export differs from provider evidence")
    return normalized
