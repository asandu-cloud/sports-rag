"""Pure validation of archived history before an additive canonical import.

Provider status alone does not establish a regulation-time target. These
checks preserve missing statistics and reject contradictory period evidence.
Competition/season identity and archive provenance are checked by the caller.
"""
from __future__ import annotations

from datetime import datetime, timezone

from Scripts.team_stat_export import statistic_number

from ..sync.upserts import _STAT_KEY_MAP


FRACTIONS = frozenset({"possession", "pass_accuracy"})
CONTINUOUS = FRACTIONS | {"expected_goals", "goals_prevented"}


def _valid_id(value):
    return type(value) is int and value > 0


def _integer(value):
    if isinstance(value, str) and "%" in value:
        return None
    number = statistic_number(value)
    return number if number is not None and number >= 0 and number.is_integer() else None


def _mapping(value):
    return value if isinstance(value, dict) else {}


def regulation_issues(raw_fixture, *, as_of=None):
    """Return ordered exclusion reasons without changing the provider payload.

    ``as_of`` permits deterministic replay. Explicit extra-time/penalty scores,
    including 0–0, are evidence of another period and are never normalized away.
    """
    if not isinstance(raw_fixture, dict):
        return ["invalid_fixture_payload"]
    as_of = as_of or datetime.now(timezone.utc)
    if not isinstance(as_of, datetime) or as_of.tzinfo is None:
        raise ValueError("as_of must be a timezone-aware datetime")
    fixture = _mapping(raw_fixture.get("fixture"))
    teams = _mapping(raw_fixture.get("teams"))
    status = _mapping(fixture.get("status"))
    goals = _mapping(raw_fixture.get("goals"))
    score = _mapping(raw_fixture.get("score"))
    fulltime = _mapping(score.get("fulltime"))
    issues = []
    if not _valid_id(fixture.get("id")):
        issues.append("invalid_fixture_id")
    team_ids = [_mapping(teams.get(side)).get("id") for side in ("home", "away")]
    if not all(_valid_id(team_id) for team_id in team_ids) or team_ids[0] == team_ids[1]:
        issues.append("invalid_fixture_team_ids")
    try:
        kickoff = datetime.fromisoformat(fixture["date"].replace("Z", "+00:00"))
        if kickoff.tzinfo is None or kickoff >= as_of:
            issues.append("invalid_or_future_kickoff")
    except (KeyError, TypeError, ValueError, AttributeError):
        issues.append("invalid_or_future_kickoff")
    if status.get("short") != "FT":
        issues.append("not_regulation_finished")
    elapsed = status.get("elapsed")
    if elapsed is not None and (_integer(elapsed) is None or _integer(elapsed) > 90):
        issues.append("invalid_or_extra_time_elapsed")
    valid_goals = all(_integer(goals.get(side)) is not None for side in ("home", "away"))
    valid_fulltime = all(_integer(fulltime.get(side)) is not None for side in ("home", "away"))
    if not valid_goals:
        issues.append("invalid_final_goals")
    if not valid_fulltime:
        issues.append("missing_or_invalid_fulltime_score")
    if valid_goals and valid_fulltime and any(
        _integer(goals[side]) != _integer(fulltime[side]) for side in ("home", "away")
    ):
        issues.append("goals_fulltime_mismatch")
    for period in ("extratime", "penalty"):
        values = score.get(period)
        if values is not None and not isinstance(values, dict):
            issues.append(f"invalid_{period}_score")
        elif isinstance(values, dict) and any(value is not None for value in values.values()):
            issues.append(f"nonnull_{period}_score")
    return issues


def normalize_team_statistics(raw_fixture, blocks):
    """Return canonical columns keyed by exact provider team ID.

    Empty provider responses remain empty. Explicit null and zero values stay
    distinct, while missing statistic types are omitted. Malformed non-null
    values and inconsistent identities raise ValueError instead of becoming null.
    This function does not certify card settlement semantics or fixture period.
    """
    fixture = _mapping(raw_fixture)
    fixture_id = _mapping(fixture.get("fixture")).get("id")
    teams = _mapping(fixture.get("teams"))
    ids = [_mapping(teams.get(side)).get("id") for side in ("home", "away")]
    if not _valid_id(fixture_id) or not all(_valid_id(team_id) for team_id in ids) or ids[0] == ids[1]:
        raise ValueError("Invalid fixture/team provider identity")
    if not isinstance(blocks, list):
        raise ValueError("Statistics response must be a list")
    if not blocks:
        return {}
    block_ids = [_mapping(_mapping(block).get("team")).get("id") for block in blocks]
    if len(blocks) != 2 or not all(_valid_id(team_id) for team_id in block_ids) or set(block_ids) != set(ids):
        raise ValueError("Statistics must identify exactly the two fixture teams")
    result = {}
    for block, team_id in zip(blocks, block_ids):
        entries = block.get("statistics")
        if not isinstance(entries, list):
            raise ValueError("Team statistics must be a list")
        seen, normalized = set(), {}
        for entry in entries:
            key = _mapping(entry).get("type")
            if not isinstance(key, str) or not key.strip() or key in seen:
                raise ValueError("Missing or duplicate statistic type")
            seen.add(key)
            column = _STAT_KEY_MAP.get(key)
            if column is None:
                continue
            if "value" not in entry:
                raise ValueError("Statistic value is absent: " + key)
            value = entry["value"]
            if value is None:
                normalized[column] = None
                continue
            number = statistic_number(value)
            if (number is None or (column != "goals_prevented" and number < 0)
                    or (column in FRACTIONS and number > 1)
                    or (column not in CONTINUOUS and not number.is_integer())
                    or (isinstance(value, str) and "%" in value and column not in FRACTIONS)):
                raise ValueError("Invalid numeric statistic: " + key)
            normalized[column] = number if column in CONTINUOUS else int(number)
        for part, total in (("shots_on", "shots_total"), ("shots_off", "shots_total"),
                            ("shots_blocked", "shots_total"), ("passes_accurate", "passes_total")):
            if (normalized.get(part) is not None and normalized.get(total) is not None
                    and normalized[part] > normalized[total]):
                raise ValueError(f"Inconsistent statistics: {part} exceeds {total}")
        result[team_id] = normalized
    return result
