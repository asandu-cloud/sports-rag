"""Read-only canonical inputs for the candidate pre-match feature contract.

This adapter deliberately does not fill gaps with present-day Chroma profiles.
Missing historical coverage is a reportable exclusion, not a fabricated prior.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import sqlite3

from ..registry.loader import load_registry
from Scripts.rag_ingest.core.model_features import capture_snapshot, fixture_identity, number


def supported_competitions() -> dict[str, str]:
    return {entry.code: entry.competition_type for entry in load_registry()
            if entry.enabled and entry.competition_type in {"domestic_league", "continental_cup"}}


def _timestamp(value):
    if not value:
        return None
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    # Canonical SQLite columns are UTC; their serializer omits tzinfo.
    return parsed.replace(tzinfo=timezone.utc).isoformat() if parsed.tzinfo is None else parsed.astimezone(timezone.utc).isoformat()


def load_canonical_inputs(database: Path) -> tuple[list[dict], list[dict], dict]:
    """Consistent short read transaction; no schema changes or provider calls."""
    competitions = supported_competitions()
    with sqlite3.connect(Path(database).resolve().as_uri() + "?mode=ro", uri=True) as db:
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA query_only=ON")
        db.execute("BEGIN")
        fixtures = [dict(row) for row in db.execute("""
            SELECT f.*, c.code AS competition, s.year AS season,
                   s.competition_id AS season_competition_id,
                   h.api_football_id AS home_api_id, a.api_football_id AS away_api_id
            FROM fixtures f JOIN competitions c ON c.id=f.competition_id
            JOIN seasons s ON s.id=f.season_id
            JOIN teams h ON h.id=f.home_team_id JOIN teams a ON a.id=f.away_team_id
            ORDER BY f.kickoff_utc, f.api_football_id
        """)]
        stats = defaultdict(list)
        for row in db.execute("SELECT * FROM fixture_team_stats"):
            stats[row["fixture_id"]].append(dict(row))
    targets, history, quarantined = [], [], []
    coverage = defaultdict(lambda: {"fixtures": 0, "completed": 0})
    for fixture in fixtures:
        if fixture["competition"] not in competitions:
            continue
        key = f"{fixture['competition']}:{fixture['season']}"
        coverage[key]["fixtures"] += 1
        try:
            if fixture["season_competition_id"] != fixture["competition_id"]:
                raise ValueError("Fixture and provider season belong to different competitions")
            item = fixture_identity({"fixture_id": fixture["api_football_id"], "competition": fixture["competition"],
                                     "season": fixture["season"], "home_team_id": fixture["home_api_id"],
                                     "away_team_id": fixture["away_api_id"], "kickoff": _timestamp(fixture["kickoff_utc"])}, competitions)
            observed = _timestamp(fixture["last_fetched_at"])
            item.update(referee=fixture["referee"], referee_observed_at=observed,
                        observed_at=observed, status=fixture["status"])
            target = dict(item)
            if fixture["status"] != "FT":
                targets.append(target)
                continue  # AET/PEN stats cannot be assumed to be 90-minute totals.
            paired = {}
            observed_times = [observed] if observed else []
            for row in stats[fixture["id"]]:
                home = row["team_id"] == fixture["home_team_id"]
                side = "home" if home else "away"
                expected_team = fixture[f"{side}_team_id"]
                expected_opponent = fixture["away_team_id" if home else "home_team_id"]
                if row["team_id"] != expected_team or row["opponent_team_id"] != expected_opponent or bool(row["is_home"]) != home or side in paired:
                    raise ValueError("Ambiguous/mismatched team-stat pairing")
                paired[side] = row
                if row.get("updated_at"):
                    observed_times.append(_timestamp(row["updated_at"]))
            item["observed_at"] = max(observed_times) if observed_times else None
            for side in ("home", "away"):
                row = paired.get(side, {})
                yellow, red = number(row.get("yellow_cards")), number(row.get("red_cards"))
                goals = number(fixture[f"{side}_goals"])
                stats_goals = number(row.get("goals"))
                if goals is not None and stats_goals is not None and goals != stats_goals:
                    goals = None  # Conflicting scores cannot supply a goals label.
                item[side] = {"goals": goals, "corners": number(row.get("corners")),
                              "sot": number(row.get("shots_on")),
                              "cards": yellow + red if yellow is not None and red is not None else None,
                              "shots": number(row.get("shots_total")), "fouls": number(row.get("fouls_committed")),
                              "xg": number(row.get("expected_goals")), "possession": number(row.get("possession"))}
            targets.append(target)
            history.append(item)
            coverage[key]["completed"] += 1
        except (ValueError, TypeError, OverflowError) as exc:
            quarantined.append({"fixture_id": fixture["api_football_id"], "competition": fixture["competition"],
                                "season": fixture["season"], "reason": str(exc)})
    return targets, history, {"sources": dict(coverage), "quarantined": quarantined,
                              "availability_note": "observed_at is latest canonical fetch/update, not an immutable original observation",
                              "period_note": "Only FT fixtures supply history/labels; AET/PEN totals are excluded"}


def capture_fixture_snapshot(database: Path, fixture_id: int, *, as_of: datetime) -> dict:
    """Prospective/candidate inference uses the same inputs as dataset export.

    This is not a website endpoint and does not generate/publish a prediction.
    Retrospective availability assumptions are intentionally not exposed here.
    """
    fixtures, history, _ = load_canonical_inputs(database)
    matches = [row for row in fixtures if row["fixture_id"] == fixture_id]
    if len(matches) != 1:
        raise ValueError(f"Fixture {fixture_id} is missing or quarantined")
    return capture_snapshot(matches[0], history, as_of=as_of, competitions=supported_competitions())
