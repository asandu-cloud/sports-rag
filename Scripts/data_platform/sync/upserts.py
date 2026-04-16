"""Pure upsert helpers for fixtures, stats, features, etc.

Split out from the pipeline so the same primitives work for both
bootstrap backfill, incremental refresh, and the JSON-file importer
used to bring historical ``Output/*`` data into the DB without re-hitting
API-Football.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import date, datetime, timezone
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..models import (
    Competition,
    Fixture,
    FixturePlayerStats,
    FixtureTeamStats,
    Player,
    PlayerFeatureSnapshot,
    Season,
    SyncWatermark,
    Team,
    TeamFeatureSnapshot,
)
from .apifootball import CompetitionSpec

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _digest(payload: Any) -> str:
    body = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(body).hexdigest()


def _parse_dt(value: Any) -> Optional[datetime]:
    """Accept ISO strings or numeric epochs (API-Football mixes the two)."""
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(float(value), tz=timezone.utc)
        except (ValueError, OSError):
            return None
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    return None


def _euro_season_label(year: int) -> str:
    return f"{year}/{str(year + 1)[-2:]}"


def _first_numeric(value: Any) -> Optional[float]:
    """API-Football stats come as ints, floats, strings, or ``"62%"``."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        v = value.strip().replace("%", "")
        try:
            f = float(v)
        except ValueError:
            return None
        return f / 100.0 if "%" in value else f
    return None


def _coerce_int(value: Any) -> Optional[int]:
    try:
        f = _first_numeric(value)
        return None if f is None else int(f)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Catalogue upserts
# ---------------------------------------------------------------------------


def upsert_competition(session: Session, spec: CompetitionSpec) -> Competition:
    row = session.scalar(
        select(Competition).where(Competition.api_football_id == spec.api_football_id)
    )
    if row is None:
        row = Competition(
            code=spec.code,
            name=spec.name,
            country=spec.country,
            api_football_id=spec.api_football_id,
            competition_type=spec.competition_type,
        )
        session.add(row)
        session.flush()
    else:
        row.code = spec.code
        row.name = spec.name
        row.country = spec.country
        row.competition_type = spec.competition_type
    return row


def upsert_season(
    session: Session,
    *,
    competition: Competition,
    year: int,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    is_current: Optional[bool] = None,
) -> Season:
    row = session.scalar(
        select(Season).where(
            Season.competition_id == competition.id,
            Season.year == year,
        )
    )
    if row is None:
        row = Season(
            competition_id=competition.id,
            year=year,
            label=_euro_season_label(year),
            start_date=start_date,
            end_date=end_date,
            is_current=bool(is_current) if is_current is not None else False,
        )
        session.add(row)
        session.flush()
    else:
        if start_date is not None:
            row.start_date = start_date
        if end_date is not None:
            row.end_date = end_date
        if is_current is not None:
            row.is_current = bool(is_current)
    return row


def upsert_team(session: Session, *, api_id: int, name: str, **extra: Any) -> Team:
    row = session.scalar(select(Team).where(Team.api_football_id == api_id))
    if row is None:
        row = Team(api_football_id=api_id, name=name, **{k: v for k, v in extra.items() if v is not None})
        session.add(row)
        session.flush()
    else:
        if name and row.name != name:
            row.name = name
        for key, value in extra.items():
            if value is None:
                continue
            if hasattr(row, key):
                setattr(row, key, value)
    return row


def upsert_player(session: Session, *, api_id: int, name: str, **extra: Any) -> Player:
    row = session.scalar(select(Player).where(Player.api_football_id == api_id))
    if row is None:
        row = Player(api_football_id=api_id, name=name, **{k: v for k, v in extra.items() if v is not None})
        session.add(row)
        session.flush()
    else:
        if name and row.name != name:
            row.name = name
        for key, value in extra.items():
            if value is None:
                continue
            if hasattr(row, key):
                setattr(row, key, value)
    return row


# ---------------------------------------------------------------------------
# Fixture upserts from API-Football /fixtures row
# ---------------------------------------------------------------------------


def upsert_fixture_from_api_row(
    session: Session,
    *,
    api_row: Mapping[str, Any],
    competition: Competition,
    season: Season,
) -> Tuple[Fixture, bool]:
    """Upsert a fixture row. Returns ``(fixture, is_new_or_changed)``.

    The change flag lets the caller decide whether to re-fetch stats.
    """
    fixture_block = api_row.get("fixture") or {}
    teams_block = api_row.get("teams") or {}
    goals_block = api_row.get("goals") or {}
    league_block = api_row.get("league") or {}

    api_fid = fixture_block.get("id")
    if api_fid is None:
        raise ValueError("Missing fixture.id in API row")

    home = teams_block.get("home") or {}
    away = teams_block.get("away") or {}
    if not home.get("id") or not away.get("id"):
        raise ValueError(f"Missing team ids on fixture {api_fid}")

    home_team = upsert_team(session, api_id=home["id"], name=home.get("name") or f"Team {home['id']}",
                            logo_url=home.get("logo"))
    away_team = upsert_team(session, api_id=away["id"], name=away.get("name") or f"Team {away['id']}",
                            logo_url=away.get("logo"))

    kickoff_utc = _parse_dt(fixture_block.get("date"))
    last_update = _parse_dt(fixture_block.get("timestamp"))

    status_block = fixture_block.get("status") or {}
    status = status_block.get("short")
    elapsed = _coerce_int(status_block.get("elapsed"))

    payload_digest = _digest({
        "teams": teams_block,
        "goals": goals_block,
        "status": status_block,
        "league_round": league_block.get("round"),
        "venue": fixture_block.get("venue"),
        "referee": fixture_block.get("referee"),
        "kickoff": fixture_block.get("date"),
    })

    existing = session.scalar(select(Fixture).where(Fixture.api_football_id == api_fid))
    changed = False
    if existing is None:
        fixture = Fixture(
            api_football_id=api_fid,
            competition_id=competition.id,
            season_id=season.id,
            home_team_id=home_team.id,
            away_team_id=away_team.id,
            kickoff_utc=kickoff_utc,
            venue=(fixture_block.get("venue") or {}).get("name"),
            city=(fixture_block.get("venue") or {}).get("city"),
            status=status,
            elapsed=elapsed,
            home_goals=_coerce_int(goals_block.get("home")),
            away_goals=_coerce_int(goals_block.get("away")),
            round=league_block.get("round"),
            referee=fixture_block.get("referee"),
            last_fetched_at=datetime.now(timezone.utc),
            last_api_updated_at=last_update,
            payload_digest=payload_digest,
        )
        session.add(fixture)
        session.flush()
        changed = True
    else:
        fixture = existing
        fixture.last_fetched_at = datetime.now(timezone.utc)
        if fixture.payload_digest != payload_digest:
            fixture.competition_id = competition.id
            fixture.season_id = season.id
            fixture.home_team_id = home_team.id
            fixture.away_team_id = away_team.id
            fixture.kickoff_utc = kickoff_utc
            fixture.venue = (fixture_block.get("venue") or {}).get("name")
            fixture.city = (fixture_block.get("venue") or {}).get("city")
            fixture.status = status
            fixture.elapsed = elapsed
            fixture.home_goals = _coerce_int(goals_block.get("home"))
            fixture.away_goals = _coerce_int(goals_block.get("away"))
            fixture.round = league_block.get("round")
            fixture.referee = fixture_block.get("referee")
            fixture.last_api_updated_at = last_update
            fixture.payload_digest = payload_digest
            changed = True
    return fixture, changed


# ---------------------------------------------------------------------------
# Fixture statistics upserts
# ---------------------------------------------------------------------------


_STAT_KEY_MAP = {
    "Shots on Goal": "shots_on",
    "Shots off Goal": "shots_off",
    "Total Shots": "shots_total",
    "Blocked Shots": "shots_blocked",
    "Shots insidebox": "shots_inside_box",
    "Shots outsidebox": "shots_outside_box",
    "Fouls": "fouls_committed",
    "Corner Kicks": "corners",
    "Offsides": "offsides",
    "Ball Possession": "possession",
    "Yellow Cards": "yellow_cards",
    "Red Cards": "red_cards",
    "Goalkeeper Saves": "goalkeeper_saves",
    "Total passes": "passes_total",
    "Passes accurate": "passes_accurate",
    "Passes %": "pass_accuracy",
    "expected_goals": "expected_goals",
    "goals_prevented": "goals_prevented",
}


def _stats_list_to_dict(stats: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for entry in stats or ():
        key = entry.get("type")
        col = _STAT_KEY_MAP.get(key)
        if col is None:
            continue
        out[col] = _first_numeric(entry.get("value"))
    return out


def upsert_fixture_team_stats(
    session: Session,
    *,
    fixture: Fixture,
    stats_response: Iterable[Mapping[str, Any]],
) -> int:
    """Upsert both teams' stat rows from an API-Football ``/fixtures/statistics`` response."""
    rows_upserted = 0
    # Resolve team ids by team.id (the API returns team.id per block)
    for block in stats_response:
        team_info = block.get("team") or {}
        api_tid = team_info.get("id")
        if api_tid is None:
            continue
        team = upsert_team(session, api_id=api_tid, name=team_info.get("name") or f"Team {api_tid}")
        is_home = team.id == fixture.home_team_id
        opponent_id = fixture.away_team_id if is_home else fixture.home_team_id
        if team.id not in (fixture.home_team_id, fixture.away_team_id):
            logger.warning("Team %s (api=%s) not part of fixture %s — skipping stats row.",
                           team.name, api_tid, fixture.api_football_id)
            continue

        parsed = _stats_list_to_dict(block.get("statistics") or [])
        digest = _digest({"team": api_tid, "stats": parsed, "fixture": fixture.api_football_id})

        existing = session.scalar(
            select(FixtureTeamStats).where(
                FixtureTeamStats.fixture_id == fixture.id,
                FixtureTeamStats.team_id == team.id,
            )
        )
        goals_value = fixture.home_goals if is_home else fixture.away_goals
        if existing is None:
            session.add(FixtureTeamStats(
                fixture_id=fixture.id,
                team_id=team.id,
                opponent_team_id=opponent_id,
                is_home=is_home,
                goals=goals_value,
                raw_payload_digest=digest,
                stats_json=parsed,
                **{k: parsed.get(k) for k in _STAT_KEY_MAP.values() if k in parsed},
            ))
            session.flush()
            rows_upserted += 1
        elif existing.raw_payload_digest != digest:
            existing.opponent_team_id = opponent_id
            existing.is_home = is_home
            existing.goals = goals_value
            existing.raw_payload_digest = digest
            existing.stats_json = parsed
            for col in _STAT_KEY_MAP.values():
                if col in parsed:
                    setattr(existing, col, parsed[col])
            rows_upserted += 1
    return rows_upserted


# ---------------------------------------------------------------------------
# Player stats from /fixtures/players
# ---------------------------------------------------------------------------


def _player_stat_get(stats_block: Mapping[str, Any], path: str) -> Any:
    """Navigate the nested API-Football player-stats dict, e.g. ``shots.total``."""
    cur: Any = stats_block
    for part in path.split("."):
        if not isinstance(cur, Mapping):
            return None
        cur = cur.get(part)
    return cur


def upsert_fixture_player_stats(
    session: Session,
    *,
    fixture: Fixture,
    players_response: Iterable[Mapping[str, Any]],
) -> int:
    rows_upserted = 0
    for team_entry in players_response:
        team_info = team_entry.get("team") or {}
        api_tid = team_info.get("id")
        if api_tid is None:
            continue
        team = upsert_team(session, api_id=api_tid, name=team_info.get("name") or f"Team {api_tid}")
        for player_block in team_entry.get("players") or []:
            player_info = player_block.get("player") or {}
            api_pid = player_info.get("id")
            if api_pid is None:
                continue
            statistics = player_block.get("statistics") or []
            if not statistics:
                continue
            s0 = statistics[0] or {}
            player = upsert_player(
                session,
                api_id=api_pid,
                name=player_info.get("name") or f"Player {api_pid}",
                position=(s0.get("games") or {}).get("position"),
            )

            parsed = {
                "position": (s0.get("games") or {}).get("position"),
                "minutes": _coerce_int((s0.get("games") or {}).get("minutes")),
                "rating": _first_numeric((s0.get("games") or {}).get("rating")),
                "captain": (s0.get("games") or {}).get("captain"),
                "substitute": (s0.get("games") or {}).get("substitute"),
                "goals": _coerce_int(_player_stat_get(s0, "goals.total")),
                "assists": _coerce_int(_player_stat_get(s0, "goals.assists")),
                "shots_total": _coerce_int(_player_stat_get(s0, "shots.total")),
                "shots_on": _coerce_int(_player_stat_get(s0, "shots.on")),
                "passes_total": _coerce_int(_player_stat_get(s0, "passes.total")),
                "passes_accurate": _coerce_int(_player_stat_get(s0, "passes.accuracy")),
                "tackles": _coerce_int(_player_stat_get(s0, "tackles.total")),
                "interceptions": _coerce_int(_player_stat_get(s0, "tackles.interceptions")),
                "duels_won": _coerce_int(_player_stat_get(s0, "duels.won")),
                "duels_total": _coerce_int(_player_stat_get(s0, "duels.total")),
                "yellow_cards": _coerce_int(_player_stat_get(s0, "cards.yellow")),
                "red_cards": _coerce_int(_player_stat_get(s0, "cards.red")),
                "fouls_committed": _coerce_int(_player_stat_get(s0, "fouls.committed")),
                "fouls_drawn": _coerce_int(_player_stat_get(s0, "fouls.drawn")),
            }
            if parsed["passes_total"] and parsed["passes_accurate"] is not None:
                if parsed["passes_total"] > 0:
                    parsed["pass_accuracy"] = parsed["passes_accurate"] / parsed["passes_total"]
            digest = _digest({"pid": api_pid, "fid": fixture.api_football_id, "stats": parsed})

            existing = session.scalar(
                select(FixturePlayerStats).where(
                    FixturePlayerStats.fixture_id == fixture.id,
                    FixturePlayerStats.player_id == player.id,
                )
            )
            if existing is None:
                session.add(FixturePlayerStats(
                    fixture_id=fixture.id,
                    player_id=player.id,
                    team_id=team.id,
                    raw_payload_digest=digest,
                    stats_json=parsed,
                    **{k: v for k, v in parsed.items() if k not in {"stats_json", "raw_payload_digest"}},
                ))
                session.flush()
                rows_upserted += 1
            elif existing.raw_payload_digest != digest:
                existing.team_id = team.id
                existing.raw_payload_digest = digest
                existing.stats_json = parsed
                for col, value in parsed.items():
                    if col in {"stats_json", "raw_payload_digest"}:
                        continue
                    setattr(existing, col, value)
                rows_upserted += 1
    return rows_upserted


# ---------------------------------------------------------------------------
# Watermarks
# ---------------------------------------------------------------------------


def get_watermark(session: Session, scope: str) -> Optional[SyncWatermark]:
    return session.scalar(select(SyncWatermark).where(SyncWatermark.scope == scope))


def upsert_watermark(
    session: Session,
    *,
    scope: str,
    last_successful_at: Optional[datetime] = None,
    last_cursor: Optional[str] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> SyncWatermark:
    row = get_watermark(session, scope)
    if row is None:
        row = SyncWatermark(
            scope=scope,
            last_successful_at=last_successful_at,
            last_cursor=last_cursor,
            meta=meta or {},
        )
        session.add(row)
        session.flush()
        return row
    if last_successful_at is not None:
        row.last_successful_at = last_successful_at
    if last_cursor is not None:
        row.last_cursor = last_cursor
    if meta is not None:
        merged = dict(row.meta or {})
        merged.update(meta)
        row.meta = merged
    return row


# ---------------------------------------------------------------------------
# Feature snapshot upserts
# ---------------------------------------------------------------------------


def upsert_team_feature_snapshot(
    session: Session,
    *,
    team: Team,
    season: Season,
    as_of_date: date,
    fields: Mapping[str, Any],
    build_run_id: Optional[int] = None,
    extras: Optional[Mapping[str, Any]] = None,
    feature_build_id: Optional[int] = None,
    feature_version: Optional[str] = None,
    as_of_fixture_api_id: Optional[int] = None,
) -> TeamFeatureSnapshot:
    existing = session.scalar(
        select(TeamFeatureSnapshot).where(
            TeamFeatureSnapshot.team_id == team.id,
            TeamFeatureSnapshot.season_id == season.id,
            TeamFeatureSnapshot.as_of_date == as_of_date,
        )
    )
    built_at = datetime.now(timezone.utc)
    if existing is None:
        clean = {k: v for k, v in fields.items() if hasattr(TeamFeatureSnapshot, k)}
        existing = TeamFeatureSnapshot(
            team_id=team.id,
            season_id=season.id,
            as_of_date=as_of_date,
            build_run_id=build_run_id,
            feature_build_id=feature_build_id,
            feature_version=feature_version,
            as_of_fixture_api_id=as_of_fixture_api_id,
            built_at=built_at,
            extras=dict(extras) if extras is not None else None,
            **clean,
        )
        session.add(existing)
        session.flush()
        return existing
    for key, value in fields.items():
        if hasattr(TeamFeatureSnapshot, key):
            setattr(existing, key, value)
    if build_run_id is not None:
        existing.build_run_id = build_run_id
    if feature_build_id is not None:
        existing.feature_build_id = feature_build_id
    if feature_version is not None:
        existing.feature_version = feature_version
    if as_of_fixture_api_id is not None:
        existing.as_of_fixture_api_id = as_of_fixture_api_id
    existing.built_at = built_at
    if extras is not None:
        existing.extras = dict(extras)
    return existing


def upsert_player_feature_snapshot(
    session: Session,
    *,
    player: Player,
    season: Season,
    as_of_date: date,
    fields: Mapping[str, Any],
    team: Optional[Team] = None,
    build_run_id: Optional[int] = None,
    extras: Optional[Mapping[str, Any]] = None,
    feature_build_id: Optional[int] = None,
    feature_version: Optional[str] = None,
    as_of_fixture_api_id: Optional[int] = None,
) -> PlayerFeatureSnapshot:
    existing = session.scalar(
        select(PlayerFeatureSnapshot).where(
            PlayerFeatureSnapshot.player_id == player.id,
            PlayerFeatureSnapshot.season_id == season.id,
            PlayerFeatureSnapshot.as_of_date == as_of_date,
        )
    )
    built_at = datetime.now(timezone.utc)
    if existing is None:
        clean = {k: v for k, v in fields.items() if hasattr(PlayerFeatureSnapshot, k)}
        existing = PlayerFeatureSnapshot(
            player_id=player.id,
            team_id=team.id if team is not None else None,
            season_id=season.id,
            as_of_date=as_of_date,
            build_run_id=build_run_id,
            feature_build_id=feature_build_id,
            feature_version=feature_version,
            as_of_fixture_api_id=as_of_fixture_api_id,
            built_at=built_at,
            extras=dict(extras) if extras is not None else None,
            **clean,
        )
        session.add(existing)
        session.flush()
        return existing
    for key, value in fields.items():
        if hasattr(PlayerFeatureSnapshot, key):
            setattr(existing, key, value)
    if team is not None:
        existing.team_id = team.id
    if build_run_id is not None:
        existing.build_run_id = build_run_id
    if feature_build_id is not None:
        existing.feature_build_id = feature_build_id
    if feature_version is not None:
        existing.feature_version = feature_version
    if as_of_fixture_api_id is not None:
        existing.as_of_fixture_api_id = as_of_fixture_api_id
    existing.built_at = built_at
    if extras is not None:
        existing.extras = dict(extras)
    return existing
