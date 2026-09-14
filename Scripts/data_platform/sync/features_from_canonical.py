"""Build live-compatible feature snapshots directly from canonical fixture data.

The established domestic leagues still have a legacy ``Output/`` feature path.
New competitions must not need copied collector/feature-engineering scripts just
to become prediction-ready.  This builder turns the canonical API-Football
fixture, team-stat, and player-stat records into the same snapshot vocabulary
used by the existing Chroma/Match Read prediction path.

It intentionally changes no projection or selection weights.  It is an input
adapter: after a canonical bootstrap, callers build snapshots, enqueue KB
documents, and retain the existing prediction logic unchanged.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from sqlalchemy import select
from sqlalchemy.orm import Session, aliased

from ..models import (
    Competition,
    Fixture,
    FixturePlayerStats,
    FixtureTeamStats,
    Player,
    Season,
    SyncRun,
    Team,
)
from .apifootball import iter_competitions
from .team_snapshot_derivation import derive_team_snapshot_rows
from .upserts import upsert_player_feature_snapshot, upsert_team_feature_snapshot


FINISHED_STATUSES = frozenset({"FT", "AET", "PEN"})
PIPELINE_NAME = "canonical_fixture_stats_v1"


@dataclass
class CanonicalFeatureStats:
    competitions: int = 0
    seasons: int = 0
    fixture_rows: int = 0
    team_snapshots: int = 0
    player_snapshots: int = 0

    def as_dict(self) -> Dict[str, int]:
        return {
            "competitions": self.competitions,
            "seasons": self.seasons,
            "fixture_rows": self.fixture_rows,
            "team_snapshots": self.team_snapshots,
            "player_snapshots": self.player_snapshots,
        }


def _float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _date_before_or_on(value: Optional[datetime], as_of: date) -> bool:
    """Treat canonical timestamp values conservatively across SQLite/Postgres."""
    if value is None:
        return False
    try:
        return value.date() <= as_of
    except AttributeError:
        return False


def _season_as_of(season: Season, requested: date) -> date:
    """Keep completed campaigns immutable while current campaigns use today."""
    if season.end_date is not None and season.end_date < requested:
        return season.end_date
    # API-Football start-year seasons normally end next June.  Historical
    # bootstraps do not always populate season end dates, so retain a stable
    # equivalent anchor in that case.
    if season.year < requested.year - 1:
        return date(season.year + 1, 6, 30)
    return requested


def _team_rows(
    session: Session,
    *,
    competition: Competition,
    season: Season,
    as_of: date,
) -> List[Dict[str, Any]]:
    opponent = aliased(Team)
    rows = session.execute(
        select(FixtureTeamStats, Fixture, Team, opponent)
        .join(Fixture, Fixture.id == FixtureTeamStats.fixture_id)
        .join(Team, Team.id == FixtureTeamStats.team_id)
        .join(opponent, opponent.id == FixtureTeamStats.opponent_team_id)
        .where(
            Fixture.competition_id == competition.id,
            Fixture.season_id == season.id,
            Fixture.status.in_(FINISHED_STATUSES),
        )
        .order_by(Fixture.kickoff_utc.asc(), Fixture.api_football_id.asc())
    ).all()

    normalized: List[Dict[str, Any]] = []
    for stats, fixture, team, opp in rows:
        if not _date_before_or_on(fixture.kickoff_utc, as_of):
            continue
        home = bool(stats.is_home)
        fixture_name = f"{team.name} vs {opp.name}" if home else f"{opp.name} vs {team.name}"
        yellow = _float(stats.yellow_cards) or 0.0
        red = _float(stats.red_cards) or 0.0
        passes = _float(stats.passes_total) or 0.0
        accurate = _float(stats.passes_accurate) or 0.0
        possession = _float(stats.possession)
        if possession is not None and possession > 1.0:
            possession /= 100.0
        # These are the existing feature vocabulary, derived from the same
        # raw API-Football fields as the legacy per-league scripts.  They are
        # deliberately descriptive rather than new model logic.
        pass_accuracy = (accurate / passes) if passes > 0 else _float(stats.pass_accuracy)
        cards_total = yellow + red
        normalized.append({
            "team": team.name,
            "team_id": team.id,
            "opponent": opp.name,
            "fixture": fixture_name,
            "fixture_id": fixture.api_football_id,
            "fixture_date": fixture.kickoff_utc.date().isoformat(),
            "home_away": "home" if home else "away",
            "goals": _float(stats.goals),
            "assists": 0.0,
            "shots_total": _float(stats.shots_total),
            "shots_on": _float(stats.shots_on),
            "corners": _float(stats.corners),
            "fouls_committed": _float(stats.fouls_committed),
            "yellow_cards": yellow,
            "red_cards": red,
            "cards_total": cards_total,
            "possession": possession,
            "expected_goals": _float(stats.expected_goals),
            "passes_total": passes,
            "accurate_passes": accurate,
            "pass_accuracy_team": pass_accuracy,
            "cards_per_90_team": cards_total,
            "fouls_per_90_team": _float(stats.fouls_committed),
            "cards_per_foul_team": (cards_total / (_float(stats.fouls_committed) or 1.0)),
        })
    return normalized


def _player_snapshot_fields(rows: Sequence[tuple[FixturePlayerStats, Fixture]]) -> Dict[str, Any]:
    played = [stats for stats, _fixture in rows if int(stats.minutes or 0) > 0]
    minutes = sum(int(stats.minutes or 0) for stats in played)
    appearances = len(played)
    starters = [stats for stats in played if int(stats.minutes or 0) >= 60]
    starter_minutes = sum(int(stats.minutes or 0) for stats in starters)

    def total(name: str, source: Sequence[FixturePlayerStats] = played) -> float:
        return sum(float(getattr(item, name) or 0) for item in source)

    def per90(name: str, source: Sequence[FixturePlayerStats], denominator: int) -> float:
        return (total(name, source) * 90.0 / denominator) if denominator else 0.0

    ratings = [float(stats.rating) for stats, _ in rows if stats.rating is not None]
    cards90 = per90("yellow_cards", played, minutes) + per90("red_cards", played, minutes)
    fouls90 = per90("fouls_committed", played, minutes)
    if appearances == 0:
        risk = "bench"
    elif len(starters) / appearances >= 0.80 and minutes / appearances >= 75:
        risk = "nailed_on"
    elif len(starters) / appearances >= 0.60 and minutes / appearances >= 60:
        risk = "likely_starter"
    elif len(starters) / appearances >= 0.30:
        risk = "rotation"
    else:
        risk = "bench"
    return {
        "appearances_total": appearances,
        "appearances_as_starter": len(starters),
        "start_rate": (len(starters) / appearances) if appearances else 0.0,
        "avg_minutes_per_appearance": (minutes / appearances) if appearances else 0.0,
        "minutes_risk": risk,
        "recent_role_trend": "starting" if len(starters) >= min(4, appearances) else "mixed",
        "goals_per_90": per90("goals", played, minutes),
        "assists_per_90": per90("assists", played, minutes),
        "sot_per_90": per90("shots_on", played, minutes),
        "cards_per_90": cards90,
        "starter_goals_per_90": per90("goals", starters, starter_minutes),
        "starter_assists_per_90": per90("assists", starters, starter_minutes),
        "starter_sot_per_90": per90("shots_on", starters, starter_minutes),
        "starter_cards_per_90": (
            per90("yellow_cards", starters, starter_minutes)
            + per90("red_cards", starters, starter_minutes)
        ),
        "form_index": (sum(ratings[-4:]) / min(4, len(ratings))) if ratings else 0.0,
        "aggression_index_norm": min(1.0, cards90 / 0.6),
        "control_index": 0.0,
        "expected_card_risk": cards90,
        "expected_foul_pressure": fouls90,
        "total_minutes": minutes,
    }


def _player_rows(
    session: Session,
    *,
    competition: Competition,
    season: Season,
    as_of: date,
) -> Dict[int, List[tuple[FixturePlayerStats, Fixture]]]:
    rows = session.execute(
        select(FixturePlayerStats, Fixture)
        .join(Fixture, Fixture.id == FixturePlayerStats.fixture_id)
        .where(
            Fixture.competition_id == competition.id,
            Fixture.season_id == season.id,
            Fixture.status.in_(FINISHED_STATUSES),
        )
        .order_by(Fixture.kickoff_utc.asc(), Fixture.api_football_id.asc())
    ).all()
    grouped: Dict[int, List[tuple[FixturePlayerStats, Fixture]]] = defaultdict(list)
    for stats, fixture in rows:
        if _date_before_or_on(fixture.kickoff_utc, as_of):
            grouped[stats.player_id].append((stats, fixture))
    return grouped


def build_feature_snapshots_from_canonical(
    session: Session,
    *,
    codes: Optional[Iterable[str]] = None,
    seasons: Optional[Iterable[int]] = None,
    as_of_date: Optional[date] = None,
) -> Dict[str, Any]:
    """Build additive team/player feature snapshots from canonical tables.

    Only finished fixtures on or before the appropriate as-of date contribute.
    Re-running the command updates the same snapshot key and is therefore safe
    after each matchweek.
    """
    requested_as_of = as_of_date or date.today()
    requested_seasons = {int(value) for value in seasons} if seasons else None
    wanted = set(codes) if codes else None
    stats = CanonicalFeatureStats()
    run = SyncRun(
        run_kind="canonical_feature_build",
        scope=f"{','.join(sorted(wanted)) if wanted else 'ALL'}:{','.join(map(str, sorted(requested_seasons))) if requested_seasons else 'ALL'}",
        started_at=datetime.now(timezone.utc),
        status="running",
        stats={},
    )
    session.add(run)
    session.flush()
    try:
        for spec in iter_competitions(codes):
            if wanted is not None and spec.code not in wanted:
                continue
            competition = session.scalar(select(Competition).where(Competition.code == spec.code))
            if competition is None:
                continue
            season_rows = session.scalars(
                select(Season)
                .where(Season.competition_id == competition.id)
                .order_by(Season.year.asc())
            ).all()
            seen_competition = False
            for season in season_rows:
                if requested_seasons is not None and season.year not in requested_seasons:
                    continue
                effective_as_of = _season_as_of(season, requested_as_of)
                team_rows = _team_rows(
                    session, competition=competition, season=season, as_of=effective_as_of,
                )
                if not team_rows:
                    continue
                seen_competition = True
                stats.seasons += 1
                stats.fixture_rows += len(team_rows)
                derived = derive_team_snapshot_rows(team_rows)
                latest_fixture_id = max(int(row["fixture_id"]) for row in team_rows)
                teams = {team.id: team for team in session.scalars(select(Team).where(Team.id.in_({int(row["team_id"]) for row in derived}))).all()}
                for row in derived:
                    team = teams.get(int(row["team_id"]))
                    if team is None:
                        continue
                    fields = {key: value for key, value in row.items() if key not in {"team", "team_id"}}
                    upsert_team_feature_snapshot(
                        session,
                        team=team,
                        season=season,
                        as_of_date=effective_as_of,
                        fields=fields,
                        build_run_id=run.id,
                        feature_version=PIPELINE_NAME,
                        as_of_fixture_api_id=latest_fixture_id,
                        extras={"source": "canonical_fixture_team_stats", "pipeline": PIPELINE_NAME},
                    )
                    stats.team_snapshots += 1

                players = _player_rows(session, competition=competition, season=season, as_of=effective_as_of)
                for player_id, player_rows in players.items():
                    player = session.get(Player, player_id)
                    if player is None:
                        continue
                    latest_stats = player_rows[-1][0]
                    team = session.get(Team, latest_stats.team_id)
                    upsert_player_feature_snapshot(
                        session,
                        player=player,
                        team=team,
                        season=season,
                        as_of_date=effective_as_of,
                        fields=_player_snapshot_fields(player_rows),
                        build_run_id=run.id,
                        feature_version=PIPELINE_NAME,
                        as_of_fixture_api_id=latest_fixture_id,
                        extras={"source": "canonical_fixture_player_stats", "pipeline": PIPELINE_NAME},
                    )
                    stats.player_snapshots += 1
            if seen_competition:
                stats.competitions += 1
        run.status = "completed"
        run.finished_at = datetime.now(timezone.utc)
        run.stats = stats.as_dict()
        return {"run_id": run.id, **stats.as_dict()}
    except Exception as exc:
        run.status = "failed"
        run.finished_at = datetime.now(timezone.utc)
        run.stats = stats.as_dict()
        run.error_text = str(exc)
        raise
