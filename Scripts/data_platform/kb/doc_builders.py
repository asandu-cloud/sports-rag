"""Build Chroma docs from canonical DB rows.

Doc text is intentionally compatible with the output of
``Scripts/rag_ingest/00_normalize_to_docs.py`` so existing retrieval /
ranking code keeps working. Doc ids use the same MD5 hash the normalizer
computes, so the platform can adopt legacy docs without re-embedding
unchanged entities.
"""

from __future__ import annotations

import hashlib
import logging
from datetime import date
from typing import Any, Dict, Iterable, List, Optional

from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from ..db import session_scope
from ..models import (
    Competition,
    Fixture,
    FixturePlayerStats,
    FixtureTeamStats,
    Player,
    PlayerFeatureSnapshot,
    Season,
    Team,
    TeamFeatureSnapshot,
)

logger = logging.getLogger(__name__)


def make_doc_id(parts: Iterable) -> str:
    base = "|".join("" if p is None else str(p) for p in parts)
    return hashlib.md5(base.encode()).hexdigest()


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.3f}".rstrip("0").rstrip(".")
    return str(value)


def build_docs_for_entity(
    entity_type: str,
    entity_key: str,
    *,
    session: Optional[Session] = None,
) -> List[Dict[str, Any]]:
    """Entry point — returns a list of docs for a given entity.

    ``entity_type`` values:
      * ``team`` — key = ``<league_code>:<team_api_football_id>``
      * ``player`` — key = ``<league_code>:<player_api_football_id>``
      * ``fixture`` — key = ``<fixture_api_football_id>``

    Each returned doc has the shape expected by ``KBService.refresh``:
    ``{doc_id, entity_type, entity_id, league, season, doc_type, text, metadata}``.
    """
    builders = {
        "team": _build_team_docs,
        "player": _build_player_docs,
        "fixture": _build_fixture_docs,
    }
    builder = builders.get(entity_type)
    if builder is None:
        return []
    if session is not None:
        return builder(session, entity_key)
    with session_scope() as s:
        return builder(s, entity_key)


# ---------------------------------------------------------------------------
# Team docs
# ---------------------------------------------------------------------------


def _build_team_docs(session: Session, entity_key: str) -> List[Dict[str, Any]]:
    try:
        league_code, team_api_id = entity_key.split(":", 1)
        team_api_id_int = int(team_api_id)
    except ValueError:
        logger.warning("Invalid team entity_key: %s", entity_key)
        return []
    team = session.scalar(select(Team).where(Team.api_football_id == team_api_id_int))
    if team is None:
        return []
    competition = session.scalar(select(Competition).where(Competition.code == league_code))
    if competition is None:
        return []

    snapshot = session.scalar(
        select(TeamFeatureSnapshot)
        .join(Season, Season.id == TeamFeatureSnapshot.season_id)
        .where(TeamFeatureSnapshot.team_id == team.id, Season.competition_id == competition.id)
        .order_by(desc(TeamFeatureSnapshot.as_of_date))
        .limit(1)
    )
    docs: List[Dict[str, Any]] = []
    if snapshot is not None:
        season = session.get(Season, snapshot.season_id)
        docs.append(_team_profile_doc(team, season, competition, snapshot))

    # Fixture docs for this team's most recent matches
    recent = session.execute(
        select(FixtureTeamStats, Fixture, Season)
        .join(Fixture, Fixture.id == FixtureTeamStats.fixture_id)
        .join(Season, Season.id == Fixture.season_id)
        .where(
            FixtureTeamStats.team_id == team.id,
            Fixture.competition_id == competition.id,
        )
        .order_by(desc(Fixture.kickoff_utc))
        .limit(10)
    ).all()
    for stats, fixture, season in recent:
        opponent = session.get(Team, stats.opponent_team_id)
        docs.append(_team_fixture_doc(team, opponent, fixture, season, competition, stats))
    return docs


def _team_profile_doc(
    team: Team,
    season: Optional[Season],
    competition: Competition,
    snapshot: TeamFeatureSnapshot,
) -> Dict[str, Any]:
    season_label = season.label if season else ""
    league = competition.code
    text = "\n".join([
        f"{league} {season_label} | Team Profile",
        f"{team.name} | MP:{snapshot.matches_played or 0}",
        f"G/Match:{_fmt(snapshot.goals_for_pm)} G Against/Match:{_fmt(snapshot.goals_against_pm)} Poss:{_fmt(snapshot.possession)}",
        f"Shots For/Match:{_fmt(snapshot.shots_for_pm)} | SoT For/Match:{_fmt(snapshot.sot_for_pm)} | SoT Against/Match:{_fmt(snapshot.sot_against_pm)}",
        f"Corners For/Match:{_fmt(snapshot.corners_pm)} | Corners Against/Match:{_fmt(snapshot.corners_against_pm)}",
        f"Cards/Match:{_fmt(snapshot.cards_pm)} Yellows/Match:{_fmt(snapshot.yellows_pm)} Reds/Match:{_fmt(snapshot.reds_pm)}",
        f"Cards/90:{_fmt(snapshot.cards_per_90_team)} Fouls/90:{_fmt(snapshot.fouls_per_90_team)} Cards/Foul:{_fmt(snapshot.cards_per_foul_team)}",
        f"Dominance:{_fmt(snapshot.dominance_index)} Control:{_fmt(snapshot.control_index)} Archetype:{snapshot.archetype or ''}",
        f"Opp Cards Induced/Match:{_fmt(snapshot.opp_cards_induced_pm)}",
    ])
    metadata = {
        "entity_type": "team",
        "doc_type": "team_profile",
        "league": league,
        "season": season_label,
        "team": team.name,
        "team_id": team.api_football_id,
        "matches_played": snapshot.matches_played,
        "goals_for_pm": _to_float(snapshot.goals_for_pm),
        "goals_against_pm": _to_float(snapshot.goals_against_pm),
        "corners_pm": _to_float(snapshot.corners_pm),
        "corners_against_pm": _to_float(snapshot.corners_against_pm),
        "sot_for_pm": _to_float(snapshot.sot_for_pm),
        "sot_against_pm": _to_float(snapshot.sot_against_pm),
        "cards_per_90_team": _to_float(snapshot.cards_per_90_team),
        "fouls_per_90_team": _to_float(snapshot.fouls_per_90_team),
        "aggression_index_norm": _to_float(snapshot.aggression_index_norm),
        "form_index_team": _to_float(snapshot.form_index_team),
        "control_index": _to_float(snapshot.control_index),
        "dominance_index": _to_float(snapshot.dominance_index),
        "archetype": snapshot.archetype,
        "as_of_date": snapshot.as_of_date.isoformat() if snapshot.as_of_date else None,
    }
    doc_id = make_doc_id([league, season_label, "team_profile", team.name])
    return {
        "doc_id": doc_id,
        "entity_type": "team",
        "entity_id": str(team.api_football_id),
        "league": league,
        "season": season_label,
        "doc_type": "team_profile",
        "text": text,
        "metadata": metadata,
    }


def _team_fixture_doc(
    team: Team,
    opponent: Optional[Team],
    fixture: Fixture,
    season: Optional[Season],
    competition: Competition,
    stats: FixtureTeamStats,
) -> Dict[str, Any]:
    season_label = season.label if season else ""
    league = competition.code
    is_home = bool(stats.is_home)
    opp_name = opponent.name if opponent else "?"
    fixture_str = f"{team.name} vs {opp_name}" if is_home else f"{opp_name} vs {team.name}"
    text = "\n".join([
        f"{league} {season_label} | Team Fixture",
        f"Date: {fixture.kickoff_utc.date().isoformat() if fixture.kickoff_utc else ''} | Fixture: {fixture_str}",
        f"Team: {team.name} ({'home' if is_home else 'away'}) vs {opp_name}",
        f"Goals:{_fmt(stats.goals)} xG:{_fmt(stats.expected_goals)} Possession:{_fmt(stats.possession)}",
        f"Shots For:{_fmt(stats.shots_total)} (On:{_fmt(stats.shots_on)})",
        f"Corners For:{_fmt(stats.corners)}",
        f"Fouls:{_fmt(stats.fouls_committed)} YC:{_fmt(stats.yellow_cards)} RC:{_fmt(stats.red_cards)}",
    ])
    metadata = {
        "entity_type": "team",
        "doc_type": "team_fixture",
        "league": league,
        "season": season_label,
        "fixture": fixture_str,
        "fixture_date": fixture.kickoff_utc.date().isoformat() if fixture.kickoff_utc else None,
        "team": team.name,
        "team_id": team.api_football_id,
        "opponent": opp_name,
        "home_away": "home" if is_home else "away",
        "shots_for": stats.shots_total,
        "sot_for": stats.shots_on,
        "corners_for": stats.corners,
        "possession": _to_float(stats.possession),
        "yellow_cards": stats.yellow_cards,
        "red_cards": stats.red_cards,
        "fouls_committed": stats.fouls_committed,
        "fixture_api_id": fixture.api_football_id,
    }
    doc_id = make_doc_id([league, season_label, "team_fixture", team.name, fixture_str])
    return {
        "doc_id": doc_id,
        "entity_type": "team",
        "entity_id": str(team.api_football_id),
        "league": league,
        "season": season_label,
        "doc_type": "team_fixture",
        "text": text,
        "metadata": metadata,
    }


# ---------------------------------------------------------------------------
# Player docs
# ---------------------------------------------------------------------------


def _build_player_docs(session: Session, entity_key: str) -> List[Dict[str, Any]]:
    try:
        league_code, player_api_id = entity_key.split(":", 1)
        player_api_id_int = int(player_api_id)
    except ValueError:
        return []
    player = session.scalar(select(Player).where(Player.api_football_id == player_api_id_int))
    if player is None:
        return []
    competition = session.scalar(select(Competition).where(Competition.code == league_code))
    if competition is None:
        return []

    snapshot = session.scalar(
        select(PlayerFeatureSnapshot)
        .join(Season, Season.id == PlayerFeatureSnapshot.season_id)
        .where(PlayerFeatureSnapshot.player_id == player.id, Season.competition_id == competition.id)
        .order_by(desc(PlayerFeatureSnapshot.as_of_date))
        .limit(1)
    )
    docs: List[Dict[str, Any]] = []
    if snapshot is not None:
        season = session.get(Season, snapshot.season_id)
        team = session.get(Team, snapshot.team_id) if snapshot.team_id else None
        docs.append(_player_profile_doc(player, team, season, competition, snapshot))
    return docs


def _player_profile_doc(
    player: Player,
    team: Optional[Team],
    season: Optional[Season],
    competition: Competition,
    snapshot: PlayerFeatureSnapshot,
) -> Dict[str, Any]:
    season_label = season.label if season else ""
    league = competition.code
    team_name = team.name if team else "-"
    text = "\n".join([
        f"{league} {season_label} | Player Profile",
        f"{player.name} (id:{player.api_football_id}, {team_name}) – pos:{player.position} ",
        f"Apps:{snapshot.appearances_total or 0} Starts:{snapshot.appearances_as_starter or 0} StartRate:{_fmt(snapshot.start_rate)} AvgMin:{_fmt(snapshot.avg_minutes_per_appearance)}",
        f"G/90:{_fmt(snapshot.goals_per_90)} A/90:{_fmt(snapshot.assists_per_90)} SoT/90:{_fmt(snapshot.sot_per_90)} Cards/90:{_fmt(snapshot.cards_per_90)}",
        f"StarterOnly G/90:{_fmt(snapshot.starter_goals_per_90)} A/90:{_fmt(snapshot.starter_assists_per_90)} SoT/90:{_fmt(snapshot.starter_sot_per_90)} Cards/90:{_fmt(snapshot.starter_cards_per_90)}",
        f"FormIdx:{_fmt(snapshot.form_index)} AggIdx:{_fmt(snapshot.aggression_index_norm)} CtrlIdx:{_fmt(snapshot.control_index)}",
    ])
    metadata = {
        "entity_type": "player",
        "doc_type": "player_profile",
        "league": league,
        "season": season_label,
        "team": team_name,
        "player_id": player.api_football_id,
        "player_name": player.name,
        "position": player.position,
        "minutes_risk": snapshot.minutes_risk,
        "start_rate": _to_float(snapshot.start_rate),
        "appearances_total": snapshot.appearances_total,
        "appearances_as_starter": snapshot.appearances_as_starter,
        "goals_per_90": _to_float(snapshot.goals_per_90),
        "assists_per_90": _to_float(snapshot.assists_per_90),
        "sot_per_90": _to_float(snapshot.sot_per_90),
        "cards_per_90": _to_float(snapshot.cards_per_90),
        "starter_goals_per_90": _to_float(snapshot.starter_goals_per_90),
        "starter_assists_per_90": _to_float(snapshot.starter_assists_per_90),
        "starter_sot_per_90": _to_float(snapshot.starter_sot_per_90),
        "starter_cards_per_90": _to_float(snapshot.starter_cards_per_90),
        "total_minutes": snapshot.total_minutes,
    }
    doc_id = make_doc_id([league, season_label, "player_profile", player.api_football_id])
    return {
        "doc_id": doc_id,
        "entity_type": "player",
        "entity_id": str(player.api_football_id),
        "league": league,
        "season": season_label,
        "doc_type": "player_profile",
        "text": text,
        "metadata": metadata,
    }


# ---------------------------------------------------------------------------
# Fixture docs
# ---------------------------------------------------------------------------


def _build_fixture_docs(session: Session, entity_key: str) -> List[Dict[str, Any]]:
    try:
        fixture_api_id = int(entity_key)
    except ValueError:
        return []
    fixture = session.scalar(select(Fixture).where(Fixture.api_football_id == fixture_api_id))
    if fixture is None:
        return []
    competition = session.get(Competition, fixture.competition_id)
    season = session.get(Season, fixture.season_id)
    if competition is None:
        return []
    team_stats = session.scalars(
        select(FixtureTeamStats).where(FixtureTeamStats.fixture_id == fixture.id)
    ).all()
    home_team = session.get(Team, fixture.home_team_id)
    away_team = session.get(Team, fixture.away_team_id)
    docs: List[Dict[str, Any]] = []
    for stats in team_stats:
        team = home_team if stats.team_id == (home_team.id if home_team else None) else away_team
        opp = away_team if team is home_team else home_team
        if team is None:
            continue
        docs.append(_team_fixture_doc(team, opp, fixture, season, competition, stats))
    return docs


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
