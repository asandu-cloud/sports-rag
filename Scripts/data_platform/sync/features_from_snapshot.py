"""Import feature snapshots from existing ``Output/*_feature_engineering`` JSON.

The existing pipeline writes season-to-date team/player features into
per-league JSON files. To avoid re-running feature engineering, V1 ships
an importer that reads those files and upserts snapshots into the
canonical DB using the same variable vocabulary. Callers can re-run it
any time; the upsert is keyed on ``(team_id|player_id, season_id, as_of_date)``.
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

from sqlalchemy.orm import Session

from ..config import SETTINGS
from ..models import Player, SyncRun, Team
from .apifootball import COMPETITIONS, CompetitionSpec, iter_competitions
from .team_snapshot_derivation import derive_team_snapshot_rows
from .upserts import (
    upsert_competition,
    upsert_player,
    upsert_player_feature_snapshot,
    upsert_season,
    upsert_team,
    upsert_team_feature_snapshot,
)

logger = logging.getLogger(__name__)


OUTPUT_DIR = SETTINGS.project_root / "Output"

LEAGUE_DIR_MAP: Dict[str, str] = {
    "EPL": "Prem_feature_engineering",
    "LaLiga": "LaLiga_feature_engineering",
    "SerieA": "SeriaA_feature_engineering",
    "Bundesliga": "Bundesliga_feature_engineering",
    "Ligue1": "Ligue1_feature_engineering",
    "UCL": "UCL_feature_engineering",
    "UEL": "UEL_feature_engineering",
    "UECL": "UECL_feature_engineering",
}

# Only columns that already exist on the snapshot tables are persisted;
# the importer ignores anything else silently so new fields added to the
# feature pipeline don't break the import.
_TEAM_SNAPSHOT_COLS = {
    "matches_played", "goals_for_pm", "goals_against_pm", "assists_pm",
    "corners_pm", "corners_against_pm", "sot_for_pm", "sot_against_pm",
    "shots_for_pm", "cards_pm", "yellows_pm", "reds_pm",
    "cards_per_90_team", "fouls_per_90_team", "cards_per_foul_team",
    "aggression_index_norm", "form_index_team", "control_index",
    "dominance_index", "possession", "expected_goals", "pass_accuracy_pct",
    "opp_cards_induced_pm", "goals_home_pm", "goals_away_pm",
    "corners_home_pm", "corners_away_pm", "cards_home_pm", "cards_away_pm",
    "corners_against_home_pm", "corners_against_away_pm",
    "sot_against_home_pm", "sot_against_away_pm",
    "cards_induced_home_pm", "cards_induced_away_pm",
    "xg_home_pm", "xg_away_pm", "sot_home_pm", "sot_away_pm",
    "fouls_home_pm", "fouls_away_pm",
    "corners_var", "goals_var", "cards_var", "sot_var",
    "archetype",
}
_PLAYER_SNAPSHOT_COLS = {
    "appearances_total", "appearances_as_starter", "start_rate",
    "avg_minutes_per_appearance", "minutes_risk", "recent_role_trend",
    "goals_per_90", "assists_per_90", "sot_per_90", "cards_per_90",
    "starter_goals_per_90", "starter_assists_per_90",
    "starter_sot_per_90", "starter_cards_per_90",
    "form_index", "aggression_index_norm", "control_index",
    "expected_card_risk", "expected_foul_pressure",
    "total_minutes",
}


def _coerce_field(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (int, float, bool, str)):
        return value
    if isinstance(value, (list, dict)):
        return value
    return value


def _split_fields(row: Mapping[str, Any], allowed: Iterable[str]) -> Dict[str, Any]:
    allowed_set = set(allowed)
    out = {}
    for key, value in row.items():
        if key in allowed_set:
            out[key] = _coerce_field(value)
    return out


def _jsonl_or_json(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    if path.suffix == ".jsonl":
        rows = []
        with path.open("r") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return rows
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return []


def _season_to_year(season_label: str) -> Optional[int]:
    """Accept '2025', '2025/26', '2024-25' — return the API-Football season year."""
    if not season_label:
        return None
    text = str(season_label).strip()
    digits = ""
    for ch in text:
        if ch.isdigit():
            digits += ch
            if len(digits) == 4:
                break
    if len(digits) != 4:
        return None
    try:
        return int(digits)
    except ValueError:
        return None


def _effective_as_of(season_year: int, today: date) -> date:
    """Cap the ``as_of_date`` at season end so historical snapshots stay deterministic."""
    season_end = date(season_year + 1, 6, 30)
    return min(today, season_end)


def build_feature_snapshots_from_files(
    session: Session,
    *,
    codes: Optional[Iterable[str]] = None,
    seasons: Optional[Iterable[int]] = None,
    output_dir: Optional[Path] = None,
    as_of_date: Optional[date] = None,
    auto_fetch_teams: bool = True,
) -> Dict[str, Any]:
    """Read existing ``Output/*_feature_engineering`` files into feature snapshots.

    Parameters
    ----------
    codes:
        Competition codes to import. ``None`` = all known competitions.
    seasons:
        Season years to consider. ``None`` = every season year found on disk.
    output_dir:
        Root ``Output/`` directory; defaults to the repo's ``Output``.
    as_of_date:
        ``as_of_date`` to stamp on every row. Defaults to today, clamped
        at the end of each season so re-running after a season ends stays
        idempotent.
    auto_fetch_teams:
        When ``True`` (default), make one cheap API-Football
        ``/teams?league=X&season=Y`` call per (competition, season) so
        team rows exist before we try to attach team feature snapshots.
        Disable when running offline; team snapshots will be skipped for
        teams not already in the canonical store.
    """
    root = Path(output_dir) if output_dir else OUTPUT_DIR
    today = as_of_date or date.today()
    api_client = None
    if auto_fetch_teams:
        try:
            from .apifootball import ApiFootballClient
            api_client = ApiFootballClient()
        except Exception as exc:
            logger.warning("auto_fetch_teams=True but ApiFootballClient unavailable (%s); "
                           "team snapshots will be skipped for unknown teams.", exc)
            api_client = None

    run = SyncRun(
        run_kind="feature_build",
        scope=f"{','.join(codes) if codes else 'ALL'}:{','.join(map(str, seasons)) if seasons else 'ALL'}",
        started_at=datetime.now(timezone.utc),
        status="running",
        stats={},
    )
    session.add(run)
    session.flush()

    team_count = 0
    team_engineered_count = 0
    player_count = 0
    files_read = 0

    try:
        for spec in iter_competitions(codes):
            folder = LEAGUE_DIR_MAP.get(spec.code)
            if not folder:
                continue
            base = root / folder
            if not base.exists():
                logger.info("No feature dir for %s (%s) — skipping", spec.code, base)
                continue

            competition = upsert_competition(session, spec)

            # Profile files are already season-rolled up; engineered files
            # carry per-fixture rows but also include the team's averages.
            profile_files = sorted(base.glob("team_profiles_*.jsonl")) + sorted(base.glob("team_profiles_*.json"))
            player_profile_files = sorted(base.glob("player_profiles_*.jsonl")) + sorted(base.glob("player_profiles_*.json"))
            team_engineered_files = (
                sorted(base.glob("team_engineered_features_*.jsonl"))
                + sorted(base.glob("team_engineered_features_*.json"))
            )
            player_engineered_files = sorted(base.glob("player_engineered_features_*.json"))

            # Track which (league, season) combos we've prefetched teams for
            teams_prefetched: set = set()
            # A deliberately generated profile remains the preferred source.
            # The engineered-file path below is a fallback for current seasons
            # where that profile file was never produced.
            profile_team_keys_by_season: Dict[int, set] = {}

            def prefetch_teams_if_needed(season_year: int) -> None:
                if api_client is None or (spec.code, season_year) in teams_prefetched:
                    return
                try:
                    _prefetch_teams(session, api_client, spec=spec, season_year=season_year)
                except Exception as exc:
                    logger.warning("teams prefetch failed for %s %s (%s) — "
                                   "team snapshots may be skipped",
                                   spec.code, season_year, exc)
                teams_prefetched.add((spec.code, season_year))

            # Try the profile files first (they carry season-level aggregates).
            for path in profile_files:
                season_year = _extract_year_from_filename(path.name)
                if season_year is None or (seasons and season_year not in seasons):
                    continue
                season = upsert_season(session, competition=competition, year=season_year)
                files_read += 1
                as_of = _effective_as_of(season_year, today)
                rows = _jsonl_or_json(path)
                for row in rows:
                    team_name = row.get("team") or row.get("name")
                    if team_name:
                        profile_team_keys_by_season.setdefault(season_year, set()).add(
                            str(team_name).strip().casefold()
                        )

                # Lazy team bootstrap: one /teams?league=X&season=Y call per slice
                prefetch_teams_if_needed(season_year)

                for row in rows:
                    team = _upsert_team_from_row(session, row)
                    if team is None:
                        continue
                    fields = _split_fields(row, _TEAM_SNAPSHOT_COLS)
                    if not fields:
                        continue
                    upsert_team_feature_snapshot(
                        session,
                        team=team, season=season, as_of_date=as_of,
                        fields=fields, build_run_id=run.id,
                        extras={"source_file": path.name},
                    )
                    team_count += 1

            # Current legacy season runs produce fixture-level engineered rows
            # but do not always generate team_profiles_YYYY.  Derive a
            # deterministic season-to-date snapshot from those rows instead of
            # leaving the canonical database with no team features.  Keep this
            # after explicit profiles so a deliberately generated profile stays
            # authoritative when both sources are available.
            for path in team_engineered_files:
                season_year = _extract_year_from_filename(path.name)
                if season_year is None or (seasons and season_year not in seasons):
                    continue
                season = upsert_season(session, competition=competition, year=season_year)
                files_read += 1
                as_of = _effective_as_of(season_year, today)
                prefetch_teams_if_needed(season_year)
                rows = _jsonl_or_json(path)
                profile_team_keys = profile_team_keys_by_season.get(season_year, set())
                for row in derive_team_snapshot_rows(rows):
                    # Do not overwrite an explicit profile.  Filling only a
                    # missing team also makes a partially generated profile
                    # file recoverable instead of silently dropping teams.
                    if str(row.get("team") or "").strip().casefold() in profile_team_keys:
                        continue
                    team = _upsert_team_from_row(session, row)
                    if team is None:
                        continue
                    fields = _split_fields(row, _TEAM_SNAPSHOT_COLS)
                    if not fields:
                        continue
                    upsert_team_feature_snapshot(
                        session,
                        team=team, season=season, as_of_date=as_of,
                        fields=fields, build_run_id=run.id,
                        extras={
                            "source_file": path.name,
                            "source_kind": "derived_team_engineered_features",
                            "aggregation": "season_to_date_fixture_rows_v1",
                        },
                    )
                    team_count += 1
                    team_engineered_count += 1

            for path in player_profile_files:
                season_year = _extract_year_from_filename(path.name)
                if season_year is None or (seasons and season_year not in seasons):
                    continue
                season = upsert_season(session, competition=competition, year=season_year)
                files_read += 1
                as_of = _effective_as_of(season_year, today)
                for row in _jsonl_or_json(path):
                    player, team = _upsert_player_and_team_from_row(session, row)
                    if player is None:
                        continue
                    fields = _split_fields(row, _PLAYER_SNAPSHOT_COLS)
                    if not fields and row.get("cards_per_90_calc") is None:
                        continue
                    # Older files use cards_per_90_calc / fouls_per_90_calc
                    # rather than cards_per_90; map them so downstream code
                    # sees consistent names.
                    if "cards_per_90_calc" in row and "cards_per_90" not in fields:
                        fields["cards_per_90"] = _coerce_field(row["cards_per_90_calc"])
                    upsert_player_feature_snapshot(
                        session,
                        player=player, team=team, season=season, as_of_date=as_of,
                        fields=fields, build_run_id=run.id,
                        extras={"source_file": path.name},
                    )
                    player_count += 1

            # Engineered files are only used as a fallback for team snapshots;
            # player snapshots continue to come from their dedicated profiles.
            logger.debug(
                "%s: %s team-profile files, %s player-profile files, %s team-eng, %s player-eng",
                spec.code, len(profile_files), len(player_profile_files),
                len(team_engineered_files), len(player_engineered_files),
            )

        run.finished_at = datetime.now(timezone.utc)
        run.status = "completed"
        run.stats = {
            "team_snapshots": team_count,
            "team_snapshots_from_engineered": team_engineered_count,
            "player_snapshots": player_count,
            "files_read": files_read,
        }
        return {
            "run_id": run.id,
            "team_snapshots": team_count,
            "team_snapshots_from_engineered": team_engineered_count,
            "player_snapshots": player_count,
            "files_read": files_read,
        }
    except Exception as exc:
        run.finished_at = datetime.now(timezone.utc)
        run.status = "failed"
        run.error_text = str(exc)
        raise


def _extract_year_from_filename(name: str) -> Optional[int]:
    digits = ""
    for ch in name:
        if ch.isdigit():
            digits += ch
            if len(digits) == 4:
                try:
                    year = int(digits)
                except ValueError:
                    return None
                if 2000 <= year <= 2100:
                    return year
                digits = digits[1:]
        else:
            digits = ""
    return None


def _prefetch_teams(
    session: Session,
    client,
    *,
    spec,
    season_year: int,
) -> int:
    """Populate the canonical teams table for one (league, season) slice.

    Reads ``/teams?league=X&season=Y`` from API-Football. Cheap — one call,
    one upsert per team. Run lazily so callers without an API key still
    work (they just lose team feature snapshots for unknown teams).
    """
    rows = client.teams_for_league(league=spec.api_football_id, season=season_year)
    if not rows:
        return 0
    inserted = 0
    for row in rows:
        team_block = row.get("team") or {}
        api_id = team_block.get("id")
        name = team_block.get("name")
        if not api_id or not name:
            continue
        upsert_team(
            session,
            api_id=int(api_id),
            name=str(name),
            country=team_block.get("country"),
            short_code=team_block.get("code"),
            founded=team_block.get("founded"),
            logo_url=team_block.get("logo"),
        )
        inserted += 1
    session.flush()
    logger.info("Prefetched %s teams for %s %s", inserted, spec.code, season_year)
    return inserted


def _upsert_team_from_row(session: Session, row: Mapping[str, Any]) -> Optional[Team]:
    api_id = row.get("team_id") or row.get("api_football_id")
    name = row.get("team") or row.get("name")
    if not name:
        return None
    if api_id is None:
        # Fall back: look up an existing team by name
        existing = session.query(Team).filter(Team.name == name).first()
        if existing is None:
            logger.warning("Team missing api id: %s — skipping", name)
            return None
        return existing
    try:
        api_id_int = int(api_id)
    except (TypeError, ValueError):
        return None
    return upsert_team(session, api_id=api_id_int, name=str(name))


def _upsert_player_and_team_from_row(session: Session, row: Mapping[str, Any]):
    player_api = row.get("player_id") or row.get("api_football_id")
    player_name = row.get("name") or row.get("player_name")
    if not player_name:
        return None, None
    if player_api is None:
        return None, None
    try:
        player_api_int = int(player_api)
    except (TypeError, ValueError):
        return None, None
    player = upsert_player(
        session, api_id=player_api_int, name=str(player_name),
        position=row.get("position"),
    )
    team = None
    team_name = row.get("team")
    if team_name:
        team_api = row.get("team_id")
        if team_api is not None:
            try:
                team_api_int = int(team_api)
                team = upsert_team(session, api_id=team_api_int, name=str(team_name))
            except (TypeError, ValueError):
                team = None
        else:
            team = session.query(Team).filter(Team.name == str(team_name)).first()
    return player, team
