"""Conservative recovery of legacy team history, separate from live ingestion.

Legacy exports dropped provider team IDs and changed null statistics to zero.
Only an independently supplied API fixture list can recover identities. Exact
names are used *inside that identified fixture*, never as global team identity.
Old exported zero statistics remain unknown; new exports can retain explicit
zeros with matching per-team provider evidence. API score zeros are retained.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import math

from sqlalchemy import select

from Scripts.team_stat_export import EVIDENCE_KEY, verified_export_values

from ..models import Competition, Fixture, FixtureTeamStats, Season, Team
from ..sync.apifootball import COMPETITIONS
from ..sync.upserts import _STAT_KEY_MAP, _digest, _parse_dt


LEGACY_FILES = {
    "EPL": "Prem_teams/team_fixture_stats_{season}.json",
    "LaLiga": "LaLiga_Output/LaLiga_team_fixture_stats_{season}.json",
    "SerieA": "SeriaA_Output/SeriaA_team_fixture_stats_{season}.json",
    "Bundesliga": "Bundesliga_Output/Bundesliga_team_fixture_stats_{season}.json",
    "Ligue1": "Ligue1_Output/Ligue1_team_fixture_stats_{season}.json",
    "UCL": "UCL_teams/team_fixture_stats_{season}.json",
    "UEL": "UEL_teams/team_fixture_stats_{season}.json",
    "UECL": "UECL_teams/team_fixture_stats_{season}.json",
}
FRACTIONS = {"possession", "pass_accuracy"}
CONTINUOUS = FRACTIONS | {"expected_goals", "goals_prevented"}


def numeric(value):
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (ValueError, TypeError):
        return None


def provider_id(value):
    n = numeric(value)
    if n is None or n <= 0 or not n.is_integer():
        raise ValueError("Missing/invalid provider ID")
    return int(n)


def checked_fixture(row, code, season, as_of):
    fixture, league, teams = row["fixture"], row["league"], row["teams"]
    if league["id"] != COMPETITIONS[code].api_football_id or league["season"] != season:
        raise ValueError("Provider competition/season mismatch")
    fid = provider_id(fixture["id"])
    home, away = provider_id(teams["home"]["id"]), provider_id(teams["away"]["id"])
    if home == away or not all(teams[s].get("name") for s in ("home", "away")):
        raise ValueError("Invalid fixture teams")
    kickoff = _parse_dt(fixture["date"])
    if kickoff is None or kickoff.tzinfo is None or kickoff >= as_of:
        raise ValueError("Invalid or future kickoff")
    if fixture["status"]["short"] != "FT":
        raise ValueError("Not a regulation-time completed fixture")
    for side in ("home", "away"):
        score = numeric(row["goals"][side])
        if score is None or score < 0 or not score.is_integer():
            raise ValueError("Invalid final score")
    return fid


def recover_statistics(rows, api_row):
    """Return two safely paired rows, or reject the entire ambiguous pairing."""
    if not rows:
        return [{}, {}], ["legacy_statistics_absent"]
    if len(rows) != 2:
        raise ValueError("Expected exactly two legacy team rows")
    names = [api_row["teams"][s]["name"] for s in ("home", "away")]
    if len(set(names)) != 2 or sorted(r.get("team", "") for r in rows) != sorted(names):
        raise ValueError("Legacy team names do not match identified fixture sides")
    paired = {r["team"]: r for r in rows}
    values, warnings = [], []
    for name in names:
        row = paired[name]
        side = "home" if name == names[0] else "away"
        evidenced = verified_export_values(row, fixture_id=api_row["fixture"]["id"],
                                          team_id=api_row["teams"][side]["id"])
        if [row.get("home_team"), row.get("away_team")] != names:
            raise ValueError("Legacy home/away mismatch")
        if _parse_dt(row.get("fixture_date_utc")) != _parse_dt(api_row["fixture"]["date"]):
            raise ValueError("Legacy kickoff mismatch")
        if any(numeric(row.get(s + "_goals")) != numeric(api_row["goals"][s]) for s in ("home", "away")):
            raise ValueError("Legacy score mismatch")
        stats = {}
        for key, column in _STAT_KEY_MAP.items():
            n = numeric(row.get(key))
            if n == 0 and evidenced.get(key) != 0:
                warnings.append("ambiguous_zero:" + column)
                n = None
            elif n is not None and ((column != "goals_prevented" and n < 0)
                                    or (column not in CONTINUOUS and not n.is_integer())
                                    or (column in FRACTIONS and n > 1)):
                warnings.append("invalid_statistic:" + column)
                n = None
            stats[column] = n if column in CONTINUOUS or n is None else int(n)
        values.append(stats)
    return values, warnings


def build_recovery_plan(metadata, legacy, *, season, sources, as_of=None):
    as_of = as_of or datetime.now(timezone.utc)
    entries, excluded, cohorts = [], [], {}
    # Duplicate IDs, even identical copies, are never silently last-write-wins.
    counts = Counter(row.get("fixture", {}).get("id") for rows in metadata.values() for row in rows)
    for code, fixtures in metadata.items():
        grouped = defaultdict(list)
        for row in legacy.get(code, []):
            try:
                grouped[provider_id(row.get("fixture_id"))].append(row)
            except ValueError:
                excluded.append({"league": code, "reason": "legacy_invalid_fixture_id"})
        warnings = Counter()
        usable, missing, partial = 0, 0, 0
        for row in fixtures:
            fid = row.get("fixture", {}).get("id")
            try:
                fid = checked_fixture(row, code, season, as_of)
                if counts[fid] != 1:
                    raise ValueError("Duplicate provider fixture ID")
                try:
                    stats, notices = recover_statistics(grouped.get(fid, []), row)
                except ValueError as exc:
                    stats, notices = [{}, {}], ["legacy_statistics_quarantined"]
                    excluded.append({"league": code, "fixture_id": fid, "reason": str(exc), "scope": "statistics_only"})
                warnings.update(notices)
                if not all(stats):
                    missing += 1
                else:
                    usable += 1
                if any(any(s.get(k) is None for k in ("corners", "shots_on", "yellow_cards", "red_cards")) for s in stats):
                    partial += 1
                entries.append({"league": code, "season": season, "api_row": row,
                                "stats": stats, "source": sources[code]})
                evidence = [r[EVIDENCE_KEY] for r in grouped.get(fid, []) if r.get(EVIDENCE_KEY)]
                if evidence and all(stats):
                    entries[-1]["statistics_evidence"] = evidence
            except (ValueError, KeyError, TypeError) as exc:
                excluded.append({"league": code, "fixture_id": fid, "reason": str(exc), "scope": "fixture"})
        provider_ids = {r.get("fixture", {}).get("id") for r in fixtures}
        for fid in sorted(set(grouped) - provider_ids):
            excluded.append({"league": code, "fixture_id": fid, "reason": "legacy_fixture_absent_from_provider_list", "scope": "fixture"})
        cohorts[code] = {"provider_fixtures": len(fixtures), "legacy_fixtures": len(grouped),
                         "paired_legacy_statistics": usable, "metadata_only": missing,
                         "fixtures_needing_raw_statistics": partial, "warnings": dict(warnings)}
    return {"schema": "historical-recovery.v1", "prepared_at": as_of.isoformat(),
            "entries": entries, "coverage": cohorts, "quarantined": excluded,
            "policy": "additive_only; provider_fixture_identity; legacy_zero_is_unknown; no_player_import"}


def apply_recovery_plan(session, plan):
    """Within caller's transaction/gate: insert absent records, never overwrite.

    No watermarks/KB queues/models are touched. Missing player/raw statistics
    keep fixture.payload_digest unset, allowing a later proper API bootstrap.
    """
    counts = Counter()
    rejected = []
    now = datetime.now(timezone.utc)
    for entry in plan["entries"]:
        row, code, year = entry["api_row"], entry["league"], entry["season"]
        fid = checked_fixture(row, code, year, now)
        competition = session.scalar(select(Competition).where(Competition.code == code))
        if competition is None or competition.api_football_id != COMPETITIONS[code].api_football_id:
            raise ValueError("Missing or conflicting competition catalogue: " + code)
        season = session.scalar(select(Season).where(Season.competition_id == competition.id, Season.year == year))
        if season is None:
            season = Season(competition_id=competition.id, year=year, label=f"{year}/{str(year+1)[-2:]}", is_current=False)
            session.add(season)
            session.flush()
        teams = []
        for side in ("home", "away"):
            raw = row["teams"][side]
            team = session.scalar(select(Team).where(Team.api_football_id == raw["id"]))
            if team is None:
                team = Team(api_football_id=raw["id"], name=raw["name"], logo_url=raw.get("logo"))
                session.add(team)
                session.flush()
            teams.append(team)
        fixture = session.scalar(select(Fixture).where(Fixture.api_football_id == fid))
        if fixture is not None:
            identity = (fixture.competition_id, fixture.season_id, fixture.home_team_id, fixture.away_team_id)
            if identity != (competition.id, season.id, teams[0].id, teams[1].id):
                rejected.append({"fixture_id": fid, "reason": "existing_fixture_identity_conflict"})
                continue
            counts["existing_fixtures_preserved"] += 1
            # Even missing rows on an existing fixture require a separately
            # reviewed repair: do not attach old stats to a corrected score.
            continue
        fixture = Fixture(api_football_id=fid, competition_id=competition.id, season_id=season.id,
                          home_team_id=teams[0].id, away_team_id=teams[1].id,
                          kickoff_utc=_parse_dt(row["fixture"]["date"]), status="FT",
                          home_goals=int(row["goals"]["home"]), away_goals=int(row["goals"]["away"]),
                          referee=row["fixture"].get("referee"), round=row["league"].get("round"),
                          last_fetched_at=now, payload_digest=None)
        session.add(fixture)
        session.flush()
        counts["fixtures_inserted"] += 1
        for index, side in enumerate(("home", "away")):
            values = entry["stats"][index]
            provenance = {"source": entry["source"], "prepared_at": plan["prepared_at"],
                          "zero_policy": "legacy_zero_is_unknown", "source_kind": "legacy_team_export_verified_fixture"}
            if entry.get("statistics_evidence"):
                provenance["statistics_evidence"] = entry["statistics_evidence"]
                provenance["zero_policy"] = "explicit_provider_zero_or_unknown"
            session.add(FixtureTeamStats(
                fixture_id=fixture.id, team_id=teams[index].id, opponent_team_id=teams[1-index].id,
                is_home=index == 0, goals=int(row["goals"][side]), **values,
                stats_json={**values, "_history_recovery": provenance}, raw_payload_digest=_digest(provenance),
            ))
            counts["team_rows_inserted"] += 1
    session.flush()
    return {"counts": dict(counts), "quarantined": rejected, "models_changed": False,
            "watermarks_changed": False, "players_imported": False}
