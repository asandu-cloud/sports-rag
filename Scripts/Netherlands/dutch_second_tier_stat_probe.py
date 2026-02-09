#!/usr/bin/env python3
"""
Probe script: Dutch 2nd tier (Eerste Divisie) available statistics.

This script does not write files and does not run feature engineering.
It only calls API-Football and prints which stat fields are available.
"""

import argparse
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import requests
from dotenv import dotenv_values, load_dotenv


BASE_URL = "https://v3.football.api-sports.io"
DEFAULT_LEAGUE_ID = 89  # Eerste Divisie in API-Football
ENV_PATH = Path(__file__).resolve().parents[2] / ".env"


def is_present(value) -> bool:
    if value is None:
        return False
    if isinstance(value, str) and not value.strip():
        return False
    return True


def flatten_dict(data: Dict, parent: str = "") -> Iterable[Tuple[str, object]]:
    for key, value in (data or {}).items():
        k = f"{parent}.{key}" if parent else str(key)
        if isinstance(value, dict):
            yield from flatten_dict(value, k)
        else:
            yield (k, value)


def api_get(api_key: str, path: str, params: Dict) -> List[Dict]:
    url = f"{BASE_URL}{path}"
    headers = {"x-apisports-key": api_key}
    resp = requests.get(url, headers=headers, params=params, timeout=30)
    if resp.status_code != 200:
        msg = f"HTTP {resp.status_code} for {path}: {resp.text[:300]}"
        raise RuntimeError(msg)
    payload = resp.json()
    errors = payload.get("errors") or {}
    if errors:
        print(f"API warnings for {path}: {errors}", file=sys.stderr)
    return payload.get("response", []) or []


def get_finished_fixture_rows(api_key: str, league_id: int, season: int) -> List[Dict]:
    rows = api_get(
        api_key,
        "/fixtures",
        {"league": league_id, "season": season, "status": "FT"},
    )
    if rows:
        return rows
    # fallback if FT filter returns 0
    return api_get(api_key, "/fixtures", {"league": league_id, "season": season})


def collect_team_fixture_stat_types(api_key: str, fixture_ids: List[int]) -> Tuple[Counter, Counter]:
    seen = Counter()
    non_null = Counter()
    for fixture_id in fixture_ids:
        rows = api_get(api_key, "/fixtures/statistics", {"fixture": fixture_id})
        for team_block in rows:
            for stat in team_block.get("statistics", []) or []:
                name = stat.get("type")
                value = stat.get("value")
                if not name:
                    continue
                seen[name] += 1
                if is_present(value):
                    non_null[name] += 1
    return seen, non_null


def collect_player_stat_fields(api_key: str, fixture_ids: List[int]) -> Tuple[Counter, Counter]:
    seen = Counter()
    non_null = Counter()
    for fixture_id in fixture_ids:
        rows = api_get(api_key, "/fixtures/players", {"fixture": fixture_id})
        for team_block in rows:
            for player in team_block.get("players", []) or []:
                stats = (player.get("statistics") or [{}])[0] or {}
                for section, payload in stats.items():
                    if isinstance(payload, dict):
                        for sub_key, value in payload.items():
                            field = f"{section}.{sub_key}"
                            seen[field] += 1
                            if is_present(value):
                                non_null[field] += 1
                    else:
                        field = str(section)
                        seen[field] += 1
                        if is_present(payload):
                            non_null[field] += 1
    return seen, non_null


def collect_team_season_fields(
    api_key: str, league_id: int, season: int, team_ids: List[int]
) -> Tuple[Counter, Counter]:
    seen = Counter()
    non_null = Counter()
    for team_id in team_ids:
        rows = api_get(
            api_key,
            "/teams/statistics",
            {"league": league_id, "season": season, "team": team_id},
        )
        if not rows:
            continue
        # API-Football may return a dict for this endpoint (not always a list).
        if isinstance(rows, dict):
            data = rows
        elif isinstance(rows, list):
            data = rows[0] if rows else {}
        else:
            data = {}

        if not isinstance(data, dict) or not data:
            continue

        for field, value in flatten_dict(data):
            # skip metadata fields
            if field.startswith("league.") or field.startswith("team."):
                continue
            seen[field] += 1
            if is_present(value):
                non_null[field] += 1
    return seen, non_null


def print_field_list(title: str, seen: Counter, non_null: Counter) -> None:
    print(f"\n{title}")
    if not seen:
        print("  (none found)")
        return
    for name in sorted(seen.keys()):
        print(f"  - {name} [{non_null[name]}/{seen[name]} non-null]")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print available stat fields for Dutch 2nd tier from API-Football."
    )
    parser.add_argument("--league-id", type=int, default=DEFAULT_LEAGUE_ID)
    parser.add_argument("--season", type=int, default=2025)
    parser.add_argument("--max-fixtures", type=int, default=20)
    parser.add_argument("--max-teams", type=int, default=10)
    args = parser.parse_args()

    # Explicit .env load from project root.
    if ENV_PATH.exists():
        load_dotenv(dotenv_path=ENV_PATH, override=False)

    api_key = os.getenv("API-FOOTBALL-KEY")
    if not api_key and ENV_PATH.exists():
        # Fallback: read key directly in case shell/env handling skips dashed names.
        api_key = dotenv_values(ENV_PATH).get("API-FOOTBALL-KEY")
    if not api_key:
        print("Missing API key. Set API-FOOTBALL-KEY in .env or environment.")
        sys.exit(1)

    fixtures = get_finished_fixture_rows(api_key, args.league_id, args.season)
    if not fixtures:
        print("No fixtures returned for this league/season.")
        sys.exit(1)

    fixture_ids = [f["fixture"]["id"] for f in fixtures[: args.max_fixtures] if f.get("fixture", {}).get("id")]
    team_ids = []
    seen_teams = set()
    for f in fixtures:
        for side in ("home", "away"):
            tid = (f.get("teams", {}).get(side) or {}).get("id")
            if tid and tid not in seen_teams:
                seen_teams.add(tid)
                team_ids.append(tid)
    team_ids = team_ids[: args.max_teams]

    print(f"League ID: {args.league_id}")
    print(f"Season: {args.season}")
    print(f"Fixtures scanned: {len(fixture_ids)}")
    print(f"Teams scanned (season stats): {len(team_ids)}")

    tf_seen, tf_non_null = collect_team_fixture_stat_types(api_key, fixture_ids)
    pf_seen, pf_non_null = collect_player_stat_fields(api_key, fixture_ids)
    ts_seen, ts_non_null = collect_team_season_fields(api_key, args.league_id, args.season, team_ids)

    print_field_list("Team fixture statistic types", tf_seen, tf_non_null)
    print_field_list("Player statistic fields", pf_seen, pf_non_null)
    print_field_list("Team season statistic fields", ts_seen, ts_non_null)


if __name__ == "__main__":
    main()
