"""Legacy-model refresh orchestrator.

Pulls API-Football data, runs feature engineering, rebuilds domestic player
profiles, then optionally normalizes and embeds the changed season.  It is
used by ``spixctl refresh-season`` while the live prediction path still reads
the legacy Output/Chroma artefacts.

Usage:
    # Pull raw data + feature engineering for seasons 2023 and 2024:
    python Scripts/pull_season.py --season 2023 2024

    # Full pipeline (pull + normalize + embed):
    python Scripts/pull_season.py --season 2023 2024 --normalize --embed

    # Single league:
    python Scripts/pull_season.py --season 2024 --league EPL

    # Skip data pull, just normalize + embed existing files:
    python Scripts/pull_season.py --skip-pull --normalize --embed
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Sequence

ROOT = Path(__file__).resolve().parents[1]

# ---- Script registry: league → scripts ----
LEAGUES = {
    "EPL": {
        "team_stats":  "Scripts/Premier_League/PL_stats_for_teams.py",
        "player_stats": "Scripts/Premier_League/prem_player_stats_per_gw.py",
        "feature_eng":  "Scripts/Premier_League/PL_feature_engineering.py",
        "player_profile": "Scripts/Premier_League/Prem_player_profile.py",
        "feature_dir": "Prem_feature_engineering",
    },
    "LaLiga": {
        "team_stats":  "Scripts/La_Liga/LaLiga_stats_for_teams.py",
        "player_stats": "Scripts/La_Liga/laliga_player_stats_per_gw.py",
        "feature_eng":  "Scripts/La_Liga/LaLiga_feature_engineering.py",
        "player_profile": "Scripts/La_Liga/LaLiga_player_profile.py",
        "feature_dir": "LaLiga_feature_engineering",
    },
    "SerieA": {
        "team_stats":  "Scripts/Seria_A/SeriaA_team_stats.py",
        "player_stats": "Scripts/Seria_A/SeriaA_player_stats.py",
        "feature_eng":  "Scripts/Seria_A/SeriaA_feature_engineering.py",
        "player_profile": "Scripts/Seria_A/SeriaA_player_profile.py",
        "feature_dir": "SeriaA_feature_engineering",
    },
    "Bundesliga": {
        "team_stats":  "Scripts/Bundesliga/bundesliga_stats_for_teams.py",
        "player_stats": "Scripts/Bundesliga/bundesliga_stats_for_players.py",
        "feature_eng":  "Scripts/Bundesliga/Bundesliga_feature_engineering.py",
        "player_profile": "Scripts/Bundesliga/Bundesliga_player_profile.py",
        "feature_dir": "Bundesliga_feature_engineering",
    },
    "Ligue1": {
        "team_stats":  "Scripts/Ligue_1/Ligue1_stats_for_teams.py",
        "player_stats": "Scripts/Ligue_1/Ligue_1_Stats_for_players.py",
        "feature_eng":  "Scripts/Ligue_1/Ligue1_feature_engineering.py",
        "player_profile": "Scripts/Ligue_1/Ligue1_player_profile.py",
        "feature_dir": "Ligue1_feature_engineering",
    },
    "UCL": {
        "team_stats":   "Scripts/Champions_League/UCL_stats_for_teams.py",
        "player_stats": "Scripts/Champions_League/UCL_player_stats.py",
        "feature_eng":  "Scripts/Champions_League/UCL_feature_engineering.py",
    },
    "UEL": {
        "team_stats":   "Scripts/Europa_League/UEL_stats_for_teams.py",
        "player_stats": "Scripts/Europa_League/UEL_player_stats.py",
        "feature_eng":  "Scripts/Europa_League/UEL_feature_engineering.py",
    },
    "UECL": {
        "team_stats":   "Scripts/Conference_League/UECL_stats_for_teams.py",
        "player_stats": "Scripts/Conference_League/UECL_player_stats.py",
        "feature_eng":  "Scripts/Conference_League/UECL_feature_engineering.py",
    },
}


def run_command(command: Sequence[str], label: str) -> bool:
    """Run one child stage and make failures visible to the caller."""
    print(f"  [RUN] {label}: {' '.join(command)}")
    result = subprocess.run(command, cwd=str(ROOT))
    if result.returncode != 0:
        print(f"  [FAIL] {label} exited with code {result.returncode}")
        return False
    print(f"  [OK]  {label}")
    return True


def run_script(script_rel: str, season: int, label: str) -> bool:
    """Run a per-league collection/feature script with its season argument."""
    script = ROOT / script_rel
    if not script.exists():
        print(f"  [FAIL] {label}: {script} not found")
        return False

    return run_command([sys.executable, str(script), "--season", str(season)], label)


def season_label(year: int) -> str:
    """Return the repository's human-readable season label for an API year."""
    return f"{year}/{str(year + 1)[-2:]}"


def run_player_profile(league: str, season: int) -> bool:
    """Build a current player-profile file when that competition supports it.

    UEFA player-profile aggregation has not been implemented yet.  It is an
    intentional skip rather than a successful profile refresh, and the caller
    keeps the team-market path usable for those competitions.
    """
    config = LEAGUES[league]
    profile_rel = config.get("player_profile")
    feature_dir = config.get("feature_dir")
    if not profile_rel or not feature_dir:
        print(f"  [SKIP] {league} player profiles: no profile builder registered")
        return True

    script = ROOT / profile_rel
    input_path = ROOT / "Output" / feature_dir / f"player_engineered_features_{season}.json"
    if not script.exists():
        print(f"  [FAIL] {league} player profiles: {script} not found")
        return False
    if not input_path.exists():
        print(f"  [FAIL] {league} player profiles: expected {input_path} was not created")
        return False

    command = [
        sys.executable,
        str(script),
        "--input", str(input_path),
        "--league", league,
        "--season", season_label(season),
        "--profiles_basename", f"player_profiles_{season}",
    ]
    return run_command(command, f"{league} player profiles")


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Pull API-Football data and run feature engineering for multiple seasons."
    )
    parser.add_argument(
        "--season", type=int, nargs="+", required=True,
        help="Season year(s) to process (e.g. 2023 2024 2025)"
    )
    parser.add_argument(
        "--league", type=str, nargs="*", default=None,
        help="League(s) to process (e.g. EPL LaLiga). Default: all registered leagues."
    )
    parser.add_argument(
        "--skip-pull", action="store_true",
        help="Skip API data pull (stages 1-2), only run normalize/embed."
    )
    parser.add_argument(
        "--normalize", action="store_true",
        help="Run 00_normalize_to_docs.py --all after data pull."
    )
    parser.add_argument(
        "--embed", action="store_true",
        help="Run 01_embed_and_upsert.py --all-leagues --skip-existing after normalization."
    )
    parser.add_argument(
        "--build-referees", action="store_true",
        help="Rebuild referee profiles + fixture-referee detail (after feature engineering, before normalize)."
    )
    parser.add_argument(
        "--retrain-ml", action="store_true",
        help="Retrain cumulative-profile ML models after normalize/embed."
    )
    parser.add_argument(
        "--skip-player-profiles", action="store_true",
        help="Skip domestic player-profile aggregation (normally rebuilt after every refresh).",
    )
    args = parser.parse_args(argv)

    leagues = args.league if args.league else list(LEAGUES.keys())
    # Validate league names
    for lg in leagues:
        if lg not in LEAGUES:
            print(f"Unknown league: {lg}. Valid: {', '.join(LEAGUES.keys())}")
            return 1

    if not args.skip_pull:
        for season in args.season:
            print(f"\n{'='*60}")
            print(f"SEASON {season} ({season}/{season+1 - 2000})")
            print(f"{'='*60}")

            for lg in leagues:
                scripts = LEAGUES[lg]
                print(f"\n--- {lg} (season {season}) ---")

                # Stage 1: Data collection
                if not run_script(scripts["team_stats"], season, f"{lg} team stats"):
                    return 1
                if not run_script(scripts["player_stats"], season, f"{lg} player stats"):
                    return 1

                # Stage 2: Feature engineering
                if not run_script(scripts["feature_eng"], season, f"{lg} feature engineering"):
                    return 1

                if not args.skip_player_profiles:
                    if not run_player_profile(lg, season):
                        return 1

    # Stage 2.5: Build referee profiles + fixture-referee detail
    if args.build_referees:
        print(f"\n{'='*60}")
        print("BUILDING REFEREE PROFILES + FIXTURE DETAIL")
        print(f"{'='*60}")
        ref_script = ROOT / "Scripts" / "rag_ingest" / "referee_data.py"
        cmd = [sys.executable, str(ref_script), "--build"]
        if not run_command(cmd, "referee profiles + fixture detail"):
            return 1

    # Stage 3: Normalize (incremental when single season, full when multiple)
    if args.normalize:
        print(f"\n{'='*60}")
        norm_script = ROOT / "Scripts" / "rag_ingest" / "00_normalize_to_docs.py"
        if len(args.season) == 1:
            # Single season: only normalize files matching that season (fast)
            season_year = args.season[0]
            print(f"NORMALIZING (incremental — season {season_year} only)")
            print(f"{'='*60}")
            if args.league:
                for lg in leagues:
                    cmd = [sys.executable, str(norm_script), "--league", lg, "--season", str(season_year)]
                    if not run_command(cmd, f"normalize {lg} {season_year}"):
                        return 1
            else:
                cmd = [sys.executable, str(norm_script), "--all", "--season", str(season_year)]
                if not run_command(cmd, f"normalize season {season_year}"):
                    return 1
        else:
            # Multiple seasons: normalize everything
            print("NORMALIZING (full — multiple seasons)")
            print(f"{'='*60}")
            cmd = [sys.executable, str(norm_script), "--all"]
            if not run_command(cmd, "normalize all seasons"):
                return 1

    # Stage 4: Embed (incremental by default — skip-existing is the default behavior)
    if args.embed:
        print(f"\n{'='*60}")
        print("EMBEDDING (incremental — skipping completed fixtures, updating profiles)")
        print(f"{'='*60}")
        embed_script = ROOT / "Scripts" / "rag_ingest" / "01_embed_and_upsert.py"
        cmd = [sys.executable, str(embed_script), "--all-leagues"]
        if len(args.season) == 1:
            cmd.extend(["--season", str(args.season[0])])
        if not run_command(cmd, "embed normalized documents"):
            return 1

    # Stage 5: Retrain ML models
    if args.retrain_ml:
        print(f"\n{'='*60}")
        print("RETRAINING ML MODELS (ml_edge.py --train-cumulative)")
        print(f"{'='*60}")
        ml_script = ROOT / "Scripts" / "rag_ingest" / "ml_edge.py"
        cmd = [sys.executable, str(ml_script), "--train-cumulative"]
        if not run_command(cmd, "retrain cumulative ML models"):
            return 1

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
