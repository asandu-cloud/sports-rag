"""
Orchestrator: pull API-Football data, run feature engineering, normalize, and embed
for one or more seasons across all 5 leagues.

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

ROOT = Path(__file__).resolve().parents[1]

# ---- Script registry: league → scripts ----
LEAGUES = {
    "EPL": {
        "team_stats":  "Scripts/Premier_League/PL_stats_for_teams.py",
        "player_stats": "Scripts/Premier_League/prem_player_stats_per_gw.py",
        "feature_eng":  "Scripts/Premier_League/PL_feature_engineering.py",
    },
    "LaLiga": {
        "team_stats":  "Scripts/La_Liga/LaLiga_stats_for_teams.py",
        "player_stats": "Scripts/La_Liga/laliga_player_stats_per_gw.py",
        "feature_eng":  "Scripts/La_Liga/LaLiga_feature_engineering.py",
    },
    "SerieA": {
        "team_stats":  "Scripts/Seria_A/SeriaA_team_stats.py",
        "player_stats": "Scripts/Seria_A/SeriaA_player_stats.py",
        "feature_eng":  "Scripts/Seria_A/SeriaA_feature_engineering.py",
    },
    "Bundesliga": {
        "team_stats":  "Scripts/Bundesliga/bundesliga_stats_for_teams.py",
        "player_stats": "Scripts/Bundesliga/bundesliga_stats_for_players.py",
        "feature_eng":  "Scripts/Bundesliga/Bundesliga_feature_engineering.py",
    },
    "Ligue1": {
        "team_stats":  "Scripts/Ligue_1/Ligue1_stats_for_teams.py",
        "player_stats": "Scripts/Ligue_1/Ligue_1_Stats_for_players.py",
        "feature_eng":  "Scripts/Ligue_1/Ligue1_feature_engineering.py",
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


def run_script(script_rel: str, season: int, label: str) -> bool:
    """Run a Python script with --season flag. Returns True on success."""
    script = ROOT / script_rel
    if not script.exists():
        print(f"  [SKIP] {label}: {script} not found")
        return False

    cmd = [sys.executable, str(script), "--season", str(season)]
    print(f"  [RUN] {label}: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(ROOT))
    if result.returncode != 0:
        print(f"  [FAIL] {label} exited with code {result.returncode}")
        return False
    print(f"  [OK]  {label}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Pull API-Football data and run feature engineering for multiple seasons."
    )
    parser.add_argument(
        "--season", type=int, nargs="+", required=True,
        help="Season year(s) to process (e.g. 2023 2024 2025)"
    )
    parser.add_argument(
        "--league", type=str, nargs="*", default=None,
        help="League(s) to process (e.g. EPL LaLiga). Default: all 5 leagues."
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
        "--retrain-ml", action="store_true",
        help="Retrain ML models (Elo + regression) after normalize/embed."
    )
    args = parser.parse_args()

    leagues = args.league if args.league else list(LEAGUES.keys())
    # Validate league names
    for lg in leagues:
        if lg not in LEAGUES:
            print(f"Unknown league: {lg}. Valid: {', '.join(LEAGUES.keys())}")
            sys.exit(1)

    if not args.skip_pull:
        for season in args.season:
            print(f"\n{'='*60}")
            print(f"SEASON {season} ({season}/{season+1 - 2000})")
            print(f"{'='*60}")

            for lg in leagues:
                scripts = LEAGUES[lg]
                print(f"\n--- {lg} (season {season}) ---")

                # Stage 1: Data collection
                run_script(scripts["team_stats"], season, f"{lg} team stats")
                run_script(scripts["player_stats"], season, f"{lg} player stats")

                # Stage 2: Feature engineering
                run_script(scripts["feature_eng"], season, f"{lg} feature engineering")

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
                    print(f"  [RUN] {' '.join(cmd)}")
                    subprocess.run(cmd, cwd=str(ROOT))
            else:
                cmd = [sys.executable, str(norm_script), "--all", "--season", str(season_year)]
                print(f"  [RUN] {' '.join(cmd)}")
                subprocess.run(cmd, cwd=str(ROOT))
        else:
            # Multiple seasons: normalize everything
            print("NORMALIZING (full — multiple seasons)")
            print(f"{'='*60}")
            cmd = [sys.executable, str(norm_script), "--all"]
            print(f"  [RUN] {' '.join(cmd)}")
            subprocess.run(cmd, cwd=str(ROOT))

    # Stage 4: Embed (incremental by default — skip-existing is the default behavior)
    if args.embed:
        print(f"\n{'='*60}")
        print("EMBEDDING (incremental — skipping existing fixtures/players)")
        print(f"{'='*60}")
        embed_script = ROOT / "Scripts" / "rag_ingest" / "01_embed_and_upsert.py"
        cmd = [sys.executable, str(embed_script), "--all-leagues"]
        print(f"  [RUN] {' '.join(cmd)}")
        subprocess.run(cmd, cwd=str(ROOT))

    # Stage 5: Retrain ML models
    if args.retrain_ml:
        print(f"\n{'='*60}")
        print("RETRAINING ML MODELS (ml_edge.py --train)")
        print(f"{'='*60}")
        ml_script = ROOT / "Scripts" / "rag_ingest" / "ml_edge.py"
        cmd = [sys.executable, str(ml_script), "--train"]
        print(f"  [RUN] {' '.join(cmd)}")
        subprocess.run(cmd, cwd=str(ROOT))

    print("\nDone.")


if __name__ == "__main__":
    main()
