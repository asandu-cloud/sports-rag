"""Ligue1 player profile aggregation -- thin wrapper around EPL core logic."""

import argparse
import os
import sys

# Allow importing from the Premier_League directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Premier_League"))
from Prem_player_profile import load_player_fixtures, aggregate_player_profiles, write_outputs


def parse_args():
    p = argparse.ArgumentParser(description="Aggregate Ligue1 player profiles.")
    p.add_argument(
        "--input",
        default="/Users/sanduandrei/Desktop/Betting_RAG/Output/Ligue1_feature_engineering/player_engineered_features_2025.json",
        help="Path to player_engineered_features_YYYY.json",
    )
    p.add_argument("--outdir", default=None, help="Output directory (defaults to input file's directory).")
    p.add_argument("--league", default="Ligue1", help="League code.")
    p.add_argument("--season", default="2024/25", help="Season string.")
    p.add_argument("--profiles_basename", default="player_profiles_2025", help="Base filename for outputs.")
    return p.parse_args()


def main():
    args = parse_args()
    in_path = args.input
    outdir = args.outdir or os.path.dirname(os.path.abspath(in_path))
    basename = args.profiles_basename

    print(f"Loading player fixture data from: {in_path}")
    df = load_player_fixtures(in_path)
    print(f"Loaded {len(df):,} player-fixture rows")

    print("Aggregating to player profiles...")
    profiles = aggregate_player_profiles(df, args.league, args.season)
    print(f"Built {len(profiles):,} player profiles")

    print("Writing outputs...")
    jsonl_path, csv_path = write_outputs(profiles, outdir, basename)
    print(f"JSONL: {jsonl_path}")
    print(f"CSV  : {csv_path}")
    print("Done.")


if __name__ == "__main__":
    main()
