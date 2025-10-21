# LaLiga team profiles

import argparse
import json
import math
import os
from collections import Counter

import numpy as np
import pandas as pd

LEAGUE = 'LaLiga'

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input",
        default="/Users/sanduandrei/Desktop/Betting_RAG/Output/LaLiga_feature_engineering/team_engineered_features_2025.json",
        help="Path to team_engineered_features_YYYY.json",
    )
    p.add_argument(
        "--outdir",
        default=None,
        help="Output directory (defaults to input file's directory).",
    )
    p.add_argument("--league", default="EPL", help="League code or name.")
    p.add_argument("--season", default="2025/26", help="Season string.")
    p.add_argument("--basename", default="team_profiles_2025", help="Base filename for outputs.")
    return p.parse_args()


def load_team_fixtures(path: str) -> pd.DataFrame:
    # Accepts a JSON array or JSONL
    try:
        df = pd.read_json(path, lines=False)
    except ValueError:
        df = pd.read_json(path, lines=True)

    # Numeric coercion for known fields (safe if missing)
    numeric_cols = [
        "goals","assists","shots_total_x","shots_on_x","passes_total","accurate_passes",
        "fouls_committed","yellow_cards","red_cards","minutes","tackles","interceptions",
        "shots_accuracy_for","pass_accuracy_team","expected_goals","goals_prevented",
        "possession","corners","shots_total_y","shots_on_y","shots_on","shots_total",
        "dominance_index","goal_conversion_rate","shot_on_target_ratio","control_index",
        "cards_total","fouls_per_90_team","cards_per_90_team","cards_per_foul_team",
        "fouls_per_duel_team","aggression_index_raw","aggression_index_norm",
        "avg_goals_last_5","avg_cards_last_5","form_index_team"
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Ensure identity columns exist
    if "team" not in df.columns:
        df["team"] = None
    # Some datasets include "opponent" and "fixture" as strings; keep if present.
    return df


def pick_first_nonnull(row, candidates, default=np.nan):
    for c in candidates:
        if c in row and pd.notna(row[c]):
            return row[c]
    return default


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create stable 'for' and 'against' convenience columns without mutating originals.
    Many pipelines produce *_x as 'for' and *_y as 'against'; others just use base names.
    """
    # For-side shots
    df["_shots_for"] = df.apply(lambda r: pick_first_nonnull(r, ["shots_total_x", "shots_total"], 0.0), axis=1)
    df["_sot_for"]   = df.apply(lambda r: pick_first_nonnull(r, ["shots_on_x", "shots_on"], 0.0), axis=1)

    # Against-side shots (if available)
    df["_shots_against"] = df.apply(lambda r: pick_first_nonnull(r, ["shots_total_y"], np.nan), axis=1)
    df["_sot_against"]   = df.apply(lambda r: pick_first_nonnull(r, ["shots_on_y"], np.nan), axis=1)

    # Passing accuracy: prefer provided team accuracy; otherwise recompute from totals
    if "pass_accuracy_team" in df.columns:
        df["_pass_acc_for"] = df["pass_accuracy_team"] * 100.0 if df["pass_accuracy_team"].max() <= 1.0 else df["pass_accuracy_team"]
    else:
        if "passes_total" in df.columns and "accurate_passes" in df.columns:
            df["_pass_acc_for"] = 100.0 * (df["accurate_passes"] / df["passes_total"].replace(0, np.nan))
        else:
            df["_pass_acc_for"] = np.nan

    # SOT ratio and conversion helpers
    if "shot_on_target_ratio" in df.columns:
        df["_sot_ratio_for"] = df["shot_on_target_ratio"]
    else:
        df["_sot_ratio_for"] = df["_sot_for"] / df["_shots_for"].replace(0, np.nan)

    if "goal_conversion_rate" in df.columns:
        df["_conv_rate_for"] = df["goal_conversion_rate"]
    else:
        df["_conv_rate_for"] = df["goals"] / df["_shots_for"].replace(0, np.nan)

    return df


def aggregate_team_profiles(df: pd.DataFrame, league: str, season: str) -> pd.DataFrame:
    df = normalize_columns(df)

    # How many fixtures per team (robust to missing minutes)
    games_per_team = df.groupby("team").size().rename("matches_played")

    # Sum minutes then derive effective match-minutes (~ 11 * on-field minutes total).
    # This is OPTIONAL and only used to compute 'per-90 team' rates when useful.
    minutes_sum = df.groupby("team")["minutes"].sum(min_count=1).rename("sum_minutes")

    # Core sums
    sums = df.groupby("team").agg({
        "goals": "sum",
        "assists": "sum",
        "fouls_committed": "sum",
        "yellow_cards": "sum",
        "red_cards": "sum",
        "corners": "sum",
        "_shots_for": "sum",
        "_sot_for": "sum",
        "_shots_against": "sum",
        "_sot_against": "sum",
        "passes_total": "sum",
        "accurate_passes": "sum"
    })

    # Means for shape/quality indices and rates
    means = df.groupby("team").agg({
        "expected_goals": "mean",
        "goals_prevented": "mean",
        "possession": "mean",
        "_pass_acc_for": "mean",
        "_sot_ratio_for": "mean",
        "_conv_rate_for": "mean",
        "dominance_index": "mean",
        "control_index": "mean",
        "aggression_index_norm": "mean",
        "fouls_per_90_team": "mean",
        "cards_per_90_team": "mean",
        "cards_per_foul_team": "mean",
        "fouls_per_duel_team": "mean",
        "avg_goals_last_5": "mean",
        "avg_cards_last_5": "mean",
        "form_index_team": "mean"
    })

    # Combine
    prof = pd.concat([games_per_team, minutes_sum, sums, means], axis=1).reset_index()

    # Derived per-match averages
    mp = prof["matches_played"].replace(0, np.nan)
    prof["goals_for_pm"] = prof["goals"] / mp
    prof["assists_pm"] = prof["assists"] / mp
    prof["shots_for_pm"] = prof["_shots_for"] / mp
    prof["sot_for_pm"] = prof["_sot_for"] / mp
    prof["shots_against_pm"] = prof["_shots_against"] / mp
    prof["sot_against_pm"] = prof["_sot_against"] / mp
    prof["corners_pm"] = prof["corners"] / mp
    prof["fouls_pm"] = prof["fouls_committed"] / mp
    prof["yellows_pm"] = prof["yellow_cards"] / mp
    prof["reds_pm"] = prof["red_cards"] / mp

    # Cards per match (count reds as 1 card; adjust if you prefer weight 2)
    if "cards_total" in df.columns:
        cards_sum = df.groupby("team")["cards_total"].sum(min_count=1).rename("cards_total_sum")
        prof = prof.merge(cards_sum, on="team", how="left")
        prof["cards_pm"] = prof["cards_total_sum"] / mp
    else:
        prof["cards_pm"] = (prof["yellow_cards"] + prof["red_cards"]) / mp

    # Recompute whole-season pass accuracy from totals (more stable than mean of means)
    prof["pass_accuracy_pct"] = 100.0 * prof["accurate_passes"] / prof["passes_total"].replace(0, np.nan)
    # Fallback to mean if recomputed is NaN
    prof["pass_accuracy_pct"] = prof["pass_accuracy_pct"].fillna(prof["_pass_acc_for"])

    # Optional team-per-90 using sum of minutes / 11 (approx effective team minutes played)
    # This keeps comparability when some fixtures have extra time or player-minutes vary.
    team_min_per90_denom = (prof["sum_minutes"] / (90.0 * 11.0)).replace(0, np.nan)
    prof["fouls_per_90_calc"] = prof["fouls_committed"] / team_min_per90_denom
    prof["cards_per_90_calc"] = (prof["yellow_cards"] + prof["red_cards"]) / team_min_per90_denom

    # Style tags
    def make_style_tags(row):
        tags = []
        if pd.notna(row["possession"]) and row["possession"] >= 0.55:
            tags.append("possession-heavy")
        if pd.notna(row["possession"]) and row["possession"] <= 0.45:
            tags.append("counter-oriented")
        if pd.notna(row["aggression_index_norm"]) and row["aggression_index_norm"] >= 0.25:
            tags.append("aggressive")
        if pd.notna(row["cards_pm"]) and row["cards_pm"] >= 3.0:
            tags.append("card-prone")
        if pd.notna(row["cards_pm"]) and row["cards_pm"] < 1.5 and pd.notna(row["fouls_pm"]) and row["fouls_pm"] < 10:
            tags.append("disciplined")
        if pd.notna(row["control_index"]) and row["control_index"] >= 0.60:
            tags.append("control-high")
        if pd.notna(row["dominance_index"]) and row["dominance_index"] >= 0.60:
            tags.append("dominant")
        if pd.notna(row["_sot_ratio_for"]) and row["_sot_ratio_for"] >= 0.40:
            tags.append("efficient-finishing")
        return tags

    prof["style_tags"] = prof.apply(make_style_tags, axis=1)

    # Natural-language summary
    def summarize_team(row):
        name = row["team"]
        gpm = row.get("goals_for_pm", np.nan)
        xg = row.get("expected_goals", np.nan)
        poss = row.get("possession", np.nan)
        sotr = row.get("_sot_ratio_for", np.nan)
        ctrl = row.get("control_index", np.nan)
        dom = row.get("dominance_index", np.nan)
        cards = row.get("cards_pm", np.nan)
        fouls = row.get("fouls_pm", np.nan)
        tags = ", ".join(row.get("style_tags", [])) if row.get("style_tags") else "no special tags"

        parts = [
            f"{name}: {row['matches_played']} matches",
            f"{gpm:.2f} goals/match",
            f"xG {xg:.2f}/match" if pd.notna(xg) else None,
            f"possession {poss:.0%}" if pd.notna(poss) else None,
            f"SOT ratio {sotr:.2f}" if pd.notna(sotr) else None,
            f"control {ctrl:.2f}" if pd.notna(ctrl) else None,
            f"dominance {dom:.2f}" if pd.notna(dom) else None,
            f"cards {cards:.2f}/match" if pd.notna(cards) else None,
            f"fouls {fouls:.1f}/match" if pd.notna(fouls) else None,
            f"tags: {tags}",
        ]
        return " | ".join([p for p in parts if p])

    prof["summary_nl"] = prof.apply(summarize_team, axis=1)

    # Metadata for RAG
    prof["entity_type"] = "team"
    prof["league"] = league
    prof["season"] = season

    # Friendly output order
    front_cols = [
        "entity_type","league","season",
        "team","matches_played",
        "goals","assists","goals_for_pm","assists_pm",
        "shots_for_pm","sot_for_pm","shots_against_pm","sot_against_pm",
        "expected_goals","goals_prevented",
        "possession","pass_accuracy_pct","_sot_ratio_for","_conv_rate_for",
        "dominance_index","control_index","aggression_index_norm",
        "corners_pm",
        "fouls_committed","fouls_pm","yellows_pm","reds_pm","cards_pm",
        "fouls_per_90_team","cards_per_90_team","cards_per_foul_team","fouls_per_duel_team",
        "avg_goals_last_5","avg_cards_last_5","form_index_team",
        "style_tags","summary_nl"
    ]
    # Add any columns used in derivations if missing from df (safe guard)
    for c in front_cols:
        if c not in prof.columns:
            prof[c] = np.nan

    remaining = [c for c in prof.columns if c not in front_cols]
    prof = prof[front_cols + remaining]

    # Clean NaNs (JSONL-friendly)
    prof = prof.replace({np.nan: None})

    return prof


def write_outputs(df: pd.DataFrame, outdir: str, basename: str):
    os.makedirs(outdir, exist_ok=True)
    jsonl_path = os.path.join(outdir, f"{basename}.jsonl")
    csv_path = os.path.join(outdir, f"{basename}.csv")

    # JSONL
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            f.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")

    # CSV
    df.to_csv(csv_path, index=False)

    return jsonl_path, csv_path


def main():
    args = parse_args()
    in_path = args.input
    outdir = args.outdir or os.path.dirname(os.path.abspath(in_path))
    basename = args.basename

    print(f"📥 Loading team fixture data from: {in_path}")
    df = load_team_fixtures(in_path)
    print(f"✅ Loaded {len(df):,} team-fixture rows")

    print("🧮 Aggregating to team profiles…")
    profiles = aggregate_team_profiles(df, args.league, args.season)
    print(f"✅ Built {len(profiles):,} team profiles")

    print("💾 Writing outputs…")
    jsonl_path, csv_path = write_outputs(profiles, outdir, basename)
    print(f"📤 JSONL: {jsonl_path}")
    print(f"📤 CSV  : {csv_path}")
    print("🎉 Done.")


if __name__ == "__main__":
    main()