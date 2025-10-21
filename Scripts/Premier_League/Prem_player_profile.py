# Prem Player Profiles 

import argparse
import json
import math
import os
from collections import Counter

import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input",
        default="/Users/sanduandrei/Desktop/Betting_RAG/Output/Prem_feature_engineering/player_engineered_features_2025.json",
        help="Path to player_engineered_features_YYYY.json",
    )
    p.add_argument(
        "--outdir",
        default=None,
        help="Output directory (defaults to input file's directory).",
    )
    p.add_argument("--league", default="EPL", help="League code or name.")
    p.add_argument("--season", default="2025/26", help="Season string.")
    p.add_argument("--profiles_basename", default="player_profiles_2025", help="Base filename for outputs.")
    return p.parse_args()


def safe_div(n, d):
    try:
        if d is None or d == 0 or (isinstance(d, float) and math.isclose(d, 0.0)):
            return 0.0
        return float(n) / float(d)
    except Exception:
        return 0.0


def most_frequent(series):
    vals = [v for v in series if pd.notna(v)]
    if not vals:
        return None
    c = Counter(vals)
    return c.most_common(1)[0][0]


def load_player_fixtures(path: str) -> pd.DataFrame:
    # Accepts either a JSON array or JSONL
    try:
        df = pd.read_json(path, lines=False)
    except ValueError:
        df = pd.read_json(path, lines=True)
    # Standardize column names just in case (keep your originals too)
    # NOTE: We do NOT rename your fields; we only ensure numeric types.
    numeric_cols = [
        "minutes", "goals", "assists", "shots_total", "shots_on", "passes_total",
        "accurate_passes", "pass_accuracy_%", "tackles", "interceptions",
        "fouls_committed", "fouls_drawn", "yellow_cards", "red_cards",
        "duels_won", "duels_total", "shots_on_target_ratio", "goal_conversion_rate",
        "duel_win_ratio", "tackle_success_rate", "goal_involvement", "cards_total",
        "cards_per_90", "fouls_per_90", "cards_per_foul", "duel_foul_ratio",
        "avg_rating_last_4", "avg_goals_last_4", "avg_assists_last_4",
        "goal_involvement_last_4", "form_index", "performance_variance",
        "aggression_index_norm", "form_index_team", "control_index",
        "expected_foul_pressure", "expected_card_risk"
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Drop rows with no minutes (cannot compute per-90s)
    if "minutes" in df.columns:
        df = df[df["minutes"] > 0].copy()

    return df


def aggregate_player_profiles(df: pd.DataFrame, league: str, season: str) -> pd.DataFrame:
    # Ensure required identity columns exist
    for col in ["player_id", "name", "team", "position"]:
        if col not in df.columns:
            df[col] = None

    # Build aggregation dictionary (sum where per-90 will be computed; mean for ratios/indices)
    agg = {
        "minutes": "sum",
        "goals": "sum",
        "assists": "sum",
        "shots_total": "sum",
        "shots_on": "sum",
        "passes_total": "sum",
        "accurate_passes": "sum",
        "tackles": "sum",
        "interceptions": "sum",
        "fouls_committed": "sum",
        "fouls_drawn": "sum",
        "yellow_cards": "sum",
        "red_cards": "sum",
        "duels_won": "sum",
        "duels_total": "sum",
        "goal_involvement": "sum",
        "cards_total": "sum",
        # Averages of quality/shape indices
        "rating": "mean" if "rating" in df.columns else "mean",
        "pass_accuracy_%": "mean" if "pass_accuracy_%" in df.columns else "mean",
        "shots_on_target_ratio": "mean" if "shots_on_target_ratio" in df.columns else "mean",
        "goal_conversion_rate": "mean" if "goal_conversion_rate" in df.columns else "mean",
        "duel_win_ratio": "mean" if "duel_win_ratio" in df.columns else "mean",
        "tackle_success_rate": "mean" if "tackle_success_rate" in df.columns else "mean",
        "cards_per_90": "mean" if "cards_per_90" in df.columns else "mean",
        "fouls_per_90": "mean" if "fouls_per_90" in df.columns else "mean",
        "cards_per_foul": "mean" if "cards_per_foul" in df.columns else "mean",
        "duel_foul_ratio": "mean" if "duel_foul_ratio" in df.columns else "mean",
        "form_index": "mean" if "form_index" in df.columns else "mean",
        "performance_variance": "mean" if "performance_variance" in df.columns else "mean",
        "aggression_index_norm": "mean" if "aggression_index_norm" in df.columns else "mean",
        "form_index_team": "mean" if "form_index_team" in df.columns else "mean",
        "control_index": "mean" if "control_index" in df.columns else "mean",
        "expected_foul_pressure": "mean" if "expected_foul_pressure" in df.columns else "mean",
        "expected_card_risk": "mean" if "expected_card_risk" in df.columns else "mean",
    }

    group_cols = ["player_id"]

    # We also want a representative (mode) team, name, and position for the profile
    def mode_after_group(s):
        return most_frequent(s)

    grouped = (
        df.groupby(group_cols)
          .agg({**agg, "team": mode_after_group, "name": mode_after_group, "position": mode_after_group})
          .reset_index()
    )

    # Per-90s & derived metrics
    m = grouped["minutes"].replace({0: np.nan})
    grouped["goals_per_90"] = grouped["goals"] / (m / 90.0)
    grouped["assists_per_90"] = grouped["assists"] / (m / 90.0)
    grouped["shots_per_90"] = grouped["shots_total"] / (m / 90.0)
    grouped["sot_per_90"] = grouped["shots_on"] / (m / 90.0)
    grouped["tackles_per_90"] = grouped["tackles"] / (m / 90.0)
    grouped["interceptions_per_90"] = grouped["interceptions"] / (m / 90.0)
    grouped["fouls_per_90_calc"] = grouped["fouls_committed"] / (m / 90.0)
    grouped["cards_per_90_calc"] = grouped["cards_total"] / (m / 90.0)
    grouped["passes_per_90"] = grouped["passes_total"] / (m / 90.0)

    # Passing accuracy recomputed from sums (more stable than averaging percents)
    grouped["pass_accuracy_pct"] = 100.0 * grouped["accurate_passes"].pipe(
        lambda s: s / grouped["passes_total"].replace({0: np.nan})
    )
    # Fallback to mean column if recomputed is NaN
    if "pass_accuracy_%" in grouped.columns:
        grouped["pass_accuracy_pct"] = grouped["pass_accuracy_pct"].fillna(grouped["pass_accuracy_%"])

    # Duels win %
    grouped["duel_win_pct"] = 100.0 * grouped["duels_won"] / grouped["duels_total"].replace({0: np.nan})

    # Discipline helpers
    grouped["total_cards"] = grouped.get("cards_total", pd.Series([0] * len(grouped)))
    grouped["total_yellows"] = grouped.get("yellow_cards", pd.Series([0] * len(grouped)))
    grouped["total_reds"] = grouped.get("red_cards", pd.Series([0] * len(grouped)))

    # Style tags
    def make_style_tags(row):
        tags = []
        if pd.notna(row.get("aggression_index_norm", np.nan)) and row["aggression_index_norm"] > 0.25:
            tags.append("aggressive")
        if pd.notna(row.get("fouls_per_90_calc", np.nan)) and row["fouls_per_90_calc"] >= 2.0:
            tags.append("foul-prone")
        if pd.notna(row.get("cards_per_90_calc", np.nan)) and row["cards_per_90_calc"] >= 0.30:
            tags.append("card-risk")
        if pd.notna(row.get("duel_win_pct", np.nan)) and row["duel_win_pct"] >= 55:
            tags.append("strong-duels")
        if pd.notna(row.get("pass_accuracy_pct", np.nan)) and pd.notna(row.get("passes_per_90", np.nan)):
            if row["pass_accuracy_pct"] >= 85 and row["passes_per_90"] >= 30:
                tags.append("safe-passer")
        if pd.notna(row.get("shots_per_90", np.nan)) and row["shots_per_90"] >= 2.0:
            tags.append("shooter")
        if pd.notna(row.get("assists_per_90", np.nan)) and row["assists_per_90"] >= 0.20:
            tags.append("creator")
        return tags

    grouped["style_tags"] = grouped.apply(make_style_tags, axis=1)

    # Natural-language summary
    def summarize_player(row):
        name = row.get("name") or f"Player {row['player_id']}"
        team = row.get("team") or "Unknown Team"
        pos = row.get("position") or "N/A"
        goals90 = row.get("goals_per_90", 0.0)
        ast90 = row.get("assists_per_90", 0.0)
        duel = row.get("duel_win_pct", np.nan)
        pass_acc = row.get("pass_accuracy_pct", np.nan)
        style = ", ".join(row.get("style_tags", [])) if row.get("style_tags") else "no special tags"
        rat = row.get("rating", np.nan)
        cards90 = row.get("cards_per_90_calc", 0.0)

        parts = [
            f"{name} ({team}) – {pos}",
            f"{goals90:.2f} G/90, {ast90:.2f} A/90",
            f"duel win {duel:.0f}%" if pd.notna(duel) else None,
            f"pass acc {pass_acc:.0f}%" if pd.notna(pass_acc) else None,
            f"cards {cards90:.2f}/90",
            f"avg rating {rat:.2f}" if pd.notna(rat) else None,
            f"tags: {style}",
        ]
        return " | ".join([p for p in parts if p])

    grouped["summary_nl"] = grouped.apply(summarize_player, axis=1)

    # Static metadata for RAG filtering
    grouped["entity_type"] = "player"
    grouped["league"] = league
    grouped["season"] = season

    # Reorder columns (nice-to-have)
    col_order_front = [
        "entity_type", "league", "season", "player_id", "name", "team", "position",
        "minutes", "goals", "assists",
        "goals_per_90", "assists_per_90", "shots_per_90", "sot_per_90",
        "passes_per_90", "pass_accuracy_pct",
        "tackles_per_90", "interceptions_per_90",
        "fouls_per_90_calc", "cards_per_90_calc", "duel_win_pct",
        "rating", "aggression_index_norm", "form_index", "control_index",
        "expected_foul_pressure", "expected_card_risk",
        "total_yellows", "total_reds", "total_cards",
        "style_tags", "summary_nl",
    ]
    # Keep any remaining columns too
    remaining = [c for c in grouped.columns if c not in col_order_front]
    grouped = grouped[col_order_front + remaining]

    # Clean NaNs
    grouped = grouped.replace({np.nan: None})

    return grouped


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
    basename = args.profiles_basename

    print(f"📥 Loading player fixture data from: {in_path}")
    df = load_player_fixtures(in_path)
    print(f"✅ Loaded {len(df):,} player-fixture rows")

    print("🧮 Aggregating to player profiles…")
    profiles = aggregate_player_profiles(df, args.league, args.season)
    print(f"✅ Built {len(profiles):,} player profiles")

    print("💾 Writing outputs…")
    jsonl_path, csv_path = write_outputs(profiles, outdir, basename)
    print(f"📤 JSONL: {jsonl_path}")
    print(f"📤 CSV  : {csv_path}")
    print("🎉 Done.")


if __name__ == "__main__":
    main()
