# Prem Player Profiles

import argparse
import json
import math
import os
from collections import Counter

import numpy as np
import pandas as pd


# ── Constants ────────────────────────────────────────────────────────────────
STARTER_MINUTES_THRESHOLD = 60   # >= 60 min = counted as a starter appearance
RECENT_WINDOW = 5                # last N team fixtures for recent role analysis


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

    # Mark starter appearances (>= STARTER_MINUTES_THRESHOLD)
    if "minutes" in df.columns:
        df["_is_starter"] = (df["minutes"] >= STARTER_MINUTES_THRESHOLD).astype(int)
    else:
        df["_is_starter"] = 0

    return df


# ── Fix 1: Starter/Sub Role Classification ──────────────────────────────────
def compute_role_classification(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per player_id, compute role classification fields from the raw fixture rows.
    Returns a DataFrame indexed by player_id with new role columns.
    """
    role_agg = df.groupby("player_id").agg(
        appearances_total=("minutes", "count"),
        appearances_as_starter=("_is_starter", "sum"),
        avg_minutes_per_appearance=("minutes", "mean"),
    ).reset_index()

    role_agg["start_rate"] = (
        role_agg["appearances_as_starter"] / role_agg["appearances_total"]
    ).round(3)

    role_agg["appearances_as_starter"] = role_agg["appearances_as_starter"].astype(int)
    role_agg["avg_minutes_per_appearance"] = role_agg["avg_minutes_per_appearance"].round(1)

    return role_agg


# ── Fix 2: Starter-Only Per-90 Rates ────────────────────────────────────────
def compute_starter_per90(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-90 rates using ONLY starter appearances (>= STARTER_MINUTES_THRESHOLD).
    This eliminates the sub-appearance rate inflation problem.
    """
    starters = df[df["_is_starter"] == 1].copy()
    if starters.empty:
        return pd.DataFrame(columns=["player_id"])

    starter_agg = starters.groupby("player_id").agg(
        starter_minutes=("minutes", "sum"),
        starter_goals=("goals", "sum"),
        starter_assists=("assists", "sum"),
        starter_shots_on=("shots_on", "sum"),
        starter_shots_total=("shots_total", "sum"),
        starter_fouls=("fouls_committed", "sum"),
        starter_cards_total=("cards_total", "sum"),
    ).reset_index()

    m = starter_agg["starter_minutes"].replace({0: np.nan})
    starter_agg["starter_goals_per_90"] = (starter_agg["starter_goals"] / (m / 90.0)).round(3)
    starter_agg["starter_assists_per_90"] = (starter_agg["starter_assists"] / (m / 90.0)).round(3)
    starter_agg["starter_sot_per_90"] = (starter_agg["starter_shots_on"] / (m / 90.0)).round(3)
    starter_agg["starter_shots_per_90"] = (starter_agg["starter_shots_total"] / (m / 90.0)).round(3)
    starter_agg["starter_fouls_per_90"] = (starter_agg["starter_fouls"] / (m / 90.0)).round(3)
    starter_agg["starter_cards_per_90"] = (starter_agg["starter_cards_total"] / (m / 90.0)).round(3)

    # Drop intermediate sum columns (keep only per-90 rates + starter_minutes)
    keep_cols = [
        "player_id", "starter_minutes",
        "starter_goals_per_90", "starter_assists_per_90",
        "starter_sot_per_90", "starter_shots_per_90",
        "starter_fouls_per_90", "starter_cards_per_90",
    ]
    return starter_agg[keep_cols]


# ── Fix 4: Minutes Risk Score ───────────────────────────────────────────────
def compute_minutes_risk(row) -> str:
    """
    Classify a player's minutes risk based on start_rate and avg minutes.
    """
    sr = row.get("start_rate") or 0.0
    avg_min = row.get("avg_minutes_per_appearance") or 0.0

    if sr >= 0.80 and avg_min >= 75:
        return "nailed_on"
    elif sr >= 0.60 and avg_min >= 60:
        return "likely_starter"
    elif sr >= 0.30:
        return "rotation"
    else:
        return "bench"


# ── Fix 5: Recent Form with Minutes Context ────────────────────────────────
def compute_recent_minutes_context(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each player, look at the last RECENT_WINDOW *team* fixtures and compute:
    - last_5_minutes: list of minutes played (0 if player did not appear)
    - recent_starts_last_5: count of starts in those fixtures
    - recent_role_trend: "starting" / "benched" / "mixed"
    """
    # Build per-team fixture ordering.  We use the fixture string as a proxy
    # for ordering (the engineered features are already sorted chronologically
    # within the feature engineering pipeline via fixture_date).
    # First, assign a fixture sequence number per team.
    team_fixture_order = (
        df.drop_duplicates(subset=["team", "fixture"])
          .sort_values(["team", "fixture"])
          .groupby("team")
          .cumcount()
    )
    # Build a map: (team, fixture) -> sequence_number
    fixture_seq_df = df.drop_duplicates(subset=["team", "fixture"]).sort_values(["team", "fixture"]).copy()
    fixture_seq_df["_seq"] = team_fixture_order.values

    # For each team, identify the last RECENT_WINDOW fixtures
    max_seq = fixture_seq_df.groupby("team")["_seq"].max().reset_index()
    max_seq.columns = ["team", "_max_seq"]

    fixture_seq_df = fixture_seq_df.merge(max_seq, on="team")
    recent_fixtures = fixture_seq_df[
        fixture_seq_df["_seq"] > fixture_seq_df["_max_seq"] - RECENT_WINDOW
    ][["team", "fixture", "_seq"]].copy()

    # For each player, check which of their team's recent fixtures they appeared in
    results = []
    for pid, player_rows in df.groupby("player_id"):
        team = most_frequent(player_rows["team"])
        if team is None:
            results.append({
                "player_id": pid,
                "last_5_minutes": [],
                "recent_starts_last_5": 0,
                "recent_role_trend": "bench",
            })
            continue

        team_recent = recent_fixtures[
            recent_fixtures["team"].str.lower().str.strip() == str(team).lower().strip()
        ].sort_values("_seq")

        if team_recent.empty:
            results.append({
                "player_id": pid,
                "last_5_minutes": [],
                "recent_starts_last_5": 0,
                "recent_role_trend": "bench",
            })
            continue

        # For each recent team fixture, find if the player appeared and how many minutes
        player_fixture_map = {}
        for _, pr in player_rows.iterrows():
            fx_key = str(pr.get("fixture", "")).lower().strip()
            player_fixture_map[fx_key] = int(pr.get("minutes") or 0)

        last_5_minutes = []
        starts = 0
        for _, rf in team_recent.iterrows():
            fx_key = str(rf["fixture"]).lower().strip()
            mins = player_fixture_map.get(fx_key, 0)
            last_5_minutes.append(mins)
            if mins >= STARTER_MINUTES_THRESHOLD:
                starts += 1

        # Determine trend from last RECENT_WINDOW team fixtures
        if starts >= 4:
            trend = "starting"
        elif starts <= 1:
            trend = "benched"
        else:
            trend = "mixed"

        results.append({
            "player_id": pid,
            "last_5_minutes": last_5_minutes,
            "recent_starts_last_5": starts,
            "recent_role_trend": trend,
        })

    return pd.DataFrame(results)


def aggregate_player_profiles(df: pd.DataFrame, league: str, season: str) -> pd.DataFrame:
    # Ensure required identity columns exist
    for col in ["player_id", "name", "team", "position"]:
        if col not in df.columns:
            df[col] = None

    # ── Fix 1: Compute role classification before grouping ───────────────
    role_df = compute_role_classification(df)

    # ── Fix 2: Compute starter-only per-90 rates ────────────────────────
    starter_per90_df = compute_starter_per90(df)

    # ── Fix 5: Compute recent minutes context ───────────────────────────
    recent_ctx_df = compute_recent_minutes_context(df)

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

    # ── Merge role classification (Fix 1) ────────────────────────────────
    grouped = grouped.merge(role_df, on="player_id", how="left")

    # ── Merge starter per-90 rates (Fix 2) ──────────────────────────────
    if not starter_per90_df.empty:
        grouped = grouped.merge(starter_per90_df, on="player_id", how="left")
    else:
        for col in ["starter_minutes", "starter_goals_per_90", "starter_assists_per_90",
                     "starter_sot_per_90", "starter_shots_per_90",
                     "starter_fouls_per_90", "starter_cards_per_90"]:
            grouped[col] = None

    # ── Merge recent minutes context (Fix 5) ────────────────────────────
    grouped = grouped.merge(recent_ctx_df, on="player_id", how="left")

    # Per-90s & derived metrics (all appearances -- kept for backward compatibility)
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

    # ── Fix 4: Minutes risk classification ──────────────────────────────
    grouped["minutes_risk"] = grouped.apply(compute_minutes_risk, axis=1)

    # Style tags (updated to include minutes risk tags)
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
        # Minutes risk tag for downstream visibility
        mr = row.get("minutes_risk", "")
        if mr == "nailed_on":
            tags.append("nailed-on-starter")
        elif mr == "bench":
            tags.append("bench-player")
        elif mr == "rotation":
            tags.append("rotation-risk")
        return tags

    grouped["style_tags"] = grouped.apply(make_style_tags, axis=1)

    # Natural-language summary (updated to include role and minutes context)
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

        # New: role context
        apps = row.get("appearances_total", 0) or 0
        starts = row.get("appearances_as_starter", 0) or 0
        sr = row.get("start_rate", 0.0) or 0.0
        avg_min = row.get("avg_minutes_per_appearance", 0.0) or 0.0
        mr = row.get("minutes_risk", "unknown")
        trend = row.get("recent_role_trend", "unknown")

        # Use starter per-90 when available (more accurate for starters)
        sg90 = row.get("starter_goals_per_90")
        sa90 = row.get("starter_assists_per_90")
        rate_label = "all"
        disp_g90 = goals90 or 0.0
        disp_a90 = ast90 or 0.0
        if sg90 is not None and pd.notna(sg90) and starts >= 3:
            disp_g90 = sg90
            disp_a90 = sa90 if (sa90 is not None and pd.notna(sa90)) else (ast90 or 0.0)
            rate_label = "starter"

        parts = [
            f"{name} ({team}) - {pos}",
            f"{disp_g90:.2f} G/90 ({rate_label}), {disp_a90:.2f} A/90 ({rate_label})",
            f"role: {mr} ({starts}/{apps} starts, avg {avg_min:.0f}min, trend:{trend})",
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
        # All-appearances per-90 (backward compat)
        "goals_per_90", "assists_per_90", "shots_per_90", "sot_per_90",
        "passes_per_90", "pass_accuracy_pct",
        "tackles_per_90", "interceptions_per_90",
        "fouls_per_90_calc", "cards_per_90_calc", "duel_win_pct",
        # Starter-only per-90 (Fix 2)
        "starter_minutes", "starter_goals_per_90", "starter_assists_per_90",
        "starter_sot_per_90", "starter_shots_per_90",
        "starter_fouls_per_90", "starter_cards_per_90",
        # Role classification (Fix 1)
        "appearances_total", "appearances_as_starter", "start_rate",
        "avg_minutes_per_appearance",
        # Minutes risk (Fix 4)
        "minutes_risk",
        # Recent context (Fix 5)
        "recent_starts_last_5", "recent_role_trend", "last_5_minutes",
        # Existing
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
