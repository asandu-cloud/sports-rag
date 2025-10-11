# Premier League feature engineering

# feature_engineering.py

import pandas as pd
from pathlib import Path
from datetime import datetime

# --- CONFIG ---
LEAGUE = "Premier League"
SEASON = 2025

# Input data paths
PLAYER_PATH = Path("/Users/sanduandrei/Desktop/Betting_RAG/Output/Prem_output/player_fixture_stats_2025.json")
FIXTURE_PATH = Path("/Users/sanduandrei/Desktop/Betting_RAG/Output/Prem_teams/team_fixture_stats_2025.json")

# Output directory
OUTPUT_PATH = Path("/Users/sanduandrei/Desktop/Betting_RAG/Output/Prem_feature_engineering")
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)


# ===============================================================
# 1️⃣ DATA LOADING
# ===============================================================
def load_player_data():
    df = pd.read_json(PLAYER_PATH)
    print(f"📂 Loaded {len(df)} player-fixture rows from {PLAYER_PATH.name}")
    return df


def load_fixture_stats():
    fx = pd.read_json(FIXTURE_PATH)

    # Normalize column names
    fx.columns = fx.columns.str.strip().str.lower().str.replace(" ", "_")

    # Rename for consistency
    rename_map = {
        "corner_kicks": "corners",
        "ball_possession": "possession",
        "total_shots": "shots_total",
        "shots_on_goal": "shots_on",
        "shots_on_target": "shots_on",
        "shots_on_goals": "shots_on",
        "total_passes": "passes_total",
        "passes_accurate": "accurate_passes",
        "passes_%": "pass_accuracy_rate",
        "yellow_cards": "yellow_cards",
        "red_cards": "red_cards",
        "fouls": "fouls_committed",
    }
    fx.rename(columns={k: v for k, v in rename_map.items() if k in fx.columns}, inplace=True)

    # Ensure numeric consistency
    numeric_cols = [
        "shots_on", "shots_total", "corners", "possession",
        "fouls_committed", "yellow_cards", "red_cards",
        "expected_goals", "goals_prevented",
    ]
    for col in numeric_cols:
        if col not in fx.columns:
            print(f"⚠️ Column '{col}' missing in fixture data — creating zeros.")
            fx[col] = 0
        fx[col] = pd.to_numeric(fx[col], errors="coerce").fillna(0)

    print("📋 Fixture columns after load/rename:", list(fx.columns))
    print(f"📂 Loaded {len(fx)} fixture-level rows from {FIXTURE_PATH.name}")
    return fx


# ===============================================================
# 2️⃣ PLAYER-LEVEL FEATURE ENGINEERING
# ===============================================================
def engineer_player_efficiency(df):
    for col in ["shots_on", "shots_total", "duels_total", "tackles"]:
        if col in df.columns:
            df[col] = df[col].replace(0, pd.NA)

    df["shots_on_target_ratio"] = df["shots_on"] / df["shots_total"]
    df["goal_conversion_rate"] = df["goals"] / df["shots_on"]
    df["duel_win_ratio"] = df["duels_won"] / df["duels_total"]
    df["tackle_success_rate"] = df["tackles"] / df["duels_total"]
    df["goal_involvement"] = df["goals"].fillna(0) + df["assists"].fillna(0)

    df.replace([float("inf"), -float("inf")], pd.NA, inplace=True)
    df.fillna(0, inplace=True)
    print("✅ Engineered player-level efficiency features (1.1)")
    return df


def engineer_player_aggression(df):
    for col in ["minutes", "duels_total", "fouls_committed"]:
        if col in df.columns:
            df[col] = df[col].replace(0, pd.NA)

    df["cards_total"] = df["yellow_cards"].fillna(0) + df["red_cards"].fillna(0)
    df["cards_per_90"] = (df["cards_total"] / df["minutes"]) * 90
    df["fouls_per_90"] = (df["fouls_committed"] / df["minutes"]) * 90
    df["cards_per_foul"] = df["cards_total"] / df["fouls_committed"]
    df["duel_foul_ratio"] = df["fouls_committed"] / df["duels_total"]

    df.replace([float("inf"), -float("inf")], pd.NA, inplace=True)
    df.fillna(0, inplace=True)
    print("✅ Engineered player aggression & disciplinary features (1.2)")
    return df


def engineer_player_form(df):
    df = df.sort_values(by=["player_id", "fixture"]).copy()
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")

    group = df.groupby("player_id")
    df["avg_rating_last_4"] = group["rating"].transform(lambda x: x.rolling(4, 1).mean())
    df["avg_goals_last_4"] = group["goals"].transform(lambda x: x.rolling(4, 1).mean())
    df["avg_assists_last_4"] = group["assists"].transform(lambda x: x.rolling(4, 1).mean())

    df["goal_involvement_last_4"] = df["avg_goals_last_4"] + df["avg_assists_last_4"]
    df["form_index"] = (
        df["avg_rating_last_4"].fillna(0) * 0.6
        + df["goal_involvement_last_4"].fillna(0) * 0.4
    )

    df["performance_variance"] = group["rating"].transform(lambda x: x.rolling(4, 2).var())
    df.fillna(0, inplace=True)
    print("✅ Engineered player form & consistency features (1.3)")
    return df


# ===============================================================
# 3️⃣ TEAM-LEVEL FEATURE ENGINEERING
# ===============================================================
def engineer_team_core_features(df):
    agg_map = {
        "goals": "sum",
        "assists": "sum",
        "shots_total": "sum",
        "shots_on": "sum",
        "passes_total": "sum",
        "accurate_passes": "sum",
        "fouls_committed": "sum",
        "yellow_cards": "sum",
        "red_cards": "sum",
        "minutes": "sum",
        "tackles": "sum",
        "interceptions": "sum",
    }
    team_fixture = df.groupby(["fixture", "team"], dropna=False).agg(agg_map).reset_index()
    team_fixture["shots_accuracy_for"] = team_fixture["shots_on"] / team_fixture["shots_total"].replace(0, pd.NA)
    team_fixture["pass_accuracy_team"] = team_fixture["accurate_passes"] / team_fixture["passes_total"].replace(0, pd.NA)
    team_fixture.fillna(0, inplace=True)
    print(f"✅ Engineered team-level core aggregates (2.1): {len(team_fixture)} rows")
    return team_fixture


def merge_team_with_fixture(team_df, fixture_df):
    if fixture_df is None or fixture_df.empty:
        print("⚠️ Fixture stats missing, skipping merge.")
        return team_df

    fx = fixture_df.copy()
    df = team_df.copy()
    fx["fixture_key"] = fx["home_team"].str.lower().str.strip() + "_vs_" + fx["away_team"].str.lower().str.strip()
    df["fixture_key"] = df["fixture"].str.lower().str.strip()

    merged = df.merge(
        fx[
            ["fixture_key", "team", "expected_goals", "goals_prevented", "possession",
             "corners", "shots_total", "shots_on"]
        ],
        on=["fixture_key", "team"],
        how="left",
    )

    for col in ["shots_on", "shots_total", "corners", "possession"]:
        if col not in merged.columns:
            print(f"⚠️ Column '{col}' missing — filling with zeros.")
            merged[col] = 0
        merged[col] = merged[col].fillna(0)

    merged["dominance_index"] = (
        (merged["possession"] * 0.4)
        + ((merged["shots_on"] / merged["shots_total"].replace(0, pd.NA)).fillna(0) * 0.3)
        + ((merged["corners"] / (merged["corners"].max() or 1)) * 0.3)
    )
    print("✅ Successfully merged fixture-level stats.")
    return merged


def engineer_team_strength_efficiency(team_df):
    df = team_df.copy()
    for col in ["shots_total", "passes_total", "tackles", "fouls_committed"]:
        df[col] = df[col].replace(0, pd.NA)

    df["goal_conversion_rate"] = df["goals"] / df["shots_total"]
    df["shot_on_target_ratio"] = df["shots_on"] / df["shots_total"]
    df["control_index"] = (
        (df["pass_accuracy_team"].fillna(0) * 0.5)
        + ((df["passes_total"].fillna(0) / df["passes_total"].max()) * 0.3)
        + ((df["corners"].fillna(0) / (df["corners"].max() or 1)) * 0.2)
    )

    df.fillna(0, inplace=True)
    print(f"✅ Engineered team strength & efficiency metrics (2.2): {len(df)} rows")
    return df


def engineer_team_aggression_discipline(team_df):
    """(2.3) Compute team-level aggression and discipline metrics."""
    df = team_df.copy()

    # --- SAFETY ---
    for col in ["fouls_committed", "yellow_cards", "red_cards", "tackles", "interceptions"]:
        if col not in df.columns:
            df[col] = 0

    # --- BASE RATES ---
    df["cards_total"] = df["yellow_cards"] + df["red_cards"]

    # ✅ FIX: make sure cards_per_90_team always exists (per match)
    df["fouls_per_90_team"] = df["fouls_committed"].fillna(0)
    df["cards_per_90_team"] = df["cards_total"].fillna(0)

    df["cards_per_foul_team"] = df["cards_total"] / df["fouls_committed"].replace(0, pd.NA)
    df["fouls_per_duel_team"] = df["fouls_committed"] / (
        df["tackles"] + df["interceptions"]
    ).replace(0, pd.NA)

    # --- AGGRESSION INDEX ---
    df["aggression_index_raw"] = (
        df["fouls_per_90_team"].fillna(0) * 0.5
        + df["cards_per_90_team"].fillna(0) * 0.3
        + df["fouls_per_duel_team"].fillna(0) * 0.2
    )

    min_val, max_val = df["aggression_index_raw"].min(), df["aggression_index_raw"].max()
    df["aggression_index_norm"] = (
        (df["aggression_index_raw"] - min_val) / (max_val - min_val)
        if max_val > min_val else 0
    )

    df.fillna(0, inplace=True)
    print(f"✅ Engineered team aggression & discipline metrics (2.3): {len(df)} rows")
    return df



def engineer_team_form_consistency(team_df):
    df = team_df.copy().sort_values(by=["team", "fixture"]).reset_index(drop=True)
    group = df.groupby("team")
    w = 5
    df["avg_goals_last_5"] = group["goals"].transform(lambda x: x.rolling(w, 1).mean())
    df["avg_cards_last_5"] = group["cards_total"].transform(lambda x: x.rolling(w, 1).mean())
    df["form_index_team"] = (
        df["avg_goals_last_5"].fillna(0) * 0.5
        - df["avg_cards_last_5"].fillna(0) * 0.3
        + df["goal_conversion_rate"].fillna(0) * 0.2
    )
    df.fillna(0, inplace=True)
    print(f"✅ Engineered team form & consistency metrics (2.4): {len(df)} rows")
    return df


# ===============================================================
# 4️⃣ CONTEXTUAL (PLAYER + OPPONENT)
# ===============================================================
def engineer_player_contextual_features(player_df, team_df):
    """(1.4) Merge opponent-based contextual features for each player record."""
    df = player_df.copy()
    team = team_df.copy()

    # ✅ Ensure expected team-level columns exist before merge
    required_cols = [
        "cards_per_90_team", "fouls_per_90_team",
        "aggression_index_norm", "form_index_team", "control_index"
    ]
    for col in required_cols:
        if col not in team.columns:
            print(f"⚠️ '{col}' missing in team data — creating zeros.")
            team[col] = 0


    fixture_teams = team.groupby("fixture")["team"].apply(list).reset_index(name="teams")
    fixture_teams = fixture_teams[fixture_teams["teams"].apply(len) == 2]

    opponent_map = {}
    for _, row in fixture_teams.iterrows():
        t1, t2 = row["teams"]
        opponent_map[(row["fixture"], t1)] = t2
        opponent_map[(row["fixture"], t2)] = t1

    df = df.merge(team[["fixture", "team", "aggression_index_norm", "form_index_team", "control_index"]],
                  on=["fixture", "team"], how="left")

    df["opponent"] = df.apply(lambda x: opponent_map.get((x["fixture"], x["team"])), axis=1)

    df = df.merge(
        team[["fixture", "team", "aggression_index_norm", "cards_per_90_team",
              "fouls_per_90_team", "form_index_team", "control_index"]],
        left_on=["fixture", "opponent"], right_on=["fixture", "team"],
        how="left", suffixes=("", "_opp")
    )

    if "team_opp" in df.columns:
        df.drop(columns=["team_opp"], inplace=True)

    df["agg_diff"] = df["aggression_index_norm"] - df["aggression_index_norm_opp"]
    df["form_diff"] = df["form_index_team"] - df["form_index_team_opp"]
    df["control_diff"] = df["control_index"] - df["control_index_opp"]

    df["is_defensive"] = df["position"].fillna("").str.contains("DF|CB|LB|RB", case=False).astype(int)
    df["expected_foul_pressure"] = df["aggression_index_norm_opp"] * df["is_defensive"]
    df["expected_card_risk"] = (
        df["cards_per_90_team_opp"].fillna(0)
        * df["aggression_index_norm_opp"].fillna(0)
        * df["is_defensive"]
    )

    df.fillna(0, inplace=True)
    print(f"✅ Engineered contextual opponent-based player features (1.4): {len(df)} rows")
    return df


# ===============================================================
# 5️⃣ MASTER PIPELINE
# ===============================================================
def run_feature_engineering():
    df = load_player_data()
    fixture_stats = load_fixture_stats()

    df = engineer_player_efficiency(df)
    df = engineer_player_aggression(df)
    df = engineer_player_form(df)

    team_features = engineer_team_core_features(df)
    team_features = merge_team_with_fixture(team_features, fixture_stats)
    team_features = engineer_team_strength_efficiency(team_features)
    team_features = engineer_team_aggression_discipline(team_features)
    team_features = engineer_team_form_consistency(team_features)

    df = engineer_player_contextual_features(df, team_features)

    # Save outputs
    df.to_csv(OUTPUT_PATH / f"player_engineered_features_{SEASON}.csv", index=False)
    df.to_json(OUTPUT_PATH / f"player_engineered_features_{SEASON}.json", orient="records", indent=4)
    team_features.to_csv(OUTPUT_PATH / f"team_engineered_features_{SEASON}.csv", index=False)
    team_features.to_json(OUTPUT_PATH / f"team_engineered_features_{SEASON}.json", orient="records", indent=4)

    print(f"💾 Saved engineered datasets → {OUTPUT_PATH}")
    print(f"   • Player rows: {len(df)}")
    print(f"   • Team rows: {len(team_features)}")
    return df, team_features


# ===============================================================
# MAIN
# ===============================================================
if __name__ == "__main__":
    start = datetime.now()
    run_feature_engineering()
    print(f"⏱️ Completed in {(datetime.now() - start).total_seconds():.2f} seconds")