# Premier League feature engineering

# feature_engineering.py

import pandas as pd
from pathlib import Path
from datetime import datetime

# --- CONFIG ---
LEAGUE = 'Premier League'
SEASON = 2025
DATA_DIR = Path("/Users/sanduandrei/Desktop/Betting_RAG/Output/Prem_output/player_fixture_stats_2025.json")
OUTPUT_PATH = Path('/Users/sanduandrei/Desktop/Betting_RAG/Output/Prem_feature_engineering')
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

# --- LOADING DATA ---
def load_player_data():
    df = pd.read_json(DATA_DIR)
    print(f"Loaded {len(df)} player-fixture rows from {DATA_DIR.name}")
    return df

# --- PLAYER LEVEL ---
def engineer_player_efficiency(df):
    """Compute finishing, conversion, duel and defensive efficiency metrics."""

    for col in ["shots_on", "shots_total", "duels_total", "tackles"]:
        df[col] = df[col].replace(0, pd.NA)

    df["shots_on_target_ratio"] = df["shots_on"] / df["shots_total"]
    df["goal_conversion_rate"] = df["goals"] / df["shots_on"]
    df["duel_win_ratio"] = df["duels_won"] / df["duels_total"]
    df["tackle_success_rate"] = df["tackles"] / df["duels_total"]
    df["goal_involvement"] = df["goals"].fillna(0) + df["assists"].fillna(0)

    # Handle division by zero or missing data
    df.replace([float("inf"), -float("inf")], pd.NA, inplace=True)
    df.fillna(0, inplace=True)

    print("Engineered player-level efficiency features (1.1)")
    return df


# --- AGGRESION & DISCIPLINARY ---
def engineer_player_aggression(df):

    # No div errors
    df["minutes"].replace(0, pd.NA, inplace=True)
    df["duels_total"].replace(0, pd.NA, inplace=True)
    df["fouls_committed"].replace(0, pd.NA, inplace=True)

    # Indicators for aggression
    df["cards_total"] = df["yellow_cards"].fillna(0) + df["red_cards"].fillna(0)
    df["cards_per_90"] = (df["cards_total"] / df["minutes"]) * 90
    df["fouls_per_90"] = (df["fouls_committed"] / df["minutes"]) * 90
    df["cards_per_foul"] = df["cards_total"] / df["fouls_committed"]
    df["duel_foul_ratio"] = df["fouls_committed"] / df["duels_total"]

    # Clean up 
    df.replace([float("inf"), -float("inf")], pd.NA, inplace=True)
    df.fillna(0, inplace=True)

    print("Engineered player aggression & disciplinary features (1.2)")
    return df

def engineer_player_form(df):
    """(1.3) Player form and consistency metrics based on last 4 fixtures."""

    # Ensure correct sorting
    df = df.sort_values(by=["player_id", "fixture"]).copy()

    # Convert rating to numeric (API returns strings sometimes)
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")

    # Group by player for rolling calculations
    df["avg_rating_last_4"] = (
        df.groupby("player_id")["rating"]
        .transform(lambda x: x.rolling(window=4, min_periods=1).mean())
    )

    df["avg_goals_last_4"] = (
        df.groupby("player_id")["goals"]
        .transform(lambda x: x.rolling(window=4, min_periods=1).mean())
    )

    df["avg_assists_last_4"] = (
        df.groupby("player_id")["assists"]
        .transform(lambda x: x.rolling(window=4, min_periods=1).mean())
    )

    # Compute goal involvement rolling average
    df["goal_involvement_last_4"] = (
        df["avg_goals_last_4"] + df["avg_assists_last_4"]
    )

    # Composite form index — combines rating and productivity
    df["form_index"] = (
        (df["avg_rating_last_4"].fillna(0) * 0.6)
        + (df["goal_involvement_last_4"].fillna(0) * 0.4)
    )

    # Consistency indicator — rating variance (lower is better)
    df["performance_variance"] = (
        df.groupby("player_id")["rating"]
        .transform(lambda x: x.rolling(window=4, min_periods=2).var())
    )

    df.fillna(0, inplace=True)

    print("Engineered player form & consistency features (1.3, window=4)")
    return df

# ---   ---
def engineer_team_core_features(df):
    """
    (2.1) Aggregate player-level data into team-level performance metrics per fixture.
    Includes core stats like goals, fouls, cards, corners, and derived ratios.
    """

    # --- Base aggregation ---
    agg_map = {
        "goals": "sum",
        "assists": "sum",
        "shots_total": "sum",
        "shots_on": "sum",
        "passes_total": "sum",
        "accurate_passes": "sum",
        "fouls_committed": "sum",
        "fouls_drawn": "sum",
        "yellow_cards": "sum",
        "red_cards": "sum",
        "minutes": "sum",
        "tackles": "sum",
        "interceptions": "sum",
    }

    # Aggregate to team x fixture
    team_fixture = (
        df.groupby(["fixture", "team"], dropna=False)
        .agg(agg_map)
        .reset_index()
    )

    # --- Add derived aggregates ---
    team_fixture["goal_difference"] = team_fixture["goals"] - team_fixture["assists"]  # placeholder (replace with match data later)
    team_fixture["shots_accuracy_for"] = team_fixture["shots_on"] / team_fixture["shots_total"].replace(0, pd.NA)
    team_fixture["pass_accuracy_team"] = team_fixture["accurate_passes"] / team_fixture["passes_total"].replace(0, pd.NA)
    team_fixture["fouls_per_card"] = team_fixture["fouls_committed"] / (
        (team_fixture["yellow_cards"] + team_fixture["red_cards"]).replace(0, pd.NA)
    )

    # --- Add placeholders for corners (will be merged later) ---
    team_fixture["corners_won"] = pd.NA
    team_fixture["corners_conceded"] = pd.NA
    team_fixture["corners_balance"] = team_fixture["corners_won"].fillna(0) - team_fixture["corners_conceded"].fillna(0)

    # Clean up
    team_fixture.fillna(0, inplace=True)

    print(f"✅ Engineered team-level aggregates (2.1): {len(team_fixture)} team-fixture rows")
    return team_fixture

# --- ---
def engineer_team_strength_efficiency(team_df):
    """
    (2.2) Compute advanced team efficiency metrics for attack, defense, and control.
    """
    df = team_df.copy()

    # --- SAFETY ---
    for col in ["shots_total", "passes_total", "tackles", "fouls_committed", "corners_won"]:
        df[col] = df[col].replace(0, pd.NA)

    # --- ATTACK EFFICIENCY ---
    df["goal_conversion_rate"] = df["goals"] / df["shots_total"]
    df["shot_on_target_ratio"] = df["shots_on"] / df["shots_total"]
    df["goals_per_90"] = (df["goals"] / df["minutes"]) * 90

    # --- DEFENSIVE STRENGTH ---
    df["defensive_duel_efficiency"] = df["tackles"] / (df["tackles"] + df["interceptions"])
    df["fouls_to_tackles_ratio"] = df["fouls_committed"] / df["tackles"]
    df["cards_total"] = df["yellow_cards"] + df["red_cards"]
    df["cards_per_foul_team"] = df["cards_total"] / df["fouls_committed"]

    # --- POSSESSION / CONTROL ---
    df["build_up_intensity"] = df["passes_total"] / 90
    df["control_index"] = (
        (df["pass_accuracy_team"].fillna(0) * 0.5)
        + ((df["passes_total"].fillna(0) / df["passes_total"].max()) * 0.3)
        + ((df["corners_won"].fillna(0) / (df["corners_won"].max() or 1)) * 0.2)
    )

    # --- SET PIECES ---
    df["corner_efficiency"] = df["goals"] / df["corners_won"]
    df["corners_balance"] = df["corners_won"].fillna(0) - df["corners_conceded"].fillna(0)

    # --- CLEANUP ---
    df.replace([float("inf"), -float("inf")], pd.NA, inplace=True)
    df.fillna(0, inplace=True)

    print(f"✅ Engineered team strength & efficiency metrics (2.2) for {len(df)} rows")
    return df

# --- ---
def engineer_team_aggression_discipline(team_df):
    """
    (2.3) Compute team-level aggression and discipline metrics.
    Includes foul/card rates, duel ratios, and normalized aggression index.
    """

    df = team_df.copy()
    # Safety: ensure aggression_index_norm exists in both dataframes
    if "aggression_index_norm" not in team.columns:
        team["aggression_index_norm"] = 0


    # --- SAFETY ---
    for col in ["minutes", "fouls_committed", "yellow_cards", "red_cards", "tackles", "interceptions"]:
        df[col] = df[col].replace(0, pd.NA)

    # --- BASE RATES ---
    df["cards_total"] = df["yellow_cards"] + df["red_cards"]
    df["fouls_per_90_team"] = (df["fouls_committed"] / df["minutes"]) * 90
    df["cards_per_90_team"] = (df["cards_total"] / df["minutes"]) * 90
    df["cards_per_foul_team"] = df["cards_total"] / df["fouls_committed"]
    df["fouls_per_duel_team"] = df["fouls_committed"] / (df["tackles"] + df["interceptions"])

    # --- AGGRESSION INDEX ---
    df["aggression_index_raw"] = (
        (df["fouls_per_90_team"].fillna(0) * 0.5)
        + (df["cards_per_90_team"].fillna(0) * 0.3)
        + (df["fouls_per_duel_team"].fillna(0) * 0.2)
    )

    # Normalize aggression index between 0 and 1
    min_val = df["aggression_index_raw"].min()
    max_val = df["aggression_index_raw"].max()
    df["aggression_index_norm"] = (df["aggression_index_raw"] - min_val) / (max_val - min_val)

    # --- OPPONENT IMPACT PLACEHOLDER ---
    # These will be merged later using match-level data (requires home/away info)
    df["opponent_card_tendency"] = pd.NA
    df["opponent_foul_tendency"] = pd.NA

    # Ensure aggression_index_norm exists (fallback)
    if "aggression_index_norm" not in df.columns or df["aggression_index_norm"].isna().all():
        df["aggression_index_norm"] = df["aggression_index_raw"].fillna(0)
        # Normalize safely if possible
        if df["aggression_index_norm"].max() != df["aggression_index_norm"].min():
            df["aggression_index_norm"] = (
                (df["aggression_index_norm"] - df["aggression_index_norm"].min())
                / (df["aggression_index_norm"].max() - df["aggression_index_norm"].min())
            )
        else:
            df["aggression_index_norm"] = 0


    # --- CLEANUP ---
    df.replace([float("inf"), -float("inf")], pd.NA, inplace=True)
    df.fillna(0, inplace=True)

    print(f"✅ Engineered team aggression & discipline metrics (2.3) for {len(df)} rows")
    if "cards_per_90_team" not in df.columns:
        df["cards_per_90_team"] = (df["cards_total"] / df["minutes"]) * 90
    if "fouls_per_90_team" not in df.columns:
        df["fouls_per_90_team"] = (df["fouls_committed"] / df["minutes"]) * 90

    return df

# --- --- 
def engineer_team_form_consistency(team_df):
    """
    (2.4) Compute rolling form, momentum, and consistency metrics for teams.
    Uses a 5-match rolling window per team.
    """
    df = team_df.copy()
    df = df.sort_values(by=["team", "fixture"]).reset_index(drop=True)

    # --- Rolling averages for offensive & defensive features ---
    rolling_window = 5

    group = df.groupby("team")

    df["avg_goals_last_5"] = group["goals"].transform(lambda x: x.rolling(window=rolling_window, min_periods=1).mean())
    df["avg_shots_on_last_5"] = group["shots_on"].transform(lambda x: x.rolling(window=rolling_window, min_periods=1).mean())
    df["avg_fouls_last_5"] = group["fouls_committed"].transform(lambda x: x.rolling(window=rolling_window, min_periods=1).mean())
    df["avg_cards_last_5"] = group["cards_total"].transform(lambda x: x.rolling(window=rolling_window, min_periods=1).mean())

    # --- Goal conversion trend ---
    df["goal_conversion_last_5"] = df["avg_goals_last_5"] / (group["shots_total"].transform(lambda x: x.rolling(window=rolling_window, min_periods=1).mean()))

    # --- Momentum (change vs previous average) ---
    df["attack_momentum"] = group["goals"].transform(lambda x: x - x.shift(1))
    df["discipline_momentum"] = group["cards_total"].transform(lambda x: x - x.shift(1))

    # --- Composite form index ---
    df["form_index_team"] = (
        (df["avg_goals_last_5"].fillna(0) * 0.4)
        + (df["goal_conversion_last_5"].fillna(0) * 0.3)
        - (df["avg_cards_last_5"].fillna(0) * 0.15)
        - (df["avg_fouls_last_5"].fillna(0) * 0.15)
    )

    # --- Consistency indicator ---
    df["performance_variance_team"] = group["goals"].transform(lambda x: x.rolling(window=rolling_window, min_periods=2).var())

    # --- Placeholder for clean sheet ratio (needs goals_conceded data) ---
    df["clean_sheet_ratio_last_5"] = pd.NA

    # --- Cleanup ---
    df.replace([float("inf"), -float("inf")], pd.NA, inplace=True)
    df.fillna(0, inplace=True)

    print(f"✅ Engineered team form, consistency & momentum metrics (2.4, window={rolling_window}) for {len(df)} rows")
    return df


def engineer_player_contextual_features(player_df, team_df):
    """
    (1.4) Merge opponent-based contextual features for each player record.
    """

    df = player_df.copy()
    team = team_df.copy()

    # Step 1 — Create opponent mapping for each fixture
    fixture_teams = (
        team.groupby("fixture")["team"]
        .apply(list)
        .reset_index(name="teams")
    )

    # Fixtures with 2 teams only (valid matches)
    fixture_teams = fixture_teams[fixture_teams["teams"].apply(len) == 2]

    # Build mapping: (fixture, team) → opponent
    opponent_map = {}
    for _, row in fixture_teams.iterrows():
        t1, t2 = row["teams"]
        opponent_map[(row["fixture"], t1)] = t2
        opponent_map[(row["fixture"], t2)] = t1

    # Step 2 — Merge player's team stats
    df = df.merge(
        team[["fixture", "team", "aggression_index_norm", "form_index_team", "control_index"]],
        on=["fixture", "team"],
        how="left",
        suffixes=("", "_team")
    )

    # Step 3 — Add opponent stats via mapping
    df["opponent"] = df.apply(lambda x: opponent_map.get((x["fixture"], x["team"])), axis=1)

    df = df.merge(
        team[["fixture", "team", "aggression_index_norm", "cards_per_90_team",
            "fouls_per_90_team", "control_index", "form_index_team"]],
        left_on=["fixture", "opponent"],
        right_on=["fixture", "team"],
        how="left"
    )

# Step 3b — Rename opponent columns explicitly
    df.rename(
        columns={
            "aggression_index_norm": "aggression_index_norm_opp",
            "cards_per_90_team": "cards_per_90_team_opp",
            "fouls_per_90_team": "fouls_per_90_team_opp",
            "control_index": "control_index_opp",
            "form_index_team": "form_index_team_opp"
        },
        inplace=True
    )


    # Step 4 — Compute differential & interaction features
    df["agg_diff"] = df["aggression_index_norm"] - df["aggression_index_norm_opp"]
    df["form_diff"] = df["form_index_team"] - df["form_index_team_opp"]
    df["control_diff"] = df["control_index"] - df["control_index_opp"]

    # Step 5 — Player-specific interaction risk
    df["is_defensive"] = df["position"].fillna("").str.contains("DF|CB|LB|RB", case=False).astype(int)
    df["expected_foul_pressure"] = df["aggression_index_norm_opp"] * df["is_defensive"]
    df["expected_card_risk"] = (
        df["cards_per_90_team_opp"].fillna(0)
        * df["aggression_index_norm_opp"].fillna(0)
        * df["is_defensive"]
    )

    df.fillna(0, inplace=True)

    print(f"✅ Engineered contextual opponent-based player features (1.4) for {len(df)} rows")
    return df

# --- MASTER PIPELINE ---
# --- MASTER PIPELINE ---
def run_feature_engineering():
    # --- Load base player data ---
    df = load_player_data()

    # ===============================================================
    # 🧍 PLAYER-LEVEL FEATURES
    # ===============================================================
    df = engineer_player_efficiency(df)       # (1.1) Finishing, duels, involvements
    df = engineer_player_aggression(df)       # (1.2) Cards/fouls & aggression ratios
    df = engineer_player_form(df)             # (1.3) Rolling form & consistency

    # ===============================================================
    # 🏟️ TEAM-LEVEL FEATURES
    # ===============================================================
    team_features = engineer_team_core_features(df)              # (2.1) Core aggregates
    team_features = engineer_team_strength_efficiency(team_features)  # (2.2) Strength & efficiency
    team_features = engineer_team_aggression_discipline(team_features) # (2.3) Aggression & discipline
    team_features = engineer_team_form_consistency(team_features)      # (2.4) Form & momentum

    # ===============================================================
    # 🔁 CONTEXTUAL (PLAYER + OPPONENT INTERACTION)
    # ===============================================================
    df = engineer_player_contextual_features(df, team_features)  # (1.4) Opponent-based context

    # ===============================================================
    # 💾 SAVE OUTPUTS
    # ===============================================================
    OUTPUT_PATH_DIR = Path(OUTPUT_PATH)
    OUTPUT_PATH_DIR.mkdir(parents=True, exist_ok=True)

    # --- Player-level enriched dataset ---
    player_csv = OUTPUT_PATH_DIR / f"player_engineered_features_{SEASON}.csv"
    player_json = OUTPUT_PATH_DIR / f"player_engineered_features_{SEASON}.json"
    df.to_csv(player_csv, index=False)
    df.to_json(player_json, orient="records", indent=4)

    # --- Team-level aggregates ---
    team_csv = OUTPUT_PATH_DIR / f"team_engineered_features_{SEASON}.csv"
    team_json = OUTPUT_PATH_DIR / f"team_engineered_features_{SEASON}.json"
    team_features.to_csv(team_csv, index=False)
    team_features.to_json(team_json, orient="records", indent=4)

    print(f"💾 Saved engineered player + team datasets to {OUTPUT_PATH_DIR}")
    print(f"   • Player features: {len(df)} rows")
    print(f"   • Team features: {len(team_features)} rows")

    # ===============================================================
    # ✅ Return both for downstream use (embedding, analytics, etc.)
    # ===============================================================
    return df, team_features


if __name__ == "__main__":
    start = datetime.now()
    run_feature_engineering()
    print(f"⏱️ Completed in {(datetime.now() - start).total_seconds():.2f} seconds")

