# Premier League feature engineering

import pandas as pd
from pathlib import Path
from datetime import datetime

LEAGUE = "Premier League"
SEASON = 2025

# Input data
PLAYER_PATH = Path("/Users/sanduandrei/Desktop/Betting_RAG/Output/Prem_output/player_fixture_stats_2025.json")
FIXTURE_PATH = Path("/Users/sanduandrei/Desktop/Betting_RAG/Output/Prem_teams/team_fixture_stats_2025.json")

# Output dir
OUTPUT_PATH = Path("/Users/sanduandrei/Desktop/Betting_RAG/Output/Prem_feature_engineering")
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)


# ---------------------------
# LOADERS
# ---------------------------
def load_player_data() -> pd.DataFrame:
    df = pd.read_json(PLAYER_PATH)
    print(f"📂 Loaded {len(df)} player-fixture rows from {PLAYER_PATH.name}")
    return df


def load_fixture_stats() -> pd.DataFrame:
    fx = pd.read_json(FIXTURE_PATH)

    # Ensure metadata columns from PL_stats_for_teams.py exist
    required_meta_cols = [
        "fixture",
        "team",
        "fixture_id",
        "fixture_date",
        "fixture_date_utc",
        "final_score",
        "final_score_string",
        "home_team",
        "away_team",
        "home_goals",
        "away_goals",
    ]
    for col in required_meta_cols:
        if col not in fx.columns:
            fx[col] = None

    print(f"📂 Loaded {len(fx)} fixture-level rows from {FIXTURE_PATH.name}")
    return fx


# PLAYER FEATURES
# ---------------------------
def engineer_player_efficiency(df: pd.DataFrame) -> pd.DataFrame:
    # Avoid divide-by-zero explosions by temporarily NA-ing 0 denominators
    for col in ["shots_on", "shots_total", "duels_total", "tackles"]:
        if col in df.columns:
            df[col] = df[col].replace(0, pd.NA)

    df["shots_on_target_ratio"] = df["shots_on"] / df["shots_total"]
    df["goal_conversion_rate"]  = df["goals"] / df["shots_on"]
    df["duel_win_ratio"]        = df["duels_won"] / df["duels_total"]
    df["tackle_success_rate"]   = df["tackles"] / df["duels_total"]

    df["goal_involvement"] = df["goals"].fillna(0) + df["assists"].fillna(0)

    df.replace([float("inf"), -float("inf")], pd.NA, inplace=True)
    df.fillna(0, inplace=True)
    print("Engineered player-level efficiency features (1.1)")
    return df


def engineer_player_aggression(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["minutes", "duels_total", "fouls_committed"]:
        if col in df.columns:
            df[col] = df[col].replace(0, pd.NA)

    df["cards_total"]    = df["yellow_cards"].fillna(0) + df["red_cards"].fillna(0)
    df["cards_per_90"]   = (df["cards_total"] / df["minutes"]) * 90
    df["fouls_per_90"]   = (df["fouls_committed"] / df["minutes"]) * 90
    df["cards_per_foul"] = df["cards_total"] / df["fouls_committed"]
    df["duel_foul_ratio"] = df["fouls_committed"] / df["duels_total"]

    df.replace([float("inf"), -float("inf")], pd.NA, inplace=True)
    df.fillna(0, inplace=True)
    print("✅ Engineered player aggression & disciplinary features (1.2)")
    return df


def engineer_player_form(df: pd.DataFrame) -> pd.DataFrame:
    # Sort per player for rolling stats
    df = df.sort_values(by=["player_id", "fixture"]).copy()
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")

    group = df.groupby("player_id")

    df["avg_rating_last_4"]  = group["rating"].transform(lambda x: x.rolling(4, 1).mean())
    df["avg_goals_last_4"]   = group["goals"].transform(lambda x: x.rolling(4, 1).mean())
    df["avg_assists_last_4"] = group["assists"].transform(lambda x: x.rolling(4, 1).mean())

    df["goal_involvement_last_4"] = df["avg_goals_last_4"] + df["avg_assists_last_4"]

    # form_index ~ blend of rating & direct attacking contrib
    df["form_index"] = (
        df["avg_rating_last_4"].fillna(0) * 0.6
        + df["goal_involvement_last_4"].fillna(0) * 0.4
    )

    # Stability / volatility
    df["performance_variance"] = group["rating"].transform(lambda x: x.rolling(4, 2).var())

    df.fillna(0, inplace=True)
    print("Engineered player form & consistency features (1.3)")
    return df


# ---------------------------
# TEAM FEATURES (AGGREGATE FROM PLAYER DATA)
# ---------------------------
def engineer_team_core_features(player_df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse all player rows per (fixture, team)
    into a single team performance row.
    """
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

    team_fixture = (
        player_df.groupby(["fixture", "team"], dropna=False)
                 .agg(agg_map)
                 .reset_index()
    )

    # shooting and passing quality
    team_fixture["shots_accuracy_for"] = (
        team_fixture["shots_on"] / team_fixture["shots_total"].replace(0, pd.NA)
    )
    team_fixture["pass_accuracy_team"] = (
        team_fixture["accurate_passes"] / team_fixture["passes_total"].replace(0, pd.NA)
    )

    team_fixture.fillna(0, inplace=True)
    print(f"✅ Engineered team-level core aggregates (2.1): {len(team_fixture)} rows")
    return team_fixture


def merge_team_with_fixture(team_df: pd.DataFrame, fixture_df: pd.DataFrame) -> pd.DataFrame:
    """
    Attach match metadata from fixture_df to each (fixture, team)
    row in team_df. We ONLY pull the fields we care about for RAG:
      - fixture_id
      - fixture_date
      - fixture_date_utc
      - final_score_string
      - home_team / away_team / home_goals / away_goals
    We do NOT attempt to merge extra numeric stats here. That avoids KeyErrors.
    """
    if fixture_df is None or fixture_df.empty:
        print("⚠️ Fixture stats missing, skipping merge.")
        return team_df

    df = team_df.copy()
    fx = fixture_df.copy()

    # Normalize join keys
    df["fixture_join"] = df["fixture"].astype(str).str.lower().str.strip()
    df["team_join"]    = df["team"].astype(str).str.lower().str.strip()

    fx["fixture_join"] = fx["fixture"].astype(str).str.lower().str.strip()
    fx["team_join"]    = fx["team"].astype(str).str.lower().str.strip()

    carry_cols = [
        "fixture_join",
        "team_join",
        "fixture_id",
        "fixture_date",
        "fixture_date_utc",
        "final_score",
        "final_score_string",
        "home_team",
        "away_team",
        "home_goals",
        "away_goals",
    ]
    for c in carry_cols:
        if c not in fx.columns:
            fx[c] = None

    merged = df.merge(
        fx[carry_cols],
        on=["fixture_join", "team_join"],
        how="left"
    )

    merged.drop(columns=["fixture_join", "team_join"], inplace=True, errors="ignore")

    print("✅ Merged fixture date + final score metadata into team features.")
    return merged


def engineer_team_strength_efficiency(team_df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds:
    - goal_conversion_rate
    - shot_on_target_ratio
    - control_index (possession/territory proxy)
    Guarded so missing columns won't kill us.
    """
    df = team_df.copy()

    # make sure the columns we rely on exist
    needed_cols = [
        "shots_total",
        "shots_on",
        "passes_total",
        "pass_accuracy_team",
        "corners",
        "tackles",
        "fouls_committed",
    ]
    for col in needed_cols:
        if col not in df.columns:
            df[col] = 0

    # avoid div-by-zero for rate stats
    df["shots_total"]       = df["shots_total"].replace(0, pd.NA)
    df["passes_total"]      = df["passes_total"].replace(0, pd.NA)
    df["tackles"]           = df["tackles"].replace(0, pd.NA)
    df["fouls_committed"]   = df["fouls_committed"].replace(0, pd.NA)

    df["goal_conversion_rate"] = df["goals"]    / df["shots_total"]
    df["shot_on_target_ratio"] = df["shots_on"] / df["shots_total"]

    # control_index:
    # - passing accuracy (keep ball)
    # - volume of passes (tempo/ball retention)
    # - corners (territory/pressure proxy)
    passes_norm = (
        df["passes_total"].fillna(0) / df["passes_total"].fillna(0).max()
        if df["passes_total"].notna().any() else 0
    )
    corners_norm = (
        df["corners"].fillna(0) / df["corners"].fillna(0).max()
        if "corners" in df.columns and df["corners"].notna().any() else 0
    )

    df["control_index"] = (
        (df["pass_accuracy_team"].fillna(0) * 0.5)
        + (passes_norm * 0.3)
        + (corners_norm * 0.2)
    )

    df.fillna(0, inplace=True)
    print(f"✅ Engineered team strength & efficiency metrics (2.2): {len(df)} rows")
    return df


def engineer_team_aggression_discipline(team_df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds fouls/cards rates, aggression_index_norm.
    """
    df = team_df.copy()

    for col in ["fouls_committed", "yellow_cards", "red_cards", "tackles", "interceptions"]:
        if col not in df.columns:
            df[col] = 0

    df["cards_total"]        = df["yellow_cards"] + df["red_cards"]
    df["fouls_per_90_team"]  = df["fouls_committed"].fillna(0)  # per match basis
    df["cards_per_90_team"]  = df["cards_total"].fillna(0)

    df["cards_per_foul_team"] = df["cards_total"] / df["fouls_committed"].replace(0, pd.NA)
    df["fouls_per_duel_team"] = df["fouls_committed"] / (
        df["tackles"] + df["interceptions"]
    ).replace(0, pd.NA)

    df["aggression_index_raw"] = (
        df["fouls_per_90_team"].fillna(0) * 0.5
        + df["cards_per_90_team"].fillna(0) * 0.3
        + df["fouls_per_duel_team"].fillna(0) * 0.2
    )

    min_val = df["aggression_index_raw"].min()
    max_val = df["aggression_index_raw"].max()
    if max_val > min_val:
        df["aggression_index_norm"] = (df["aggression_index_raw"] - min_val) / (max_val - min_val)
    else:
        df["aggression_index_norm"] = 0

    df.fillna(0, inplace=True)
    print(f"✅ Engineered team aggression & discipline metrics (2.3): {len(df)} rows")
    return df


def engineer_team_form_consistency(team_df: pd.DataFrame) -> pd.DataFrame:
    """
    Rolling performance form per team.
    We will sort by fixture_date if available (more reliable than fixture string),
    otherwise fall back to fixture text.
    """
    sort_key = "fixture_date" if "fixture_date" in team_df.columns else "fixture"

    df = team_df.copy().sort_values(by=["team", sort_key]).reset_index(drop=True)
    group = df.groupby("team")

    w = 5  # rolling window (5 matches)
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


# ---------------------------
# PLAYER CONTEXT FEATURES
# ---------------------------
def engineer_player_contextual_features(player_df: pd.DataFrame,
                                        team_df: pd.DataFrame) -> pd.DataFrame:
    """
    Joins player rows with team-level aggression/form/control,
    and opponent aggression/form/control, so players carry matchup context.
    """
    df = player_df.copy()
    teams = team_df.copy()

    # normalize join keys
    df["fixture"]  = df["fixture"].astype(str).str.lower().str.strip()
    df["team"]     = df["team"].astype(str).str.lower().str.strip()
    teams["fixture"] = teams["fixture"].astype(str).str.lower().str.strip()
    teams["team"]    = teams["team"].astype(str).str.lower().str.strip()

    # ensure columns exist in teams
    needed_cols = [
        "cards_per_90_team", "fouls_per_90_team",
        "aggression_index_norm", "form_index_team", "control_index"
    ]
    for col in needed_cols:
        if col not in teams.columns:
            teams[col] = 0

    # build opponent map for each fixture
    fixture_teams = teams.groupby("fixture")["team"].apply(list).reset_index(name="teams")
    fixture_teams = fixture_teams[fixture_teams["teams"].apply(len) == 2]

    opp_map = {}
    for _, row in fixture_teams.iterrows():
        t1, t2 = row["teams"]
        opp_map[(row["fixture"], t1)] = t2
        opp_map[(row["fixture"], t2)] = t1

    # merge team metrics for player's own team
    df = df.merge(
        teams[["fixture", "team", "aggression_index_norm",
               "form_index_team", "control_index"]],
        on=["fixture", "team"],
        how="left"
    )

    # map opponent name onto each player row
    df["opponent"] = df.apply(
        lambda x: opp_map.get((x["fixture"], x["team"])),
        axis=1
    )

    # merge opponent metrics
    df = df.merge(
        teams[["fixture", "team",
               "aggression_index_norm", "cards_per_90_team",
               "fouls_per_90_team", "form_index_team", "control_index"]],
        left_on=["fixture", "opponent"],
        right_on=["fixture", "team"],
        how="left",
        suffixes=("", "_opp")
    )

    # clean duplicate merge key
    if "team_opp" in df.columns:
        df.drop(columns=["team_opp"], inplace=True)

    # make sure opponent-derived columns exist
    ensure_cols = [
        "cards_per_90_team_opp", "fouls_per_90_team_opp",
        "aggression_index_norm_opp", "form_index_team_opp",
        "control_index_opp"
    ]
    for col in ensure_cols:
        if col not in df.columns:
            df[col] = 0

    # contextual deltas
    df["agg_diff"]     = df["aggression_index_norm"] - df["aggression_index_norm_opp"]
    df["form_diff"]    = df["form_index_team"]       - df["form_index_team_opp"]
    df["control_diff"] = df["control_index"]         - df["control_index_opp"]

    # role tagging
    df["is_defensive"] = df["position"].fillna("").str.contains("DF|CB|LB|RB", case=False).astype(int)

    # foul/card pressure estimates
    df["expected_foul_pressure"] = df["aggression_index_norm_opp"].fillna(0) * df["is_defensive"]
    df["expected_card_risk"] = (
        df["cards_per_90_team_opp"].fillna(0)
        * df["aggression_index_norm_opp"].fillna(0)
        * df["is_defensive"]
    )

    df.fillna(0, inplace=True)
    print(f"✅ Engineered contextual opponent-based player features (1.4): {len(df)} rows")
    return df


# ---------------------------
# MASTER PIPELINE
# ---------------------------
def run_feature_engineering():
    # 1. Load raw inputs
    player_df     = load_player_data()
    fixture_stats = load_fixture_stats()

    # 2. Player features
    player_df = engineer_player_efficiency(player_df)
    player_df = engineer_player_aggression(player_df)
    player_df = engineer_player_form(player_df)

    # 3. Team features (per fixture, per team)
    team_df = engineer_team_core_features(player_df)
    team_df = merge_team_with_fixture(team_df, fixture_stats)
    team_df = engineer_team_strength_efficiency(team_df)
    team_df = engineer_team_aggression_discipline(team_df)
    team_df = engineer_team_form_consistency(team_df)

    # 4. Player contextual features
    player_df = engineer_player_contextual_features(player_df, team_df)

    # 5. FINAL OUTPUT CLEANUP
    #    - Rename shots_total_y -> shots_total_team
    #    - Rename shots_on_y   -> shots_on_team
    #    - Drop shots_total_x, shots_on_x, shots_total, shots_on
    rename_map = {
        "shots_total_y": "shots_total_team",
        "shots_on_y":   "shots_on_team",
    }
    for old_col, new_col in rename_map.items():
        if old_col in team_df.columns:
            team_df.rename(columns={old_col: new_col}, inplace=True)

    drop_cols = [
        "shots_total_x",
        "shots_on_x",
        "shots_total",
        "shots_on",
    ]
    for col in drop_cols:
        if col in team_df.columns:
            team_df.drop(columns=[col], inplace=True)

    # NOTE: We are explicitly KEEPING:
    # - fixture_date, fixture_date_utc
    # - final_score_string
    # - fixture_id, home_team, away_team, home_goals, away_goals
    # These will then become part of team_engineered_features_2025.json,
    # and will be picked up by 00_normalize_to_docs.py.

    # 6. Save
    player_out_csv = OUTPUT_PATH / f"player_engineered_features_{SEASON}.csv"
    player_out_json = OUTPUT_PATH / f"player_engineered_features_{SEASON}.json"
    team_out_csv = OUTPUT_PATH / f"team_engineered_features_{SEASON}.csv"
    team_out_json = OUTPUT_PATH / f"team_engineered_features_{SEASON}.json"

    player_df.to_csv(player_out_csv, index=False)
    player_df.to_json(player_out_json, orient="records", indent=4)

    team_df.to_csv(team_out_csv, index=False)
    team_df.to_json(team_out_json, orient="records", indent=4)

    print(f"💾 Saved engineered datasets → {OUTPUT_PATH}")
    print(f"   • Player rows: {len(player_df)}")
    print(f"   • Team rows: {len(team_df)}")

    return player_df, team_df


# ---------------------------
# MAIN
# ---------------------------
if __name__ == "__main__":
    start = datetime.now()
    run_feature_engineering()
    print(f"⏱️ Completed in {(datetime.now() - start).total_seconds():.2f} seconds")