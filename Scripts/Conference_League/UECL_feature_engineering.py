# Conference League feature engineering

import argparse
import pandas as pd
from pathlib import Path
from datetime import datetime

LEAGUE = "Conference League"

_parser = argparse.ArgumentParser()
_parser.add_argument("--season", type=int, default=2025,
                     help="API-Football season year (e.g. 2025 for 2025/26)")
SEASON = _parser.parse_args().season

PLAYER_PATH = Path(f"/Users/sanduandrei/Desktop/Betting_RAG/Output/UECL_output/player_fixture_stats_{SEASON}.json")
FIXTURE_PATH = Path(f"/Users/sanduandrei/Desktop/Betting_RAG/Output/UECL_teams/team_fixture_stats_{SEASON}.json")

OUTPUT_PATH = Path("/Users/sanduandrei/Desktop/Betting_RAG/Output/UECL_feature_engineering")
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

# LOADERS
def load_player_data() -> pd.DataFrame:
    df = pd.read_json(PLAYER_PATH)
    print(f"📂 Loaded {len(df)} player-fixture rows from {PLAYER_PATH.name}")
    return df


def load_fixture_stats() -> pd.DataFrame:
    fx = pd.read_json(FIXTURE_PATH)

    # 1. Normalize column names to snake_case lowercase
    fx.columns = (
        fx.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    # Rename API stat keys to canonical names we use downstream
    # This is CRITICAL for corners.
    rename_map = {
        "corner_kicks": "corners",
        "corner_kick": "corners",
        "corners": "corners",  # passthrough if already good

        "ball_possession": "possession",
        "possession_%": "possession",

        "total_shots": "shots_total",
        "shots_total": "shots_total",

        "shots_on_goal": "shots_on",
        "shots_on_goals": "shots_on",
        "shots_on_target": "shots_on",
        "shots_on": "shots_on",

        "total_passes": "passes_total",
        "passes_total": "passes_total",

        "passes_accurate": "accurate_passes",
        "accurate_passes": "accurate_passes",

        "passes_%": "pass_accuracy_rate",

        "fouls": "fouls_committed",
    }
    fx.rename(
        columns={k: v for k, v in rename_map.items() if k in fx.columns},
        inplace=True,
    )

    # make sure every per-team stat column we care about exists and is numeric
    numeric_cols = [
        "shots_total",
        "shots_on",
        "corners",
        "possession",
        "fouls_committed",
        "yellow_cards",
        "red_cards",
        "expected_goals",
        "goals_prevented",
    ]
    for col in numeric_cols:
        if col not in fx.columns:
            fx[col] = 0
        fx[col] = pd.to_numeric(fx[col], errors="coerce").fillna(0)

    # make sure metadata columns  ALSOexist, because we want them in final output
    meta_cols = [
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
    for col in meta_cols:
        if col not in fx.columns:
            # numeric vs string defaults
            if col in ["home_goals", "away_goals"]:
                fx[col] = 0
            else:
                fx[col] = None

    # 5. Some API dumps don't explicitly give "fixture" as "Home vs Away".
    # If fixture is missing or empty, reconstruct it.
    if ("fixture" not in fx.columns) or (fx["fixture"].isna().all()):
        fx["fixture"] = (
            fx["home_team"].fillna("").astype(str).str.strip()
            + " vs "
            + fx["away_team"].fillna("").astype(str).str.strip()
        )

    print(f"📂 Loaded {len(fx)} fixture-level rows from {FIXTURE_PATH.name}")
    return fx

# PLAYER FEATURES
def engineer_player_efficiency(df: pd.DataFrame) -> pd.DataFrame:
    # Avoid div-by-zero during ratios
    for col in ["shots_on", "shots_total", "duels_total", "tackles"]:
        if col in df.columns:
            df[col] = df[col].replace(0, pd.NA)

    df["shots_on_target_ratio"] = df["shots_on"] / df["shots_total"]
    df["goal_conversion_rate"]  = df["goals"] / df["shots_on"]
    df["duel_win_ratio"]        = df["duels_won"] / df["duels_total"]
    df["tackle_success_rate"]   = df["tackles"] / df["duels_total"]

    df["goal_involvement"] = df["goals"].fillna(0) + df["assists"].fillna(0)

    df.fillna(0, inplace=True)
    print("Player efficiency engineered.")
    return df


def engineer_player_aggression(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["minutes", "duels_total", "fouls_committed"]:
        if col in df.columns:
            df[col] = df[col].replace(0, pd.NA)

    df["cards_total"]     = df["yellow_cards"].fillna(0) + df["red_cards"].fillna(0)
    df["cards_per_90"]    = (df["cards_total"] / df["minutes"]) * 90
    df["fouls_per_90"]    = (df["fouls_committed"] / df["minutes"]) * 90
    df["cards_per_foul"]  = df["cards_total"] / df["fouls_committed"]
    df["duel_foul_ratio"] = df["fouls_committed"] / df["duels_total"]

    df.fillna(0, inplace=True)
    print("Player aggression engineered.")
    return df


def engineer_player_form(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(by=["player_id", "fixture"]).copy()
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    g = df.groupby("player_id")

    df["avg_rating_last_4"]  = g["rating"].transform(lambda x: x.rolling(4, 1).mean())
    df["avg_goals_last_4"]   = g["goals"].transform(lambda x: x.rolling(4, 1).mean())
    df["avg_assists_last_4"] = g["assists"].transform(lambda x: x.rolling(4, 1).mean())

    df["goal_involvement_last_4"] = df["avg_goals_last_4"] + df["avg_assists_last_4"]
    df["form_index"] = (
        df["avg_rating_last_4"].fillna(0) * 0.6
        + df["goal_involvement_last_4"].fillna(0) * 0.4
    )

    df["performance_variance"] = g["rating"].transform(lambda x: x.rolling(4, 2).var())

    df.fillna(0, inplace=True)
    print("✅ Player form engineered.")
    return df


# TEAM (AGGREGATED FROM PLAYERS, THEN ENRICHED)
def engineer_team_core_features(player_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate all player rows into per-(fixture, team) rows.
    This gives us:
    - goals, assists
    - shots_total, shots_on
    - passes_total, accurate_passes
    - fouls_committed, yellow_cards, red_cards
    - tackles, interceptions
    - plus computed shots_accuracy_for, pass_accuracy_team
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

    # compute accuracy metrics at team level
    team_fixture["shots_accuracy_for"] = (
        team_fixture["shots_on"] / team_fixture["shots_total"].replace(0, pd.NA)
    )
    team_fixture["pass_accuracy_team"] = (
        team_fixture["accurate_passes"] / team_fixture["passes_total"].replace(0, pd.NA)
    )

    # we DO NOT want minutes in final output
    if "minutes" in team_fixture.columns:
        # keep it only as a helper for now if downstream needs per-90,
        # we'll drop it later right before export
        pass

    team_fixture.fillna(0, inplace=True)
    print("✅ Team core aggregated.")
    return team_fixture


def _norm_merge_str(series: pd.Series) -> pd.Series:
    """
    Normalize strings for merge keys:
    - lowercase
    - collapse multiple spaces
    - strip ends
    """
    return (
        series.astype(str)
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def merge_team_with_fixture(team_df: pd.DataFrame, fixture_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge aggregated player-based team stats (team_df)
    with fixture-level stats + metadata (fixture_df)
    using normalized (fixture, team).

    After this merge, each row will have:
    - shots_total, shots_on, corners, possession, expected_goals, goals_prevented
    - fixture_date, final_score_string, home_team, away_team, etc.
    """

    if fixture_df is None or fixture_df.empty:
        print("⚠️ Fixture stats missing, skipping merge.")
        return team_df

    df = team_df.copy()
    fx = fixture_df.copy()

    # 1. Normalize join keys
    df["fixture_norm"] = _norm_merge_str(df["fixture"])
    df["team_norm"]    = _norm_merge_str(df["team"])

    fx["fixture_norm"] = _norm_merge_str(fx["fixture"])
    fx["team_norm"]    = _norm_merge_str(fx["team"])

    # 2. Make sure fixture_df has all numeric + metadata columns we want to pull in
    # Numeric per-team stats needed downstream
    numeric_cols = [
        "shots_total",
        "shots_on",
        "corners",
        "possession",
        "expected_goals",
        "goals_prevented",
        "fouls_committed",
        "yellow_cards",
        "red_cards",
    ]
    # Metadata we also want merged through for final output
    meta_cols = [
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

    for col in numeric_cols:
        if col not in fx.columns:
            fx[col] = 0
        fx[col] = pd.to_numeric(fx[col], errors="coerce").fillna(0)

    for col in meta_cols:
        if col not in fx.columns:
            if col in ["home_goals", "away_goals"]:
                fx[col] = 0
            else:
                fx[col] = None

    # 3. Merge
    keep_cols = ["fixture_norm", "team_norm"] + numeric_cols + meta_cols
    merged = df.merge(
        fx[keep_cols],
        on=["fixture_norm", "team_norm"],
        how="left"
    )

    # 3b. Resolve duplicate columns (player aggregation vs fixture stats) by preferring fixture stats, fallback to player agg
    for col in numeric_cols:
        cx, cy = f"{col}_x", f"{col}_y"
        if cx in merged.columns or cy in merged.columns:
            merged[col] = merged.get(cy)
            if col in merged and merged[col].isna().any() and cx in merged:
                merged[col] = merged[col].fillna(merged[cx])
    # Drop suffix columns
    merged.drop(columns=[c for c in merged.columns if c.endswith("_x") or c.endswith("_y")], inplace=True, errors="ignore")

    # 4. Clean helper cols
    merged.drop(columns=["fixture_norm", "team_norm"], inplace=True, errors="ignore")

    # 5. Ensure numeric integrity post-merge (safety)
    for col in numeric_cols:
        if col not in merged.columns:
            merged[col] = 0
        merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0)

    # 6. Compute dominance_index now that shots, possession, corners are in place.
    # You were using a weighted blend of possession, shot quality, and corners.
    merged["dominance_index"] = (
        (merged["possession"].fillna(0) * 0.4)
        + (
            (
                merged["shots_on"] /
                merged["shots_total"].replace(0, pd.NA)
            ).fillna(0) * 0.3
        )
        + (
            merged["corners"].fillna(0) /
            (merged["corners"].max() if merged["corners"].max() not in [0, None] else 1)
        ) * 0.3
    )

    merged.fillna(0, inplace=True)
    print("Merged fixture stats + metadata (including corners) into team rows.")
    return merged



def engineer_team_strength_efficiency(team_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute:
    - goal_conversion_rate
    - shot_on_target_ratio
    - control_index
    These must exist in final output.
    """

    df = team_df.copy()

    # Safety: ensure columns exist even if some fixture rows were weird
    for col in ["shots_total", "shots_on", "passes_total", "pass_accuracy_team", "corners"]:
        if col not in df.columns:
            df[col] = 0

    # Goal conversion = goals / shots_total
    df["goal_conversion_rate"] = (
        df["goals"] / df["shots_total"].replace(0, pd.NA)
    )

    # Shot on target ratio = shots_on / shots_total
    df["shot_on_target_ratio"] = (
        df["shots_on"] / df["shots_total"].replace(0, pd.NA)
    )

    # Control index:
    # - pass accuracy (keep ball)/(quality of ball use)
    # - passing volume (tempo/control proxy)
    # - corners (territory/pressure proxy)
    denom_passes = df["passes_total"].max() if "passes_total" in df.columns and df["passes_total"].max() not in [0, None] else 1
    denom_corners = df["corners"].max() if "corners" in df.columns and df["corners"].max() not in [0, None] else 1

    df["control_index"] = (
        (df["pass_accuracy_team"].fillna(0) * 0.5)
        + ((df["passes_total"].fillna(0) / denom_passes) * 0.3)
        + ((df["corners"].fillna(0) / denom_corners) * 0.2)
    )

    df.fillna(0, inplace=True)
    print("Team efficiency/control metrics computed.")
    return df


def engineer_team_aggression_discipline(team_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute fouls/cards metrics and aggression profile.
    Must output fouls_per_90_team, cards_per_90_team,
    cards_per_foul_team, fouls_per_duel_team,
    aggression_index_raw, aggression_index_norm,
    cards_total.
    """

    df = team_df.copy()
    for col in ["fouls_committed", "yellow_cards", "red_cards", "tackles", "interceptions"]:
        if col not in df.columns:
            df[col] = 0

    df["cards_total"] = df["yellow_cards"] + df["red_cards"]

    # Since we dropped minutes for teams (team-level is per 90 ~= per match),
    # fouls_per_90_team is just fouls in that match
    df["fouls_per_90_team"] = df["fouls_committed"]
    df["cards_per_90_team"] = df["cards_total"]

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
        df["aggression_index_norm"] = (
            (df["aggression_index_raw"] - min_val) / (max_val - min_val)
        )
    else:
        df["aggression_index_norm"] = 0

    df.fillna(0, inplace=True)
    print("Team aggression metrics computed.")
    return df


def engineer_team_form_consistency(team_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute rolling form using fixture_date (chronological).
    Must output avg_goals_last_5, avg_cards_last_5, form_index_team.
    """

    # If fixture_date didn't merge for some row, fallback so sort doesn't die
    sort_key = "fixture_date" if "fixture_date" in team_df.columns else "fixture"

    df = team_df.copy().sort_values(by=["team", sort_key]).reset_index(drop=True)

    g = df.groupby("team")
    df["avg_goals_last_5"] = g["goals"].transform(lambda x: x.rolling(5, 1).mean())
    df["avg_cards_last_5"] = g["cards_total"].transform(lambda x: x.rolling(5, 1).mean())

    df["form_index_team"] = (
        df["avg_goals_last_5"].fillna(0) * 0.5
        - df["avg_cards_last_5"].fillna(0) * 0.3
        + df["goal_conversion_rate"].fillna(0) * 0.2
    )

    df.fillna(0, inplace=True)
    print("Team form metrics computed.")
    return df


# =====================================================
# PLAYER CONTEXT FEATURES
# =====================================================
def engineer_player_contextual_features(player_df: pd.DataFrame,
                                        team_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add team-level + opponent-level aggression/control/form
    to each player row.
    """

    df = player_df.copy()
    teams = team_df.copy()

    # Normalize keys
    df["fixture"] = df["fixture"].astype(str).str.lower().str.strip()
    df["team"]    = df["team"].astype(str).str.lower().str.strip()
    teams["fixture"] = teams["fixture"].astype(str).str.lower().str.strip()
    teams["team"]    = teams["team"].astype(str).str.lower().str.strip()

    # Ensure required columns exist before merge
    req_cols = [
        "cards_per_90_team", "fouls_per_90_team",
        "aggression_index_norm", "form_index_team",
        "control_index"
    ]
    for col in req_cols:
        if col not in teams.columns:
            teams[col] = 0

    # Opponent mapping for each fixture (2 teams per match)
    fixture_team_lists = (
        teams.groupby("fixture")["team"].apply(list).reset_index(name="teams")
    )
    fixture_team_lists = fixture_team_lists[fixture_team_lists["teams"].apply(len) == 2]

    opp_lookup = {}
    for _, row in fixture_team_lists.iterrows():
        t1, t2 = row["teams"]
        opp_lookup[(row["fixture"], t1)] = t2
        opp_lookup[(row["fixture"], t2)] = t1

    # Merge self team metrics
    df = df.merge(
        teams[[
            "fixture", "team",
            "aggression_index_norm", "form_index_team", "control_index"
        ]],
        on=["fixture", "team"],
        how="left"
    )

    # Add opponent team name
    df["opponent"] = df.apply(
        lambda x: opp_lookup.get((x["fixture"], x["team"])),
        axis=1
    )

    # Merge opponent metrics
    df = df.merge(
        teams[[
            "fixture", "team",
            "aggression_index_norm",
            "cards_per_90_team", "fouls_per_90_team",
            "form_index_team", "control_index"
        ]],
        left_on=["fixture", "opponent"],
        right_on=["fixture", "team"],
        suffixes=("", "_opp"),
        how="left"
    )

    if "team_opp" in df.columns:
        df.drop(columns=["team_opp"], inplace=True)

    # Make sure opponent-derived cols exist
    backup_cols = [
        "cards_per_90_team_opp",
        "fouls_per_90_team_opp",
        "aggression_index_norm_opp",
        "form_index_team_opp",
        "control_index_opp"
    ]
    for c in backup_cols:
        if c not in df.columns:
            df[c] = 0

    # Derived matchup deltas
    df["agg_diff"]     = df["aggression_index_norm"] - df["aggression_index_norm_opp"]
    df["form_diff"]    = df["form_index_team"]       - df["form_index_team_opp"]
    df["control_diff"] = df["control_index"]         - df["control_index_opp"]

    # Role tagging for foul/card props
    df["is_defensive"] = df["position"].fillna("").str.contains("DF|CB|LB|RB", case=False).astype(int)

    df["expected_foul_pressure"] = df["aggression_index_norm_opp"].fillna(0) * df["is_defensive"]
    df["expected_card_risk"] = (
        df["cards_per_90_team_opp"].fillna(0)
        * df["aggression_index_norm_opp"].fillna(0)
        * df["is_defensive"]
    )

    df.fillna(0, inplace=True)
    print("Player contextual features engineered.")
    return df



# MASTER PIPELINE
def run_feature_engineering():
    # Load
    player_df     = load_player_data()
    fixture_stats = load_fixture_stats()

    # Player-level enrich
    player_df = engineer_player_efficiency(player_df)
    player_df = engineer_player_aggression(player_df)
    player_df = engineer_player_form(player_df)

    # Team-level enrich
    team_df = engineer_team_core_features(player_df)
    team_df = merge_team_with_fixture(team_df, fixture_stats)
    team_df = engineer_team_strength_efficiency(team_df)
    team_df = engineer_team_aggression_discipline(team_df)
    team_df = engineer_team_form_consistency(team_df)

    # Player contextual (team + opponent metrics)
    player_df = engineer_player_contextual_features(player_df, team_df)

    # FINAL CLEANUPS BEFORE EXPORT

    # 1. Drop minutes from team_df (you said it's irrelevant in final output)
    if "minutes" in team_df.columns:
        team_df.drop(columns=["minutes"], inplace=True)

    # 2. Ensure all columns you expect exist even if NaN in some matches
    expected_final_cols = [
        "fixture", "team",
        "goals", "assists",
        "passes_total", "accurate_passes",
        "fouls_committed", "yellow_cards", "red_cards",
        "tackles", "interceptions",
        "shots_total", "shots_on",
        "shots_accuracy_for", "pass_accuracy_team",
        "fixture_id", "fixture_date", "fixture_date_utc",
        "final_score", "final_score_string",
        "home_team", "away_team",
        "home_goals", "away_goals",
        "corners",
        "goal_conversion_rate", "shot_on_target_ratio", "control_index",
        "cards_total", "fouls_per_90_team", "cards_per_90_team",
        "cards_per_foul_team", "fouls_per_duel_team",
        "aggression_index_raw", "aggression_index_norm",
        "avg_goals_last_5", "avg_cards_last_5", "form_index_team",
    ]

    for col in expected_final_cols:
        if col not in team_df.columns:
            team_df[col] = 0 if col not in [
                "fixture", "team",
                "fixture_id", "fixture_date", "fixture_date_utc",
                "final_score", "final_score_string",
                "home_team", "away_team",
            ] else None

    # SAVE
    player_out = OUTPUT_PATH / f"player_engineered_features_{SEASON}.json"
    team_out   = OUTPUT_PATH / f"team_engineered_features_{SEASON}.json"

    player_df.to_json(player_out, orient="records", indent=4)
    team_df.to_json(team_out, orient="records", indent=4)

    print(f"Saved engineered datasets → {OUTPUT_PATH}")
    print(f"   • Player rows: {len(player_df)}")
    print(f"   • Team rows: {len(team_df)}")


if __name__ == "__main__":
    start = datetime.now()
    run_feature_engineering()
    print(f"Completed in {(datetime.now() - start).total_seconds():.2f} seconds")
