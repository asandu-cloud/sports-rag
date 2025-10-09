# Statistics for teams

import requests
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
import time

# --- CONFIGURATION ---
API_KEY = "63dbcf42afa65f2e2769daee817f6e48"
LEAGUE_ID = 39  # Premier League
SEASON = 2025
OUTPUT_DIR = Path('/Users/sanduandrei/Desktop/Betting_RAG/Output/Prem_teams')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --- STEP 1: Fetch all fixture IDs and metadata ---
def get_fixture_info():
    url = "https://v3.football.api-sports.io/fixtures"
    headers = {"x-apisports-key": API_KEY}
    params = {"league": LEAGUE_ID, "season": SEASON, "status": "FT"}

    response = requests.get(url, headers=headers, params=params)
    data = response.json()
    fixture_map = {}

    for f in data.get("response", []):
        fid = f["fixture"]["id"]
        home = f["teams"]["home"]["name"]
        away = f["teams"]["away"]["name"]

        # ✅ Extract the final score
        home_goals = f["goals"]["home"]
        away_goals = f["goals"]["away"]

        fixture_map[fid] = {
            "fixture_name": f"{home} vs {away}",
            "home_team": home,
            "away_team": away,
            "home_goals": home_goals,
            "away_goals": away_goals,
            "final_score": f"{home_goals}-{away_goals}"
        }

    print(f"✅ Retrieved {len(fixture_map)} fixtures with names and scores.")
    return fixture_map


# --- STEP 2: Fetch per-fixture team stats ---
def fetch_fixture_team_stats(fixture_id, fixture_info):
    url = "https://v3.football.api-sports.io/fixtures/statistics"
    headers = {"x-apisports-key": API_KEY}
    params = {"fixture": fixture_id}

    response = requests.get(url, headers=headers, params=params)
    data = response.json()

    teams_data = []
    for entry in data.get("response", []):
        team_name = entry.get("team", {}).get("name")
        stats_list = entry.get("statistics", [])

        # ✅ Start with fixture context (score, teams, etc.)
        team_stats = {
            "fixture": fixture_info.get("fixture_name"),
            "team": team_name,
            "home_team": fixture_info.get("home_team"),
            "away_team": fixture_info.get("away_team"),
            "home_goals": fixture_info.get("home_goals"),
            "away_goals": fixture_info.get("away_goals"),
            "final_score": fixture_info.get("final_score"),
        }

        for stat in stats_list:
            key = stat.get("type")
            value = stat.get("value")

            # Convert percentages or None types
            if isinstance(value, str) and "%" in value:
                value = float(value.replace("%", "")) / 100
            elif value is None:
                value = 0

            team_stats[key] = value

        teams_data.append(team_stats)

    if not teams_data:
        print(f"⚠️ No stats found for fixture {fixture_id}")
    return teams_data


# --- STEP 3: Iterate through all fixtures ---
def fetch_all_team_fixtures():
    fixture_map = get_fixture_info()
    fixture_ids = list(fixture_map.keys())
    all_team_stats = []

    for i, fid in enumerate(fixture_ids, start=1):
        fixture_info = fixture_map[fid]
        try:
            team_stats = fetch_fixture_team_stats(fid, fixture_info)
            all_team_stats.extend(team_stats)
        except Exception as e:
            print(f"⚠️ Error fetching fixture {fid}: {e}")
        time.sleep(0.5)
        if i % 10 == 0:
            print(f"Progress: {i}/{len(fixture_ids)} fixtures processed")

    return all_team_stats


# --- STEP 4: Aggregate per-team averages ---
def aggregate_team_stats(team_fixture_data):
    df = pd.DataFrame(team_fixture_data)

    # Convert numeric-like columns
    for col in df.columns:
        if col not in ["fixture", "team", "home_team", "away_team", "final_score"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Compute per-team averages
    team_avg = df.groupby("team").mean(numeric_only=True).reset_index()

    # Opponent averages: what happens *against* each team
    opponent_avg = (
        df.merge(df, on="fixture")
        .query("team_x != team_y")
        .groupby("team_y")
        .mean(numeric_only=True)
        .reset_index()
        .rename(columns={"team_y": "team"})
    )

    # Merge both averages
    merged = team_avg.merge(
        opponent_avg,
        on="team",
        suffixes=("_for", "_against"),
        how="left"
    )

    return df, merged

# --- STEP 5: Save outputs ---
def save_team_outputs(per_fixture_df, aggregate_df):
    json_path1 = OUTPUT_DIR / f"team_fixture_stats_{SEASON}.json"
    csv_path1 = OUTPUT_DIR / f"team_fixture_stats_{SEASON}.csv"
    per_fixture_df.to_json(json_path1, orient="records", indent=4)
    per_fixture_df.to_csv(csv_path1, index=False)

    json_path2 = OUTPUT_DIR / f"team_aggregate_stats_{SEASON}.json"
    csv_path2 = OUTPUT_DIR / f"team_aggregate_stats_{SEASON}.csv"
    aggregate_df.to_json(json_path2, orient="records", indent=4)
    aggregate_df.to_csv(csv_path2, index=False)

    print(f"💾 Saved {len(per_fixture_df)} fixture team entries and {len(aggregate_df)} team aggregates")


# --- STEP 6: Main execution ---
def main():
    start = datetime.now()
    all_team_stats = fetch_all_team_fixtures()
    per_fixture_df, aggregate_df = aggregate_team_stats(all_team_stats)
    save_team_outputs(per_fixture_df, aggregate_df)
    print(f"⏱️ Completed in {(datetime.now() - start).seconds} seconds")


if __name__ == "__main__":
    main()