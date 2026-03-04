# SeriaA team stats 

import argparse
import requests
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
import time
from dotenv import load_dotenv
import os

# --- CONFIGURATION ---
load_dotenv()
API_KEY = os.getenv('API-FOOTBALL-KEY')
LEAGUE_ID = 135  # Serie A

_parser = argparse.ArgumentParser()
_parser.add_argument("--season", type=int, default=2025,
                     help="API-Football season year (e.g. 2025 for 2025/26)")
SEASON = _parser.parse_args().season
OUTPUT_DIR = Path('/Users/sanduandrei/Desktop/Betting_RAG/Output/SeriaA_Output')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Fetch all fixture IDs and metadata ---
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
            "fixture_id": fid,
            "fixture_name": f"{home} vs {away}",
            "home_team": home,
            "away_team": away,
            "home_goals": home_goals,
            "away_goals": away_goals,
            "final_score": f"{home_goals}-{away_goals}",
            "final_score_string": f"{home} {home_goals} - {away_goals} {away}",
            "fixture_date_utc": f["fixture"]["date"],
            "fixture_date": f["fixture"]["date"][:10],
        }

    print(f"✅ Retrieved {len(fixture_map)} fixtures with names, scores, and dates.")
    return fixture_map


# --- Fetch per-fixture team stats ---
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

        # ✅ Start with fixture context (score, teams, date, etc.)
        team_stats = {
            "fixture_id": fixture_info.get("fixture_id"),
            "fixture": fixture_info.get("fixture_name"),
            "fixture_date_utc": fixture_info.get("fixture_date_utc"),
            "fixture_date": fixture_info.get("fixture_date"),
            "team": team_name,
            "home_team": fixture_info.get("home_team"),
            "away_team": fixture_info.get("away_team"),
            "home_goals": fixture_info.get("home_goals"),
            "away_goals": fixture_info.get("away_goals"),
            "final_score": fixture_info.get("final_score"),
            "final_score_string": fixture_info.get("final_score_string"),
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


# --- Iterate through all fixtures ---
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


# --- Aggregate per-team averages ---
def aggregate_team_stats(team_fixture_data):
    import pandas as pd

    df = pd.DataFrame(team_fixture_data)

    # --- Convert numeric-like columns ---
    for col in df.columns:
        if col not in ["fixture", "team", "home_team", "away_team", "final_score"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # --- 1️⃣ Compute team "for" averages ---
    team_for = (
        df.groupby("team", dropna=False)
        .mean(numeric_only=True)
        .reset_index()
        .rename(columns=lambda c: f"{c}_for" if c not in ["team"] else c)
    )

    # --- 2️⃣ Compute team "against" averages ---
    paired = (
        df.merge(df, on="fixture", suffixes=("_team", "_opp"))
        .query("team_team != team_opp")
        .reset_index(drop=True)
    )

    # Keep only opponent stats and who they played against
    opp_cols = [c for c in paired.columns if c.endswith("_opp")]
    opp_stats = paired[["team_team"] + opp_cols].copy()

    # Rename cleanly
    opp_stats.columns = ["team"] + [c.replace("_opp", "") for c in opp_cols]

    # --- 💡 Ensure 'team' is the only team column and is 1D ---
    opp_stats = opp_stats.loc[:, ~opp_stats.columns.duplicated()]
    opp_stats["team"] = opp_stats["team"].astype(str)

    # Drop any potential multi-index issues
    opp_stats.columns = opp_stats.columns.get_level_values(0)

    # --- ✅ Group safely ---
    team_against = (
        opp_stats.groupby("team", dropna=False)
        .mean(numeric_only=True)
        .reset_index()
        .rename(columns=lambda c: f"{c}_against" if c not in ["team"] else c)
    )

    # --- 3️⃣ Merge both sets of averages ---
    merged = team_for.merge(team_against, on="team", how="left")

    # --- 4️⃣ Round numeric columns for readability ---
    numeric_cols = merged.select_dtypes(include=["float", "int"]).columns
    merged[numeric_cols] = merged[numeric_cols].round(1)

    # --- 5️⃣ Sort alphabetically ---
    merged = merged.sort_values(by="team").reset_index(drop=True)

    return df, merged



# --- Save outputs ---
def save_team_outputs(per_fixture_df, aggregate_df):
    json_path1 = OUTPUT_DIR / f"SeriaA_team_fixture_stats_{SEASON}.json"
    csv_path1 = OUTPUT_DIR / f"SeriaA_team_fixture_stats_{SEASON}.csv"
    per_fixture_df.to_json(json_path1, orient="records", indent=4)
    per_fixture_df.to_csv(csv_path1, index=False)

    json_path2 = OUTPUT_DIR / f"SeriaA_team_aggregate_stats_{SEASON}.json"
    csv_path2 = OUTPUT_DIR / f"SeriaA_team_aggregate_stats_{SEASON}.csv"
    aggregate_df.to_json(json_path2, orient="records", indent=4)
    aggregate_df.to_csv(csv_path2, index=False)

    print(f"💾 Saved {len(per_fixture_df)} fixture team entries and {len(aggregate_df)} team aggregates")


# --- Main execution ---
def main():
    start = datetime.now()
    all_team_stats = fetch_all_team_fixtures()
    per_fixture_df, aggregate_df = aggregate_team_stats(all_team_stats)
    save_team_outputs(per_fixture_df, aggregate_df)
    print(f"⏱️ Completed in {(datetime.now() - start).seconds} seconds")

if __name__ == "__main__":
    main()