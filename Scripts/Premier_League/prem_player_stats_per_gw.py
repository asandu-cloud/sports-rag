# Prem players per gw

import requests
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
import time 
from dotenv import load_dotenv
import os

load_dotenv()
API_KEY = os.getenv('API-FOOTBALL-KEY')
LEAGUE_ID = 39   # Premier League
SEASON = 2025
OUTPUT_DIR = Path('/Users/sanduandrei/Desktop/Betting_RAG/Output/Prem_output')
MIN_MINUTES = 1 

# --- GET FIXTURE ID'S ---
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
        fixture_map[fid] = f"{home} vs {away}"

    print(f"✅ Retrieved {len(fixture_map)} fixtures with names.")
    return fixture_map


# --- GET PLAYER STATS PER FIXTURE ---
def fetch_fixture_player_stats(fixture_id, fixture_map):
    url = "https://v3.football.api-sports.io/fixtures/players"
    headers = {"x-apisports-key": API_KEY}
    params = {"fixture": fixture_id}

    r = requests.get(url, headers=headers, params=params)
    data = r.json()
    response = data.get("response", [])
    players = []

    def _to_float(x):
        try:
            if isinstance(x, str):
                x = x.strip().replace("%", "")
            return float(x)
        except (TypeError, ValueError):
            return None

    for team_block in response:
        team = (team_block.get("team") or {}).get("name")
        for player_entry in team_block.get("players", []):

            player = player_entry.get("player", {}) or {}
            stats  = (player_entry.get("statistics") or [{}])[0]
            games  = stats.get("games", {}) or {}
            minutes = games.get("minutes") or 0
            if not minutes or minutes <= MIN_MINUTES:
                continue

            passes = stats.get("passes", {}) or {}
            passes_total = passes.get("total") or 0
            acc_raw = passes.get("accuracy")  # may be count OR percent

            accurate_passes = None
            pass_accuracy_pct = None

            acc_val = _to_float(acc_raw)
            pt = _to_float(passes_total) or 0.0

            if pt > 0 and acc_val is not None:
                if 0 <= acc_val <= pt:
                    # Treat as COUNT of accurate passes
                    accurate_passes = int(round(acc_val))
                    pass_accuracy_pct = round((accurate_passes / pt) * 100.0, 1)
                elif 0 <= acc_val <= 100:
                    # Treat as PERCENT of accurate passes
                    pass_accuracy_pct = round(acc_val, 1)
                    accurate_passes = int(round((pt * acc_val) / 100.0))
                # else: unusable → leave both as None

            row = {
                "fixture": fixture_map.get(fixture_id, f"Fixture {fixture_id}"),
                "team": team,
                "player_id": player.get("id"),
                "name": player.get("name"),
                "position": games.get("position"),
                "minutes": minutes,
                "rating": games.get("rating"),
                "goals": (stats.get("goals", {}) or {}).get("total"),
                "assists": (stats.get("goals", {}) or {}).get("assists"),
                "shots_total": (stats.get("shots", {}) or {}).get("total"),
                "shots_on": (stats.get("shots", {}) or {}).get("on"),
                "passes_total": passes_total,
                # keep exact field names
                "accurate_passes": accurate_passes,
                "pass_accuracy_%": pass_accuracy_pct,
                "tackles": (stats.get("tackles", {}) or {}).get("total"),
                "interceptions": (stats.get("tackles", {}) or {}).get("interceptions"),
                "fouls_committed": (stats.get("fouls", {}) or {}).get("committed"),
                "fouls_drawn": (stats.get("fouls", {}) or {}).get("drawn"),
                "yellow_cards": (stats.get("cards", {}) or {}).get("yellow"),
                "red_cards": (stats.get("cards", {}) or {}).get("red"),
                "duels_won": (stats.get("duels", {}) or {}).get("won"),
                "duels_total": (stats.get("duels", {}) or {}).get("total"),
            }
            players.append(row)

    print(f"Fetched {len(players)} players from fixture {fixture_id}")
    return players

# --- ITERATE THROUGH ALL FIXTURES AND CREATE STATS ---
def fetch_all_fixtures():
    fixture_map = get_fixture_info()
    fixture_ids = list(fixture_map.keys())
    all_players = []

    for i, fid in enumerate(fixture_ids, start=1):
        try:
            fixture_players = fetch_fixture_player_stats(fid, fixture_map)
            all_players.extend(fixture_players)
        except Exception as e:
            print(f"⚠️ Error fetching fixture {fid}: {e}")
        time.sleep(0.5)  # avoid hitting rate limit
        if i % 10 == 0:
            print(f"Progress: {i}/{len(fixture_ids)} fixtures processed")

    return all_players

# --- Cluster player performances per individual ---
def cluster_player_performances(player_fixture_data):
    from collections import defaultdict

    clustered = defaultdict(lambda: {
        "player_id": None,
        "name": None,
        "team": None,
        "position": None,
        "performances": []
    })

    for row in player_fixture_data:
        pid = row.get("player_id")
        if pid is None:
            continue

        if clustered[pid]["player_id"] is None:
            clustered[pid]["player_id"] = pid
            clustered[pid]["name"] = row.get("name")
            clustered[pid]["team"] = row.get("team")
            clustered[pid]["position"] = row.get("position")

        perf = {
            "fixture": row.get("fixture"),
            "minutes": row.get("minutes"),
            "rating": row.get("rating"),
            "goals": row.get("goals"),
            "assists": row.get("assists"),
            "shots_total": row.get("shots_total"),
            "shots_on": row.get("shots_on"),
            "passes_total": row.get("passes_total"),
            "accurate_passes": row.get("accurate_passes"),
            "pass_accuracy_pct": row.get("pass_accuracy_pct"),
            "tackles": row.get("tackles"),
            "interceptions": row.get("interceptions"),
            "fouls_committed": row.get("fouls_committed"),
            "fouls_drawn": row.get("fouls_drawn"),
            "yellow_cards": row.get("yellow_cards"),
            "red_cards": row.get("red_cards"),
            "duels_won": row.get("duels_won"),
            "duels_total": row.get("duels_total")
        }
        clustered[pid]["performances"].append(perf)

    print(f"✅ Clustered performances for {len(clustered)} players.")
    return list(clustered.values())

# --- AGGREGATE PER PLAYER TOTALS WITH FEATURE ENGINEERING --- 
def aggregate_player_totals(player_fixture_data):
    df = pd.DataFrame(player_fixture_data)

    # Step 1 — Aggregate raw statistics
    agg_map = {
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
        "duels_total": "sum"
    }

    totals = (
        df.groupby(["player_id", "name", "team", "position"], dropna=False)
        .agg(agg_map)
        .reset_index()
    )

    # Step 2 — Derived metrics (feature engineering)
    totals["goals_per_90"] = totals.apply(
        lambda x: (x["goals"] / x["minutes"]) * 90 if x["minutes"] > 0 else 0, axis=1
    )
    totals["assists_per_90"] = totals.apply(
        lambda x: (x["assists"] / x["minutes"]) * 90 if x["minutes"] > 0 else 0, axis=1
    )
    totals["shots_accuracy"] = totals.apply(
        lambda x: (x["shots_on"] / x["shots_total"]) if x["shots_total"] > 0 else None, axis=1
    )
    totals["pass_accuracy_pct"] = totals.apply(
        lambda x: (x["accurate_passes"] / x["passes_total"]) * 100 if x["passes_total"] > 0 else None, axis=1
    )
    totals["duel_success_rate"] = totals.apply(
        lambda x: (x["duels_won"] / x["duels_total"]) * 100 if x["duels_total"] > 0 else None, axis=1
    )
    totals["defensive_contrib_per_90"] = totals.apply(
        lambda x: ((x["tackles"] + x["interceptions"]) / x["minutes"]) * 90 if x["minutes"] > 0 else 0, axis=1
    )
    totals["discipline_index"] = totals.apply(
        lambda x: ((x["yellow_cards"] + 2 * x["red_cards"] + 0.25 * x["fouls_committed"]) / x["minutes"]) * 90 if x["minutes"] > 0 else 0,
        axis=1
    )

    # Step 3 — Round selected columns for readability
    numeric_cols = ["goals_per_90", "assists_per_90", "shots_accuracy", "pass_accuracy_pct",
                    "duel_success_rate", "defensive_contrib_per_90", "discipline_index"]
    totals[numeric_cols] = totals[numeric_cols].round(3)

    print(f"✅ Aggregated {len(totals)} players with derived performance features.")
    return df, totals



# --- NEW FUNCTION: save_clustered_performances ---

def save_clustered_performances(clustered_players):
    """
    Saves the clustered per-player match performance data
    (list of fixtures per player) as JSON and CSV.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # JSON output
    json_path = OUTPUT_DIR / f"player_clustered_performances_{SEASON}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(clustered_players, f, indent=4, ensure_ascii=False)

    # Flatten nested 'performances' for CSV export
    csv_rows = []
    for player in clustered_players:
        pid = player.get("player_id")
        name = player.get("name")
        team = player.get("team")
        position = player.get("position")

        for perf in player.get("performances", []):
            csv_rows.append({
                "player_id": pid,
                "name": name,
                "team": team,
                "position": position,
                "fixture": perf.get("fixture"),
                "minutes": perf.get("minutes"),
                "rating": perf.get("rating"),
                "goals": perf.get("goals"),
                "assists": perf.get("assists"),
                "shots_total": perf.get("shots_total"),
                "shots_on": perf.get("shots_on"),
                "passes_total": perf.get("passes_total"),
                "accurate_passes": perf.get("accurate_passes"),
                "pass_accuracy_pct": perf.get("pass_accuracy_pct"),
                "tackles": perf.get("tackles"),
                "interceptions": perf.get("interceptions"),
                "fouls_committed": perf.get("fouls_committed"),
                "fouls_drawn": perf.get("fouls_drawn"),
                "yellow_cards": perf.get("yellow_cards"),
                "red_cards": perf.get("red_cards"),
                "duels_won": perf.get("duels_won"),
                "duels_total": perf.get("duels_total")
            })

    df_csv = pd.DataFrame(csv_rows)
    csv_path = OUTPUT_DIR / f"player_clustered_performances_{SEASON}.csv"
    df_csv.to_csv(csv_path, index=False)

    print(f"💾 Saved clustered performances: {json_path.name} and {csv_path.name} ({len(clustered_players)} players)")

# --- SAVE TO JSON & CSV --- 
def save_outputs(per_fixture_df, total_df):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Per fixture
    json_path1 = OUTPUT_DIR / f"player_fixture_stats_{SEASON}.json"
    csv_path1 = OUTPUT_DIR / f"player_fixture_stats_{SEASON}.csv"
    per_fixture_df.to_json(json_path1, orient="records", indent=4)
    per_fixture_df.to_csv(csv_path1, index=False)

    # Aggregated totals
    json_path2 = OUTPUT_DIR / f"player_total_stats_{SEASON}.json"
    csv_path2 = OUTPUT_DIR / f"player_total_stats_{SEASON}.csv"
    total_df.to_json(json_path2, orient="records", indent=4)
    total_df.to_csv(csv_path2, index=False)

    print(f"💾 Saved {len(per_fixture_df)} fixture entries and {len(total_df)} totals")

# --- MAIN EXECUTION ---
def main():
    start = datetime.now()
    all_players = fetch_all_fixtures()
    per_fixture_df, total_df = aggregate_player_totals(all_players)
    clustered_players = cluster_player_performances(all_players)
    
    # Existing saves
    save_outputs(per_fixture_df, total_df)
    
    # 🆕 New clustered output file
    save_clustered_performances(clustered_players)

    print(f"⏱️ Completed in {(datetime.now() - start).seconds} seconds")

if __name__ == "__main__":
    main()

