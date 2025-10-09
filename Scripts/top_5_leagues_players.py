# Script for creating the JSON Files

import requests
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
import os
import time

# -- CONFIGURATION --

load_dotenv()
API_KEY = '63dbcf42afa65f2e2769daee817f6e48'
OUTPUT_directory = Path('/Users/sanduandrei/Desktop/Betting_RAG/Output/Prem_output')
SEASON = 2025
min_minutes = 1

LEAGUES = {"Premier_League": 39}

# -- FUNCTIONS --

# Player stats for specific league & season
def fetch_player_stats(league_id: int, season: int):
    url = 'https://v3.football.api-sports.io/players'
    headers = {'x-apisports-key': API_KEY}
    players = []
    page = 1
    saved = {}
    total_seen = 0 

    while True:
        params = {'league': league_id, 'season': season, 'page': page}
        response = requests.get(url, headers=headers, params=params)
        data = response.json()

        resp_list = data.get('response', [])
        errors = data.get('errors')
        paging = data.get('paging', {}) or {}
        cur = paging.get('current', page)
        total_pages = paging.get('total', page)


        print(f"Page {page} | API errors: {errors} | players returned: {len(resp_list)} | paging {cur}/{total_pages}")

        
        # Debug output
        print(f"League {league_id} | Page {page} | Errors: {data.get('errors')} | Results: {len(data.get('response', []))}")

        if not resp_list:
            # Could be rate limit or end; if rate limited, sleep and retry once
            if errors and ("requests" in errors or "Too Many" in str(errors)):
                time.sleep(2)
                continue
            break

        for entry in resp_list:
            player = entry.get("player", {}) or {}
            stats_list = entry.get("statistics", []) or []

            # Sum minutes across all statistical entries (handles mid-season transfers)
            total_minutes = 0
            best_stat = None
            best_appear = -1

            for st in stats_list:
                g = st.get("games", {}) or {}
                m = g.get("minutes") or 0
                total_minutes += m

                # Keep the team block with max appearances as "representative"
                ap = g.get("appearences") or 0
                if ap > best_appear:
                    best_appear = ap
                    best_stat = st

            if total_minutes > min_minutes:
                # Choose fields from best_stat (most appearances), but minutes is total
                team_name = (best_stat or {}).get("team", {}) or {}
                league_name = (best_stat or {}).get("league", {}) or {}
                games = (best_stat or {}).get("games", {}) or {}
                cards = (best_stat or {}).get("cards", {}) or {}
                fouls = (best_stat or {}).get("fouls", {}) or {}
                tackles = (best_stat or {}).get("tackles", {}) or {}
                duels = (best_stat or {}).get("duels", {}) or {}
                goals = (best_stat or {}).get("goals", {}) or {}
                shots = (best_stat or {}).get("shots", {}) or {}
                passes = (best_stat or {}).get("passes", {}) or {}

                row = {
                    "player_id": player.get("id"),
                    "name": player.get("name"),
                    "firstname": player.get("firstname"),
                    "lastname": player.get("lastname"),
                    "age": player.get("age"),
                    "nationality": player.get("nationality"),
                    "team": team_name.get("name"),
                    "league": league_name.get("name"),
                    "position": games.get("position"),
                    "appearances": games.get("appearences"),
                    "league_starts": games.get("lineups"),
                    "minutes": int(total_minutes),
                    "rating": games.get("rating"),
                    "yellow_cards": cards.get("yellow"),
                    "red_cards": cards.get("red"),
                    "fouls_committed": fouls.get("committed"),
                    "fouls_drawn": fouls.get("drawn"),
                    "tackles": tackles.get("total"),
                    "interceptions": tackles.get("interceptions"),
                    "duels_won": duels.get("won"),
                    "duels_total": duels.get("total"),
                    "goals": goals.get("total"),
                    "assists": goals.get("assists"),
                    "shots_total": shots.get("total"),
                    "shots_on": shots.get("on"),
                    "passes_total": passes.get("total"),
                    "passes_accuracy": passes.get("accuracy"),
                }

                pid = row["player_id"]
                # Deduplicate by player_id, keep the row with higher minutes
                if pid is not None:
                    if pid not in saved or (row["minutes"] or 0) > (saved[pid]["minutes"] or 0):
                        saved[pid] = row

            total_seen += 1

        # move to next page using API's paging info
        if cur >= total_pages:
            break
        page += 1

        # Small pause helps avoid hitting per-second limits on lower plans
        time.sleep(0.1)

    print(f"Total API entries seen: {total_seen} | players saved (minutes>{min_minutes}): {len(saved)}")
    return list(saved.values())


# Save data as CSV and JSON
def save_files(players, league_name):
    OUTPUT_directory.mkdir(parents=True, exist_ok=True)

    # JSON
    json_path = OUTPUT_directory / f'{league_name}_players_{SEASON}.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(players, f, indent=4, ensure_ascii=False)

    df = pd.DataFrame(players)
    csv_path = OUTPUT_directory / f'{league_name}_players_{SEASON}.csv'
    df.to_csv(csv_path, index=False)

    print(f"💾 Saved {len(players)} players: {json_path.name} and {csv_path.name}")


def main():
    start = datetime.now()
    for league_name, league_id in LEAGUES.items():
        print(f"Fetching {league_name} ({league_id})...")
        data = fetch_player_stats(league_id, SEASON)
        save_files(data, league_name)
    print(f"⏱️ Completed in {(datetime.now() - start).seconds} seconds.")


if __name__ == "__main__":
    main()

