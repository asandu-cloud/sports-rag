# Script for scraps (total players, etc.)

import requests
import json 
import pandas as pd
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
import os

# === CONFIGURATION ===
load_dotenv()

# Replace with your API key (or load from .env)
API_KEY = "63dbcf42afa65f2e2769daee817f6e48"

# Output folder
OUTPUT_DIR = Path("/Users/sanduandrei/Desktop/Betting_RAG/Output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Premier League info
LEAGUE_ID = 39   # Premier League
SEASON = 2025    # Current active season

# === FUNCTION ===
def fetch_all_players(league_id: int, season: int):
    """
    Fetch all players for a given league and season.
    """
    url = "https://v3.football.api-sports.io/players"
    headers = {"x-apisports-key": API_KEY}
    players = []
    page = 1

    print(f"Fetching players for league {league_id} - season {season}...")

    while True:
        params = {"league": league_id, "season": season, "page": page}
        response = requests.get(url, headers=headers, params=params)
        data = response.json()

        errors = data.get("errors")
        response_data = data.get("response", [])

        print(f"Page {page} | Errors: {errors} | Players returned: {len(response_data)}")

        if not response_data:
            break

        for entry in response_data:
            player = entry.get("player", {})
            stats = entry.get("statistics", [{}])[0]

            players.append({
                "player_id": player.get("id"),
                "name": player.get("name"),
                "firstname": player.get("firstname"),
                "lastname": player.get("lastname"),
                "age": player.get("age"),
                "birth_date": player.get("birth", {}).get("date"),
                "nationality": player.get("nationality"),
                "height": player.get("height"),
                "weight": player.get("weight"),
                "photo": player.get("photo"),
                "team": stats.get("team", {}).get("name"),
                "league": stats.get("league", {}).get("name"),
                "position": stats.get("games", {}).get("position"),
                "appearances": stats.get("games", {}).get("appearences"),
                "minutes": stats.get("games", {}).get("minutes"),
                "rating": stats.get("games", {}).get("rating"),
                "goals": stats.get("goals", {}).get("total"),
                "assists": stats.get("goals", {}).get("assists"),
                "yellow_cards": stats.get("cards", {}).get("yellow"),
                "red_cards": stats.get("cards", {}).get("red")
            })

        page += 1

        # Stop if fewer than 20 players on this page (last page)
        if len(response_data) < 20:
            break

    return players


def save_player_data(players):
    """
    Save players to JSON and CSV files.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    json_path = OUTPUT_DIR / f"PremierLeague_Players_{SEASON}_{timestamp}.json"
    csv_path = OUTPUT_DIR / f"PremierLeague_Players_{SEASON}_{timestamp}.csv"

    # JSON
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(players, f, indent=4, ensure_ascii=False)

    # CSV
    df = pd.DataFrame(players)
    df.to_csv(csv_path, index=False)

    print(f"\n✅ Saved {len(players)} players to:")
    print(f"  - {json_path.name}")
    print(f"  - {csv_path.name}")


def main():
    players = fetch_all_players(LEAGUE_ID, SEASON)
    print(f"\nTotal Premier League players found: {len(players)}")
    save_player_data(players)


if __name__ == "__main__":
    main()