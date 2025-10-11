import requests
import json
from dotenv import load_dotenv
import os

# === CONFIG ===
load_dotenv()
API_KEY = os.getenv("API-FOOTBALL-KEY")
BASE_URL = "https://v3.football.api-sports.io"
HEADERS = {"x-apisports-key": API_KEY}
LEAGUE_ID = 39   # Premier League
SEASON = 2025

# --- Helper Functions ---
def safe_request(endpoint, params):
    """Generic GET request"""
    try:
        r = requests.get(f"{BASE_URL}/{endpoint}", headers=HEADERS, params=params, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"⚠️ Error calling {endpoint}: {e}")
        return None


def get_fixture_id():
    """Fetch a finished fixture ID to use for match-based endpoints"""
    data = safe_request("fixtures", {"league": LEAGUE_ID, "season": SEASON, "status": "FT"})
    if data and data.get("response"):
        return data["response"][0]["fixture"]["id"]
    return None


def extract_keys(data, prefix=""):
    """
    Recursively extract all keys from nested dicts/lists.
    Returns a list of key paths.
    """
    keys = []
    if isinstance(data, dict):
        for k, v in data.items():
            new_prefix = f"{prefix}.{k}" if prefix else k
            keys.append(new_prefix)
            keys.extend(extract_keys(v, new_prefix))
    elif isinstance(data, list):
        if len(data) > 0:
            keys.extend(extract_keys(data[0], prefix))
    return keys


def inspect_endpoint(endpoint, params, save_name=None):
    """Fetch endpoint, extract all nested keys, and print/save them."""
    print(f"\n🔍 Inspecting endpoint: /{endpoint}")
    data = safe_request(endpoint, params)
    if not data or not data.get("response"):
        print(" → No data returned.")
        return

    sample = data["response"][0]
    all_keys = sorted(set(extract_keys(sample)))

    print(f"✅ Found {len(all_keys)} keys.")
    for k in all_keys:
        print(" ", k)

    # Optionally save results
    if save_name:
        with open(save_name, "w", encoding="utf-8") as f:
            json.dump(all_keys, f, indent=4, ensure_ascii=False)
        print(f"💾 Saved keys to {save_name}")


# --- MAIN ---
def main():
    fixture_id = get_fixture_id()
    if not fixture_id:
        print("❌ Could not get fixture ID. Check your key or season.")
        return

    # Inspect 4 key endpoints
    inspect_endpoint("players", {"league": LEAGUE_ID, "season": SEASON, "page": 1},
                     save_name="player_season_keys.json")

    inspect_endpoint("fixtures/players", {"fixture": fixture_id},
                     save_name="player_fixture_keys.json")

    inspect_endpoint("teams/statistics", {"league": LEAGUE_ID, "season": SEASON, "team": 33},
                     save_name="team_season_keys.json")

    inspect_endpoint("fixtures/statistics", {"fixture": fixture_id},
                     save_name="team_fixture_keys.json")


if __name__ == "__main__":
    main()