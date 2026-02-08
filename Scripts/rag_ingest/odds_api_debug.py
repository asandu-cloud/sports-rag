#!/usr/bin/env python3
"""
Standalone Odds API diagnostic script.

Purpose:
- Verify API key visibility in runtime environment.
- Verify connectivity and response from /v4/sports.
- Verify EPL odds payload from /v4/sports/soccer_epl/odds.

This script intentionally does not import project RAG modules.
"""

import argparse
import os
import sys
from typing import Dict, Optional

import requests


BASE_URL = "https://api.the-odds-api.com/v4"


def mask_key(key: str) -> str:
    if len(key) <= 8:
        return "*" * len(key)
    return f"{key[:4]}...{key[-4:]}"


def get_key(cli_key: Optional[str]) -> Optional[str]:
    if cli_key:
        return cli_key.strip()
    # Requested key name
    key = os.environ.get("ODDS-API")
    if key:
        return key.strip()
    # Helpful fallback check for common naming mismatch
    alt = os.environ.get("ODDS_API_KEY")
    if alt:
        print("NOTE: Found ODDS_API_KEY but ODDS-API is missing.")
        print("      Your app currently expects ODDS-API.")
        return None
    return None


def print_headers(headers: Dict[str, str]) -> None:
    remaining = headers.get("x-requests-remaining")
    used = headers.get("x-requests-used")
    if remaining is not None or used is not None:
        print(f"Quota headers: x-requests-remaining={remaining}, x-requests-used={used}")


def get_json(url: str, params: Dict[str, str], timeout: int = 20):
    r = requests.get(url, params=params, timeout=timeout)
    return r


def check_sports(api_key: str) -> bool:
    url = f"{BASE_URL}/sports"
    params = {"apiKey": api_key}
    print(f"\n[1/2] GET {url}")
    r = get_json(url, params=params)
    print(f"Status: {r.status_code}")
    print_headers(r.headers)
    if r.status_code != 200:
        print(f"Body: {r.text[:600]}")
        return False

    try:
        data = r.json()
    except Exception as exc:
        print(f"JSON parse error on /sports: {exc}")
        print(f"Body: {r.text[:600]}")
        return False

    keys = [x.get("key") for x in data if isinstance(x, dict)]
    print(f"Sports returned: {len(keys)}")
    print("Sample sport keys:", keys[:12])
    print("Contains soccer_epl:", "soccer_epl" in keys)
    return True


def check_epl_odds(api_key: str, regions: str, markets: str, max_events: int, raw: bool) -> bool:
    url = f"{BASE_URL}/sports/soccer_epl/odds"
    params = {
        "apiKey": api_key,
        "regions": regions,
        "markets": markets,
        "oddsFormat": "decimal",
        "dateFormat": "iso",
    }
    print(f"\n[2/2] GET {url}")
    print(f"Params: regions={regions}, markets={markets}")
    r = get_json(url, params=params)
    print(f"Status: {r.status_code}")
    print_headers(r.headers)
    if r.status_code != 200:
        print(f"Body: {r.text[:1000]}")
        return False

    try:
        data = r.json()
    except Exception as exc:
        print(f"JSON parse error on /odds: {exc}")
        print(f"Body: {r.text[:1000]}")
        return False

    if not isinstance(data, list):
        print("Unexpected payload type for /odds.")
        print(str(data)[:1000])
        return False

    print(f"Upcoming EPL events: {len(data)}")
    for i, ev in enumerate(data[:max_events], start=1):
        home = ev.get("home_team")
        away = ev.get("away_team")
        when = ev.get("commence_time")
        books = ev.get("bookmakers") or []
        print(f"{i}. {home} vs {away} | {when} | bookmakers={len(books)}")

        best = {}
        for bm in books:
            for mk in bm.get("markets", []):
                if mk.get("key") != "h2h":
                    continue
                for out in mk.get("outcomes", []):
                    name = out.get("name")
                    price = out.get("price")
                    try:
                        price = float(price)
                    except Exception:
                        continue
                    if name not in best or price > best[name]:
                        best[name] = price
        if best:
            print(f"   Best H2H: {best}")

    if raw:
        print("\nRaw payload preview:")
        print(str(data[:2])[:2000])

    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Debug Odds API connectivity and payloads.")
    parser.add_argument("--api-key", default=None, help="Override API key (otherwise reads ODDS-API env var).")
    parser.add_argument("--regions", default="uk,eu,us", help="Odds API regions param.")
    parser.add_argument("--markets", default="h2h", help="Odds API markets param.")
    parser.add_argument("--max-events", type=int, default=5, help="How many events to print.")
    parser.add_argument("--raw", action="store_true", help="Print raw payload preview.")
    args = parser.parse_args()

    key = get_key(args.api_key)
    if not key:
        print("ERROR: Missing ODDS API key.")
        print("Expected runtime env var: ODDS-API")
        print("Tip: for this run, you can pass --api-key directly.")
        return 1

    print("Key detected:", mask_key(key))
    try:
        ok_sports = check_sports(key)
        ok_odds = check_epl_odds(key, args.regions, args.markets, args.max_events, args.raw)
    except requests.RequestException as exc:
        print(f"Network/request error: {exc}")
        return 2
    except Exception as exc:
        print(f"Unexpected error: {exc}")
        return 3

    if ok_sports and ok_odds:
        print("\nDiagnostics passed.")
        return 0

    print("\nDiagnostics found issues.")
    return 4


if __name__ == "__main__":
    sys.exit(main())
