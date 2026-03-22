"""
odds_provider.py
================
Drop-in replacement for The Odds API odds fetching, backed by API-Football.

Produces event dicts in the exact format consumed by rag_cli_v2.py:

    {
        "id": "<fixture_id>",
        "home_team": "Arsenal",
        "away_team": "Chelsea",
        "commence_time": "2026-03-19T20:00:00Z",
        "bookmakers": [
            {
                "title": "Bet365",
                "markets": [
                    {
                        "key": "h2h",
                        "outcomes": [
                            {"name": "Home", "price": 2.05},
                            {"name": "Draw", "price": 3.40},
                            ...
                        ]
                    }
                ]
            }
        ]
    }

Usage:
    from odds_provider import fetch_events_apifootball

    events, notes = fetch_events_apifootball("EPL", date.today())
"""

from __future__ import annotations

import logging
import os
import re
import time
from datetime import date, datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

import requests

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger("odds_provider")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[odds_provider] %(levelname)s  %(message)s"))
    logger.addHandler(_h)
    logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
API_FOOTBALL_BASE = "https://v3.football.api-sports.io"

LEAGUE_TO_API_ID: Dict[str, int] = {
    "EPL": 39,
    "LaLiga": 140,
    "SerieA": 135,
    "Bundesliga": 78,
    "Ligue1": 61,
    "UCL": 2,
    "UEL": 3,
    "UECL": 848,
}

# Bookmakers to prefer, in priority order.  When multiple bookmakers carry the
# same bet type we keep only the highest-priority one so the downstream code
# sees a single, clean set of odds per market.
_PREFERRED_BOOKMAKERS: List[str] = [
    "Bet365",
    "William Hill",
    "10Bet",
]

# API-Football bet-type ID  ->  Odds-API-compatible market key
_BET_ID_TO_MARKET_KEY: Dict[int, str] = {
    1: "h2h",
    4: "spreads",
    5: "totals",
    8: "btts",
    10: "correctscore",
    16: "team_totals_home_goals",
    17: "team_totals_away_goals",
    45: "totals_corners_over_under",
    55: "corners_1x2",
    57: "team_totals_home_corners",
    58: "team_totals_away_corners",
    80: "totals_cards_over_under",
    82: "team_totals_home_cards",
    83: "team_totals_away_cards",
    87: "shots_on_target_over_under",
    92: "anytime_goalscorer",
    153: "totals_yellow_cards",
    212: "player_assists",
    242: "player_sot",
    266: "player_fouls",
    271: "player_fouls_home",
    277: "player_fouls_away",
    102: "player_booked",
    251: "player_booked",
}

# Reverse lookup: market key -> bet ID (used for targeted enrichment)
_MARKET_KEY_TO_BET_ID: Dict[str, int] = {v: k for k, v in _BET_ID_TO_MARKET_KEY.items()}

# Current season for API-Football queries
_CURRENT_SEASON = 2025

# Cache: (league, date_str) -> (events, notes, timestamp)
_events_cache: Dict[Tuple[str, str], Tuple[List[Dict], List[str], float]] = {}
_CACHE_TTL = 600  # 10 minutes

# Per-fixture odds cache: fixture_id -> (bookmakers_list, timestamp)
_fixture_odds_cache: Dict[int, Tuple[List[Dict], float]] = {}

# ---------------------------------------------------------------------------
# API key
# ---------------------------------------------------------------------------
_api_key_warned = False


def _api_football_key() -> Optional[str]:
    """Resolve the API-Football key from environment."""
    for k in ("API-FOOTBALL-KEY", "API_FOOTBALL_KEY"):
        v = os.environ.get(k)
        if v and str(v).strip():
            return str(v).strip().strip("'\"")
    return None


# ---------------------------------------------------------------------------
# Low-level API call
# ---------------------------------------------------------------------------
def _api_get(endpoint: str, params: Dict[str, Any]) -> Tuple[Optional[List], Optional[str]]:
    """
    Call API-Football.  Returns (response_list, None) on success or
    (None, error_message) on failure.
    """
    global _api_key_warned
    key = _api_football_key()
    if not key:
        if not _api_key_warned:
            logger.warning("API-FOOTBALL-KEY not set. Odds fetching disabled.")
            _api_key_warned = True
        return None, "Missing API-FOOTBALL-KEY environment variable."

    url = f"{API_FOOTBALL_BASE}{endpoint}"
    headers = {"x-apisports-key": key}
    try:
        logger.debug("GET %s  params=%s", endpoint, params)
        r = requests.get(url, headers=headers, params=params, timeout=30)
        if r.status_code != 200:
            msg = f"API-Football HTTP {r.status_code}: {(r.text or '')[:200]}"
            logger.error(msg)
            return None, msg
        data = r.json()
        errors = data.get("errors")
        if errors:
            # API-Football returns {"errors": {"requests": "..."}} on rate limit
            msg = f"API-Football errors: {errors}"
            logger.error(msg)
            return None, msg
        return data.get("response", []), None
    except requests.exceptions.Timeout:
        msg = "API-Football request timed out."
        logger.error(msg)
        return None, msg
    except Exception as exc:
        msg = f"API-Football request failed: {exc}"
        logger.error(msg)
        return None, msg


# ---------------------------------------------------------------------------
# Value parsing helpers
# ---------------------------------------------------------------------------
# Patterns for "Over 2.5", "Under 3.5", "Home -1", "Away +1.5", etc.
_OVER_UNDER_RE = re.compile(
    r"^(Over|Under)\s+([\d]+(?:\.\d+)?)$", re.IGNORECASE
)
_HANDICAP_RE = re.compile(
    r"^(.+?)\s+([+-]?\d+(?:\.\d+)?)$"
)
_CORRECT_SCORE_RE = re.compile(
    r"^\d+:\d+$"
)


def _parse_value(value_str: str, home_team: str, away_team: str) -> Dict[str, Any]:
    """
    Parse an API-Football bet value string into an outcome dict with
    'name', 'price' (to be filled later), and optionally 'point'.

    Examples:
        "Over 2.5"  -> {"name": "Over",  "point": 2.5}
        "Under 3.5" -> {"name": "Under", "point": 3.5}
        "Home"       -> {"name": home_team}
        "Away"       -> {"name": away_team}
        "Draw"       -> {"name": "Draw"}
        "Home -1"    -> {"name": home_team, "point": -1.0}
        "Away +1.5"  -> {"name": away_team, "point": 1.5}
        "1:0"        -> {"name": "1:0"}
        "Yes"        -> {"name": "Yes"}
        "No"         -> {"name": "No"}
    """
    v = str(value_str).strip()
    result: Dict[str, Any] = {}

    # Over/Under with line
    m = _OVER_UNDER_RE.match(v)
    if m:
        result["name"] = m.group(1).capitalize()  # "Over" or "Under"
        result["point"] = float(m.group(2))
        return result

    # Correct score  "1:0", "2:1", etc.
    if _CORRECT_SCORE_RE.match(v):
        result["name"] = v
        return result

    # Yes / No  (BTTS)
    if v.lower() in ("yes", "no"):
        result["name"] = v.capitalize()
        return result

    # Handicap values:  "Home -1", "Away +1.5", or team-name based
    m = _HANDICAP_RE.match(v)
    if m:
        label = m.group(1).strip()
        point = float(m.group(2))
        # Normalize "Home"/"Away" to actual team names
        if label.lower() == "home":
            label = home_team
        elif label.lower() == "away":
            label = away_team
        result["name"] = label
        result["point"] = point
        return result

    # Plain labels: "Home", "Away", "Draw"
    if v.lower() == "home":
        result["name"] = home_team
    elif v.lower() == "away":
        result["name"] = away_team
    elif v.lower() == "draw":
        result["name"] = "Draw"
    else:
        # Goalscorer names, etc. — pass through as-is
        result["name"] = v

    return result


# ---------------------------------------------------------------------------
# Odds transformation
# ---------------------------------------------------------------------------
def _transform_apifootball_odds(
    api_bookmakers: List[Dict],
    home_team: str,
    away_team: str,
) -> List[Dict]:
    """
    Transform API-Football bookmaker/bets structure into the flat
    bookmaker/markets/outcomes structure expected by rag_cli_v2.py.

    Returns a list of bookmaker dicts, one per API-Football bookmaker,
    each containing a 'markets' list with 'key' and 'outcomes'.
    """
    transformed: List[Dict] = []

    for bm in api_bookmakers:
        bm_name = bm.get("name", "Unknown")
        markets: List[Dict] = []

        for bet in bm.get("bets", []):
            bet_id = bet.get("id")
            bet_name = bet.get("name", "")
            market_key = _BET_ID_TO_MARKET_KEY.get(bet_id)

            if market_key is None:
                # Unknown bet type — skip
                logger.debug(
                    "Skipping unknown bet type id=%s name='%s' from %s",
                    bet_id, bet_name, bm_name,
                )
                continue

            outcomes: List[Dict] = []
            is_spread = market_key == "spreads"
            for val in bet.get("values", []):
                odd_str = val.get("odd")
                value_str = val.get("value", "")
                if not odd_str:
                    continue
                try:
                    price = float(odd_str)
                except (ValueError, TypeError):
                    continue
                if price < 1.01:
                    continue  # nonsensical odds

                outcome = _parse_value(value_str, home_team, away_team)
                outcome["price"] = price

                # Fix Asian Handicap: API-Football shows "Home -1" / "Away -1"
                # where both sides have the SAME negative point. The away side
                # actually RECEIVES the handicap (positive), so flip its sign.
                if is_spread and outcome.get("point") is not None:
                    raw_val = str(value_str).strip().lower()
                    if raw_val.startswith("away"):
                        outcome["point"] = -outcome["point"]

                outcomes.append(outcome)

            if outcomes:
                markets.append({"key": market_key, "outcomes": outcomes})

        if markets:
            transformed.append({"title": bm_name, "markets": markets})

    # Sort bookmakers: preferred ones first, then alphabetical
    pref_lower = [p.lower() for p in _PREFERRED_BOOKMAKERS]

    def _bm_sort_key(bm_dict: Dict) -> Tuple[int, str]:
        title = bm_dict.get("title", "").lower()
        try:
            idx = pref_lower.index(title)
        except ValueError:
            idx = len(pref_lower)
        return (idx, bm_dict.get("title", ""))

    transformed.sort(key=_bm_sort_key)
    return transformed


# ---------------------------------------------------------------------------
# Fixture discovery
# ---------------------------------------------------------------------------
def _fetch_fixtures(league: str, target_date: date) -> Tuple[List[Dict], Optional[str]]:
    """
    Fetch fixtures for a league on a given date via API-Football.
    Returns (fixtures_list, error_or_None).
    """
    league_id = LEAGUE_TO_API_ID.get(league)
    if league_id is None:
        return [], f"Unknown league '{league}'. Valid: {list(LEAGUE_TO_API_ID.keys())}"

    date_str = target_date.strftime("%Y-%m-%d")
    params = {
        "league": league_id,
        "season": _CURRENT_SEASON,
        "date": date_str,
    }
    rows, err = _api_get("/fixtures", params)
    if err:
        return [], err
    if not rows:
        logger.info("No fixtures found for %s on %s", league, date_str)
        return [], None
    logger.info("Found %d fixture(s) for %s on %s", len(rows), league, date_str)
    return rows, None


# ---------------------------------------------------------------------------
# Per-fixture odds fetch
# ---------------------------------------------------------------------------
def _fetch_fixture_odds(fixture_id: int) -> Tuple[List[Dict], Optional[str]]:
    """
    Fetch odds for a single fixture.  Returns the raw bookmakers list
    from the API-Football response, or ([], error).

    Results are cached for _CACHE_TTL seconds.
    """
    now = time.time()
    cached = _fixture_odds_cache.get(fixture_id)
    if cached and (now - cached[1]) < _CACHE_TTL:
        logger.debug("Odds cache hit for fixture %d", fixture_id)
        return cached[0], None

    rows, err = _api_get("/odds", {"fixture": fixture_id})
    if err:
        return [], err
    if not rows:
        logger.info("No odds available for fixture %d", fixture_id)
        return [], None

    # API-Football returns a list with one element per fixture
    bookmakers = rows[0].get("bookmakers", []) if rows else []
    _fixture_odds_cache[fixture_id] = (bookmakers, now)
    return bookmakers, None


# ---------------------------------------------------------------------------
# Date normalization for commence_time
# ---------------------------------------------------------------------------
def _normalize_datetime(date_str: str) -> str:
    """
    Normalize API-Football date string to ISO-8601 UTC format matching
    The Odds API convention: '2026-03-19T20:00:00Z'
    """
    if not date_str:
        return ""
    # API-Football format: "2026-03-19T20:00:00+00:00"
    # We want: "2026-03-19T20:00:00Z"
    try:
        dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception:
        # Fallback: strip timezone offset and append Z
        if "+" in date_str:
            return date_str.split("+")[0] + "Z"
        return date_str


# ---------------------------------------------------------------------------
# Team name normalization
# ---------------------------------------------------------------------------
# API-Football sometimes uses slightly different team names than The Odds API.
# This mapping corrects common discrepancies so Chroma lookups work.
_TEAM_NAME_OVERRIDES: Dict[str, str] = {
    # Add overrides here as needed, e.g.:
    # "Manchester United FC": "Manchester United",
    # "FC Barcelona": "Barcelona",
}


def _normalize_team_name(raw: str) -> str:
    """Return the canonical team name used in the Chroma KB."""
    name = str(raw or "").strip()
    return _TEAM_NAME_OVERRIDES.get(name, name)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def fetch_events_apifootball(
    league: str,
    target_date: date,
    *,
    requested_bet_ids: Optional[Set[int]] = None,
) -> Tuple[List[Dict], List[str]]:
    """
    Fetch fixtures + odds for *league* on *target_date* from API-Football
    and return them in the exact event-dict format expected by rag_cli_v2.py.

    Parameters
    ----------
    league : str
        One of LEAGUE_TO_API_ID keys (e.g. "EPL", "LaLiga", "UCL").
    target_date : date
        The match date to fetch.
    requested_bet_ids : set of int, optional
        If provided, only include these API-Football bet IDs in the output.
        When None, all recognized bet types are included.

    Returns
    -------
    (events, notes) : Tuple[List[Dict], List[str]]
        events  — list of event dicts compatible with rag_cli_v2.py
        notes   — list of informational/warning strings
    """
    notes: List[str] = []
    date_str = target_date.isoformat()

    # Check cache
    cache_key = (league, date_str)
    now = time.time()
    cached = _events_cache.get(cache_key)
    if cached and (now - cached[2]) < _CACHE_TTL:
        logger.debug("Events cache hit for %s on %s", league, date_str)
        return cached[0], cached[1]

    # 1. Discover fixtures
    fixtures, fix_err = _fetch_fixtures(league, target_date)
    if fix_err:
        notes.append(fix_err)
        return [], notes
    if not fixtures:
        notes.append(f"No fixtures for {league} on {date_str}.")
        return [], notes

    events: List[Dict] = []

    # 2. For each fixture, fetch odds and transform
    for fix in fixtures:
        fix_data = fix.get("fixture", {})
        teams_data = fix.get("teams", {})
        fixture_id = fix_data.get("id")
        if not fixture_id:
            continue

        home_raw = teams_data.get("home", {}).get("name", "Home")
        away_raw = teams_data.get("away", {}).get("name", "Away")
        home_team = _normalize_team_name(home_raw)
        away_team = _normalize_team_name(away_raw)
        commence_time = _normalize_datetime(fix_data.get("date", ""))

        logger.info(
            "Fetching odds for fixture %d: %s vs %s", fixture_id, home_team, away_team
        )

        api_bookmakers, odds_err = _fetch_fixture_odds(fixture_id)
        if odds_err:
            notes.append(f"Odds unavailable for {home_team} vs {away_team}: {odds_err}")
            # Still include the event with empty bookmakers so the match
            # shows up in fixture lists even without odds
            api_bookmakers = []

        bookmakers = _transform_apifootball_odds(api_bookmakers, home_team, away_team)

        # Filter to requested bet types if specified
        if requested_bet_ids is not None:
            allowed_keys = {
                _BET_ID_TO_MARKET_KEY[bid]
                for bid in requested_bet_ids
                if bid in _BET_ID_TO_MARKET_KEY
            }
            for bm in bookmakers:
                bm["markets"] = [
                    m for m in bm["markets"] if m["key"] in allowed_keys
                ]
            bookmakers = [bm for bm in bookmakers if bm["markets"]]

        event: Dict[str, Any] = {
            "id": str(fixture_id),
            "home_team": home_team,
            "away_team": away_team,
            "commence_time": commence_time,
            "bookmakers": bookmakers,
            # Extra metadata useful for downstream enrichment
            "_fixture_id": fixture_id,
            "_league": league,
        }
        events.append(event)

    if not events:
        notes.append(f"Found fixtures but no odds available for {league} on {date_str}.")

    # Cache result
    _events_cache[cache_key] = (events, notes, now)

    logger.info(
        "Returning %d event(s) with odds for %s on %s", len(events), league, date_str
    )
    return events, notes


def fetch_events_apifootball_multi(
    leagues: List[str],
    target_date: date,
) -> Tuple[List[Dict], List[str]]:
    """
    Multi-league variant — fetches events from all specified leagues and
    tags each event with its source league via the '_league' key.

    Drop-in replacement for fetch_events_multi() in rag_cli_v2.py.
    """
    all_events: List[Dict] = []
    all_notes: List[str] = []
    for league in leagues:
        events, notes = fetch_events_apifootball(league, target_date)
        for ev in events:
            ev["_league"] = league
        all_events.extend(events)
        all_notes.extend(notes)
    return all_events, all_notes


def enrich_event_odds(
    event: Dict,
    league: str,
    extra_bet_ids: Optional[Set[int]] = None,
) -> Tuple[Set[str], Optional[str]]:
    """
    Re-fetch odds for a single event to discover additional market keys
    (e.g. corners, cards, SoT) that may not have been in the initial pull.

    Merges new markets into the event's bookmakers in-place.

    Returns (set_of_all_market_keys, error_or_None).
    """
    fixture_id = event.get("_fixture_id")
    if not fixture_id:
        # Try parsing from string id
        try:
            fixture_id = int(event.get("id", 0))
        except (ValueError, TypeError):
            return set(), "No fixture_id on event."

    home = str(event.get("home_team", "Home"))
    away = str(event.get("away_team", "Away"))

    api_bookmakers, err = _fetch_fixture_odds(fixture_id)
    if err:
        return set(), err

    new_bms = _transform_apifootball_odds(api_bookmakers, home, away)

    # Build set of existing (bookmaker_title, market_key) pairs
    existing_pairs: Set[Tuple[str, str]] = set()
    for bm in event.get("bookmakers", []):
        for mk in bm.get("markets", []):
            existing_pairs.add((bm["title"], mk["key"]))

    # Merge new markets into existing bookmakers or add new bookmaker entries
    existing_bm_map: Dict[str, Dict] = {
        bm["title"]: bm for bm in event.get("bookmakers", [])
    }

    for new_bm in new_bms:
        title = new_bm["title"]
        if title in existing_bm_map:
            for mk in new_bm["markets"]:
                if (title, mk["key"]) not in existing_pairs:
                    existing_bm_map[title]["markets"].append(mk)
                    existing_pairs.add((title, mk["key"]))
        else:
            event.setdefault("bookmakers", []).append(new_bm)
            existing_bm_map[title] = new_bm
            for mk in new_bm["markets"]:
                existing_pairs.add((title, mk["key"]))

    # Collect all market keys now present
    all_keys: Set[str] = set()
    for bm in event.get("bookmakers", []):
        for mk in bm.get("markets", []):
            all_keys.add(mk["key"])

    return all_keys, None


# ---------------------------------------------------------------------------
# Cache management
# ---------------------------------------------------------------------------
def clear_cache() -> None:
    """Clear all in-memory caches (useful for testing or forced refresh)."""
    _events_cache.clear()
    _fixture_odds_cache.clear()
    logger.info("All odds caches cleared.")


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import json
    import sys

    logging.basicConfig(level=logging.DEBUG)
    logger.setLevel(logging.DEBUG)

    league_arg = sys.argv[1] if len(sys.argv) > 1 else "EPL"
    date_arg = sys.argv[2] if len(sys.argv) > 2 else date.today().isoformat()

    try:
        target = date.fromisoformat(date_arg)
    except ValueError:
        print(f"Invalid date: {date_arg}. Use YYYY-MM-DD format.")
        sys.exit(1)

    print(f"Fetching {league_arg} events for {target} ...")
    evts, nts = fetch_events_apifootball(league_arg, target)

    if nts:
        print("\nNotes:")
        for n in nts:
            print(f"  - {n}")

    print(f"\n{len(evts)} event(s) returned.\n")

    for ev in evts:
        print(f"  {ev['home_team']} vs {ev['away_team']}  ({ev['commence_time']})")
        market_keys = set()
        for bm in ev.get("bookmakers", []):
            for mk in bm.get("markets", []):
                market_keys.add(mk["key"])
        print(f"    Markets: {sorted(market_keys)}")
        print(f"    Bookmakers: {[bm['title'] for bm in ev.get('bookmakers', [])]}")

    if evts:
        print("\nFull first event JSON:")
        print(json.dumps(evts[0], indent=2, default=str))
