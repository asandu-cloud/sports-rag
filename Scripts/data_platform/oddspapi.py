"""OddsPapi access shared by the closing capture and the price studies (rework A1 step 4, A5.3).

OddsPapi's ``/v4/historical-odds`` is free (never counted against the quota) and returns every price
change of up to three bookmakers for one fixture, each with its time and whether the selection was
active. Every other endpoint costs one request from a quota that resets on the 1st of each month (250 on
the free plan). The client caches every successful response under ``Index/oddspapi/``, waits out each
endpoint's cooldown and stops before a billable budget. The API key is passed in, never written to disk.
"""
from __future__ import annotations

import difflib
from datetime import datetime, timedelta
import gzip
import hashlib
import json
from pathlib import Path
import re
import time

import requests

from .reference_price import norm

BASE = "https://api.oddspapi.io"
CACHE = Path(__file__).resolve().parents[2] / "Index/oddspapi"
FREE = {"/v4/historical-odds", "/v4/account"}
COOLDOWN = {"/v4/fixtures": 2.1, "/v4/historical-odds": 5.1}
TOURNAMENTS = {"EPL": 17, "LaLiga": 8, "SerieA": 23, "Bundesliga": 35, "Ligue1": 34, "Championship": 18,
               "Eredivisie": 37, "PrimeiraLiga": 238, "BelgianProLeague": 38, "UCL": 7, "UEL": 679, "UECL": 34480,
               "SuperLig": 52}
# API-Football bookmaker name -> OddsPapi slug. OddsPapi does not carry BetVictor; its Betano RO is a
# clone of "betano". Bet365 prices differ by country site: API-Football's match the Spanish, Italian, Greek
# and Dutch sites (20 Oct pilot), not the UK ("bet365") or German ones.
BOOKS = {"Pinnacle": "pinnacle", "Bet365": "bet365.es", "1xBet": "1xbet", "William Hill": "williamhill",
         "Marathonbet": "marathonbet", "SBO": "sbobet", "Dafabet": "dafabet", "Betfair": "betfair-ex",
         "Betano": "betano"}
MARKET_TYPES = {"goals": "totals", "corners": "totals-corners", "cards": "totals-bookings",
                "btts": "bothteamsscore", "moneyline": "1x2", "spreads": "spreads"}
OUTCOME_NAMES = {"over": "Over", "under": "Under", "yes": "Yes", "no": "No", "home": "1", "draw": "X", "away": "2"}
NOISE = {"fc", "afc", "cf", "sc", "ac", "ssc", "as", "cd", "ud", "rc", "sv", "vfb", "vfl", "tsg", "fsv", "bc",
         "sk", "fk", "the", "club", "de", "calcio", "1"}


class BudgetExceeded(RuntimeError):
    pass


def utc(text):
    return datetime.fromisoformat(str(text).replace("Z", "+00:00"))


class OddsPapiClient:
    """Cached, cooldown-respecting, budget-capped OddsPapi client."""

    def __init__(self, key, cache_dir=CACHE, *, budget=40, session=None, sleep=time.sleep, clock=time.monotonic,
                 compress=False):
        self.key, self.cache = key, Path(cache_dir)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.budget, self.session, self.sleep, self.clock = budget, session or requests.Session(), sleep, clock
        self.compress = compress  # gzip new cache files (full price histories run to ~10 MB each)
        self._last = {}

    def _usage(self):
        path = self.cache / "usage.json"
        return json.loads(path.read_text()) if path.exists() else {"billable": 0}

    def get(self, path, **params):
        name = hashlib.sha1(json.dumps([path, params], sort_keys=True).encode()).hexdigest()[:20]
        cached, packed = self.cache / f"{name}.json", self.cache / f"{name}.json.gz"
        if cached.exists():
            return json.loads(cached.read_text())["body"]
        if packed.exists():
            return json.loads(gzip.decompress(packed.read_bytes()))["body"]
        billable = path not in FREE
        usage = self._usage()
        if billable and usage["billable"] >= self.budget:
            raise BudgetExceeded(f"OddsPapi budget of {self.budget} billable requests reached")
        wait = COOLDOWN.get(path, 1.1) - (self.clock() - self._last.get(path, -1e9))
        if wait > 0:
            self.sleep(wait)
        response = self.session.get(BASE + path, params={**params, "apiKey": self.key}, timeout=60)
        self._last[path] = self.clock()
        if billable:
            usage["billable"] += 1
            (self.cache / "usage.json").write_text(json.dumps(usage))
        body = response.json()
        if response.status_code == 200:
            text = json.dumps({"path": path, "params": params, "body": body})
            if self.compress:
                packed.write_bytes(gzip.compress(text.encode()))
            else:
                cached.write_text(text)
        return body


def _tokens(name):
    words = re.sub(r"[^a-z0-9 ]", " ", str(name).lower().replace("ü", "u").replace("ö", "o").replace("é", "e")).split()
    return {w for w in words if w not in NOISE}


def _names_match(a, b):
    ta, tb = _tokens(a), _tokens(b)
    return bool(ta & tb) or difflib.SequenceMatcher(None, " ".join(sorted(ta)), " ".join(sorted(tb))).ratio() >= 0.75


def match_fixtures(ours, theirs, *, window=timedelta(minutes=20)):
    """{our fixture id: OddsPapi fixtureId} by kickoff time and both team names."""
    mapping = {}
    for fixture in ours:
        hits = [t for t in theirs if abs(utc(t["startTime"]) - fixture["kickoff"]) <= window
                and _names_match(fixture["home"], t["participant1Name"]) and _names_match(fixture["away"], t["participant2Name"])]
        if len(hits) == 1:
            mapping[fixture["fixture_id"]] = hits[0]["fixtureId"]
    return mapping


def market_index(markets):
    """(market type, handicap) -> (market id, {outcome name: outcome id}) for football's full-time markets."""
    index = {}
    for m in markets or []:
        if m.get("sportId") != 10 or m.get("period") != "fulltime" or m.get("playerProp"):
            continue
        index[(m["marketType"], float(m["handicap"]))] = (
            str(m["marketId"]), {o["outcomeName"]: str(o["outcomeId"]) for o in m.get("outcomes", [])})
    return index


def locate(index, line):
    """(market id, outcome id) of one line's side in OddsPapi, or None.

    ``line`` has ``market`` (goals, corners, cards, btts, moneyline, spreads), ``side`` (normalised; a team
    name for handicaps), ``point``, ``home`` and ``away``. Asian handicap markets carry the home team's
    handicap: outcome "1" is the home team at the market's handicap, outcome "2" the away team at minus it.
    """
    market = line["market"]
    if market == "spreads":
        if line["point"] is None:
            return None
        side, home, away = line["side"], norm(line["home"]), norm(line["away"])
        if side == home and side != away:
            handicap, name = float(line["point"]), "1"
        elif side == away and side != home:
            handicap, name = -float(line["point"]), "2"
        else:
            return None
    elif market in ("moneyline", "btts"):
        handicap, name = 0.0, OUTCOME_NAMES.get(line["side"])
    else:
        if line["point"] is None:
            return None
        handicap, name = float(line["point"]), OUTCOME_NAMES.get(line["side"])
    entry = index.get((MARKET_TYPES.get(market), handicap + 0.0))
    if not entry or name not in entry[1]:
        return None
    return entry[0], entry[1][name]


def series(body, slug, market_id, outcome_id):
    """[(time, price or None, active)] of one outcome at one bookmaker, in time order."""
    markets = ((((body or {}).get("bookmakers") or {}).get(slug) or {}).get("markets") or {})
    entries = ((((markets.get(market_id) or {}).get("outcomes") or {}).get(outcome_id) or {}).get("players") or {}).get("0") or []
    points = [(utc(e["createdAt"]), float(e["price"]) if e.get("price") else None, e.get("active", True) is not False)
              for e in entries if e.get("createdAt")]
    return sorted(points, key=lambda p: p[0])


def last_before(points, moment):
    """(time, price) of the last entry strictly before ``moment`` if it is an active price, else None.

    A line whose last entry is a suspension was withdrawn: it has no price at ``moment``, and an older
    active price is not used. (20 Oct: Pinnacle's live lines end a median 1.2 minutes before kickoff with
    an active price; the inactive endings were lines withdrawn about 22 days earlier.)
    """
    for when, price, active in reversed(points):
        if when < moment:
            return (when, price) if active and price else None
    return None
