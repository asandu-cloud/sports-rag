"""Live odds ingestion and normalization for the live betting stack."""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("live_odds")

# One-shot diagnostic: log the bet inventory the first time we see an in-play
# row so we can verify bet-ID mapping without spamming every poll.
_INPLAY_BETS_LOGGED = False


# ---------------------------------------------------------------------------
# In-play bet-ID remap
# ---------------------------------------------------------------------------
# API-Football's `/odds/live` endpoint uses a different bet-ID space from the
# pre-match `/odds` endpoint. `_transform_apifootball_odds` only understands the
# pre-match IDs (1, 5, 8, 45, 80, 87…), so we rewrite the in-play IDs to their
# pre-match equivalents before handing off. Mapping verified from the
# INPLAY_BETS_DIAGNOSTIC log dump.
_INPLAY_TO_PREMATCH_BET_ID: Dict[int, int] = {
    59: 1,    # Fulltime Result    -> Match Winner (h2h)
    25: 5,    # Match Goals        -> Goals Over/Under (totals)
    69: 8,    # Both Teams to Score -> Both Teams Score (btts)
    37: 45,   # Total Corners      -> Corners Over/Under
    119: 80,  # Total Cards        -> Cards Over/Under
    150: 87,  # Total shots on goal -> Shots on Target Over/Under
    72: 12,   # Double Chance      -> Double Chance (if extractor supports it)
}


def _wrap_inplay_row_as_bookmaker(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Convert an API-Football `/odds/live` row into the `bookmakers` shape the
    pre-match transformer expects.

    In-play responses have a flat ``row["odds"]`` list (43+ bet markets, each
    with ``id``/``name``/``values``). Pre-match responses wrap bets inside
    ``row["bookmakers"][].bets[]``. We synthesize a single pseudo-bookmaker
    named "API-Football Live" and rewrite each bet's id to the pre-match
    equivalent so ``_transform_apifootball_odds`` can consume the payload
    unchanged.
    """
    bets = row.get("odds") or []
    if not isinstance(bets, list) or not bets:
        return []

    remapped_bets: List[Dict[str, Any]] = []
    for bet in bets:
        if not isinstance(bet, dict):
            continue
        inplay_id = bet.get("id")
        prematch_id = _INPLAY_TO_PREMATCH_BET_ID.get(inplay_id)
        if prematch_id is None:
            # Not a market we consume — skip silently (the pre-match transformer
            # would also skip it).
            continue

        # Rewrite each value to match the pre-match shape. The in-play response
        # puts the totals line in a separate `handicap` field (e.g.
        # value="Over", handicap="7.5"), whereas pre-match embeds it in the
        # value string ("Over 7.5"). The existing _parse_value regex only
        # matches the embedded form, so we splice the handicap back into the
        # value string for Over/Under markets.
        rewritten_values: List[Dict[str, Any]] = []
        for val in bet.get("values") or []:
            if not isinstance(val, dict):
                continue
            if val.get("suspended"):
                continue  # suspended market — drop
            value_str = str(val.get("value") or "").strip()
            handicap = val.get("handicap")
            if handicap not in (None, "") and value_str.lower() in ("over", "under"):
                value_str = f"{value_str} {handicap}"
            rewritten_values.append({
                "value": value_str,
                "odd": val.get("odd"),
                "handicap": handicap,
            })

        if not rewritten_values:
            continue

        remapped_bets.append({
            "id": prematch_id,
            "name": bet.get("name", ""),
            "values": rewritten_values,
        })

    if not remapped_bets:
        return []

    return [{
        "id": 0,
        "name": "API-Football Live",
        "bets": remapped_bets,
    }]

_SCRIPTS_DIR = Path(__file__).resolve().parents[1]
_RAG_INGEST_DIR = _SCRIPTS_DIR / "rag_ingest"
if str(_RAG_INGEST_DIR) not in sys.path:
    sys.path.insert(0, str(_RAG_INGEST_DIR))

from odds_provider import _transform_apifootball_odds  # type: ignore
from core.odds_extraction import (  # type: ignore
    extract_btts_odds,
    extract_moneyline_odds,
    extract_total_line_options,
)
from prob_models import implied_prob, remove_vig_two_way  # type: ignore


API_FOOTBALL_BASE = "https://v3.football.api-sports.io"
_CACHE_TTL = 20
_live_odds_cache: Dict[int, Tuple[Dict[str, Any], float]] = {}


def _api_football_key() -> Optional[str]:
    for k in ("API-FOOTBALL-KEY", "API_FOOTBALL_KEY"):
        v = os.environ.get(k)
        if v and str(v).strip():
            return str(v).strip().strip("'\"")
    return None


async def _async_api_get(endpoint: str, params: Dict[str, Any], session: Optional[Any] = None) -> Optional[List[Dict[str, Any]]]:
    key = _api_football_key()
    if not key:
        logger.warning("API-FOOTBALL-KEY not set — cannot fetch %s", endpoint)
        return None

    url = f"{API_FOOTBALL_BASE}{endpoint}"
    headers = {"x-apisports-key": key}

    try:
        import aiohttp

        if session is not None:
            resp = await session.get(url, headers=headers, params=params, timeout=aiohttp.ClientTimeout(total=20))
            if resp.status != 200:
                body = ""
                try:
                    body = (await resp.text())[:200]
                except Exception:
                    pass
                logger.warning("API-Football %s HTTP %s params=%s body=%s", endpoint, resp.status, params, body)
                return None
            data = await resp.json()
            return data.get("response", [])

        async with aiohttp.ClientSession() as local_session:
            resp = await local_session.get(url, headers=headers, params=params, timeout=aiohttp.ClientTimeout(total=20))
            if resp.status != 200:
                body = ""
                try:
                    body = (await resp.text())[:200]
                except Exception:
                    pass
                logger.warning("API-Football %s HTTP %s params=%s body=%s", endpoint, resp.status, params, body)
                return None
            data = await resp.json()
            return data.get("response", [])
    except ImportError:
        pass

    import requests

    def _sync_get():
        try:
            r = requests.get(url, headers=headers, params=params, timeout=20)
            if r.status_code != 200:
                logger.warning("API-Football %s HTTP %s params=%s body=%s", endpoint, r.status_code, params, r.text[:200])
                return None
            return r.json().get("response", [])
        except Exception as exc:
            logger.warning("API-Football %s sync request failed: %s", endpoint, exc)
            return None

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _sync_get)


def _build_event(bookmakers: List[Dict[str, Any]], home_team: str, away_team: str) -> Dict[str, Any]:
    return {
        "home_team": home_team,
        "away_team": away_team,
        "bookmakers": bookmakers,
    }


def _best_total_offers(options: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_line_book: Dict[Tuple[float, str], Dict[str, Any]] = {}
    for opt in options:
        point = float(opt["point"])
        bookmaker = str(opt.get("bookmaker") or "")
        entry = by_line_book.setdefault((point, bookmaker), {"line": point, "bookmaker": bookmaker})
        side = str(opt.get("side") or "").lower()
        if side == "over":
            entry["over_odds"] = float(opt["odds"])
        elif side == "under":
            entry["under_odds"] = float(opt["odds"])

    grouped: Dict[float, Dict[str, Any]] = {}
    for (_line, _bookmaker), entry in by_line_book.items():
        over_odds = entry.get("over_odds")
        under_odds = entry.get("under_odds")
        if over_odds is None or under_odds is None:
            continue
        fair_over, fair_under = remove_vig_two_way(float(over_odds), float(under_odds))
        offer = {
            "bookmaker": entry["bookmaker"],
            "over_odds": float(over_odds),
            "under_odds": float(under_odds),
            "fair_over_prob": fair_over,
            "fair_under_prob": fair_under,
        }
        line_block = grouped.setdefault(entry["line"], {"line": entry["line"], "offers": []})
        line_block["offers"].append(offer)

    result: List[Dict[str, Any]] = []
    for line in sorted(grouped):
        block = grouped[line]
        offers = block["offers"]
        best_over = max(offers, key=lambda row: row["over_odds"])
        best_under = max(offers, key=lambda row: row["under_odds"])
        block["best_over"] = {"bookmaker": best_over["bookmaker"], "odds": best_over["over_odds"], "fair_prob": best_over["fair_over_prob"]}
        block["best_under"] = {"bookmaker": best_under["bookmaker"], "odds": best_under["under_odds"], "fair_prob": best_under["fair_under_prob"]}
        result.append(block)
    return result


def _btts_offers(event: Dict[str, Any]) -> List[Dict[str, Any]]:
    by_book: Dict[str, Dict[str, Any]] = {}
    for opt in extract_btts_odds(event):
        bookmaker = str(opt.get("bookmaker") or "")
        entry = by_book.setdefault(bookmaker, {"bookmaker": bookmaker})
        side = str(opt.get("side") or "").lower()
        if side == "yes":
            entry["yes_odds"] = float(opt["odds"])
        elif side == "no":
            entry["no_odds"] = float(opt["odds"])
    offers = []
    for entry in by_book.values():
        yes_odds = entry.get("yes_odds")
        no_odds = entry.get("no_odds")
        if yes_odds is None or no_odds is None:
            continue
        fair_yes, fair_no = remove_vig_two_way(float(yes_odds), float(no_odds))
        offers.append(
            {
                "bookmaker": entry["bookmaker"],
                "yes_odds": float(yes_odds),
                "no_odds": float(no_odds),
                "fair_yes_prob": fair_yes,
                "fair_no_prob": fair_no,
            }
        )
    return offers


def _moneyline_offers(event: Dict[str, Any]) -> List[Dict[str, Any]]:
    by_book: Dict[str, Dict[str, Any]] = {}
    for opt in extract_moneyline_odds(event):
        bookmaker = str(opt.get("bookmaker") or "")
        entry = by_book.setdefault(bookmaker, {"bookmaker": bookmaker})
        side = str(opt.get("side") or "").lower()
        entry[f"{side}_odds"] = float(opt["odds"])

    offers = []
    for entry in by_book.values():
        home_odds = entry.get("home_odds")
        draw_odds = entry.get("draw_odds")
        away_odds = entry.get("away_odds")
        if home_odds is None or draw_odds is None or away_odds is None:
            continue
        raw_home = implied_prob(float(home_odds))
        raw_draw = implied_prob(float(draw_odds))
        raw_away = implied_prob(float(away_odds))
        total = raw_home + raw_draw + raw_away
        if total <= 0:
            continue
        offers.append(
            {
                "bookmaker": entry["bookmaker"],
                "home_odds": float(home_odds),
                "draw_odds": float(draw_odds),
                "away_odds": float(away_odds),
                "fair_home_prob": raw_home / total,
                "fair_draw_prob": raw_draw / total,
                "fair_away_prob": raw_away / total,
            }
        )
    return offers


def summarize_live_odds(bookmakers: List[Dict[str, Any]], home_team: str, away_team: str, *, captured_at: Optional[str] = None) -> Dict[str, Any]:
    event = _build_event(bookmakers, home_team, away_team)
    return {
        "captured_at": captured_at or datetime.now(timezone.utc).isoformat(),
        "bookmakers_count": len(bookmakers),
        "totals": {
            "goals": _best_total_offers(extract_total_line_options(event, "goals")),
            "corners": _best_total_offers(extract_total_line_options(event, "corners")),
            "cards": _best_total_offers(extract_total_line_options(event, "cards")),
            "sot": _best_total_offers(extract_total_line_options(event, "sot")),
        },
        "btts": _btts_offers(event),
        "moneyline": _moneyline_offers(event),
    }


async def fetch_live_fixture_odds(
    fixture_id: int,
    home_team: str,
    away_team: str,
    *,
    session: Optional[Any] = None,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    now = time.time()
    cached = _live_odds_cache.get(int(fixture_id))
    if cached and (now - cached[1]) < _CACHE_TTL:
        logger.debug("Live odds cache hit fixture=%d", fixture_id)
        return cached[0], None

    source = "inplay"
    rows = await _async_api_get("/odds/live", {"fixture": str(fixture_id)}, session=session)
    if not rows:
        logger.info(
            "Live odds /odds/live empty for fixture=%d — falling back to pre-match /odds",
            fixture_id,
        )
        source = "prematch_fallback"
        rows = await _async_api_get("/odds", {"fixture": str(fixture_id)}, session=session)
        if not rows:
            logger.warning(
                "Live odds both endpoints empty for fixture=%d (no in-play, no pre-match)",
                fixture_id,
            )
            return None, "No live odds returned by API-Football."
    else:
        logger.info("Live odds /odds/live returned %d row(s) for fixture=%d", len(rows), fixture_id)

    row = rows[0] if rows else {}

    # In-play rows have no `bookmakers` wrapper — the bets live flat on
    # `row["odds"]`. Wrap them in a synthetic bookmaker (with bet IDs remapped
    # from in-play -> pre-match) so the existing transformer can consume them.
    if source == "inplay" and isinstance(row, dict) and "odds" in row and "bookmakers" not in row:
        api_bookmakers = _wrap_inplay_row_as_bookmaker(row)

        # One-shot diagnostic: log a sample of values from each remapped bet so
        # we can verify the value shape (e.g. "Over 2.5"/"Under 2.5" strings)
        # matches what core.odds_extraction expects.
        global _INPLAY_BETS_LOGGED
        if not _INPLAY_BETS_LOGGED and api_bookmakers:
            sample = []
            for bet in (api_bookmakers[0].get("bets") or [])[:6]:
                values = bet.get("values") or []
                sample.append({
                    "id": bet.get("id"),
                    "name": bet.get("name"),
                    "n_values": len(values),
                    "first_value": values[0] if values else None,
                })
            logger.warning(
                "INPLAY_REMAPPED_BETS fixture=%d remapped=%d sample=%s",
                fixture_id,
                len(api_bookmakers[0].get("bets") or []),
                sample,
            )
            _INPLAY_BETS_LOGGED = True
    else:
        api_bookmakers = row.get("bookmakers", []) if isinstance(row, dict) else []
    transformed = _transform_apifootball_odds(api_bookmakers, home_team, away_team)
    # _transform_apifootball_odds returns a list of bookmakers directly.
    transformed_list = transformed if isinstance(transformed, list) else []
    total_markets = sum(len(bm.get("markets") or []) for bm in transformed_list)
    logger.info(
        "Live odds fixture=%d source=%s api_bookmakers=%d transformed_bookmakers=%d markets=%d",
        fixture_id,
        source,
        len(api_bookmakers),
        len(transformed_list),
        total_markets,
    )
    if api_bookmakers and not total_markets:
        # Critical signal: API returned data, transformation dropped every market.
        logger.warning(
            "Live odds transformation produced zero markets fixture=%d source=%s "
            "(bet-id remap may be incomplete or value shape differs from pre-match)",
            fixture_id,
            source,
        )

    snapshot = summarize_live_odds(
        transformed,
        home_team,
        away_team,
        captured_at=datetime.now(timezone.utc).isoformat(),
    )
    # Tag snapshot with the source so downstream logs can distinguish
    # inplay vs prematch_fallback.
    if isinstance(snapshot, dict):
        snapshot["source"] = source
    _live_odds_cache[int(fixture_id)] = (snapshot, now)
    return snapshot, None


def best_total_offer(snapshot: Dict[str, Any], stat_group: str, line: float, side: str) -> Optional[Dict[str, Any]]:
    totals = (snapshot or {}).get("totals", {})
    blocks = totals.get(stat_group, []) if isinstance(totals, dict) else []
    for block in blocks:
        if abs(float(block.get("line", -999.0)) - float(line)) > 1e-9:
            continue
        offers = block.get("offers", [])
        if side == "over":
            key = "over_odds"
            fair_key = "fair_over_prob"
        else:
            key = "under_odds"
            fair_key = "fair_under_prob"
        valid = [offer for offer in offers if offer.get(key) is not None]
        if not valid:
            return None
        best = max(valid, key=lambda row: float(row[key]))
        return {
            "bookmaker": best["bookmaker"],
            "odds": float(best[key]),
            "fair_prob": float(best[fair_key]),
            "line": float(line),
            "side": side,
            "stat_group": stat_group,
        }
    return None


def best_btts_offer(snapshot: Dict[str, Any], side: str) -> Optional[Dict[str, Any]]:
    offers = (snapshot or {}).get("btts", [])
    key = "yes_odds" if side == "yes" else "no_odds"
    fair_key = "fair_yes_prob" if side == "yes" else "fair_no_prob"
    valid = [offer for offer in offers if offer.get(key) is not None]
    if not valid:
        return None
    best = max(valid, key=lambda row: float(row[key]))
    return {
        "bookmaker": best["bookmaker"],
        "odds": float(best[key]),
        "fair_prob": float(best[fair_key]),
        "side": side,
    }


def best_moneyline_offer(snapshot: Dict[str, Any], side: str) -> Optional[Dict[str, Any]]:
    offers = (snapshot or {}).get("moneyline", [])
    key = f"{side}_odds"
    fair_key = f"fair_{side}_prob"
    valid = [offer for offer in offers if offer.get(key) is not None]
    if not valid:
        return None
    best = max(valid, key=lambda row: float(row[key]))
    return {
        "bookmaker": best["bookmaker"],
        "odds": float(best[key]),
        "fair_prob": float(best[fair_key]),
        "side": side,
    }
