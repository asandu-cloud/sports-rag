"""Closing prices from API-Football's final pre-kickoff odds snapshot.

API-Football refreshes pre-match odds periodically and keeps a fixture's last
refresh for several days after the match (measured 15 Oct 2026: the last
refresh lands a median ~59 minutes before kickoff). The true kickoff price is
not available from this provider, so the closing price is that final snapshot,
fetched once the match is over and stored with its own timestamp:

* accepted only if the snapshot is stamped strictly before kickoff;
* the same bookmaker's exact selection gives ``closing_odds`` and the CLV;
* every bookmaker pricing the full market gives a consensus fair probability
  (proportional margin removal, median across books), kept with the per-book
  prices so later weighting can recompute it.
"""
from datetime import datetime, timedelta, timezone
import gzip
import json
import math
from pathlib import Path

from sqlalchemy import update

from ..models import OddsSnapshot, Prediction
from ..settlement import MARKET_KEYS, handicap_side, provider_id, utc_datetime
from ..closing_contract import CLV_DEFINITION, FINAL_SNAPSHOT
from ..reference_price import devig_consensus, norm

SOURCE = FINAL_SNAPSHOT
DUE_AFTER = timedelta(hours=2)        # fetch once the match is surely over
GIVE_UP_AFTER = timedelta(days=7)     # API-Football has dropped the odds by then
NOT_PLAYED = {"PST", "CANC", "ABD", "AWD", "WO"}
STARTED = {"1H", "HT", "2H", "ET", "BT", "P", "FT", "AET", "PEN", "LIVE", "INT", "SUSP"}


def _transform():
    try:
        from Scripts.rag_ingest.odds_provider import _BET_ID_TO_MARKET_KEY, _transform_apifootball_odds
    except ImportError:
        from rag_ingest.odds_provider import _BET_ID_TO_MARKET_KEY, _transform_apifootball_odds
    return _BET_ID_TO_MARKET_KEY, _transform_apifootball_odds


def book_outcomes(bookmaker, market_key, home, away):
    """[(side, point, price)] for one bookmaker and market, normalised like the published quotes."""
    bet_keys, transform = _transform()
    out = []
    for bet in bookmaker.get("bets", []):
        if bet_keys.get(bet.get("id")) != market_key:
            continue
        for market in (transform([{"name": bookmaker.get("name"), "bets": [bet]}], home, away) or [{}])[0].get("markets", []):
            for outcome in market.get("outcomes", []):
                price = outcome.get("price")
                if isinstance(price, (int, float)) and math.isfinite(price) and price > 1:
                    out.append((norm(outcome.get("name")), outcome.get("point"), float(price)))
    return out


def price_selection(prediction, odds_row):
    """(same-book price or None, consensus dict, issue) for one pick from one snapshot row."""
    selection = (prediction.get("tracking") or {}).get("selection") or {}
    if selection.get("period") != "regulation_time" or selection.get("participant") is not None:
        return None, None, "unsupported_period_or_participant"
    key = selection.get("market_key")
    if key not in MARKET_KEYS.get(prediction.get("market"), set()):
        return None, None, "unsupported_market"
    spreads = prediction.get("market") == "spreads"
    if spreads:  # handicap sides are team names; transform with our own names to match them
        home, away = prediction.get("home_team"), prediction.get("away_team")
        side = {"home": home, "away": away}.get(handicap_side(prediction))
    else:
        home, away, side = "home", "away", selection.get("side")
    if not norm(side):
        return None, None, "invalid_side"
    books = {b.get("name"): book_outcomes(b, key, home, away) for b in odds_row.get("bookmakers", [])}
    same, consensus, issue = devig_consensus(books, side, selection.get("line"), spreads=spreads,
                                             wanted_book=selection.get("bookmaker"))
    if same:
        bookmaker = next(b for b in odds_row["bookmakers"] if b.get("name") == same["bookmaker"])
        same = {"odds": same["odds"], "bookmaker": same["bookmaker"], "bookmaker_id": bookmaker.get("id")}
    return same, consensus, issue


class ClosingPriceService:
    def __init__(self, repo, *, client, guard, clock=None):
        self.repo, self.client, self.guard = repo, client, guard
        self.clock = clock or (lambda: datetime.now(timezone.utc))

    def process(self, predictions, *, dry_run=False):
        fid = provider_id(predictions[0]["fixture_id"])
        self.guard()
        fixtures = self.client.fixture(int(fid))
        if len(fixtures) != 1 or provider_id(fixtures[0].get("fixture", {}).get("id")) != fid:
            raise ValueError("Closing fixture response mismatch")
        from ..sync.apifootball import COMPETITIONS
        league = COMPETITIONS.get(predictions[0]["league"])
        league_id = str(league.api_football_id) if league else None
        if league_id is None or provider_id(fixtures[0].get("league", {}).get("id")) != league_id:
            raise ValueError("Closing competition response mismatch")
        fixture = fixtures[0]["fixture"]
        kickoff, status = utc_datetime(fixture.get("date")), (fixture.get("status") or {}).get("short")
        now = self.clock()
        if kickoff is None:
            return {"captured": 0, "pending_reason": "missing_fixture_kickoff"}
        if status in NOT_PLAYED:
            return {"captured": 0, "pending_reason": "fixture_not_played", "finished": True}
        if status not in STARTED or now < kickoff + DUE_AFTER:
            return {"captured": 0, "pending_reason": "not_due"}
        payload = self.client._get("/odds", {"fixture": int(fid)})
        captured_at = self.clock()
        self.guard()
        rows = payload.get("response") or []
        if len(rows) != 1:
            reason = "odds_unavailable" if not rows else "ambiguous_odds_rows"
            return {"captured": 0, "pending_reason": reason, "finished": now > kickoff + GIVE_UP_AFTER}
        row = rows[0]
        if provider_id((row.get("fixture") or {}).get("id")) != fid:
            return {"captured": 0, "pending_reason": "fixture_identity_mismatch", "finished": True}
        if provider_id((row.get("league") or {}).get("id")) != league_id:
            return {"captured": 0, "pending_reason": "odds_competition_mismatch", "finished": True}
        if utc_datetime((row.get("fixture") or {}).get("date")) != kickoff:
            return {"captured": 0, "pending_reason": "odds_kickoff_mismatch", "finished": True}
        quote_time = utc_datetime(row.get("update"))
        if quote_time is None:
            return {"captured": 0, "pending_reason": "quote_time_unavailable", "finished": True}
        if quote_time >= kickoff:
            return {"captured": 0, "pending_reason": "snapshot_after_kickoff", "finished": True}

        captures, reasons = [], {}
        for pred in predictions:
            if (pred.get("closing_capture") or {}).get("status") == "final":
                continue
            same_book, consensus, issue = price_selection(pred, row)
            if issue:
                reasons[issue] = reasons.get(issue, 0) + 1
            if same_book or consensus:
                captures.append((pred, same_book, consensus))
        if dry_run:
            return {"captured": len(captures), "finished": True, "pending_reasons": reasons,
                    "pending_reason": next(iter(reasons), None)}
        saved = 0
        with self.repo._factory() as session:
            snapshot = OddsSnapshot(event_id=fid, league=predictions[0]["league"], kind="final_prematch",
                                    kickoff_utc=kickoff, kickoff_date=kickoff.date().isoformat(),
                                    captured_at=captured_at, odds_json=payload)
            session.add(snapshot)
            session.flush()
            for pred, same_book, consensus in captures:
                self.guard()
                record = session.get(Prediction, pred["id"])
                extras = dict(record.extras or {})
                if (extras.get("closing_capture") or {}).get("status") == "final":
                    continue
                extras["closing_capture"] = {
                    "status": "final", "source": SOURCE, "clv_definition": CLV_DEFINITION,
                    "quote_time": quote_time.isoformat(), "quote_time_source": "api_football_odds_row.update",
                    "minutes_before_kickoff": round((kickoff - quote_time).total_seconds() / 60, 2),
                    "captured_at": captured_at.isoformat(), "finalized_at": captured_at.isoformat(),
                    "kickoff": kickoff.isoformat(), "snapshot_id": snapshot.id,
                    "selection_key": (pred.get("tracking") or {}).get("selection_key"),
                    **(same_book or {"odds": None}), "consensus": consensus,
                }
                values = {"extras": extras}
                opening = float(record.odds) if record.odds is not None else None
                if same_book and opening and math.isfinite(opening) and opening > 1:
                    closing_odds = same_book["odds"]
                    values.update(closing_odds=closing_odds, closing_implied_prob=1 / closing_odds,
                                  clv=opening / closing_odds - 1)
                changed = session.execute(update(Prediction).where(
                    Prediction.id == record.id, Prediction.updated_at == record.updated_at,
                ).values(**values).execution_options(synchronize_session=False))
                if changed.rowcount != 1:
                    raise RuntimeError("Closing write conflicted; retry next tick")
                saved += 1
            self.guard()
            session.commit()
        return {"captured": saved, "finished": True, "snapshot_id": snapshot.id, "kickoff": kickoff.isoformat(),
                "minutes_before_kickoff": round((kickoff - quote_time).total_seconds() / 60, 2),
                "pending_reasons": reasons, "pending_reason": next(iter(reasons), None)}


class SavedOddsClient:
    """Serve ``/odds?fixture=`` from post-match odds saved to disk; everything else goes to ``client``."""

    def __init__(self, client, directory):
        self.client, self.directory = client, Path(directory)

    def _get(self, path, params):
        fixture = (params or {}).get("fixture")
        saved = self.directory / f"{fixture}.json.gz"
        if path == "/odds" and fixture is not None and saved.exists():
            with gzip.open(saved, "rt", encoding="utf-8") as handle:
                return json.load(handle)["body"]
        return self.client._get(path, params)

    def __getattr__(self, name):
        return getattr(self.client, name)
