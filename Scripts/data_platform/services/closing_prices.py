"""Exact, prospective closing prices. Never use a post-kickoff odds request."""
from datetime import datetime, timedelta, timezone
import math

from sqlalchemy import update

from ..models import OddsSnapshot, Prediction
from ..settlement import MARKET_KEYS, provider_id, utc_datetime
from ..closing_contract import WINDOW_MINUTES, MAX_QUOTE_AGE_MINUTES, CLV_DEFINITION


def exact_price(prediction, odds_row, *, kickoff, captured_at):
    """Retain exact raw-provider quote evidence; no bookmaker/market substitutes."""
    selection = (prediction.get("tracking") or {}).get("selection") or {}
    key, group = selection.get("market_key"), prediction.get("market")
    quote_time = utc_datetime(odds_row.get("update"))
    if provider_id((odds_row.get("fixture") or {}).get("id")) != provider_id(prediction.get("fixture_id")):
        return None, "fixture_identity_mismatch"
    if selection.get("period") != "regulation_time" or selection.get("participant") is not None:
        return None, "unsupported_period_or_participant"
    if key not in MARKET_KEYS.get(group, set()):
        return None, "unsupported_market"
    if quote_time is None:
        return None, "quote_time_unavailable"
    if not quote_time <= captured_at < kickoff:
        return None, "invalid_quote_chronology"
    if not 0 < (kickoff - quote_time).total_seconds() <= MAX_QUOTE_AGE_MINUTES * 60:
        return None, "stale_quote"
    if not 0 < (kickoff - captured_at).total_seconds() <= WINDOW_MINUTES * 60:
        return None, "outside_capture_window"
    if selection.get("fixture_id") != prediction.get("fixture_id") or any(
        selection.get(field) != prediction.get(field) for field in ("side", "line", "bookmaker")
    ) or selection.get("market_group") != group:
        return None, "conflicting_selection_identity"
    # Reuse precisely the market/value normalization that produced the original
    # published quote, without its preferred-bookmaker selection/fallback.
    try:
        from Scripts.rag_ingest.odds_provider import _BET_ID_TO_MARKET_KEY, _transform_apifootball_odds
    except ImportError:
        from rag_ingest.odds_provider import _BET_ID_TO_MARKET_KEY, _transform_apifootball_odds
    books = [book for book in odds_row.get("bookmakers", [])
             if str(book.get("name", "")).strip().casefold() == str(selection.get("bookmaker", "")).strip().casefold()]
    if len(books) != 1:
        return None, "bookmaker_unavailable_or_ambiguous"
    matches = []
    side = str(selection.get("side", "")).lower()
    for bet in books[0].get("bets", []):
        if _BET_ID_TO_MARKET_KEY.get(bet.get("id")) != key:
            continue
        for raw in bet.get("values", []):
            transformed = _transform_apifootball_odds([
                {"name": books[0]["name"], "bets": [{"id": bet["id"], "values": [raw]}]}
            ], "home", "away")
            if not transformed:
                continue
            outcome = transformed[0]["markets"][0]["outcomes"][0]
            name = str(outcome["name"]).lower().replace(":", "-")
            expected = side.replace(":", "-")
            line = selection.get("line")
            if name != expected or outcome.get("point") != line:
                continue
            price = float(outcome["price"])
            if not math.isfinite(price) or price <= 1:
                continue
            matches.append({"odds": price, "bookmaker_id": books[0].get("id"),
                            "bookmaker": books[0]["name"], "provider_market_id": bet["id"],
                            "market_key": key, "period": "regulation_time", "side": side, "line": line,
                            "raw_value": raw.get("value"), "quote_time": quote_time.isoformat(),
                            "quote_time_source": "api_football_odds_row.update",
                            "captured_at": captured_at.isoformat(), "kickoff": kickoff.isoformat(),
                            "selection_key": (prediction.get("tracking") or {}).get("selection_key")})
    return (matches[0], None) if len(matches) == 1 else (None, "exact_quote_unavailable_or_ambiguous")


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
        expected_league = COMPETITIONS.get(predictions[0]["league"])
        if expected_league is None or provider_id(fixtures[0].get("league", {}).get("id")) != str(expected_league.api_football_id):
            raise ValueError("Closing competition response mismatch")
        fixture = fixtures[0]["fixture"]
        kickoff = utc_datetime(fixture.get("date"))
        status = (fixture.get("status") or {}).get("short")
        if kickoff is None:
            return {"pending_reason": "missing_fixture_kickoff", "captured": 0}
        now = self.clock()
        if now >= kickoff:
            if status in {"NS", "TBD", "PST", "CANC", "ABD", "AWD", "WO"}:
                return {"pending_reason": "kickoff_or_status_unverified", "captured": 0}
            if status not in {"1H", "HT", "2H", "ET", "BT", "P", "FT", "AET", "PEN"}:
                return {"pending_reason": "kickoff_or_status_unverified", "captured": 0}
            finalized = 0
            unavailable = 0
            for pred in predictions:
                self.guard()
                candidate = pred.get("closing_capture") or {}
                if candidate.get("status") == "final":
                    continue
                captured, quoted = utc_datetime(candidate.get("captured_at")), utc_datetime(candidate.get("quote_time"))
                valid = (captured is not None and quoted is not None and quoted <= captured < kickoff
                         and utc_datetime(candidate.get("kickoff")) == kickoff
                         and (kickoff - quoted).total_seconds() <= MAX_QUOTE_AGE_MINUTES * 60
                         and (kickoff - captured).total_seconds() <= WINDOW_MINUTES * 60
                         and candidate.get("selection_key") == (pred.get("tracking") or {}).get("selection_key"))
                if not valid:
                    unavailable += 1
                    continue
                if not dry_run:
                    self._finalize(pred, candidate, self.clock())
                finalized += 1
            return {"captured": 0, "finalized": finalized, "unavailable": unavailable,
                    "finished": True, "pending_reason": "no_eligible_pre_kickoff_quote" if unavailable else None}
        if status != "NS" or (kickoff - now).total_seconds() > WINDOW_MINUTES * 60:
            return {"captured": 0, "pending_reason": "outside_capture_window_or_started"}
        payload = self.client._get("/odds", {"fixture": int(fid)})
        captured_at = self.clock()  # RESPONSE time, never request-start time.
        self.guard()
        if captured_at >= kickoff:
            return {"captured": 0, "pending_reason": "response_after_kickoff"}
        rows = payload["response"]
        if len(rows) != 1:
            return {"captured": 0, "pending_reason": "missing_or_ambiguous_odds_row"}
        # The odds row also carries the fixture date. Reject reschedule mismatch.
        if utc_datetime((rows[0].get("fixture") or {}).get("date")) != kickoff:
            return {"captured": 0, "pending_reason": "odds_kickoff_mismatch"}
        if provider_id((rows[0].get("league") or {}).get("id")) != str(expected_league.api_football_id):
            return {"captured": 0, "pending_reason": "odds_competition_mismatch"}
        if provider_id((rows[0].get("fixture") or {}).get("id")) != fid:
            return {"captured": 0, "pending_reason": "fixture_identity_mismatch"}
        quotes, reasons = [], {}
        for pred in predictions:
            quote, reason = exact_price(pred, rows[0], kickoff=kickoff, captured_at=captured_at)
            if quote is not None:
                quotes.append((pred, quote))
            else:
                reasons[reason] = reasons.get(reason, 0) + 1
        saved = 0
        snapshot_id = None
        if not dry_run:
            self.guard()
            with self.repo._factory() as session:
                snapshot = OddsSnapshot(event_id=fid, league=predictions[0]["league"], kind="closing",
                                        kickoff_utc=kickoff, kickoff_date=kickoff.date().isoformat(),
                                        captured_at=captured_at, odds_json=payload)
                session.add(snapshot)
                session.flush()
                snapshot_id = snapshot.id
                for pred, quote in quotes:
                    row = session.get(Prediction, pred["id"])
                    extras = dict(row.extras or {})
                    previous = extras.get("closing_capture") or {}
                    if previous.get("status") == "final":
                        continue
                    previous_time = utc_datetime(previous.get("quote_time"))
                    if previous_time and previous_time > utc_datetime(quote["quote_time"]):
                        continue
                    extras["closing_capture"] = {**quote, "status": "provisional", "snapshot_id": snapshot.id,
                                                 "clv_definition": CLV_DEFINITION}
                    changed = session.execute(update(Prediction).where(Prediction.id == row.id,
                        Prediction.updated_at == row.updated_at).values(extras=extras).execution_options(synchronize_session=False))
                    if changed.rowcount != 1:
                        raise RuntimeError("Closing write conflicted; retry next tick")
                    saved += 1
                self.guard()
        return {"captured": len(quotes) if dry_run else saved, "snapshot_id": snapshot_id, "pending_reasons": reasons,
                "pending_reason": next(iter(reasons), None), "kickoff": kickoff.isoformat()}

    def _finalize(self, pred, candidate, now):
        with self.repo._factory() as session:
            row = session.get(Prediction, pred["id"])
            extras = dict(row.extras or {})
            if extras.get("closing_capture") != candidate:
                raise RuntimeError("Closing evidence changed during finalization")
            opening, closing = float(row.odds), float(candidate["odds"])
            if not all(math.isfinite(value) and value > 1 for value in (opening, closing)):
                raise ValueError("Invalid recorded decimal odds")
            extras["closing_capture"] = {**candidate, "status": "final", "finalized_at": now.isoformat()}
            result = session.execute(update(Prediction).where(Prediction.id == row.id,
                Prediction.updated_at == row.updated_at).values(
                    closing_odds=closing, closing_implied_prob=1 / closing,
                    clv=opening / closing - 1, extras=extras,
                ).execution_options(synchronize_session=False))
            if result.rowcount != 1:
                raise RuntimeError("Closing finalization conflicted; retry next tick")
            self.guard()
