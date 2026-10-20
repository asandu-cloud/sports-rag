"""True closing prices from OddsPapi's timestamped history (rework A1 step 4, tracker D18).

API-Football's final pre-kickoff snapshot (``closing_prices.py``) lands a median ~60 minutes before kickoff,
and its copies of bookmakers' prices lag (D15: Pinnacle a median 130 minutes in the last three hours).
OddsPapi's free historical odds give every price change with its time, so a bookmaker's true close is its
last active price before kickoff. For each published pick, once the match is over:

* ``consensus``: Pinnacle's last pre-kickoff prices for the pick's line with the margin removed
  (proportional, as ``reference_price`` today, and power for A3.7 stage 2), stored with the prices so the
  probability can be recomputed;
* ``same_book``: the pick's own bookmaker's last pre-kickoff price for the selection, at the country site
  API-Football copies (``oddspapi.BOOKS``); absent where OddsPapi does not carry the bookmaker (BetVictor).

Both go to ``extras["true_close"]``. API-Football's ``closing_capture`` is untouched and stays the fallback
where a fixture cannot be matched (D18). Fixture ids cost one billable request per competition and week
(cached); price histories are free.
"""
from datetime import datetime, timedelta, timezone
import time

from sqlalchemy import update
from sqlalchemy.exc import OperationalError

from .. import oddspapi
from ..models import Prediction
from ..reference_price import METHOD_VERSIONS, devig_consensus, norm
from ..settlement import utc_datetime

# v2 (20 Oct): a line withdrawn before kickoff has no close (v1 fell back to its last active price).
SOURCE = "oddspapi_true_close.v2"
DUE_AFTER = timedelta(hours=2)
SIDES = {"goals": ("over", "under"), "corners": ("over", "under"), "cards": ("over", "under"),
         "btts": ("yes", "no"), "moneyline": ("home", "draw", "away")}
METHODS = ("proportional", "power")


def week_window(kickoff):
    """OddsPapi fixture-list window for the kickoff's week, Monday to Monday, padded by a day each side."""
    monday = datetime(kickoff.year, kickoff.month, kickoff.day, tzinfo=timezone.utc) - timedelta(days=kickoff.weekday())
    return ((monday - timedelta(days=1)).strftime("%Y-%m-%dT00:00:00Z"),
            (monday + timedelta(days=8)).strftime("%Y-%m-%dT00:00:00Z"))


def pick_line(prediction):
    """The pick as an ``oddspapi.locate`` line: market, normalised side, point, team names."""
    selection = (prediction.get("tracking") or {}).get("selection") or {}
    return {"market": prediction.get("market"), "side": norm(selection.get("side")), "point": selection.get("line"),
            "home": prediction.get("home_team"), "away": prediction.get("away_team"),
            "bookmaker": selection.get("bookmaker") or prediction.get("bookmaker")}


def line_outcomes(line):
    """Every outcome of the pick's line, as (side, point) pairs in our normalised form."""
    market = line["market"]
    if market == "spreads":
        if line["point"] is None:
            return []
        home, away = norm(line["home"]), norm(line["away"])
        home_line = float(line["point"]) if line["side"] == home else -float(line["point"])
        return [(home, home_line), (away, -home_line)]
    return [(side, None if market in ("btts", "moneyline") else line["point"]) for side in SIDES.get(market, ())]


def close_from_history(body, slug, index, line, kickoff):
    """[(side, point, price, time)] of the line's last active pre-kickoff prices at one bookmaker, or None."""
    out = []
    for side, point in line_outcomes(line):
        place = oddspapi.locate(index, {**line, "side": side, "point": point})
        last = oddspapi.last_before(oddspapi.series(body, slug, *place), kickoff) if place else None
        if last is None:
            return None
        out.append((side, point, last[1], last[0]))
    return out or None


def true_close(prediction, bodies, index, kickoff):
    """The ``true_close`` record for one pick from Pinnacle's and its own bookmaker's histories."""
    line = pick_line(prediction)
    spreads = line["market"] == "spreads"
    record = {"source": SOURCE, "kickoff": kickoff.isoformat()}
    pinnacle = close_from_history(bodies.get("pinnacle"), "pinnacle", index, line, kickoff)
    if pinnacle:
        books = {"Pinnacle": [(s, p, price) for s, p, price, _ in pinnacle]}
        quote_time = max(t for *_, t in pinnacle)
        record["consensus"] = {"bookmaker": "Pinnacle", "prices": [[s, p, price] for s, p, price, _ in pinnacle],
                               "quote_time": quote_time.isoformat(),
                               "minutes_before_kickoff": round((kickoff - quote_time).total_seconds() / 60, 2)}
        for method in METHODS:
            _, consensus, _ = devig_consensus(books, line["side"], line["point"], spreads=spreads, method=method)
            if consensus:
                record["consensus"][f"fair_probability_{method}"] = consensus["fair_probability"]
                record["consensus"][f"method_{method}"] = METHOD_VERSIONS[method]
    slug = oddspapi.BOOKS.get(line["bookmaker"])
    place = oddspapi.locate(index, line) if slug else None
    own = oddspapi.last_before(oddspapi.series(bodies.get(slug), slug, *place), kickoff) if place else None
    if own:
        record["same_book"] = {"bookmaker": line["bookmaker"], "site": slug, "odds": own[1],
                               "quote_time": own[0].isoformat(),
                               "minutes_before_kickoff": round((kickoff - own[0]).total_seconds() / 60, 2)}
    found = [k for k in ("consensus", "same_book") if k in record]
    record["status"] = "final" if found else "unavailable"
    if not found:
        record["reason"] = "line_not_in_history"
    return record


class TrueCloseService:
    """Writes ``extras["true_close"]`` for published picks whose match is over (OddsPapi history)."""

    def __init__(self, repo, client, index, *, clock=None):
        self.repo, self.client, self.index = repo, client, index
        self.clock = clock or (lambda: datetime.now(timezone.utc))

    def due(self, predictions):
        now = self.clock()
        out = []
        for p in predictions:
            kickoff = utc_datetime(p.get("kickoff"))
            done = (p.get("true_close") or {}).get("source") == SOURCE
            if kickoff and not done and now >= kickoff + DUE_AFTER and p.get("market") in (*SIDES, "spreads"):
                out.append(p)
        return out

    def fixture_ids(self, predictions):
        """{fixture id: OddsPapi fixtureId}: one fixture list per competition and week."""
        groups = {}
        for p in predictions:
            if p.get("league") not in oddspapi.TOURNAMENTS:
                continue
            kickoff = utc_datetime(p["kickoff"])
            groups.setdefault((p["league"], week_window(kickoff)), {})[str(p["fixture_id"])] = {
                "fixture_id": str(p["fixture_id"]), "kickoff": kickoff, "home": p["home_team"], "away": p["away_team"]}
        mapping = {}
        for (league, (start, end)), fixtures in sorted(groups.items()):
            theirs = self.client.get("/v4/fixtures", tournamentId=oddspapi.TOURNAMENTS[league], **{"from": start, "to": end})
            mapping.update(oddspapi.match_fixtures(list(fixtures.values()), theirs if isinstance(theirs, list) else []))
        return mapping

    def run(self, predictions, *, limit=None):
        """Capture every due pick (at most ``limit`` fixtures); returns counts by status."""
        due = self.due(predictions)
        mapping = self.fixture_ids(due)
        by_fixture = {}
        for p in due:
            by_fixture.setdefault(str(p["fixture_id"]), []).append(p)
        counts = {"due": len(due), "fixtures": len(by_fixture), "matched": 0, "final": 0, "unavailable": 0,
                  "unmatched": 0, "conflicts": 0}
        for n, (fixture, picks) in enumerate(sorted(by_fixture.items())):
            if limit is not None and n >= limit:
                break
            odds_id = mapping.get(fixture)
            if odds_id is None:
                counts["unmatched"] += len(picks)
                continue
            counts["matched"] += 1
            kickoff = utc_datetime(picks[0]["kickoff"])
            slugs = {"pinnacle"} | {oddspapi.BOOKS[b] for b in (pick_line(p)["bookmaker"] for p in picks) if b in oddspapi.BOOKS}
            bodies = {}
            others = sorted(slugs - {"pinnacle"})
            for group in [["pinnacle"]] + [others[i:i + 3] for i in range(0, len(others), 3)]:
                body = self.client.get("/v4/historical-odds", fixtureId=odds_id, bookmakers=",".join(group))
                bodies.update({slug: body if isinstance(body, dict) else {} for slug in group})
            for p in picks:
                record = {**true_close(p, bodies, self.index, kickoff), "captured_at": self.clock().isoformat(),
                          "oddspapi_fixture_id": odds_id}
                result = self._write(p["id"], record)
                counts["conflicts" if result == "conflict" else record["status"]] += 1
        return counts

    def _write(self, prediction_id, record, *, attempts=5):
        """Write one record; SQLite's "database is locked" (the measurement worker writes every minute) is retried."""
        for attempt in range(attempts):
            try:
                return self._write_once(prediction_id, record)
            except OperationalError as exc:
                if "locked" not in str(exc) or attempt == attempts - 1:
                    raise
                time.sleep(1 + attempt)

    def _write_once(self, prediction_id, record):
        with self.repo._factory() as session:
            row = session.get(Prediction, prediction_id)
            if row is None:
                return "conflict"
            extras = dict(row.extras or {})
            if (extras.get("true_close") or {}).get("source") == SOURCE:
                return "unchanged"
            extras["true_close"] = record
            changed = session.execute(update(Prediction).where(
                Prediction.id == row.id, Prediction.updated_at == row.updated_at,
            ).values(extras=extras).execution_options(synchronize_session=False))
            session.commit()
            return "applied" if changed.rowcount == 1 else "conflict"
