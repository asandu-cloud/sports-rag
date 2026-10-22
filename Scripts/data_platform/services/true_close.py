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

An OddsPapi error records nothing: the fixture's picks stay due and the next run retries them. Only after
``GIVE_UP`` is a fixture that still errors, or was never found in OddsPapi's lists, recorded as unavailable
with its reason. ``run(recapture=...)`` replaces the records of named fixtures (22 Oct: 129 picks recorded
from error answers on 21 Oct).
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
GIVE_UP = timedelta(days=3)  # after this, an erroring or unmatched fixture is recorded as unavailable
# OddsPapi's "No historical odds found" means a bookmaker has no prices for the fixture only once its
# history has settled; sooner it is retried (21 Oct: 8 fixtures failed 2.5 h after kickoff).
SETTLED = timedelta(hours=6)
SIDES = {"goals": ("over", "under"), "corners": ("over", "under"), "cards": ("over", "under"),
         "btts": ("yes", "no"), "moneyline": ("home", "draw", "away")}
METHODS = ("proportional", "power")


def week_window(kickoff):
    """OddsPapi fixture-list window for the kickoff's week, Monday to Monday, padded by a day each side."""
    monday = datetime(kickoff.year, kickoff.month, kickoff.day, tzinfo=timezone.utc) - timedelta(days=kickoff.weekday())
    return ((monday - timedelta(days=1)).strftime("%Y-%m-%dT00:00:00Z"),
            (monday + timedelta(days=8)).strftime("%Y-%m-%dT00:00:00Z"))


def kickoff_of(prediction):
    """The match's kickoff: the settled result's where there is one (a match can be moved after publication)."""
    result = (prediction.get("settlement") or {}).get("result") or {}
    return utc_datetime(result.get("kickoff") or prediction.get("kickoff"))


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

    def due(self, predictions, recapture=()):
        now = self.clock()
        out = []
        for p in predictions:
            kickoff = kickoff_of(p)
            done = (p.get("true_close") or {}).get("source") == SOURCE and str(p.get("fixture_id")) not in recapture
            if kickoff and not done and now >= kickoff + DUE_AFTER and p.get("market") in (*SIDES, "spreads"):
                out.append(p)
        return out

    def fixture_ids(self, predictions):
        """({fixture id: OddsPapi fixtureId}, {fixture ids whose list failed}): one list per competition and week."""
        groups = {}
        for p in predictions:
            if p.get("league") not in oddspapi.TOURNAMENTS:
                continue
            kickoff = kickoff_of(p)
            groups.setdefault((p["league"], week_window(kickoff)), {})[str(p["fixture_id"])] = {
                "fixture_id": str(p["fixture_id"]), "kickoff": kickoff, "home": p["home_team"], "away": p["away_team"]}
        mapping, failed = {}, set()
        for (league, (start, end)), fixtures in sorted(groups.items()):
            try:
                theirs = self.client.get("/v4/fixtures", tournamentId=oddspapi.TOURNAMENTS[league], **{"from": start, "to": end})
            except oddspapi.OddsPapiError:
                failed.update(fixtures)
                continue
            mapping.update(oddspapi.match_fixtures(list(fixtures.values()), theirs if isinstance(theirs, list) else []))
        return mapping, failed

    def histories(self, odds_id, picks, kickoff):
        """{slug: price history} for Pinnacle and the picks' own bookmakers (an exchange per picked outcome)."""
        slugs, outcomes = {"pinnacle"}, {}
        for p in picks:
            line = pick_line(p)
            slug = oddspapi.BOOKS.get(line["bookmaker"])
            place = oddspapi.locate(self.index, line) if slug in oddspapi.EXCHANGES else None
            if place:
                outcomes.setdefault(slug, set()).add(place[1])
            if slug:
                slugs.add(slug)
        return oddspapi.histories(self.client, odds_id, slugs, outcomes, missing_ok=self.clock() >= kickoff + SETTLED)

    def run(self, predictions, *, limit=None, recapture=()):
        """Capture every due pick (at most ``limit`` fixtures); returns counts by status.

        ``recapture`` names fixtures (our ids) whose existing records are replaced.
        """
        due = self.due(predictions, recapture)
        mapping, failed = self.fixture_ids(due)
        by_fixture = {}
        for p in due:
            by_fixture.setdefault(str(p["fixture_id"]), []).append(p)
        counts = {"due": len(due), "fixtures": len(by_fixture), "matched": 0, "final": 0, "unavailable": 0,
                  "unmatched": 0, "errors": 0, "conflicts": 0}
        for n, (fixture, picks) in enumerate(sorted(by_fixture.items())):
            if limit is not None and n >= limit:
                break
            kickoff = kickoff_of(picks[0])
            missing = {"source": SOURCE, "kickoff": kickoff.isoformat(), "status": "unavailable"}
            given_up, replace = self.clock() >= kickoff + GIVE_UP, fixture in recapture
            odds_id = mapping.get(fixture)
            if fixture in failed:
                counts["errors"] += len(picks)
                continue
            if odds_id is None:
                if not given_up:
                    counts["unmatched"] += len(picks)
                    continue
                for p in picks:
                    self._save(p, {**missing, "reason": "fixture_not_matched"}, counts, replace)
                continue
            counts["matched"] += 1
            failure = None
            try:
                bodies = self.histories(odds_id, picks, kickoff)
            except oddspapi.OddsPapiError as exc:
                if not given_up:
                    counts["errors"] += len(picks)
                    continue
                failure = {**missing, "reason": "oddspapi_error", "error": exc.code or exc.status}
            for p in picks:
                record = failure or true_close(p, bodies, self.index, kickoff)
                self._save(p, {**record, "oddspapi_fixture_id": odds_id}, counts, replace)
        return counts

    def _save(self, prediction, record, counts, replace):
        result = self._write(prediction["id"], {**record, "captured_at": self.clock().isoformat()}, replace=replace)
        counts["conflicts" if result == "conflict" else record["status"]] += 1

    def _write(self, prediction_id, record, *, replace=False, attempts=5):
        """Write one record; SQLite's "database is locked" (the measurement worker writes every minute) is retried."""
        for attempt in range(attempts):
            try:
                return self._write_once(prediction_id, record, replace)
            except OperationalError as exc:
                if "locked" not in str(exc) or attempt == attempts - 1:
                    raise
                time.sleep(1 + attempt)

    def _write_once(self, prediction_id, record, replace):
        with self.repo._factory() as session:
            row = session.get(Prediction, prediction_id)
            if row is None:
                return "conflict"
            extras = dict(row.extras or {})
            if not replace and (extras.get("true_close") or {}).get("source") == SOURCE:
                return "unchanged"
            extras["true_close"] = record
            changed = session.execute(update(Prediction).where(
                Prediction.id == row.id, Prediction.updated_at == row.updated_at,
            ).values(extras=extras).execution_options(synchronize_session=False))
            session.commit()
            return "applied" if changed.rowcount == 1 else "conflict"
