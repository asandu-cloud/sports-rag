"""Price confirmation before publishing (rework A5.3, tracker D40 (a), D42).

A market-value pick takes the best price any bookmaker showed in the read, but API-Football copies prices
with a delay (D15), and only about half of the best prices were still on offer at the read (D39; D41: 53%
of 538 checked picks). Before a read is stored, each recommended result's quote is checked against the
quoting bookmaker's own price in OddsPapi's free history, requested fresh (never through the client's
cache, so a pre-kickoff history cannot later be read as the true close):

* ``confirmed``: the bookmaker's price now is at least the quoted price (0.5% tolerance, as the price
  check); the quote is kept;
* ``repriced``: it is shorter but the value at it still clears the selection's threshold: the pick is
  published at the bookmaker's own price (owner, 22 Oct);
* ``not_confirmed``: anything else is a no-bet (fail closed): the price is gone, the line is suspended or
  not listed, OddsPapi has not seen the bookmaker within ``MAX_AGE``, the bookmaker or match is not in
  OddsPapi, the bookmaker is an exchange (commission not modelled yet), or OddsPapi answered an error.

OddsPapi records changes only and checks each bookmaker on its own schedule (22 Oct: Betano about every
10 minutes, with missed checks of 20-56 minutes), so a line's price is taken as current only when the
bookmaker was seen on any line of the match within ``MAX_AGE`` (``None``: no limit; the record keeps when it was
seen either way).

The evidence goes to the result's ``context["price_confirmation"]``. Only reads made under
``SPIX_SELECTION_POLICY=market_value`` (A3.7 stage 4) are checked. Fixture ids come from weekly fixture
lists (one billable request per competition and week) cached apart from the true close's, so a list
fetched before the matches never stands in for the one the true close fetches after them.
"""
from datetime import datetime, timedelta, timezone
import json
import os

from .. import oddspapi
from ..reference_price import devig_consensus, norm
from ..settlement import utc_datetime
from .match_read_compiler import _selection_policy
from .true_close import line_outcomes, week_window

try:
    from Scripts.rag_ingest.prob_models import asian_total_expected_value
except ImportError:  # pragma: no cover - the rag_ingest path layout
    from rag_ingest.prob_models import asian_total_expected_value  # type: ignore[import]

VERSION = "price-confirmation.v1"
TOLERANCE = 0.005  # as the price check (ops/price_check.py)
MAX_AGE = timedelta(minutes=60)  # owner, 28 Oct (D45); None would take the newest price however old, with its time
CACHE = oddspapi.CACHE / "confirmation"
REASONS = {
    "price_gone": "the bookmaker's price is now shorter and no longer has value",
    "suspended": "the bookmaker has suspended the line",
    "line_not_listed": "the bookmaker does not list the line",
    "not_recent": "OddsPapi has not checked the bookmaker recently",
    "no_history": "OddsPapi has no prices from the bookmaker for this match",
    "bookmaker_not_carried": "OddsPapi does not carry the bookmaker",
    "exchange": "exchange prices are not confirmed yet",
    "fixture_not_matched": "the match is not in OddsPapi's fixture list",
    "fixture_list_error": "OddsPapi's fixture list could not be read",
    "market_list_error": "OddsPapi's market list could not be read",
    "history_error": "OddsPapi's price history could not be read",
    "no_oddspapi_key": "no OddsPapi key is set",
}


class PriceConfirmation:
    """Checks one fixture's recommended results against the quoting bookmakers' own prices now."""

    def __init__(self, client, index, *, min_ev, max_age=MAX_AGE, clock=None):
        self.client, self.index, self.min_ev, self.max_age = client, index, min_ev, max_age
        self.clock = clock or (lambda: datetime.now(timezone.utc))

    def __call__(self, results):
        results = list(results)
        snapshots = [_as_dict(r) for r in results]
        picks = [i for i, s in enumerate(snapshots) if _recommended(s)]
        if _selection_policy(snapshots) != "market_value" or not picks:
            return results
        now = self.clock()
        records = self._check(snapshots, picks, now)
        return [_rebuild(r, _apply(snapshots[i], records[i])) if i in records else r for i, r in enumerate(results)]

    def _check(self, snapshots, picks, now):
        lines = {i: _line(snapshots[i]) for i in picks}
        records, slugs = {}, {}
        for i, line in lines.items():
            slug = oddspapi.BOOKS.get(line["bookmaker"])
            if not slug or slug in oddspapi.EXCHANGES:
                records[i] = _record(snapshots[i], now, slug, reason="exchange" if slug else "bookmaker_not_carried")
            else:
                slugs[i] = slug
        if not slugs:
            return records
        try:
            odds_id = self._fixture_id(snapshots[picks[0]]["fixture"])
        except (oddspapi.OddsPapiError, oddspapi.BudgetExceeded):
            return {**records, **{i: _record(snapshots[i], now, s, reason="fixture_list_error") for i, s in slugs.items()}}
        if odds_id is None:
            return {**records, **{i: _record(snapshots[i], now, s, reason="fixture_not_matched") for i, s in slugs.items()}}
        bodies, failed = self._histories(odds_id, set(slugs.values()))
        for i, slug in slugs.items():
            known = {"oddspapi_fixture_id": odds_id}
            records[i] = (_record(snapshots[i], now, slug, reason="history_error", **known) if slug in failed
                          else self._verdict(snapshots[i], lines[i], slug, bodies.get(slug), now, known))
        return records

    def _fixture_id(self, fixture):
        """OddsPapi's fixtureId for the match, from the cached weekly list of its competition, or None."""
        kickoff = utc_datetime(fixture.get("kickoff"))
        if fixture.get("league") not in oddspapi.TOURNAMENTS or kickoff is None:
            return None
        start, end = week_window(kickoff)
        theirs = self.client.get("/v4/fixtures", tournamentId=oddspapi.TOURNAMENTS[fixture["league"]],
                                 **{"from": start, "to": end})
        ours = {"fixture_id": "match", "kickoff": kickoff, "home": fixture.get("home_team"),
                "away": fixture.get("away_team")}
        return oddspapi.match_fixtures([ours], theirs if isinstance(theirs, list) else []).get("match")

    def _histories(self, odds_id, slugs):
        """({slug: history now}, {slugs whose request failed}): three bookmakers per request."""
        ordered, bodies, failed = sorted(slugs), {}, set()
        for group in (ordered[i:i + 3] for i in range(0, len(ordered), 3)):
            try:
                body = self.client.fresh("/v4/historical-odds", fixtureId=odds_id, bookmakers=",".join(group))
            except oddspapi.OddsPapiError as exc:
                if exc.code != "NOT_FOUND":  # "No historical odds found": none of these bookmakers priced it
                    failed.update(group)
                continue
            bodies.update({slug: body for slug in group})
        return bodies, failed

    def _verdict(self, snapshot, line, slug, body, now, known):
        seen = oddspapi.last_seen(body, slug)
        if seen is None:
            return _record(snapshot, now, slug, reason="no_history", **known)
        known["bookmaker_seen_at"] = seen.isoformat()
        if self.max_age is not None and now - seen > self.max_age:
            return _record(snapshot, now, slug, reason="not_recent", **known)
        points = self._points(body, slug, line, now)
        if not points:
            return _record(snapshot, now, slug, reason="line_not_listed", **known)
        when, own, active = points[-1]
        known["own_odds_since"] = when.isoformat()
        if not active or not own:
            return _record(snapshot, now, slug, reason="suspended", **known)
        known["own_odds"] = own
        decision = snapshot["decision"]
        if own >= decision["quote"]["odds"] * (1 - TOLERANCE):
            return _record(snapshot, now, slug, status="confirmed", **known)
        if _expected_value(decision, own) >= self.min_ev:
            return {**_record(snapshot, now, slug, status="repriced", **known),
                    "fair_probability": self._own_fair(snapshot, line, slug, body, now)}
        return _record(snapshot, now, slug, reason="price_gone", **known)

    def _points(self, body, slug, line, now):
        """[(time, price or None, active)] of the line's side at the bookmaker, up to ``now``."""
        place = oddspapi.locate(self.index, line)
        return [p for p in oddspapi.series(body, slug, *place) if p[0] <= now] if place else []

    def _own_fair(self, snapshot, line, slug, body, now):
        """The bookmaker's own fair probability for the pick now, from all its prices on the line, or None."""
        prices = []
        for side, point in line_outcomes(line):
            points = self._points(body, slug, {**line, "side": side, "point": point}, now)
            if not points or not points[-1][2] or not points[-1][1]:
                return None
            prices.append((side, point, points[-1][1]))
        switches = (snapshot.get("context") or {}).get("engine_switches") or {}
        _, consensus, _ = devig_consensus({line["bookmaker"]: prices}, line["side"], line["point"],
                                          spreads=line["market"] == "spreads",
                                          method=switches.get("SPIX_DEVIG_METHOD") or "proportional")
        return consensus["fair_probability"] if consensus else None


def confirm_prices(results, *, min_ev, clock=None):
    """The Match Read worker's confirmation: results pass unchanged unless made under ``market_value``.

    ``min_ev`` is the market-value selection's threshold (``line_selection.MARKET_VALUE_MIN_EV``): a shorter
    price is published only while its value still clears it.
    """
    results = list(results)
    snapshots = [_as_dict(r) for r in results]
    if _selection_policy(snapshots) != "market_value" or not any(_recommended(s) for s in snapshots):
        return results
    now = (clock or (lambda: datetime.now(timezone.utc)))()
    key = os.environ.get("ODDSPAPI_API_KEY")
    if not key:
        return _report(_fail(results, snapshots, now, "no_oddspapi_key"))
    try:
        client, index = _client(key)
    except (oddspapi.OddsPapiError, oddspapi.BudgetExceeded):
        return _report(_fail(results, snapshots, now, "market_list_error"))
    client.budget = client._usage()["billable"] + 2  # at most the market list and one fixture list per call
    return _report(PriceConfirmation(client, index, min_ev=min_ev, clock=lambda: now)(results))


def _report(results):
    """Counts the checked picks for the owner's chat (``ops_alerts``: most of a day unchecked is a fault)."""
    from . import ops_alerts
    records = [s["context"]["price_confirmation"] for s in map(_as_dict, results)
               if "price_confirmation" in (s.get("context") or {})]
    if records:
        ops_alerts.price_confirmations(records)
    return results


_shared = {}  # key -> (client, market index): one client per process, so OddsPapi's cooldown holds across fixtures


def _client(key):
    if key not in _shared:
        client = oddspapi.OddsPapiClient(key, CACHE, compress=True)
        client.budget = client._usage()["billable"] + 1
        _shared[key] = (client, oddspapi.market_index(client.get("/v4/markets")))
    return _shared[key]


def _fail(results, snapshots, now, reason):
    return [_rebuild(r, _apply(snapshots[i], _record(snapshots[i], now, oddspapi.BOOKS.get(_line(snapshots[i])["bookmaker"]),
                                                     reason=reason)))
            if _recommended(snapshots[i]) else r for i, r in enumerate(results)]


def _as_dict(result):
    to_dict = getattr(result, "to_dict", None)
    return json.loads(json.dumps(to_dict() if callable(to_dict) else dict(result)))


def _rebuild(original, snapshot):
    return type(original).from_dict(snapshot) if callable(getattr(original, "to_dict", None)) else snapshot


def _recommended(snapshot):
    return (snapshot.get("decision") or {}).get("status") == "recommended"


def _line(snapshot):
    """The result's pick as an ``oddspapi.locate`` line (as ``true_close.pick_line`` for a published pick)."""
    fixture, quote = snapshot.get("fixture") or {}, (snapshot.get("decision") or {}).get("quote") or {}
    return {"market": (snapshot.get("market") or {}).get("group"), "side": norm(quote.get("side")),
            "point": quote.get("line"), "home": fixture.get("home_team"), "away": fixture.get("away_team"),
            "bookmaker": quote.get("bookmaker")}


def _record(snapshot, now, slug, *, status="not_confirmed", reason=None, **found):
    quote = snapshot["decision"]["quote"]
    record = {"version": VERSION, "status": status, "bookmaker": quote.get("bookmaker"), "site": slug,
              "quoted_odds": quote.get("odds"), "checked_at": now.isoformat(), **found}
    return {**record, "reason": reason} if reason else record


def _expected_value(decision, odds):
    """Value per unit stake at ``odds``, on the decision's settlement profile (a half-win line) or win chance."""
    p = decision["model_probability"]
    profile = decision.get("settlement_profile") or {"full_win": p, "half_win": 0.0, "push": 0.0, "half_loss": 0.0,
                                                     "full_loss": 1.0 - p}
    return asian_total_expected_value(profile, odds)


def _apply(snapshot, record):
    """The result with the confirmation applied: kept, repriced at the bookmaker's own price, or no-bet."""
    out = {**snapshot, "context": {**(snapshot.get("context") or {}), "price_confirmation": {
        k: v for k, v in record.items() if k != "fair_probability"}}}
    decision = dict(snapshot["decision"])
    if record["status"] == "repriced":
        own, fair = record["own_odds"], record.get("fair_probability")
        decision.update(quote={**decision["quote"], "odds": own}, expected_value=_expected_value(decision, own),
                        implied_probability=fair,
                        value_edge=decision["model_probability"] - fair if fair is not None else None)
    elif record["status"] == "not_confirmed":
        decision.update(status="no_bet", reason=f"Price not confirmed: {REASONS[record['reason']]}.")
    out["decision"] = decision
    return out
