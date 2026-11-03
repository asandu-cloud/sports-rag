"""A5.4 replay (D58, D61): what cards priced from Pinnacle's fair price would have published on the stored reads.

For every stored read (latest before kickoff, per fixture and stage), the cards selector runs again on the prices
the decision audit stored, as ``market_service`` calls it under ``SPIX_CARDS_REFERENCE=pinnacle`` with stage 4's
switches: each cards line at Pinnacle's own fair price (power margin removal), a pick where another bookmaker's
price has at least 2% value against it (``line_selection.MARKET_VALUE_MIN_EV``), none on a line Pinnacle does not
price on both sides, at most one per match. A read whose guardrail blocked cards for any reason but the referee
publishes nothing, as live. The three-picks-per-match limit (the compiler's) is not applied, so a cards pick another
market's pick would displace is still counted.

Each pick is graded on the bookmakers' card count (``calibration_data.fixture_outcomes``: yellow 1, red 2, at most 3
per player) and valued at Pinnacle's closing price: Pinnacle's prices in API-Football's final pre-kickoff snapshot,
margin removed (a lineup read is often made after that snapshot, so only pre-match reads are valued at the close).
Results are split by side, since Pinnacle also overstated card overs on 9-22 Oct (Findings, 24 Oct).

With ``--pinnacle-then``, Pinnacle's own OddsPapi price at the read says whether price confirmation's Pinnacle check
(D61) would have kept the pick, and its last price before kickoff gives the value at the true close (D16): only
fixture lists and histories already downloaded are read (no billable request);
a pick whose history is not downloaded is counted as unknown. The key is read from ``.env`` and never printed. The
stored reads are copied behind the refresh gate; the replay runs after the gate is released.

    python -m Scripts.ops.cards_reference_replay --out Research/cards-reference-<date>
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path

from Scripts.data_platform import disagreement as dis, oddspapi
from Scripts.data_platform.calibration_data import _time, grade as grade_selection
from Scripts.data_platform.reference_price import devig_consensus, norm
from Scripts.data_platform.services import price_confirmation as pcf, true_close
from Scripts.ops import confidence_age_study as age, stage4_replay as s4
from Scripts.ops.odds_archive import ROOT

MARKET_KEY = "totals_cards_over_under"
REFERENCE = ("Pinnacle",)
REFEREE_REASON = "profiled referee"  # prediction_guardrails' referee rule, skipped for these picks (D61)
SIDES = ("over", "under")


def _publishable(result):
    """Live, and no guardrail reason but the referee's (the profile guardrails still apply)."""
    context = result.get("context") or {}
    guardrail = (context.get("data_quality") or {}).get("recommendation_guardrail") or {}
    others = [r for r in guardrail.get("reasons") or [] if REFEREE_REASON not in str(r)]
    eligible = guardrail.get("eligible") is True or (guardrail.get("eligible") is False and not others)
    return eligible and (context.get("release_policy") or {}).get("mode") == "live"


def replay_read(data, *, threshold=0.02, books="all"):
    """[pick] (at most one) that cards from Pinnacle's fair price would recommend in one stored read."""
    line_selection, _, decision_from_selector = s4._engine()
    result = next((r for r in data.get("canonical_results", []) if (r.get("market") or {}).get("group") == "cards"),
                  None)
    if result is None or not _publishable(result):
        return []
    options = deepcopy(((result.get("context") or {}).get("decision_audit") or {}).get("input_options") or [])
    projection = result.get("projection") or {}
    if not options or projection.get("value") is None:  # live does not run the selector without a projection
        return []
    with s4.bundle(threshold, books):
        selected = line_selection.select_best_total_recommendation(
            options, projection["value"], projection.get("variance"), anchor_points="all", anchor_books=REFERENCE)
        decision = decision_from_selector(selected)
    if decision is None or not decision.is_recommended or decision.quote is None:
        return []
    quote = decision.quote
    return [{"market": "cards", "side": norm(quote.side), "point": quote.line, "odds": quote.odds,
             "bookmaker": quote.bookmaker, "p": decision.model_probability, "ev": decision.expected_value}]


def pinnacle_close(payload, side, point, home, away):
    """Pinnacle's closing fair probability of one cards line's side (final pre-kickoff snapshot), or None."""
    books = dis._snapshot_books(payload, MARKET_KEY, home, away)
    pinnacle = {name: outcomes for name, outcomes in books.items() if norm(name) == "pinnacle" and outcomes}
    _, consensus, _ = devig_consensus(pinnacle, side, point, spreads=False, method="power")
    return consensus["fair_probability"] if consensus else None


def grade(pick, outcome, payload, home, away):
    """The pick with its result (``y`` win share, ``w`` weight) and Pinnacle's closing fair probability."""
    selection = {"market": "cards", "side": pick["side"], "point": pick["point"], "home": home, "away": away}
    graded = grade_selection(selection, outcome)
    y, w = graded if graded else (None, 0.0)
    p_close = pinnacle_close(payload, pick["side"], pick["point"], home, away) if payload else None
    return {**pick, "y": y, "w": w, "p_close": p_close}


def pinnacle_then(pick, fixture, body, at, index, *, kickoff=None, threshold=0.02, max_age=60):
    """Price confirmation's Pinnacle check (D61) at the read, on Pinnacle's OddsPapi history as it stood then; with
    ``kickoff``, also Pinnacle's true closing fair probability (its last active prices before kickoff)."""
    if body is None:
        return {"status": "unknown", "reason": "history_not_cached"}
    checker = pcf.PriceConfirmation(None, index, min_ev=threshold, max_age=timedelta(minutes=max_age),
                                    clock=lambda: at)
    snapshot = {"decision": {"quote": {"odds": pick["odds"]}}, "context": {"engine_switches": dict(s4.BUNDLE)}}
    line = {"market": "cards", "side": pick["side"], "point": pick["point"], "home": fixture.get("home_team"),
            "away": fixture.get("away_team"), "bookmaker": pick["bookmaker"]}
    record = checker._against_reference(snapshot, line, {"status": "confirmed"}, s4.history_at(body, at), at)
    out = {"status": "kept"} if record["status"] == "confirmed" else {"status": "dropped", "reason": record["reason"]}
    if "reference" in record:
        out["fair_probability"] = record["reference"]["fair_probability"]
    close = true_close.close_from_history(body, "pinnacle", index, line, kickoff) if kickoff else None
    if close:
        _, consensus, _ = devig_consensus({"Pinnacle": [(s, p, price) for s, p, price, _ in close]}, line["side"],
                                          line["point"], spreads=False, method="power")
        out["true_close_probability"] = consensus["fair_probability"] if consensus else None
    return out


def _group(rows, resamples):
    out = s4._group(rows, resamples)
    graded = [r for r in rows if r.get("w")]

    def gap(sample):
        weights = [r["w"] for r in sample if r.get("w")]
        return (sum(r["w"] * (r["y"] - r["p"]) for r in sample if r.get("w")) / sum(weights)) if weights else None
    closed = [r for r in rows if r.get("p_true_close") is not None]

    def true_value(sample):
        return sum(r["p_true_close"] * r["odds"] - 1 for r in sample) / len(sample) if sample else None
    out.update(true_closing_picks=len(closed), value_at_true_close=true_value(closed),
               value_at_true_close_interval95=dis.fixture_bootstrap(closed, true_value, resamples=resamples)
               if closed else [None, None])
    out["won_minus_stated"] = gap(graded) if graded else None
    out["won_minus_stated_interval95"] = (dis.fixture_bootstrap(graded, gap, resamples=resamples) if graded
                                          else [None, None])
    return out


def summarize(rows, *, resamples=1000):
    """Over all picks and per side: picks, stated against won (with its interval), value at the read and close."""
    out = {"all": _group(rows, resamples)}
    out.update({side: _group([r for r in rows if r["side"] == side], resamples)
                for side in SIDES if any(r["side"] == side for r in rows)})
    return out


def run(reads, outcomes, snapshots, *, pinnacle=None, threshold=0.02, resamples=1000):
    """Picks per read, graded; with ``pinnacle`` (row -> record) also Pinnacle's check at the read."""
    rows = []
    for read in reads:
        fid, data = read["fixture_id"], read["data"]
        fixture = next((r.get("fixture") for r in data["canonical_results"] if r.get("fixture")), {}) or {}
        home, away = fixture.get("home_team"), fixture.get("away_team")
        for pick in replay_read(data, threshold=threshold):
            payload = snapshots.get(fid) if read["stage"] == "pre_match" else None
            row = {**(grade(pick, outcomes[fid], payload, home, away) if fid in outcomes else
                      {**pick, "y": None, "w": 0.0, "p_close": None}),
                   "fixture_id": fid, "stage": read["stage"], "fixture": fixture, "evaluated_at": read["evaluated_at"]}
            if pinnacle is not None:
                row["pinnacle_then"] = pinnacle(row)
                if read["stage"] == "pre_match":  # as the close: a lineup read can postdate Pinnacle's last move
                    row["p_true_close"] = row["pinnacle_then"].get("true_close_probability")
            rows.append(row)
    out = {"reads": len(reads), "fixtures": len({r["fixture_id"] for r in reads}),
           "reads_with_cards_prices": sum(any(o.get("bookmaker") for o in _cards_options(r)) for r in reads),
           "reads_with_pinnacle": sum(any(norm(o.get("bookmaker")) == "pinnacle" for o in _cards_options(r))
                                      for r in reads),
           "picks": summarize(rows, resamples=resamples)}
    if pinnacle is not None:
        out["pinnacle_then"] = dict(Counter(r["pinnacle_then"]["status"] for r in rows))
        out["pinnacle_then_reasons"] = dict(Counter(r["pinnacle_then"].get("reason") for r in rows
                                                    if r["pinnacle_then"]["status"] != "kept"))
        out["kept_against_pinnacle_then"] = summarize([r for r in rows if r["pinnacle_then"]["status"] == "kept"],
                                                      resamples=resamples)
    out["rows"] = [{k: v for k, v in r.items() if k != "fixture"} for r in rows]
    return out


def _cards_options(read):
    result = next((r for r in read["data"].get("canonical_results", [])
                   if (r.get("market") or {}).get("group") == "cards"), None) or {}
    return ((result.get("context") or {}).get("decision_audit") or {}).get("input_options") or []


def _pinnacle_checker(reads):
    """row -> Pinnacle's check at its read, from OddsPapi fixture lists and histories already downloaded."""
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    client = oddspapi.OddsPapiClient(os.environ["ODDSPAPI_API_KEY"], compress=True)
    client.budget = client._usage()["billable"]  # cached answers only for anything billable
    fixtures = []
    for read in reads:
        fixture = next((r.get("fixture") for r in read["data"]["canonical_results"] if r.get("fixture")), {}) or {}
        kickoff = s4.utc_datetime(fixture.get("kickoff"))
        if kickoff is not None:
            fixtures.append({"fixture_id": read["fixture_id"], "kickoff": kickoff, "league": fixture.get("league"),
                             "home": fixture.get("home_team"), "away": fixture.get("away_team")})
    mapping, skipped = age._fixture_map(client, list({f["fixture_id"]: f for f in fixtures}.values()))
    index = oddspapi.market_index(client.get("/v4/markets"))

    def check(row):
        odds_id = mapping.get(str(row["fixture_id"]))
        if odds_id is None:
            return {"status": "unknown", "reason": "fixture_not_in_cached_lists"}
        body = age._history(client, odds_id, "pinnacle", cached_only=True)
        return pinnacle_then(row, row["fixture"], body, _time(row["evaluated_at"]), index,
                             kickoff=s4.utc_datetime(row["fixture"].get("kickoff")))
    return check, {"fixtures_listed": len(mapping), "skipped_lists": skipped}


def _pct(value):
    return "n/a" if value is None else f"{value * 100:+.1f}%"


def _line(label, group):
    if not group["picks"]:
        return f"  {label}: no picks"
    lo, hi = group["won_minus_stated_interval95"]
    vlo, vhi = group["value_at_close_interval95"]
    stated = "n/a" if group["stated"] is None else f"{group['stated'] * 100:.1f}%"
    won = "n/a" if group["won"] is None else f"{group['won'] * 100:.1f}%"
    return (f"  {label}: {group['picks']} picks ({group['fixtures']} fixtures), median odds {group['median_odds']:.2f}; "
            f"stated {stated}, won {won} (won − stated {_pct(group['won_minus_stated'])}, {_pct(lo)} to {_pct(hi)}); "
            f"value at the read {_pct(group['value_at_read'])}, at Pinnacle's close {_pct(group['value_at_close'])} "
            f"({_pct(vlo)} to {_pct(vhi)}, {group['closing_picks']} picks)"
            + (f", at Pinnacle's true close {_pct(group['value_at_true_close'])} "
               f"({_pct(group['value_at_true_close_interval95'][0])} to "
               f"{_pct(group['value_at_true_close_interval95'][1])}, {group['true_closing_picks']} picks)"
               if group["true_closing_picks"] else ""))


def summary(result, title):
    lines = [f"{title}: {result['reads']} reads ({result['fixtures']} fixtures); cards prices in "
             f"{result['reads_with_cards_prices']}, Pinnacle's in {result['reads_with_pinnacle']}"]
    lines += [_line(label, group) for label, group in result["picks"].items()]
    if "pinnacle_then" in result:
        lines.append(f"  Pinnacle's own price at the read: {result['pinnacle_then']} "
                     f"(not kept: {result['pinnacle_then_reasons'] or 'none'})")
        lines += [_line(f"kept, {label}", group) for label, group in result["kept_against_pinnacle_then"].items()]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--db", type=Path, default=ROOT / "Index/platform.db")
    parser.add_argument("--backfill", type=Path, default=ROOT / "Index/closing_backfill")
    parser.add_argument("--from", dest="since", default="2026-10-09")
    parser.add_argument("--to", dest="until", default=datetime.now(timezone.utc).date().isoformat(),
                        help="kickoffs before this date (default: today, UTC)")
    parser.add_argument("--resamples", type=int, default=1000)
    parser.add_argument("--pinnacle-then", action="store_true",
                        help="Pinnacle's own OddsPapi price at each read, from histories already downloaded")
    args = parser.parse_args(argv)
    reads, outcomes, snapshots = s4.load(args.db, args.backfill, args.since, args.until)
    pre_match = [r for r in reads if r["stage"] == "pre_match"]
    lineups = [r for r in reads if r["stage"] == "confirmed_lineups"]
    checker, coverage = _pinnacle_checker(reads) if args.pinnacle_then else (None, None)
    result = {"from": args.since, "to": args.until, "oddspapi_coverage": coverage,
              "pre_match": run(pre_match, outcomes, snapshots, pinnacle=checker, resamples=args.resamples),
              "confirmed_lineups": run(lineups, outcomes, snapshots, pinnacle=checker, resamples=args.resamples)}
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "cards_reference_replay.json").write_text(json.dumps(result, indent=2, default=str))
    if coverage:
        print(f"OddsPapi fixture lists cached for {coverage['fixtures_listed']} fixtures; "
              f"weeks without a cached list: {coverage['skipped_lists'] or 'none'}")
    print(summary(result["pre_match"], "Pre-match reads (latest per fixture)") + "\n\n"
          + summary(result["confirmed_lineups"], "Lineup reads (no value at the close)"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
