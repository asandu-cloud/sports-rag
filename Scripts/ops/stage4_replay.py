"""Stage 4 replay (A3.7 stage 4 bundle, D33 (b), D35): what the bundle would publish on the reads already stored.

For every audited read (latest before kickoff, per fixture and stage), the engine's own selectors run again on the
prices the decision audit stored, under the bundle's switches: power margin removal, the market anchor in full (M1,
solved from the read's own prices, for every goal line, the handicap and BTTS; 1X2, O/U 2.5 and each corner line at
the consensus) and the market-value rule, at every bookmaker or Romania's only (Betano, Superbet). The value
threshold (``line_selection.MARKET_VALUE_MIN_EV``, 2%) is varied. A market the read's data-quality guardrail blocked
publishes nothing, as live. Shots on target keep today's rule (the read's own pick, at an allowed bookmaker); cards
are blocked by the guardrail in every read. A Match Read then publishes at most three compatible picks, market-priced
ones first by value (the compiler's own rule, D35).

With ``--confirm``, each published pick then goes through price confirmation (D42) as at its read: the quoting
bookmaker's OddsPapi history as it stood at the read, at each freshness limit (``MAX_AGES``). Only histories already
downloaded are read (the D42 replay's), unless ``--fetch-histories`` (free requests); a pick whose history or match
listing is not cached is counted as unknown, not as a failure. No billable request: the client's budget is the
billable count already used. The key is read from ``.env`` and never printed.

Each published pick is graded on the result and valued at the close (API-Football's final pre-kickoff snapshot, the
consensus fair price). Headline counts use each fixture's final read (confirmed lineups when there is one), beside
what today's rule published in the same reads (the reads' own selections). The stored reads are copied behind the refresh gate in short statements;
the replay runs after the gate is released.

    python -m Scripts.ops.stage4_replay --out Research/stage4-replay-<date>
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

from Scripts.data_platform import disagreement as dis, oddspapi
from Scripts.data_platform.calibration_data import _time, fixture_outcomes, grade as grade_selection, latest_read_rows
from Scripts.data_platform.m1_gate import read_fit
from Scripts.data_platform.reference_price import norm
from Scripts.data_platform.services import match_read_compiler as compiler, price_confirmation as pcf
from Scripts.data_platform.services.refresh_coordination import data_access
from Scripts.data_platform.settlement import utc_datetime
from Scripts.ops import confidence_age_study as age
from Scripts.ops.odds_archive import ROOT

BUNDLE = {"SPIX_DEVIG_METHOD": "power", "SPIX_MARKET_ANCHOR": "full", "SPIX_SELECTION_POLICY": "market_value"}
THRESHOLDS = (0.01, 0.02, 0.03, 0.05)
BOOK_SETS = ("all", "romania")
MARKETS = ("goals", "corners", "btts", "moneyline", "spreads")  # priced against the market at stage 4
KEPT = ("sot",)  # today's rule; cards are blocked by the guardrail in every read
GOAL_MODEL_MARKETS = ("goals", "btts", "spreads")
MAX_AGES = (30, 60, None)  # price confirmation's freshness limit in minutes; None: no limit (owner to choose, D42)
_SOLVE = object()


def _engine():
    path = str(ROOT / "Scripts/rag_ingest")
    if path not in sys.path:
        sys.path.insert(0, path)
    from core import line_selection, market_goal_model
    from core.market_result import decision_from_selector
    return line_selection, market_goal_model, decision_from_selector


@contextmanager
def bundle(threshold, books):
    """The stage 4 switches and the value threshold for the duration of the block, then exactly as they were."""
    line_selection = _engine()[0]
    settings = {**BUNDLE, "SPIX_BOOKMAKER_SET": books}
    saved, saved_threshold = {name: os.environ.get(name) for name in settings}, line_selection.MARKET_VALUE_MIN_EV
    os.environ.update(settings)
    line_selection.MARKET_VALUE_MIN_EV = threshold
    try:
        yield
    finally:
        line_selection.MARKET_VALUE_MIN_EV = saved_threshold
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def solve_fit(data):
    """M1 from one read's stored prices, with the bundle's margin removal (the threshold and books play no part)."""
    with bundle(_engine()[0].MARKET_VALUE_MIN_EV, "all"):
        return read_fit(data)


def _publishable(result):
    context = result.get("context") or {}
    guardrail = (context.get("data_quality") or {}).get("recommendation_guardrail") or {}
    return guardrail.get("eligible") is True and (context.get("release_policy") or {}).get("mode") == "live"


def _select(market, result, fit):
    """The stage 4 selector's response for one stored market result, as ``market_service`` calls it; or None."""
    line_selection, mgm, _ = _engine()
    options = deepcopy(((result.get("context") or {}).get("decision_audit") or {}).get("input_options") or [])
    projection, fixture = result.get("projection") or {}, result.get("fixture") or {}
    value, home, away = projection.get("value"), fixture.get("home_team"), fixture.get("away_team")
    if not options or value is None:
        return None
    if market == "goals":
        return line_selection.select_best_total_recommendation(
            options, value, projection.get("variance"), anchor_points=(2.5,),
            market_count_probabilities=mgm.total_probabilities(fit))
    if market == "corners":
        return line_selection.select_best_total_recommendation(options, value, projection.get("variance"),
                                                               anchor_points="all")
    if market == "btts":
        return line_selection.select_best_btts_recommendation(options, mgm.btts_probability(fit), spix_p_yes=value)
    if market == "spreads":
        return line_selection.select_best_spread_recommendation(
            options, value, home, away_team=away, market_score_probs=mgm.score_probabilities(fit),
            market_diff=fit["home"] - fit["away"])
    components = projection.get("components") or {}
    probabilities = [components.get(f"{side}_probability") for side in ("home", "draw", "away")]
    if None in probabilities:
        return None
    return line_selection.choose_best_moneyline_side(*probabilities, options, home, away, anchor=True)


def replay_read(data, *, threshold=0.02, books="all", fit=_SOLVE):
    """{"picks": [...], "unfitted": bool}: what the stage 4 bundle recommends in one stored read.

    Without M1 (the prices cannot identify both teams) the goal markets would fall back to Spix's own count
    distribution, which a read does not store, so they are left out and the read is flagged.
    """
    _, _, decision_from_selector = _engine()
    from core import bookmaker_sets
    if fit is _SOLVE:
        fit = solve_fit(data)
    results = {(r.get("market") or {}).get("group"): r for r in data.get("canonical_results", [])}
    picks = []
    with bundle(threshold, books):
        for market in MARKETS:
            result = results.get(market)
            if result is None or not _publishable(result) or (market in GOAL_MODEL_MARKETS and fit is None):
                continue
            selected = _select(market, result, fit)
            decision = decision_from_selector(selected) if selected is not None else None
            if decision is None or not decision.is_recommended or decision.quote is None:
                continue
            quote = decision.quote
            picks.append({"market": market, "side": norm(quote.side), "point": quote.line, "odds": quote.odds,
                          "bookmaker": quote.bookmaker, "p": decision.model_probability, "ev": decision.expected_value,
                          "profile": decision.settlement_profile})
        for market in KEPT:
            kept = _stored_pick(results.get(market) or {})
            if kept and _publishable(results[market]) and bookmaker_sets.allowed(kept["bookmaker"]):
                picks.append(kept)
    return {"picks": picks, "unfitted": fit is None}


def _stored_pick(result):
    """The read's own recommended pick for one market result, in the replay's shape; or None."""
    market, decision = (result.get("market") or {}).get("group"), result.get("decision") or {}
    quote = decision.get("quote") or {}
    if decision.get("status") != "recommended" or not quote.get("odds"):
        return None
    return {"market": market, "side": norm(quote.get("side")), "point": quote.get("line"), "odds": quote["odds"],
            "bookmaker": quote.get("bookmaker"), "p": decision.get("model_probability"),
            "ev": decision.get("expected_value"), "profile": decision.get("settlement_profile")}


def recommended_today(data):
    """Every pick today's rule recommended in the read's markets."""
    return [pick for result in data.get("canonical_results", [])
            if (result.get("market") or {}).get("group") in MARKETS + KEPT and (pick := _stored_pick(result))]


def published_today(data):
    """The picks the read published (its own selections) in the replayed markets."""
    results = data.get("canonical_results", [])
    return [pick for index in data.get("published") or [] if index < len(results)
            and (results[index].get("market") or {}).get("group") in MARKETS + KEPT
            and (pick := _stored_pick(results[index]))]


def publish(picks, home, away):
    """What one Match Read publishes under stage 4: market-priced picks first by value, then the others, at most
    ``MAX_ACTIONABLE_SELECTIONS`` that do not contradict each other (``match_read_compiler``, D35)."""
    fixture = {"fixture": {"home_team": home, "away_team": away}}

    def rank(pick):
        return (pick["market"] not in MARKETS, -(pick["ev"] or 0.0), -(pick["p"] or 0.0),
                compiler._MARKET_TIEBREAK_ORDER.get(pick["market"], 99))
    chosen = []
    for pick in sorted(picks, key=rank):
        if len(chosen) >= compiler.MAX_ACTIONABLE_SELECTIONS:
            break
        candidate = SimpleNamespace(market_group=pick["market"],
                                    tags=compiler._candidate_tags(fixture, pick["market"], "", pick["side"]))
        if all(compiler._compatible(candidate, other) for _, other in chosen):
            chosen.append((pick, candidate))
    return [pick for pick, _ in chosen]


def grade(pick, outcome, payload, home, away):
    """The pick with its result (``y`` win share, ``w`` weight) and closing fair probability; None if ungradable."""
    selection = {"market": pick["market"], "side": pick["side"], "point": pick["point"], "home": home, "away": away}
    graded = grade_selection(selection, outcome)
    if graded is None or graded[1] <= 0:
        return None
    line = {**selection, "market_key": dis.MARKET_KEYS[pick["market"]]}
    return {**pick, "y": graded[0], "w": graded[1], "p_close": dis.closing_probability(payload, line) if payload else None}


def _age_label(max_age):
    return "none" if max_age is None else str(max_age)


def history_at(body, at):
    """A fixture's price history as it stood at ``at``: only the entries OddsPapi had created by then."""
    out = {"bookmakers": {}}
    for slug, book in ((body or {}).get("bookmakers") or {}).items():
        markets = {}
        for market_id, market in (book.get("markets") or {}).items():
            for outcome_id, outcome in (market.get("outcomes") or {}).items():
                entries = [e for e in (outcome.get("players") or {}).get("0") or []
                           if e.get("createdAt") and oddspapi.utc(e["createdAt"]) <= at]
                if entries:
                    markets.setdefault(market_id, {"outcomes": {}})["outcomes"][outcome_id] = {"players": {"0": entries}}
        out["bookmakers"][slug] = {"markets": markets}
    return out


class _AtTheRead(pcf.PriceConfirmation):
    """The Match Read worker's price confirmation, run at a past read on the bookmaker's history as it stood then."""

    def __init__(self, index, odds_id, history, at, **kwargs):
        super().__init__(None, index, clock=lambda: at, **kwargs)
        self._odds_id, self._history, self._at = odds_id, history, at

    def _fixture_id(self, fixture):
        return self._odds_id

    def _histories(self, odds_id, slugs):
        return {slug: history_at(self._history, self._at) for slug in slugs}, set()


def confirm_pick(pick, fixture, *, at, index, odds_id, history, threshold, max_age):
    """Price confirmation's record for one replayed pick at its read (``max_age`` minutes; None: no limit)."""
    quote = {"side": pick["side"], "line": pick["point"], "odds": pick["odds"], "bookmaker": pick["bookmaker"]}
    result = {"fixture": fixture, "market": {"group": pick["market"]},
              "decision": {"status": "recommended", "model_probability": pick["p"],
                           "settlement_profile": pick.get("profile"), "quote": quote},
              "context": {"engine_switches": dict(BUNDLE)}}
    checker = _AtTheRead(index, odds_id, history, at, min_ev=threshold,
                         max_age=None if max_age is None else timedelta(minutes=max_age))
    [checked] = checker([result])
    return checked["context"]["price_confirmation"]


def confirmations(row, client, index, mapping, *, max_ages=MAX_AGES, cached_only=True):
    """{freshness limit: confirmation record} for one published pick; "unknown" where nothing is downloaded."""
    def unknown(reason):
        return {_age_label(m): {"status": "unknown", "reason": reason} for m in max_ages}
    fixture, at = row["fixture"], _time(row["evaluated_at"])  # the database writes "YYYY-MM-DD HH:MM:SS.ffffff"
    slug = oddspapi.BOOKS.get(row["bookmaker"])
    odds_id, body = mapping.get(str(fixture.get("event_id"))), None
    if slug and slug not in oddspapi.EXCHANGES:  # else confirmation says why without a history
        if odds_id is None:
            return unknown("fixture_not_in_cached_lists")
        if cached_only and not client.cached("/v4/historical-odds", fixtureId=odds_id, bookmakers=slug):
            return unknown("history_not_cached")
        body = age._history(client, odds_id, slug, cached_only=cached_only)
    return {_age_label(m): confirm_pick(row, fixture, at=at, index=index, odds_id=odds_id, history=body,
                                        threshold=row["threshold"], max_age=m) for m in max_ages}


def confirmed_summary(rows, label, *, resamples=1000):
    """Published picks after price confirmation at one freshness limit; repriced picks at the bookmaker's own price."""
    records = [r["confirmation"][label] for r in rows]
    statuses = Counter(record["status"] for record in records)
    kept = [{**r, "odds": record.get("own_odds") or r["odds"]} for r, record in zip(rows, records)
            if record["status"] in ("confirmed", "repriced")]
    return {**_group(kept, resamples), "published": len(kept), "confirmed": statuses["confirmed"],
            "repriced": statuses["repriced"], "not_confirmed": statuses["not_confirmed"], "unknown": statuses["unknown"],
            "reasons": dict(Counter(record.get("reason") for record in records if record["status"] == "not_confirmed")),
            "unknown_reasons": dict(Counter(record.get("reason") for record in records if record["status"] == "unknown"))}


def _group(rows, resamples):
    closed = [r for r in rows if r.get("p_close") is not None]
    graded = [r for r in rows if r.get("w")]

    def value(sample):
        return float(np.mean([r["p_close"] * r["odds"] - 1 for r in sample])) if sample else None
    weights = np.array([r["w"] for r in graded])
    return {"picks": len(rows), "fixtures": len({r["fixture_id"] for r in rows}),
            "median_odds": float(np.median([r["odds"] for r in rows])) if rows else None,
            "value_at_read": float(np.mean([r["ev"] for r in rows if r.get("ev") is not None])) if rows else None,
            "graded": len(graded),
            "stated": float((weights * [r["p"] for r in graded]).sum() / weights.sum()) if graded else None,
            "won": float((weights * [r["y"] for r in graded]).sum() / weights.sum()) if graded else None,
            "closing_picks": len(closed), "value_at_close": value(closed),
            "value_at_close_interval95": dis.fixture_bootstrap(closed, value, resamples=resamples)
            if closed else [None, None]}


def summarize(rows, *, resamples=1000):
    """Per market and over all markets: picks, fixtures, value at the read and at the close, stated against won."""
    out = {market: _group([r for r in rows if r["market"] == market], resamples)
           for market in MARKETS + KEPT if any(r["market"] == market for r in rows)}
    out["all"] = _group(rows, resamples)
    return out


def _compact(data):
    """The parts of a stored read the replay uses (a read's JSON is ~0.75 MB)."""
    keep = []
    for result in data.get("canonical_results", []):
        context = result.get("context") or {}
        keep.append({"market": result.get("market"), "fixture": result.get("fixture"),
                     "projection": result.get("projection"), "decision": result.get("decision"),
                     "context": {"decision_audit": {"input_options": (context.get("decision_audit") or {})
                                                    .get("input_options")},
                                 "data_quality": {"recommendation_guardrail": (context.get("data_quality") or {})
                                                  .get("recommendation_guardrail")},
                                 "release_policy": context.get("release_policy")}})
    return {"canonical_results": keep,
            "published": [s["result_index"] for s in data.get("selections") or [] if isinstance(s.get("result_index"), int)]}


def load(db_path, backfill_dir, since, until):
    """Stored reads with kickoffs in [since, until), their results and closing snapshots, behind the refresh gate."""
    with data_access():  # never read while refresh-season writes
        outcomes = fixture_outcomes(db_path)
        snapshots = dis.closing_snapshots(db_path, backfill_dir)
        reads = []
        for row in latest_read_rows(db_path):
            if since <= str(row["kickoff"])[:10] < until:
                data = json.loads(row["read_json"]) if isinstance(row["read_json"], str) else row["read_json"]
                reads.append({"fixture_id": str(row["fixture_api_id"]), "stage": row["stage"],
                              "kickoff": str(row["kickoff"]), "evaluated_at": str(row["evaluated_at"]),
                              "data": _compact(data)})
    return reads, outcomes, {fid: snapshots[r["fixture_id"]] for r in reads if (fid := r["fixture_id"]) in snapshots}


def final_reads(reads):
    """Each fixture's last read: confirmed lineups when there is one, else pre-match."""
    order = {"pre_match": 0, "confirmed_lineups": 1}
    best = {}
    for read in reads:
        current = best.get(read["fixture_id"])
        if current is None or order.get(read["stage"], -1) > order.get(current["stage"], -1):
            best[read["fixture_id"]] = read
    return list(best.values())


def run(reads, outcomes, snapshots, *, thresholds=THRESHOLDS, book_sets=BOOK_SETS, resamples=1000, confirm=None):
    """Picks per rule; with ``confirm`` (row -> {freshness limit: record}) the stage 4 picks also go through price
    confirmation at their read."""
    names = ["today"] + [f"{threshold:.2f}/{books}" for books in book_sets for threshold in thresholds]
    rows, unfitted = {name: {"recommended": [], "published": []} for name in names}, 0
    for read in reads:
        data, fid = read["data"], read["fixture_id"]
        fixture = next((r.get("fixture") for r in data["canonical_results"] if r.get("fixture")), {}) or {}
        home, away = fixture.get("home_team"), fixture.get("away_team")
        fit = solve_fit(data)
        unfitted += fit is None
        variants = {"today": (None, recommended_today(data), published_today(data))}
        for books in book_sets:
            for threshold in thresholds:
                picks = replay_read(data, threshold=threshold, books=books, fit=fit)["picks"]
                variants[f"{threshold:.2f}/{books}"] = (threshold, picks, publish(picks, home, away))
        for name, (threshold, recommended, published) in variants.items():
            for kind, picks in (("recommended", recommended), ("published", published)):
                for pick in picks:
                    graded = grade(pick, outcomes[fid], snapshots.get(fid), home, away) if fid in outcomes else None
                    row = {**(graded or pick), "fixture_id": fid, "stage": read["stage"], "fixture": fixture,
                           "evaluated_at": read["evaluated_at"], "threshold": threshold}
                    if confirm is not None and threshold is not None and kind == "published":
                        row["confirmation"] = confirm(row)
                    rows[name][kind].append(row)
    out = {name: {kind: summarize(group, resamples=resamples) for kind, group in kinds.items()}
           for name, kinds in rows.items()}
    if confirm is not None:
        labels = sorted({label for kinds in rows.values() for r in kinds["published"] for label in r.get("confirmation", {})},
                        key=lambda label: float("inf") if label == "none" else float(label))
        for name, kinds in rows.items():
            if name != "today":
                out[name]["confirmed"] = {label: confirmed_summary(kinds["published"], label, resamples=resamples)
                                          for label in labels}
    return {"reads": len(reads), "fixtures": len({r["fixture_id"] for r in reads}), "unfitted_reads": unfitted,
            "variants": out}


def _pct(value):
    return "n/a" if value is None else f"{value * 100:+.1f}%"


def summary(result, title):
    lines = [f"{title}: {result['reads']} reads ({result['fixtures']} fixtures); M1 not solved: "
             f"{result['unfitted_reads']}. Published picks (at most three per Match Read); recommended in brackets.",
             "", "| Rule | Goals | Corners | BTTS | 1X2 | Handicap | SoT | All | Fixtures | Value at read | "
             "Value at close (95%) | Stated / won |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, kinds in result["variants"].items():
        by_market, recommended = kinds["published"], kinds["recommended"]
        counts = [f"{by_market.get(m, {}).get('picks', 0)} ({recommended.get(m, {}).get('picks', 0)})"
                  for m in MARKETS + KEPT]
        a = by_market["all"]
        lo, hi = a["value_at_close_interval95"]
        interval = "" if lo is None else f" ({lo * 100:+.1f} to {hi * 100:+.1f})"
        stated = "n/a" if a["stated"] is None else f"{a['stated']:.1%} / {a['won']:.1%} ({a['graded']})"
        lines.append(f"| {name} | {' | '.join(counts)} | {a['picks']} ({recommended['all']['picks']}) | {a['fixtures']} | "
                     f"{_pct(a['value_at_read'])} | "
                     f"{_pct(a['value_at_close'])}{interval} | {stated} |")
    confirmed = {name: kinds["confirmed"] for name, kinds in result["variants"].items() if "confirmed" in kinds}
    if confirmed:
        labels = list(next(iter(confirmed.values())))
        heads = [f"{'no limit' if label == 'none' else label + ' min'}" for label in labels]
        lines += ["", "After price confirmation at the read: published (unknown: no downloaded history), value at close.",
                  "", "| Rule | Published | " + " | ".join(heads) + " |", "|---|---|" + "---|" * len(labels)]
        for name, by_label in confirmed.items():
            cells = []
            for label in labels:
                c = by_label[label]
                cells.append(f"{c['published']} ({c['unknown']}) {_pct(c['value_at_close'])}")
            lines.append(f"| {name} | {result['variants'][name]['published']['all']['picks']} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


class _Remembered:
    """The OddsPapi client with the last few answers kept in memory: a read's picks are confirmed together."""

    def __init__(self, client, size=8):
        self.client, self.size, self.memory = client, size, {}

    def cached(self, path, **params):
        return self.client.cached(path, **params)

    def get(self, path, **params):
        key = (path, tuple(sorted(params.items())))
        if key not in self.memory:
            if len(self.memory) >= self.size:
                self.memory.pop(next(iter(self.memory)))
            self.memory[key] = self.client.get(path, **params)
        return self.memory[key]


def _confirmer(reads, *, cached_only):
    """row -> {freshness limit: record}, from OddsPapi answers already downloaded (histories too unless asked)."""
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    client = oddspapi.OddsPapiClient(os.environ["ODDSPAPI_API_KEY"], compress=True)
    client.budget = client._usage()["billable"]  # cached answers only for anything billable
    fixtures = []
    for read in reads:
        fixture = next((r.get("fixture") for r in read["data"]["canonical_results"] if r.get("fixture")), {}) or {}
        if utc_datetime(fixture.get("kickoff")) is not None:
            fixtures.append({"fixture_id": read["fixture_id"], "kickoff": utc_datetime(fixture["kickoff"]),
                             "league": fixture.get("league"), "home": fixture.get("home_team"),
                             "away": fixture.get("away_team")})
    unique = list({f["fixture_id"]: f for f in fixtures}.values())
    mapping, skipped = age._fixture_map(client, unique)
    index = oddspapi.market_index(client.get("/v4/markets"))
    remembered, done = _Remembered(client), {}

    def confirm(row):
        key = (row["fixture_id"], row["evaluated_at"], row["market"], row["side"], row["point"], row["odds"],
               row["bookmaker"], row["threshold"])
        if key not in done:
            done[key] = confirmations(row, remembered, index, mapping, cached_only=cached_only)
        return done[key]
    return confirm, {"fixtures_listed": len(mapping), "fixtures": len(unique), "skipped_lists": skipped}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--db", type=Path, default=ROOT / "Index/platform.db")
    parser.add_argument("--backfill", type=Path, default=ROOT / "Index/closing_backfill")
    parser.add_argument("--from", dest="since", default="2026-10-09")
    parser.add_argument("--to", dest="until", default=datetime.now(timezone.utc).date().isoformat(),
                        help="kickoffs before this date (default: today, UTC)")
    parser.add_argument("--resamples", type=int, default=1000)
    parser.add_argument("--confirm", action="store_true",
                        help="price confirmation at each read, from OddsPapi histories already downloaded")
    parser.add_argument("--fetch-histories", action="store_true",
                        help="with --confirm, request histories not yet downloaded (free, but real OddsPapi requests)")
    args = parser.parse_args(argv)
    reads, outcomes, snapshots = load(args.db, args.backfill, args.since, args.until)
    confirm, coverage = _confirmer(reads, cached_only=not args.fetch_histories) if args.confirm else (None, None)
    result = {"from": args.since, "to": args.until, "confirmation_coverage": coverage,
              "final": run(final_reads(reads), outcomes, snapshots, resamples=args.resamples, confirm=confirm),
              "pre_match": run([r for r in reads if r["stage"] == "pre_match"], outcomes, snapshots,
                               resamples=args.resamples, confirm=confirm)}
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "stage4_replay.json").write_text(json.dumps(result, indent=2))
    if coverage:
        print(f"OddsPapi fixture lists cached for {coverage['fixtures_listed']} of {coverage['fixtures']} fixtures; "
              f"weeks without a cached list: {coverage['skipped_lists'] or 'none'}")
    print(summary(result["final"], "Final read per fixture") + "\n\n" + summary(result["pre_match"], "Pre-match reads"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
