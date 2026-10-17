"""Golden snapshots of the Match Read engine (rework step A0).

``record`` evaluates real fixtures exactly as the Match Read worker does and
saves, per fixture, the event (odds) payload, every market result, and every
outside read the engine made on the way: Chroma ``get``/``query`` calls and
HTTP requests (API-Football). ``check`` re-runs the current code with those
reads served from the bundle and lists every changed probability, line, pick
or other result field. Fields that change on every code edit (timestamps, the
code fingerprints and the engine-switch record) are ignored.

A read that was not recorded fails the replay loudly (``GoldenMiss``), so a new
data dependency shows up instead of silently reading live data. Model files
under ``Index/ml_models`` are treated as code, not as recorded inputs.

``preview`` replays the same fixtures and summarises what a switched-on stage does to the
published picks (added, removed, moved to another side or line, price only), per market.

    python -m Scripts.ops.golden_snapshots record --date 2026-10-17 --date 2026-10-18
    python -m Scripts.ops.golden_snapshots check
    SPIX_MARKET_ANCHOR=goals python -m Scripts.ops.golden_snapshots preview
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import date, datetime, timezone
import functools
import gzip
import json
import math
import os
from pathlib import Path
import sys
from unittest import mock

# Chroma's anonymous usage telemetry posts over HTTP at random moments; replayed, each post is an
# unrecorded read. Switched off before any Chroma client exists so record and check stay deterministic.
os.environ["ANONYMIZED_TELEMETRY"] = "False"

import requests
from requests.models import Response

from Scripts.ops.odds_archive import ROOT

BUNDLE = ROOT / "Index/golden/match_read_engine.json.gz"
BUNDLE_VERSION = 1
LEAGUES = ("EPL", "LaLiga", "SerieA", "Bundesliga", "Ligue1", "Championship", "Eredivisie",
           "PrimeiraLiga", "BelgianProLeague", "SuperLig", "UCL", "UEL", "UECL")
# Paths whose values change with every code edit or run, not with predictions.
VOLATILE = {
    ("provenance", "generated_at"), ("provenance", "model_version"), ("provenance", "system_version"),
    ("context", "prediction_system"), ("context", "engine_switches"), ("decision_audit", "prediction_version"),
    ("decision_audit", "release_identity"),  # fingerprint of the communication files (communication_release.py)
    ("decision_audit", "policy_version"),    # fingerprint of the selection files (decision_audit.policy_identity)
}
REL_TOLERANCE = 1e-9


class GoldenMiss(LookupError):
    """The engine made a read the bundle does not hold."""


def _chroma_targets():
    from chromadb.api.models.Collection import Collection
    from Scripts.data_platform.vectors.document_store import DocumentStore
    return [(Collection, "get"), (Collection, "query"), (DocumentStore, "get")]


def _key(**parts):
    return json.dumps(parts, sort_keys=True, default=str)


def _plain(value):
    """Round-trip through JSON so recorded and replayed values look identical."""
    return json.loads(json.dumps(value, default=str))


class Capture:
    """Record or replay outside reads. ``pool`` holds {"chroma": {...}, "http": {...}}."""

    def __init__(self, mode, pool, *, chroma_targets=None, http=True):
        if mode not in {"record", "replay"}:
            raise ValueError(mode)
        self.mode, self.http = mode, http
        self.pool = pool
        pool.setdefault("chroma", {})
        pool.setdefault("http", {})
        self.targets = chroma_targets if chroma_targets is not None else _chroma_targets()
        self.misses = []  # kept even when the engine catches GoldenMiss and falls back
        self._stack = ExitStack()

    def __enter__(self):
        for owner, name in self.targets:
            self._stack.enter_context(mock.patch.object(owner, name, self._chroma(getattr(owner, name), name)))
        if self.http:
            self._stack.enter_context(mock.patch.object(requests.Session, "request", self._request(requests.Session.request)))
        return self

    def __exit__(self, *exc):
        return self._stack.__exit__(*exc)

    def _chroma(self, real, method):
        store, mode, misses = self.pool["chroma"], self.mode, self.misses

        @functools.wraps(real)
        def wrapper(collection, *args, **kwargs):
            key = _key(method=method, collection=getattr(collection, "name", None), args=args, kwargs=kwargs)
            if mode == "replay":
                if key not in store:
                    misses.append(f"chroma {key}")
                    raise GoldenMiss(f"chroma read not in bundle: {key[:300]}")
                return store[key]
            result = _plain(real(collection, *args, **kwargs))
            store[key] = result
            return result
        return wrapper

    def _request(self, real):
        store, mode, misses = self.pool["http"], self.mode, self.misses

        @functools.wraps(real)
        def wrapper(session, method, url, params=None, data=None, headers=None, json=None, **kwargs):
            # Headers are never part of the key or the record: they carry the API key.
            key = _key(method=str(method).upper(), url=url, params=params, data=data, json=json)
            if mode == "replay":
                if key not in store:
                    misses.append(f"http {key}")
                    raise GoldenMiss(f"http request not in bundle: {key[:300]}")
                return _response(store[key], url)
            response = real(session, method, url, params=params, data=data, headers=headers, json=json, **kwargs)
            store[key] = dict(status=response.status_code, body=response.content.decode("utf-8", "replace"),
                              content_type=response.headers.get("Content-Type"))
            return response
        return wrapper


def _response(saved, url):
    response = Response()
    response.status_code = saved["status"]
    response._content = saved["body"].encode()
    if saved.get("content_type"):
        response.headers["Content-Type"] = saved["content_type"]
    response.url = url
    response.encoding = "utf-8"
    return response


def _volatile(path):
    keys = tuple(p for p in path if isinstance(p, str))
    return any(keys[i:i + len(v)] == v for v in VOLATILE for i in range(len(keys)))


def compare(old, new, path=()):
    """List (path, old, new) for every difference, ignoring volatile fields."""
    if _volatile(path):
        return []
    label = "".join(f"[{p}]" if isinstance(p, int) else f".{p}" for p in path) or "."
    if isinstance(old, dict) and isinstance(new, dict):
        return [d for k in sorted(set(old) | set(new), key=str)
                for d in compare(old.get(k, "<missing>"), new.get(k, "<missing>"), path + (k,))]
    if isinstance(old, list) and isinstance(new, list):
        if len(old) != len(new):
            return [(label + ".length", len(old), len(new))]
        return [d for i, (a, b) in enumerate(zip(old, new)) for d in compare(a, b, path + (i,))]
    if isinstance(old, (int, float)) and isinstance(new, (int, float)) and not isinstance(old, bool) \
            and not isinstance(new, bool):
        return [] if math.isclose(old, new, rel_tol=REL_TOLERANCE, abs_tol=1e-12) else [(label, old, new)]
    return [] if old == new else [(label, old, new)]


def _as_dicts(results):
    return _plain([r.to_dict() if hasattr(r, "to_dict") else r for r in results])


def _label(fixture):
    return f"{fixture['league']} {fixture['fixture_date']} {fixture['event'].get('id')}"


def record(fixtures, evaluate, path=BUNDLE, *, generated_at, reset=None, chroma_targets=None, system_version=None):
    """Evaluate each fixture live, keep its outputs and every outside read, save the bundle."""
    pool, saved = {}, []
    with Capture("record", pool, chroma_targets=chroma_targets):
        for fixture in fixtures:
            if reset:
                reset()
            kwargs = dict(fixture.get("kwargs") or {}, fixture_date=fixture["fixture_date"], generated_at=generated_at)
            outputs = _as_dicts(evaluate(fixture["event"], fixture["league"], **kwargs))
            saved.append(dict(fixture, kwargs=kwargs, outputs=outputs))
    bundle = dict(version=BUNDLE_VERSION, recorded_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                  system_version=system_version, fixtures=saved, reads=pool)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(bundle, handle)
    return bundle


def load(path=BUNDLE):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        bundle = json.load(handle)
    if bundle.get("version") != BUNDLE_VERSION:
        raise ValueError(f"unsupported golden bundle version {bundle.get('version')}")
    return bundle


def check(path, evaluate, *, reset=None, chroma_targets=None):
    """Replay every fixture; return one entry per fixture that changed or failed."""
    bundle = load(path)
    report = []
    with Capture("replay", bundle["reads"], chroma_targets=chroma_targets) as capture:
        for fixture in bundle["fixtures"]:
            if reset:
                reset()
            capture.misses.clear()
            entry = dict(fixture=_label(fixture))
            try:
                outputs = _as_dicts(evaluate(fixture["event"], fixture["league"], **fixture["kwargs"]))
                entry["changes"] = compare(fixture["outputs"], outputs)
            except Exception as exc:  # report every fixture, not just the first failure
                entry["error"] = f"{type(exc).__name__}: {exc}"
            if capture.misses:
                entry["missing_reads"] = list(capture.misses)
            if entry.get("changes") or entry.get("error") or entry.get("missing_reads"):
                report.append(entry)
    return report


def pick(result):
    """The published-pick facts of one result."""
    decision = result.get("decision") or {}
    quote = decision.get("quote") or {}
    return {"status": decision.get("status"), "side": quote.get("side"), "line": quote.get("line"),
            "odds": quote.get("odds"), "bookmaker": quote.get("bookmaker"),
            "probability": decision.get("model_probability")}


def classify(old, new):
    was, now = old["status"] == "recommended", new["status"] == "recommended"
    if was and not now:
        return "removed"
    if now and not was:
        return "added"
    if not was:
        return None
    if (old["side"], old["line"]) != (new["side"], new["line"]):
        return "moved"
    if (old["odds"], old["bookmaker"]) != (new["odds"], new["bookmaker"]):
        return "price_only"
    return "kept"


def preview(path, evaluate, *, reset=None):
    """Old and new pick for every result of every fixture, replayed from the bundle."""
    bundle = load(path)
    rows = []
    with Capture("replay", bundle["reads"]) as capture:
        for fixture in bundle["fixtures"]:
            if reset:
                reset()
            capture.misses.clear()
            outputs = _as_dicts(evaluate(fixture["event"], fixture["league"], **fixture["kwargs"]))
            for old, new in zip(fixture["outputs"], outputs):
                rows.append({"fixture": _label(fixture), "market": old["market"]["group"],
                             "old": pick(old), "new": pick(new), "change": classify(pick(old), pick(new))})
    return rows


def summarize_preview(rows, limit=40):
    from collections import Counter
    markets = sorted({r["market"] for r in rows})
    lines = ["| Market | Picks before | After | Added | Removed | Moved | Price only |", "|---|---|---|---|---|---|---|"]
    for market in markets:
        sub = [r for r in rows if r["market"] == market]
        count = Counter(r["change"] for r in sub)
        before = sum(r["old"]["status"] == "recommended" for r in sub)
        after = sum(r["new"]["status"] == "recommended" for r in sub)
        lines.append(f"| {market} | {before} | {after} | {count['added']} | {count['removed']} | {count['moved']} | {count['price_only']} |")
    changed = [r for r in rows if r["change"] in ("added", "removed", "moved")]
    for r in changed[:limit]:
        o, n = r["old"], r["new"]
        lines.append(f"{r['fixture']} {r['market']}: {r['change']} | {o['status']} {o['side']} {o['line']} @ {o['odds']} "
                     f"(p {o['probability']}) -> {n['status']} {n['side']} {n['line']} @ {n['odds']} (p {n['probability']})")
    if len(changed) > limit:
        lines.append(f"... {len(changed) - limit} more")
    return "\n".join(lines)


# --- the real engine -------------------------------------------------------------------------

def _engine():
    for path in (ROOT / "Scripts/rag_ingest", ROOT / "Scripts"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    from core import market_service, team_resolution
    return market_service, team_resolution


def engine_evaluate(event, league, **kwargs):
    market_service, _ = _engine()
    return market_service.evaluate_event(event, league, market_service.SUPPORTED_MARKETS, **kwargs)


def engine_reset():
    market_service, team_resolution = _engine()
    team_resolution.clear_profile_caches()
    market_service.invalidate_statistical_cache()


def collect_fixtures(dates, per_league, leagues=LEAGUES):
    """Fetch and enrich real events the way the Match Read worker does (live API calls)."""
    _engine()
    from core.events import fetch_events, filter_events_by_exact_date
    from core.parlay import enrich_events_for_groups
    from rendering.match_read_dispatch import CANONICAL_ODDS_GROUPS
    fixtures = []
    for day in dates:
        for league in leagues:
            events, _notes = fetch_events(league, target_date=day)
            events = filter_events_by_exact_date(events or [], day)[:per_league]
            if not events:
                continue
            enriched, _notes = enrich_events_for_groups(events, league, set(CANONICAL_ODDS_GROUPS))
            for event in enriched or events:
                fixtures.append(dict(league=league, fixture_date=day.isoformat(), event=_plain(event),
                                     kwargs={"context": {"delivery_surface": "match_read_shadow",
                                                         "match_read_stage": "pre_match"}}))
    return fixtures


def _importance(change):
    """Picks first, then projections, then everything else (audit detail)."""
    path = change[0]
    return (0 if ".decision." in path else 1 if ".projection." in path else 2, path)


def summarize(report, limit=12):
    lines = []
    for entry in report:
        if "error" in entry:
            lines.append(f"{entry['fixture']}: ERROR {entry['error']}")
        for miss in entry.get("missing_reads", [])[:limit]:
            lines.append(f"{entry['fixture']}: read not in bundle: {miss[:240]}")
        changes = entry.get("changes") or []
        if changes:
            lines.append(f"{entry['fixture']}: {len(changes)} changed values")
        for path, old, new in sorted(changes, key=_importance)[:limit]:
            lines.append(f"    {path}: {old!r} -> {new!r}")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["record", "check", "preview"])
    parser.add_argument("--bundle", type=Path, default=BUNDLE)
    parser.add_argument("--date", action="append", type=date.fromisoformat, default=[],
                        help="fixture date to record (repeatable)")
    parser.add_argument("--per-league", type=int, default=4)
    parser.add_argument("--live-documents", action="store_true",
                        help="check: read documents from the configured store (DOCUMENT_STORE), replay only HTTP")
    args = parser.parse_args(argv)
    if args.command == "record":
        if not args.date:
            parser.error("record needs at least one --date")
        fixtures = collect_fixtures(args.date, args.per_league)
        market_service, _ = _engine()
        bundle = record(fixtures, engine_evaluate, args.bundle, generated_at="2000-01-01T00:00:00Z",
                        reset=engine_reset, system_version=market_service.prediction_system_manifest()["version"])
        print(f"recorded {len(bundle['fixtures'])} fixtures, {len(bundle['reads']['chroma'])} Chroma reads, "
              f"{len(bundle['reads']['http'])} HTTP reads -> {args.bundle}")
        return 0
    if args.command == "preview":
        print(summarize_preview(preview(args.bundle, engine_evaluate, reset=engine_reset)))
        return 0
    report = check(args.bundle, engine_evaluate, reset=engine_reset,
                   chroma_targets=[] if args.live_documents else None)
    fixtures = len(load(args.bundle)["fixtures"])
    print(summarize(report) or f"no changes across {fixtures} fixtures")
    return 1 if report else 0


if __name__ == "__main__":
    raise SystemExit(main())
