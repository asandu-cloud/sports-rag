"""Bounded Phase 1 diagnostic. Writes evidence only, never production data.

prepare: read-only local audit + reproducible sample, no API calls.
fetch: explicitly request sampled statistics/events and competition coverage;
       durable 120-attempt cap, no redirects, no automatic transport retries.
analyze: compare the frozen local snapshot with saved provider responses.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from Scripts.data_platform.features.history_diagnostic import audit_snapshot, choose_sample, compare_fixture, snapshot
from Scripts.data_platform.features.model_dataset import supported_competitions
from Scripts.data_platform.sync.apifootball import API_BASE, COMPETITIONS
from Scripts.ops.prediction_features import experiment_directory
from Scripts.ops.prediction_history import digest, write


MAX_ATTEMPTS = 120


def prepare(root, database, name, seasons, sample_size=52):
    leagues = list(supported_competitions())
    minimum_requests = len(leagues) + sample_size * 2
    if minimum_requests > MAX_ATTEMPTS:
        raise ValueError("Sample plus coverage requests exceeds the hard 120-request cap")
    data = snapshot(database, seasons)
    sample = choose_sample(data, leagues, size=sample_size)
    target = experiment_directory(root, name)
    target.chmod(0o700)
    write(target / "local-snapshot.json", data)
    write(target / "local-audit.json", audit_snapshot(data, leagues))
    write(target / "sample.json", sample)
    requests = [{"endpoint": "/leagues", "params": {"id": COMPETITIONS[league].api_football_id}} for league in leagues]
    requests += [{"endpoint": endpoint, "params": {"fixture": item["fixture_id"]}}
                 for item in sample["fixtures"] for endpoint in ("/fixtures/statistics", "/fixtures/events")]
    write(target / "request-plan.json", {"max_attempts": MAX_ATTEMPTS, "requests": requests,
                                        "includes_retries_and_failed_attempts": True})
    write(target / "PREPARED.json", {p.name: digest(p) for p in sorted(target.glob("*.json"))})
    return target


def verify_prepared(path):
    expected = {"local-snapshot.json", "local-audit.json", "sample.json", "request-plan.json"}
    manifest = json.loads((path / "PREPARED.json").read_text())
    if set(manifest) != expected:
        raise ValueError("Unexpected diagnostic artifacts")
    for name, checksum in manifest.items():
        if (path / name).is_symlink() or digest(path / name) != checksum:
            raise ValueError("Diagnostic artifact checksum mismatch: " + name)
    plan = json.loads((path / "request-plan.json").read_text())
    if plan["max_attempts"] != MAX_ATTEMPTS or len(plan["requests"]) > MAX_ATTEMPTS:
        raise ValueError("Invalid request budget")
    return plan


def request_key(request):
    return json.dumps(request, sort_keys=True, separators=(",", ":"))


class RequestBudgetExceeded(RuntimeError):
    pass


class EvidenceClient:
    """Call only while holding the experiment lock; reserve before sending.

    Failed/interrupted requests consume an attempt; restarts cannot reset the
    budget. API keys and request headers are never written to the evidence log.
    """
    def __init__(self, directory, session, *, limit=MAX_ATTEMPTS, pause=0.4):
        if not 0 < limit <= MAX_ATTEMPTS:
            raise ValueError("Invalid hard request cap")
        self.directory = directory
        directory.mkdir(exist_ok=True)
        self.session, self.limit, self.pause = session, limit, pause

    @property
    def attempts(self):
        return len(list(self.directory.glob("*.request.json")))

    def successes(self):
        result = {}
        for receipt in sorted(self.directory.glob("*.receipt.json")):
            info = json.loads(receipt.read_text())
            if not info.get("success"):
                continue
            request_file = receipt.with_name(receipt.name.replace(".receipt.json", ".request.json"))
            request = json.loads(request_file.read_text())["request"]
            body = receipt.with_name(receipt.name.replace(".receipt.json", ".body"))
            if body.is_symlink() or digest(body) != info["body_sha256"]:
                raise ValueError("Provider evidence checksum mismatch")
            result[request_key(request)] = json.loads(body.read_bytes())
        return result

    def fetch(self, request):
        if self.attempts >= self.limit:
            raise RequestBudgetExceeded("Hard diagnostic request budget exhausted")
        if request["endpoint"] not in {"/leagues", "/fixtures/statistics", "/fixtures/events"}:
            raise ValueError("Endpoint outside diagnostic scope")
        start = datetime.now(timezone.utc).isoformat()
        stem = f"{self.attempts + 1:03d}"
        write(self.directory / (stem + ".request.json"), {"request": request, "started_at": start})
        info = {"success": False}
        try:
            response = self.session.get(API_BASE + request["endpoint"], params=request["params"],
                                        timeout=30, allow_redirects=False)
            body = response.content
            with (self.directory / (stem + ".body")).open("xb") as handle:
                handle.write(body)
            info.update(status_code=response.status_code, body_sha256=hashlib.sha256(body).hexdigest(),
                        quota={key: value for key, value in response.headers.items() if key.lower() in {
                            "x-ratelimit-requests-remaining", "x-ratelimit-requests-limit",
                            "x-ratelimit-remaining", "x-ratelimit-limit"}})
            if response.status_code != 200:
                raise RuntimeError(f"Provider HTTP status {response.status_code}; evidence saved, stopping")
            payload = json.loads(body)
            if not isinstance(payload, dict) or payload.get("errors") or not isinstance(payload.get("response"), list):
                raise RuntimeError("Provider error/invalid envelope; evidence saved, stopping")
            pagination = payload.get("paging") or {}
            if pagination.get("total", 1) != 1:
                raise RuntimeError("Unexpected pagination; no automatic extra requests allowed")
            info["success"] = True
        except Exception as exc:
            # No raw headers, session repr or credential-bearing exceptions.
            info["error_type"] = type(exc).__name__
            raise
        finally:
            info["completed_at"] = datetime.now(timezone.utc).isoformat()
            write(self.directory / (stem + ".receipt.json"), info)
        if self.pause:
            time.sleep(self.pause)
        return payload


@contextmanager
def evidence_lock(path):
    with (path / ".fetch.lock").open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def fetch(path):
    import requests
    from Scripts.data_platform.config import SETTINGS
    plan = verify_prepared(path)
    if not SETTINGS.api_football_key:
        raise ValueError("API-Football key is not configured")
    with evidence_lock(path), requests.Session() as session:
        # A plain requests session has no automatic transport retries; each
        # send is accounted for here, and redirects cannot add uncounted calls.
        session.headers.update({"x-apisports-key": SETTINGS.api_football_key})
        client = EvidenceClient(path / "provider", session)
        done = client.successes()
        for request in plan["requests"]:
            key = request_key(request)
            if key in done:
                continue
            print(f"Attempt {client.attempts + 1}/{MAX_ATTEMPTS}: {request['endpoint']} {request['params']}", flush=True)
            done[key] = client.fetch(request)
        print(json.dumps({"completed_requests": len(done), "attempts_including_failures": client.attempts}))


def analyze(path):
    plan = verify_prepared(path)
    if not (path / "provider").is_dir():
        raise ValueError("No provider evidence captured")
    client = EvidenceClient(path / "provider", None)
    responses = client.successes()
    pending = [r for r in plan["requests"] if request_key(r) not in responses]
    local = json.loads((path / "local-snapshot.json").read_text())
    fixtures = {f["fixture_id"]: f for f in local["fixtures"]}
    sample = json.loads((path / "sample.json").read_text())
    comparisons = []
    for item in sample["fixtures"]:
        stats = responses.get(request_key({"endpoint": "/fixtures/statistics", "params": {"fixture": item["fixture_id"]}}))
        events = responses.get(request_key({"endpoint": "/fixtures/events", "params": {"fixture": item["fixture_id"]}}))
        if stats is not None and events is not None:
            result = compare_fixture(fixtures[item["fixture_id"]], stats["response"], events["response"])
            result["selection"] = item["selection"]
            comparisons.append(result)
    totals, by_source, by_selection = Counter(), {}, {}
    for result in comparisons:
        source = by_source.setdefault(result.get("source", "unavailable"), Counter())
        selection = by_selection.setdefault("random_control" if result["selection"] == "random_control" else "targeted_or_extra", Counter())
        for entry in result.get("comparisons", []):
            key = entry["field"] + ":" + entry["comparison"]
            totals[key] += 1
            source[key] += 1
            selection[key] += 1
    coverage = {}
    for request in plan["requests"]:
        if request["endpoint"] != "/leagues":
            continue
        response = responses.get(request_key(request))
        if response:
            for league in response["response"]:
                if league.get("league", {}).get("id") != request["params"]["id"]:
                    raise ValueError("Coverage response league ID mismatch")
                for season in league.get("seasons", []):
                    if season["year"] in local["seasons"]:
                        coverage[f"{request['params']['id']}:{season['year']}"] = season.get("coverage")
    compared = sum(bool(r["comparisons"]) for r in comparisons)
    report = {"schema": "history-diagnostic.v2", "analyzed_at": datetime.now(timezone.utc).isoformat(),
              "sample_size": len(sample["fixtures"]), "processed_fixtures": len(comparisons),
              "compared_fixtures": compared,
              "valid_statistics_pairs": sum(r["valid_statistics_pair"] for r in comparisons),
              "unavailable_statistics": dict(Counter(r["issue"] for r in comparisons if r.get("issue"))),
              "compared_team_sides": sum(len({c["team_id"] for c in r["comparisons"]}) for r in comparisons),
              "attempts_including_failures": client.attempts, "successful_requests": len(responses),
              "pending_requests": pending, "comparison_totals": totals, "by_source": by_source,
              "by_selection": by_selection, "provider_coverage": coverage, "fixtures": comparisons,
              "production_writes": False, "automatic_repairs": False,
              "source_sha256": {str(p.relative_to(ROOT)): digest(p) for p in (
                  Path(__file__).resolve(), ROOT / "Scripts/data_platform/features/history_diagnostic.py")},
              "prepared_manifest_sha256": digest(path / "PREPARED.json"),
              "qualification": "Diagnostic sample only. No automatic null-to-zero rule or historical-vintage claim."}
    # New analysis names permit offline re-analysis without replacing evidence.
    index = len(list(path.glob("analysis-*.json"))) + 1
    destination = path / f"analysis-{index:03d}.json"
    write(destination, report)
    print(json.dumps({"report": str(destination), "sample_size": report["sample_size"],
                      "processed_fixtures": len(comparisons), "compared_fixtures": compared,
                      "unavailable_statistics": report["unavailable_statistics"], "pending_requests": len(pending),
                      "attempts_including_failures": client.attempts}, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--experiment", required=True)
    p.add_argument("--database", type=Path, default=ROOT / "Index/platform.db")
    p.add_argument("--season", type=int, nargs="+", default=[2025, 2026])
    p.add_argument("--sample-size", type=int, default=52)
    for name in ("fetch", "analyze"):
        p = sub.add_parser(name)
        p.add_argument("--experiment", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        print(prepare(ROOT, args.database, args.experiment, args.season, args.sample_size))
    elif args.command == "fetch":
        fetch(args.experiment)
    else:
        analyze(args.experiment)


if __name__ == "__main__":
    main()
