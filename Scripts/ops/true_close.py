"""Capture true closing prices from OddsPapi for published picks (rework A1 step 4, tracker D18).

Runs ``data_platform.services.true_close`` over the measured cohort's published picks whose match is
over and that have no true close yet. Billable OddsPapi requests (fixture lists, one per competition and
week) are kept below the account's monthly quota minus a reserve, read from ``/v4/account`` (unmetered).
``--recapture`` replaces the records of the named fixtures (our fixture ids).

    python -m Scripts.ops.true_close [--limit 50] [--reserve 40] [--recapture 1635647 1635684]
"""
from __future__ import annotations

import argparse
import json
import os

from Scripts.ops.odds_archive import ROOT


class AccountUnavailable(RuntimeError):
    pass


def quota_budget(client, key, reserve):
    """Local billable count allowed for this run: what the account has left this month, minus ``reserve``."""
    import requests
    try:
        account = requests.get("https://api.oddspapi.io/v4/account", params={"apiKey": key}, timeout=30).json()
    except (requests.RequestException, ValueError) as exc:  # a RequestException's message holds the key
        raise AccountUnavailable(type(exc).__name__) from None
    active = next((s for s in account.get("subscriptions") or [] if s.get("is_active")), {})
    left = int(active.get("request_limit") or 0) - int(active.get("request_count") or 0)
    return client._usage()["billable"] + max(0, left - reserve), left


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, default=None, help="fixtures per run")
    parser.add_argument("--reserve", type=int, default=40, help="billable requests left unused each month")
    parser.add_argument("--recapture", nargs="+", default=(), metavar="FIXTURE_ID", help="replace these fixtures' records")
    args = parser.parse_args(argv)
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    key = os.environ.get("ODDSPAPI_API_KEY")
    if not key:
        parser.error("ODDSPAPI_API_KEY is not set")
    from Scripts.data_platform import oddspapi
    from Scripts.data_platform.repositories.predictions import PredictionRepository
    from Scripts.data_platform.services.measurement_runtime import CONTROL, MeasurementRuntime
    from Scripts.data_platform.services.refresh_coordination import RefreshBusy, data_access
    from Scripts.data_platform.services.true_close import TrueCloseService
    client = oddspapi.OddsPapiClient(key, compress=True)
    try:
        client.budget, left = quota_budget(client, key, args.reserve)
    except AccountUnavailable as exc:  # OddsPapi unreachable; the next run catches up
        print(json.dumps({"status": "oddspapi_unreachable", "reason": str(exc)}))
        return 0
    repo = PredictionRepository()
    boundary = int(MeasurementRuntime(repo._factory).state(CONTROL).get("after_recommendation_id") or 0)
    try:
        with data_access():
            rows = [r for r in repo.get_recent(days=3650, published_only=True, publication_scope="all", limit=10 ** 7)
                    if (r.get("recommendation_id") or 0) > boundary]
            service = TrueCloseService(repo, client, oddspapi.market_index(client.get("/v4/markets")))
            counts = service.run(rows, limit=args.limit, recapture={str(f) for f in args.recapture})
    except RefreshBusy as exc:  # the data refresh is running; the next run catches up
        print(json.dumps({"status": "refresh_deferred", "reason": str(exc)}))
        return 0
    print(json.dumps({**counts, "oddspapi_requests_left_before_run": left}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
