"""Bounded, validated API-Football downloads shared by refresh collectors."""
from __future__ import annotations

import logging
import time
from urllib.parse import urlsplit

import requests

logger = logging.getLogger(__name__)
REQUEST_TIMEOUT = (5, 30)  # connect, socket read; stage deadline bounds total work
MAX_ATTEMPTS = 3


class ApiFootballResponseError(RuntimeError):
    """A failed/partial response must never masquerade as an empty dataset."""


def validate_envelope(data, path):
    if not isinstance(data, dict) or not isinstance(data.get("response"), list):
        raise ApiFootballResponseError(f"API-Football {path}: invalid response envelope")
    if data.get("errors"):
        raise ApiFootballResponseError(f"API-Football {path}: {data['errors']}")
    rows = data["response"]
    if "results" in data and data["results"] != len(rows):
        raise ApiFootballResponseError(f"API-Football {path}: incomplete result count")
    paging = data.get("paging") or {}
    if paging.get("total", 1) != 1:
        raise ApiFootballResponseError(f"API-Football {path}: unconsumed pagination")
    return data


def get_json(url, *, params=None, headers=None, session=None, sleep=time.sleep):
    """At most three attempts. Retry transport/429/5xx, not auth or API errors.

    Do not log headers, request objects, or raw transport exception messages.
    Empty successful responses are provider coverage, not transport success
    with fabricated statistics; consumers must preserve prior coverage.
    """
    path = urlsplit(url).path
    transport = session or requests
    for attempt in range(MAX_ATTEMPTS):
        response = None
        try:
            response = transport.get(url, params=params or {}, headers=headers,
                                     timeout=REQUEST_TIMEOUT)
            status = response.status_code
            if status == 429 or 500 <= status < 600:
                failure = f"HTTP {status}"
            elif status != 200:
                raise ApiFootballResponseError(f"API-Football {path}: HTTP {status}")
            else:
                try:
                    data = response.json()
                except ValueError:
                    raise ApiFootballResponseError(f"API-Football {path}: invalid JSON") from None
                return validate_envelope(data, path)
        except (requests.Timeout, requests.ConnectionError) as exc:
            failure = type(exc).__name__
        finally:
            if response is not None:
                response.close()
        if attempt + 1 < MAX_ATTEMPTS:
            logger.warning("API-Football %s %s; retry %s/%s", path, failure, attempt + 2, MAX_ATTEMPTS)
            sleep(2 ** attempt)
    raise ApiFootballResponseError(f"API-Football {path} failed after {MAX_ATTEMPTS} attempts ({failure})")


def legacy_json(url, *, headers, params):
    """Validate team/player blocks before a season-level file is replaced."""
    data = get_json(url, headers=headers, params=params)
    return validate_stat_blocks(data, urlsplit(url).path, params=params)


def validate_stat_blocks(data, path, *, params=None):
    """Nonempty fixture-detail downloads must cover both teams."""
    params = params or {}
    rows = data["response"]
    if path in {"/fixtures/statistics", "/fixtures/players"}:
        if not rows:
            logger.warning("API-Football %s fixture=%s: provider has no statistics", path, params.get("fixture"))
            return data
        ids = {(row.get("team") or {}).get("id") for row in rows}
        field = "players" if path.endswith("players") else "statistics"
        if len(rows) != 2 or len(ids) != 2 or None in ids or any(
            not isinstance(row.get(field), list) or not row[field] for row in rows
        ):
            raise ApiFootballResponseError(f"API-Football {path} fixture={params.get('fixture')}: incomplete team blocks")
        if field == "players" and any(
            not entry.get("statistics") for row in rows for entry in row["players"]
        ):
            raise ApiFootballResponseError(f"API-Football {path}: incomplete player statistics")
        if field == "players" and any(len(row["players"]) < 11 for row in rows):
            # Some qualifiers have sparse but structurally valid provider
            # coverage, also present in the historical baseline. Don't turn
            # this transport fix into a new model-eligibility policy.
            logger.warning("API-Football %s fixture=%s: limited player coverage (team roster sizes %s)",
                           path, params.get("fixture"), [len(row["players"]) for row in rows])
    return data


def preserve_fixture_coverage(rows, output_dir, pattern):
    """Refuse to erase previously collected fixture/team coverage.

    Only compare the selected season's per-fixture files, never aggregates or
    other seasons. Provider-null values stay null; this is not imputation.
    Do not freeze player IDs or participation minutes: the provider corrects
    those, including replacing placeholder IDs. Raw roster structure is
    validated and sparse coverage warned about before the minutes filter.
    """
    import json
    def keys(items):
        return {(str(row.get("fixture")), str(row.get("team"))) for row in items}
    current = keys(rows)
    for path in output_dir.glob(pattern):
        previous = json.loads(path.read_text())
        missing = keys(previous) - current
        if missing:
            raise ApiFootballResponseError(
                f"Refusing incomplete refresh: {len(missing)} prior fixture/team entries absent from {path.name}; old file retained"
            )
