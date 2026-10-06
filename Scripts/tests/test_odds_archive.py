from copy import deepcopy
from datetime import datetime, timedelta, timezone
import gzip
import hashlib
import json
import plistlib
import sqlite3

import pytest
import requests

from Scripts.ops import odds_archive as archive
from Scripts.ops import odds_archive_launchd as schedule
from Scripts.ops.match_read_launchd import build_paths

NOW = datetime(2026, 10, 6, 9, tzinfo=timezone.utc)
PARAMS = dict(league=39, season=2026, date="2026-10-06", page=1)


def payload(params=None, *, pages=1, empty=False):
    params = params or PARAMS
    return dict(get="odds", parameters={k: str(v) for k, v in params.items()}, errors=[],
                results=0 if empty else 1, paging=dict(current=params["page"], total=pages),
                response=[] if empty else [dict(league=dict(id=params["league"], season=params["season"]),
                    fixture=dict(id=123, date=f'{params["date"]}T18:00:00+00:00'),
                    update="2026-10-06T08:00:00+00:00", bookmakers=[dict(id=8, name="Book",
                        bets=[dict(id=5, name="Goals Over/Under", values=[
                            dict(value="Over 2.5", odd="2.075"), dict(value="Under 2.5", odd=None)])])])])


class Response:
    def __init__(self, data, status=200, headers=None):
        self.content = json.dumps(data).encode() if not isinstance(data, bytes) else data
        self.status_code = status
        self.headers = requests.structures.CaseInsensitiveDict(headers or {"x-ratelimit-requests-remaining": "74000"})
        self.closed = False

    def close(self):
        self.closed = True


class Transport:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def get(self, url, **kwargs):
        assert url == archive.URL
        assert kwargs["allow_redirects"] is False
        self.calls.append(kwargs)
        result = next(self.responses)
        if isinstance(result, Exception):
            raise result
        return result


@pytest.fixture
def db():
    connection = sqlite3.connect(":memory:")
    connection.execute("PRAGMA foreign_keys=ON")
    archive.initialize(connection)
    yield connection
    connection.close()


def collect(db, responses, **settings):
    client = Transport(responses)
    result = archive.collect(db, "test-only-key", archive.config(["EPL"], days_ahead=0, **settings),
        session=client, clock=lambda: NOW, sleep=lambda _: None)
    return result, client


def all_quotes(db):
    return [quote for row in db.execute("SELECT id FROM requests WHERE status='ok'")
            for quote in archive.quote_rows(db, row[0])]


def test_exact_raw_prices_nulls_source_time_and_multiple_observations(db):
    body = payload()
    response = Response(body)
    collect(db, [response])
    collect(db, [Response(body)])
    assert response.closed
    assert db.execute("SELECT count(*) FROM raw_bodies").fetchone()[0] == 1
    assert db.execute("SELECT count(*) FROM requests").fetchone()[0] == 2
    quotes = all_quotes(db)
    assert len(quotes) == 4
    assert db.execute("SELECT count(*) FROM quotes").fetchone()[0] == 0
    assert db.execute("SELECT sum(quote_count) FROM fixture_snapshots").fetchone()[0] == 4
    row = (quotes[0][13], quotes[0][6], quotes[0][11])
    assert row == ("2.075", 3600, '"Over 2.5"')
    assert (quotes[1][13], quotes[1][12]) == (None, "null")
    digest, compressed = db.execute("SELECT * FROM raw_bodies").fetchone()
    assert gzip.decompress(compressed) == response.content
    assert hashlib.sha256(response.content).hexdigest() == digest
    for sql in ("DELETE FROM fixture_snapshots", "UPDATE raw_bodies SET sha256='changed'", "DELETE FROM requests"):
        with pytest.raises(sqlite3.IntegrityError):
            db.execute(sql)


def test_all_books_and_market_labels_preserved():
    data = payload()
    second = deepcopy(data["response"][0]["bookmakers"][0])
    second.update(id=9, name="Other Book")
    second["bets"][0].update(id=80, name="Cards Over/Under")
    data["response"][0]["bookmakers"].append(second)
    _, quotes = archive.normalize(data, PARAMS, NOW)
    assert len(quotes) == 4
    assert quotes[2][7:11] == (9, "Other Book", 80, "Cards Over/Under")


@pytest.mark.parametrize("field,value,issue", [
    ("update", None, "missing_source_update"),
    ("update", "2026-10-06T10:00:00+00:00", "future_source_update"),
    ("update", "2026-10-06T08:00:00", "missing_source_update"),
])
def test_missing_and_future_timestamps_never_replaced_with_fetch_time(field, value, issue):
    data = payload()
    data["response"][0][field] = value
    _, quotes = archive.normalize(data, PARAMS, NOW)
    assert issue in json.loads(quotes[0][-1])
    if issue == "missing_source_update":
        assert quotes[0][5:7] == (None, None)


def test_post_kickoff_receipt_cannot_be_prematch_evidence():
    _, quotes = archive.normalize(payload(), PARAMS, NOW + timedelta(hours=10))
    assert "observed_at_or_after_kickoff" in json.loads(quotes[0][-1])


def test_complete_pagination(db):
    second_params = {**PARAMS, "page": 2}
    second = payload(second_params, pages=2)
    second["response"][0]["fixture"]["id"] = 456
    result, client = collect(db, [Response(payload(pages=2)), Response(second)])
    assert result["status"] == "complete" and len(client.calls) == 2
    assert db.execute("SELECT count(DISTINCT fixture_id) FROM fixture_snapshots").fetchone()[0] == 2


def test_incomplete_pagination_preserves_received_evidence(db):
    result, client = collect(db, [Response(payload(pages=2))], max_requests=1)
    assert result["status"] == "incomplete"
    assert result["stop_reason"] == "local_request_budget"
    assert len(all_quotes(db)) == 2


def test_changed_pagination_fails_scope(db):
    result, _ = collect(db, [Response(payload(pages=2)), Response(payload({**PARAMS, "page": 2}, pages=3))])
    assert result["status"] == "incomplete"
    assert db.execute("SELECT reason FROM scopes").fetchone()[0] == "pagination_changed"


def test_duplicate_fixture_across_pages_is_incomplete(db):
    result, _ = collect(db, [Response(payload(pages=2)), Response(payload({**PARAMS, "page": 2}, pages=2))])
    assert result["status"] == "incomplete"
    assert db.execute("SELECT reason FROM scopes").fetchone()[0] == "duplicate_fixture_across_pages"


def test_interrupted_run_is_recorded_without_losing_receipts(db):
    with db:
        db.execute("INSERT INTO runs VALUES('old',?,NULL,'running','{}','test')", (archive.stamp(NOW),))
        db.execute("INSERT INTO requests(id,run_id,requested_at,params_json,status) VALUES('old_request','old',?,'{}','pending')", (archive.stamp(NOW),))
    collect(db, [Response(payload())])
    assert db.execute("SELECT status FROM runs WHERE id='old'").fetchone()[0] == "interrupted"
    assert db.execute("SELECT error_code FROM requests WHERE id='old_request'").fetchone()[0] == "interrupted"


@pytest.mark.parametrize("body,status", [
    (b"not json", 200), (b"unavailable", 503),
    ({**payload(), "errors": {"requests": "quota"}}, 200),
    ({**payload(), "results": 5}, 200),
    ({**payload(), "parameters": {}}, 200),
])
def test_failures_remain_raw_evidence_not_empty_success(db, body, status):
    result, _ = collect(db, [Response(body, status=status)])
    assert result["status"] == "incomplete"
    assert db.execute("SELECT count(*) FROM raw_bodies").fetchone()[0] == 1
    assert db.execute("SELECT count(*) FROM quotes").fetchone()[0] == 0


def test_successful_empty_scope_is_visible(db):
    result, _ = collect(db, [Response(payload(empty=True))])
    assert result["status"] == "complete"
    assert db.execute("SELECT count(*) FROM quotes").fetchone()[0] == 0
    assert db.execute("SELECT status FROM scopes").fetchone()[0] == "complete"


def test_quota_floor_persists_across_runs(db):
    collect(db, [Response(payload(), headers={"x-ratelimit-requests-remaining": "999"})])
    result, client = collect(db, [])
    assert not client.calls and result["stop_reason"] == "provider_quota_reserve"


def test_daily_cap_counts_failed_requests_and_restart(db):
    collect(db, [requests.Timeout("Do not print secrets")], daily_budget=1)
    result, client = collect(db, [], daily_budget=1)
    assert not client.calls and result["stop_reason"] == "local_request_budget"
    assert db.execute("SELECT error_code FROM requests").fetchone()[0] == "transport_error"


def test_secret_headers_never_stored(db):
    collect(db, [Response(payload(), headers={"x-apisports-key": "private", "Set-Cookie": "private", "Date": "date"})])
    assert json.loads(db.execute("SELECT headers_json FROM requests").fetchone()[0]) == {"date": "date"}


def test_read_only_defaults_and_status(tmp_path):
    result = archive.run(tmp_path)
    assert result["minimum_requests_per_cycle"] == 195
    assert result["contract"]["training_authorized"] is False
    assert not archive.archive_path(tmp_path).exists()
    assert archive.run(tmp_path, show_status=True)["initialized"] is False
    assert not (tmp_path / "Index").exists()


def test_refuse_alias_and_unrecognized_database(tmp_path):
    (tmp_path / "Index").symlink_to(tmp_path)
    with pytest.raises(ValueError):
        archive.archive_path(tmp_path)
    db = sqlite3.connect(":memory:")
    db.execute("CREATE TABLE production(x)")
    with pytest.raises(ValueError):
        archive.initialize(db)
    db.close()


@pytest.mark.parametrize("value", [None, True, 0, 1, -2, "NaN", "Infinity", "bad"])
def test_invalid_prices_never_become_zero(value):
    assert archive.price(value) is None


def test_schedule_is_separate_and_has_no_secrets_or_publication(tmp_path):
    paths = build_paths(root=tmp_path, plist_path=tmp_path / "job.plist")
    settings = plistlib.loads(schedule.render(paths).encode())
    assert settings["StartInterval"] == 10800
    assert settings["Label"] == "com.bettingrag.odds-archive"
    assert settings["ProgramArguments"][-2:] == ["Scripts.ops.odds_archive", "--execute"]
    assert "EnvironmentVariables" not in settings
    assert schedule.operate("install", paths, dry_run=True)["action"] == "install"
    assert not paths.plist_path.exists()


def test_minute_quota_stops_between_scopes(db):
    client = Transport([Response(payload(), headers={"x-ratelimit-remaining": "0"})])
    result = archive.collect(db, "test", archive.config(["EPL", "LaLiga"], days_ahead=0),
        session=client, clock=lambda: NOW, sleep=lambda _: None)
    assert result["stop_reason"] == "provider_minute_quota"
    assert result["incomplete_scopes"] == 1 and len(client.calls) == 1


def test_compact_status_and_disk_guard(tmp_path, monkeypatch):
    path = tmp_path / "odds.sqlite3"
    db = sqlite3.connect(path)
    archive.initialize(db)
    collect(db, [Response(payload())])
    report = archive.status(path)
    assert (report["quotes"], report["fixtures"], report["requests"]) == (2, 1, 1)
    monkeypatch.setattr(archive.shutil, "disk_usage", lambda _: type("Usage", (), {"free": 1000})())
    result, client = collect(db, [])
    assert result["stop_reason"] == "low_disk_space" and not client.calls
    assert archive.status(path)["quotes"] == 2
    db.close()


def test_lock_prevents_overlapping_provider_requests(tmp_path, monkeypatch):
    monkeypatch.setenv("API_FOOTBALL_KEY", "test-only")
    path = archive.archive_path(tmp_path)
    path.parent.mkdir(parents=True)
    with (path.parent / ".collector.lock").open("a") as lock:
        archive.fcntl.flock(lock, archive.fcntl.LOCK_EX | archive.fcntl.LOCK_NB)
        assert archive.run(tmp_path, execute=True)["status"] == "already_running"
        assert not path.exists()


def test_missing_fixture_metadata_still_preserves_raw_receipt(db):
    data = payload()
    data["response"][0]["fixture"] = None
    result, _ = collect(db, [Response(data)])
    assert result["status"] == "complete"
    assert "missing_fixture_identity" in json.loads(all_quotes(db)[0][-1])


def test_malformed_bookmaker_name_is_preserved_as_failed_response(db):
    data = payload()
    data["response"][0]["bookmakers"][0]["name"] = {"bad": "shape"}
    result, _ = collect(db, [Response(data)])
    assert result["status"] == "incomplete"
    assert db.execute("SELECT count(*) FROM raw_bodies").fetchone()[0] == 1
