"""No external calls: transport errors, old-file retention and hung child cleanup."""
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import time
from unittest.mock import Mock

import pytest
import requests

from Scripts import football_http as http
from Scripts.pull_season import LEAGUES
from Scripts.refresh_runtime import run_bounded


def response(status=200, payload=None):
    result = Mock(status_code=status)
    result.json.return_value = payload if payload is not None else {"response": [], "errors": [], "results": 0}
    return result


@pytest.mark.parametrize("failure", [requests.ReadTimeout(), requests.ConnectionError(), response(429), response(503)])
def test_network_failure_has_three_attempts_and_explicit_timeout(failure):
    session = Mock()
    session.get.side_effect = [failure, failure, failure]
    sleep = Mock()
    with pytest.raises(http.ApiFootballResponseError, match="3 attempts"):
        http.get_json("https://example.test/fixtures", session=session, sleep=sleep)
    assert session.get.call_count == 3
    assert all(c.kwargs["timeout"] == (5, 30) for c in session.get.call_args_list)
    assert [c.args[0] for c in sleep.call_args_list] == [1, 2]


@pytest.mark.parametrize("result", [response(401), response(403),
    response(payload={"errors": {"quota": "exceeded"}, "response": []}),
    response(payload={"response": [], "results": 2}),
    response(payload={"response": [], "paging": {"total": 2}}),
    response(payload={"response": None})])
def test_invalid_download_is_not_success_or_retried(result):
    session = Mock()
    session.get.return_value = result
    with pytest.raises(http.ApiFootballResponseError):
        http.get_json("https://example.test/fixtures", session=session)
    assert session.get.call_count == 1


def test_successful_retry_closes_responses():
    session = Mock()
    first, second = response(503), response()
    session.get.side_effect = [first, second]
    assert http.get_json("https://example.test/fixtures", session=session, sleep=lambda _: None)["response"] == []
    first.close.assert_called_once()
    second.close.assert_called_once()


def test_partial_team_blocks_rejected(monkeypatch):
    monkeypatch.setattr(http, "get_json", lambda *a, **kw: {"response": [{"team": {"id": 1}, "statistics": [1]}]})
    with pytest.raises(http.ApiFootballResponseError, match="incomplete team blocks"):
        http.legacy_json("https://example.test/fixtures/statistics", headers={}, params={"fixture": 5})


def test_canonical_client_rejects_incomplete_detail_download():
    from Scripts.data_platform.sync.apifootball import ApiFootballClient
    session = Mock()
    session.get.return_value = response(payload={"response": [{"team": {"id": 1}, "statistics": [1]}]})
    client = ApiFootballClient(api_key="test", session=session, request_pause_s=0)
    with pytest.raises(http.ApiFootballResponseError, match="incomplete team blocks"):
        client.fixture_statistics(5)


def test_coverage_loss_keeps_previous_files(tmp_path):
    path = tmp_path / "team_fixture_stats_2026.json"
    path.write_text(json.dumps([{"fixture": "A vs B", "team": "A"}]))
    before = path.read_bytes()
    with pytest.raises(http.ApiFootballResponseError, match="old file retained"):
        http.preserve_fixture_coverage([], tmp_path, "*team_fixture_stats_2026.json")
    assert path.read_bytes() == before
    http.preserve_fixture_coverage([{"fixture": "A vs B", "team": "A"}], tmp_path, "*team_fixture_stats_2026.json")


def test_corrected_player_ids_and_minutes_do_not_falsely_block_refresh(tmp_path):
    rows = [{"fixture": "A vs B", "team": "A", "player_id": i} for i in (0, 2)]
    path = tmp_path / "player_fixture_stats_2026.json"
    path.write_text(json.dumps(rows))
    http.preserve_fixture_coverage([{"fixture": "A vs B", "team": "A", "player_id": 659107}],
                                  tmp_path, "*player_fixture_stats_2026.json")


def test_sparse_provider_coverage_is_explicit_without_changing_model_policy(monkeypatch, caplog):
    payload = {"response": [{"team": {"id": i}, "players": [{"statistics": [{}]}] * 10} for i in (1, 2)]}
    monkeypatch.setattr(http, "get_json", lambda *a, **kw: payload)
    assert http.legacy_json("https://example.test/fixtures/players", headers={}, params={"fixture": 5}) == payload
    assert "limited player coverage" in caplog.text


def test_malformed_player_stats_are_rejected(monkeypatch):
    payload = {"response": [{"team": {"id": i}, "players": [{"statistics": []}]} for i in (1, 2)]}
    monkeypatch.setattr(http, "get_json", lambda *a, **kw: payload)
    with pytest.raises(http.ApiFootballResponseError, match="incomplete player statistics"):
        http.legacy_json("https://example.test/fixtures/players", headers={}, params={"fixture": 5})


COLLECTORS = [config[k] for config in LEAGUES.values() for k in ("team_stats", "player_stats")]


@pytest.mark.parametrize("script", COLLECTORS)
def test_every_active_collector_stops_before_saving_partial_download(script, monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", [script, "--season", "2026"])
    module = runpy.run_path(script)
    main = module["main"]
    namespace = main.__globals__
    namespace["get_fixture_info"] = lambda: {1: {}, 2: {}}
    field = "fetch_fixture_player_stats" if "fetch_fixture_player_stats" in namespace else "fetch_fixture_team_stats"
    namespace[field] = Mock(side_effect=[[{"fixture": "A vs B", "team": "A"}], requests.ReadTimeout()])
    namespace["OUTPUT_DIR"] = tmp_path
    namespace["save_outputs"] = save = Mock()
    namespace["save_team_outputs"] = save
    monkeypatch.setattr(time, "sleep", lambda _: None)
    with pytest.raises(requests.ReadTimeout):
        main()
    save.assert_not_called()
    assert "requests.get(" not in Path(script).read_text()


def test_deadline_kills_grandchild_and_releases_inherited_lock(tmp_path, monkeypatch):
    import fcntl
    monkeypatch.delenv("BETTING_REFRESH_STAGE_GROUP", raising=False)
    lock_path = tmp_path / "gate"
    pid_path = tmp_path / "pid"
    child = ("import os,sys,subprocess,time; "
             "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'],pass_fds=(int(sys.argv[1]),)); "
             "open(sys.argv[2],'w').write(str(p.pid)); time.sleep(60)")
    with lock_path.open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        started = time.monotonic()
        with pytest.raises(subprocess.TimeoutExpired):
            run_bounded([sys.executable, "-c", child, str(handle.fileno()), str(pid_path)],
                        cwd=str(tmp_path), timeout=0.5, pass_fds=(handle.fileno(),))
        assert time.monotonic() - started < 5
    assert pid_path.exists()
    with lock_path.open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)


def test_nested_deadline_cleans_up_entire_outer_stage(tmp_path, monkeypatch):
    monkeypatch.delenv("BETTING_REFRESH_STAGE_GROUP", raising=False)
    root = str(Path(__file__).resolve().parents[2])
    nested = ("from Scripts.refresh_runtime import run_bounded; import sys; "
              "run_bounded([sys.executable,'-c','import time; time.sleep(60)'],cwd='.',timeout=0.1)")
    result = run_bounded([sys.executable, "-c", nested], cwd=root, timeout=5)
    assert result.returncode != 0


def test_referee_rebuild_fails_closed_but_live_lookup_can_degrade(monkeypatch):
    from Scripts.rag_ingest import referee_data
    monkeypatch.setattr(referee_data, "_api_football_key", lambda: "test")
    monkeypatch.setattr(referee_data, "get_json", Mock(side_effect=http.ApiFootballResponseError("failed")))
    with pytest.raises(http.ApiFootballResponseError):
        referee_data.fetch_completed_fixtures(39, 2026)
    assert referee_data._api_get("/fixtures", {}) is None
