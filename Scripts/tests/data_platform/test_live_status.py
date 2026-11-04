"""Live status for started matches on the public board (D70): batched, kept five minutes, capped, never stored."""
from datetime import datetime, timedelta, timezone

from data_platform.services import live_status

NOW = datetime(2026, 11, 4, 20, 30, tzinfo=timezone.utc)


def fixture(event_id, minutes_ago):
    return {"event_id": str(event_id), "kickoff": (NOW - timedelta(minutes=minutes_ago)).isoformat()}


def row(event_id, short, elapsed=None):
    return {"fixture": {"id": int(event_id), "status": {"long": "x", "short": short, "elapsed": elapsed}}}


class Provider:
    def __init__(self, statuses=None, fail=False):
        self.calls, self.statuses, self.fail = [], dict(statuses or {}), fail

    def __call__(self, ids):
        self.calls.append(list(ids))
        if self.fail:
            raise RuntimeError("provider down")
        return [row(i, *self.statuses[i]) for i in ids if i in self.statuses]


def test_started_matches_are_asked_for_in_one_batch_and_kept_for_five_minutes():
    provider = Provider({"1": ("1H", 30), "2": ("HT", 45), "3": ("NS", None)})
    statuses = live_status.LiveStatuses(provider)
    board = [fixture(1, 30), fixture(2, 50), fixture(3, 5), fixture(4, -60)]  # 4 kicks off in an hour
    assert statuses.lookup(board, now=NOW) == {"1": {"short": "1H", "elapsed": 30},
                                               "2": {"short": "HT", "elapsed": 45},
                                               "3": {"short": "NS", "elapsed": None}}
    assert provider.calls == [["1", "2", "3"]]
    statuses.lookup(board, now=NOW + timedelta(minutes=4))
    assert len(provider.calls) == 1
    provider.statuses["2"] = ("2H", 50)
    later = statuses.lookup(board, now=NOW + timedelta(minutes=5))
    assert provider.calls[1] == ["1", "2", "3"] and later["2"] == {"short": "2H", "elapsed": 50}


def test_a_finished_or_called_off_match_is_never_asked_for_again():
    provider = Provider({"1": ("FT", 90), "2": ("PST", None), "3": ("2H", 70)})
    statuses = live_status.LiveStatuses(provider)
    board = [fixture(1, 120), fixture(2, 60), fixture(3, 80)]
    statuses.lookup(board, now=NOW)
    again = statuses.lookup(board, now=NOW + timedelta(minutes=30))
    assert provider.calls == [["1", "2", "3"], ["3"]]
    assert again["1"] == {"short": "FT", "elapsed": 90} and again["2"]["short"] == "PST"


def test_more_than_twenty_started_matches_take_one_request_per_twenty():
    provider = Provider({str(i): ("1H", 10) for i in range(45)})
    live_status.LiveStatuses(provider).lookup([fixture(i, 10) for i in range(45)], now=NOW)
    assert [len(call) for call in provider.calls] == [20, 20, 5]


def test_a_failed_request_keeps_what_was_known_and_waits_five_minutes_before_trying_again():
    provider = Provider({"1": ("1H", 20)})
    statuses = live_status.LiveStatuses(provider)
    board = [fixture(1, 20), fixture(2, 20)]
    statuses.lookup(board, now=NOW)
    provider.fail = True
    failed = statuses.lookup(board, now=NOW + timedelta(minutes=5))
    assert failed == {"1": {"short": "1H", "elapsed": 20}, "2": None}
    statuses.lookup(board, now=NOW + timedelta(minutes=6))
    assert len(provider.calls) == 2
    statuses.lookup(board, now=NOW + timedelta(minutes=10))
    assert len(provider.calls) == 3


def test_the_daily_cap_stops_requests_until_the_next_utc_day():
    provider = Provider({"1": ("1H", 5)})
    statuses = live_status.LiveStatuses(provider, daily_cap=2)
    board = [fixture(1, 5)]
    for minutes in (0, 5, 10, 15):
        statuses.lookup(board, now=NOW + timedelta(minutes=minutes))
    assert len(provider.calls) == 2
    statuses.lookup([fixture(1, 5)], now=NOW.replace(hour=0) + timedelta(days=1))
    assert len(provider.calls) == 3


def test_a_request_already_running_is_not_waited_for():
    provider = Provider({"1": ("1H", 5)})
    statuses = live_status.LiveStatuses(provider)
    with statuses._lock:  # another board request is asking the provider right now
        assert statuses.lookup([fixture(1, 5)], now=NOW) == {"1": None}
    assert provider.calls == []


def test_the_switch_is_off_unless_set_to_on(monkeypatch):
    monkeypatch.delenv("SPIX_LIVE_STATUS", raising=False)
    assert not live_status.enabled()
    monkeypatch.setenv("SPIX_LIVE_STATUS", "On ")
    assert live_status.enabled()


def test_the_provider_is_asked_once_with_the_ids_joined_and_a_failure_is_not_retried(monkeypatch):
    from data_platform.sync import apifootball

    sent, replies = [], [(200, {"errors": [], "results": 1, "response": [row(7, "2H", 61)]}), (500, {})]

    class Response:
        def __init__(self, status, body):
            self.status_code, self._body = status, body

        def json(self):
            return self._body

        def close(self):
            pass

    class Session:
        def get(self, url, *, params, timeout):
            sent.append((url, params, timeout))
            return Response(*replies.pop(0))

        def close(self):
            pass

    class Client:
        def __init__(self, *, request_pause_s):
            self.session = Session()

    monkeypatch.setattr(apifootball, "ApiFootballClient", Client)
    assert live_status.fetch_statuses(["7", "8"]) == [row(7, "2H", 61)]
    assert sent == [(f"{apifootball.API_BASE}/fixtures", {"ids": "7-8"}, live_status.TIMEOUT)]
    statuses = live_status.LiveStatuses()
    assert statuses.lookup([fixture(9, 10)], now=NOW) == {"9": None}  # HTTP 500: one attempt, no status
    assert len(sent) == 2 and not replies
