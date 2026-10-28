"""Operations messages to the owner's private chat (28 Oct): problems with a sound, the day's rhythm quietly."""
from datetime import datetime, timedelta, timezone

import pytest

from data_platform.services import ops_alerts

TZ = timezone(timedelta(hours=2))  # Bucharest in October


@pytest.fixture
def sent(tmp_path, monkeypatch):
    """Messages sent, as (text, silent); state kept in a fresh folder."""
    monkeypatch.setenv("SPIX_OPS_ALERT_DIR", str(tmp_path))
    messages = []
    monkeypatch.setattr(ops_alerts, "notify", lambda text, silent=False: messages.append((text, silent)) or True)
    return messages


def at(hour, minute=0, day=28):
    return datetime(2026, 10, day, hour, minute, tzinfo=TZ)


def test_a_message_that_cannot_be_sent_never_fails_the_job(monkeypatch):
    def broken(message, *, silent=False):
        raise RuntimeError("network down")
    monkeypatch.setattr("data_platform.services.refresh_alerts._send", broken)
    assert ops_alerts.notify("Spix · test") is False


def test_a_hook_that_breaks_never_fails_the_job(sent, monkeypatch):
    monkeypatch.setattr(ops_alerts, "streak", lambda *a, **k: 1 / 0)
    ops_alerts.measurement_cycle({"status": "partial", "errors": 1})  # no exception


# The Match Read worker -------------------------------------------------------------------------------------------

def cycle(errors=(), skipped=None, releases=()):
    return {"dry_run": False, "errors": list(errors), "skipped_reason": skipped, "releases": list(releases)}


def test_the_worker_alerts_once_after_two_failed_runs_and_quietly_on_recovery(sent):
    ops_alerts.match_read_cycle(cycle(errors=["SerieA 2026-10-31 pre_match: fixture 1 was not returned"]))
    assert sent == []
    ops_alerts.match_read_cycle(cycle(errors=["a", "b", "c"]))
    ops_alerts.match_read_cycle(cycle(errors=["a"]))
    assert len(sent) == 1 and not sent[0][1]
    assert "2 runs in a row" in sent[0][0] and "3 errors" in sent[0][0]
    ops_alerts.match_read_cycle(cycle())
    assert len(sent) == 2 and sent[1][1] and "3 failed runs" in sent[1][0]
    ops_alerts.match_read_cycle(cycle(errors=["a"]))
    assert len(sent) == 2  # a single failure after recovery starts a new count


def test_runs_skipped_while_the_gate_is_shut_neither_fail_nor_recover(sent):
    ops_alerts.match_read_cycle(cycle(errors=["a"]))
    ops_alerts.match_read_cycle(cycle(skipped="A full season refresh did not complete"))
    ops_alerts.match_read_cycle(cycle(errors=["a"]))
    assert len(sent) == 1 and "2 runs in a row" in sent[0][0]


def test_error_text_never_reaches_the_chat(sent):
    secret = "dispatcher failed (HTTPError: https://api.oddspapi.io/v4/fixtures?apiKey=s3cret)"
    for _ in range(2):
        ops_alerts.match_read_cycle(cycle(errors=[secret]))
    assert "s3cret" not in sent[0][0] and "oddspapi" not in sent[0][0]


def release(league, fixture, *, stage="pre_match", kickoff="2026-10-31T13:00:00+00:00", picks=2, created=True,
            label="Home vs Away", date="2026-10-31"):
    return {"league": league, "date": date, "stage": stage, "kickoff": kickoff, "fixture": label,
            "released": [{"fixture_id": fixture, "match_read_id": 1, "delivery_created": created,
                          "recommendations_created": picks}]}


def test_only_new_cards_count_as_published():
    report = cycle(releases=[release("EPL", "1"), release("EPL", "2", created=False)])
    assert [i["fixture_id"] for i in ops_alerts.release_items(report)] == ["1"]


def test_published_reads_are_grouped_by_day_and_league():
    items = ops_alerts.release_items(cycle(releases=[
        release("SerieA", "1", label="Frosinone vs Torino", kickoff="2026-10-31T13:00:00+00:00", picks=2),
        release("SerieA", "2", label="Genoa vs Lazio", kickoff="2026-10-31T16:00:00+00:00", picks=3),
        release("SerieA", "1", label="Frosinone vs Torino", stage="confirmed_lineups", picks=1),
        release("EPL", "3", label="Arsenal vs Leeds", kickoff="2026-11-01T14:30:00+00:00", picks=4,
                date="2026-11-01"),
    ]))
    text = ops_alerts.releases_message(items, tz=TZ)
    # A fixture read again (confirmed lineups) counts once, with its latest read's picks.
    assert text.splitlines()[0] == "Spix · Match Reads published: 3 matches, 8 picks"
    assert "Sat 31 Oct" in text and "Sun 1 Nov" in text
    assert "SerieA: Frosinone vs Torino, Genoa vs Lazio · 4 picks · first kickoff 15:00 · 1 with confirmed lineups" in text
    assert "EPL: Arsenal vs Leeds · 4 picks · first kickoff 16:30" in text


def test_many_matches_in_a_league_are_counted_not_listed():
    items = ops_alerts.release_items(cycle(releases=[release("LaLiga", str(n), label=f"Team {n} vs X")
                                                     for n in range(5)]))
    assert "LaLiga: 5 matches · 10 picks" in ops_alerts.releases_message(items, tz=TZ)


def test_published_reads_arrive_quietly_at_most_once_an_hour(sent):
    first = cycle(releases=[release("EPL", "1")])
    ops_alerts.releases_due(ops_alerts.release_items(first), now=at(14), tz=TZ)
    assert len(sent) == 1 and sent[0][1] is True
    ops_alerts.releases_due(ops_alerts.release_items(cycle(releases=[release("EPL", "2")])), now=at(14, 10), tz=TZ)
    ops_alerts.releases_due([], now=at(14, 50), tz=TZ)
    assert len(sent) == 1
    ops_alerts.releases_due([], now=at(15, 0), tz=TZ)
    assert len(sent) == 2 and "1 match," in sent[1][0]
    ops_alerts.releases_due([], now=at(16, 30), tz=TZ)
    assert len(sent) == 2  # nothing pending, nothing sent


def test_published_reads_wait_when_the_chat_is_unreachable(sent, monkeypatch):
    monkeypatch.setattr(ops_alerts, "notify", lambda text, silent=False: False)
    ops_alerts.releases_due(ops_alerts.release_items(cycle(releases=[release("EPL", "1")])), now=at(14), tz=TZ)
    monkeypatch.setattr(ops_alerts, "notify", lambda text, silent=False: sent.append((text, silent)) or True)
    ops_alerts.releases_due(ops_alerts.release_items(cycle(releases=[release("EPL", "2")])), now=at(14, 10), tz=TZ)
    assert len(sent) == 1 and "2 matches" in sent[0][0]


# The daily refresh -----------------------------------------------------------------------------------------------

def test_the_refresh_reports_its_start_and_end_quietly(sent):
    ops_alerts.refresh_started(attempt=2)
    ops_alerts.refresh_finished(ok=True, seconds=3780, attempt=2, now=at(9, 18))
    assert sent == [("Spix · daily refresh started (try 2 of 6 today).", True),
                    ("Spix · daily refresh done in 1 h 03 min. Match Reads and measurement resume.", True)]


def test_a_failed_refresh_makes_a_sound_and_says_when_it_tries_again(sent):
    ops_alerts.refresh_finished(ok=False, seconds=2520, attempt=1, now=at(7, 57))
    text, silent = sent[0]
    assert not silent and "failed after 42 min" in text and "tries again at 08:15 (try 2 of 6)" in text


def test_the_last_failed_try_says_it_has_stopped_once_a_day(sent):
    ops_alerts.refresh_finished(ok=False, seconds=600, attempt=6, now=at(12, 25))
    ops_alerts.refresh_skipped("attempt limit reached today (6 of 6)", gate_shut=True, now=at(13, 15))
    assert len(sent) == 1 and not sent[0][1] and "stopped trying" in sent[0][0]
    ops_alerts.refresh_skipped("attempt limit reached today (6 of 6)", gate_shut=True, now=at(13, 15, day=29))
    assert len(sent) == 2


def test_a_skipped_refresh_is_quiet_unless_it_gave_up_with_the_gate_shut(sent):
    ops_alerts.refresh_skipped("today's refresh already ran", gate_shut=False, now=at(9, 15))
    ops_alerts.refresh_skipped("attempt limit reached today (6 of 6)", gate_shut=False, now=at(13, 15))
    assert sent == []


# OddsPapi, measurement, the true close -----------------------------------------------------------------------------

def test_oddspapi_warns_once_per_level_and_again_after_the_allowance_renews(sent):
    ops_alerts.oddspapi_left(149, reserve=40)
    ops_alerts.oddspapi_left(50, reserve=40)
    ops_alerts.oddspapi_left(48, reserve=40)
    assert len(sent) == 1 and "50 requests left" in sent[0][0] and not sent[0][1]
    ops_alerts.oddspapi_left(40, reserve=40)
    assert len(sent) == 2 and "stopped buying" in sent[1][0]
    ops_alerts.oddspapi_left(250, reserve=40)
    ops_alerts.oddspapi_left(45, reserve=40)
    assert len(sent) == 3


def test_measurement_alerts_after_ten_failed_runs_and_ignores_deferred_ones(sent):
    for _ in range(9):
        ops_alerts.measurement_cycle({"status": "partial", "errors": 2})
    ops_alerts.measurement_cycle({"status": "refresh_deferred", "errors": 0})
    ops_alerts.measurement_cycle({"status": "warning", "errors": 0})
    assert sent == []
    for _ in range(10):
        ops_alerts.measurement_cycle({"status": "partial", "errors": 1})
    assert len(sent) == 1 and "10 runs in a row" in sent[0][0]


@pytest.mark.parametrize("counts, problem, failed", [
    ({"fixtures": 2, "errors": 45, "final": 0, "unavailable": 0}, None, False),  # one match retrying: normal
    ({"fixtures": 10, "errors": 45, "final": 162, "unavailable": 14}, None, False),
    ({"fixtures": 4, "errors": 60, "final": 0, "unavailable": 0}, None, True),
    (None, "unreachable", True),
    (None, "crashed", True),
])
def test_the_true_close_fails_when_it_records_nothing_or_cannot_run(sent, counts, problem, failed):
    for _ in range(3):
        ops_alerts.true_close_run(counts, problem=problem)
    assert len(sent) == (1 if failed else 0)
    if failed:
        assert "3 runs in a row" in sent[0][0]


# Price confirmation (stage 4) --------------------------------------------------------------------------------------

def record(reason=None, status="not_confirmed"):
    return {"status": "confirmed"} if reason is None and status == "confirmed" else {"status": status, "reason": reason}


def test_price_confirmation_alerts_once_a_day_when_most_picks_cannot_be_checked(sent):
    ops_alerts.price_confirmations([record("price_gone")] * 6, now=at(11))  # the market moved: not a fault
    ops_alerts.price_confirmations([record(status="confirmed")] * 2 + [record("fixture_not_matched")] * 4, now=at(12))
    assert sent == []
    ops_alerts.price_confirmations([record("fixture_list_error")] * 4, now=at(13))
    assert len(sent) == 0  # 8 faults of 16: not most
    ops_alerts.price_confirmations([record("history_error")], now=at(14))
    assert len(sent) == 1 and not sent[0][1]
    assert "9 of today's 17 picks" in sent[0][0]
    assert "the match is not in OddsPapi's fixture list (4)" in sent[0][0]
    ops_alerts.price_confirmations([record("history_error")] * 5, now=at(15))
    assert len(sent) == 1
    ops_alerts.price_confirmations([record("history_error")] * 5, now=at(9, day=29))
    assert len(sent) == 2


# The Monday scorecard ----------------------------------------------------------------------------------------------

def test_the_scorecard_headline():
    slice_ = {"published": 2424, "settled": 2132, "win_rate": 0.6618, "stated": 0.7493, "market": 0.6430,
              "true_close_value": -0.0234, "true_close_count": 1506,
              "intervals": {"true_close_value": [-0.0306, -0.0166]}}
    card = {"as_of": "2026-11-02", "method": "proportional", "cohorts": {"selections": {
        "overall": slice_,
        "by_engine": {"defaults": {**slice_, "true_close_value": -0.022, "true_close_count": 1000, "intervals": None},
                      "SPIX_COUNT_DISPERSION=fixed + SPIX_DEVIG_METHOD=power": {
                          "true_close_value": None, "true_close_count": 0}}}}}
    text = ops_alerts.scorecard_message(card)
    assert text.splitlines()[0] == "Spix · scorecard 2026-11-02"
    assert "Value at Pinnacle's close: -2.3% (-3.1% to -1.7%) on 1,506 picks" in text
    assert "Spix said 74.9%, won 66.2%; the market said 64.3%." in text
    assert "defaults: -2.2% on 1,000" in text
    assert "COUNT_DISPERSION=fixed + DEVIG_METHOD=power: no picks at the close yet" in text
    assert "Research/scorecards/2026-11-02/scorecard.md" in text


# Wiring: the scheduled commands report to the chat ----------------------------------------------------------------

def test_the_worker_command_reports_each_run_but_not_a_dry_run(monkeypatch):
    from contextlib import nullcontext
    from types import SimpleNamespace
    from data_platform import cli
    from data_platform.services import match_read_cycle, worker_runtime
    seen = []

    class Report:
        errors = []

        def as_dict(self):
            return {"dry_run": False, "errors": [], "releases": []}

    class Service:
        def __init__(self, settings):
            pass

        def run_once(self, **_):
            return Report()
    monkeypatch.setattr(match_read_cycle, "MatchReadCycleService", Service)
    monkeypatch.setattr(worker_runtime, "prevent_idle_sleep", nullcontext)
    monkeypatch.setattr(ops_alerts, "match_read_cycle", lambda report, **_: seen.append(report))
    args = SimpleNamespace(league=None, mode="website", now=None, dry_run=False)
    assert cli.cmd_match_read_cycle(args) == 0 and seen == [Report().as_dict()]
    cli.cmd_match_read_cycle(SimpleNamespace(**{**vars(args), "dry_run": True}))
    assert len(seen) == 1


def test_the_measurement_command_reports_each_run_but_not_a_dry_run(monkeypatch):
    from types import SimpleNamespace
    from data_platform import cli
    seen = []
    monkeypatch.setattr("data_platform.services.measurement_cycle.run_measurement_cycle",
                        lambda **_: {"status": "partial", "errors": 1})
    monkeypatch.setattr(ops_alerts, "measurement_cycle", seen.append)
    args = SimpleNamespace(command="measurement-cycle", settlement_only=False, closing_only=False, dry_run=False)
    assert cli.cmd_measurement(args) == 1 and seen == [{"status": "partial", "errors": 1}]
    cli.cmd_measurement(SimpleNamespace(**{**vars(args), "dry_run": True}))
    assert len(seen) == 1


def refresh_args(result):
    from types import SimpleNamespace
    return SimpleNamespace(command="refresh-season", dry_run=False, verbose=False, if_due=True, func=lambda _: result)


@pytest.mark.parametrize("result, ok", [(0, True), (1, False)])
def test_the_refresh_reports_its_start_and_end(settings, engine, monkeypatch, result, ok):
    from types import SimpleNamespace
    from data_platform import cli
    from data_platform.services import refresh_retry
    calls = []
    monkeypatch.setattr(cli, "build_parser", lambda: SimpleNamespace(parse_args=lambda _: refresh_args(result)))
    monkeypatch.setattr(refresh_retry, "check_due", lambda: (True, "today's refresh has not run"))
    monkeypatch.setattr(refresh_retry, "attempts_today", lambda: 2)
    monkeypatch.setattr("data_platform.services.measurement_cycle.run_measurement_cycle", lambda **_: {})
    monkeypatch.setattr(ops_alerts, "refresh_started", lambda attempt=None: calls.append(("started", attempt)))
    monkeypatch.setattr(ops_alerts, "refresh_finished",
                        lambda *, ok, seconds, attempt=None: calls.append(("finished", ok, attempt, seconds >= 0)))
    assert cli.main([]) == result
    assert calls == [("started", 2), ("finished", ok, 2, True)]


def test_a_skipped_refresh_is_reported_with_its_reason(settings, engine, monkeypatch):
    from types import SimpleNamespace
    from data_platform import cli
    from data_platform.services import refresh_retry
    skipped = []
    monkeypatch.setattr(cli, "build_parser", lambda: SimpleNamespace(parse_args=lambda _: refresh_args(0)))
    monkeypatch.setattr(refresh_retry, "check_due", lambda: (False, "attempt limit reached today (6 of 6)"))
    monkeypatch.setattr(ops_alerts, "refresh_skipped", skipped.append)
    assert cli.main([]) == 0 and skipped == ["attempt limit reached today (6 of 6)"]
