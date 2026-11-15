"""The daily refresh retries itself hourly until the day's run completes (28 Oct)."""
from datetime import datetime, timedelta, timezone
import json
from types import SimpleNamespace

from data_platform.services import refresh_retry as rr

TZ = timezone(timedelta(hours=2))


def at(hour, minute=15, day=28):
    return datetime(2026, 10, day, hour, minute, tzinfo=TZ)


def test_the_daily_refresh_is_due_from_seven_fifteen():
    assert rr.refresh_due(at(7), gate_shut=False, attempts=[])[0]
    assert not rr.refresh_due(at(7, 14), gate_shut=False, attempts=[])[0]


def test_nothing_runs_before_seven_fifteen_even_with_the_gate_shut():
    due, reason = rr.refresh_due(at(3), gate_shut=True, attempts=[at(23, day=27)])
    assert not due and "before" in reason


def test_a_completed_run_today_is_not_repeated():
    assert not rr.refresh_due(at(9), gate_shut=False, attempts=[at(7)])[0]


def test_a_failed_run_is_retried_every_hour_up_to_the_daily_limit():
    attempts = [at(7)]
    for hour in range(8, 8 + rr.MAX_ATTEMPTS - 1):
        assert rr.refresh_due(at(hour), gate_shut=True, attempts=attempts)[0]
        attempts.append(at(hour))
    due, reason = rr.refresh_due(at(8 + rr.MAX_ATTEMPTS), gate_shut=True, attempts=attempts)
    assert len(attempts) == rr.MAX_ATTEMPTS and not due and "limit" in reason


def test_yesterdays_attempts_do_not_count_today():
    yesterday = [at(hour, day=27) for hour in range(7, 7 + rr.MAX_ATTEMPTS)]
    assert rr.refresh_due(at(7), gate_shut=True, attempts=yesterday)[0]
    assert rr.refresh_due(at(7), gate_shut=False, attempts=yesterday)[0]


def fake_run(battery, lid_closed):
    outputs = {"pmset": "Now drawing from 'Battery Power'\n" if battery else "Now drawing from 'AC Power'\n",
               "ioreg": f'    | "AppleClamshellState" = {"Yes" if lid_closed else "No"}\n'}
    return lambda command, **_: SimpleNamespace(returncode=0, stdout=outputs[command[0].rsplit("/", 1)[-1]])


def test_only_a_shut_mac_on_battery_cannot_stay_awake():
    assert not rr.can_stay_awake(run=fake_run(battery=True, lid_closed=True))
    assert rr.can_stay_awake(run=fake_run(battery=True, lid_closed=False))
    assert rr.can_stay_awake(run=fake_run(battery=False, lid_closed=True))
    def broken(*_, **__):
        raise OSError("no pmset")
    assert rr.can_stay_awake(run=broken)


def test_a_due_run_is_recorded_and_a_skipped_one_is_not(tmp_path):
    path = tmp_path / "attempts.json"
    awake = dict(stay_awake=lambda: True, reachable=lambda: True)
    assert rr.check_due(at(7), gate_shut=False, attempts_path=path, **awake)[0]
    assert [datetime.fromisoformat(a) for a in json.loads(path.read_text())["attempts"]] == [at(7)]
    assert not rr.check_due(at(8), gate_shut=False, attempts_path=path, **awake)[0]
    asleep = rr.check_due(at(9), gate_shut=True, attempts_path=path, stay_awake=lambda: False, reachable=lambda: True)
    offline = rr.check_due(at(9), gate_shut=True, attempts_path=path, stay_awake=lambda: True, reachable=lambda: False)
    assert not asleep[0] and not offline[0]
    assert len(json.loads(path.read_text())["attempts"]) == 1


def test_if_due_skips_without_taking_the_gate_or_refreshing(settings, engine, monkeypatch):
    from data_platform import cli
    calls = []
    args = SimpleNamespace(command="refresh-season", dry_run=False, verbose=False, if_due=True,
                           func=lambda _: calls.append(1) or 0)
    monkeypatch.setattr(cli, "build_parser", lambda: SimpleNamespace(parse_args=lambda _: args))
    monkeypatch.setattr(rr, "check_due", lambda: (False, "today's refresh already ran"))
    assert cli.main([]) == 0 and calls == []
    monkeypatch.setattr(rr, "check_due", lambda: (True, "today's refresh has not run"))
    monkeypatch.setattr("data_platform.services.measurement_cycle.run_measurement_cycle", lambda **_: {})
    assert cli.main([]) == 0 and calls == [1]


def test_the_refresh_command_takes_if_due():
    from data_platform import cli
    args = cli.build_parser().parse_args(["refresh-season", "--include-europe", "--if-due"])
    assert args.if_due is True


def test_the_attempts_today_number_the_running_try(tmp_path):
    path = tmp_path / "attempts.json"
    path.write_text(json.dumps({"attempts": [at(23, day=27).isoformat(), at(7).isoformat(), at(8).isoformat()]}))
    assert rr.attempts_today(at(8, 20), attempts_path=path) == 2
    assert rr.attempts_today(at(6), attempts_path=path) == 0  # before today's refresh time


def run_if_due(tmp_path, monkeypatch, *, blocked=False, result=0):
    """One scheduled run through the real command, returning the attempts left on record."""
    from contextlib import contextmanager
    from data_platform import cli
    from data_platform.services import refresh_coordination as rc
    path = tmp_path / "attempts.json"
    check_due = rr.check_due
    monkeypatch.setattr(rr, "_attempts_path", lambda: path)
    monkeypatch.setattr(rr, "check_due", lambda: check_due(at(9), gate_shut=False, attempts_path=path,
                                                           stay_awake=lambda: True, reachable=lambda: True))
    if blocked:
        @contextmanager
        def held(**_):
            raise rc.RefreshBusy("Data refresh or fixture generation is active; retry on the next tick.")
            yield
        monkeypatch.setattr(rc, "data_access", held)
    args = SimpleNamespace(command="refresh-season", dry_run=False, verbose=False, if_due=True,
                           func=lambda _: result)
    monkeypatch.setattr(cli, "build_parser", lambda: SimpleNamespace(parse_args=lambda _: args))
    cli.main([])
    return rr._load(path)


def test_a_run_that_never_got_the_gate_is_not_counted(tmp_path, settings, engine, monkeypatch):
    # 1 Nov: the 07:16 run timed out waiting for the gate, and "today's refresh already ran" held all day.
    assert run_if_due(tmp_path, monkeypatch, blocked=True) == []
    assert rr.refresh_due(at(10), gate_shut=False, attempts=[])[0]


def test_a_run_that_started_and_failed_still_counts(tmp_path, settings, engine, monkeypatch):
    assert run_if_due(tmp_path, monkeypatch, result=1) == [at(9)]


def test_withdrawing_removes_only_the_latest_attempt(tmp_path):
    path = tmp_path / "attempts.json"
    path.write_text(json.dumps({"attempts": [at(7).isoformat(), at(8).isoformat()]}))
    rr.withdraw_attempt(attempts_path=path)
    assert rr._load(path) == [at(7)]


# Held for match time (D109, 15 Nov) -----------------------------------------------------------------------------
# A late refresh (the Mac slept through 07:15) waits while a match it would run into is close, so the worker keeps
# checking cards in each match's last two hours. 7 Nov: the 07:18 refresh ran until about 15:20 over the midday
# kickoffs; 8 Nov: the 16:15 refresh ran over the 16:15 and 16:30 UTC kickoffs (D108).

def ko(hour, minute=0, league="EPL", day=28):
    return league, datetime(2026, 10, day, hour, minute, tzinfo=TZ)


def never_read(*_):
    raise AssertionError("kickoffs read although no hold can apply")


def hold(now, kickoffs, *, gate_shut=False, runs=(60,) * 5):
    return rr.refresh_hold(now, gate_shut=gate_shut, runs=list(runs), window_minutes=120,
                           kickoffs=kickoffs if callable(kickoffs) else lambda *_: kickoffs)


def first_start(kickoffs, *, runs=(60,) * 5):
    """The first hourly tick from 09:15 at which a late refresh starts."""
    return next(hour for hour in range(9, 24) if hold(at(hour), kickoffs, runs=runs) is None)


def test_a_late_refresh_with_no_match_close_runs_and_counts(tmp_path):
    path = tmp_path / "attempts.json"
    due, reason = rr.check_due(at(10), gate_shut=False, attempts_path=path, stay_awake=lambda: True,
                               reachable=lambda: True, kickoffs=lambda *_: [ko(16)], window_minutes=120)
    assert due and reason == "today's refresh has not run"
    assert rr._load(path) == [at(10)]


def test_a_late_refresh_waits_when_a_match_kicks_off_soon_and_does_not_count(tmp_path):
    path = tmp_path / "attempts.json"
    due, reason = rr.check_due(at(10), gate_shut=False, attempts_path=path, stay_awake=lambda: True,
                               reachable=lambda: True, kickoffs=lambda *_: [ko(13, 30, "Ligue1")], window_minutes=120)
    assert not due and rr.HELD in reason and "Ligue1 kicks off at 13:30" in reason
    assert rr._load(path) == []


def test_the_morning_run_and_a_retry_after_a_failure_are_never_held():
    assert hold(at(7), never_read) is None
    assert hold(at(8, 14), never_read) is None  # still the normal run's hour
    assert hold(at(10), never_read, gate_shut=True) is None  # no cards at all until a refresh completes
    assert hold(at(8), [ko(9)]) is not None


def test_a_match_in_its_last_two_hours_holds_only_until_the_run_would_end_before_them():
    # A 60-minute run started at 10:15 ends 11:15; a 13:15 kickoff's last two hours start then.
    assert hold(at(10), [ko(13, 15)]) is not None
    assert hold(at(10), [ko(13, 16)]) is None
    assert hold(at(10), [ko(10, 0)]) is None  # already kicked off


def test_a_busy_day_takes_the_first_real_gap():
    assert first_start([ko(12), ko(14, 30), ko(21)]) == 15


def test_a_day_held_until_the_last_kickoff_runs_after_it():
    assert first_start([ko(hour) for hour in range(12, 23, 2)]) == 22


def test_the_last_start_of_the_day_runs_whatever_the_kickoffs():
    assert hold(at(22), [ko(23, 30, "PrimeiraLiga")]) is not None
    assert hold(at(23), [ko(23, 30, "PrimeiraLiga")]) is None


def test_kickoffs_that_cannot_be_read_never_hold_the_refresh(tmp_path):
    from data_platform.services.refresh_coordination import RefreshBusy

    def busy(*_):
        raise RefreshBusy("Data refresh or fixture generation is active; retry on the next tick.")
    due, _ = rr.check_due(at(10), gate_shut=False, attempts_path=tmp_path / "attempts.json",
                          stay_awake=lambda: True, reachable=lambda: True, kickoffs=busy, window_minutes=120)
    assert due


def test_the_likely_run_time_is_the_upper_quartile_of_recent_runs_between_one_and_three_hours():
    assert rr.likely_minutes([50] * 4) == rr.RUN_DEFAULT == 120  # too few runs on record
    assert rr.likely_minutes([40] * 14) == 60
    assert rr.likely_minutes([400] * 14) == 180
    # The completed runs of 2-15 Nov: about 45 minutes awake, longer after a weekend, hours on a sleeping Mac.
    recent = [52, 44, 49, 59, 49, 493, 150, 50, 46, 47, 46, 56, 97, 317]
    assert 100 <= rr.likely_minutes(recent) <= 120
    assert rr.likely_minutes([600] * 10 + recent) == rr.likely_minutes(recent)  # only the last 14 count


def test_a_longer_likely_run_looks_further_ahead():
    # From 10:15, a 60-minute run looks to 13:15 and a 150-minute one to 14:45.
    assert hold(at(10), [ko(14, 30)], runs=[60] * 5) is None
    assert hold(at(10), [ko(14, 30)], runs=[150] * 5) is not None


def test_completed_runs_are_kept_beside_the_attempts_and_failed_ones_are_not(tmp_path):
    path = tmp_path / "attempts.json"
    awake = dict(stay_awake=lambda: True, reachable=lambda: True)
    rr.check_due(at(7), gate_shut=False, attempts_path=path, **awake)
    rr.record_run(47.5, ok=True, now=at(8), attempts_path=path)
    rr.record_run(3, ok=False, now=at(8), attempts_path=path)
    rr.check_due(at(9), gate_shut=True, attempts_path=path, **awake)
    rr.withdraw_attempt(attempts_path=path)
    assert rr._load(path) == [at(7)]
    assert rr._runs(path) == [47.5]


def test_the_kickoffs_are_the_workers_schedule_read_inside_the_gate(settings, monkeypatch):
    from contextlib import contextmanager
    from types import SimpleNamespace as Row
    from data_platform.services import match_read_cycle, refresh_coordination
    gate, asked = [], []

    @contextmanager
    def data_access(**kwargs):
        gate.append(kwargs)
        yield

    def loader(*, now, until, leagues):
        asked.append((now, until, tuple(leagues)))
        return [Row(league="UCL", kickoff_utc=datetime(2026, 10, 28, 19, tzinfo=timezone.utc))]
    monkeypatch.setattr(refresh_coordination, "data_access", data_access)
    monkeypatch.setattr(match_read_cycle, "_load_upcoming_platform_fixtures", loader)
    monkeypatch.delenv("MATCH_READ_WORKER_LEAGUES", raising=False)
    found = rr.upcoming_kickoffs(at(10), at(14))
    assert found == [("UCL", datetime(2026, 10, 28, 19, tzinfo=timezone.utc))]
    assert gate == [{}] and asked[0][:2] == (at(10), at(14))
    assert {"EPL", "Championship", "UCL", "UECL"} <= set(asked[0][2])


def test_a_completed_refresh_records_how_long_it_took(tmp_path, settings, engine, monkeypatch):
    monkeypatch.setattr(rr, "upcoming_kickoffs", lambda *_: [])
    run_if_due(tmp_path, monkeypatch)
    assert len(rr._runs(tmp_path / "attempts.json")) == 1


def test_a_failed_refresh_records_no_run_time(tmp_path, settings, engine, monkeypatch):
    monkeypatch.setattr(rr, "upcoming_kickoffs", lambda *_: [])
    run_if_due(tmp_path, monkeypatch, result=1)
    assert rr._runs(tmp_path / "attempts.json") == []
