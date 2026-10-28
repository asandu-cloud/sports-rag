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
