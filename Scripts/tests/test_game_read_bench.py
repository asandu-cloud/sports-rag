"""Game-read bench, task 7: the bench command, the report, Opportunities and the locked-run guard (D77)."""
from __future__ import annotations

from datetime import date, timedelta
import json
import random

import numpy as np
import pytest

from Scripts.data_platform import market_goals
from Scripts.data_platform.game_read import opportunities, pool
from Scripts.data_platform.game_read.forecast import Forecast
from Scripts.data_platform.game_read.history import Match, Prices
from Scripts.data_platform.game_read.models.market import MarketCopy
from Scripts.ops import game_read_bench as bench


def priced(fair, booksum):
    return tuple(round(p, 2) for p in pool.inverse_power(fair, booksum))


def prices(home, away, line, best):
    result = market_goals.result_probabilities(home, away)
    over = market_goals.over_probability(home, away, 2.5)
    cover = market_goals.handicap_probability(home, away, line)

    def pair(fair, booksum):
        average = priced(fair, booksum)
        return average, tuple(round(p * best, 2) for p in average)
    (result_avg, result_max), handicap, over25 = pair(result, 1.05), pair([cover, 1 - cover], 1.04), pair(
        [over, 1 - over], 1.04)
    return Prices(result_avg=result_avg, result_max=result_max, handicap_line=line, handicap_avg=handicap[0],
                  handicap_max=handicap[1], over25_avg=over25[0], over25_max=over25[1])


def season(year, n, rng, goals, first_id, teams):
    """``n`` matches, alternating EPL and LaLiga; each league's matches pair its own teams in turn."""
    out = []
    for i in range(n):
        home, away = rng.uniform(0.7, 2.3), rng.uniform(0.5, 1.7)
        line, best, drift = rng.choice([-1.5, -0.5, 0.5, -0.75]), rng.uniform(1.03, 1.10), rng.uniform(0.9, 1.1)
        league = ("EPL", "LaLiga")[i % 2]
        names, k = teams[league], i // 2
        out.append(Match(id=first_id + i, competition=league, country="X", season=year,
                         day=date(year, 8, 10) + timedelta(days=(i // 4) * 3), home=names[k % len(names)],
                         away=names[(k + 1) % len(names)], home_goals=int(goals.poisson(home)),
                         away_goals=int(goals.poisson(away)), pre=prices(home, away, line, best),
                         close=prices(home * drift, away / drift, line, best)))
    return out


@pytest.fixture(scope="module")
def history():
    rng, goals = random.Random(5), np.random.default_rng(5)
    england, spain = [f"X:E{i}" for i in range(7)], [f"X:L{i}" for i in range(6)]
    return (season(2018, 24, rng, goals, 1, {"EPL": england[:6], "LaLiga": spain})
            + season(2019, 96, rng, goals, 100, {"EPL": england[1:], "LaLiga": spain})  # E6 is new, E0 gone
            + season(2023, 48, rng, goals, 300, {"EPL": england, "LaLiga": spain}))


@pytest.fixture
def stubbed(monkeypatch, tmp_path, history):
    monkeypatch.setattr(bench, "_load", lambda leagues: [m for m in history if not leagues or m.competition in leagues])
    monkeypatch.setattr(bench, "_named", lambda: {})
    monkeypatch.setattr(bench, "_bench_code_changes", lambda: "")
    monkeypatch.setattr(bench, "_commit", lambda: "abc1234")
    monkeypatch.setattr(bench, "LEDGER", tmp_path / "ledger" / "locked-runs.jsonl")
    return tmp_path


def test_a_locked_run_must_be_named(stubbed):
    out = stubbed / "out"
    assert bench.main(["run", "--block", "locked", "--out", str(out)]) == 2
    assert not out.exists() and not bench.LEDGER.exists()


def test_a_locked_run_refuses_uncommitted_bench_code(stubbed, monkeypatch):
    monkeypatch.setattr(bench, "_bench_code_changes", lambda: " M Scripts/data_platform/game_read/pool.py")
    out = stubbed / "out"
    assert bench.main(["run", "--block", "locked", "--confirm-locked", "--out", str(out)]) == 2
    assert not out.exists() and not bench.LEDGER.exists()


def test_a_locked_run_is_written_to_the_ledger(stubbed):
    out = stubbed / "out"
    assert bench.main(["run", "--block", "locked", "--confirm-locked", "--out", str(out)]) == 0
    lines = bench.LEDGER.read_text().splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["commit"] == "abc1234" and entry["models"] == ["market", "control"] and entry["block"] == "locked"
    assert {"time", "headline"} <= set(entry) and set(entry["headline"]) == {"market", "control"}
    summary = json.loads((out / "summary.json").read_text())
    assert summary["run"]["matches_scored"] == 48  # the 2023 matches only
    assert bench.main(["run", "--block", "locked", "--confirm-locked", "--out", str(stubbed / "again")]) == 0
    assert len(bench.LEDGER.read_text().splitlines()) == 2  # every locked run is kept


class Broken(MarketCopy):
    """A zero line that adds 0.01 to every chance."""

    def predict(self, matches):
        found = []
        for match, forecast in zip(matches, super().predict(matches)):
            options = pool.build_pool(match)
            found.append(Forecast(result=None if forecast.result is None else tuple(p + 0.01 for p in forecast.result),
                                  lines={pool.option_key(o): o["chance"] + 0.01 for o in options}))
        return found


def test_a_broken_zero_line_stops_the_run(stubbed, monkeypatch):
    monkeypatch.setitem(bench.MODELS, "market", Broken)
    out = stubbed / "out"
    assert bench.main(["run", "--block", "tuning", "--out", str(out)]) == 2
    assert not out.exists()


def test_opportunities_never_look_at_locked_seasons(history):
    with pytest.raises(ValueError):
        opportunities.scan(history)
    found = opportunities.scan([m for m in history if m.season < 2023])
    assert found["drift"]["home win"]["all"]["n"] == 96  # tuning matches only; the warm-up season is not measured
    assert sum(band["n"] for band in found["calibration"].values()) == 96 * 3
    assert found["promoted"]["teams"] == 1 and found["promoted"]["n"] == 10  # E6's first 10 matches
    for lead in found["leads"]:
        assert lead["low"] > 0 or lead["high"] < 0
    returns = [path for path, group in opportunities._walk(found["calibration"]) if path[-1:] == ("return",)]
    assert returns and opportunities._lead(("calibration", "1–1.5", "return"), {"low": -0.2, "high": -0.1}) is False
    assert opportunities._lead(("calibration", "1–1.5", "return"), {"low": 0.01, "high": 0.2}) is True
    assert opportunities._lead(("drift", "draw", "all"), {"low": -0.2, "high": -0.1}) is True


def test_the_report_has_every_section(stubbed):
    out = stubbed / "out"
    assert bench.main(["run", "--block", "tuning", "--out", str(out)]) == 0
    report = (out / "report.md").read_text()
    for heading in ("# Game-read bench: tuning seasons", "## The zero line", "## Accuracy",
                    "## Adds to the market (blend)", "## Most likely picks", "## Value", "## The closing check",
                    "## Leagues", "## Verdicts", "## The control line", "## The market against the named books",
                    "## Opportunities"):
        assert heading in report, heading
    summary = json.loads((out / "summary.json").read_text())
    assert summary["zero_line"]["passed"] is True
    assert set(summary["models"]) == {"market", "control"}
    assert summary["run"]["matches_scored"] == 96 and summary["run"]["block"] == "tuning"
    assert set(summary["verdicts"]) == {"edge", "blend"}
    assert {"reference", "opportunities"} <= set(summary)
    assert not bench.LEDGER.exists()  # only locked runs go to the ledger
