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
            + season(2023, 48, rng, goals, 300, {"EPL": england, "LaLiga": spain})
            + season(2026, 16, rng, goals, 400, {"EPL": england, "LaLiga": spain}))


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


def test_a_fresh_run_must_be_named(stubbed, monkeypatch):
    out = stubbed / "out"
    assert bench.main(["run", "--block", "fresh", "--out", str(out)]) == 2
    assert not out.exists() and not bench.LEDGER.exists()
    monkeypatch.setattr(bench, "_bench_code_changes", lambda: " M Scripts/data_platform/game_read/pool.py")
    assert bench.main(["run", "--block", "fresh", "--confirm-locked", "--out", str(out)]) == 2
    assert not out.exists() and not bench.LEDGER.exists()
    monkeypatch.setattr(bench, "_bench_code_changes", lambda: "")
    assert bench.main(["run", "--block", "fresh", "--confirm-locked", "--out", str(out)]) == 0
    entry = json.loads(bench.LEDGER.read_text().splitlines()[0])
    assert entry["block"] == "fresh"
    assert json.loads((out / "summary.json").read_text())["run"]["matches_scored"] == 16  # the 2026 matches only


def test_a_locked_run_never_predicts_the_fresh_season(stubbed, monkeypatch):
    seen = []

    class Watching(MarketCopy):
        def predict(self, matches):
            seen.extend(m.season for m in matches)
            return super().predict(matches)

        def learn(self, matches):
            seen.extend(m.season for m in matches)

    monkeypatch.setitem(bench.MODELS, "control", Watching)
    assert bench.main(["run", "--block", "locked", "--confirm-locked", "--out", str(stubbed / "out")]) == 0
    assert seen and max(seen) == 2023
    seen.clear()
    assert bench.main(["run", "--block", "tuning", "--out", str(stubbed / "tuning")]) == 0
    assert seen and max(seen) == 2019


def test_opportunities_never_look_at_fresh_seasons(history):
    with pytest.raises(ValueError):
        opportunities.scan([m for m in history if m.season != 2023])


def test_opportunities_never_look_at_locked_seasons(history):
    with pytest.raises(ValueError):
        opportunities.scan([m for m in history if m.season < 2026])
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


def test_the_average_runs_its_members_and_fits_its_weights_on_tuning_seasons_only(stubbed, monkeypatch):
    seen, fit_all = [], bench.average.fit_all
    monkeypatch.setattr(bench.average, "fit_all", lambda rows: seen.append(rows) or fit_all(rows))
    out = stubbed / "out"
    assert bench.main(["run", "--models", "average", "--block", "locked", "--confirm-locked", "--out", str(out)]) == 0
    assert len(seen) == 1 and set(seen[0]) == {"elo", "pi", "dc", "drift"}
    assert {m.season for rows in seen[0].values() for m, _ in rows} == {2019}  # never the locked 2023 matches
    summary = json.loads((out / "summary.json").read_text())
    assert summary["run"]["models"] == ["market", "control", "elo", "pi", "dc", "drift", "average"]
    weights = summary["models"]["average"]["weights"]
    assert set(weights["goals"]) == {"dc", "drift"} and set(weights["result"]) == {"elo", "pi", "dc", "drift"}
    assert all(sum(part.values()) == pytest.approx(1) for part in weights.values())
    report = (out / "report.md").read_text()
    assert "## The average's weights" in report and "in-sample" in report


def test_a_variant_runs_its_base_and_is_gated_against_it(stubbed, monkeypatch, history):
    from Scripts.data_platform.game_read.stats import MatchStats
    rng = np.random.default_rng(9)
    table = {m.id: MatchStats(home_box=int(rng.poisson(9)), away_box=int(rng.poisson(7)), home_outside=5,
                              away_outside=4, home_target=3, away_target=2) for m in history}
    monkeypatch.setattr(bench, "_stats", lambda: table)
    out = stubbed / "out"
    assert bench.main(["run", "--block", "tuning", "--models", "dc+box", "--out", str(out)]) == 0
    summary = json.loads((out / "summary.json").read_text())
    assert summary["run"]["models"] == ["market", "control", "dc", "dc+box"]
    found = summary["models"]["dc+box"]["ablation"]
    assert found["base"] == "dc" and set(found["tuning"]) == {"result", "over25"}
    assert found["tuning"]["result"]["n"] == 96 and isinstance(found["passed"], bool)
    assert "ablation" not in summary["models"]["dc"]
    assert "## Ablation" in (out / "report.md").read_text()
    assert bench.main(["run", "--block", "tuning", "--models", "dc+nothing", "--out", str(stubbed / "x")]) == 2


def test_variables_combine_in_a_name(monkeypatch):
    monkeypatch.setattr(bench, "_stats", lambda: {})
    model = bench._factory("dc+box+rest")()
    assert model.name == "dc+box+rest" and [v.name for v in model.variables] == ["box", "rest"]
    assert bench._factory("drift+rest")().name == "drift+rest"
    assert bench._factory("dc+nothing") is None and bench._factory("elo+rest") is None
    assert bench._factory("market") is bench.MODELS["market"]


def test_a_variable_carried_by_the_owner_is_said_in_the_report(stubbed, monkeypatch, history):
    from Scripts.data_platform.game_read.stats import MatchStats
    monkeypatch.setattr(bench, "_stats", lambda: {m.id: MatchStats(home_box=8, away_box=6, home_outside=5,
                                                                   away_outside=4) for m in history})
    out = stubbed / "out"
    assert bench.main(["run", "--block", "tuning", "--models", "dc+xg", "--out", str(out)]) == 0
    summary = json.loads((out / "summary.json").read_text())
    assert set(summary["carried"]) == {"xg"} and summary["run"]["models"] == ["market", "control", "dc", "dc+xg"]
    assert "owner's choice" in (out / "report.md").read_text()
    assert bench.main(["run", "--block", "tuning", "--models", "dc", "--out", str(stubbed / "plain")]) == 0
    assert json.loads((stubbed / "plain" / "summary.json").read_text())["carried"] == {}
