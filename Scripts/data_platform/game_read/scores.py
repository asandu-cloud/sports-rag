"""Game-read bench (D77): the five scores of a model, from the day-by-day runner's (match, forecast) rows.

- Accuracy: log loss against the market's on the result, a half-line main handicap and over/under 2.5.
- Adds to the market: the blend of the market and the model fitted on the tuning seasons (``blend.fit``), scored
  against the market-only recalibration.
- Most likely picks: today's lists against the engine's, wins minus the market's expected wins per 100 picks.
- Value: today's rule and the engine's, return per unit and closing line value.
- The closing check: do closing prices move towards the engine's Most likely picks more than towards today's?

Every interval resamples whole weeks (Monday to Sunday), 95%, seeded. Each score sums a few numbers per match;
a resample adds up its weeks' sums and the score is computed from the totals, so 2,000 resamples stay quick.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from functools import lru_cache
import math
from typing import Callable, Mapping, Optional, Sequence

import numpy as np

from Scripts.data_platform import blend, devig, market_goals, recommendations
from Scripts.data_platform.features.benchmarks.evaluation import calendar_block
from Scripts.data_platform.game_read import pool as pools
from Scripts.data_platform.game_read import rules
from Scripts.data_platform.game_read.forecast import Forecast, chance
from Scripts.data_platform.game_read.history import Match, market_handicap, market_over25, market_result

RESAMPLES, SEED = 2000, 20261115
MARKETS = ("result", "handicap", "over25")
EPS = 1e-12
MIN_VALUE_PICKS = 300
MAX_LEAGUE_SHORTFALL = 2.0  # wins per 100 picks
MIN_NAMED_BOOKS = 3
SCOPES = ("all", "real", "derived")


def _finite(value) -> Optional[float]:
    return float(value) if value is not None and math.isfinite(value) else None


def _ratio(top, bottom) -> float:
    return float(top) / float(bottom) if bottom else math.nan


def _week(match: Match) -> str:
    return calendar_block(datetime(match.day.year, match.day.month, match.day.day, tzinfo=timezone.utc))


def week_interval(rows: Sequence, statistic: Callable, *, resamples: int = RESAMPLES,
                  seed: int = SEED) -> tuple:
    """The 95% interval of ``statistic`` over resamples of whole weeks. ``rows`` are (match, values) pairs; a
    resample draws as many weeks as there are, with replacement, and ``statistic`` turns the summed values of its
    rows into the estimate. (None, None) when there is nothing to resample."""
    weeks = defaultdict(list)
    for match, values in rows:
        weeks[_week(match)].append(np.asarray(values, dtype=float))
    if not weeks:
        return None, None
    sums = np.array([np.sum(values, axis=0) for values in weeks.values()])
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(sums), size=(resamples, len(sums)))
    counts = np.zeros((resamples, len(sums)))
    np.add.at(counts, (np.arange(resamples)[:, None], draws), 1)
    estimates = np.array([statistic(total) for total in counts @ sums.reshape(len(sums), -1)], dtype=float)
    estimates = estimates[np.isfinite(estimates)]
    if not len(estimates):
        return None, None
    low, high = np.percentile(estimates, [2.5, 97.5])
    return float(low), float(high)


def _summary(rows: Sequence, statistic: Callable) -> dict:
    """The estimate, its interval, ``n`` and each league's estimate."""
    leagues = defaultdict(list)
    for match, values in rows:
        leagues[match.competition].append(values)
    low, high = week_interval(rows, statistic)
    return {"estimate": _finite(statistic(np.sum([v for _, v in rows], axis=0))) if rows else None,
            "low": low, "high": high, "n": len(rows),
            "by_league": {league: {"estimate": _finite(statistic(np.sum(values, axis=0))), "n": len(values)}
                          for league, values in sorted(leagues.items())}}


def _mean(total) -> float:
    return _ratio(total[0], total[1])


# Accuracy and the blend

def _probabilities(match: Match, forecast: Forecast, market: str) -> Optional[tuple]:
    """(model chances, market chances, index of what happened), or None when either side is missing."""
    prices, home, away = match.pre, match.home_goals, match.away_goals
    if market == "result":
        found = market_result(prices)
        model = [chance(forecast, {"market": "moneyline", "side": side, "line": None})
                 for side in ("home", "draw", "away")]
        if found is None or None in model:
            return None
        return model, list(found), 0 if home > away else 1 if home == away else 2
    if market == "handicap":
        found = market_handicap(prices)
        if found is None or not pools._half(prices.handicap_line):
            return None
        option = {"market": "spreads", "side": "home", "line": float(prices.handicap_line)}
    else:
        found = market_over25(prices)
        option = {"market": "goals", "side": "over", "line": 2.5}
    model = chance(forecast, option) if found else None
    if model is None:
        return None
    # both sides as [p, 1 − p]: the market's two chances sum to 1 only up to float noise
    return [model, 1 - model], [found[0], 1 - found[0]], 0 if recommendations.wins(option, home, away) else 1


def _log_loss(probabilities, outcome) -> float:
    return -math.log(min(max(probabilities[outcome], EPS), 1.0))


def accuracy(rows: Sequence, market: str) -> dict:
    """Mean log loss of the model and of the market, and the model's minus the market's (below zero is better)."""
    found = []
    for match, forecast in rows:
        probabilities = _probabilities(match, forecast, market)
        if probabilities:
            model, odds, outcome = probabilities
            mine, theirs = _log_loss(model, outcome), _log_loss(odds, outcome)
            found.append((match, (mine - theirs, mine, theirs, 1.0)))
    totals = np.sum([v for _, v in found], axis=0) if found else None
    return {"market": market, **_summary(found, lambda t: _ratio(t[0], t[3])),
            "model_log_loss": _finite(_ratio(totals[1], totals[3])) if found else None,
            "market_log_loss": _finite(_ratio(totals[2], totals[3])) if found else None}


def blend_score(tuning_rows: Sequence, scored_rows: Sequence, market: str) -> dict:
    """The blend fitted on the tuning rows, with and without the model; on the scored rows, its ``c`` and the log
    loss of the blend minus the market-only recalibration's (below zero: the model adds to the market)."""
    def arrays(rows):
        kept = [(match, found) for match, forecast in rows if (found := _probabilities(match, forecast, market))]
        return ([m for m, _ in kept], [f[1] for _, f in kept], [f[0] for _, f in kept], [f[2] for _, f in kept])

    _, market_fit, model_fit, outcomes = arrays(tuning_rows)
    matches, market_now, model_now, happened = arrays(scored_rows)
    if not outcomes or not happened:
        return {"market": market, "estimate": None, "low": None, "high": None, "n": 0, "by_league": {},
                "c": None, "b": None, "fit_n": len(outcomes)}
    with_model = blend.fit(market_fit, model_fit, outcomes)
    market_only = blend.fit(market_fit, model_fit, outcomes, use_spix=False)
    blended = blend.predict(with_model, market_now, model_now)
    recalibrated = blend.predict(market_only, market_now, model_now)
    rows = [(match, (_log_loss(b, y) - _log_loss(r, y), 1.0))
            for match, b, r, y in zip(matches, blended, recalibrated, happened)]
    return {"market": market, **_summary(rows, _mean), "c": with_model["c"], "b": with_model["b"],
            "fit_n": len(outcomes)}


# Most likely picks, Value and the closing check

@lru_cache(maxsize=40_000)
def _pool(match: Match, phase: str = "pre") -> tuple:
    return tuple(pools.build_pool(match, phase=phase))


@lru_cache(maxsize=40_000)
def _closing(match: Match) -> dict:
    return {pools.option_key(option): option["chance"] for option in _pool(match, "close")}


def closing_chance(match: Match, option: Mapping) -> Optional[float]:
    """The closing fair chance of the option's own line: the real closing price when the line has one, else M1
    solved on the closing prices (as when the closing main handicap sits on another line)."""
    found = _closing(match).get(pools.option_key(option))
    if found is not None:
        return found
    means = pools.m1_means(match.close)
    if means is None:
        return None
    matrix = market_goals._matrix(*means)
    return chance(Forecast(score_matrix=matrix / matrix.sum()), option)


def _engine(options, forecast: Forecast) -> dict:
    return {pools.option_key(o): found for o in options if (found := chance(forecast, o)) is not None}


def _lists(match: Match, forecast: Forecast, strength) -> tuple:
    options = _pool(match)
    return (rules.todays_most_likely(options),
            rules.engine_most_likely(options, _engine(options, forecast), strength=strength))


def _tally(picks, match: Match) -> list:
    """picks, wins, expected wins: for all picks, real lines only, derived lines only."""
    out = []
    for scope in SCOPES:
        kept = [o for o in picks if scope == "all" or o.get("real") == (scope == "real")]
        out += [len(kept), sum(recommendations.wins(o, match.home_goals, match.away_goals) for o in kept),
                sum(o["chance"] for o in kept)]
    return out


def _per_100(total, at: int) -> float:
    return 100 * _ratio(total[at + 1] - total[at + 2], total[at])


def _list_summary(total, at: int) -> dict:
    return {"picks": float(total[at]), "wins": float(total[at + 1]), "expected": float(total[at + 2]),
            "per_100": _finite(_per_100(total, at)), "win_rate": _finite(_ratio(total[at + 1], total[at]))}


def picks_score(rows: Sequence, *, strength) -> dict:
    """Today's Most likely lists and the engine's: wins minus the market's expected wins per 100 picks, and the
    paired difference engine minus today (``estimate``), for all picks and on real or derived lines only."""
    found = []
    for match, forecast in rows:
        today, engine = _lists(match, forecast, strength)
        found.append((match, np.array(_tally(today, match) + _tally(engine, match), dtype=float)))
    width = 3 * len(SCOPES)
    out = {}
    for s, scope in enumerate(SCOPES):
        today_at, engine_at = 3 * s, width + 3 * s
        summary = _summary(found, lambda t, a=today_at, b=engine_at: _per_100(t, b) - _per_100(t, a))
        totals = np.sum([v for _, v in found], axis=0) if found else np.zeros(2 * width)
        summary.update(today=_list_summary(totals, today_at), engine=_list_summary(totals, engine_at))
        if scope == "all":
            out.update(summary)
        else:
            out[scope] = {key: summary[key] for key in ("today", "engine", "estimate", "low", "high")}
    return out


def value_score(rows: Sequence, *, price: str) -> dict:
    """Today's Value rule and the engine's at the average or best price: picks, return per unit, and closing line
    value ``price × closing fair chance − 1``. ``estimate`` and its interval are the engine's closing line value."""
    found = []
    for match, forecast in rows:
        candidates = rules.value_candidates(match, price=price)
        values = []
        for picks in (rules.todays_value(candidates), rules.engine_value(candidates, _engine(candidates, forecast))):
            closing = [closing_chance(match, pick) for pick in picks]
            values += [len(picks),
                       sum(p["price"] - 1 if recommendations.wins(p, match.home_goals, match.away_goals) else -1
                           for p in picks),
                       sum(c is not None for c in closing),
                       sum(p["price"] * c - 1 for p, c in zip(picks, closing) if c is not None)]
        found.append((match, np.array(values, dtype=float)))
    totals = np.sum([v for _, v in found], axis=0) if found else np.zeros(8)

    def rule(at):
        low, high = week_interval(found, lambda t: _ratio(t[at + 3], t[at + 2]))
        return {"picks": float(totals[at]), "return": _finite(_ratio(totals[at + 1], totals[at])),
                "clv": _finite(_ratio(totals[at + 3], totals[at + 2])), "clv_n": float(totals[at + 2]),
                "clv_low": low, "clv_high": high}

    engine = rule(4)
    summary = _summary(found, lambda t: _ratio(t[7], t[6]))
    return {"price": price, "today": rule(0), "engine": engine, "estimate": engine["clv"],
            "low": engine["clv_low"], "high": engine["clv_high"], "n": engine["picks"],
            "by_league": summary["by_league"]}


def closing_check(rows: Sequence, *, strength) -> dict:
    """The mean move of the fair chance from pick time to the close, engine's Most likely picks minus today's."""
    found = []
    for match, forecast in rows:
        values = []
        for picks in _lists(match, forecast, strength):
            moves = [c - o["chance"] for o in picks if (c := closing_chance(match, o)) is not None]
            values += [len(moves), sum(moves)]
        found.append((match, np.array(values, dtype=float)))
    totals = np.sum([v for _, v in found], axis=0) if found else np.zeros(4)
    summary = _summary(found, lambda t: _ratio(t[3], t[2]) - _ratio(t[1], t[0]))
    return {**summary, "today": _finite(_ratio(totals[1], totals[0])),
            "engine": _finite(_ratio(totals[3], totals[2]))}


# Checks and verdicts

def reference_check(matches: Sequence[Match], named: Mapping[int, list]) -> dict:
    """Per season, the mean and 95th percentile of the largest gap, over the three outcomes, between the average
    price's result chance and the median of the named books' power-method chances (matches with 3 books or more)."""
    gaps = defaultdict(list)
    for match in matches:
        books, average = named.get(match.id) or [], market_result(match.pre)
        if average is None or len(books) < MIN_NAMED_BOOKS:
            continue
        median = np.median([devig.power(book) for book in books], axis=0)
        gaps[match.season].append(float(np.max(np.abs(np.asarray(average) - median))))
    return {season: {"mean": float(np.mean(found)), "p95": float(np.percentile(found, 95)), "n": len(found)}
            for season, found in sorted(gaps.items())}


def _blend_passes(score: Mapping) -> bool:
    return score.get("c") is not None and score["c"] > 0 and score.get("high") is not None and score["high"] < 0


def _picks_pass(score: Mapping, control: Optional[Mapping]) -> bool:
    if score.get("low") is None or score["low"] <= 0:
        return False
    if any(league.get("estimate") is not None and league["estimate"] < -MAX_LEAGUE_SHORTFALL
           for league in score.get("by_league", {}).values()):
        return False
    mine = (score.get("derived") or {}).get("estimate")
    theirs = ((control or {}).get("derived") or {}).get("estimate")
    return mine is not None and theirs is not None and mine > theirs


def _value_passes(score: Mapping) -> bool:
    engine = score.get("engine") or {}
    return (score.get("price") == "avg" and engine.get("picks", 0) >= MIN_VALUE_PICKS
            and score.get("low") is not None and score["low"] > 0
            and engine.get("return") is not None and engine["return"] >= 0)


def verdicts(results: Mapping[str, Mapping]) -> dict:
    """Pass or fail per model and score, by the spec's rules. ``results`` maps a model's name to its ``blend``
    (per market), ``picks`` and ``value`` (average price) scores; the ``control`` model's picks are the bar on the
    derived lines."""
    control = (results.get("control") or {}).get("picks")
    out = {}
    for name, found in results.items():
        verdict = {}
        if "blend" in found:
            verdict["blend"] = {market: _blend_passes(score) for market, score in found["blend"].items()}
        if "picks" in found:
            verdict["picks"] = _picks_pass(found["picks"], control)
        if "value" in found:
            verdict["value"] = _value_passes(found["value"])
        out[name] = verdict
    return out
