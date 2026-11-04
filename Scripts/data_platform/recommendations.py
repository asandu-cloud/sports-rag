"""Recommendation layer (tracker D52, D53): what Spix would take in each market of a match, from a read's prices.

Every option is a line of Result, Goals, Both teams to score, Handicap or Corners that at least
``fair_prices.MIN_BOOKS`` bookmakers price on every side, half lines only, so no bet ends with the stake returned or
split. Its chance is the market's fair chance (power method, median across books, the fair-price card's numbers);
its typical price is the median of the bookmakers' best prices, with no bookmaker named.

Spix's suggestion in a market is its likeliest option at a chance of at least ``MIN_CHANCE`` and a typical price of
at least ``MIN_PRICE``. Most likely is the top ``MAX_MOST_LIKELY`` markets' suggestions that can all win together:
goal-market legs are checked against the scorelines 0-0 to ``MAX_SCORE``-``MAX_SCORE``; corners never conflict.
Value is the read's released selections (stage 4) in the layer's markets, each with "Worth it at X or better": the
fair-price card's minimum odds at the outcome's fair chance, any line; a cards or shots-on-target pick stays out
until the owner adds those markets (D61). No engine output changes: the lists are compiled from what the
read already stores. The block reaches the card only when the display switch ``SPIX_RECOMMENDATIONS`` is ``on``.
"""
from __future__ import annotations

import itertools
import os
from typing import Any, Dict, Iterable, List, Mapping, Optional

from . import fair_prices as fp

VERSION = "recommendations.v1"
MIN_CHANCE = 0.50
MIN_PRICE = 1.30
MAX_MOST_LIKELY = 3
MAX_VALUE = 2
LONG_SHOT = 0.25
MAX_SCORE = 10
MARKETS = ("moneyline", "goals", "btts", "spreads", "corners")
GOAL_MARKETS = MARKETS[:4]


def _half_line(line) -> bool:
    return line is None or (isinstance(line, (int, float)) and line % 1 == 0.5)


def _signed(line: float) -> str:
    return f"{'−' if line < 0 else '+'}{abs(line):g}"


def _label(result: Mapping[str, Any], market: str, side: str, line) -> str:
    fixture = result.get("fixture") or {}
    team = {"home": fixture.get("home_team"), "away": fixture.get("away_team")}
    if market in ("goals", "corners"):
        return f"{side.capitalize()} {line:g} {market}"
    if market == "moneyline":
        return "Draw" if side == "draw" else f"{team[side]} to win"
    if market == "btts":
        return "Both teams to score" if side == "yes" else "Not both teams to score"
    return f"{team[side]} {_signed(line)}"


def _market_options(result: Mapping[str, Any], market: str) -> List[dict]:
    books, found = fp._books(result, market), []
    for line in fp._lines(result, market, books):
        if not _half_line(line):
            continue
        entry = fp._entry(result, market, books, line)
        if entry is None:
            continue
        priced = [(o, fp.typical_price(books, market, o["side"], o["line"])) for o in entry["outcomes"]]
        if any(price is None for _, price in priced):
            continue
        for index, (outcome, price) in enumerate(priced):
            side = ("home", "away")[index] if market == "spreads" else outcome["side"]
            chance = outcome["fair_probability"]
            found.append({"market": market, "side": side, "line": outcome["line"],
                          "label": _label(result, market, side, outcome["line"]), "chance": chance,
                          "typical_price": price, "long_shot": chance < LONG_SHOT})
    return found


def options(canonical_results: Iterable[Mapping[str, Any]]) -> Dict[str, List[dict]]:
    """Every option of each market the read covers, lines ascending, sides in the card's order (handicap: home
    team first, each side with its own handicap)."""
    results = fp._results(canonical_results)
    return {market: _market_options(results[market], market) for market in MARKETS if market in results}


def wins(option: Mapping[str, Any], home: int, away: int) -> bool:
    """Whether a goal-market option wins on the scoreline ``home``-``away``."""
    market, side, line = option.get("market"), option.get("side"), option.get("line")
    if market == "goals":
        return home + away > line if side == "over" else home + away < line
    if market == "btts":
        return (home > 0 and away > 0) == (side == "yes")
    if market == "moneyline":
        return {"home": home > away, "draw": home == away, "away": away > home}[side]
    if market == "spreads":
        return home + line > away if side == "home" else away + line > home
    raise ValueError(f"not a goal market: {market!r}")


def can_win_together(a: Mapping[str, Any], b: Mapping[str, Any]) -> bool:
    """True when at least one scoreline wins both; corners never conflict with anything."""
    if "corners" in (a.get("market"), b.get("market")):
        return True
    return any(wins(a, h, w) and wins(b, h, w) for h, w in _scorelines())


def _scorelines():
    return itertools.product(range(MAX_SCORE + 1), repeat=2)


def same_bet(a: Mapping[str, Any], b: Mapping[str, Any]) -> bool:
    """True when two goal-market options win on exactly the same scorelines (e.g. a team −0.5 and that team to win)."""
    if "corners" in (a.get("market"), b.get("market")):
        return False
    return all(wins(a, h, w) == wins(b, h, w) for h, w in _scorelines())


def _rank(option: Mapping[str, Any]):
    return -option["chance"], -option["typical_price"]


def suggestion(market_options: List[dict]) -> Optional[dict]:
    """Spix's suggestion in one market: the likeliest option that clears both floors, ties to the higher price."""
    eligible = [o for o in market_options or [] if o["chance"] >= MIN_CHANCE and o["typical_price"] >= MIN_PRICE]
    return min(eligible, key=_rank) if eligible else None


def most_likely(suggestions: Iterable[Optional[dict]]) -> List[dict]:
    """Up to ``MAX_MOST_LIKELY`` markets' suggestions, by chance, each able to win with every one kept before it and
    none the same bet as one kept under another market's name."""
    kept: List[dict] = []
    for option in sorted((s for s in suggestions if s), key=_rank):
        if len(kept) == MAX_MOST_LIKELY:
            break
        if all(k["market"] != option["market"] and can_win_together(k, option) and not same_bet(k, option)
               for k in kept):
            kept.append(option)
    return kept


def enabled() -> bool:
    """The display switch: the card carries the block only when ``SPIX_RECOMMENDATIONS`` is ``on``."""
    return os.environ.get("SPIX_RECOMMENDATIONS", "").strip().lower() == "on"


def _figure(value) -> Optional[float]:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def why(option: Mapping[str, Any], metrics: Mapping[str, Any], fixture: Mapping[str, Any], *,
        corners: Optional[float] = None) -> Optional[str]:
    """One plain sentence from the read's own figures (owner signs off the copy); None when a figure is missing."""
    market = option.get("market")
    if market == "goals":
        total = _figure((metrics or {}).get("projected_total_goals"))
        return None if total is None else f"Spix expects about {total:.1f} goals."
    if market == "corners":
        return None if corners is None else f"Spix expects about {corners:.1f} corners."
    team_goals = (metrics or {}).get("team_goals") or {}
    home, away = _figure(team_goals.get("home")), _figure(team_goals.get("away"))
    if home is None or away is None:
        return None
    return (f"Spix expects {(fixture or {}).get('home_team')} about {home:.1f} goals "
            f"and {(fixture or {}).get('away_team')} about {away:.1f}.")


def _quote(selection: Mapping[str, Any]) -> Mapping[str, Any]:
    stored = ((selection.get("data") or {}).get("decision") or {}).get("quote")
    return stored if isinstance(stored, Mapping) else selection.get("quote") or {}


def _value(canonical_results, selection: Mapping[str, Any], fixture: Mapping[str, Any]) -> dict:
    """One Value item: the pick's chance and "Worth it at", and its side, line and label as its option has them.
    A handicap quote names its team in ``side``; the options say home or away."""
    market, quote = (selection.get("market") or {}).get("group"), _quote(selection)
    side, line = quote.get("side"), None if quote.get("line") is None else float(quote.get("line"))
    outcome = fp.fair_outcome(canonical_results, market, side, line) if side else None
    chance = outcome["fair_probability"] if outcome else None
    if market == "spreads":
        side = dict(zip(fp._teams({"fixture": fixture}), ("home", "away"))).get(fp.norm(side))
    named = side in (("home", "away") if market == "spreads" else fp.SIDES.get(market, ())) and (
        line is not None or market in ("moneyline", "btts"))
    return {"position": selection.get("position"), "market": market, "side": side if named else None, "line": line,
            "label": _label({"fixture": fixture}, market, side, line) if named else selection.get("pick"),
            "pick": selection.get("pick"), "chance": chance, "worth_it_at": fp.minimum_odds(chance) if chance else None,
            "published_recommendation_id": selection.get("published_recommendation_id")}


def _score_grid(results: Mapping[str, Mapping[str, Any]]) -> Optional[dict]:
    for market in ("goals", "btts", "spreads"):
        fit = (((results.get(market) or {}).get("context") or {}).get("market_anchor") or {}).get("goal_model")
        if isinstance(fit, Mapping) and _figure(fit.get("home")) is not None and _figure(fit.get("away")) is not None:
            return {"home": float(fit["home"]), "away": float(fit["away"])}
    return None


def block(canonical_results: Iterable[Mapping[str, Any]], *, selections: Iterable[Mapping[str, Any]],
          metrics: Mapping[str, Any], prices_as_of) -> dict:
    """The ``recommendations.v1`` block of one read: Most likely, Value, every market's options and suggestion, and
    the score grid same-match goal legs combine through (null when the prices could not fit one)."""
    canonical_results = [r for r in canonical_results or [] if isinstance(r, Mapping)]
    results, by_market = fp._results(canonical_results), options(canonical_results)
    markets = [{"market": m, "suggestion": suggestion(by_market.get(m, [])), "options": by_market.get(m, [])}
               for m in MARKETS]
    corners = _figure(((results.get("corners") or {}).get("projection") or {}).get("value"))
    fixture = next((r.get("fixture") for r in canonical_results if isinstance(r.get("fixture"), Mapping)), {})
    likely = [{**o, "why": why(o, metrics, fixture, corners=corners)}
              for o in most_likely(m["suggestion"] for m in markets)]
    ordered = sorted((s for s in selections or [] if isinstance(s, Mapping)
                      and (s.get("market") or {}).get("group") in MARKETS),  # cards and SoT stay out (D61)
                     key=lambda s: s.get("position") or 0)
    return {"version": VERSION, "prices_as_of": prices_as_of, "most_likely": likely,
            "value": [_value(canonical_results, s, fixture) for s in ordered[:MAX_VALUE]], "markets": markets,
            "score_grid": _score_grid(results)}
