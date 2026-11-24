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

D129, behind the engine switch ``SPIX_BEST_BETS=fairest``: Most likely becomes the fairest bets in each payout range.
Options priced ``MIN_PRICE`` to under ``MAX_FAIR_PRICE`` where the bookmaker keeps at most ``MAX_KEEP`` of each stake;
the fairest under ``PAYOUT_SPLIT``, the fairest from it, then the next fairest, from three markets that can all win
together, listed fairest first. Each market marks its bet among them, or its own fairest. The rule is the one the
read recorded when it was made, so a card already shown keeps its bets; a read from before the switch is likeliest.
"""
from __future__ import annotations

import itertools
import os
from typing import Any, Dict, Iterable, List, Mapping, Optional

from Scripts.rag_ingest.core import engine_switches

from . import fair_prices as fp

VERSION = "recommendations.v1"
MIN_CHANCE = 0.50
MIN_PRICE = 1.30
MAX_MOST_LIKELY = 3
MAX_VALUE = 2
LONG_SHOT = 0.25
MAX_SCORE = 10
MAX_FAIR_PRICE = 2.00  # D129: from here the fair bets on one match nearly always back the other side
PAYOUT_SPLIT = 1.60
MAX_KEEP = 0.06
BEST_BETS_SWITCH = "SPIX_BEST_BETS"
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


def keeps(option: Mapping[str, Any]) -> Optional[float]:
    """The bookmaker's share of each stake at the typical price, by the market's own chance; None without a price."""
    price = option.get("typical_price")
    return None if price is None else 1 - option["chance"] * price


def fair_options(market_options: Iterable[Mapping[str, Any]]) -> List[dict]:
    """The options priced ``MIN_PRICE`` to under ``MAX_FAIR_PRICE`` where the bookmaker keeps at most ``MAX_KEEP``."""
    return [o for o in market_options or [] if o.get("typical_price") is not None
            and MIN_PRICE <= o["typical_price"] < MAX_FAIR_PRICE and keeps(o) <= MAX_KEEP]


def _fairness(option: Mapping[str, Any]):
    return keeps(option), -option["chance"]


def _all_win_together(options: List[Mapping[str, Any]]) -> bool:
    """True when one scoreline wins every goal-market option at once; corners never conflict."""
    goals = [o for o in options if o.get("market") != "corners"]
    return any(all(wins(o, h, w) for o in goals) for h, w in _scorelines())


def best_bets(market_options: Iterable[Mapping[str, Any]]) -> List[dict]:
    """D129: the fairest option under ``PAYOUT_SPLIT``, the fairest from it, then the next fairest, from three markets
    that can all win together and never the same bet twice; listed fairest first."""
    order = sorted(fair_options(market_options), key=_fairness)
    kept: List[dict] = []
    for pool in ([o for o in order if o["typical_price"] < PAYOUT_SPLIT],
                 [o for o in order if o["typical_price"] >= PAYOUT_SPLIT], order):
        for option in pool:
            if len(kept) < MAX_MOST_LIKELY and option not in kept \
                    and all(k["market"] != option["market"] and not same_bet(k, option) for k in kept) \
                    and _all_win_together(kept + [option]):
                kept.append(option)
                break
    return sorted(kept, key=_fairness)


def fairest_suggestion(market_options: List[dict], picks: List[dict]) -> Optional[dict]:
    """One market's mark under the fairest rule: its bet among ``picks``, else its fairest option that is not the
    same bet as one of them."""
    mine = [p for p in picks if p in (market_options or [])]
    if mine:
        return mine[0]
    fair = [o for o in fair_options(market_options) if not any(same_bet(o, p) for p in picks)]
    return min(fair, key=_fairness) if fair else None


def rule(canonical_results: Iterable[Mapping[str, Any]]) -> str:
    """The Most likely rule the read recorded with its engine switches; likeliest when it recorded none or another."""
    default, accepted = engine_switches.SWITCHES[BEST_BETS_SWITCH]
    for result in canonical_results or []:
        switches = (result.get("context") or {}).get("engine_switches") if isinstance(result, Mapping) else None
        if isinstance(switches, Mapping) and BEST_BETS_SWITCH in switches:
            recorded = str(switches[BEST_BETS_SWITCH]).strip().lower()
            return recorded if recorded in accepted else default
    return default


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
    The chance is the one the pick was made with (D68), so "Worth it at" is never above the price Spix found; a pick
    without one takes its option's fair-card chance. A handicap quote names its team in ``side``; the options say
    home or away."""
    market, quote = (selection.get("market") or {}).get("group"), _quote(selection)
    side, line = quote.get("side"), None if quote.get("line") is None else float(quote.get("line"))
    picked = selection.get("model_probability")
    if isinstance(picked, (int, float)) and not isinstance(picked, bool) and 0 < picked < 1:
        chance = float(picked)
    else:
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
    if rule(canonical_results) == "fairest":
        picks = best_bets(o for m in MARKETS for o in by_market.get(m, []))
        markets = [{"market": m, "suggestion": fairest_suggestion(by_market.get(m, []), picks),
                    "options": by_market.get(m, [])} for m in MARKETS]
    else:
        markets = [{"market": m, "suggestion": suggestion(by_market.get(m, [])), "options": by_market.get(m, [])}
                   for m in MARKETS]
        picks = most_likely(m["suggestion"] for m in markets)
    corners = _figure(((results.get("corners") or {}).get("projection") or {}).get("value"))
    fixture = next((r.get("fixture") for r in canonical_results if isinstance(r.get("fixture"), Mapping)), {})
    likely = [{**o, "why": why(o, metrics, fixture, corners=corners)} for o in picks]
    ordered = sorted((s for s in selections or [] if isinstance(s, Mapping)
                      and (s.get("market") or {}).get("group") in MARKETS),  # cards and SoT stay out (D61)
                     key=lambda s: s.get("position") or 0)
    return {"version": VERSION, "prices_as_of": prices_as_of, "most_likely": likely,
            "value": [_value(canonical_results, s, fixture) for s in ordered[:MAX_VALUE]], "markets": markets,
            "score_grid": _score_grid(results)}
