"""Pick evidence on the match card (tracker D82, design ``docs/superpowers/specs/2026-11-09-pick-evidence-design.md``).

For each Most likely pick: three groups of past league matches chosen only by how the bookmakers priced them (each
team's matches with its own win odds closest to tonight's, and the league's matches with home and away odds closest
to tonight's), how many of each the pick would have won, a verdict set by a rule on those counts, and one detail line
in fixed wording. Above the picks, "How Spix reads it": a lead sentence and the card's own figures. Pure: the
history comes in as plain dicts (``services.pick_evidence_service`` loads it), and nothing here changes a pick.

A history match is ``{date, season, competition, home_id, away_id, home, away, home_goals, away_goals, corners,
home_price, draw_price, away_price}``.
"""
from __future__ import annotations

from datetime import date
import math
from typing import Any, Dict, List, Mapping, Optional, Sequence

from . import recommendations

VERSION = "pick-evidence.v1"
SINCE = "2024-08-01"
DOMESTIC = ("EPL", "Championship", "LaLiga", "SerieA", "Bundesliga", "Ligue1", "Eredivisie", "PrimeiraLiga",
            "SuperLig", "BelgianProLeague")
TEAM_MATCHES, TEAM_MINIMUM = 12, 8
LEAGUE_MATCHES, LEAGUE_MINIMUM = 40, 20
BACKS, AGAINST = 0.6, 0.5
LEAD_WORDS, DETAIL_WORDS = 14, 25
SHOWN = ("date", "season", "competition", "home", "away", "home_price", "draw_price", "away_price", "home_goals",
         "away_goals", "corners")


def _newest_first(row) -> int:
    return -date.fromisoformat(row["date"]).toordinal()


def team_group(history: Sequence[Mapping[str, Any]], team_id: int, price: float, before: str, *,
               n: int = TEAM_MATCHES) -> List[dict]:
    """The team's ``n`` matches before ``before`` whose own win odds are closest to ``price`` (log distance, ties to
    the newer match), newest first, each with ``venue`` and ``team_price``."""
    rows = []
    for row in history:
        if row["date"] >= before:
            continue
        if row["home_id"] == team_id:
            rows.append({**row, "venue": "H", "team_price": row["home_price"]})
        elif row["away_id"] == team_id:
            rows.append({**row, "venue": "A", "team_price": row["away_price"]})
    rows.sort(key=lambda r: (abs(math.log(r["team_price"]) - math.log(price)), _newest_first(r)))
    return sorted(rows[:n], key=_newest_first)


def league_group(history: Sequence[Mapping[str, Any]], league: str, home_price: float, away_price: float, before: str,
                 *, n: int = LEAGUE_MATCHES) -> List[dict]:
    """The league's ``n`` matches before ``before`` whose home and away odds are closest to tonight's, newest first."""
    rows = [dict(r) for r in history if r["competition"] == league and r["date"] < before]
    rows.sort(key=lambda r: (math.hypot(math.log(r["home_price"] / home_price), math.log(r["away_price"] / away_price)),
                             _newest_first(r)))
    return sorted(rows[:n], key=_newest_first)


def roles(row: Mapping[str, Any], view: str) -> tuple:
    """(goals of tonight's home side's stand-in, goals of the away side's) in a past match. In the home team's group
    that team is the home side; in the away team's group its opponent is; in the league group the venue decides."""
    hg, ag = row["home_goals"], row["away_goals"]
    if view == "league":
        return hg, ag
    group_team_was_home = row["venue"] == "H"
    if view == "home":
        return (hg, ag) if group_team_was_home else (ag, hg)
    return (ag, hg) if group_team_was_home else (hg, ag)


def won(pick: Mapping[str, Any], row: Mapping[str, Any], *, view: str) -> bool:
    if pick["market"] == "corners":
        return row["corners"] > pick["line"] if pick["side"] == "over" else row["corners"] < pick["line"]
    home, away = roles(row, view)
    return recommendations.wins(pick, home, away)


def _margin_phrase(subject: str, line: float, *, own: bool) -> str:
    """What ``margin + line > 0`` means for the picked side (``own``), or for the other side, whose margin is the
    picked side's turned round."""
    k = int(abs(line) - 0.5)
    if own:
        if line > 0:
            return f"{subject} did not lose" + (f" by more than {k}" if k else "")
        return f"{subject} won" + (f" by {k + 1} or more" if k else "")
    if line > 0:
        return f"{subject} did not win" + (f" by more than {k}" if k else "")
    return f"{subject} lost" + (f" by {k + 1} or more" if k else "")


def means(pick: Mapping[str, Any], *, view: str, home: str, away: str) -> str:
    """What the pick winning was, in plain words, as seen from one group."""
    market, side, line = pick["market"], pick["side"], pick.get("line")
    if market in ("goals", "corners"):
        edge = math.ceil(line) if side == "over" else math.floor(line)
        return f"the match had {edge} or {'more' if side == 'over' else 'fewer'} {market}"
    if market == "btts":
        return "both teams scored" if side == "yes" else "at least one team failed to score"
    if market == "moneyline" and side == "draw":
        return "the match was drawn"
    line = -0.5 if market == "moneyline" else line
    if view == "league":
        return _margin_phrase(f"the {side} side", line, own=True)
    team = {"home": home, "away": away}[view]
    return _margin_phrase(team, line, own=view == side)


def verdict(rates: Sequence[float]) -> str:
    """A group backs the pick at ``BACKS`` or more and leans against it under ``AGAINST``; two or three groups."""
    n, backs, against = len(rates), sum(r >= BACKS for r in rates), sum(r < AGAINST for r in rates)
    if backs == n:
        return "All three groups agree." if n == 3 else "Both groups agree."
    if against == n:
        return "All three groups lean against it." if n == 3 else "Both groups lean against it."
    if against >= 2:
        return "Two of the three groups lean against it."
    if backs == 2:
        return "Two of the three groups agree."
    if backs and against:
        return "The groups are split."
    if backs:
        return "Only one group clearly agrees."
    return "The groups don't settle it."


def _words(text: str) -> int:
    return len(text.split())


def lead(home: str, away: str, chances: Mapping[str, float], goals: Optional[Mapping[str, float]]) -> str:
    """Who the favourite is and where the goals are expected, in at most ``LEAD_WORDS`` words."""
    (fav, p_fav), (_, p_other) = sorted(((home, chances["home"]), (away, chances["away"])), key=lambda t: -t[1])
    if p_fav >= 0.60:
        first = f"{fav} are clear favourites"
    elif p_fav - p_other >= 0.12:
        first = f"{fav} are favourites"
    elif p_fav - p_other >= 0.04:
        first = f"{fav} are slight favourites"
    else:
        first = "Too close to call"
    if not goals or goals.get("home") is None or goals.get("away") is None:
        return first + "."
    g_home, g_away = goals["home"], goals["away"]
    if g_home + g_away >= 3.0:
        second = "with plenty of goals expected"
    elif g_home + g_away <= 2.2:
        second = "with few goals expected"
    elif min(g_home, g_away) >= 1.1:
        second = "with goals expected at both ends"
    else:
        second = f"with most goals expected from {home if g_home >= g_away else away}"
    both = f"{first}, {second}."
    return both if _words(both) <= LEAD_WORDS else first + "."


def _record(rows: Sequence[Mapping[str, Any]], view: str, side: str) -> tuple:
    """(won, drew, lost) of the picked side's stand-in across a group."""
    margins = []
    for row in rows:
        home, away = roles(row, view)
        margins.append(home - away if side == "home" else away - home)
    return sum(m > 0 for m in margins), sum(m == 0 for m in margins), sum(m < 0 for m in margins)


def _scored(rows: Sequence[Mapping[str, Any]]) -> int:
    return sum((r["home_goals"] if r["venue"] == "H" else r["away_goals"]) > 0 for r in rows)


def _first_sentence(pick, groups, home, away) -> str:
    market, side = pick["market"], pick["side"]
    by_key = {g["key"]: g["rows"] for g in groups}
    if market in ("goals", "corners"):
        field = (lambda r: r["home_goals"] + r["away_goals"]) if market == "goals" else (lambda r: r["corners"])
        averages = sorted(round(sum(field(r) for r in rows) / len(rows), 1) for rows in by_key.values() if rows)
        if not averages:
            return ""
        spread = f"{averages[0]:.1f}" if averages[0] == averages[-1] else f"{averages[0]:.1f} to {averages[-1]:.1f}"
        return f"Similar matches averaged {spread} {market}."
    if market == "btts":
        parts = [(name, _scored(by_key[key]), len(by_key[key])) for key, name in (("home", home), ("away", away))
                 if by_key.get(key)]
        if len(parts) == 2:
            return f"{parts[0][0]} scored in {parts[0][1]} of their {parts[0][2]}, {parts[1][0]} in {parts[1][1]} of theirs."
        if parts:
            return f"{parts[0][0]} scored in {parts[0][1]} of their {parts[0][2]} similar matches."
        return ""
    if side == "draw":
        rows = by_key.get("league") or []
        drawn = sum(r["home_goals"] == r["away_goals"] for r in rows)
        return f"{drawn} of the {len(rows)} league matches at similar odds were drawn." if rows else ""
    other = "away" if side == "home" else "home"
    for key in (side, other):
        if by_key.get(key):
            w, d, l = _record(by_key[key], key, key)
            team = home if key == "home" else away
            return f"{team} won {w}, drew {d} and lost {l} of their {w + d + l} matches at similar odds."
    return ""


def _safer(pick: Mapping[str, Any], options: Sequence[Mapping[str, Any]]) -> Optional[Mapping[str, Any]]:
    """The likelier option on the same side of the same market that pays under ``MIN_PRICE``, the closest one."""
    safer = [o for o in options or [] if o.get("market") == pick["market"] and o.get("side") == pick["side"]
             and o.get("chance") is not None and o["chance"] > pick["chance"]
             and o.get("typical_price") is not None and o["typical_price"] < recommendations.MIN_PRICE]
    return min(safer, key=lambda o: o["chance"]) if safer else None


def detail(pick: Mapping[str, Any], groups: Sequence[Mapping[str, Any]], options: Sequence[Mapping[str, Any]],
           home: str, away: str) -> str:
    """What the rows don't show: an average, a record or each side's scoring, then the safer line and its price.
    At most ``DETAIL_WORDS`` words; a sentence that would break that is left out."""
    sentences = [_first_sentence(pick, groups, home, away)]
    safer = _safer(pick, options)
    if safer:
        sentences.append(f"{safer['label']} is safer but pays only {safer['typical_price']:.2f}.")
    kept: List[str] = []
    for sentence in (s for s in sentences if s):
        if _words(" ".join(kept + [sentence])) <= DETAIL_WORDS:
            kept.append(sentence)
    return " ".join(kept)


def _figure(value) -> Optional[float]:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def story(fixture: Mapping[str, Any], figures: Mapping[str, Any]) -> Optional[dict]:
    """How Spix reads it: the lead and the card's figures; None without the result chances."""
    chances = figures.get("result_chances")
    if not isinstance(chances, Mapping) or any(_figure(chances.get(k)) is None for k in ("home", "draw", "away")):
        return None
    goals = figures.get("goals_expected")
    goals = goals if isinstance(goals, Mapping) and all(_figure(goals.get(k)) is not None for k in ("home", "away")) \
        else None
    return {"lead": lead(fixture["home_team"], fixture["away_team"], chances, goals), "result_chances": dict(chances),
            "goals_expected": dict(goals) if goals else None,
            "corners_expected": _figure(figures.get("corners_expected"))}


def _tonight(block: Mapping[str, Any]) -> Optional[dict]:
    moneyline = next((m for m in block.get("markets") or [] if m.get("market") == "moneyline"), None)
    prices = {o.get("side"): _figure(o.get("typical_price")) for o in (moneyline or {}).get("options") or []}
    if any(not prices.get(side) or prices[side] <= 1 for side in ("home", "draw", "away")):
        return None
    return {side: prices[side] for side in ("home", "draw", "away")}


def _shown(row: Mapping[str, Any], *, team: bool) -> dict:
    out = {key: row[key] for key in SHOWN}
    if team:
        out["venue"] = row["venue"]
    return out


def build(block: Mapping[str, Any], fixture: Mapping[str, Any], *, history: Sequence[Mapping[str, Any]],
          team_ids: Optional[tuple], figures: Mapping[str, Any]) -> Dict[str, Any]:
    """``{"story", "evidence", "picks"}`` for one card: ``evidence`` holds the shared groups (None outside the
    domestic leagues, without tonight's 1X2 prices or the teams, or with fewer than two groups); ``picks`` maps each
    Most likely position to its verdict, detail and counts."""
    out: Dict[str, Any] = {"story": story(fixture, figures), "evidence": None, "picks": {}}
    league, tonight = fixture.get("league"), _tonight(block)
    if league not in DOMESTIC or tonight is None or not team_ids or None in team_ids:
        return out
    before = str(fixture.get("kickoff") or "")[:10] or "9999-12-31"
    home, away = fixture["home_team"], fixture["away_team"]
    candidates = [("home", home, team_group(history, team_ids[0], tonight["home"], before), TEAM_MINIMUM),
                  ("away", away, team_group(history, team_ids[1], tonight["away"], before), TEAM_MINIMUM),
                  ("league", None, league_group(history, league, tonight["home"], tonight["away"], before),
                   LEAGUE_MINIMUM)]
    groups = [{"key": key, "team": team, "rows": rows} for key, team, rows, least in candidates if len(rows) >= least]
    if len(groups) < 2:
        return out
    out["evidence"] = {
        "version": VERSION, "since": SINCE, "competition": league, "tonight": tonight,
        "groups": [{"key": g["key"], "team": g["team"],
                    "price_from": round(min(r["team_price"] for r in g["rows"]), 2) if g["team"] else None,
                    "price_to": round(max(r["team_price"] for r in g["rows"]), 2) if g["team"] else None,
                    "matches": [_shown(r, team=bool(g["team"])) for r in g["rows"]]} for g in groups]}
    options = [o for m in block.get("markets") or [] for o in m.get("options") or []]
    for index, pick in enumerate(block.get("most_likely") or []):
        if pick.get("market") not in recommendations.MARKETS or pick.get("side") is None:
            continue
        counted = []
        for g in groups:
            results = [won(pick, r, view=g["key"]) for r in g["rows"]]
            counted.append({"key": g["key"], "won": sum(results), "of": len(results),
                            "means": means(pick, view=g["key"], home=home, away=away), "results": results})
        out["picks"][index] = {"verdict": verdict([c["won"] / c["of"] for c in counted]),
                               "detail": detail(pick, groups, options, home, away), "groups": counted}
    return out
