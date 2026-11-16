"""One match's research view for the Research Area (tracker D110, the overhaul's B2, D107).

The match, Spix's read of it and the past numbers that decide each market. Spix's figures are the read's own: the
figures the match page shows (``match_read_metrics.match_metrics``), and for each over/under market every half line
at least ``fair_prices.MIN_BOOKS`` bookmakers price on both sides, with the market's fair chance and the typical
price, as the recommendation layer lists its options. Cards lines come from the read's cards result and never carry
a suggestion: cards stay out of the picks (D61). Nothing is predicted afresh. The read keeps only whether line-ups
were confirmed, not the players, so ``lineups`` is null.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from .. import fair_prices as fp
from ..reference_price import audit_books
from .match_read_metrics import match_metrics
from .research_profiles import iso
from .research_upcoming import Read, _figure, _time, match_entry, one_decimal, refs_for

SCHEMA_VERSION = "research-match.v1"
TOTALS_KEYS = {"cards": "totals_cards_over_under", "goals": "totals", "corners": "totals_corners_over_under"}


def _results(canonical_results) -> List[Mapping[str, Any]]:
    results = canonical_results if isinstance(canonical_results, Sequence) and not isinstance(
        canonical_results, (str, bytes)) else []
    return [r for r in results if isinstance(r, Mapping)]


def total_lines(canonical_results, market: str) -> List[dict]:
    """Every half line of an over/under market with both sides' fair chance and typical price, lines ascending.
    Each market has the goals market's over/under shape, priced on its own bookmakers' prices."""
    result = fp._results(_results(canonical_results)).get(market)
    if result is None:
        return []
    options = ((result.get("context") or {}).get("decision_audit") or {}).get("input_options")
    books, lines = audit_books(options, TOTALS_KEYS[market]), []
    for line in fp._lines(result, "goals", books):
        if line % 1 != 0.5:
            continue
        entry = fp._entry(result, "goals", books, line)
        if entry is None:
            continue
        sides = {o["side"]: {"chance": o["fair_probability"],
                             "typical_price": fp.typical_price(books, "goals", o["side"], o["line"])}
                 for o in entry["outcomes"]}
        if all(side["typical_price"] is not None for side in sides.values()):
            lines.append({"line": line, **sides})
    return lines


def read_block(card: Mapping[str, Any], canonical_results, *, status: str) -> dict:
    """The ``read`` of the match view from the card the match page shows and the read's canonical results."""
    results = _results(canonical_results)
    metrics = match_metrics(results)
    outcome = metrics.get("result_probabilities") or {}
    result = {k: outcome[k] for k in ("home", "draw", "away")} if all(k in outcome for k in ("home", "draw", "away")) \
        else None
    team_goals = metrics.get("team_goals") or {}
    home, away = one_decimal(team_goals.get("home")), one_decimal(team_goals.get("away"))
    goals = {"home": home, "away": away, "total": _figure(metrics, "goals")} if home is not None and away is not None \
        else None
    block = card.get("recommendations")
    markets = block.get("markets") if isinstance(block, Mapping) else None
    suggestions = {m.get("market"): m.get("suggestion") for m in markets or [] if isinstance(m, Mapping)}
    return {"status": status, "read_at": _time(card.get("created_at")),
            "prices_as_of": (card.get("provenance") or {}).get("evaluated_at"),
            "result": result, "goals": goals, "btts_yes": team_goals.get("both_teams_score_probability"),
            "corners_expected": _figure(metrics, "corners"), "cards_expected": _figure(metrics, "cards"),
            "lines": {market: total_lines(results, market) for market in TOTALS_KEYS},
            "suggestions": {"cards": None, "goals": suggestions.get("goals"), "corners": suggestions.get("corners")},
            "markets": markets}


class ResearchMatchService:
    def __init__(self, repo, *, load_read: Callable[[str], Optional[Read]], now: Callable[[], datetime],
                 outlook_hours: int):
        self.repo, self.load_read, self.now, self.outlook_hours = repo, load_read, now, outlook_hours

    def match(self, event_id: str) -> dict:
        now = self.now()
        with self.repo.session_factory() as session:
            row = self.repo.fixture(session, int(event_id))
            if row is None:
                raise LookupError("Unknown match")
            refs = refs_for(self.repo, session, [row["referee"]])
        read = self.load_read(str(row["api_football_id"]))
        fixture = {**match_entry(row, refs, load_read=lambda _: read, now=now, outlook_hours=self.outlook_hours),
                   "status": row["status"]}
        return {"schema_version": SCHEMA_VERSION, "fixture": fixture,
                "read": None if read is None else read_block(read[0], read[1], status=fixture["read"]["status"]),
                "teams": None, "referee": None, "h2h": [], "lineups": None,
                "data_basis": {"generated_at": iso(now)}}
