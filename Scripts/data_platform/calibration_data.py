"""A4.1: calibration dataset from the decision audits of match reads.

Every evaluated candidate (not only published picks) is taken from the latest match
read made before kickoff, per fixture and stage, de-duplicated per market, side and
line (bookmakers quoting the same line share one model probability; the first bookmaker
with a two-sided price supplies the de-vigged market probability). Each selection
is graded with the settlement's own Asian rules: half results count half, pushes and
voids carry no weight (as in the scorecard).

Cards are graded as yellow + red cards from the match statistics; bookmakers'
card rules differ (bookings points, second yellows), so card calibration is
indicative until the card definition is qualified.
"""
from __future__ import annotations

from collections import defaultdict
from contextlib import closing
from datetime import datetime, timezone
import json
import sqlite3

from .scorecard import outcome_weight
from .settlement import asian_outcome, utc_datetime

TOTALS = {"goals", "corners", "cards"}


def _time(value):
    """ISO time with a zone, or the database's naive UTC form ('2026-10-16 18:00:59.000000')."""
    parsed = utc_datetime(value)
    if parsed is not None or value is None:
        return parsed
    try:
        naive = datetime.fromisoformat(str(value).strip().replace(" ", "T"))
    except ValueError:
        return None
    return naive.replace(tzinfo=timezone.utc) if naive.tzinfo is None else naive.astimezone(timezone.utc)


def latest_reads(rows):
    """Per (fixture, stage), the latest read evaluated strictly before kickoff."""
    best = {}
    for row in rows:
        evaluated, kickoff = _time(row["evaluated_at"]), _time(row["kickoff"])
        if evaluated is None or kickoff is None or evaluated >= kickoff:
            continue
        key = (str(row["fixture_api_id"]), row["stage"])
        if key not in best or evaluated > _time(best[key]["evaluated_at"]):
            best[key] = row
    return list(best.values())



READ_COLUMNS = ("fixture_api_id", "stage", "evaluated_at", "kickoff")


def latest_read_rows(db_path, *, batch=500):
    """latest_reads() of the database's match reads, each read's JSON fetched on its own as it is used.

    Short statements only: the database is in rollback-journal mode and its writers wait 2 s for a lock, and
    a read's JSON is ~0.75 MB, so one scan of every read holds the lock long enough to fail the jobs' commits.
    """
    with closing(sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=60)) as db:
        rows, last = [], 0
        while chunk := db.execute(f"SELECT rowid, {', '.join(READ_COLUMNS)} FROM match_reads WHERE rowid > ? "
                                  "ORDER BY rowid LIMIT ?", (last, batch)).fetchall():
            rows += [{"rowid": r[0], **dict(zip(READ_COLUMNS, r[1:]))} for r in chunk]
            last = chunk[-1][0]
        for row in latest_reads(rows):
            found = db.execute("SELECT read_json FROM match_reads WHERE rowid = ?", (row.pop("rowid"),)).fetchall()
            if found:
                yield {**row, "read_json": found[0][0]}

def selections(row):
    data = json.loads(row["read_json"]) if isinstance(row["read_json"], str) else row["read_json"]
    out, seen = [], {}
    for res in data.get("canonical_results", []):
        group = (res.get("market") or {}).get("group")
        fixture = res.get("fixture") or {}
        projection = res.get("projection") or {}
        for candidate in ((res.get("context") or {}).get("decision_audit") or {}).get("candidates") or []:
            option = candidate.get("option") or {}
            side = option.get("side") if option.get("side") is not None else option.get("team")
            probability = option.get("_model_prob")
            if side is None or probability is None:
                continue
            key = (group, option.get("market_key"), str(side).casefold(), option.get("point"))
            market_prob = option.get("_implied_prob") if option.get("_vig_adjusted_pair") else None
            if key in seen:
                seen[key]["selected"] = seen[key]["selected"] or bool(candidate.get("selector_selected"))
                if seen[key]["market_prob"] is None:
                    seen[key]["market_prob"] = market_prob
                continue
            seen[key] = {"fixture_id": str(row["fixture_api_id"]), "stage": row["stage"], "kickoff": row["kickoff"],
                         "market": group, "market_key": option.get("market_key"), "side": side,
                         "point": option.get("point"), "model_prob": float(probability),
                         "mean": projection.get("value"), "variance": projection.get("variance"),
                         "basis": option.get("_probability_basis"), "home": fixture.get("home_team"),
                         "away": fixture.get("away_team"), "selected": bool(candidate.get("selector_selected")),
                         "market_prob": market_prob}
            out.append(seen[key])
    return out


def _label(selection, outcome):
    market, side, point = selection["market"], str(selection["side"]).casefold(), selection["point"]
    home_goals, away_goals = outcome.get("home_goals"), outcome.get("away_goals")
    if home_goals is None or away_goals is None:
        return None
    if market in TOTALS:
        value = home_goals + away_goals if market == "goals" else outcome.get(market)
        if value is None or point is None or side not in {"over", "under"}:
            return None
        return asian_outcome(value, point, over=side == "over")
    if market == "spreads":
        home, away = str(selection.get("home") or "").casefold(), str(selection.get("away") or "").casefold()
        if point is None or side not in {home, away} or home == away:
            return None
        margin = home_goals - away_goals if side == home else away_goals - home_goals
        return asian_outcome(margin, -float(point))
    if market == "btts":
        both = home_goals > 0 and away_goals > 0
        return "hit" if both == (side == "yes") else "miss" if side in {"yes", "no"} else None
    if market == "moneyline":
        winner = "home" if home_goals > away_goals else "away" if away_goals > home_goals else "draw"
        return ("hit" if side == winner else "miss") if side in {"home", "away", "draw"} else None
    return None


def grade(selection, outcome):
    """(win share, weight) like the scorecard, or None when the selection cannot be graded."""
    label = _label(selection, outcome)
    return None if label is None else outcome_weight(label)


def fixture_outcomes(db_path):
    query_goals = "SELECT id, api_football_id, home_goals, away_goals FROM fixtures WHERE status IN ('FT', 'AET', 'PEN')"
    query_stats = """SELECT fixture_id, SUM(corners), SUM(COALESCE(yellow_cards, 0) + COALESCE(red_cards, 0)),
                            COUNT(corners), COUNT(yellow_cards), COUNT(*)
                     FROM fixture_team_stats GROUP BY fixture_id"""
    with closing(sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=60)) as db:
        stats = {fid: (corners if n_corners == 2 else None, cards if n_cards == 2 else None)
                 for fid, corners, cards, n_corners, n_cards, n in db.execute(query_stats)}
        return {str(api): {"home_goals": hg, "away_goals": ag, "corners": stats.get(fid, (None, None))[0],
                           "cards": stats.get(fid, (None, None))[1]}
                for fid, api, hg, ag in db.execute(query_goals)}


def build(db_path):
    outcomes = fixture_outcomes(db_path)
    dataset = []
    for row in latest_read_rows(db_path):
        outcome = outcomes.get(str(row["fixture_api_id"]))
        if not outcome:
            continue
        for selection in selections(row):
            graded = grade(selection, outcome)
            if graded and graded[1] > 0:
                dataset.append({**selection, "y": graded[0], "w": graded[1]})
    return dataset
