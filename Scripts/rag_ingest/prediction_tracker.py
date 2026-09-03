#!/usr/bin/env python3
"""
Prediction Tracker — log, resolve, and report on model predictions.

Provides:
  - log_prediction()     : persist a single prediction to SQLite
  - log_from_render_output() : parse rendered text and log predictions found
  - log_direct()         : log from a structured dict
  - resolve_outcomes()   : grade unresolved predictions against actual results
  - get_track_record()   : compute hit-rate / ROI / streak statistics

Database: Index/predictions.db (SQLite)

CLI:
  python prediction_tracker.py --resolve
  python prediction_tracker.py --resolve-date 2026-03-14
  python prediction_tracker.py --stats
  python prediction_tracker.py --stats --stats-market goals --stats-confidence high
  python prediction_tracker.py --export predictions.csv
  python prediction_tracker.py --backfill-unit-bets
  python prediction_tracker.py --clv                        # CLV report
  python prediction_tracker.py --capture-clv                # capture closing odds for today
  python prediction_tracker.py --capture-clv --resolve-date 2026-03-14  # capture for specific date
"""

from __future__ import annotations

import argparse
import csv
import difflib
import hashlib
import json
import logging
import os
import re
import sqlite3
import sys
import traceback
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parents[1]
sys.path.insert(0, str(_SCRIPT_DIR))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DB_PATH = _PROJECT_ROOT / "Index" / "predictions.db"
OUTPUT_DIR = _PROJECT_ROOT / "Output"

# ---------------------------------------------------------------------------
# V2 platform dispatch (lazy import so SQLAlchemy isn't a hard dep).
# ---------------------------------------------------------------------------

_SCRIPTS_ROOT = _PROJECT_ROOT / "Scripts"
if str(_SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_ROOT))

from data_platform.outcomes import normalize_outcome
from data_platform.tracking_metrics import (
    build_calibration,
    build_daily_breakdown,
    build_track_record,
    flat_stake_roi,
    outcome_totals,
)


def _platform_on() -> bool:
    try:
        from data_platform.compat.predictions_shim import is_platform_enabled
    except Exception:
        return False
    try:
        return is_platform_enabled()
    except Exception:
        return False

LEAGUE_TO_FE_DIR = {
    "EPL": "Prem_feature_engineering",
    "LaLiga": "LaLiga_feature_engineering",
    "SerieA": "SeriaA_feature_engineering",
    "Bundesliga": "Bundesliga_feature_engineering",
    "Ligue1": "Ligue1_feature_engineering",
    "UCL": "UCL_feature_engineering",
    "UEL": "UEL_feature_engineering",
    "UECL": "UECL_feature_engineering",
}

LEAGUE_TO_API_ID: Dict[str, int] = {
    "EPL": 39,
    "LaLiga": 140,
    "SerieA": 135,
    "Bundesliga": 78,
    "Ligue1": 61,
    "UCL": 2,
    "UEL": 3,
    "UECL": 848,
}


def _api_football_season_for_prediction_date(prediction_date: str) -> int:
    """Resolve the API-Football season year for a stored ISO prediction date."""
    try:
        target = date.fromisoformat(prediction_date[:10])
    except (TypeError, ValueError):
        target = date.today()
    return target.year if target.month >= 7 else target.year - 1

ALL_MARKETS = {"goals", "corners", "cards", "sot", "btts", "moneyline", "spreads", "correct_score"}

log = logging.getLogger("prediction_tracker")

# ---------------------------------------------------------------------------
# Database setup
# ---------------------------------------------------------------------------

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    -- Fixture info
    fixture_id TEXT,
    home_team TEXT NOT NULL,
    away_team TEXT NOT NULL,
    league TEXT NOT NULL,
    kickoff TEXT,

    -- Prediction info
    market TEXT NOT NULL,
    pick TEXT NOT NULL,
    side TEXT,
    line REAL,
    odds REAL,
    bookmaker TEXT,

    -- Model data
    model_prob REAL,
    implied_prob REAL,
    value_edge REAL,
    confidence TEXT,
    projected_total REAL,

    -- Source
    source TEXT DEFAULT 'standalone',

    -- Outcome (filled by resolver)
    actual_result REAL,
    outcome TEXT,
    graded_at TEXT,

    -- Metadata
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    season TEXT,
    prediction_date TEXT
);
"""

_CREATE_INDEXES_SQL = [
    "CREATE INDEX IF NOT EXISTS idx_predictions_date ON predictions(prediction_date);",
    "CREATE INDEX IF NOT EXISTS idx_predictions_market ON predictions(market);",
    "CREATE INDEX IF NOT EXISTS idx_predictions_confidence ON predictions(confidence);",
    "CREATE INDEX IF NOT EXISTS idx_predictions_outcome ON predictions(outcome);",
    "CREATE INDEX IF NOT EXISTS idx_predictions_league ON predictions(league);",
    "CREATE INDEX IF NOT EXISTS idx_predictions_fixture ON predictions(fixture_id);",
]

_CREATE_UNIT_BET_SLIPS_SQL = """
CREATE TABLE IF NOT EXISTS unit_bet_slips (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    slip_key TEXT NOT NULL UNIQUE,
    prediction_date TEXT NOT NULL,
    source TEXT NOT NULL DEFAULT 'unit_bets',
    title TEXT,
    fixture_summary TEXT,
    legs_json TEXT NOT NULL,
    prediction_ids_json TEXT,
    combined_odds REAL NOT NULL,
    stake_units REAL NOT NULL DEFAULT 1.0,
    outcome TEXT,
    profit_units REAL,
    graded_at TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
"""

_CREATE_UNIT_BET_INDEXES_SQL = [
    "CREATE INDEX IF NOT EXISTS idx_unit_bet_slips_date ON unit_bet_slips(prediction_date);",
    "CREATE INDEX IF NOT EXISTS idx_unit_bet_slips_source ON unit_bet_slips(source);",
    "CREATE INDEX IF NOT EXISTS idx_unit_bet_slips_outcome ON unit_bet_slips(outcome);",
]


def _migrate_schema(conn: sqlite3.Connection) -> None:
    """Run idempotent schema migrations. Safe to call on every connection."""
    _CLV_COLUMNS = [
        ("closing_odds", "REAL"),
        ("closing_implied_prob", "REAL"),
        ("clv", "REAL"),
    ]
    for col_name, col_type in _CLV_COLUMNS:
        try:
            conn.execute(f"ALTER TABLE predictions ADD COLUMN {col_name} {col_type}")
        except sqlite3.OperationalError as exc:
            # "duplicate column name" is expected on subsequent runs
            if "duplicate column" in str(exc).lower():
                pass
            else:
                raise


def _get_db(db_path: Path = DB_PATH) -> sqlite3.Connection:
    """Return a connection to the predictions database, creating it if needed."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA busy_timeout=5000;")
    conn.execute(_CREATE_TABLE_SQL)
    for idx_sql in _CREATE_INDEXES_SQL:
        conn.execute(idx_sql)
    conn.execute(_CREATE_UNIT_BET_SLIPS_SQL)
    for idx_sql in _CREATE_UNIT_BET_INDEXES_SQL:
        conn.execute(idx_sql)
    _migrate_schema(conn)
    conn.commit()
    return conn


# ===================================================================
# A. Logging Functions
# ===================================================================

def log_prediction(
    home_team: str,
    away_team: str,
    league: str,
    market: str,
    pick: str,
    side: str,
    line: float = None,
    odds: float = None,
    bookmaker: str = None,
    model_prob: float = None,
    implied_prob: float = None,
    value_edge: float = None,
    confidence: str = None,
    projected_total: float = None,
    source: str = "standalone",
    fixture_id: str = None,
    kickoff: str = None,
    season: str = None,
    *,
    db_path: Path = DB_PATH,
) -> int:
    """Log a single prediction. Returns the prediction ID.

    This function is designed to be FAST and SAFE — it will never crash the
    calling code. Any error is caught, logged, and -1 is returned.

    When ``PLATFORM_ENABLED=1`` and ``db_path`` is the default, writes land
    in Postgres via ``data_platform``.
    """
    if db_path == DB_PATH and _platform_on():
        try:
            from data_platform.compat import platform_log_prediction
            return platform_log_prediction(
                home_team=home_team, away_team=away_team, league=league,
                market=market, pick=pick, side=side,
                line=line, odds=odds, bookmaker=bookmaker,
                model_prob=model_prob, implied_prob=implied_prob,
                value_edge=value_edge, confidence=confidence,
                projected_total=projected_total, source=source,
                fixture_id=fixture_id, kickoff=kickoff, season=season,
            )
        except Exception:
            log.exception("platform log_prediction failed; falling through to SQLite")
    try:
        conn = _get_db(db_path)
        prediction_date = date.today().isoformat()
        existing = conn.execute(
            """SELECT id
               FROM predictions
               WHERE prediction_date = ?
                 AND home_team = ?
                 AND away_team = ?
                 AND league = ?
                 AND market = ?
                 AND pick = ?
                 AND source = ?
               ORDER BY id DESC
               LIMIT 1""",
            (
                prediction_date,
                home_team,
                away_team,
                league,
                market,
                pick,
                source,
            ),
        ).fetchone()
        if existing:
            conn.close()
            return int(existing["id"])

        cur = conn.execute(
            """INSERT INTO predictions
               (fixture_id, home_team, away_team, league, kickoff,
                market, pick, side, line, odds, bookmaker,
                model_prob, implied_prob, value_edge, confidence,
                projected_total, source, season, prediction_date)
               VALUES (?,?,?,?,?, ?,?,?,?,?,?, ?,?,?,?, ?,?,?,?)""",
            (
                fixture_id, home_team, away_team, league, kickoff,
                market, pick, side, line, odds, bookmaker,
                model_prob, implied_prob, value_edge, confidence,
                projected_total, source, season,
                prediction_date,
            ),
        )
        conn.commit()
        pred_id = cur.lastrowid
        conn.close()
        return pred_id
    except Exception:
        log.error("Failed to log prediction: %s", traceback.format_exc())
        return -1


def log_direct(prediction_data: dict, *, db_path: Path = DB_PATH) -> int:
    """Log a prediction from a structured dict. Keys match log_prediction params."""
    return log_prediction(
        home_team=prediction_data.get("home_team", ""),
        away_team=prediction_data.get("away_team", ""),
        league=prediction_data.get("league", ""),
        market=prediction_data.get("market", ""),
        pick=prediction_data.get("pick", ""),
        side=prediction_data.get("side", ""),
        line=prediction_data.get("line"),
        odds=prediction_data.get("odds"),
        bookmaker=prediction_data.get("bookmaker"),
        model_prob=prediction_data.get("model_prob"),
        implied_prob=prediction_data.get("implied_prob"),
        value_edge=prediction_data.get("value_edge"),
        confidence=prediction_data.get("confidence"),
        projected_total=prediction_data.get("projected_total"),
        source=prediction_data.get("source", "standalone"),
        fixture_id=prediction_data.get("fixture_id"),
        kickoff=prediction_data.get("kickoff"),
        season=prediction_data.get("season"),
        db_path=db_path,
    )


def _unit_slip_key(prediction_date: str, source: str, title: str, legs: List[dict]) -> str:
    leg_keys = []
    for leg in legs or []:
        leg_keys.append({
            "event_id": str(leg.get("event_id") or leg.get("fixture_id") or ""),
            "fixture": str(leg.get("fixture") or ""),
            "market": str(leg.get("market") or leg.get("market_group") or leg.get("market_key") or ""),
            "pick": str(leg.get("pick") or leg.get("pick_display") or ""),
            "side": str(leg.get("side") or ""),
            "line": leg.get("line"),
        })
    payload = json.dumps(
        {
            "prediction_date": prediction_date,
            "source": source,
            "title": title,
            "legs": leg_keys,
        },
        sort_keys=True,
        default=str,
    )
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:20]
    return f"{source}:{prediction_date}:{digest}"


def _normalise_market_name(market: str) -> str:
    market = str(market or "").lower()
    return "goals" if market == "totals" else market


def _normalise_unit_outcome(outcome: Any) -> Optional[str]:
    value = str(outcome or "").strip().lower()
    if value in {"hit", "win", "won"}:
        return "hit"
    if value in {"miss", "loss", "lost", "lose"}:
        return "miss"
    if value in {"push", "void", "refund", "refunded"}:
        return "push"
    return None


def _unit_bet_profit(
    leg_outcomes: List[dict],
    combined_odds: float,
    stake_units: float,
) -> Tuple[Optional[str], Optional[float]]:
    outcomes = [_normalise_unit_outcome(x.get("outcome")) for x in leg_outcomes]
    if not outcomes or any(x is None for x in outcomes):
        return None, None
    stake = float(stake_units or 1.0)
    if any(x == "miss" for x in outcomes):
        return "miss", -stake
    if all(x == "push" for x in outcomes):
        return "push", 0.0
    if all(x == "hit" for x in outcomes):
        odds = float(combined_odds or 0.0)
        if odds <= 1.0:
            odds = 1.0
            for leg in leg_outcomes:
                odds *= max(float(leg.get("odds") or 1.0), 1.0)
        return "hit", stake * (odds - 1.0)

    # At least one hit and the rest are pushes: settled at reduced parlay odds.
    reduced_odds = 1.0
    for leg, outcome in zip(leg_outcomes, outcomes):
        if outcome == "hit":
            reduced_odds *= max(float(leg.get("odds") or 1.0), 1.0)
    if reduced_odds <= 1.0:
        return "push", 0.0
    return "hit", stake * (reduced_odds - 1.0)


def log_unit_bet_slip(
    *,
    title: str,
    combined_odds: float,
    legs: List[dict],
    prediction_ids: List[int] = None,
    prediction_date: str = None,
    source: str = "unit_bets",
    stake_units: float = 1.0,
    db_path: Path = DB_PATH,
) -> int:
    """Log one posted unit-betting slip.

    Unit-betting performance is slip-level: a two-leg double is one 1-unit bet,
    not two independent 1-unit singles.
    """
    prediction_date = prediction_date or date.today().isoformat()
    title = title or "Unit Bet"
    source = source or "unit_bets"
    legs = [dict(x or {}) for x in (legs or [])]
    if not legs:
        return -1

    try:
        odds = float(combined_odds or 0.0)
    except Exception:
        odds = 0.0
    if odds <= 1.0:
        odds = 1.0
        for leg in legs:
            try:
                odds *= max(float(leg.get("odds") or 1.0), 1.0)
            except Exception:
                pass

    try:
        stake = float(stake_units or 1.0)
    except Exception:
        stake = 1.0
    if stake <= 0:
        stake = 1.0

    clean_ids = []
    for pred_id in prediction_ids or []:
        try:
            pred_id_int = int(pred_id)
        except Exception:
            continue
        if pred_id_int > 0:
            clean_ids.append(pred_id_int)

    fixture_summary = "; ".join(dict.fromkeys(
        str(leg.get("fixture") or f"{leg.get('home_team', '')} vs {leg.get('away_team', '')}").strip()
        for leg in legs
        if leg
    ))[:500]
    slip_key = _unit_slip_key(prediction_date, source, title, legs)

    try:
        conn = _get_db(db_path)
        existing = conn.execute(
            "SELECT id FROM unit_bet_slips WHERE slip_key = ?",
            [slip_key],
        ).fetchone()
        if existing:
            conn.close()
            return int(existing["id"])

        cur = conn.execute(
            """INSERT INTO unit_bet_slips
               (slip_key, prediction_date, source, title, fixture_summary,
                legs_json, prediction_ids_json, combined_odds, stake_units)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                slip_key,
                prediction_date,
                source,
                title,
                fixture_summary,
                json.dumps(legs, sort_keys=True, default=str),
                json.dumps(clean_ids),
                odds,
                stake,
            ),
        )
        conn.commit()
        slip_id = cur.lastrowid
        conn.close()
        return int(slip_id)
    except Exception:
        log.error("Failed to log unit bet slip: %s", traceback.format_exc())
        return -1


def _load_slip_legs(row: sqlite3.Row | dict) -> List[dict]:
    try:
        return [dict(x or {}) for x in json.loads((row["legs_json"] if isinstance(row, sqlite3.Row) else row.get("legs_json")) or "[]")]
    except Exception:
        return []


def _load_slip_prediction_ids(row: sqlite3.Row | dict) -> List[int]:
    try:
        raw = row["prediction_ids_json"] if isinstance(row, sqlite3.Row) else row.get("prediction_ids_json")
        return [int(x) for x in json.loads(raw or "[]") if int(x) > 0]
    except Exception:
        return []


def _unit_leg_outcomes_from_predictions(
    conn: sqlite3.Connection,
    prediction_ids: List[int],
    legs: List[dict],
) -> Optional[List[dict]]:
    if not prediction_ids:
        return None
    if len(prediction_ids) != len(legs):
        return None
    placeholders = ",".join("?" for _ in prediction_ids)
    rows = conn.execute(
        f"SELECT * FROM predictions WHERE id IN ({placeholders})",
        prediction_ids,
    ).fetchall()
    by_id = {int(row["id"]): dict(row) for row in rows}
    if len(by_id) < len(prediction_ids):
        return None

    outcomes = []
    for idx, pred_id in enumerate(prediction_ids):
        pred = by_id.get(int(pred_id))
        outcome = _normalise_unit_outcome(pred.get("outcome") if pred else None)
        if outcome is None:
            return None
        leg = legs[idx] if idx < len(legs) else {}
        outcomes.append({
            "outcome": outcome,
            "odds": pred.get("odds") or leg.get("odds"),
            "actual": pred.get("actual_result"),
            "pick": pred.get("pick") or leg.get("pick"),
            "fixture": leg.get("fixture") or f"{pred.get('home_team')} vs {pred.get('away_team')}",
        })
    return outcomes


def _unit_leg_outcomes_from_results(
    prediction_date: str,
    legs: List[dict],
    results_cache: Dict[Tuple[str, str], List[dict]],
) -> Optional[List[dict]]:
    outcomes = []
    for leg in legs:
        league = str(leg.get("league") or "").strip()
        if not league:
            return None
        cache_key = (prediction_date, league)
        if cache_key not in results_cache:
            results = _fetch_results_from_data(prediction_date, league)
            if not results:
                results = _fetch_results_from_api(prediction_date, league)
            results_cache[cache_key] = results or []
        pred = {
            "fixture_id": leg.get("fixture_id") or leg.get("event_id"),
            "home_team": leg.get("home_team"),
            "away_team": leg.get("away_team"),
            "league": league,
            "market": _normalise_market_name(leg.get("market") or leg.get("market_group")),
            "pick": leg.get("pick") or leg.get("pick_display"),
            "side": str(leg.get("side") or "").lower(),
            "line": _safe_float(leg.get("line")),
        }
        result = _match_prediction_to_result(pred, results_cache[cache_key])
        if result is None:
            return None
        outcome, actual = _grade_prediction(pred, result)
        outcome = _normalise_unit_outcome(outcome)
        if outcome is None:
            return None
        outcomes.append({
            "outcome": outcome,
            "odds": _safe_float(leg.get("odds")),
            "actual": actual,
            "pick": pred["pick"],
            "fixture": leg.get("fixture"),
        })
    return outcomes


def _unit_leg_from_prediction_row(row: dict) -> dict:
    market = _normalise_market_name(row.get("market"))
    return {
        "event_id": row.get("fixture_id"),
        "fixture_id": row.get("fixture_id"),
        "league": row.get("league"),
        "fixture": f"{row.get('home_team', '')} vs {row.get('away_team', '')}",
        "home_team": row.get("home_team"),
        "away_team": row.get("away_team"),
        "market": market,
        "market_group": "totals" if market == "goals" else market,
        "market_key": row.get("market"),
        "pick": row.get("pick"),
        "side": row.get("side"),
        "line": row.get("line"),
        "odds": row.get("odds"),
        "bookmaker": row.get("bookmaker"),
        "model_prob": row.get("model_prob"),
        "implied_prob": row.get("implied_prob"),
        "value_edge": row.get("value_edge"),
        "confidence": row.get("confidence"),
        "projected_total": row.get("projected_total"),
    }


def _combined_odds_for_rows(rows: List[dict]) -> float:
    odds = 1.0
    for row in rows:
        try:
            odds *= max(float(row.get("odds") or 1.0), 1.0)
        except Exception:
            pass
    return odds


def _odds_range_penalty(odds: float, low: float = 1.80, high: float = 2.40) -> float:
    if low <= odds <= high:
        return 0.0
    if odds < low:
        return (low - odds) * 10.0 + 2.0
    return (odds - high) * 10.0 + 2.0


def _infer_legacy_unit_slip_groups(rows: List[dict]) -> List[Tuple[str, List[dict]]]:
    """Infer old #unit-betting slip boundaries from leg order and odds.

    Before slip tracking existed, the channel logged only individual legs.
    The posting code emitted one Single and/or one Double in order.  We infer
    that shape by preferring chunks whose combined odds land in 1.80-2.40.
    """
    rows = [dict(row) for row in rows or []]
    if not rows:
        return []

    n = len(rows)
    best: Dict[int, Tuple[float, List[Tuple[str, List[dict]]]]] = {0: (0.0, [])}
    for i in range(n):
        if i not in best:
            continue
        prev_score, prev_groups = best[i]
        for size, title in ((1, "Single"), (2, "Double")):
            if i + size > n:
                continue
            chunk = rows[i:i + size]
            combined = _combined_odds_for_rows(chunk)
            score = prev_score + _odds_range_penalty(combined)
            if size == 2:
                # Slightly prefer a plausible double over two weak singles.
                score -= 0.05
            candidate = (score, prev_groups + [(title, chunk)])
            current = best.get(i + size)
            if current is None or candidate[0] < current[0]:
                best[i + size] = candidate

    return best.get(n, (0.0, [("Single", [row]) for row in rows]))[1]


def _settle_backfilled_unit_slip(
    slip_id: int,
    rows: List[dict],
    *,
    combined_odds: float,
    db_path: Path,
) -> bool:
    leg_outcomes = []
    for row in rows:
        outcome = _normalise_unit_outcome(row.get("outcome"))
        if outcome is None:
            return False
        leg_outcomes.append({
            "outcome": outcome,
            "odds": row.get("odds"),
            "actual": row.get("actual_result"),
            "pick": row.get("pick"),
            "fixture": f"{row.get('home_team', '')} vs {row.get('away_team', '')}",
        })
    outcome, profit = _unit_bet_profit(leg_outcomes, combined_odds, 1.0)
    if outcome is None or profit is None:
        return False
    try:
        conn = _get_db(db_path)
        conn.execute(
            """UPDATE unit_bet_slips
               SET outcome = ?, profit_units = ?, graded_at = ?
               WHERE id = ? AND outcome IS NULL""",
            (outcome, float(profit), datetime.now().isoformat(), slip_id),
        )
        conn.commit()
        conn.close()
        return True
    except Exception:
        log.error("Failed to settle backfilled unit bet slip: %s", traceback.format_exc())
        return False


def backfill_unit_bet_slips_from_predictions(
    prediction_date: str = None,
    source: str = "unit_bets",
    *,
    db_path: Path = DB_PATH,
) -> dict:
    """Create historical unit-bet slips from old leg-level unit predictions.

    This is idempotent by date: if a date already has any slip rows for the
    source, it is left alone.  Old rows did not store slip IDs, so boundaries
    are inferred from posting order and the 1.80-2.40 unit-bet odds band.
    """
    stats = {
        "dates_seen": 0,
        "dates_backfilled": 0,
        "dates_skipped_existing": 0,
        "source_predictions": 0,
        "slips_created": 0,
        "slips_settled": 0,
        "errors": 0,
    }

    try:
        conn = _get_db(db_path)
        params: list = [source]
        query = "SELECT * FROM predictions WHERE source = ?"
        if prediction_date:
            query += " AND prediction_date = ?"
            params.append(prediction_date)
        query += " ORDER BY prediction_date, id"
        rows = [dict(row) for row in conn.execute(query, params).fetchall()]

        existing_params: list = [source]
        existing_query = "SELECT DISTINCT prediction_date FROM unit_bet_slips WHERE source = ?"
        if prediction_date:
            existing_query += " AND prediction_date = ?"
            existing_params.append(prediction_date)
        existing_dates = {
            str(row["prediction_date"])
            for row in conn.execute(existing_query, existing_params).fetchall()
            if row["prediction_date"]
        }
        conn.close()
    except Exception:
        log.error("Failed to load historical unit bet predictions: %s", traceback.format_exc())
        stats["errors"] = 1
        return stats

    by_date: Dict[str, List[dict]] = {}
    for row in rows:
        day = row.get("prediction_date") or str(row.get("created_at") or "")[:10]
        if not day:
            continue
        by_date.setdefault(day, []).append(row)
    stats["dates_seen"] = len(by_date)
    stats["source_predictions"] = len(rows)

    for day, day_rows in sorted(by_date.items()):
        if day in existing_dates:
            stats["dates_skipped_existing"] += 1
            continue
        groups = _infer_legacy_unit_slip_groups(day_rows)
        if not groups:
            continue
        for title, group_rows in groups:
            legs = [_unit_leg_from_prediction_row(row) for row in group_rows]
            prediction_ids = [int(row["id"]) for row in group_rows if row.get("id")]
            combined_odds = _combined_odds_for_rows(group_rows)
            slip_id = log_unit_bet_slip(
                title=title,
                combined_odds=combined_odds,
                legs=legs,
                prediction_ids=prediction_ids,
                prediction_date=day,
                source=source,
                stake_units=1.0,
                db_path=db_path,
            )
            if slip_id > 0:
                stats["slips_created"] += 1
                if _settle_backfilled_unit_slip(
                    slip_id,
                    group_rows,
                    combined_odds=combined_odds,
                    db_path=db_path,
                ):
                    stats["slips_settled"] += 1
            else:
                stats["errors"] += 1
        stats["dates_backfilled"] += 1

    return stats


def resolve_unit_bet_slips(
    prediction_date: str = None,
    source: str = "unit_bets",
    dry_run: bool = False,
    *,
    db_path: Path = DB_PATH,
) -> dict:
    """Resolve posted unit-betting slips and settle flat-stake unit profit."""
    if prediction_date is None:
        prediction_date = (date.today() - timedelta(days=1)).isoformat()

    stats = {
        "graded": 0,
        "hit": 0,
        "miss": 0,
        "push": 0,
        "pending": 0,
        "errors": 0,
        "profit_units": 0.0,
        "details": [],
    }

    try:
        conn = _get_db(db_path)
    except Exception:
        log.error("Cannot open database: %s", traceback.format_exc())
        stats["errors"] = 1
        return stats

    params: list = [prediction_date]
    query = "SELECT * FROM unit_bet_slips WHERE prediction_date = ? AND outcome IS NULL"
    if source:
        query += " AND source = ?"
        params.append(source)
    query += " ORDER BY id"

    try:
        rows = conn.execute(query, params).fetchall()
    except Exception:
        log.error("Failed to query unit bet slips: %s", traceback.format_exc())
        conn.close()
        stats["errors"] = 1
        return stats

    results_cache: Dict[Tuple[str, str], List[dict]] = {}
    now_iso = datetime.now().isoformat()
    for row in rows:
        legs = _load_slip_legs(row)
        prediction_ids = _load_slip_prediction_ids(row)
        try:
            leg_outcomes = _unit_leg_outcomes_from_predictions(conn, prediction_ids, legs)
            if leg_outcomes is None:
                leg_outcomes = _unit_leg_outcomes_from_results(prediction_date, legs, results_cache)
            if leg_outcomes is None:
                stats["pending"] += 1
                stats["details"].append({
                    "id": row["id"],
                    "title": row["title"],
                    "status": "pending",
                    "fixture": row["fixture_summary"],
                })
                continue

            outcome, profit = _unit_bet_profit(
                leg_outcomes,
                combined_odds=float(row["combined_odds"] or 0.0),
                stake_units=float(row["stake_units"] or 1.0),
            )
            if outcome is None or profit is None:
                stats["pending"] += 1
                continue

            stats["graded"] += 1
            stats[outcome] = stats.get(outcome, 0) + 1
            stats["profit_units"] += float(profit)
            stats["details"].append({
                "id": row["id"],
                "title": row["title"],
                "outcome": outcome,
                "profit_units": float(profit),
                "fixture": row["fixture_summary"],
                "legs": leg_outcomes,
            })

            if not dry_run:
                conn.execute(
                    """UPDATE unit_bet_slips
                       SET outcome = ?, profit_units = ?, graded_at = ?
                       WHERE id = ?""",
                    (outcome, float(profit), now_iso, row["id"]),
                )
        except Exception:
            log.error("Failed to resolve unit bet slip %s: %s", row["id"], traceback.format_exc())
            stats["errors"] += 1

    if not dry_run:
        try:
            conn.commit()
        except Exception:
            log.error("Failed to commit unit bet slip grading: %s", traceback.format_exc())
            stats["errors"] += 1
    conn.close()
    return stats


def get_unit_bet_track_record(
    days: int = None,
    source: str = "unit_bets",
    *,
    db_path: Path = DB_PATH,
) -> dict:
    """Return all-time unit-betting record at posted-slip level."""
    try:
        conn = _get_db(db_path)
    except Exception:
        log.error("Cannot open database: %s", traceback.format_exc())
        return {"error": "database_unavailable"}

    where = ["outcome IS NOT NULL"]
    params: list = []
    if source:
        where.append("source = ?")
        params.append(source)
    if days is not None:
        cutoff = (date.today() - timedelta(days=days)).isoformat()
        where.append("prediction_date >= ?")
        params.append(cutoff)

    try:
        rows = conn.execute(
            f"SELECT * FROM unit_bet_slips WHERE {' AND '.join(where)} ORDER BY prediction_date, id",
            params,
        ).fetchall()
    except Exception:
        log.error("Failed to query unit bet track record: %s", traceback.format_exc())
        conn.close()
        return {"error": "query_failed"}
    conn.close()

    slips = [dict(row) for row in rows]
    wins = sum(1 for row in slips if row.get("outcome") == "hit")
    losses = sum(1 for row in slips if row.get("outcome") == "miss")
    pushes = sum(1 for row in slips if row.get("outcome") == "push")
    staked = sum(float(row.get("stake_units") or 0.0) for row in slips)
    profit = sum(float(row.get("profit_units") or 0.0) for row in slips)
    decided = wins + losses

    by_date: Dict[str, dict] = {}
    for row in slips:
        day = row.get("prediction_date") or ""
        bucket = by_date.setdefault(day, {"date": day, "wins": 0, "losses": 0, "pushes": 0, "profit_units": 0.0, "total": 0})
        bucket["total"] += 1
        if row.get("outcome") == "hit":
            bucket["wins"] += 1
        elif row.get("outcome") == "miss":
            bucket["losses"] += 1
        elif row.get("outcome") == "push":
            bucket["pushes"] += 1
        bucket["profit_units"] += float(row.get("profit_units") or 0.0)

    return {
        "total_graded": len(slips),
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "hit_rate": wins / decided if decided > 0 else 0.0,
        "stake_units": staked,
        "units_profit": profit,
        "roi": profit / staked if staked > 0 else 0.0,
        "daily_performance": [by_date[key] for key in sorted(by_date.keys())[-30:]],
    }


def get_unit_bet_daily_summary(
    prediction_date: str = None,
    source: str = "unit_bets",
    *,
    db_path: Path = DB_PATH,
) -> dict:
    """Return a date-level summary for posted unit-betting slips."""
    if prediction_date is None:
        prediction_date = (date.today() - timedelta(days=1)).isoformat()

    try:
        conn = _get_db(db_path)
    except Exception:
        return {"date": prediction_date, "total": 0}

    params: list = [prediction_date]
    query = "SELECT * FROM unit_bet_slips WHERE prediction_date = ?"
    if source:
        query += " AND source = ?"
        params.append(source)
    query += " ORDER BY id"

    try:
        rows = conn.execute(query, params).fetchall()
    except Exception:
        conn.close()
        return {"date": prediction_date, "total": 0}
    conn.close()

    slips = []
    wins = losses = pushes = pending = 0
    profit = 0.0
    stake = 0.0
    for row in rows:
        outcome = row["outcome"]
        if outcome == "hit":
            wins += 1
        elif outcome == "miss":
            losses += 1
        elif outcome == "push":
            pushes += 1
        else:
            pending += 1
        if outcome is not None:
            profit += float(row["profit_units"] or 0.0)
            stake += float(row["stake_units"] or 0.0)
        slips.append({
            "id": row["id"],
            "title": row["title"] or "Unit Bet",
            "fixture_summary": row["fixture_summary"],
            "combined_odds": float(row["combined_odds"] or 0.0),
            "stake_units": float(row["stake_units"] or 0.0),
            "outcome": outcome or "pending",
            "profit_units": float(row["profit_units"] or 0.0) if outcome is not None else None,
            "legs": _load_slip_legs(row),
        })

    decided = wins + losses
    return {
        "date": prediction_date,
        "total": len(rows),
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "pending": pending,
        "hit_rate": wins / decided if decided > 0 else 0.0,
        "stake_units": stake,
        "units_profit": profit,
        "roi": profit / stake if stake > 0 else 0.0,
        "slips": slips,
    }


# ===================================================================
# B. Render-output parser (lazy integration)
# ===================================================================

# Regex patterns for extracting predictions from rendered text.
# These match the output format of render_totals_line_answer, render_btts_answer, etc.

_RE_FIXTURE = re.compile(
    r"(?:^-\s*|^=+\s*)(.+?)\s+vs\s+(.+?)(?:\s*=+)?(?:\s*:|\s*$)",
    re.MULTILINE,
)

_RE_RECOMMENDED_TOTAL = re.compile(
    r"Recommended:\s*(Over|Under)\s+([\d.]+)\s*@\s*([\d.]+)\s*\(([^,]+),\s*([^)]+)\)",
    re.IGNORECASE,
)

_RE_RECOMMENDED_MODEL_ONLY_TOTAL = re.compile(
    r"Recommended:\s*(Over|Under)\s+([\d.]+)\s*\(model-only",
    re.IGNORECASE,
)

_RE_RECOMMENDED_BTTS = re.compile(
    r"Recommended:\s*BTTS\s+(Yes|No)\s*@\s*([\d.]+)\s*\(([^,]+),\s*([^)]+)\)",
    re.IGNORECASE,
)

_RE_RECOMMENDED_BTTS_MODEL = re.compile(
    r"Recommended:\s*BTTS\s+(Yes|No)\s*(?:—|--)\s*model gives\s+([\d.]+)%",
    re.IGNORECASE,
)

_RE_RECOMMENDED_MONEYLINE = re.compile(
    r">>>\s*Recommended:\s*(.+?)\s*@\s*([\d.]+)\s*\((\w+)\s+confidence\)",
    re.IGNORECASE,
)

_RE_RECOMMENDED_MONEYLINE_MODEL = re.compile(
    r">>>\s*Recommended:\s*(.+?)\s*\(([\d.]+)%\)\s*(?:—|--)\s*no moneyline",
    re.IGNORECASE,
)

_RE_RECOMMENDED_SPREAD = re.compile(
    r"Recommended:\s*(.+?)\s+([+-][\d.]+)\s*@\s*([\d.]+)\s*\(([^,]+),\s*([^)]+)\)",
    re.IGNORECASE,
)

_RE_RECOMMENDED_SPREAD_MODEL = re.compile(
    r"Recommended:\s*(.+?)\s+([+-][\d.]+)\s*\(model-only",
    re.IGNORECASE,
)

_RE_MODEL_LINE = re.compile(
    r"Model:\s*P\((.+?)\)\s*=\s*([\d.]+)%\s*\|\s*Fair implied:\s*([\d.]+)%\s*\|\s*Value edge:\s*([+-]?[\d.]+)%",
    re.IGNORECASE,
)

_RE_PROJECTED_TOTAL = re.compile(
    r"projected total\s+([\d]+\.[\d]+)",
    re.IGNORECASE,
)

_RE_CONFIDENCE = re.compile(
    r"\b(high|medium|low)\s+confidence\b",
    re.IGNORECASE,
)

_RE_EV_LINE = re.compile(
    r"Expected value:\s*([+-]?[\d.]+)\s*per unit staked\s*\((\w+)\s+confidence\)",
    re.IGNORECASE,
)

# Detect which market type is active from the header lines.
_HEADER_TO_MARKET = {
    "goals totals": "goals",
    "corner totals": "corners",
    "card totals": "cards",
    "sot totals": "sot",
    "both teams to score": "btts",
    "moneyline": "moneyline",
    "score handicap": "spreads",
    "correct score": "correct_score",
    "goal interval": "goals",
}


def _detect_market_from_header(text: str) -> str:
    """Detect the market type from the header line of rendered output."""
    first_line = text.strip().split("\n")[0].lower() if text.strip() else ""
    for keyword, market in _HEADER_TO_MARKET.items():
        if keyword in first_line:
            return market
    return "goals"


def _detect_stat_group_from_context(text_block: str) -> Optional[str]:
    """Detect stat group from surrounding context lines."""
    lower = text_block.lower()
    if "corner" in lower:
        return "corners"
    if "card" in lower:
        return "cards"
    if "sot" in lower or "shots on target" in lower:
        return "sot"
    if "goal" in lower:
        return "goals"
    return None


def log_from_render_output(
    text: str,
    league: str,
    events: list = None,
    source: str = "standalone",
    *,
    db_path: Path = DB_PATH,
) -> int:
    """Parse a rendered output string and log all predictions found in it.

    This is the lazy integration path -- parse the existing text output
    rather than modifying every render function.

    Returns number of predictions logged.
    """
    if not text or not text.strip():
        return 0

    count = 0
    market = _detect_market_from_header(text)

    # Split text into per-fixture blocks.
    # Fixtures appear as "- Home vs Away:" or "=== Home vs Away ==="
    fixture_blocks = re.split(
        r"(?=^[-=]+\s*.+?\s+vs\s+.+?(?:\s*=+)?(?:\s*:|\s*$))",
        text,
        flags=re.MULTILINE,
    )

    # If no fixture blocks found, try the entire text as one block
    if len(fixture_blocks) <= 1:
        fixture_blocks = [text]

    current_home = None
    current_away = None

    for block in fixture_blocks:
        # Try to extract fixture teams from this block
        fm = _RE_FIXTURE.search(block)
        if fm:
            current_home = fm.group(1).strip().rstrip(":")
            current_away = fm.group(2).strip().rstrip(":")
            # Clean up any leading "- " or "=== " artifacts
            current_home = re.sub(r"^[=\-\s]+", "", current_home).strip()
            current_away = re.sub(r"[=\s]+$", "", current_away).strip()

        if not current_home or not current_away:
            continue

        # Detect local market override from block context (for team totals)
        local_market = _detect_stat_group_from_context(block) or market

        # Extract projected total
        pt_match = _RE_PROJECTED_TOTAL.search(block)
        projected_total = float(pt_match.group(1)) if pt_match else None

        # Extract model probability line
        model_prob = None
        implied_prob_val = None
        value_edge_val = None
        ml_match = _RE_MODEL_LINE.search(block)
        if ml_match:
            model_prob = float(ml_match.group(2)) / 100.0
            implied_prob_val = float(ml_match.group(3)) / 100.0
            value_edge_val = float(ml_match.group(4)) / 100.0

        # Extract confidence
        conf_match = _RE_CONFIDENCE.search(block)
        confidence = conf_match.group(1).lower() if conf_match else None

        # --- Totals (Over/Under) ---
        tot_match = _RE_RECOMMENDED_TOTAL.search(block)
        if tot_match:
            side = tot_match.group(1).lower()
            line = float(tot_match.group(2))
            odds = float(tot_match.group(3))
            bookmaker = tot_match.group(4).strip()
            pick = f"{side.title()} {line:g}"
            log_prediction(
                home_team=current_home, away_team=current_away, league=league,
                market=local_market, pick=pick, side=side, line=line,
                odds=odds, bookmaker=bookmaker, model_prob=model_prob,
                implied_prob=implied_prob_val, value_edge=value_edge_val,
                confidence=confidence, projected_total=projected_total,
                source=source, db_path=db_path,
            )
            count += 1
            continue

        # Model-only totals
        tot_mo = _RE_RECOMMENDED_MODEL_ONLY_TOTAL.search(block)
        if tot_mo:
            side = tot_mo.group(1).lower()
            line = float(tot_mo.group(2))
            pick = f"{side.title()} {line:g}"
            log_prediction(
                home_team=current_home, away_team=current_away, league=league,
                market=local_market, pick=pick, side=side, line=line,
                model_prob=model_prob, implied_prob=implied_prob_val,
                value_edge=value_edge_val, confidence=confidence,
                projected_total=projected_total, source=source, db_path=db_path,
            )
            count += 1
            continue

        # --- BTTS ---
        btts_match = _RE_RECOMMENDED_BTTS.search(block)
        if btts_match:
            side = btts_match.group(1).lower()
            odds = float(btts_match.group(2))
            bookmaker = btts_match.group(3).strip()
            pick = f"BTTS {side.title()}"
            log_prediction(
                home_team=current_home, away_team=current_away, league=league,
                market="btts", pick=pick, side=side, odds=odds,
                bookmaker=bookmaker, model_prob=model_prob,
                implied_prob=implied_prob_val, value_edge=value_edge_val,
                confidence=confidence, projected_total=projected_total,
                source=source, db_path=db_path,
            )
            count += 1
            continue

        # BTTS model-only
        btts_mo = _RE_RECOMMENDED_BTTS_MODEL.search(block)
        if btts_mo:
            side = btts_mo.group(1).lower()
            mp = float(btts_mo.group(2)) / 100.0
            pick = f"BTTS {side.title()}"
            log_prediction(
                home_team=current_home, away_team=current_away, league=league,
                market="btts", pick=pick, side=side, model_prob=mp,
                confidence=confidence, projected_total=projected_total,
                source=source, db_path=db_path,
            )
            count += 1
            continue

        # --- Moneyline ---
        ml_rec = _RE_RECOMMENDED_MONEYLINE.search(block)
        if ml_rec:
            team = ml_rec.group(1).strip()
            odds = float(ml_rec.group(2))
            conf = ml_rec.group(3).lower()
            # Determine side
            if _names_match(team, current_home):
                side = "home"
            elif _names_match(team, current_away):
                side = "away"
            elif team.lower() == "draw":
                side = "draw"
            else:
                side = "home"
            pick = f"{team} Win" if side != "draw" else "Draw"
            log_prediction(
                home_team=current_home, away_team=current_away, league=league,
                market="moneyline", pick=pick, side=side, odds=odds,
                model_prob=model_prob, implied_prob=implied_prob_val,
                value_edge=value_edge_val, confidence=conf,
                projected_total=projected_total, source=source, db_path=db_path,
            )
            count += 1
            continue

        # Moneyline model-only
        ml_mo = _RE_RECOMMENDED_MONEYLINE_MODEL.search(block)
        if ml_mo:
            team = ml_mo.group(1).strip()
            mp = float(ml_mo.group(2)) / 100.0
            if _names_match(team, current_home):
                side = "home"
            elif _names_match(team, current_away):
                side = "away"
            elif team.lower() == "draw":
                side = "draw"
            else:
                side = "home"
            pick = f"{team} Win" if side != "draw" else "Draw"
            log_prediction(
                home_team=current_home, away_team=current_away, league=league,
                market="moneyline", pick=pick, side=side, model_prob=mp,
                confidence=confidence, source=source, db_path=db_path,
            )
            count += 1
            continue

        # --- Spreads ---
        sp_match = _RE_RECOMMENDED_SPREAD.search(block)
        if sp_match:
            team = sp_match.group(1).strip()
            handicap = float(sp_match.group(2))
            odds = float(sp_match.group(3))
            bookmaker = sp_match.group(4).strip()
            if _names_match(team, current_home):
                side = "home"
            else:
                side = "away"
            pick = f"{team} {handicap:+g}"
            log_prediction(
                home_team=current_home, away_team=current_away, league=league,
                market="spreads", pick=pick, side=side, line=handicap,
                odds=odds, bookmaker=bookmaker, model_prob=model_prob,
                implied_prob=implied_prob_val, value_edge=value_edge_val,
                confidence=confidence, projected_total=projected_total,
                source=source, db_path=db_path,
            )
            count += 1
            continue

        # Spreads model-only
        sp_mo = _RE_RECOMMENDED_SPREAD_MODEL.search(block)
        if sp_mo:
            team = sp_mo.group(1).strip()
            handicap = float(sp_mo.group(2))
            if _names_match(team, current_home):
                side = "home"
            else:
                side = "away"
            pick = f"{team} {handicap:+g}"
            log_prediction(
                home_team=current_home, away_team=current_away, league=league,
                market="spreads", pick=pick, side=side, line=handicap,
                model_prob=model_prob, confidence=confidence,
                projected_total=projected_total, source=source, db_path=db_path,
            )
            count += 1
            continue

    return count


# ===================================================================
# C. Outcome Resolution
# ===================================================================

def _normalize(name: str) -> str:
    """Normalize a team name for matching."""
    return re.sub(r"[^a-z0-9\s]", " ", (name or "").lower()).strip()


def _names_match(a: str, b: str) -> bool:
    """Check if two team names are a fuzzy match."""
    na, nb = _normalize(a), _normalize(b)
    if not na or not nb:
        return False
    if na == nb:
        return True
    # One contained in the other
    if na in nb or nb in na:
        return True
    # difflib similarity
    ratio = difflib.SequenceMatcher(None, na, nb).ratio()
    return ratio >= 0.75


def _fetch_results_from_data(
    prediction_date: str,
    league: str = None,
) -> List[dict]:
    """Load actual fixture results from local feature engineering data.

    Searches Output/*_feature_engineering/team_engineered_features_*.json
    for fixtures matching the prediction date.

    Returns list of dicts with standardized keys.
    """
    results: List[dict] = []
    seen_fixtures = set()

    leagues_to_search = [league] if league else list(LEAGUE_TO_FE_DIR.keys())

    for lg in leagues_to_search:
        fe_dir_name = LEAGUE_TO_FE_DIR.get(lg)
        if not fe_dir_name:
            continue
        fe_dir = OUTPUT_DIR / fe_dir_name
        if not fe_dir.is_dir():
            continue

        # Glob all team_engineered_features files (multiple seasons)
        for fp in sorted(fe_dir.glob("team_engineered_features_*.json")):
            try:
                with open(fp, "r") as f:
                    data = json.load(f)
            except Exception:
                continue

            if not isinstance(data, list):
                continue

            for row in data:
                fix_date = str(row.get("fixture_date", ""))
                # Match by date (YYYY-MM-DD)
                if not fix_date.startswith(prediction_date):
                    continue

                home = str(row.get("home_team", ""))
                away = str(row.get("away_team", ""))
                team = str(row.get("team", ""))
                fix_id = str(row.get("fixture_id", ""))

                # Each fixture appears twice in team_engineered_features (once per team).
                # Only process the home team's row to avoid duplicates.
                if not _names_match(team, home):
                    continue

                fixture_key = f"{_normalize(home)}|{_normalize(away)}|{fix_date}"
                if fixture_key in seen_fixtures:
                    continue
                seen_fixtures.add(fixture_key)

                # We need to find the away team's row for their stats
                away_row = None
                for r2 in data:
                    if (str(r2.get("fixture_id", "")) == fix_id
                            and _names_match(str(r2.get("team", "")), away)):
                        away_row = r2
                        break

                goals_home = row.get("goals") or row.get("home_goals")
                goals_away = row.get("away_goals")
                if goals_away is None and away_row:
                    goals_away = away_row.get("goals")

                corners_home = row.get("corners")
                corners_away = away_row.get("corners") if away_row else None

                cards_home = row.get("cards_total")
                cards_away = away_row.get("cards_total") if away_row else None

                sot_home = row.get("shots_on")
                sot_away = away_row.get("shots_on") if away_row else None

                result = {
                    "home_team": home,
                    "away_team": away,
                    "fixture_id": fix_id,
                    "fixture_date": fix_date,
                    "league": lg,
                    "goals_home": _safe_float(goals_home),
                    "goals_away": _safe_float(goals_away),
                    "corners_home": _safe_float(corners_home),
                    "corners_away": _safe_float(corners_away),
                    "cards_home": _safe_float(cards_home),
                    "cards_away": _safe_float(cards_away),
                    "sot_home": _safe_float(sot_home),
                    "sot_away": _safe_float(sot_away),
                    "final_score": str(row.get("final_score", "")),
                }
                results.append(result)

    return results


def _fetch_results_from_api(
    prediction_date: str,
    league: str = None,
) -> List[dict]:
    """Fetch actual fixture results from API-Football when local data is unavailable.

    Calls the matching API-Football season for each prediction date, then
    fetches per-fixture statistics for corners, cards, and SoT.
    Then fetches per-fixture statistics for corners, cards, SoT.
    """
    try:
        import requests
    except ImportError:
        return []

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    api_key = os.environ.get("API-FOOTBALL-KEY") or os.environ.get("API_FOOTBALL_KEY")
    if not api_key:
        log.warning("No API-Football key for result resolution")
        return []

    results: List[dict] = []
    leagues_to_search = [league] if league else list(LEAGUE_TO_API_ID.keys())

    for lg in leagues_to_search:
        api_id = LEAGUE_TO_API_ID.get(lg)
        if not api_id:
            continue

        try:
            resp = requests.get(
                "https://v3.football.api-sports.io/fixtures",
                headers={"x-apisports-key": api_key},
                params={
                    "league": api_id,
                    "season": _api_football_season_for_prediction_date(prediction_date),
                    "date": prediction_date,
                    "status": "FT",
                },
                timeout=15,
            )
            resp.raise_for_status()
            fixtures = resp.json().get("response", [])
        except Exception as exc:
            log.warning("API-Football fixture fetch failed for %s: %s", lg, exc)
            continue

        for fx in fixtures:
            teams = fx.get("teams", {})
            goals = fx.get("goals", {})
            fixture_info = fx.get("fixture", {})
            fixture_id = str(fixture_info.get("id", ""))

            home = teams.get("home", {}).get("name", "")
            away = teams.get("away", {}).get("name", "")
            goals_home = goals.get("home")
            goals_away = goals.get("away")

            # Fetch fixture statistics for corners, cards, SoT
            corners_home = corners_away = cards_home = cards_away = sot_home = sot_away = None
            if fixture_id:
                try:
                    stat_resp = requests.get(
                        "https://v3.football.api-sports.io/fixtures/statistics",
                        headers={"x-apisports-key": api_key},
                        params={"fixture": fixture_id},
                        timeout=15,
                    )
                    stat_data = stat_resp.json().get("response", [])
                    for team_stats in stat_data:
                        team_name = team_stats.get("team", {}).get("name", "")
                        is_home = _names_match(team_name, home)
                        stats_list = team_stats.get("statistics", [])
                        for s in stats_list:
                            stat_type = (s.get("type") or "").lower()
                            val = s.get("value")
                            if stat_type == "corner kicks":
                                if is_home:
                                    corners_home = _safe_float(val)
                                else:
                                    corners_away = _safe_float(val)
                            elif stat_type == "yellow cards":
                                yc = _safe_float(val) or 0
                                rc_stat = next((x for x in stats_list if (x.get("type") or "").lower() == "red cards"), None)
                                rc = _safe_float(rc_stat.get("value")) if rc_stat else 0
                                if is_home:
                                    cards_home = yc + (rc or 0)
                                else:
                                    cards_away = yc + (rc or 0)
                            elif stat_type == "shots on goal":
                                if is_home:
                                    sot_home = _safe_float(val)
                                else:
                                    sot_away = _safe_float(val)
                except Exception as exc:
                    log.debug("Stats fetch failed for fixture %s: %s", fixture_id, exc)

            result = {
                "home_team": home,
                "away_team": away,
                "fixture_id": fixture_id,
                "fixture_date": prediction_date,
                "league": lg,
                "goals_home": _safe_float(goals_home),
                "goals_away": _safe_float(goals_away),
                "corners_home": corners_home,
                "corners_away": corners_away,
                "cards_home": cards_home,
                "cards_away": cards_away,
                "sot_home": sot_home,
                "sot_away": sot_away,
                "final_score": f"{home} {goals_home}-{goals_away} {away}",
            }
            results.append(result)

        log.info("Fetched %d results from API-Football for %s on %s", len(fixtures), lg, prediction_date)

    return results


def _safe_float(v: Any) -> Optional[float]:
    """Convert to float, returning None on failure."""
    if v is None:
        return None
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


# ===================================================================
# C2. Closing Line Value (CLV) Capture
# ===================================================================

def _match_odds_to_prediction(
    pred: dict,
    bookmakers: list,
    home_team: str,
    away_team: str,
) -> Optional[float]:
    """Find the closing odds that match a prediction's market/side/line.

    Searches through transformed bookmaker data (Odds-API-compatible format)
    for the outcome matching the prediction. Returns the decimal odds or None.
    """
    market = str(pred.get("market", "")).lower()
    side = str(pred.get("side", "")).lower()
    line = pred.get("line")
    pick = str(pred.get("pick", ""))

    # Map prediction market to API-Football market keys
    market_key_map = {
        "goals": ["totals"],
        "corners": ["totals_corners_over_under"],
        "cards": ["totals_cards_over_under", "totals_yellow_cards"],
        "sot": ["shots_on_target_over_under"],
        "btts": ["btts"],
        "moneyline": ["h2h"],
        "spreads": ["spreads"],
    }
    target_keys = market_key_map.get(market, [])
    if not target_keys:
        return None

    for bm in bookmakers:
        for mkt in bm.get("markets", []):
            if mkt.get("key") not in target_keys:
                continue
            for outcome in mkt.get("outcomes", []):
                price = outcome.get("price")
                if not price or price < 1.01:
                    continue
                name = str(outcome.get("name", "")).lower()
                point = outcome.get("point")

                # --- Over/Under markets ---
                if market in ("goals", "corners", "cards", "sot"):
                    if side == "over" and name == "over" and point is not None:
                        if line is not None and abs(float(point) - float(line)) < 0.01:
                            return float(price)
                    elif side == "under" and name == "under" and point is not None:
                        if line is not None and abs(float(point) - float(line)) < 0.01:
                            return float(price)

                # --- BTTS ---
                elif market == "btts":
                    if side == "yes" and name == "yes":
                        return float(price)
                    elif side == "no" and name == "no":
                        return float(price)

                # --- Moneyline ---
                elif market == "moneyline":
                    outcome_name = str(outcome.get("name", ""))
                    if side == "home" and _names_match(outcome_name, home_team):
                        return float(price)
                    elif side == "away" and _names_match(outcome_name, away_team):
                        return float(price)
                    elif side == "draw" and name == "draw":
                        return float(price)

                # --- Spreads ---
                elif market == "spreads":
                    outcome_name = str(outcome.get("name", ""))
                    if point is not None and line is not None:
                        if abs(float(point) - float(line)) < 0.01:
                            if side == "home" and _names_match(outcome_name, home_team):
                                return float(price)
                            elif side == "away" and _names_match(outcome_name, away_team):
                                return float(price)

    return None


def capture_closing_odds(
    prediction_date: str = None,
    *,
    db_path: Path = DB_PATH,
) -> dict:
    """Fetch closing odds for unresolved predictions and compute CLV.

    For each unique fixture among unresolved predictions that lack closing_odds,
    fetches current odds from API-Football and matches them to logged predictions.

    Returns {"captured": int, "skipped": int, "errors": int}.
    """
    if prediction_date is None:
        prediction_date = date.today().isoformat()

    stats = {"captured": 0, "skipped": 0, "errors": 0}

    try:
        conn = _get_db(db_path)
    except Exception:
        log.error("Cannot open database for CLV capture: %s", traceback.format_exc())
        stats["errors"] = 1
        return stats

    # Fetch predictions that need closing odds
    try:
        rows = conn.execute(
            """SELECT * FROM predictions
               WHERE prediction_date = ?
                 AND closing_odds IS NULL
                 AND odds IS NOT NULL""",
            [prediction_date],
        ).fetchall()
    except Exception:
        log.error("Failed to query predictions for CLV: %s", traceback.format_exc())
        conn.close()
        stats["errors"] = 1
        return stats

    if not rows:
        log.info("No predictions need closing odds for %s", prediction_date)
        conn.close()
        return stats

    preds = [dict(r) for r in rows]

    # Group by fixture_id for efficient API calls
    fixture_preds: Dict[str, List[dict]] = {}
    for p in preds:
        fid = p.get("fixture_id") or ""
        if fid:
            fixture_preds.setdefault(fid, []).append(p)
        else:
            # No fixture_id — try to resolve via league + teams + date
            fixture_preds.setdefault(f"_noid_{p['id']}", []).append(p)

    # Try importing odds_provider for fixture odds fetching
    try:
        from odds_provider import _fetch_fixture_odds, _transform_apifootball_odds
        has_odds_provider = True
    except ImportError:
        has_odds_provider = False

    if not has_odds_provider:
        # Fallback: direct API-Football call
        try:
            import requests
        except ImportError:
            log.warning("Neither odds_provider nor requests available for CLV capture")
            conn.close()
            stats["errors"] = 1
            return stats

    api_key = os.environ.get("API-FOOTBALL-KEY") or os.environ.get("API_FOOTBALL_KEY")
    if not api_key and not has_odds_provider:
        log.warning("No API-Football key for CLV capture")
        conn.close()
        return stats

    for fid, fid_preds in fixture_preds.items():
        if fid.startswith("_noid_"):
            stats["skipped"] += len(fid_preds)
            continue

        try:
            fixture_id_int = int(fid)
        except (ValueError, TypeError):
            stats["skipped"] += len(fid_preds)
            continue

        # Fetch odds for this fixture
        bookmakers_raw = None
        if has_odds_provider:
            try:
                bm_list, err = _fetch_fixture_odds(fixture_id_int)
                if err:
                    log.debug("CLV odds fetch error for fixture %s: %s", fid, err)
                    stats["skipped"] += len(fid_preds)
                    continue
                bookmakers_raw = bm_list
            except Exception as exc:
                log.debug("CLV odds_provider error for fixture %s: %s", fid, exc)
                stats["skipped"] += len(fid_preds)
                continue
        else:
            try:
                import requests
                resp = requests.get(
                    "https://v3.football.api-sports.io/odds",
                    headers={"x-apisports-key": api_key},
                    params={"fixture": fixture_id_int},
                    timeout=15,
                )
                resp.raise_for_status()
                api_rows = resp.json().get("response", [])
                bookmakers_raw = api_rows[0].get("bookmakers", []) if api_rows else []
            except Exception as exc:
                log.debug("CLV API fetch error for fixture %s: %s", fid, exc)
                stats["skipped"] += len(fid_preds)
                continue

        if not bookmakers_raw:
            stats["skipped"] += len(fid_preds)
            continue

        # Transform to standard format
        home_team = fid_preds[0].get("home_team", "Home")
        away_team = fid_preds[0].get("away_team", "Away")

        if has_odds_provider:
            try:
                bookmakers = _transform_apifootball_odds(bookmakers_raw, home_team, away_team)
            except Exception:
                bookmakers = bookmakers_raw
        else:
            bookmakers = bookmakers_raw

        # Match each prediction to closing odds
        for p in fid_preds:
            closing_price = _match_odds_to_prediction(p, bookmakers, home_team, away_team)
            if closing_price is None or closing_price < 1.01:
                stats["skipped"] += 1
                continue

            opening_odds = p.get("odds")
            if not opening_odds or opening_odds < 1.01:
                stats["skipped"] += 1
                continue

            closing_implied = 1.0 / closing_price
            opening_implied = 1.0 / opening_odds
            clv = (opening_implied / closing_implied) - 1.0  # positive = beat the close

            try:
                conn.execute(
                    """UPDATE predictions
                       SET closing_odds = ?, closing_implied_prob = ?, clv = ?
                       WHERE id = ?""",
                    (closing_price, closing_implied, clv, p["id"]),
                )
                stats["captured"] += 1
            except Exception:
                log.error("Failed to update CLV for prediction %d: %s",
                          p["id"], traceback.format_exc())
                stats["errors"] += 1

    try:
        conn.commit()
    except Exception:
        log.error("Failed to commit CLV updates: %s", traceback.format_exc())
        stats["errors"] += 1

    conn.close()
    log.info("CLV capture for %s: captured=%d, skipped=%d, errors=%d",
             prediction_date, stats["captured"], stats["skipped"], stats["errors"])
    return stats


def _match_prediction_to_result(
    pred: dict,
    results: List[dict],
) -> Optional[dict]:
    """Find the result matching a prediction by team names and date."""
    pred_home = str(pred.get("home_team", ""))
    pred_away = str(pred.get("away_team", ""))

    best_match = None
    best_score = 0.0

    for r in results:
        r_home = str(r.get("home_team", ""))
        r_away = str(r.get("away_team", ""))

        home_match = _names_match(pred_home, r_home)
        away_match = _names_match(pred_away, r_away)

        if home_match and away_match:
            # Compute a tighter score for tie-breaking
            h_ratio = difflib.SequenceMatcher(
                None, _normalize(pred_home), _normalize(r_home)
            ).ratio()
            a_ratio = difflib.SequenceMatcher(
                None, _normalize(pred_away), _normalize(r_away)
            ).ratio()
            score = h_ratio + a_ratio
            if score > best_score:
                best_score = score
                best_match = r

    return best_match


def _grade_prediction(pred: dict, result: dict) -> Tuple[Optional[str], Optional[float]]:
    """Grade a single prediction against an actual result.

    Returns (outcome, actual_value). Asian totals may settle as ``half_hit``
    or ``half_miss`` in addition to hit/miss/push.
    """
    market = str(pred.get("market", "")).lower()
    side = str(pred.get("side", "")).lower()
    line = pred.get("line")

    def _grade_asian_total(total: float) -> Optional[str]:
        if line is None or side not in {"over", "under"}:
            return None
        try:
            from prob_models import asian_total_settlement_outcome
        except ImportError:
            from Scripts.rag_ingest.prob_models import asian_total_settlement_outcome  # type: ignore[import]
        settlement = asian_total_settlement_outcome(int(total), float(line), side)
        return {
            "full_win": "hit",
            "half_win": "half_hit",
            "push": "push",
            "half_loss": "half_miss",
            "full_loss": "miss",
        }[settlement]

    # --- Goals Over/Under ---
    if market == "goals":
        gh = result.get("goals_home")
        ga = result.get("goals_away")
        if gh is None or ga is None:
            return None, None
        total = gh + ga
        return _grade_asian_total(total), total

    # --- Corners Over/Under ---
    if market == "corners":
        ch = result.get("corners_home")
        ca = result.get("corners_away")
        if ch is None or ca is None:
            return None, None
        total = ch + ca
        return _grade_asian_total(total), total

    # --- Cards Over/Under ---
    if market == "cards":
        ch = result.get("cards_home")
        ca = result.get("cards_away")
        if ch is None or ca is None:
            return None, None
        total = ch + ca
        return _grade_asian_total(total), total

    # --- SoT Over/Under ---
    if market == "sot":
        sh = result.get("sot_home")
        sa = result.get("sot_away")
        if sh is None or sa is None:
            return None, None
        total = sh + sa
        return _grade_asian_total(total), total

    # --- BTTS ---
    if market == "btts":
        gh = result.get("goals_home")
        ga = result.get("goals_away")
        if gh is None or ga is None:
            return None, None
        both_scored = (gh >= 1 and ga >= 1)
        actual_val = 1.0 if both_scored else 0.0
        if side == "yes":
            return ("hit" if both_scored else "miss"), actual_val
        elif side == "no":
            return ("hit" if not both_scored else "miss"), actual_val
        return None, actual_val

    # --- Moneyline ---
    if market == "moneyline":
        gh = result.get("goals_home")
        ga = result.get("goals_away")
        if gh is None or ga is None:
            return None, None
        if gh > ga:
            actual_outcome = "home"
        elif ga > gh:
            actual_outcome = "away"
        else:
            actual_outcome = "draw"
        actual_val = {"home": 1.0, "draw": 0.5, "away": 0.0}[actual_outcome]
        if side == actual_outcome:
            return "hit", actual_val
        else:
            return "miss", actual_val

    # --- Spreads ---
    if market == "spreads":
        gh = result.get("goals_home")
        ga = result.get("goals_away")
        if gh is None or ga is None:
            return None, None
        actual_diff = gh - ga  # positive = home won by that margin
        if line is None:
            return None, actual_diff
        # Spread logic: the "side" tells us home or away.
        # line is the handicap (e.g., -1.5 for favorites).
        # A "home -1.5" pick means home must win by > 1.5.
        # The stored line already has the sign: home_team -1.5 → line = -1.5
        # Cover condition: actual_diff + line > 0 for the picked side.
        if side == "home":
            adjusted = actual_diff + line
        else:  # away
            adjusted = -actual_diff + line

        if adjusted > 0:
            return "hit", actual_diff
        elif adjusted < 0:
            return "miss", actual_diff
        else:
            return "push", actual_diff

    # --- Correct Score ---
    if market == "correct_score":
        gh = result.get("goals_home")
        ga = result.get("goals_away")
        if gh is None or ga is None:
            return None, None
        actual_score = f"{int(gh)}-{int(ga)}"
        # The pick is stored as "X-Y"
        pick = str(pred.get("pick", ""))
        pick_clean = re.sub(r"\s+", "", pick)
        if pick_clean == actual_score:
            return "hit", None
        else:
            return "miss", None

    return None, None


def _maybe_platform_resolve(kwargs: Dict[str, Any]):
    """When running against the canonical store, route outcome grading
    through :class:`PredictionService`.

    Grading logic (API-Football fetch, name-matching, market-specific
    rules) stays in the legacy helpers — we only change where rows are
    read from and where outcomes are written to.
    """
    if kwargs.get("db_path", DB_PATH) != DB_PATH or not _platform_on():
        return None
    try:
        from data_platform.compat import (
            platform_get_unresolved_for_grading,
            platform_mark_outcome,
        )
    except Exception:
        return None
    try:
        prediction_date = kwargs.get("prediction_date")
        before = None
        if prediction_date:
            try:
                before = date.fromisoformat(str(prediction_date))
            except ValueError:
                before = None
        unresolved = platform_get_unresolved_for_grading(on_or_before=before)
        if not unresolved:
            return {"graded": 0, "errors": 0, "skipped": 0}

        league = kwargs.get("league")
        if league:
            unresolved = [p for p in unresolved if (p.get("league") or "") == league]

        # Group by (date, league) so we only hit API-Football once per slice.
        by_slice: Dict[Tuple[str, str], List[dict]] = {}
        for pred in unresolved:
            key = (pred.get("prediction_date") or "", pred.get("league") or "")
            by_slice.setdefault(key, []).append(pred)

        graded = 0
        errors = 0
        skipped = 0
        outcome_counts = {"hit": 0, "half_hit": 0, "miss": 0, "half_miss": 0, "push": 0}
        for (slice_date, slice_league), preds in by_slice.items():
            try:
                results = _fetch_results_from_api(slice_date, slice_league)
            except Exception:
                log.exception("platform _fetch_results_from_api failed for %s %s",
                              slice_date, slice_league)
                errors += 1
                continue
            if not results:
                skipped += len(preds)
                continue
            for pred in preds:
                match = _match_prediction_to_result(pred, results)
                if match is None:
                    skipped += 1
                    continue
                outcome, actual = _grade_prediction(pred, match)
                if outcome is None:
                    skipped += 1
                    continue
                if platform_mark_outcome(int(pred["id"]), outcome=outcome, actual_result=actual):
                    graded += 1
                    canonical = normalize_outcome(outcome)
                    if canonical == "void":
                        canonical = "push"
                    if canonical in outcome_counts:
                        outcome_counts[canonical] += 1
                else:
                    errors += 1
        return {"graded": graded, "errors": errors, "skipped": skipped, **outcome_counts}
    except Exception:
        log.exception("platform resolve_outcomes failed; falling through to SQLite")
        return None


def resolve_outcomes(
    prediction_date: str = None,
    league: str = None,
    dry_run: bool = False,
    *,
    db_path: Path = DB_PATH,
) -> dict:
    """Fetch actual match results and grade unresolved predictions.

    Returns {"graded": int, "hit": int, "miss": int, "push": int, "errors": int, "details": [...]}
    """
    if prediction_date is None:
        prediction_date = (date.today() - timedelta(days=1)).isoformat()

    stats = {"graded": 0, "hit": 0, "miss": 0, "push": 0, "errors": 0, "details": []}

    if not dry_run:
        platform_result = _maybe_platform_resolve({
            "prediction_date": prediction_date, "league": league, "db_path": db_path,
        })
        if platform_result is not None:
            return {
                "graded": platform_result.get("graded", 0),
                "hit": platform_result.get("hit", 0),
                "half_hit": platform_result.get("half_hit", 0),
                "miss": platform_result.get("miss", 0),
                "half_miss": platform_result.get("half_miss", 0),
                "push": platform_result.get("push", 0),
                "errors": platform_result.get("errors", 0),
                "details": [],
                "backend": "platform",
                "skipped": platform_result.get("skipped", 0),
            }

    try:
        conn = _get_db(db_path)
    except Exception:
        log.error("Cannot open database: %s", traceback.format_exc())
        stats["errors"] = 1
        return stats

    # Fetch unresolved predictions for the given date
    query = "SELECT * FROM predictions WHERE prediction_date = ? AND outcome IS NULL"
    params: list = [prediction_date]
    if league:
        query += " AND league = ?"
        params.append(league)

    try:
        rows = conn.execute(query, params).fetchall()
    except Exception:
        log.error("Failed to query predictions: %s", traceback.format_exc())
        stats["errors"] = 1
        conn.close()
        return stats

    if not rows:
        log.info("No unresolved predictions found for %s%s",
                 prediction_date, f" ({league})" if league else "")
        conn.close()
        return stats

    # Fetch actual results — try local data first, fallback to API-Football
    results = _fetch_results_from_data(prediction_date, league)
    if not results:
        results = _fetch_results_from_api(prediction_date, league)
    log.info("Found %d actual results for %s%s",
             len(results), prediction_date, f" ({league})" if league else "")

    now_iso = datetime.now().isoformat()

    for row in rows:
        pred = dict(row)
        result = _match_prediction_to_result(pred, results)

        if result is None:
            detail = {
                "id": pred["id"],
                "fixture": f"{pred['home_team']} vs {pred['away_team']}",
                "status": "no_result_found",
            }
            stats["details"].append(detail)
            continue

        outcome, actual_val = _grade_prediction(pred, result)
        if outcome is None:
            detail = {
                "id": pred["id"],
                "fixture": f"{pred['home_team']} vs {pred['away_team']}",
                "status": "cannot_grade",
                "market": pred["market"],
            }
            stats["details"].append(detail)
            stats["errors"] += 1
            continue

        detail = {
            "id": pred["id"],
            "fixture": f"{pred['home_team']} vs {pred['away_team']}",
            "market": pred["market"],
            "pick": pred["pick"],
            "outcome": outcome,
            "actual": actual_val,
        }
        stats["details"].append(detail)
        stats["graded"] += 1
        stats[outcome] = stats.get(outcome, 0) + 1

        if not dry_run:
            try:
                conn.execute(
                    "UPDATE predictions SET outcome = ?, actual_result = ?, graded_at = ? WHERE id = ?",
                    (outcome, actual_val, now_iso, pred["id"]),
                )
            except Exception:
                log.error("Failed to update prediction %d: %s", pred["id"], traceback.format_exc())
                stats["errors"] += 1

            # Compute CLV if closing odds are present (may have been captured earlier)
            opening_odds = pred.get("odds")
            closing_odds = pred.get("closing_odds")
            if opening_odds and closing_odds and opening_odds > 1.0 and closing_odds > 1.0:
                if pred.get("clv") is None:
                    opening_implied = 1.0 / opening_odds
                    closing_implied = 1.0 / closing_odds
                    clv_val = (opening_implied / closing_implied) - 1.0
                    try:
                        conn.execute(
                            "UPDATE predictions SET clv = ? WHERE id = ?",
                            (clv_val, pred["id"]),
                        )
                    except Exception:
                        log.debug("Failed to update CLV for prediction %d", pred["id"])

    # Attempt to capture closing odds for predictions that still lack them
    if not dry_run:
        try:
            missing_clv = conn.execute(
                """SELECT id, fixture_id, home_team, away_team, market, side, line,
                          pick, odds FROM predictions
                   WHERE prediction_date = ? AND closing_odds IS NULL AND odds IS NOT NULL""",
                [prediction_date],
            ).fetchall()
            if missing_clv:
                log.info("Attempting CLV capture for %d predictions still missing closing odds",
                         len(missing_clv))
        except Exception:
            missing_clv = []

        # Commit the grading results first
        try:
            conn.commit()
        except Exception:
            log.error("Failed to commit: %s", traceback.format_exc())
            stats["errors"] += 1

    conn.close()

    # If there are predictions without closing odds, attempt capture now
    if not dry_run and prediction_date:
        try:
            clv_stats = capture_closing_odds(prediction_date=prediction_date, db_path=db_path)
            if clv_stats.get("captured", 0) > 0:
                log.info("Post-resolve CLV capture: %d closing odds captured", clv_stats["captured"])
        except Exception:
            log.debug("Post-resolve CLV capture failed: %s", traceback.format_exc())

    return stats


# ===================================================================
# D. Track Record Statistics
# ===================================================================

def _maybe_platform_track_record(kwargs: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if kwargs.get("db_path", DB_PATH) != DB_PATH:
        return None
    if not _platform_on():
        return None
    try:
        from data_platform.compat import platform_get_track_record
        return platform_get_track_record(**{k: v for k, v in kwargs.items() if k != "db_path"})
    except Exception:
        log.exception("platform get_track_record failed; falling through to SQLite")
        return None


def get_track_record(
    days: int = None,
    league: str = None,
    market: str = None,
    confidence: str = None,
    source: str = None,
    published_only: bool = False,
    *,
    db_path: Path = DB_PATH,
) -> dict:
    """Compute track record statistics from graded predictions.

    Returns a dict with overall stats, breakdowns by confidence/market/league,
    streak info, daily performance, and ROI calculations.
    """
    platform_result = _maybe_platform_track_record({
        "days": days, "league": league, "market": market,
        "confidence": confidence, "source": source,
        "published_only": published_only, "db_path": db_path,
    })
    if platform_result is not None:
        return platform_result
    try:
        conn = _get_db(db_path)
    except Exception:
        log.error("Cannot open database: %s", traceback.format_exc())
        return {"error": "database_unavailable"}

    # Build query with filters
    where_clauses = ["outcome IS NOT NULL"]
    params: list = []

    if days is not None:
        cutoff = (date.today() - timedelta(days=days)).isoformat()
        where_clauses.append("prediction_date >= ?")
        params.append(cutoff)
    if league:
        where_clauses.append("league = ?")
        params.append(league)
    if market:
        where_clauses.append("market = ?")
        params.append(market)
    if confidence:
        where_clauses.append("confidence = ?")
        params.append(confidence)
    if source:
        where_clauses.append("source = ?")
        params.append(source)
    if published_only:
        # The platform implementation uses the immutable publication join.
        # This fallback keeps the old SQLite path useful by selecting only
        # the dedicated release source (including release-specific suffixes).
        where_clauses.append("(source = ? OR source LIKE ?)")
        params.extend(["canonical_published", "canonical_published:%"])

    where_sql = " AND ".join(where_clauses)

    try:
        rows = conn.execute(
            f"SELECT * FROM predictions WHERE {where_sql} ORDER BY prediction_date, id",
            params,
        ).fetchall()
    except Exception:
        log.error("Failed to query: %s", traceback.format_exc())
        conn.close()
        return {"error": "query_failed"}

    conn.close()

    return build_track_record([dict(row) for row in rows])


def _outcome_stakes(preds: List[dict]) -> Tuple[float, float]:
    """Return winning and resolved stake units, including Asian half-settles."""
    totals = outcome_totals(preds)
    return totals["win_units"], totals["resolved_units"]


def _compute_roi(preds: List[dict]) -> float:
    """Compute ROI for flat 1-unit stakes on all predictions with logged odds."""
    return flat_stake_roi(preds)[0]


def _compute_streaks(preds: List[dict]) -> Tuple[int, int]:
    """Compute current and best streak from ordered predictions.

    Positive = win streak, negative = loss streak.
    """
    if not preds:
        return 0, 0

    # Filter to only full hit/miss (skip partial outcomes, push and void).
    decided = [
        normalize_outcome(p.get("outcome"))
        for p in preds
        if normalize_outcome(p.get("outcome")) in ("hit", "miss")
    ]
    if not decided:
        return 0, 0

    # Current streak
    current = 0
    last_outcome = decided[-1]
    for outcome in reversed(decided):
        if outcome == last_outcome:
            current += 1
        else:
            break
    if last_outcome == "miss":
        current = -current

    # Best win streak
    best_win = 0
    run = 0
    for outcome in decided:
        if outcome == "hit":
            run += 1
            best_win = max(best_win, run)
        else:
            run = 0

    return current, best_win


def _compute_daily_performance(preds: List[dict], days: int = 30) -> List[dict]:
    """Compute daily hit-rate for the most recent N days."""
    cutoff = (date.today() - timedelta(days=days)).isoformat()
    recent = [p for p in preds if (p.get("prediction_date") or "") >= cutoff]

    by_date: Dict[str, List[dict]] = {}
    for p in recent:
        d = p.get("prediction_date", "")
        if d:
            by_date.setdefault(d, []).append(p)

    daily = []
    for d in sorted(by_date.keys()):
        day_preds = by_date[d]
        totals = outcome_totals(day_preds)
        daily.append({
            "date": d,
            "total": len(day_preds),
            "hits": int(totals["hits"]),
            "hit_rate": totals["win_units"] / totals["resolved_units"] if totals["resolved_units"] else 0.0,
        })

    return daily


def _maybe_platform_daily(kwargs: Dict[str, Any]):
    if kwargs.get("db_path", DB_PATH) != DB_PATH:
        return None
    if not _platform_on():
        return None
    try:
        from data_platform.compat import platform_get_daily_breakdown
        return platform_get_daily_breakdown(**{k: v for k, v in kwargs.items() if k != "db_path"})
    except Exception:
        log.exception("platform get_daily_breakdown failed; falling through to SQLite")
        return None


def get_daily_breakdown(
    prediction_date: str = None,
    published_only: bool = False,
    *,
    db_path: Path = DB_PATH,
) -> dict:
    """Get yesterday's (or given date's) graded predictions broken down by market and league.

    Returns {
        "date": "YYYY-MM-DD",
        "total": int, "hits": int, "misses": int, "pushes": int,
        "hit_rate": float,
        "by_market": {market: {"hits": int, "total": int, "hit_rate": float, "pushes": int}},
        "by_league": {league: {"hits": int, "total": int, "hit_rate": float, "pushes": int}},
        "by_confidence": {level: {"hits": int, "total": int, "hit_rate": float}},
        "notable_hits": [...], "notable_misses": [...],
    }
    """
    if prediction_date is None:
        prediction_date = (date.today() - timedelta(days=1)).isoformat()

    platform = _maybe_platform_daily({
        "target_date": prediction_date,
        "published_only": published_only,
        "db_path": db_path,
    })
    if platform is not None:
        return platform

    try:
        conn = _get_db(db_path)
    except Exception:
        return {"date": prediction_date, "total": 0}

    try:
        query = "SELECT * FROM predictions WHERE prediction_date = ? AND outcome IS NOT NULL"
        params = [prediction_date]
        if published_only:
            query += " AND (source = ? OR source LIKE ?)"
            params.extend(["canonical_published", "canonical_published:%"])
        rows = conn.execute(f"{query} ORDER BY id", params).fetchall()
    except Exception:
        conn.close()
        return {"date": prediction_date, "total": 0}

    conn.close()

    return build_daily_breakdown([dict(row) for row in rows], target_date=prediction_date)


def _maybe_platform_recent(kwargs: Dict[str, Any]):
    if kwargs.get("db_path", DB_PATH) != DB_PATH:
        return None
    if not _platform_on():
        return None
    try:
        from data_platform.compat import platform_get_recent_predictions
        return platform_get_recent_predictions(**{k: v for k, v in kwargs.items() if k != "db_path"})
    except Exception:
        log.exception("platform get_recent_predictions failed; falling through to SQLite")
        return None


def get_recent_predictions(
    limit: int = 20,
    league: str = None,
    graded_only: bool = True,
    published_only: bool = False,
    *,
    db_path: Path = DB_PATH,
) -> List[dict]:
    """Fetch the most recent predictions, optionally filtered."""
    platform = _maybe_platform_recent({
        "limit": limit, "league": league,
        "days": 180,  # sensible upper bound for "recent"
        "published_only": published_only,
        "db_path": db_path,
    })
    if platform is not None:
        rows = platform
        if graded_only:
            rows = [r for r in rows if r.get("outcome")]
        return rows
    try:
        conn = _get_db(db_path)
    except Exception:
        return []

    where_clauses = []
    params: list = []

    if graded_only:
        where_clauses.append("outcome IS NOT NULL")
    if league:
        where_clauses.append("league = ?")
        params.append(league)
    if published_only:
        where_clauses.append("(source = ? OR source LIKE ?)")
        params.extend(["canonical_published", "canonical_published:%"])

    where_sql = " AND ".join(where_clauses) if where_clauses else "1=1"
    params.append(limit)

    try:
        rows = conn.execute(
            f"SELECT * FROM predictions WHERE {where_sql} ORDER BY created_at DESC LIMIT ?",
            params,
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception:
        conn.close()
        return []


def export_predictions_csv(
    output_path: str,
    league: str = None,
    *,
    db_path: Path = DB_PATH,
) -> int:
    """Export all predictions to CSV. Returns count of rows written."""
    try:
        conn = _get_db(db_path)
    except Exception:
        return 0

    where = "1=1"
    params: list = []
    if league:
        where = "league = ?"
        params.append(league)

    rows = conn.execute(
        f"SELECT * FROM predictions WHERE {where} ORDER BY prediction_date, id",
        params,
    ).fetchall()
    conn.close()

    if not rows:
        return 0

    columns = rows[0].keys()
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for r in rows:
            writer.writerow(dict(r))

    return len(rows)


# ===================================================================
# E. CLI Interface
# ===================================================================

def _format_stats_report(stats: dict) -> str:
    """Format track record stats as a human-readable CLI report."""
    lines = [
        "",
        "Matchwise Track Record",
        "=" * 50,
        "",
    ]

    total = stats.get("total_graded", 0)
    hits = stats.get("hits", 0)
    misses = stats.get("misses", 0)
    pushes = stats.get("pushes", 0)
    decided = hits + misses
    hr = stats.get("hit_rate", 0.0)
    roi = stats.get("roi_flat_stake", 0.0)

    lines.append(f"Overall: {hits}/{decided} ({hr:.1%})   ROI: {roi:+.1%}")

    # CLV summary in main stats
    avg_clv = stats.get("avg_clv")
    clv_pos = stats.get("clv_positive_rate")
    clv_n = stats.get("clv_sample_size", 0)
    if avg_clv is not None and clv_n > 0:
        direction = "+" if avg_clv > 0 else ""
        lines.append(f"CLV: {direction}{avg_clv:.2%} avg   CLV+ rate: {clv_pos:.1%}  ({clv_n} picks)")

    streak = stats.get("recent_streak", 0)
    if streak > 0:
        lines.append(f"Current streak: W{streak}")
    elif streak < 0:
        lines.append(f"Current streak: L{abs(streak)}")
    else:
        lines.append("Current streak: -")

    best = stats.get("best_streak", 0)
    if best > 0:
        lines.append(f"Best win streak: W{best}")

    if pushes > 0:
        lines.append(f"Pushes: {pushes}")

    # By Confidence
    by_conf = stats.get("by_confidence", {})
    if by_conf:
        lines.append("")
        lines.append("By Confidence:")
        label_map = {"high": "Strong Edge", "medium": "Leaning", "low": "Speculative"}
        for level in ["high", "medium", "low"]:
            if level in by_conf:
                c = by_conf[level]
                d = c["hits"] + (c["total"] - c["hits"])  # approximate decided
                roi_str = f"   ROI: {c['roi']:+.1%}" if c.get("roi") is not None else ""
                label = label_map.get(level, level.title())
                lines.append(f"  {label:15s} {c['hits']}/{c['total']} ({c['hit_rate']:.1%}){roi_str}")

    # By Market
    by_market = stats.get("by_market", {})
    if by_market:
        lines.append("")
        lines.append("By Market:")
        market_labels = {
            "goals": "Goals O/U",
            "corners": "Corners O/U",
            "cards": "Cards O/U",
            "sot": "SoT O/U",
            "btts": "BTTS",
            "moneyline": "Moneyline",
            "spreads": "Spreads",
            "correct_score": "Correct Score",
        }
        for mkt in ["goals", "corners", "cards", "sot", "btts", "moneyline", "spreads", "correct_score"]:
            if mkt in by_market:
                m = by_market[mkt]
                label = market_labels.get(mkt, mkt.title())
                clv_str = ""
                if m.get("avg_clv") is not None:
                    d = "+" if m["avg_clv"] > 0 else ""
                    clv_str = f"   CLV: {d}{m['avg_clv']:.2%}"
                lines.append(f"  {label:15s} {m['hits']}/{m['total']} ({m['hit_rate']:.1%}){clv_str}")

    # By League
    by_league = stats.get("by_league", {})
    if by_league:
        lines.append("")
        lines.append("By League:")
        for lg in sorted(by_league.keys()):
            l = by_league[lg]
            lines.append(f"  {lg:15s} {l['hits']}/{l['total']} ({l['hit_rate']:.1%})")

    # Daily performance
    daily = stats.get("daily_performance", [])
    if daily:
        # Show last 7 days summary
        last7 = daily[-7:] if len(daily) >= 7 else daily
        total_7 = sum(d["total"] for d in last7)
        hits_7 = sum(d["hits"] for d in last7)
        decided_7 = sum(d["hits"] + (d["total"] - d["hits"]) for d in last7)
        hr_7 = hits_7 / decided_7 if decided_7 > 0 else 0.0
        lines.append("")
        lines.append(f"Last 7 days: {hits_7}/{decided_7} ({hr_7:.1%})")

    lines.append("")
    return "\n".join(lines)


def _format_resolve_report(stats: dict, dry_run: bool = False) -> str:
    """Format outcome resolution report."""
    lines = [
        "",
        f"{'[DRY RUN] ' if dry_run else ''}Outcome Resolution Report",
        "=" * 50,
        "",
        f"Graded: {stats['graded']}  |  Hit: {stats['hit']}  |  Miss: {stats['miss']}  |  Push: {stats['push']}  |  Errors: {stats['errors']}",
    ]

    details = stats.get("details", [])
    for d in details:
        status = d.get("outcome") or d.get("status", "?")
        fixture = d.get("fixture", "?")
        market = d.get("market", "")
        pick = d.get("pick", "")
        actual = d.get("actual")
        actual_str = f" (actual: {actual})" if actual is not None else ""
        lines.append(f"  [{status:>15s}]  {fixture} — {market} {pick}{actual_str}")

    lines.append("")
    return "\n".join(lines)


def _format_clv_report(
    stats: dict,
    daily: dict = None,
) -> str:
    """Format a CLV-focused report for the CLI."""
    lines = [
        "",
        "Closing Line Value (CLV) Report",
        "=" * 50,
        "",
    ]

    avg_clv = stats.get("avg_clv")
    clv_pos = stats.get("clv_positive_rate")
    clv_n = stats.get("clv_sample_size", 0)

    if clv_n == 0:
        lines.append("No CLV data available yet.")
        lines.append("")
        lines.append("CLV requires closing odds to be captured before kickoff.")
        lines.append("Run: python prediction_tracker.py --capture-clv")
        lines.append("")
        return "\n".join(lines)

    lines.append(f"Sample size: {clv_n} predictions with CLV data")
    lines.append("")

    if avg_clv is not None:
        direction = "+" if avg_clv > 0 else ""
        lines.append(f"Average CLV:      {direction}{avg_clv:.2%}")
    if clv_pos is not None:
        lines.append(f"CLV+ rate:        {clv_pos:.1%}  (target: >50%)")

    # Interpret the results
    if avg_clv is not None and clv_pos is not None:
        lines.append("")
        if clv_pos >= 0.55 and avg_clv > 0:
            lines.append("Assessment: STRONG — consistently beating the closing line.")
            lines.append("This is the single best predictor of long-term profitability.")
        elif clv_pos >= 0.50 and avg_clv > 0:
            lines.append("Assessment: POSITIVE — beating the close more often than not.")
        elif clv_pos >= 0.45:
            lines.append("Assessment: MARGINAL — roughly in line with the market.")
        else:
            lines.append("Assessment: NEGATIVE — the market is sharper than our timing.")
            lines.append("Consider placing bets earlier when lines first open.")

    # By market breakdown
    by_market = stats.get("by_market", {})
    market_clv = {k: v for k, v in by_market.items() if v.get("avg_clv") is not None}
    if market_clv:
        lines.append("")
        lines.append("By Market:")
        market_labels = {
            "goals": "Goals O/U",
            "corners": "Corners O/U",
            "cards": "Cards O/U",
            "sot": "SoT O/U",
            "btts": "BTTS",
            "moneyline": "Moneyline",
            "spreads": "Spreads",
            "correct_score": "Correct Score",
        }
        for mkt in ["goals", "corners", "cards", "sot", "btts", "moneyline", "spreads", "correct_score"]:
            if mkt in market_clv:
                m = market_clv[mkt]
                label = market_labels.get(mkt, mkt.title())
                clv_val = m["avg_clv"]
                direction = "+" if clv_val > 0 else ""
                lines.append(f"  {label:15s} avg CLV: {direction}{clv_val:.2%}")

    # Daily breakdown if provided
    if daily and daily.get("avg_clv") is not None:
        lines.append("")
        lines.append(f"Today ({daily['date']}):")
        d_clv = daily["avg_clv"]
        d_pos = daily.get("clv_positive_rate")
        d_n = daily.get("clv_sample_size", 0)
        direction = "+" if d_clv > 0 else ""
        lines.append(f"  Avg CLV: {direction}{d_clv:.2%}  ({d_n} picks)")
        if d_pos is not None:
            lines.append(f"  CLV+ rate: {d_pos:.1%}")

    lines.append("")
    return "\n".join(lines)


# ===================================================================
#  Calibration Analysis
# ===================================================================

def _maybe_platform_calibration(
    db_path: Path,
    buckets: int = 10,
    published_only: bool = False,
):
    if db_path != DB_PATH or not _platform_on():
        return None
    try:
        from data_platform.compat import platform_get_calibration_data
        return platform_get_calibration_data(
            buckets=buckets,
            published_only=published_only,
        )
    except Exception:
        log.exception("platform get_calibration_data failed; falling through to SQLite")
        return None


def get_calibration_data(
    *,
    published_only: bool = False,
    db_path: Path = DB_PATH,
) -> List[Dict]:
    """Return calibration data: for each probability bucket, model prob vs actual hit rate.

    Groups resolved predictions by model probability into 5% buckets (50-55%, 55-60%, etc.)
    and computes the actual hit rate for each bucket.
    """
    platform = _maybe_platform_calibration(db_path, published_only=published_only)
    if platform is not None:
        return platform
    try:
        db = _get_db(db_path)
    except Exception:
        return []

    try:
        query = "SELECT model_prob, outcome FROM predictions WHERE outcome IS NOT NULL AND model_prob IS NOT NULL"
        params = []
        if published_only:
            query += " AND (source = ? OR source LIKE ?)"
            params.extend(["canonical_published", "canonical_published:%"])
        rows = db.execute(query, params).fetchall()
    except Exception:
        db.close()
        return []

    db.close()

    return build_calibration([dict(row) for row in rows], buckets=20)


def main():
    parser = argparse.ArgumentParser(
        description="Prediction tracker -- log, resolve, and report."
    )
    parser.add_argument("--resolve", action="store_true",
                        help="Resolve yesterday's outcomes")
    parser.add_argument("--resolve-date", type=str, default=None,
                        help="Resolve outcomes for a specific date (YYYY-MM-DD)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be graded without writing")
    parser.add_argument("--stats", action="store_true",
                        help="Print track record stats")
    parser.add_argument("--stats-days", type=int, default=None,
                        help="Limit stats to last N days")
    parser.add_argument("--stats-market", type=str, default=None,
                        help="Filter stats to one market")
    parser.add_argument("--stats-confidence", type=str, default=None,
                        help="Filter stats to one confidence level")
    parser.add_argument("--export", type=str, default=None,
                        help="Export all predictions to CSV")
    parser.add_argument("--clv", action="store_true",
                        help="Print CLV (Closing Line Value) report")
    parser.add_argument("--capture-clv", action="store_true",
                        help="Capture closing odds for predictions (run before kickoff)")
    parser.add_argument("--backfill-unit-bets", action="store_true",
                        help="Create historical unit-bet slips from old source=unit_bets predictions")
    parser.add_argument("--league", type=str, default=None,
                        help="Filter to one league")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    if (args.resolve or args.resolve_date) and not args.backfill_unit_bets:
        target_date = args.resolve_date
        if target_date is None:
            target_date = (date.today() - timedelta(days=1)).isoformat()
        print(f"Resolving outcomes for {target_date}{'  [DRY RUN]' if args.dry_run else ''}...")
        result = resolve_outcomes(
            prediction_date=target_date,
            league=args.league,
            dry_run=args.dry_run,
        )
        print(_format_resolve_report(result, dry_run=args.dry_run))

    elif args.stats:
        stats = get_track_record(
            days=args.stats_days,
            league=args.league,
            market=args.stats_market,
            confidence=args.stats_confidence,
        )
        print(_format_stats_report(stats))

    elif args.backfill_unit_bets:
        result = backfill_unit_bet_slips_from_predictions(
            prediction_date=args.resolve_date,
            source="unit_bets",
        )
        if args.resolve_date:
            resolve_unit_bet_slips(prediction_date=args.resolve_date)
        record = get_unit_bet_track_record(source="unit_bets")
        print(
            "Backfilled unit bets: "
            f"{result['slips_created']} slip(s) across {result['dates_backfilled']} date(s); "
            f"settled {result['slips_settled']} from already graded legs; "
            f"skipped {result['dates_skipped_existing']} date(s) with existing slips; "
            f"errors {result['errors']}."
        )
        print(
            "Unit bet record: "
            f"{record.get('wins', 0)}-{record.get('losses', 0)}"
            + (f"-{record.get('pushes', 0)}P" if record.get("pushes", 0) else "")
            + f" | Units {record.get('units_profit', 0.0):+.2f}u | ROI {record.get('roi', 0.0):+.1%}"
        )

    elif args.capture_clv:
        target_date = args.resolve_date or date.today().isoformat()
        print(f"Capturing closing odds for {target_date}...")
        clv_result = capture_closing_odds(prediction_date=target_date)
        print(f"Captured: {clv_result['captured']}  |  Skipped: {clv_result['skipped']}  |  Errors: {clv_result['errors']}")

    elif args.clv:
        track = get_track_record(
            days=args.stats_days,
            league=args.league,
            market=args.stats_market,
            confidence=args.stats_confidence,
        )
        yesterday = (date.today() - timedelta(days=1)).isoformat()
        daily = get_daily_breakdown(prediction_date=yesterday)
        print(_format_clv_report(track, daily))

    elif args.export:
        n = export_predictions_csv(args.export, league=args.league)
        print(f"Exported {n} predictions to {args.export}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
