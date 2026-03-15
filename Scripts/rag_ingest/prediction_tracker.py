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
"""

from __future__ import annotations

import argparse
import csv
import difflib
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
    """
    try:
        conn = _get_db(db_path)
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
                date.today().isoformat(),
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


def _safe_float(v: Any) -> Optional[float]:
    """Convert to float, returning None on failure."""
    if v is None:
        return None
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


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

    Returns (outcome, actual_value) where outcome is 'hit', 'miss', 'push', or None (cannot grade).
    """
    market = str(pred.get("market", "")).lower()
    side = str(pred.get("side", "")).lower()
    line = pred.get("line")

    # --- Goals Over/Under ---
    if market == "goals":
        gh = result.get("goals_home")
        ga = result.get("goals_away")
        if gh is None or ga is None:
            return None, None
        total = gh + ga
        if line is None:
            return None, total
        if side == "over":
            if total > line:
                return "hit", total
            elif total < line:
                return "miss", total
            else:
                return "push", total
        elif side == "under":
            if total < line:
                return "hit", total
            elif total > line:
                return "miss", total
            else:
                return "push", total
        return None, total

    # --- Corners Over/Under ---
    if market == "corners":
        ch = result.get("corners_home")
        ca = result.get("corners_away")
        if ch is None or ca is None:
            return None, None
        total = ch + ca
        if line is None:
            return None, total
        if side == "over":
            return ("hit" if total > line else ("push" if total == line else "miss")), total
        elif side == "under":
            return ("hit" if total < line else ("push" if total == line else "miss")), total
        return None, total

    # --- Cards Over/Under ---
    if market == "cards":
        ch = result.get("cards_home")
        ca = result.get("cards_away")
        if ch is None or ca is None:
            return None, None
        total = ch + ca
        if line is None:
            return None, total
        if side == "over":
            return ("hit" if total > line else ("push" if total == line else "miss")), total
        elif side == "under":
            return ("hit" if total < line else ("push" if total == line else "miss")), total
        return None, total

    # --- SoT Over/Under ---
    if market == "sot":
        sh = result.get("sot_home")
        sa = result.get("sot_away")
        if sh is None or sa is None:
            return None, None
        total = sh + sa
        if line is None:
            return None, total
        if side == "over":
            return ("hit" if total > line else ("push" if total == line else "miss")), total
        elif side == "under":
            return ("hit" if total < line else ("push" if total == line else "miss")), total
        return None, total

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

    # Fetch actual results
    results = _fetch_results_from_data(prediction_date, league)
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

    if not dry_run:
        try:
            conn.commit()
        except Exception:
            log.error("Failed to commit: %s", traceback.format_exc())
            stats["errors"] += 1

    conn.close()
    return stats


# ===================================================================
# D. Track Record Statistics
# ===================================================================

def get_track_record(
    days: int = None,
    league: str = None,
    market: str = None,
    confidence: str = None,
    source: str = None,
    *,
    db_path: Path = DB_PATH,
) -> dict:
    """Compute track record statistics from graded predictions.

    Returns a dict with overall stats, breakdowns by confidence/market/league,
    streak info, daily performance, and ROI calculations.
    """
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

    if not rows:
        return {
            "total_graded": 0, "hits": 0, "misses": 0, "pushes": 0,
            "hit_rate": 0.0, "roi_flat_stake": 0.0,
            "by_confidence": {}, "by_market": {}, "by_league": {},
            "recent_streak": 0, "best_streak": 0,
            "daily_performance": [],
        }

    preds = [dict(r) for r in rows]

    # Overall counts
    hits = sum(1 for p in preds if p["outcome"] == "hit")
    misses = sum(1 for p in preds if p["outcome"] == "miss")
    pushes = sum(1 for p in preds if p["outcome"] == "push")
    decided = hits + misses  # pushes don't count for/against

    hit_rate = hits / decided if decided > 0 else 0.0

    # ROI: flat stake 1 unit on each pick at logged odds
    roi = _compute_roi(preds)

    # --- By Confidence ---
    by_confidence = {}
    for level in ["high", "medium", "low"]:
        subset = [p for p in preds if (p.get("confidence") or "").lower() == level]
        if subset:
            h = sum(1 for p in subset if p["outcome"] == "hit")
            m = sum(1 for p in subset if p["outcome"] == "miss")
            d = h + m
            by_confidence[level] = {
                "total": len(subset),
                "hits": h,
                "hit_rate": h / d if d > 0 else 0.0,
                "roi": _compute_roi(subset),
            }

    # --- By Market ---
    by_market = {}
    market_set = set(p.get("market", "") for p in preds)
    for mkt in sorted(market_set):
        if not mkt:
            continue
        subset = [p for p in preds if p.get("market") == mkt]
        if subset:
            h = sum(1 for p in subset if p["outcome"] == "hit")
            m = sum(1 for p in subset if p["outcome"] == "miss")
            d = h + m
            by_market[mkt] = {
                "total": len(subset),
                "hits": h,
                "hit_rate": h / d if d > 0 else 0.0,
            }

    # --- By League ---
    by_league = {}
    league_set = set(p.get("league", "") for p in preds)
    for lg in sorted(league_set):
        if not lg:
            continue
        subset = [p for p in preds if p.get("league") == lg]
        if subset:
            h = sum(1 for p in subset if p["outcome"] == "hit")
            m = sum(1 for p in subset if p["outcome"] == "miss")
            d = h + m
            by_league[lg] = {
                "total": len(subset),
                "hits": h,
                "hit_rate": h / d if d > 0 else 0.0,
            }

    # --- Streaks ---
    recent_streak, best_streak = _compute_streaks(preds)

    # --- Daily Performance (last 30 days) ---
    daily = _compute_daily_performance(preds, days=30)

    return {
        "total_graded": len(preds),
        "hits": hits,
        "misses": misses,
        "pushes": pushes,
        "hit_rate": hit_rate,
        "roi_flat_stake": roi,
        "by_confidence": by_confidence,
        "by_market": by_market,
        "by_league": by_league,
        "recent_streak": recent_streak,
        "best_streak": best_streak,
        "daily_performance": daily,
    }


def _compute_roi(preds: List[dict]) -> float:
    """Compute ROI for flat 1-unit stakes on all predictions with logged odds."""
    total_staked = 0
    total_returned = 0.0
    for p in preds:
        odds = p.get("odds")
        outcome = p.get("outcome")
        if odds is None or odds <= 1.0:
            continue
        if outcome not in ("hit", "miss", "push"):
            continue
        total_staked += 1
        if outcome == "hit":
            total_returned += odds  # win: stake returned + profit
        elif outcome == "push":
            total_returned += 1.0  # push: stake returned
        # miss: 0 returned
    if total_staked == 0:
        return 0.0
    return (total_returned - total_staked) / total_staked


def _compute_streaks(preds: List[dict]) -> Tuple[int, int]:
    """Compute current and best streak from ordered predictions.

    Positive = win streak, negative = loss streak.
    """
    if not preds:
        return 0, 0

    # Filter to only hit/miss (skip push, void)
    decided = [p for p in preds if p["outcome"] in ("hit", "miss")]
    if not decided:
        return 0, 0

    # Current streak
    current = 0
    last_outcome = decided[-1]["outcome"]
    for p in reversed(decided):
        if p["outcome"] == last_outcome:
            current += 1
        else:
            break
    if last_outcome == "miss":
        current = -current

    # Best win streak
    best_win = 0
    run = 0
    for p in decided:
        if p["outcome"] == "hit":
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
        total = len(day_preds)
        h = sum(1 for p in day_preds if p["outcome"] == "hit")
        m = sum(1 for p in day_preds if p["outcome"] == "miss")
        decided = h + m
        daily.append({
            "date": d,
            "total": total,
            "hits": h,
            "hit_rate": h / decided if decided > 0 else 0.0,
        })

    return daily


def get_recent_predictions(
    limit: int = 20,
    league: str = None,
    graded_only: bool = True,
    *,
    db_path: Path = DB_PATH,
) -> List[dict]:
    """Fetch the most recent predictions, optionally filtered."""
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
                lines.append(f"  {label:15s} {m['hits']}/{m['total']} ({m['hit_rate']:.1%})")

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
    parser.add_argument("--league", type=str, default=None,
                        help="Filter to one league")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    if args.resolve or args.resolve_date:
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

    elif args.export:
        n = export_predictions_csv(args.export, league=args.league)
        print(f"Exported {n} predictions to {args.export}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
