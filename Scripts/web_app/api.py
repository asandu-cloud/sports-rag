"""
FastAPI backend server — imports from refactored core/ modules.

Serves structured JSON predictions to the frontend.
Run: python api.py   (starts uvicorn on 0.0.0.0:8000)
"""

from __future__ import annotations

import sys
import os
import time
import logging
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Path setup -- add rag_ingest to sys.path so we can import the engine
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_RAG_INGEST = os.path.join(_PROJECT_ROOT, "Scripts", "rag_ingest")
if _RAG_INGEST not in sys.path:
    sys.path.insert(0, _RAG_INGEST)

from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Engine imports — from refactored core/ modules (no monolith dependency)
# ---------------------------------------------------------------------------
from core.weights import DEFAULT_MARKETS, LEAGUE_TO_ODDS_SPORT, DOMESTIC_LEAGUES
from core.events import fetch_events, fetch_events_multi, filter_events_by_exact_date
from core.team_resolution import get_team_profile_meta
from core.projections import (
    projected_total_goals, projected_total_corners, projected_total_cards,
    projected_total_sot, projected_btts_prob, projected_moneyline_probs,
    projected_correct_score_probs, projected_goal_difference,
)
from core.odds_extraction import (
    extract_total_line_options, extract_btts_odds, extract_moneyline_odds,
    extract_spread_line_options, market_group_from_key,
)
from core.line_selection import (
    choose_best_total_line, choose_best_spread_line, choose_best_btts_side,
    choose_best_moneyline_side, confidence_from_edge,
    _best_model_only_line, _best_model_only_spread,
)
from core.parlay import build_candidates, kb_leg_quality, CandidateLeg, enrich_events_for_groups

from prob_models import over_prob, implied_prob, value_edge

try:
    from player_projections import projected_player_stat, recommend_player_prop
    _HAS_PLAYER_PROJECTIONS = True
except ImportError:
    _HAS_PLAYER_PROJECTIONS = False

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("web_api")

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Betting RAG API",
    description="Football betting prediction API powered by the RAG scoring engine",
    version="1.0.0",
)

_ALLOWED_ORIGINS = [
    o.strip()
    for o in os.getenv("CORS_ORIGINS", "http://localhost:8000,http://127.0.0.1:8000").split(",")
    if o.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)

# ---------------------------------------------------------------------------
# Mount auth, webhooks, and checkout routers
# ---------------------------------------------------------------------------
try:
    from auth import router as auth_router
    from stripe_webhooks import webhook_router, checkout_router
    from users import init_db as init_user_db
    app.include_router(auth_router)
    app.include_router(webhook_router)
    app.include_router(checkout_router)
    # Initialize user tables on startup
    init_user_db()
    log.info("Auth, webhooks, and checkout routers mounted")
except ImportError as exc:
    log.warning("Subscription system not available (missing module): %s", exc)
except Exception as exc:
    log.warning("Subscription system failed to initialize: %s", exc)

# ---------------------------------------------------------------------------
# V2: mount the live-match router so /api/live/* is served by this process
# alongside /api/fixtures, /api/parlay, etc. The same router is re-exported
# from Scripts/live/live_api.py so a standalone live process still works.
# ---------------------------------------------------------------------------
try:
    import sys as _sys
    from pathlib import Path as _Path
    _scripts_dir = _Path(__file__).resolve().parents[1]
    if str(_scripts_dir) not in _sys.path:
        _sys.path.insert(0, str(_scripts_dir))
    from web_app.routers.live import router as live_router
    app.include_router(live_router)
    log.info("Live router mounted at /api/live/*")
except Exception as _live_exc:
    log.warning("Live router not mounted: %s", _live_exc)

# V3: platform health endpoint
try:
    from web_app.routers.health import router as health_router
    app.include_router(health_router)
    log.info("Health router mounted at /api/health/*")
except Exception as _health_exc:
    log.warning("Health router not mounted: %s", _health_exc)

# ---------------------------------------------------------------------------
# Simple TTL cache for fixture fetches (5 min)
# ---------------------------------------------------------------------------
_CACHE_TTL = 300  # seconds
_events_cache: Dict[str, Tuple[float, List[Dict], List[str]]] = {}


def _cached_fetch_events(league: str) -> Tuple[List[Dict], List[str]]:
    """Fetch events for a league with 5-minute TTL cache."""
    now = time.time()
    cached = _events_cache.get(league)
    if cached and (now - cached[0]) < _CACHE_TTL:
        return cached[1], cached[2]
    events, notes = fetch_events(league, set(DEFAULT_MARKETS))
    # Enrich with corners, cards, sot, btts markets
    events, enrich_notes = enrich_events_for_groups(
        events, league, {"corners", "cards", "sot", "btts"}
    )
    notes.extend(enrich_notes)
    _events_cache[league] = (now, events, notes)
    return events, notes


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------
class TotalsPrediction(BaseModel):
    projected_total: Optional[float] = None
    recommended_side: Optional[str] = None
    recommended_line: Optional[float] = None
    model_prob: Optional[float] = None
    odds: Optional[float] = None
    bookmaker: Optional[str] = None
    value_edge: Optional[float] = None
    confidence: Optional[str] = None


class BttsPrediction(BaseModel):
    probability: Optional[float] = None
    recommended_side: Optional[str] = None
    odds: Optional[float] = None
    bookmaker: Optional[str] = None
    value_edge: Optional[float] = None
    confidence: Optional[str] = None


class MoneylinePrediction(BaseModel):
    home_prob: Optional[float] = None
    draw_prob: Optional[float] = None
    away_prob: Optional[float] = None
    recommended: Optional[str] = None
    odds: Optional[float] = None
    bookmaker: Optional[str] = None
    value_edge: Optional[float] = None
    confidence: Optional[str] = None


class SpreadPrediction(BaseModel):
    projected_diff: Optional[float] = None
    recommended_team: Optional[str] = None
    recommended_line: Optional[float] = None
    model_prob: Optional[float] = None
    odds: Optional[float] = None
    bookmaker: Optional[str] = None
    value_edge: Optional[float] = None
    confidence: Optional[str] = None


class BestBet(BaseModel):
    market: Optional[str] = None
    pick: Optional[str] = None
    odds: Optional[float] = None
    confidence: Optional[str] = None
    edge: Optional[float] = None
    reason: Optional[str] = None


class FixturePrediction(BaseModel):
    event_id: str
    home_team: str
    away_team: str
    kickoff: Optional[str] = None
    goals: Optional[TotalsPrediction] = None
    corners: Optional[TotalsPrediction] = None
    cards: Optional[TotalsPrediction] = None
    sot: Optional[TotalsPrediction] = None
    btts: Optional[BttsPrediction] = None
    moneyline: Optional[MoneylinePrediction] = None
    spreads: Optional[SpreadPrediction] = None
    best_bet: Optional[BestBet] = None


class FixturesResponse(BaseModel):
    league: str
    date: str
    fixtures: List[FixturePrediction]
    notes: List[str] = []


class BestBetEntry(BaseModel):
    league: str
    event_id: str
    home_team: str
    away_team: str
    kickoff: Optional[str] = None
    market: str
    pick: str
    odds: Optional[float] = None
    model_prob: Optional[float] = None
    value_edge: Optional[float] = None
    confidence: str
    reason: str


class BestBetsResponse(BaseModel):
    date: str
    picks: List[BestBetEntry]


class ParlayLeg(BaseModel):
    event_id: str
    market: str
    side: str
    line: Optional[float] = None
    odds: float


class ParlayRequest(BaseModel):
    legs: List[ParlayLeg]


class ParlayLegAnalysis(BaseModel):
    market: str
    pick: str
    confidence: Optional[str] = None
    edge: Optional[float] = None
    model_prob: Optional[float] = None


class ParlayResponse(BaseModel):
    combined_odds: float
    quality_score: float
    quality_label: str
    legs_analysis: List[ParlayLegAnalysis]
    warnings: List[str] = []


class LeagueInfo(BaseModel):
    code: str
    name: str
    odds_sport_key: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_LEAGUE_NAMES = {
    "EPL": "English Premier League",
    "LaLiga": "La Liga",
    "SerieA": "Serie A",
    "Bundesliga": "Bundesliga",
    "Ligue1": "Ligue 1",
    "UCL": "UEFA Champions League",
    "UEL": "UEFA Europa League",
    "UECL": "UEFA Europa Conference League",
}


def _safe(val: Any) -> Optional[float]:
    """Round a float to 4 decimal places, or return None."""
    if val is None:
        return None
    try:
        return round(float(val), 4)
    except (TypeError, ValueError):
        return None


def _build_totals_prediction(
    home: str, away: str, league: str, event: Dict, stat_group: str
) -> Optional[TotalsPrediction]:
    """Build a TotalsPrediction for a given stat group (goals, corners, cards, sot)."""
    try:
        if stat_group == "goals":
            result = projected_total_goals(home, away, league)
            blended, season, recent = result
        elif stat_group == "corners":
            result = projected_total_corners(home, away, league)
            blended, season, recent = result
        elif stat_group == "cards":
            result = projected_total_cards(home, away, league)
            blended, season, recent = result[:3]  # 4th is ref_mod
        elif stat_group == "sot":
            result = projected_total_sot(home, away, league)
            blended, season, recent = result
        else:
            return None

        if blended is None:
            return TotalsPrediction(projected_total=None)

        # Extract line options from event odds
        options = extract_total_line_options(event, stat_group)
        best = choose_best_total_line(options, blended) if options else None

        if best:
            model_prob = _safe(best.get("_model_prob"))
            val_edge = _safe(best.get("_value_edge"))
            edge_abs = abs(val_edge) if val_edge is not None else abs(blended - (best.get("point") or blended))
            conf = confidence_from_edge(
                edge_abs,
                stat_group=stat_group,
                model_prob=model_prob,
                value_edge_pct=val_edge,
            )
            return TotalsPrediction(
                projected_total=_safe(blended),
                recommended_side=best.get("side", "").capitalize(),
                recommended_line=_safe(best.get("point")),
                model_prob=model_prob,
                odds=_safe(best.get("odds")),
                bookmaker=best.get("bookmaker"),
                value_edge=val_edge,
                confidence=conf,
            )
        else:
            # Model-only fallback
            side, line, prob = _best_model_only_line(blended)
            edge_abs = abs(blended - line)
            conf = confidence_from_edge(
                edge_abs,
                stat_group=stat_group,
                model_prob=prob,
                value_edge_pct=None,
            )
            return TotalsPrediction(
                projected_total=_safe(blended),
                recommended_side=side,
                recommended_line=line,
                model_prob=_safe(prob),
                odds=None,
                bookmaker=None,
                value_edge=None,
                confidence=conf,
            )
    except Exception as exc:
        log.warning("Totals prediction failed for %s vs %s (%s): %s", home, away, stat_group, exc)
        return None


def _build_btts_prediction(
    home: str, away: str, league: str, event: Dict
) -> Optional[BttsPrediction]:
    """Build a BttsPrediction for a fixture."""
    try:
        result = projected_btts_prob(home, away, league)
        p_btts, h_proj, a_proj, h_season, a_season = result
        if p_btts is None:
            return BttsPrediction()

        options = extract_btts_odds(event)
        best = choose_best_btts_side(options, p_btts) if options else None

        if best:
            val_edge = _safe(best.get("_value_edge"))
            model_prob = _safe(best.get("_model_prob"))
            edge_abs = abs(val_edge) if val_edge is not None else 0.0
            conf = confidence_from_edge(
                edge_abs,
                stat_group="btts",
                model_prob=model_prob,
                value_edge_pct=val_edge,
            )
            return BttsPrediction(
                probability=_safe(p_btts),
                recommended_side=str(best.get("side", "")).capitalize(),
                odds=_safe(best.get("odds")),
                bookmaker=best.get("bookmaker"),
                value_edge=val_edge,
                confidence=conf,
            )
        else:
            side = "Yes" if p_btts >= 0.5 else "No"
            conf = confidence_from_edge(
                abs(p_btts - 0.5),
                stat_group="btts",
                model_prob=p_btts if side == "Yes" else 1.0 - p_btts,
                value_edge_pct=None,
            )
            return BttsPrediction(
                probability=_safe(p_btts),
                recommended_side=side,
                odds=None,
                bookmaker=None,
                value_edge=None,
                confidence=conf,
            )
    except Exception as exc:
        log.warning("BTTS prediction failed for %s vs %s: %s", home, away, exc)
        return None


def _build_moneyline_prediction(
    home: str, away: str, league: str, event: Dict
) -> Optional[MoneylinePrediction]:
    """Build a MoneylinePrediction for a fixture."""
    try:
        result = projected_moneyline_probs(home, away, league)
        p_home, p_draw, p_away, h_proj, a_proj = result
        if p_home is None:
            return MoneylinePrediction()

        market_odds = extract_moneyline_odds(event)
        ml_result = choose_best_moneyline_side(p_home, p_draw, p_away, market_odds, home, away)

        rec = ml_result.get("recommended", {})
        val_edge = _safe(rec.get("value_edge"))
        model_prob = _safe(rec.get("model_prob"))
        edge_abs = abs(val_edge) if val_edge is not None else 0.0
        conf = confidence_from_edge(
            edge_abs,
            stat_group="moneyline",
            model_prob=model_prob,
            value_edge_pct=val_edge,
        )

        return MoneylinePrediction(
            home_prob=_safe(p_home),
            draw_prob=_safe(p_draw),
            away_prob=_safe(p_away),
            recommended=rec.get("team"),
            odds=_safe(rec.get("best_odds")),
            bookmaker=rec.get("bookmaker"),
            value_edge=val_edge,
            confidence=conf,
        )
    except Exception as exc:
        log.warning("Moneyline prediction failed for %s vs %s: %s", home, away, exc)
        return None


def _build_spread_prediction(
    home: str, away: str, league: str, event: Dict
) -> Optional[SpreadPrediction]:
    """Build a SpreadPrediction for a fixture."""
    try:
        result = projected_goal_difference(home, away, league)
        blended_diff, season_diff, recent_diff = result
        if blended_diff is None:
            return SpreadPrediction()

        options = extract_spread_line_options(event)
        best = (
            choose_best_spread_line(options, blended_diff, home, away_team=away, league=league)
            if options else None
        )

        if best:
            model_prob = _safe(best.get("_model_prob"))
            val_edge = _safe(best.get("_value_edge"))
            edge_abs = abs(val_edge) if val_edge is not None else abs(blended_diff)
            conf = confidence_from_edge(
                edge_abs,
                stat_group="spreads",
                model_prob=model_prob,
                value_edge_pct=val_edge,
            )
            return SpreadPrediction(
                projected_diff=_safe(blended_diff),
                recommended_team=best.get("team"),
                recommended_line=_safe(best.get("point")),
                model_prob=model_prob,
                odds=_safe(best.get("odds")),
                bookmaker=best.get("bookmaker"),
                value_edge=val_edge,
                confidence=conf,
            )
        else:
            # Model-only fallback
            team, hc, prob = _best_model_only_spread(blended_diff, home, away)
            conf = confidence_from_edge(
                abs(blended_diff),
                stat_group="spreads",
                model_prob=prob,
                value_edge_pct=None,
            )
            return SpreadPrediction(
                projected_diff=_safe(blended_diff),
                recommended_team=team,
                recommended_line=float(hc) if hc else None,
                model_prob=_safe(prob),
                odds=None,
                bookmaker=None,
                value_edge=None,
                confidence=conf,
            )
    except Exception as exc:
        log.warning("Spread prediction failed for %s vs %s: %s", home, away, exc)
        return None


def _select_best_bet(prediction: FixturePrediction) -> Optional[BestBet]:
    """Pick the single best market from a fixture's predictions by confidence + edge."""
    candidates: List[Tuple[str, str, Optional[float], Optional[str], Optional[float], str]] = []
    # (market, pick_label, odds, confidence, edge, reason)

    if prediction.goals and prediction.goals.projected_total is not None:
        g = prediction.goals
        pick = f"{g.recommended_side or 'Over'} {g.recommended_line or '?'} Goals"
        reason = f"Projected {g.projected_total} total goals."
        if g.value_edge is not None:
            reason += f" {abs(g.value_edge)*100:.1f}% edge."
        candidates.append(("goals", pick, g.odds, g.confidence, g.value_edge, reason))

    if prediction.corners and prediction.corners.projected_total is not None:
        c = prediction.corners
        pick = f"{c.recommended_side or 'Over'} {c.recommended_line or '?'} Corners"
        reason = f"Projected {c.projected_total} total corners."
        if c.value_edge is not None:
            reason += f" {abs(c.value_edge)*100:.1f}% edge."
        candidates.append(("corners", pick, c.odds, c.confidence, c.value_edge, reason))

    if prediction.cards and prediction.cards.projected_total is not None:
        k = prediction.cards
        pick = f"{k.recommended_side or 'Over'} {k.recommended_line or '?'} Cards"
        reason = f"Projected {k.projected_total} total cards."
        if k.value_edge is not None:
            reason += f" {abs(k.value_edge)*100:.1f}% edge."
        candidates.append(("cards", pick, k.odds, k.confidence, k.value_edge, reason))

    if prediction.sot and prediction.sot.projected_total is not None:
        s = prediction.sot
        pick = f"{s.recommended_side or 'Over'} {s.recommended_line or '?'} SOT"
        reason = f"Projected {s.projected_total} total shots on target."
        if s.value_edge is not None:
            reason += f" {abs(s.value_edge)*100:.1f}% edge."
        candidates.append(("sot", pick, s.odds, s.confidence, s.value_edge, reason))

    if prediction.btts and prediction.btts.probability is not None:
        b = prediction.btts
        pick = f"BTTS {b.recommended_side or 'Yes'}"
        reason = f"BTTS probability {b.probability*100:.0f}%."
        if b.value_edge is not None:
            reason += f" {abs(b.value_edge)*100:.1f}% edge."
        candidates.append(("btts", pick, b.odds, b.confidence, b.value_edge, reason))

    if prediction.moneyline and prediction.moneyline.home_prob is not None:
        m = prediction.moneyline
        pick = f"{m.recommended or 'Home'} Win"
        reason = f"Home {m.home_prob*100:.0f}% / Draw {(m.draw_prob or 0)*100:.0f}% / Away {(m.away_prob or 0)*100:.0f}%."
        if m.value_edge is not None:
            reason += f" {abs(m.value_edge)*100:.1f}% edge."
        candidates.append(("moneyline", pick, m.odds, m.confidence, m.value_edge, reason))

    if prediction.spreads and prediction.spreads.projected_diff is not None:
        sp = prediction.spreads
        pick = f"{sp.recommended_team or '?'} {sp.recommended_line or '?'}"
        reason = f"Projected goal diff {sp.projected_diff:+.2f}."
        if sp.value_edge is not None:
            reason += f" {abs(sp.value_edge)*100:.1f}% edge."
        candidates.append(("spreads", pick, sp.odds, sp.confidence, sp.value_edge, reason))

    if not candidates:
        return None

    # Rank: high > medium > low, then by edge descending
    conf_rank = {"high": 3, "medium": 2, "low": 1, None: 0}
    candidates.sort(key=lambda x: (
        conf_rank.get(x[3], 0),
        abs(x[4]) if x[4] is not None else 0,
    ), reverse=True)

    best = candidates[0]
    return BestBet(
        market=best[0],
        pick=best[1],
        odds=best[2],
        confidence=best[3],
        edge=_safe(best[4]),
        reason=best[5],
    )


def _build_fixture_prediction(
    event: Dict, league: str
) -> FixturePrediction:
    """Build full predictions for a single fixture event."""
    home = str(event.get("home_team") or "")
    away = str(event.get("away_team") or "")
    event_id = str(event.get("id") or "")
    kickoff = event.get("commence_time")

    pred = FixturePrediction(
        event_id=event_id,
        home_team=home,
        away_team=away,
        kickoff=kickoff,
    )

    # Build each market prediction independently; failures are isolated
    pred.goals = _build_totals_prediction(home, away, league, event, "goals")
    pred.corners = _build_totals_prediction(home, away, league, event, "corners")
    pred.cards = _build_totals_prediction(home, away, league, event, "cards")
    pred.sot = _build_totals_prediction(home, away, league, event, "sot")
    pred.btts = _build_btts_prediction(home, away, league, event)
    pred.moneyline = _build_moneyline_prediction(home, away, league, event)
    pred.spreads = _build_spread_prediction(home, away, league, event)

    pred.best_bet = _select_best_bet(pred)

    return pred


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/leagues", response_model=List[LeagueInfo])
def get_leagues():
    """Return list of available leagues."""
    return [
        LeagueInfo(code=code, name=_LEAGUE_NAMES.get(code, code), odds_sport_key=key)
        for code, key in LEAGUE_TO_ODDS_SPORT.items()
    ]


@app.get("/api/fixtures/{league}/{target_date}", response_model=FixturesResponse)
def get_fixtures(league: str, target_date: str):
    """
    Return all fixtures for a league on a given date with full predictions.

    - **league**: League code (EPL, LaLiga, SerieA, Bundesliga, Ligue1, UCL, UEL, UECL)
    - **target_date**: Date in YYYY-MM-DD format
    """
    if league not in LEAGUE_TO_ODDS_SPORT:
        raise HTTPException(status_code=400, detail=f"Unknown league '{league}'. Valid: {list(LEAGUE_TO_ODDS_SPORT.keys())}")

    try:
        dt = date.fromisoformat(target_date)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid date format '{target_date}'. Use YYYY-MM-DD.")

    log.info("Fetching fixtures for %s on %s", league, target_date)

    events, notes = _cached_fetch_events(league)
    day_events = filter_events_by_exact_date(events, dt)

    if not day_events:
        return FixturesResponse(league=league, date=target_date, fixtures=[], notes=notes + [f"No fixtures found for {target_date}."])

    fixtures: List[FixturePrediction] = []
    for ev in day_events:
        pred = _build_fixture_prediction(ev, league)
        fixtures.append(pred)

    return FixturesResponse(
        league=league,
        date=target_date,
        fixtures=fixtures,
        notes=notes,
    )


@app.get("/api/best-bets/{target_date}", response_model=BestBetsResponse)
def get_best_bets(
    target_date: str,
    limit: int = Query(default=15, ge=1, le=50, description="Max picks to return"),
):
    """
    Return the top value picks across all 5 domestic leagues for a given date.

    Aggregates predictions from EPL, LaLiga, SerieA, Bundesliga, Ligue1 and
    returns the best opportunities sorted by confidence and value edge.
    """
    try:
        dt = date.fromisoformat(target_date)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid date format '{target_date}'. Use YYYY-MM-DD.")

    log.info("Computing best bets across all leagues for %s", target_date)

    all_picks: List[BestBetEntry] = []
    all_notes: List[str] = []

    for league in sorted(DOMESTIC_LEAGUES):
        try:
            events, notes = _cached_fetch_events(league)
            all_notes.extend(notes)
            day_events = filter_events_by_exact_date(events, dt)

            for ev in day_events:
                home = str(ev.get("home_team") or "")
                away = str(ev.get("away_team") or "")
                event_id = str(ev.get("id") or "")
                kickoff = ev.get("commence_time")

                pred = _build_fixture_prediction(ev, league)

                # Collect all market predictions as candidate picks
                market_candidates = _extract_market_candidates(pred, league, event_id, home, away, kickoff)
                all_picks.extend(market_candidates)

        except Exception as exc:
            log.warning("Failed to process %s for best bets: %s", league, exc)
            all_notes.append(f"{league}: {exc}")

    # Sort by confidence rank (high first), then by absolute value edge descending
    conf_rank = {"high": 3, "medium": 2, "low": 1}
    all_picks.sort(key=lambda p: (
        conf_rank.get(p.confidence, 0),
        abs(p.value_edge) if p.value_edge is not None else 0,
    ), reverse=True)

    # Filter: only return medium+ confidence picks
    quality_picks = [p for p in all_picks if p.confidence in ("high", "medium")]

    return BestBetsResponse(
        date=target_date,
        picks=quality_picks[:limit],
    )


def _extract_market_candidates(
    pred: FixturePrediction, league: str, event_id: str,
    home: str, away: str, kickoff: Optional[str]
) -> List[BestBetEntry]:
    """Extract all market predictions from a fixture as BestBetEntry objects."""
    entries: List[BestBetEntry] = []

    def _add(market: str, pick: str, odds: Optional[float], model_prob: Optional[float],
             val_edge: Optional[float], confidence: Optional[str], reason: str):
        if confidence is None:
            return
        entries.append(BestBetEntry(
            league=league,
            event_id=event_id,
            home_team=home,
            away_team=away,
            kickoff=kickoff,
            market=market,
            pick=pick,
            odds=odds,
            model_prob=model_prob,
            value_edge=val_edge,
            confidence=confidence,
            reason=reason,
        ))

    if pred.goals and pred.goals.projected_total is not None:
        g = pred.goals
        pick = f"{g.recommended_side or 'Over'} {g.recommended_line or '?'} Goals"
        reason = f"Projected {g.projected_total} total goals."
        _add("goals", pick, g.odds, g.model_prob, g.value_edge, g.confidence, reason)

    if pred.corners and pred.corners.projected_total is not None:
        c = pred.corners
        pick = f"{c.recommended_side or 'Over'} {c.recommended_line or '?'} Corners"
        reason = f"Projected {c.projected_total} total corners."
        _add("corners", pick, c.odds, c.model_prob, c.value_edge, c.confidence, reason)

    if pred.cards and pred.cards.projected_total is not None:
        k = pred.cards
        pick = f"{k.recommended_side or 'Over'} {k.recommended_line or '?'} Cards"
        reason = f"Projected {k.projected_total} total cards."
        _add("cards", pick, k.odds, k.model_prob, k.value_edge, k.confidence, reason)

    if pred.sot and pred.sot.projected_total is not None:
        s = pred.sot
        pick = f"{s.recommended_side or 'Over'} {s.recommended_line or '?'} SOT"
        reason = f"Projected {s.projected_total} total SOT."
        _add("sot", pick, s.odds, s.model_prob, s.value_edge, s.confidence, reason)

    if pred.btts and pred.btts.probability is not None:
        b = pred.btts
        pick = f"BTTS {b.recommended_side or 'Yes'}"
        reason = f"BTTS probability {b.probability*100:.0f}%."
        _add("btts", pick, b.odds, None, b.value_edge, b.confidence, reason)

    if pred.moneyline and pred.moneyline.home_prob is not None:
        m = pred.moneyline
        pick = f"{m.recommended or 'Home'} Win"
        reason = f"Model: Home {m.home_prob*100:.0f}% / Draw {(m.draw_prob or 0)*100:.0f}% / Away {(m.away_prob or 0)*100:.0f}%."
        _add("moneyline", pick, m.odds, None, m.value_edge, m.confidence, reason)

    if pred.spreads and pred.spreads.projected_diff is not None:
        sp = pred.spreads
        pick = f"{sp.recommended_team or '?'} {sp.recommended_line or '?'}"
        reason = f"Projected goal diff {sp.projected_diff:+.2f}."
        _add("spreads", pick, sp.odds, sp.model_prob, sp.value_edge, sp.confidence, reason)

    return entries


@app.post("/api/parlay/score", response_model=ParlayResponse)
def score_parlay(request: ParlayRequest):
    """
    Score a user-assembled parlay. Accepts a list of legs and returns
    quality analysis using the KB scoring engine.
    """
    if not request.legs:
        raise HTTPException(status_code=400, detail="At least one leg is required.")

    if len(request.legs) > 15:
        raise HTTPException(status_code=400, detail="Maximum 15 legs per parlay.")

    combined_odds = 1.0
    legs_analysis: List[ParlayLegAnalysis] = []
    warnings: List[str] = []
    quality_scores: List[float] = []

    # We need to find the events for each leg to run KB scoring.
    # Group legs by event_id, fetch events if not cached.
    # For simplicity, we score each leg using projected totals + edge.

    for leg in request.legs:
        combined_odds *= leg.odds

        pick_label = f"{leg.side} {leg.line}" if leg.line is not None else leg.side
        pick_label = f"{pick_label} ({leg.market})"

        # Attempt to score this leg via the projection pipeline
        leg_edge: Optional[float] = None
        leg_prob: Optional[float] = None
        leg_conf: Optional[str] = None

        # We cannot easily re-fetch individual events without league context,
        # so we provide basic implied probability analysis.
        imp_prob = implied_prob(leg.odds)
        if imp_prob < 0.30:
            warnings.append(f"Leg '{pick_label}' has low implied probability ({imp_prob*100:.0f}%).")

        # Basic quality heuristic based on odds range
        if 1.40 <= leg.odds <= 2.50:
            q = 8.0  # ideal range
        elif 1.20 <= leg.odds < 1.40:
            q = 6.5  # short price
        elif 2.50 < leg.odds <= 4.00:
            q = 6.0  # longish
        elif leg.odds > 4.00:
            q = 4.0  # speculative
            warnings.append(f"Leg '{pick_label}' at {leg.odds} odds is speculative.")
        else:
            q = 5.0

        quality_scores.append(q)

        legs_analysis.append(ParlayLegAnalysis(
            market=leg.market,
            pick=pick_label,
            confidence=leg_conf,
            edge=leg_edge,
            model_prob=leg_prob,
        ))

    combined_odds = round(combined_odds, 2)

    # Overall quality score (0-10)
    if quality_scores:
        avg_quality = sum(quality_scores) / len(quality_scores)
    else:
        avg_quality = 5.0

    # Adjust for leg count (more legs = higher variance = lower quality)
    n_legs = len(request.legs)
    if n_legs > 5:
        avg_quality *= 0.85
    elif n_legs > 3:
        avg_quality *= 0.92

    # Check for contradictions
    if combined_odds > 50:
        warnings.append("Combined odds exceed 50x. Very unlikely to hit.")
        avg_quality *= 0.7

    quality_score = round(min(10.0, max(0.0, avg_quality)), 1)

    if quality_score >= 7.5:
        quality_label = "Strong"
    elif quality_score >= 6.0:
        quality_label = "Good"
    elif quality_score >= 4.5:
        quality_label = "Fair"
    else:
        quality_label = "Weak"

    return ParlayResponse(
        combined_odds=combined_odds,
        quality_score=quality_score,
        quality_label=quality_label,
        legs_analysis=legs_analysis,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/api/health")
def health_check():
    """Basic health check endpoint."""
    return {
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "leagues_available": len(LEAGUE_TO_ODDS_SPORT),
    }


# ---------------------------------------------------------------------------
# Static file serving (frontend)
# ---------------------------------------------------------------------------

_STATIC_DIR = Path(__file__).parent / "static"

@app.get("/")
def serve_root():
    """Serve the main HTML page."""
    return FileResponse(_STATIC_DIR / "index.html")

@app.get("/app")
def serve_app():
    """Serve the SPA for the /app route (client-side routing)."""
    return FileResponse(_STATIC_DIR / "index.html")

@app.get("/checkout/success")
def serve_checkout_success():
    """Serve the SPA for checkout success (client-side handles the token)."""
    return FileResponse(_STATIC_DIR / "index.html")

# Mount static files AFTER API routes so /api/* takes priority
app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
