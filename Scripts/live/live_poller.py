"""Live match polling system for API-Football.

Asyncio-based poller with two loops:
  Discovery (every 60s) — scans for new live fixtures across tracked leagues.
  Detail   (every 45s) — fetches full match state for subscribed fixtures.

After each detail poll, triggers the live projection engine (compute_snapshot)
and stores the result for API consumers.

Graceful degradation: if the live engine is not available, snapshots are None
and the poller continues collecting raw match state.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import threading
from contextlib import suppress
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Paths — ensure rag_ingest is importable for pre-match priors
# ---------------------------------------------------------------------------
_SCRIPTS_DIR = Path(__file__).resolve().parents[1]
_RAG_INGEST_DIR = _SCRIPTS_DIR / "rag_ingest"
if str(_RAG_INGEST_DIR) not in sys.path:
    sys.path.insert(0, str(_RAG_INGEST_DIR))
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

# ---------------------------------------------------------------------------
# Import live_config constants
# ---------------------------------------------------------------------------
from live.live_config import (
    ALERT_COOLDOWN_SECONDS,
    DETAIL_POLL_INTERVAL,
    DISCOVERY_POLL_INTERVAL,
)

# ---------------------------------------------------------------------------
# Optional: live engine (being built separately)
# ---------------------------------------------------------------------------
_HAS_LIVE_ENGINE = False
compute_snapshot: Optional[Callable] = None

try:
    from live.live_engine import compute_snapshot as _compute  # type: ignore[import]
    compute_snapshot = _compute
    _HAS_LIVE_ENGINE = True
except ImportError:
    pass

_HAS_ALERTS = False
detect_alerts: Optional[Callable] = None
try:
    from live.live_alerts import detect_alerts as _detect_alerts  # type: ignore[import]
    detect_alerts = _detect_alerts
    _HAS_ALERTS = True
except ImportError:
    pass

_HAS_LIVE_ODDS = False
fetch_live_fixture_odds: Optional[Callable] = None
try:
    from live.live_odds import fetch_live_fixture_odds as _fetch_live_fixture_odds  # type: ignore[import]
    fetch_live_fixture_odds = _fetch_live_fixture_odds
    _HAS_LIVE_ODDS = True
except Exception as _live_odds_import_exc:  # catch all — transitive imports can raise non-ImportError
    # logger isn't defined yet at this point in the module; defer via logging.getLogger
    logging.getLogger("live_poller").error(
        "Live odds module failed to import — live odds disabled: %s",
        _live_odds_import_exc,
        exc_info=True,
    )

# ---------------------------------------------------------------------------
# Optional: pre-match projections from rag_cli_v2
# ---------------------------------------------------------------------------
_HAS_RAG = False
try:
    from rag_cli_v2 import (
        _profile_meta,
        projected_btts_prob,
        projected_goals,
        projected_moneyline_probs,
        projected_total_cards,
        projected_total_corners,
        projected_total_goals,
        projected_total_sot,
    )
    _HAS_RAG = True
except Exception:
    pass

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger("live_poller")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(name)s] %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    ))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
API_FOOTBALL_BASE = "https://v3.football.api-sports.io"

LEAGUE_ID_TO_NAME: Dict[int, str] = {
    39: "EPL",
    140: "LaLiga",
    135: "SerieA",
    78: "Bundesliga",
    61: "Ligue1",
    2: "UCL",
    3: "UEL",
    848: "UECL",
}
TRACKED_LEAGUE_IDS: Set[int] = set(LEAGUE_ID_TO_NAME.keys())

SNAPSHOT_HISTORY_SIZE = 20

# Match statuses considered "live"
LIVE_STATUSES: Set[str] = {"1H", "HT", "2H", "ET", "BT", "P", "SUSP", "INT", "LIVE"}
FINISHED_STATUSES: Set[str] = {"FT", "AET", "PEN"}


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

class EventType(str, Enum):
    GOAL = "goal"
    YELLOW_CARD = "yellow_card"
    RED_CARD = "red_card"
    SUBSTITUTION = "substitution"
    VAR = "var"
    PENALTY_MISSED = "penalty_missed"


@dataclass
class MatchEvent:
    """A single in-match event (goal, card, sub, etc.)."""
    minute: int
    event_type: str          # EventType value
    team: str                # team name
    player: Optional[str] = None
    assist: Optional[str] = None
    detail: Optional[str] = None


@dataclass
class TeamLiveStats:
    """Per-team live statistics snapshot."""
    goals: int = 0
    corners: int = 0
    yellow_cards: int = 0
    red_cards: int = 0
    sot: int = 0
    shots: int = 0
    fouls: int = 0
    possession: float = 50.0


@dataclass
class LiveMatchState:
    """Full state of a single live match, updated by the poller."""
    fixture_id: int
    league: str                          # e.g. "EPL"
    league_id: int
    home_team: str
    away_team: str
    status: str = ""                     # 1H, HT, 2H, FT, etc.
    elapsed_min: int = 0
    home_stats: TeamLiveStats = field(default_factory=TeamLiveStats)
    away_stats: TeamLiveStats = field(default_factory=TeamLiveStats)
    events: List[MatchEvent] = field(default_factory=list)
    kickoff_utc: Optional[str] = None    # ISO timestamp

    # Pre-match priors (frozen at detection time)
    prior_total_goals: Optional[float] = None
    prior_total_corners: Optional[float] = None
    prior_total_cards: Optional[float] = None
    prior_total_sot: Optional[float] = None
    prior_home_goals: Optional[float] = None
    prior_away_goals: Optional[float] = None
    prior_btts_prob: Optional[float] = None
    prior_moneyline: Optional[Dict[str, float]] = None
    live_odds: Dict[str, Any] = field(default_factory=dict)
    live_odds_updated_at: Optional[str] = None
    live_odds_source: Optional[str] = None

    # Metadata
    last_updated: Optional[str] = None   # ISO timestamp of last detail poll

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d


@dataclass
class LiveSnapshot:
    """Projection snapshot produced by the live engine.

    This is a placeholder — the real class will be defined in live_engine.py.
    We define a minimal version here so the poller can function independently.
    """
    fixture_id: int
    timestamp: str                       # ISO UTC
    elapsed_min: int = 0
    projected_total_goals: Optional[float] = None
    projected_total_corners: Optional[float] = None
    projected_total_cards: Optional[float] = None
    projected_total_sot: Optional[float] = None
    alerts: List[Dict[str, Any]] = field(default_factory=list)
    live_odds: Dict[str, Any] = field(default_factory=dict)
    live_odds_source: Optional[str] = None
    recommendation_mode: Optional[str] = None
    recommendation_pricing_source: Optional[str] = None
    has_priced_recommendation: bool = False
    momentum_home: float = 0.0
    momentum_away: float = 0.0
    raw: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# API-Football HTTP helpers
# ---------------------------------------------------------------------------

def _api_football_key() -> Optional[str]:
    """Read API-Football key from environment (matches referee_data.py pattern)."""
    for k in ("API-FOOTBALL-KEY", "API_FOOTBALL_KEY"):
        v = os.environ.get(k)
        if v and str(v).strip():
            return str(v).strip().strip("'\"")
    return None


async def _async_api_get(
    endpoint: str,
    params: Dict[str, Any],
    session: Optional[Any] = None,
) -> Optional[list]:
    """Async HTTP GET to API-Football. Returns the 'response' list or None.

    Uses aiohttp if available; falls back to requests in a thread.
    """
    key = _api_football_key()
    if not key:
        logger.warning("API-FOOTBALL-KEY not set — live polling disabled")
        return None

    url = f"{API_FOOTBALL_BASE}{endpoint}"
    headers = {"x-apisports-key": key}

    # Try aiohttp first
    try:
        import aiohttp as _aiohttp  # noqa: F811

        if session is not None:
            resp = await session.get(url, headers=headers, params=params, timeout=_aiohttp.ClientTimeout(total=25))
            if resp.status != 200:
                text = await resp.text()
                logger.error("API-Football HTTP %d: %s", resp.status, text[:200])
                return None
            data = await resp.json()
            return data.get("response", [])
        else:
            async with _aiohttp.ClientSession() as s:
                resp = await s.get(url, headers=headers, params=params, timeout=_aiohttp.ClientTimeout(total=25))
                if resp.status != 200:
                    text = await resp.text()
                    logger.error("API-Football HTTP %d: %s", resp.status, text[:200])
                    return None
                data = await resp.json()
                return data.get("response", [])
    except ImportError:
        pass

    # Fallback: requests in a thread executor
    import requests as _requests

    def _sync_get():
        try:
            r = _requests.get(url, headers=headers, params=params, timeout=25)
            if r.status_code != 200:
                logger.error("API-Football HTTP %d: %s", r.status_code, (r.text or "")[:200])
                return None
            return r.json().get("response", [])
        except Exception as exc:
            logger.error("API-Football request failed: %s", exc)
            return None

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _sync_get)


# ---------------------------------------------------------------------------
# Response parsers
# ---------------------------------------------------------------------------

def _parse_stat_value(stat_list: list, stat_type: str) -> Any:
    """Extract a single stat value from API-Football statistics array."""
    for item in stat_list:
        if item.get("type") == stat_type:
            return item.get("value")
    return None


def _parse_possession(raw: Any) -> float:
    """Parse possession value: '58%' -> 58.0, 58 -> 58.0, None -> 50.0."""
    if raw is None:
        return 50.0
    if isinstance(raw, str):
        raw = raw.replace("%", "").strip()
    try:
        return float(raw)
    except (ValueError, TypeError):
        return 50.0


def _parse_int_stat(raw: Any) -> int:
    """Safely parse an integer stat value (None -> 0)."""
    if raw is None:
        return 0
    try:
        return int(raw)
    except (ValueError, TypeError):
        return 0


def _parse_team_stats(statistics: list) -> TeamLiveStats:
    """Parse API-Football statistics array into TeamLiveStats."""
    return TeamLiveStats(
        corners=_parse_int_stat(_parse_stat_value(statistics, "Corner Kicks")),
        yellow_cards=_parse_int_stat(_parse_stat_value(statistics, "Yellow Cards")),
        red_cards=_parse_int_stat(_parse_stat_value(statistics, "Red Cards")),
        sot=_parse_int_stat(_parse_stat_value(statistics, "Shots on Goal")),
        shots=_parse_int_stat(_parse_stat_value(statistics, "Total Shots")),
        fouls=_parse_int_stat(_parse_stat_value(statistics, "Fouls")),
        possession=_parse_possession(_parse_stat_value(statistics, "Ball Possession")),
    )


def _card_counts_from_events(
    events: List[MatchEvent],
    home_team: str,
    away_team: str,
) -> Tuple[int, int, int, int]:
    """Count yellow/red cards per team from the events list.

    API-Football's ``statistics`` array (Yellow Cards / Red Cards) lags match
    events by several minutes because stats are aggregated after the fact while
    events post in near real-time. We use events as the authoritative floor for
    card counts so the live engine doesn't recommend Under lines before the
    aggregated stats catch up.

    Returns ``(home_yellow, home_red, away_yellow, away_red)``.
    """
    hy = hr = ay = ar = 0
    home_lower = (home_team or "").lower().strip()
    away_lower = (away_team or "").lower().strip()
    for ev in events:
        if ev.event_type not in (EventType.YELLOW_CARD.value, EventType.RED_CARD.value):
            continue
        team_lower = (ev.team or "").lower().strip()
        is_home = team_lower == home_lower or (home_lower and team_lower in home_lower)
        is_away = team_lower == away_lower or (away_lower and team_lower in away_lower)
        if not is_home and not is_away:
            continue
        if ev.event_type == EventType.YELLOW_CARD.value:
            if is_home:
                hy += 1
            else:
                ay += 1
        else:  # RED_CARD
            if is_home:
                hr += 1
            else:
                ar += 1
    return hy, hr, ay, ar


def _parse_events(events_raw: list) -> List[MatchEvent]:
    """Parse API-Football events array into MatchEvent list."""
    parsed: List[MatchEvent] = []
    for ev in events_raw:
        ev_type_raw = ev.get("type", "")
        detail = ev.get("detail", "")
        team_info = ev.get("team", {})
        player_info = ev.get("player", {})
        assist_info = ev.get("assist", {})
        minute = ev.get("time", {}).get("elapsed", 0) or 0

        if ev_type_raw == "Goal":
            event_type = EventType.GOAL.value
        elif ev_type_raw == "Card":
            if "Red" in detail or "Second Yellow" in detail.replace(" ", ""):
                event_type = EventType.RED_CARD.value
            else:
                event_type = EventType.YELLOW_CARD.value
        elif ev_type_raw == "subst":
            event_type = EventType.SUBSTITUTION.value
        elif ev_type_raw == "Var":
            event_type = EventType.VAR.value
        else:
            continue  # skip unknown event types

        parsed.append(MatchEvent(
            minute=minute,
            event_type=event_type,
            team=team_info.get("name", ""),
            player=player_info.get("name"),
            assist=assist_info.get("name") if assist_info else None,
            detail=detail or None,
        ))

    return parsed


# ---------------------------------------------------------------------------
# Pre-match prior computation
# ---------------------------------------------------------------------------

def _compute_priors(home: str, away: str, league: str, state: LiveMatchState) -> None:
    """Populate pre-match prior fields on a LiveMatchState.

    Runs synchronously — called once when a fixture is first detected.
    Failures are logged and leave prior fields as None (graceful degradation).
    """
    if not _HAS_RAG:
        logger.info("rag_cli_v2 not available — priors will be None for %s vs %s", home, away)
        return

    # projected_total_* returns (blended, season, recent) — index 0 is the blended total
    try:
        result = projected_total_goals(home, away, league)
        if result and result[0] is not None:
            state.prior_total_goals = result[0]  # blended total
    except Exception as exc:
        logger.warning("Prior goals failed for %s vs %s: %s", home, away, exc)

    # Per-team goal projections for BTTS/moneyline
    try:
        hm = _profile_meta(home, league) if _profile_meta else {}
        am = _profile_meta(away, league) if _profile_meta else {}
        h_goals, a_goals = projected_goals(hm, am) if projected_goals else (None, None)
        if h_goals is not None:
            state.prior_home_goals = h_goals
        if a_goals is not None:
            state.prior_away_goals = a_goals
    except Exception as exc:
        logger.warning("Prior per-team goals failed for %s vs %s: %s", home, away, exc)

    try:
        result = projected_total_corners(home, away, league)
        if result and result[0] is not None:
            state.prior_total_corners = result[0]
    except Exception as exc:
        logger.warning("Prior corners failed for %s vs %s: %s", home, away, exc)

    try:
        result = projected_total_cards(home, away, league)
        # projected_total_cards returns (blended, season, recent, ref_mod) — 4 items
        if result and result[0] is not None:
            state.prior_total_cards = result[0]
    except Exception as exc:
        logger.warning("Prior cards failed for %s vs %s: %s", home, away, exc)

    try:
        result = projected_total_sot(home, away, league)
        if result and result[0] is not None:
            state.prior_total_sot = result[0]
    except Exception as exc:
        logger.warning("Prior SoT failed for %s vs %s: %s", home, away, exc)

    try:
        result = projected_btts_prob(home, away, league)
        if result and result[0] is not None:
            state.prior_btts_prob = result[0]
    except Exception as exc:
        logger.warning("Prior BTTS failed for %s vs %s: %s", home, away, exc)

    try:
        result = projected_moneyline_probs(home, away, league) if projected_moneyline_probs else None
        if result:
            p_home, p_draw, p_away = result[:3]
            if None not in (p_home, p_draw, p_away):
                state.prior_moneyline = {
                    "home_win": float(p_home),
                    "draw": float(p_draw),
                    "away_win": float(p_away),
                }
    except Exception as exc:
        logger.warning("Prior moneyline failed for %s vs %s: %s", home, away, exc)

    logger.info(
        "Priors computed for %s vs %s [%s]: goals=%.2f corners=%.2f cards=%.2f sot=%.2f btts=%.2f ml=%s",
        home, away, league,
        state.prior_total_goals or 0,
        state.prior_total_corners or 0,
        state.prior_total_cards or 0,
        state.prior_total_sot or 0,
        state.prior_btts_prob or 0,
        ("set" if state.prior_moneyline else "none"),
    )


# ---------------------------------------------------------------------------
# LivePoller
# ---------------------------------------------------------------------------

class LivePoller:
    """Async polling system for live match data from API-Football.

    Thread-safe: snapshot reads use a lock so the FastAPI layer can
    read snapshots while the poller writes them.
    """

    def __init__(self, auto_subscribe: bool = True, store: Optional[Any] = None):
        # Match states keyed by API-Football fixture_id
        self.states: Dict[int, LiveMatchState] = {}
        # Latest snapshot per fixture
        self.snapshots: Dict[int, LiveSnapshot] = {}
        # Snapshot history for charting (ring buffer, last N)
        self.snapshot_history: Dict[int, List[LiveSnapshot]] = {}
        # Subscribed fixture IDs (detail polling)
        self.subscribers: Set[int] = set()
        # Archived (finished) fixture IDs
        self.archived: Set[int] = set()

        # Threading lock for concurrent snapshot reads
        self._lock = threading.Lock()
        self._running = False
        self._auto_subscribe = auto_subscribe
        self._store = store

        # Metrics
        self._poll_count = 0
        self._detail_poll_count = 0
        self._errors = 0
        self._started_at: Optional[str] = None

        # aiohttp session (created on start)
        self._session: Optional[Any] = None
        self._discovery_task: Optional[asyncio.Task] = None
        self._detail_task: Optional[asyncio.Task] = None

        # Callbacks for snapshot updates (SSE push)
        self._on_snapshot: List[Callable] = []

    # -- Public API ---------------------------------------------------------

    def subscribe(self, fixture_id: int) -> bool:
        """Start detail polling for a fixture. Returns True if newly added."""
        if fixture_id in self.archived:
            logger.info("Cannot subscribe to finished fixture %d", fixture_id)
            return False
        added = fixture_id not in self.subscribers
        self.subscribers.add(fixture_id)
        if added:
            logger.info("Subscribed to fixture %d", fixture_id)
        return added

    def unsubscribe(self, fixture_id: int) -> bool:
        """Stop detail polling for a fixture. Returns True if was subscribed."""
        removed = fixture_id in self.subscribers
        self.subscribers.discard(fixture_id)
        if removed:
            logger.info("Unsubscribed from fixture %d", fixture_id)
        return removed

    def get_active_fixtures(self) -> List[Dict[str, Any]]:
        """Return summary info for all currently live fixtures."""
        with self._lock:
            result = []
            for fid, state in self.states.items():
                if fid in self.archived:
                    continue
                result.append({
                    "fixture_id": fid,
                    "home": state.home_team,
                    "away": state.away_team,
                    "score": f"{state.home_stats.goals}-{state.away_stats.goals}",
                    "elapsed": state.elapsed_min,
                    "status": state.status,
                    "league": state.league,
                    "league_id": state.league_id,
                })
            return result

    def get_state(self, fixture_id: int) -> Optional[LiveMatchState]:
        with self._lock:
            return self.states.get(fixture_id)

    def get_snapshot(self, fixture_id: int) -> Optional[LiveSnapshot]:
        with self._lock:
            return self.snapshots.get(fixture_id)

    def get_snapshot_history(self, fixture_id: int) -> List[LiveSnapshot]:
        with self._lock:
            return list(self.snapshot_history.get(fixture_id, []))

    def get_health(self) -> Dict[str, Any]:
        return {
            "running": self._running,
            "started_at": self._started_at,
            "active_fixtures": len(self.states) - len(self.archived),
            "subscribers": len(self.subscribers),
            "archived": len(self.archived),
            "poll_count": self._poll_count,
            "detail_poll_count": self._detail_poll_count,
            "errors": self._errors,
            "has_live_engine": _HAS_LIVE_ENGINE,
            "has_rag_priors": _HAS_RAG,
            "has_alert_engine": _HAS_ALERTS,
            "has_live_store": self._store is not None,
            "has_live_odds": _HAS_LIVE_ODDS,
        }

    def on_snapshot(self, callback: Callable) -> None:
        """Register a callback for snapshot updates: callback(fixture_id, snapshot)."""
        self._on_snapshot.append(callback)

    def _persist_fixture(self, state: LiveMatchState) -> None:
        if self._store is None:
            return
        try:
            self._store.ensure_fixture(
                state.fixture_id,
                state.league,
                state.home_team,
                state.away_team,
                state.kickoff_utc,
            )
        except Exception as exc:
            logger.debug("Live store ensure_fixture failed for %d: %s", state.fixture_id, exc)

    def _persist_state(self, state: LiveMatchState) -> None:
        if self._store is None:
            return
        try:
            self._store.record_state(
                state.fixture_id,
                state.to_dict(),
                status=state.status,
                elapsed_min=state.elapsed_min,
            )
        except Exception as exc:
            logger.debug("Live store record_state failed for %d: %s", state.fixture_id, exc)

    def _persist_snapshot(self, fixture_id: int, snapshot: Any) -> None:
        if self._store is None:
            return
        try:
            snapshot_payload = snapshot.to_dict() if hasattr(snapshot, "to_dict") else asdict(snapshot)
            self._store.record_snapshot(
                fixture_id,
                snapshot_payload,
                snapshot_ts=str(getattr(snapshot, "timestamp", "")),
                elapsed_min=getattr(snapshot, "elapsed_min", None),
                score=getattr(snapshot, "score", None),
            )
            self._store.record_alerts(
                fixture_id,
                str(getattr(snapshot, "timestamp", "")),
                getattr(snapshot, "alerts", []) or [],
            )
        except Exception as exc:
            logger.debug("Live store record_snapshot failed for %d: %s", fixture_id, exc)

    def _persist_live_odds(self, fixture_id: int, odds_payload: Dict[str, Any]) -> None:
        if self._store is None:
            return
        try:
            self._store.record_live_odds(
                fixture_id,
                odds_payload,
                captured_at=str(odds_payload.get("captured_at") or datetime.now(timezone.utc).isoformat()),
            )
        except Exception as exc:
            logger.debug("Live store record_live_odds failed for %d: %s", fixture_id, exc)

    def _mark_finished(self, fixture_id: int) -> None:
        if self._store is None:
            return
        try:
            self._store.mark_finished(fixture_id)
        except Exception as exc:
            logger.debug("Live store mark_finished failed for %d: %s", fixture_id, exc)

    def _attach_snapshot_alerts(self, fixture_id: int, snapshot: Any, state: LiveMatchState) -> Any:
        live_odds_source = state.live_odds_source or (state.live_odds or {}).get("source")
        try:
            snapshot.live_odds = state.live_odds or {}
            snapshot.live_odds_source = live_odds_source
            snapshot.recommendation_mode = None
            snapshot.recommendation_pricing_source = None
            snapshot.has_priced_recommendation = False
        except Exception:
            pass
        if not (_HAS_ALERTS and detect_alerts is not None and _HAS_LIVE_ENGINE):
            return snapshot
        try:
            previous = None
            history = self.snapshot_history.get(fixture_id, [])
            if history:
                previous = history[-1]
            engine_state = self._to_engine_state(state)
            alerts = detect_alerts(snapshot, previous, engine_state)
            snapshot.alerts = alerts or []
            live_bet = next((a for a in snapshot.alerts if a.get("type") == "live_bet"), None)
            if live_bet is not None:
                snapshot.recommendation_mode = live_bet.get("mode")
                snapshot.recommendation_pricing_source = live_bet.get("pricing_source")
                snapshot.has_priced_recommendation = bool(live_bet.get("mode") == "odds_aware")
        except Exception as exc:
            logger.warning("Alert attachment failed for fixture %d: %s", fixture_id, exc)
            try:
                snapshot.alerts = []
                snapshot.recommendation_mode = None
                snapshot.recommendation_pricing_source = None
                snapshot.has_priced_recommendation = False
            except Exception:
                pass
        return snapshot

    # -- Lifecycle ----------------------------------------------------------

    async def start(self) -> None:
        """Start the discovery and detail polling loops."""
        if self._running:
            logger.warning("Poller already running")
            return

        self._running = True
        self._started_at = datetime.now(timezone.utc).isoformat()

        # Try to create aiohttp session
        try:
            import aiohttp
            self._session = aiohttp.ClientSession()
        except ImportError:
            self._session = None
            logger.info("aiohttp not available, using requests fallback")

        logger.info(
            "LivePoller started (discovery=%ds, detail=%ds, auto_subscribe=%s)",
            DISCOVERY_POLL_INTERVAL, DETAIL_POLL_INTERVAL, self._auto_subscribe,
        )
        self._discovery_task = asyncio.create_task(self._discovery_loop(), name="live-discovery-loop")
        self._detail_task = asyncio.create_task(self._detail_loop(), name="live-detail-loop")

    async def stop(self) -> None:
        """Stop all polling loops and clean up."""
        self._running = False
        for task in (self._discovery_task, self._detail_task):
            if task is not None:
                task.cancel()
        for task in (self._discovery_task, self._detail_task):
            if task is not None:
                with suppress(asyncio.CancelledError):
                    await task
        self._discovery_task = None
        self._detail_task = None
        if self._session is not None:
            try:
                await self._session.close()
            except Exception:
                pass
            self._session = None
        logger.info("LivePoller stopped")

    # -- Discovery loop -----------------------------------------------------

    async def _discovery_loop(self) -> None:
        """Periodically scan for new/finished live fixtures."""
        while self._running:
            try:
                await self._poll_discovery()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._errors += 1
                logger.error("Discovery poll error: %s", exc, exc_info=True)
            await asyncio.sleep(DISCOVERY_POLL_INTERVAL)

    async def _poll_discovery(self) -> None:
        """Single discovery poll iteration."""
        response = await _async_api_get(
            "/fixtures",
            {"live": "all"},
            session=self._session,
        )
        if response is None:
            logger.debug("Discovery poll returned None (API issue or no key)")
            return

        self._poll_count += 1
        seen_ids: Set[int] = set()

        for item in response:
            fixture_info = item.get("fixture", {})
            league_info = item.get("league", {})
            teams_info = item.get("teams", {})
            goals_info = item.get("goals", {})

            league_id = league_info.get("id")
            fixture_id = fixture_info.get("id")
            status_short = fixture_info.get("status", {}).get("short", "")

            if league_id not in TRACKED_LEAGUE_IDS:
                continue
            if fixture_id is None:
                continue

            seen_ids.add(fixture_id)

            # Handle finished fixtures
            if status_short in FINISHED_STATUSES:
                if fixture_id in self.states and fixture_id not in self.archived:
                    logger.info(
                        "Fixture %d finished (%s): %s %d-%d %s",
                        fixture_id, status_short,
                        self.states[fixture_id].home_team,
                        goals_info.get("home", 0) or 0,
                        goals_info.get("away", 0) or 0,
                        self.states[fixture_id].away_team,
                    )
                    self.archived.add(fixture_id)
                    self.subscribers.discard(fixture_id)
                    self._mark_finished(fixture_id)
                continue

            if status_short not in LIVE_STATUSES:
                continue

            # New live fixture detected
            if fixture_id not in self.states:
                home_name = teams_info.get("home", {}).get("name", "Unknown")
                away_name = teams_info.get("away", {}).get("name", "Unknown")
                league_name = LEAGUE_ID_TO_NAME.get(league_id, "Unknown")

                state = LiveMatchState(
                    fixture_id=fixture_id,
                    league=league_name,
                    league_id=league_id,
                    home_team=home_name,
                    away_team=away_name,
                    status=status_short,
                    elapsed_min=fixture_info.get("status", {}).get("elapsed", 0) or 0,
                    kickoff_utc=fixture_info.get("date"),
                )
                # Set initial goals from discovery response
                state.home_stats.goals = goals_info.get("home", 0) or 0
                state.away_stats.goals = goals_info.get("away", 0) or 0

                # Compute pre-match priors (sync, runs once)
                _compute_priors(home_name, away_name, league_name, state)

                with self._lock:
                    self.states[fixture_id] = state
                    self.snapshot_history[fixture_id] = []

                self._persist_fixture(state)
                self._persist_state(state)

                logger.info(
                    "New live fixture %d: %s vs %s [%s] (status=%s, elapsed=%d)",
                    fixture_id, home_name, away_name, league_name,
                    status_short, state.elapsed_min,
                )

                if self._auto_subscribe:
                    self.subscribe(fixture_id)

        logger.debug("Discovery poll: %d tracked fixtures, %d live in feed", len(self.states), len(seen_ids))

    # -- Detail loop --------------------------------------------------------

    async def _detail_loop(self) -> None:
        """Periodically fetch detailed stats for subscribed fixtures."""
        while self._running:
            subscriber_list = list(self.subscribers)
            for fid in subscriber_list:
                if not self._running:
                    break
                if fid in self.archived:
                    continue
                try:
                    await self._poll_detail(fid)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    self._errors += 1
                    logger.error("Detail poll error for fixture %d: %s", fid, exc, exc_info=True)
            await asyncio.sleep(DETAIL_POLL_INTERVAL)

    async def _poll_detail(self, fixture_id: int) -> None:
        """Single detail poll for one fixture."""
        response = await _async_api_get(
            "/fixtures",
            {"id": str(fixture_id)},
            session=self._session,
        )
        if not response:
            logger.debug("Detail poll for fixture %d returned empty", fixture_id)
            return

        self._detail_poll_count += 1
        item = response[0]

        fixture_info = item.get("fixture", {})
        goals_info = item.get("goals", {})
        statistics_raw = item.get("statistics", [])
        events_raw = item.get("events", [])

        status_short = fixture_info.get("status", {}).get("short", "")
        elapsed = fixture_info.get("status", {}).get("elapsed", 0) or 0

        # Check if match finished
        if status_short in FINISHED_STATUSES:
            logger.info("Fixture %d finished during detail poll", fixture_id)
            self.archived.add(fixture_id)
            self.subscribers.discard(fixture_id)
            self._mark_finished(fixture_id)

        # Parse team statistics
        # API-Football returns statistics as list of 2 items: [home_stats, away_stats]
        # Each item has {"team": {...}, "statistics": [{type, value}, ...]}
        home_team_stats = TeamLiveStats()
        away_team_stats = TeamLiveStats()
        if len(statistics_raw) >= 2:
            home_stat_list = statistics_raw[0].get("statistics", [])
            away_stat_list = statistics_raw[1].get("statistics", [])
            home_team_stats = _parse_team_stats(home_stat_list)
            away_team_stats = _parse_team_stats(away_stat_list)
            logger.debug(
                "Fixture %d stats: home corners=%d sot=%d shots=%d | away corners=%d sot=%d shots=%d",
                fixture_id,
                home_team_stats.corners, home_team_stats.sot, home_team_stats.shots,
                away_team_stats.corners, away_team_stats.sot, away_team_stats.shots,
            )
        else:
            # Statistics not available — try fetching from /fixtures/statistics endpoint
            logger.warning(
                "Fixture %d: statistics array has %d items (expected 2). Stats will be zero.",
                fixture_id, len(statistics_raw),
            )

        # Set goals from goals object (more reliable than statistics)
        home_team_stats.goals = goals_info.get("home", 0) or 0
        away_team_stats.goals = goals_info.get("away", 0) or 0

        # Parse events
        events = _parse_events(events_raw)

        # Reconcile card counts from events — the events feed posts in near
        # real-time whereas the `statistics` array lags 2-5 minutes. Take the
        # max so we never under-report cards while waiting for stats to catch
        # up (which would otherwise cause the engine to recommend Under lines
        # based on a stale zero count).
        state_obj = self.states.get(fixture_id)
        home_name_for_cards = state_obj.home_team if state_obj else ""
        away_name_for_cards = state_obj.away_team if state_obj else ""
        ev_hy, ev_hr, ev_ay, ev_ar = _card_counts_from_events(
            events, home_name_for_cards, away_name_for_cards,
        )
        stats_hy = home_team_stats.yellow_cards
        stats_hr = home_team_stats.red_cards
        stats_ay = away_team_stats.yellow_cards
        stats_ar = away_team_stats.red_cards
        home_team_stats.yellow_cards = max(stats_hy, ev_hy)
        home_team_stats.red_cards = max(stats_hr, ev_hr)
        away_team_stats.yellow_cards = max(stats_ay, ev_ay)
        away_team_stats.red_cards = max(stats_ar, ev_ar)
        if (
            (ev_hy > stats_hy) or (ev_hr > stats_hr)
            or (ev_ay > stats_ay) or (ev_ar > stats_ar)
        ):
            logger.info(
                "Fixture %d card stats backfilled from events: "
                "home y=%d->%d r=%d->%d | away y=%d->%d r=%d->%d",
                fixture_id,
                stats_hy, home_team_stats.yellow_cards,
                stats_hr, home_team_stats.red_cards,
                stats_ay, away_team_stats.yellow_cards,
                stats_ar, away_team_stats.red_cards,
            )

        # Update state under lock
        now_iso = datetime.now(timezone.utc).isoformat()
        with self._lock:
            state = self.states.get(fixture_id)
            if state is None:
                logger.warning("Detail poll for unknown fixture %d — skipping", fixture_id)
                return

            state.status = status_short
            state.elapsed_min = elapsed
            state.home_stats = home_team_stats
            state.away_stats = away_team_stats
            state.events = events
            state.last_updated = now_iso

        if _HAS_LIVE_ODDS and fetch_live_fixture_odds is not None:
            try:
                odds_snapshot, odds_err = await fetch_live_fixture_odds(
                    fixture_id,
                    state.home_team,
                    state.away_team,
                    session=self._session,
                )
            except Exception as exc:
                logger.warning("Live odds fetch raised for fixture %d: %s", fixture_id, exc, exc_info=True)
                odds_snapshot, odds_err = None, str(exc)
            if odds_snapshot:
                with self._lock:
                    state.live_odds = odds_snapshot
                    state.live_odds_updated_at = str(odds_snapshot.get("captured_at") or now_iso)
                    state.live_odds_source = str(odds_snapshot.get("source") or "")
                self._persist_live_odds(fixture_id, odds_snapshot)
                # One-line observability summary: how many offers per market group
                totals = (odds_snapshot or {}).get("totals", {}) or {}
                logger.info(
                    "Live odds applied fixture=%d source=%s totals(g=%d/c=%d/cd=%d/s=%d) btts=%d ml=%d",
                    fixture_id,
                    odds_snapshot.get("source", "?"),
                    len(totals.get("goals") or []),
                    len(totals.get("corners") or []),
                    len(totals.get("cards") or []),
                    len(totals.get("sot") or []),
                    len(odds_snapshot.get("btts") or []),
                    len(odds_snapshot.get("moneyline") or []),
                )
            elif odds_err:
                logger.warning("No live odds for fixture %d: %s", fixture_id, odds_err)
        else:
            logger.debug("Live odds module disabled (_HAS_LIVE_ODDS=%s)", _HAS_LIVE_ODDS)

        self._persist_state(state)

        # Compute snapshot via live engine
        snapshot = self._try_compute_snapshot(state)
        if snapshot is not None:
            snapshot = self._attach_snapshot_alerts(fixture_id, snapshot, state)
            with self._lock:
                self.snapshots[fixture_id] = snapshot
                history = self.snapshot_history.setdefault(fixture_id, [])
                history.append(snapshot)
                if len(history) > SNAPSHOT_HISTORY_SIZE:
                    self.snapshot_history[fixture_id] = history[-SNAPSHOT_HISTORY_SIZE:]

            self._persist_snapshot(fixture_id, snapshot)

            # Notify callbacks
            for cb in self._on_snapshot:
                try:
                    cb(fixture_id, snapshot)
                except Exception as exc:
                    logger.warning("Snapshot callback error: %s", exc)

        logger.debug(
            "Detail poll fixture %d: %s %d-%d %s (elapsed=%d, status=%s)",
            fixture_id,
            state.home_team,
            home_team_stats.goals,
            away_team_stats.goals,
            state.away_team,
            elapsed,
            status_short,
        )

    # -- Snapshot computation -----------------------------------------------

    @staticmethod
    def _convert_events(poller_events: list, home_team: str, away_team: str) -> list:
        """Convert poller MatchEvents (team=name) to engine MatchEvents (team_side=home/away)."""
        try:
            from live.live_state import MatchEvent as EngineEvent
        except ImportError:
            from live_state import MatchEvent as EngineEvent

        home_lower = home_team.lower().strip() if home_team else ""
        converted = []
        for ev in poller_events:
            team_name = getattr(ev, "team", "") or ""
            team_side = "home" if team_name.lower().strip() == home_lower else "away"
            converted.append(EngineEvent(
                minute=getattr(ev, "minute", 0),
                event_type=getattr(ev, "event_type", ""),
                team_side=team_side,
                player=getattr(ev, "player", "") or "",
                detail=getattr(ev, "detail", "") or "",
            ))
        return converted

    def _to_engine_state(self, state: LiveMatchState):
        """Convert poller's LiveMatchState to the engine's LiveMatchState format.

        The engine (live_engine.py) expects flat fields like `prior_goals`,
        `goals_home`, `corners_home`, etc. The poller uses nested
        `home_stats`/`away_stats` objects and `prior_total_goals`.
        """
        try:
            from live.live_state import LiveMatchState as EngineState
        except ImportError:
            from live_state import LiveMatchState as EngineState

        hs = state.home_stats
        as_ = state.away_stats

        engine_state = EngineState(
            fixture_id=state.fixture_id,
            league=state.league,
            home_team=state.home_team,
            away_team=state.away_team,
            status=state.status,
            elapsed_min=float(state.elapsed_min),
            # Score
            goals_home=hs.goals if hs else 0,
            goals_away=as_.goals if as_ else 0,
            # Stats
            corners_home=hs.corners if hs else 0,
            corners_away=as_.corners if as_ else 0,
            cards_home=(hs.yellow_cards + hs.red_cards) if hs else 0,
            cards_away=(as_.yellow_cards + as_.red_cards) if as_ else 0,
            sot_home=hs.sot if hs else 0,
            sot_away=as_.sot if as_ else 0,
            shots_home=hs.shots if hs else 0,
            shots_away=as_.shots if as_ else 0,
            fouls_home=hs.fouls if hs else 0,
            fouls_away=as_.fouls if as_ else 0,
            possession_home=hs.possession if hs else 50.0,
            possession_away=as_.possession if as_ else 50.0,
            red_cards_home=hs.red_cards if hs else 0,
            red_cards_away=as_.red_cards if as_ else 0,
            # Events — convert poller MatchEvent (team=name) to engine MatchEvent (team_side=home/away)
            events=self._convert_events(state.events, state.home_team, state.away_team),
            # Priors (map poller field names to engine field names)
            prior_goals=state.prior_total_goals,
            prior_goals_home=state.prior_home_goals,
            prior_goals_away=state.prior_away_goals,
            prior_corners=state.prior_total_corners,
            prior_cards=state.prior_total_cards,
            prior_sot=state.prior_total_sot,
            prior_btts_prob=state.prior_btts_prob,
            prior_moneyline=state.prior_moneyline,
            live_odds=state.live_odds,
            live_odds_updated_at=state.live_odds_updated_at,
            live_odds_source=state.live_odds_source,
        )
        return engine_state

    def _try_compute_snapshot(self, state: LiveMatchState) -> Optional[LiveSnapshot]:
        """Attempt to compute a live snapshot. Returns None on failure."""
        if not _HAS_LIVE_ENGINE or compute_snapshot is None:
            # Engine not available — build a basic snapshot from raw state
            return LiveSnapshot(
                fixture_id=state.fixture_id,
                timestamp=datetime.now(timezone.utc).isoformat(),
                elapsed_min=state.elapsed_min,
                projected_total_goals=state.prior_total_goals,
                projected_total_corners=state.prior_total_corners,
                projected_total_cards=state.prior_total_cards,
                projected_total_sot=state.prior_total_sot,
                live_odds=state.live_odds,
                live_odds_source=state.live_odds_source,
            )

        try:
            # Convert poller state to engine-compatible format
            engine_state = self._to_engine_state(state)
            return compute_snapshot(engine_state)
        except Exception as exc:
            logger.warning(
                "compute_snapshot failed for fixture %d: %s (returning basic snapshot)",
                state.fixture_id, exc,
            )
            return LiveSnapshot(
                fixture_id=state.fixture_id,
                timestamp=datetime.now(timezone.utc).isoformat(),
                elapsed_min=state.elapsed_min,
                projected_total_goals=state.prior_total_goals,
                projected_total_corners=state.prior_total_corners,
                projected_total_cards=state.prior_total_cards,
                projected_total_sot=state.prior_total_sot,
                live_odds=state.live_odds,
                live_odds_source=state.live_odds_source,
            )


# ---------------------------------------------------------------------------
# Standalone runner (for testing / development)
# ---------------------------------------------------------------------------

async def _main() -> None:
    """Run the poller standalone for testing."""
    logging.basicConfig(level=logging.DEBUG)
    poller = LivePoller(auto_subscribe=True)
    await poller.start()
    try:
        while True:
            await asyncio.sleep(30)
            fixtures = poller.get_active_fixtures()
            if fixtures:
                logger.info("Active fixtures: %s", json.dumps(fixtures, indent=2))
            else:
                logger.info("No active fixtures")
            health = poller.get_health()
            logger.info("Health: %s", json.dumps(health))
    except KeyboardInterrupt:
        pass
    finally:
        await poller.stop()


if __name__ == "__main__":
    asyncio.run(_main())
