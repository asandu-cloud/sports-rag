"""Data classes for live match state and projection snapshots.

These are pure data containers with no business logic.  The engine
(live_engine.py) reads LiveMatchState and produces LiveSnapshot.
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Dict, List, Optional


@dataclass
class MatchEvent:
    """A single timestamped event within a match."""

    minute: int
    event_type: str   # "goal", "yellow_card", "red_card", "substitution",
                      # "var", "penalty", "corner", "shot_on_target"
    team_side: str    # "home" or "away"
    player: str
    detail: str       # "Normal Goal", "Yellow Card", "Second Yellow",
                      # "Own Goal", "Penalty - Scored", etc.

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class LiveMatchState:
    """Mutable representation of a match currently in play.

    Fields are updated each polling cycle.  Pre-match priors are frozen
    at kickoff and never modified.
    """

    fixture_id: int
    league: str
    home_team: str
    away_team: str
    status: str           # "NS", "1H", "HT", "2H", "ET", "FT", "AET", "PEN"
    elapsed_min: float    # minutes elapsed (0-90+; may exceed 90 in stoppage)

    # Current score
    goals_home: int = 0
    goals_away: int = 0

    # Cumulative live stats
    corners_home: int = 0
    corners_away: int = 0
    cards_home: int = 0
    cards_away: int = 0
    sot_home: int = 0
    sot_away: int = 0
    shots_home: int = 0
    shots_away: int = 0
    fouls_home: int = 0
    fouls_away: int = 0
    possession_home: float = 50.0
    possession_away: float = 50.0
    red_cards_home: int = 0
    red_cards_away: int = 0

    # Event log (chronological)
    events: List[MatchEvent] = field(default_factory=list)

    # Pre-match priors (frozen at kickoff, never mutated)
    prior_goals: Optional[float] = None         # projected total goals
    prior_goals_home: Optional[float] = None    # projected home goals
    prior_goals_away: Optional[float] = None    # projected away goals
    prior_corners: Optional[float] = None       # projected total corners
    prior_cards: Optional[float] = None         # projected total cards
    prior_sot: Optional[float] = None           # projected total SoT
    prior_btts_prob: Optional[float] = None     # P(both teams score)
    prior_moneyline: Optional[Dict[str, float]] = None  # {"home_win", "draw", "away_win"}
    live_odds: Dict = field(default_factory=dict)
    live_odds_updated_at: Optional[str] = None
    live_odds_source: Optional[str] = None

    last_updated: Optional[datetime] = None
    poll_count: int = 0

    # ---- Convenience properties ----------------------------------------

    @property
    def total_goals(self) -> int:
        return self.goals_home + self.goals_away

    @property
    def total_corners(self) -> int:
        return self.corners_home + self.corners_away

    @property
    def total_cards(self) -> int:
        return self.cards_home + self.cards_away

    @property
    def total_sot(self) -> int:
        return self.sot_home + self.sot_away

    @property
    def goal_diff(self) -> int:
        """Positive = home leading."""
        return self.goals_home - self.goals_away

    @property
    def is_first_half(self) -> bool:
        return self.status in ("1H",)

    @property
    def is_second_half(self) -> bool:
        return self.status in ("2H",)

    @property
    def is_live(self) -> bool:
        return self.status in ("1H", "HT", "2H", "ET")

    @property
    def is_finished(self) -> bool:
        return self.status in ("FT", "AET", "PEN")

    def events_in_window(self, window_min: int) -> List[MatchEvent]:
        """Return events from the last *window_min* minutes."""
        cutoff = max(0, self.elapsed_min - window_min)
        return [e for e in self.events if e.minute >= cutoff]

    def to_dict(self) -> Dict:
        data = asdict(self)
        if self.last_updated is not None:
            data["last_updated"] = self.last_updated.isoformat()
        return data


@dataclass
class LiveSnapshot:
    """Computed projection output for a single fixture at a point in time.

    Produced by ``compute_snapshot()`` in live_engine.py.  This is the
    object that gets sent to clients (Discord bot, Streamlit, API).
    """

    fixture_id: int
    home_team: str
    away_team: str
    league: str
    elapsed_min: float
    status: str
    score: str                                 # e.g. "2 - 1"

    # Remaining projections (posterior expected remaining count)
    remaining_goals: float
    remaining_corners: float
    remaining_cards: float
    remaining_sot: float

    # Updated total projections (observed + remaining)
    projected_total_goals: float
    projected_total_corners: float
    projected_total_cards: float
    projected_total_sot: float

    # Live probabilities for standard lines
    prob_over_goals: Dict[str, float]          # {"1.5": 0.85, "2.5": 0.42, ...}
    prob_over_corners: Dict[str, float]
    prob_over_cards: Dict[str, float]
    prob_over_sot: Dict[str, float]

    # Live BTTS and moneyline
    btts_prob: float
    moneyline: Dict[str, float]               # {"home_win": P, "draw": P, "away_win": P}

    # Pre-match comparison (for drift detection)
    pre_match_total_goals: Optional[float] = None
    pre_match_total_corners: Optional[float] = None
    pre_match_total_cards: Optional[float] = None
    pre_match_btts_prob: Optional[float] = None
    pre_match_moneyline: Optional[Dict[str, float]] = None

    # Momentum (home vs away share, sums to 1.0)
    momentum: Dict[str, float] = field(default_factory=lambda: {"home": 0.50, "away": 0.50})

    # xG-based adjustments (extra edge)
    live_xg_home: Optional[float] = None
    live_xg_away: Optional[float] = None
    xg_adjusted_moneyline: Optional[Dict[str, float]] = None

    # Score-state transition probabilities
    next_goal_probs: Optional[Dict[str, float]] = None   # {"home": P, "away": P, "none": P}

    # Poisson fit quality (0-1, 1 = perfect fit)
    poisson_fit: Optional[Dict[str, float]] = None       # {"goals": 0.85, "corners": 0.72, ...}

    # Alerts
    alerts: List[Dict] = field(default_factory=list)

    # Current live prices summary
    live_odds: Dict = field(default_factory=dict)
    live_odds_source: Optional[str] = None

    # Recommendation metadata
    recommendation_mode: Optional[str] = None
    recommendation_pricing_source: Optional[str] = None
    has_priced_recommendation: bool = False

    # Recent events (last 15 min, serialized for display)
    recent_events: List[Dict] = field(default_factory=list)

    timestamp: str = ""

    def to_dict(self) -> Dict:
        return asdict(self)
