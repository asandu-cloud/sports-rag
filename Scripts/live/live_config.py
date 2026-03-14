"""Live betting engine configuration constants.

All tuning parameters for the live projection engine are centralized here.
Adjust these values to calibrate Bayesian update strength, situational
modifiers, alert thresholds, and pace decay distributions.
"""

# ---------------------------------------------------------------------------
# Prior strength for Bayesian update (Gamma-Poisson conjugate)
# ---------------------------------------------------------------------------
# Conceptually, this is "how many equivalent full matches" the pre-match
# model is worth.  Higher values = trust the prior longer; lower values =
# converge to observed pace faster.
#
# Goals are rare events (~2.7 per match) so each observation carries a lot
# of information.  Corners (~10 per match) are noisier, so we let observed
# pace take over sooner.

LIVE_PRIOR_STRENGTH = {
    "goals":   6.0,   # rare events, prior matters more
    "corners": 4.0,   # volatile, trust observed pace sooner
    "cards":   5.0,   # moderate rarity
    "sot":     4.5,   # moderate-high variance
}

# ---------------------------------------------------------------------------
# Polling intervals (seconds)
# ---------------------------------------------------------------------------

DISCOVERY_POLL_INTERVAL = 60    # how often to scan for new live fixtures
DETAIL_POLL_INTERVAL    = 45    # how often to refresh each tracked fixture

# ---------------------------------------------------------------------------
# Situational multiplier config
# ---------------------------------------------------------------------------
# Applied on top of the Bayesian posterior to account for game-state effects
# that the Poisson model does not capture natively.

SITUATIONAL = {
    # Red card effects
    "red_card_attack_reduction": 0.85,    # 10-man team scores ~15% less
    "red_card_card_increase":    1.15,    # 10-man team commits more fouls/cards
    # Late-game dynamics
    "tight_finish_card_increase": 1.20,   # within 1 goal, minute >= 70
    "trailing_corner_increase":   1.10,   # trailing team pushes forward, 75'+
    "second_half_goal_boost":     1.10,   # 2H has ~10% higher goal rate
}

# ---------------------------------------------------------------------------
# Empirical time distributions (from academic research & Opta data)
# ---------------------------------------------------------------------------
# Fraction of total match events occurring in each 15-minute window.
# Used by the pace-decay model to compute how much "action" remains.

GOAL_TIME_DISTRIBUTION = {
    (0, 15):  0.145,
    (15, 30): 0.145,
    (30, 45): 0.160,   # 1H total: ~0.45
    (45, 60): 0.155,
    (60, 75): 0.175,
    (75, 90): 0.220,   # 2H total: ~0.55, late surge
}

CORNER_TIME_DISTRIBUTION = {
    (0, 15):  0.155,
    (15, 30): 0.165,
    (30, 45): 0.170,   # 1H: ~0.49
    (45, 60): 0.160,
    (60, 75): 0.170,
    (75, 90): 0.180,   # 2H: ~0.51, slight increase
}

CARD_TIME_DISTRIBUTION = {
    (0, 15):  0.10,
    (15, 30): 0.13,
    (30, 45): 0.17,    # 1H: ~0.40
    (45, 60): 0.15,
    (60, 75): 0.20,
    (75, 90): 0.25,    # 2H: ~0.60, heavy late tactical fouling
}

SOT_TIME_DISTRIBUTION = {
    (0, 15):  0.150,
    (15, 30): 0.155,
    (30, 45): 0.165,   # 1H: ~0.47
    (45, 60): 0.160,
    (60, 75): 0.175,
    (75, 90): 0.195,   # 2H: ~0.53
}

# Map stat name -> distribution
STAT_TIME_DISTRIBUTIONS = {
    "goals":   GOAL_TIME_DISTRIBUTION,
    "corners": CORNER_TIME_DISTRIBUTION,
    "cards":   CARD_TIME_DISTRIBUTION,
    "sot":     SOT_TIME_DISTRIBUTION,
}

# ---------------------------------------------------------------------------
# Alert thresholds
# ---------------------------------------------------------------------------

ALERT_THRESHOLDS = {
    "btts_threat_sot_min":     60,     # minute after which 0 SoT triggers alert
    "total_danger_low":        0.30,   # P(over) drops below -> danger alert
    "total_danger_high":       0.80,   # P(over) rises above -> opportunity alert
    "momentum_shift_threshold": 0.25,  # swing in momentum score -> alert
    "pace_anomaly_ratio":      1.5,    # observed/expected ratio for pace alert
    "value_shift_drop":        0.50,   # pre-match pick drops below this prob
    "xg_divergence_threshold": 1.5,    # |xG - actual goals| threshold
}

# Alert cooldown: don't fire same alert type for same fixture within window
ALERT_COOLDOWN_SECONDS = 300   # 5 minutes

# ---------------------------------------------------------------------------
# xG conversion rates (from historical average shot quality data)
# ---------------------------------------------------------------------------

XG_SOT_CONVERSION     = 0.30   # shot on target -> goal conversion rate
XG_OFF_TARGET_RATE     = 0.03   # off-target shot -> goal (deflections etc.)

# ---------------------------------------------------------------------------
# Momentum scoring weights
# ---------------------------------------------------------------------------

MOMENTUM_WEIGHTS = {
    "shots":           0.30,
    "possession":      0.20,
    "corners":         0.15,
    "sot":             0.20,
    "recent_goals":    1.00,
    "cards_against":  -0.10,   # negative: cards against reduce momentum
}

MOMENTUM_WINDOW_MIN = 15   # minutes to look back for recent events
