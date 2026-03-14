"""Alert detection for the live betting projection engine.

Scans LiveSnapshot data (current and previous) plus raw LiveMatchState
to detect noteworthy situations: BTTS threats, total probability
shifts, momentum swings, red card impacts, pace anomalies, value
drift, and xG divergence.

Each alert carries a type, human-readable message, severity level,
and the minute it was triggered.  A cooldown mechanism prevents
spamming the same alert type for the same fixture within a
configurable time window.
"""

import time
from typing import Dict, List, Optional

from live_config import ALERT_THRESHOLDS, ALERT_COOLDOWN_SECONDS
from live_state import LiveMatchState, LiveSnapshot


# ---------------------------------------------------------------------------
# Cooldown state — keyed by (fixture_id, alert_type) -> last_fired_timestamp
# ---------------------------------------------------------------------------
_alert_cooldowns: Dict[tuple, float] = {}


def _can_fire(fixture_id: int, alert_type: str) -> bool:
    """Check whether an alert is allowed to fire (cooldown expired)."""
    key = (fixture_id, alert_type)
    last_fired = _alert_cooldowns.get(key, 0.0)
    return (time.time() - last_fired) >= ALERT_COOLDOWN_SECONDS


def _mark_fired(fixture_id: int, alert_type: str) -> None:
    """Record that an alert just fired (reset cooldown)."""
    _alert_cooldowns[(fixture_id, alert_type)] = time.time()


def clear_cooldowns(fixture_id: Optional[int] = None) -> None:
    """Clear cooldown state.  Useful for testing or when a fixture ends.

    Parameters
    ----------
    fixture_id : int, optional
        If provided, only clear cooldowns for this fixture.
        If None, clear all cooldowns.
    """
    global _alert_cooldowns
    if fixture_id is None:
        _alert_cooldowns = {}
    else:
        _alert_cooldowns = {
            k: v for k, v in _alert_cooldowns.items()
            if k[0] != fixture_id
        }


def _make_alert(
    alert_type: str,
    message: str,
    severity: str,
    minute: float,
) -> Dict:
    """Build a standardized alert dict."""
    return {
        "type": alert_type,
        "message": message,
        "severity": severity,
        "minute": int(minute),
    }


# ---------------------------------------------------------------------------
# Individual alert detectors
# ---------------------------------------------------------------------------

def _detect_btts_threat(
    current: LiveSnapshot,
    state: LiveMatchState,
) -> List[Dict]:
    """Alert when one team has 0 goals AND 0 SoT past a threshold minute.

    This signals that BTTS Yes is in serious danger — the team is not
    creating any chances at all.
    """
    alerts = []
    threshold_min = ALERT_THRESHOLDS["btts_threat_sot_min"]

    if state.elapsed_min < threshold_min:
        return alerts

    if state.goals_home == 0 and state.sot_home == 0:
        if _can_fire(state.fixture_id, "btts_threat_home"):
            alerts.append(_make_alert(
                "btts_threat",
                f"BTTS THREAT: {state.home_team} has 0 goals and 0 shots on "
                f"target at minute {int(state.elapsed_min)}. "
                f"BTTS Yes probability: {current.btts_prob:.0%}",
                "warning",
                state.elapsed_min,
            ))
            _mark_fired(state.fixture_id, "btts_threat_home")

    if state.goals_away == 0 and state.sot_away == 0:
        if _can_fire(state.fixture_id, "btts_threat_away"):
            alerts.append(_make_alert(
                "btts_threat",
                f"BTTS THREAT: {state.away_team} has 0 goals and 0 shots on "
                f"target at minute {int(state.elapsed_min)}. "
                f"BTTS Yes probability: {current.btts_prob:.0%}",
                "warning",
                state.elapsed_min,
            ))
            _mark_fired(state.fixture_id, "btts_threat_away")

    return alerts


def _detect_total_shift(
    current: LiveSnapshot,
    state: LiveMatchState,
) -> List[Dict]:
    """Alert when P(Over X.5) crosses danger/opportunity thresholds.

    Checks the most common lines: Over 2.5 Goals, Over 9.5 Corners,
    Over 3.5 Cards.
    """
    alerts = []
    lo = ALERT_THRESHOLDS["total_danger_low"]
    hi = ALERT_THRESHOLDS["total_danger_high"]

    # Check key lines
    checks = [
        ("goals", "2.5", current.prob_over_goals),
        ("corners", "9.5", current.prob_over_corners),
        ("cards", "3.5", current.prob_over_cards),
    ]

    for stat, line, prob_dict in checks:
        p = prob_dict.get(line)
        if p is None:
            continue

        alert_key = f"total_shift_{stat}_{line}"

        if p <= lo and _can_fire(state.fixture_id, alert_key + "_low"):
            alerts.append(_make_alert(
                "total_shift",
                f"DANGER: P(Over {line} {stat}) dropped to {p:.0%}. "
                f"Under looking strong at minute {int(state.elapsed_min)}.",
                "warning",
                state.elapsed_min,
            ))
            _mark_fired(state.fixture_id, alert_key + "_low")

        elif p >= hi and _can_fire(state.fixture_id, alert_key + "_high"):
            alerts.append(_make_alert(
                "total_shift",
                f"OPPORTUNITY: P(Over {line} {stat}) reached {p:.0%}. "
                f"Over looking very likely at minute {int(state.elapsed_min)}.",
                "info",
                state.elapsed_min,
            ))
            _mark_fired(state.fixture_id, alert_key + "_high")

    return alerts


def _detect_momentum_shift(
    current: LiveSnapshot,
    previous: Optional[LiveSnapshot],
    state: LiveMatchState,
) -> List[Dict]:
    """Alert when momentum swings significantly between updates."""
    alerts = []
    threshold = ALERT_THRESHOLDS["momentum_shift_threshold"]

    if previous is None:
        return alerts

    prev_home = previous.momentum.get("home", 0.5)
    curr_home = current.momentum.get("home", 0.5)
    swing = abs(curr_home - prev_home)

    if swing >= threshold and _can_fire(state.fixture_id, "momentum_shift"):
        gaining = "home" if curr_home > prev_home else "away"
        gaining_team = state.home_team if gaining == "home" else state.away_team
        alerts.append(_make_alert(
            "momentum_shift",
            f"MOMENTUM SHIFT: {gaining_team} momentum surged "
            f"({prev_home:.0%} -> {curr_home:.0%} home share) "
            f"at minute {int(state.elapsed_min)}.",
            "info",
            state.elapsed_min,
        ))
        _mark_fired(state.fixture_id, "momentum_shift")

    return alerts


def _detect_red_card_impact(
    current: LiveSnapshot,
    previous: Optional[LiveSnapshot],
    state: LiveMatchState,
) -> List[Dict]:
    """Alert when a red card is detected (new red card events in the log)."""
    alerts = []

    # Look for red card events in recent window
    recent = state.events_in_window(5)  # last 5 minutes
    red_cards = [
        e for e in recent
        if e.event_type == "red_card"
    ]

    for rc in red_cards:
        alert_key = f"red_card_{rc.team_side}_{rc.minute}"
        if _can_fire(state.fixture_id, alert_key):
            team = state.home_team if rc.team_side == "home" else state.away_team
            alerts.append(_make_alert(
                "red_card_impact",
                f"RED CARD: {rc.player} ({team}) sent off at minute {rc.minute}. "
                f"Projections recalculated — expect fewer goals/SoT from {team}, "
                f"more cards overall.",
                "critical",
                rc.minute,
            ))
            _mark_fired(state.fixture_id, alert_key)

    return alerts


def _detect_pace_anomaly(
    current: LiveSnapshot,
    state: LiveMatchState,
) -> List[Dict]:
    """Alert when observed pace significantly exceeds or falls below expected.

    Pace ratio = observed_so_far / expected_so_far.
    Alert if ratio > threshold (high pace) or < 1/threshold (low pace).
    """
    alerts = []
    ratio_threshold = ALERT_THRESHOLDS["pace_anomaly_ratio"]

    if state.elapsed_min < 15:
        # Too early for meaningful pace assessment
        return alerts

    elapsed_frac = state.elapsed_min / 90.0

    checks = [
        ("goals", state.total_goals, state.prior_goals or 2.7),
        ("corners", state.total_corners, state.prior_corners or 10.5),
        ("cards", state.total_cards, state.prior_cards or 4.1),
    ]

    for stat, observed, prior in checks:
        expected_so_far = prior * elapsed_frac
        if expected_so_far <= 0:
            continue

        ratio = observed / expected_so_far
        alert_key = f"pace_anomaly_{stat}"

        if ratio >= ratio_threshold and _can_fire(state.fixture_id, alert_key + "_high"):
            alerts.append(_make_alert(
                "pace_anomaly",
                f"HIGH PACE: {stat.upper()} running at {ratio:.1f}x expected rate "
                f"({observed} observed vs {expected_so_far:.1f} expected "
                f"at minute {int(state.elapsed_min)}).",
                "info",
                state.elapsed_min,
            ))
            _mark_fired(state.fixture_id, alert_key + "_high")

        elif ratio <= (1.0 / ratio_threshold) and _can_fire(state.fixture_id, alert_key + "_low"):
            alerts.append(_make_alert(
                "pace_anomaly",
                f"LOW PACE: {stat.upper()} running at {ratio:.1f}x expected rate "
                f"({observed} observed vs {expected_so_far:.1f} expected "
                f"at minute {int(state.elapsed_min)}).",
                "info",
                state.elapsed_min,
            ))
            _mark_fired(state.fixture_id, alert_key + "_low")

    return alerts


def _detect_value_shift(
    current: LiveSnapshot,
    state: LiveMatchState,
) -> List[Dict]:
    """Alert when a pre-match probability drops below a threshold.

    If the pre-match moneyline had a team at >65% and now they are
    below the value_shift_drop threshold, the pre-match bet is in danger.
    """
    alerts = []
    drop_threshold = ALERT_THRESHOLDS["value_shift_drop"]

    if current.pre_match_moneyline is None:
        return alerts

    for side in ("home_win", "draw", "away_win"):
        pre = current.pre_match_moneyline.get(side, 0.0)
        live = current.moneyline.get(side, 0.0)

        if pre >= 0.65 and live < drop_threshold:
            side_label = {
                "home_win": state.home_team,
                "draw": "Draw",
                "away_win": state.away_team,
            }[side]
            alert_key = f"value_shift_{side}"

            if _can_fire(state.fixture_id, alert_key):
                alerts.append(_make_alert(
                    "value_shift",
                    f"VALUE SHIFT: {side_label} was {pre:.0%} pre-match, "
                    f"now {live:.0%} at minute {int(state.elapsed_min)}. "
                    f"Pre-match position in danger.",
                    "warning",
                    state.elapsed_min,
                ))
                _mark_fired(state.fixture_id, alert_key)

    return alerts


def _detect_xg_divergence(
    current: LiveSnapshot,
    state: LiveMatchState,
) -> List[Dict]:
    """Alert when live xG significantly differs from actual score.

    A team with high xG but no goals is "due" for regression.
    A team with goals far exceeding xG is overperforming.
    """
    alerts = []
    threshold = ALERT_THRESHOLDS.get("xg_divergence_threshold", 1.5)

    if current.live_xg_home is None or current.live_xg_away is None:
        return alerts

    if state.elapsed_min < 20:
        # Too early for meaningful xG assessment
        return alerts

    for side, xg, goals, team in [
        ("home", current.live_xg_home, state.goals_home, state.home_team),
        ("away", current.live_xg_away, state.goals_away, state.away_team),
    ]:
        divergence = xg - goals
        alert_key = f"xg_divergence_{side}"

        if abs(divergence) >= threshold and _can_fire(state.fixture_id, alert_key):
            if divergence > 0:
                msg = (
                    f"xG DIVERGENCE: {team} has xG={xg:.1f} but only "
                    f"{goals} goal(s). Underperforming — regression expected."
                )
                severity = "info"
            else:
                msg = (
                    f"xG DIVERGENCE: {team} has {goals} goal(s) but xG "
                    f"of only {xg:.1f}. Overperforming — be cautious."
                )
                severity = "warning"

            alerts.append(_make_alert(
                "xg_divergence",
                msg + f" (minute {int(state.elapsed_min)})",
                severity,
                state.elapsed_min,
            ))
            _mark_fired(state.fixture_id, alert_key)

    return alerts


# ---------------------------------------------------------------------------
# Main alert detection entry point
# ---------------------------------------------------------------------------

def detect_alerts(
    current: LiveSnapshot,
    previous: Optional[LiveSnapshot],
    state: LiveMatchState,
) -> List[Dict]:
    """Run all alert detectors and return any triggered alerts.

    This function should be called after ``compute_snapshot()`` to
    populate the snapshot's alerts field.

    Parameters
    ----------
    current : LiveSnapshot
        The just-computed snapshot.
    previous : LiveSnapshot, optional
        The previous snapshot for this fixture (for delta-based alerts).
        Pass None for the first snapshot.
    state : LiveMatchState
        Raw match state for event log access.

    Returns
    -------
    list of dict
        Each dict has keys: type, message, severity, minute.
        Severity is one of: "info", "warning", "critical".

    Examples
    --------
    >>> from live_engine import compute_snapshot
    >>> snapshot = compute_snapshot(state)
    >>> alerts = detect_alerts(snapshot, prev_snapshot, state)
    >>> snapshot.alerts = alerts
    """
    if not state.is_live:
        return []

    alerts: List[Dict] = []

    alerts.extend(_detect_btts_threat(current, state))
    alerts.extend(_detect_total_shift(current, state))
    alerts.extend(_detect_momentum_shift(current, previous, state))
    alerts.extend(_detect_red_card_impact(current, previous, state))
    alerts.extend(_detect_pace_anomaly(current, state))
    alerts.extend(_detect_value_shift(current, state))
    alerts.extend(_detect_xg_divergence(current, state))

    return alerts
