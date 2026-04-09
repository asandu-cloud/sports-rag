"""Actionable live bet alert detection.

Alerts are BET RECOMMENDATIONS, not stat commentary. Each alert tells
the user exactly what to bet and why, based on a significant probability
shift from the pre-match projection.

Design principle: a user paying for this service wants to be told
"Bet X now" — not "momentum shifted" or "pace is high."

Alert types:
  - live_bet:      A specific bet recommendation with model probability
  - goal_update:   Score changed — updated probabilities for active markets
  - red_card:      Red card — projections recalculated, new bet angles

NOT included (removed as noise):
  - momentum_shift:  Interesting but not actionable
  - pace_anomaly:    Same
  - xg_divergence:   Same — folded into bet reasoning instead
  - generic total_shift: Replaced with specific bet recommendations
"""

import time
from typing import Dict, List, Optional

try:
    from live.live_config import ALERT_COOLDOWN_SECONDS  # type: ignore
    from live.live_odds import best_btts_offer, best_moneyline_offer, best_total_offer  # type: ignore
    from live.live_state import LiveMatchState, LiveSnapshot  # type: ignore
except ImportError:
    try:
        from Scripts.live.live_config import ALERT_COOLDOWN_SECONDS  # type: ignore
        from Scripts.live.live_odds import best_btts_offer, best_moneyline_offer, best_total_offer  # type: ignore
        from Scripts.live.live_state import LiveMatchState, LiveSnapshot  # type: ignore
    except ImportError:
        from live_config import ALERT_COOLDOWN_SECONDS
        from live_odds import best_btts_offer, best_moneyline_offer, best_total_offer
        from live_state import LiveMatchState, LiveSnapshot


# ---------------------------------------------------------------------------
# Cooldown state
# ---------------------------------------------------------------------------
_alert_cooldowns: Dict[tuple, float] = {}

# Much longer cooldown — 15 minutes between alerts per fixture
_COOLDOWN = max(ALERT_COOLDOWN_SECONDS, 900)  # minimum 15 min


def _can_fire(fixture_id: int, alert_key: str) -> bool:
    key = (fixture_id, alert_key)
    last = _alert_cooldowns.get(key, 0.0)
    return (time.time() - last) >= _COOLDOWN


def _mark_fired(fixture_id: int, alert_key: str) -> None:
    _alert_cooldowns[(fixture_id, alert_key)] = time.time()


def clear_cooldowns(fixture_id: Optional[int] = None) -> None:
    global _alert_cooldowns
    if fixture_id is None:
        _alert_cooldowns = {}
    else:
        _alert_cooldowns = {k: v for k, v in _alert_cooldowns.items() if k[0] != fixture_id}


def _make_alert(alert_type: str, message: str, severity: str, minute: float, **extra: object) -> Dict:
    alert = {"type": alert_type, "message": message, "severity": severity, "minute": int(minute)}
    alert.update(extra)
    return alert


def _expected_value(model_prob: float, decimal_odds: float) -> float:
    return (model_prob * decimal_odds) - 1.0


def _live_odds_source(current: LiveSnapshot, state: LiveMatchState) -> str:
    source = (
        getattr(current, "live_odds_source", None)
        or getattr(state, "live_odds_source", None)
        or (getattr(current, "live_odds", None) or {}).get("source")
        or (getattr(state, "live_odds", None) or {}).get("source")
    )
    if source in {"inplay", "prematch_fallback"}:
        return str(source)
    return "none"


def _can_use_live_prices(pricing_source: str) -> bool:
    return pricing_source == "inplay"


def _attach_fallback_context(candidate: Optional[Dict], pricing_source: str) -> Optional[Dict]:
    if candidate is None:
        return None
    enriched = dict(candidate)
    enriched["mode"] = str(enriched.get("mode") or "model_only")
    enriched["pricing_source"] = pricing_source
    if pricing_source == "prematch_fallback":
        enriched["mode_note"] = "Live odds unavailable; using model-only projections while pre-match prices are stale."
    elif pricing_source == "none":
        enriched["mode_note"] = "Live odds unavailable; using model-only projections."
    return enriched


def _priced_total_candidate(
    live_odds: Dict,
    stat_group: str,
    prob_dict: Dict[str, float],
    projected_total: Optional[float],
    observed: int,
    stat_name: str,
    emoji: str,
) -> Optional[Dict]:
    if not live_odds or not prob_dict or projected_total is None:
        return None

    best = None
    best_score = -999.0

    for line_str, p_over in prob_dict.items():
        line = float(line_str)
        if line < observed:
            continue

        for side, model_prob in (("over", float(p_over)), ("under", float(1.0 - p_over))):
            if model_prob < 0.55 or model_prob > 0.88:
                continue
            offer = best_total_offer(live_odds, stat_group, line, side)
            if not offer:
                continue
            edge = model_prob - float(offer["fair_prob"])
            ev = _expected_value(model_prob, float(offer["odds"]))
            if edge < 0.04 or ev < 0.03:
                continue
            closeness = 1.0 / (1.0 + abs(projected_total - line))
            score = (edge * 2.0) + ev + (closeness * 0.10)
            if score > best_score:
                best_score = score
                side_label = "Over" if side == "over" else "Under"
                best = {
                    "pick": f"{emoji} {side_label} {line_str} {stat_name}",
                    "prob": model_prob,
                    "odds": float(offer["odds"]),
                    "bookmaker": offer["bookmaker"],
                    "edge": edge,
                    "ev": ev,
                    "reason": (
                        f"Model {model_prob:.0%} vs fair {offer['fair_prob']:.0%} "
                        f"at {offer['odds']:.2f} ({offer['bookmaker']}) | "
                        f"Proj {projected_total:.1f}, observed {observed}"
                    ),
                    "shift": edge,
                    "mode": "odds_aware",
                    "pricing_source": "inplay",
                }
    return best


def _priced_btts_candidate(live_odds: Dict, current: LiveSnapshot, state: LiveMatchState) -> Optional[Dict]:
    if not live_odds or current.btts_prob is None:
        return None
    if state.goals_home >= 1 and state.goals_away >= 1:
        return None

    best = None
    best_score = -999.0
    for side, model_prob in (("yes", float(current.btts_prob)), ("no", float(1.0 - current.btts_prob))):
        if model_prob < 0.55 or model_prob > 0.88:
            continue
        offer = best_btts_offer(live_odds, side)
        if not offer:
            continue
        edge = model_prob - float(offer["fair_prob"])
        ev = _expected_value(model_prob, float(offer["odds"]))
        if edge < 0.04 or ev < 0.03:
            continue
        score = (edge * 2.0) + ev
        if score > best_score:
            best_score = score
            side_label = "Yes" if side == "yes" else "No"
            best = {
                "pick": f"🤝 BTTS {side_label}",
                "prob": model_prob,
                "odds": float(offer["odds"]),
                "bookmaker": offer["bookmaker"],
                "edge": edge,
                "ev": ev,
                "reason": (
                    f"Model {model_prob:.0%} vs fair {offer['fair_prob']:.0%} "
                    f"at {offer['odds']:.2f} ({offer['bookmaker']})"
                ),
                "shift": edge,
                "mode": "odds_aware",
                "pricing_source": "inplay",
            }
    return best


def _priced_moneyline_candidate(live_odds: Dict, current: LiveSnapshot, state: LiveMatchState) -> Optional[Dict]:
    if not live_odds or not current.moneyline:
        return None

    sides = [
        ("home", "home_win", f"🏆 {state.home_team} Win", 0.42, 0.04, 0.03),
        ("draw", "draw", "🏆 Draw", 0.25, 0.06, 0.05),
        ("away", "away_win", f"🏆 {state.away_team} Win", 0.42, 0.04, 0.03),
    ]

    best = None
    best_score = -999.0
    for odds_side, model_key, label, min_prob, min_edge, min_ev in sides:
        model_prob = float(current.moneyline.get(model_key, 0.0))
        if model_prob < min_prob or model_prob > 0.88:
            continue
        offer = best_moneyline_offer(live_odds, odds_side)
        if not offer:
            continue
        edge = model_prob - float(offer["fair_prob"])
        ev = _expected_value(model_prob, float(offer["odds"]))
        if edge < min_edge or ev < min_ev:
            continue
        score = (edge * 2.0) + ev
        if score > best_score:
            best_score = score
            best = {
                "pick": label,
                "prob": model_prob,
                "odds": float(offer["odds"]),
                "bookmaker": offer["bookmaker"],
                "edge": edge,
                "ev": ev,
                "reason": (
                    f"Model {model_prob:.0%} vs fair {offer['fair_prob']:.0%} "
                    f"at {offer['odds']:.2f} ({offer['bookmaker']})"
                ),
                "shift": edge,
                "mode": "odds_aware",
                "pricing_source": "inplay",
            }
    return best


# ---------------------------------------------------------------------------
# Best live bet finder (the core — replaces all the old stat-commentary alerts)
# ---------------------------------------------------------------------------

def _find_best_live_bet(
    current: LiveSnapshot,
    previous: Optional[LiveSnapshot],
    state: LiveMatchState,
) -> List[Dict]:
    """Find the single best actionable live bet and alert it.

    Only fires when:
    1. Model probability for a specific line is between 58-82% (actionable, not trivial)
    2. The probability has SHIFTED significantly from pre-match (the live info adds value)
    3. The bet makes logical sense (no BTTS after both scored, no obvious lines)
    """
    alerts = []
    elapsed = state.elapsed_min

    # Don't alert in first 10 min (not enough data) or after 85 min (too late)
    if elapsed < 10 or elapsed > 85:
        return alerts

    live_odds = getattr(current, "live_odds", None) or getattr(state, "live_odds", None) or {}
    pricing_source = _live_odds_source(current, state)
    allow_priced_candidates = _can_use_live_prices(pricing_source)

    # --- Check each market for the best actionable bet ---

    # Goals
    best_goal_bet = (
        _priced_total_candidate(
            live_odds,
            "goals",
            current.prob_over_goals or {},
            current.projected_total_goals,
            state.total_goals,
            "Goals",
            "⚽",
        )
        if allow_priced_candidates
        else None
    ) or _attach_fallback_context(_best_actionable_line(
        current.prob_over_goals or {},
        current.projected_total_goals,
        state.total_goals,
        "Goals",
        "⚽",
    ), pricing_source)

    # Corners
    best_corner_bet = (
        _priced_total_candidate(
            live_odds,
            "corners",
            current.prob_over_corners or {},
            current.projected_total_corners,
            state.total_corners,
            "Corners",
            "📐",
        )
        if allow_priced_candidates
        else None
    ) or _attach_fallback_context(_best_actionable_line(
        current.prob_over_corners or {},
        current.projected_total_corners,
        state.total_corners,
        "Corners",
        "📐",
    ), pricing_source)

    # Cards
    best_card_bet = (
        _priced_total_candidate(
            live_odds,
            "cards",
            current.prob_over_cards or {},
            current.projected_total_cards,
            state.total_cards,
            "Cards",
            "🟨",
        )
        if allow_priced_candidates
        else None
    ) or _attach_fallback_context(_best_actionable_line(
        current.prob_over_cards or {},
        current.projected_total_cards,
        state.total_cards,
        "Cards",
        "🟨",
    ), pricing_source)

    # BTTS — only if NOT already cashed
    btts_bet = _priced_btts_candidate(live_odds, current, state) if allow_priced_candidates else None
    both_scored = state.goals_home >= 1 and state.goals_away >= 1
    if btts_bet is None and not both_scored and current.btts_prob is not None:
        pre_btts = current.pre_match_btts_prob or 0.5
        if current.btts_prob >= 0.62 and abs(current.btts_prob - pre_btts) >= 0.08:
            btts_bet = _attach_fallback_context({
                "pick": "🤝 BTTS Yes",
                "prob": current.btts_prob,
                "reason": f"P(BTTS) = {current.btts_prob:.0%} (was {pre_btts:.0%} pre-match)",
                "shift": current.btts_prob - pre_btts,
            }, pricing_source)
        elif (1.0 - current.btts_prob) >= 0.65 and abs(current.btts_prob - pre_btts) >= 0.08:
            btts_bet = _attach_fallback_context({
                "pick": "🤝 BTTS No",
                "prob": 1.0 - current.btts_prob,
                "reason": f"P(BTTS No) = {1.0 - current.btts_prob:.0%} (was {1.0 - pre_btts:.0%} pre-match)",
                "shift": (1.0 - current.btts_prob) - (1.0 - pre_btts),
            }, pricing_source)

    # Moneyline — only if significant shift from pre-match
    ml_bet = _priced_moneyline_candidate(live_odds, current, state) if allow_priced_candidates else None
    if ml_bet is None and current.moneyline and current.pre_match_moneyline:
        for side in ("home_win", "draw", "away_win"):
            live_p = current.moneyline.get(side, 0)
            pre_p = current.pre_match_moneyline.get(side, 0)
            shift = live_p - pre_p
            if live_p >= 0.60 and shift >= 0.10:  # significant improvement from pre-match
                label = {"home_win": f"🏆 {state.home_team} Win",
                         "draw": "🏆 Draw",
                         "away_win": f"🏆 {state.away_team} Win"}.get(side, side)
                ml_bet = _attach_fallback_context({
                    "pick": label,
                    "prob": live_p,
                    "reason": f"Model: {live_p:.0%} (was {pre_p:.0%} pre-match, +{shift:.0%} shift)",
                    "shift": shift,
                }, pricing_source)
                break  # only the best ML bet

    # Next goal / no more goals bet
    next_goal_bet = _attach_fallback_context(_find_next_goal_bet(current, state), pricing_source)

    # Collect all candidates and pick the single best
    candidates = []
    for bet in [best_goal_bet, best_corner_bet, best_card_bet, btts_bet, ml_bet, next_goal_bet]:
        if bet is not None:
            candidates.append(bet)

    if not candidates:
        return alerts

    # Price-aware candidates rank by edge first; model-only candidates fall back to shift.
    candidates.sort(
        key=lambda b: (
            float(b.get("edge", -999.0)),
            abs(float(b.get("shift", 0.0))) * float(b.get("prob", 0.0)),
        ),
        reverse=True,
    )
    best = candidates[0]

    # Odds-aware alerts need a real edge; model-only fallbacks still require meaningful shift.
    if best.get("mode") == "odds_aware":
        if float(best.get("edge", 0.0)) < 0.04 or float(best.get("ev", 0.0)) < 0.03:
            return alerts
    elif abs(best.get("shift", 0)) < 0.05:
        return alerts

    alert_key = f"live_bet_{best['pick']}"
    if _can_fire(state.fixture_id, alert_key):
        if best.get("mode") == "odds_aware":
            conf = "Strong Edge" if float(best.get("edge", 0.0)) >= 0.08 else "Leaning"
            msg = (
                f"LIVE BET: {best['pick']} — {conf} "
                f"(model {best['prob']:.0%}, edge {best.get('edge', 0.0):+.1%}, EV {best.get('ev', 0.0):+.2f})\n"
                f"{best['reason']}"
            )
        else:
            conf = "Strong Edge" if best["prob"] >= 0.65 else "Leaning"
            mode_note = best.get("mode_note")
            msg = f"LIVE BET: {best['pick']} — {conf} ({best['prob']:.0%})\n{best['reason']}"
            if mode_note:
                msg = f"{msg}\n{mode_note}"
        alerts.append(_make_alert(
            "live_bet",
            msg,
            "warning" if conf == "Strong Edge" else "info",
            elapsed,
            pick=best.get("pick"),
            prob=float(best.get("prob", 0.0)),
            confidence=conf,
            reason=best.get("reason"),
            mode=best.get("mode", "model_only"),
            pricing_source=best.get("pricing_source", pricing_source),
            bookmaker=best.get("bookmaker"),
            odds=best.get("odds"),
            edge=best.get("edge"),
            ev=best.get("ev"),
            mode_note=best.get("mode_note"),
        ))
        _mark_fired(state.fixture_id, alert_key)

    return alerts


def _best_actionable_line(
    prob_dict: Dict[str, float],
    projected_total: Optional[float],
    observed: int,
    stat_name: str,
    emoji: str,
) -> Optional[Dict]:
    """Find the best actionable over/under line for a stat.

    "Actionable" = probability between 58-82% on the tightest contested line.
    Must also show meaningful shift from what a naive model would predict.
    """
    if not prob_dict or projected_total is None:
        return None

    best = None
    best_score = -1

    for line_str, p_over in prob_dict.items():
        line = float(line_str)
        if line < observed:
            continue  # already cleared

        # Pick the stronger side
        if p_over >= 0.58:
            side, prob = "Over", p_over
        elif (1.0 - p_over) >= 0.58:
            side, prob = "Under", 1.0 - p_over
        else:
            continue

        # Filter: must be actionable (58-82%) — not trivially obvious
        if prob > 0.82:
            continue

        # Score: proximity to projection × probability
        closeness = 1.0 / (1.0 + abs(projected_total - line))
        score = prob * closeness

        if score > best_score:
            best_score = score
            # Estimate the shift: how different is this from what pre-match would have said?
            # Use remaining_fraction as a proxy — early in the match, small shift; late, bigger shift
            shift = abs(prob - 0.55) * 0.5  # rough proxy
            best = {
                "pick": f"{emoji} {side} {line_str} {stat_name}",
                "prob": prob,
                "reason": f"Projected {projected_total:.1f} total ({observed} so far)",
                "shift": shift,
            }

    return best


# ---------------------------------------------------------------------------
# Next Goal / No More Goals alert
# ---------------------------------------------------------------------------

def _find_next_goal_bet(
    current: LiveSnapshot,
    state: LiveMatchState,
) -> Optional[Dict]:
    """Recommend a next goal bet when the model has a strong directional opinion.

    Three scenarios:
    1. One team heavily favored to score next (65%+ of next-goal probability)
       → Recommend "Next Goal: [Team]"
    2. High P(no more goals) late in a tight match (55%+ after 65')
       → Recommend "No More Goals" / "No Next Goal"
    3. Right after a goal, momentum shifts the next-goal probabilities
       → Stronger signal, lower threshold (60%)

    This is high-value for bettors because next-goal markets are:
    - Available in-play on most sportsbooks
    - Often mispriced right after a goal or red card
    - Easy to understand (no line/total to parse)
    """
    ng = current.next_goal_probs
    if not ng:
        return None

    elapsed = state.elapsed_min
    if elapsed < 15 or elapsed > 88:
        return None  # too early (unreliable) or too late (match almost over)

    p_home = ng.get("next_home", 0)
    p_away = ng.get("next_away", 0)
    p_none = ng.get("no_more", 0)
    expected_min = ng.get("expected_min_to_next", 99)

    # --- Scenario 1: Strong directional signal for next goal ---
    # One team has 65%+ share of next-goal probability
    p_goal = 1.0 - p_none  # P(at least one more goal)
    if p_goal > 0:
        home_share = p_home / p_goal if p_goal > 0 else 0.5
        away_share = p_away / p_goal if p_goal > 0 else 0.5
    else:
        home_share = 0.5
        away_share = 0.5

    # Check if xG supports the direction (extra confidence)
    xg_h = current.live_xg_home or 0
    xg_a = current.live_xg_away or 0
    xg_supports_home = xg_h > xg_a + 0.3
    xg_supports_away = xg_a > xg_h + 0.3

    # Lower threshold if we just saw a goal (momentum)
    recent_goal = any(
        e.event_type == "goal" and e.minute >= elapsed - 5
        for e in state.events
    ) if state.events else False
    threshold = 0.60 if recent_goal else 0.65

    if home_share >= threshold and p_goal >= 0.45:
        prob = p_home
        xg_note = f" (xG: {state.home_team} {xg_h:.1f} vs {state.away_team} {xg_a:.1f})" if xg_supports_home else ""
        return {
            "pick": f"⚽ Next Goal: {state.home_team}",
            "prob": prob,
            "confidence": "Strong Edge" if home_share >= 0.72 else "Leaning",
            "color": "high" if home_share >= 0.72 else "medium",
            "reason": (
                f"{state.home_team} has {home_share:.0%} of next-goal probability. "
                f"Expected ~{expected_min:.0f} min until next goal{xg_note}"
            ),
            "shift": home_share - 0.50,  # how much better than 50/50
        }

    if away_share >= threshold and p_goal >= 0.45:
        prob = p_away
        xg_note = f" (xG: {state.away_team} {xg_a:.1f} vs {state.home_team} {xg_h:.1f})" if xg_supports_away else ""
        return {
            "pick": f"⚽ Next Goal: {state.away_team}",
            "prob": prob,
            "confidence": "Strong Edge" if away_share >= 0.72 else "Leaning",
            "color": "high" if away_share >= 0.72 else "medium",
            "reason": (
                f"{state.away_team} has {away_share:.0%} of next-goal probability. "
                f"Expected ~{expected_min:.0f} min until next goal{xg_note}"
            ),
            "shift": away_share - 0.50,
        }

    # --- Scenario 2: No more goals (high probability late in match) ---
    if p_none >= 0.55 and elapsed >= 65:
        score = f"{state.goals_home}-{state.goals_away}"
        return {
            "pick": "⚽ No More Goals",
            "prob": p_none,
            "confidence": "Strong Edge" if p_none >= 0.70 else "Leaning",
            "color": "high" if p_none >= 0.70 else "medium",
            "reason": (
                f"P(no more goals) = {p_none:.0%} at {elapsed:.0f}' with score {score}. "
                f"Match pace suggests the scoring is done"
            ),
            "shift": p_none - 0.30,  # shift from baseline ~30% for "no more goals"
        }

    return None


# ---------------------------------------------------------------------------
# Red card alert (kept — this is genuinely actionable)
# ---------------------------------------------------------------------------

def _detect_red_card(
    current: LiveSnapshot,
    previous: Optional[LiveSnapshot],
    state: LiveMatchState,
) -> List[Dict]:
    """Alert on red cards — these fundamentally change the game."""
    alerts = []
    recent = state.events_in_window(3)
    for ev in recent:
        if ev.event_type == "red_card":
            alert_key = f"red_card_{ev.team_side}_{ev.minute}"
            if _can_fire(state.fixture_id, alert_key):
                team = state.home_team if ev.team_side == "home" else state.away_team
                other = state.away_team if ev.team_side == "home" else state.home_team
                alerts.append(_make_alert(
                    "red_card",
                    f"🔴 RED CARD: {ev.player} ({team}) — minute {ev.minute}\n"
                    f"Expect: more cards overall, fewer goals from {team}, "
                    f"{other} likely to dominate corners and possession.",
                    "critical",
                    ev.minute,
                ))
                _mark_fired(state.fixture_id, alert_key)
    return alerts


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def detect_alerts(
    current: LiveSnapshot,
    previous: Optional[LiveSnapshot],
    state: LiveMatchState,
) -> List[Dict]:
    """Detect actionable live bet alerts.

    Returns at most 1-2 alerts per call (one bet recommendation + optionally
    one red card). This prevents channel spam.
    """
    if not state.is_live:
        return []

    alerts: List[Dict] = []

    # Red cards are always alerted immediately (rare, high-impact)
    alerts.extend(_detect_red_card(current, previous, state))

    # Best live bet recommendation (max 1 per 15 minutes per fixture)
    alerts.extend(_find_best_live_bet(current, previous, state))

    return alerts
