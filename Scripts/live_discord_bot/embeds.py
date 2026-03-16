"""Discord embed builders for live match data.

Creates rich, visual embeds for live alerts, scoreboards, and fixture lists.
Consumes LiveSnapshot and LiveMatchState from the live engine.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import discord

from config import (
    COLOR_BLUE,
    COLOR_GREEN,
    COLOR_ORANGE,
    COLOR_RED,
    COLOR_YELLOW,
)


# ---------------------------------------------------------------------------
# Severity -> color + emoji mapping
# ---------------------------------------------------------------------------

_SEVERITY_MAP = {
    "critical": (COLOR_RED, "\U0001f534"),       # red circle
    "warning":  (COLOR_ORANGE, "\u26a0\ufe0f"),  # warning sign
    "info":     (COLOR_BLUE, "\u2139\ufe0f"),     # info
}

# Event type -> emoji
_EVENT_EMOJI = {
    "goal":        "\u26bd",  # soccer ball
    "yellow_card": "\U0001f7e8",  # yellow square
    "red_card":    "\U0001f7e5",  # red square
    "substitution": "\U0001f504",  # arrows
    "var":         "\U0001f4fa",  # TV
    "corner":      "\U0001f6a9",  # flag
    "penalty_missed": "\u274c",   # cross mark
}


def _pct(val: Optional[float], default: str = "—") -> str:
    """Format a probability as a percentage string."""
    if val is None:
        return default
    return f"{val:.0%}"


def _f1(val: Optional[float], default: str = "—") -> str:
    """Format a float to 1 decimal place."""
    if val is None:
        return default
    return f"{val:.1f}"


def _f2(val: Optional[float], default: str = "—") -> str:
    """Format a float to 2 decimal places."""
    if val is None:
        return default
    return f"{val:.2f}"


def _momentum_bar(home_share: float, width: int = 10) -> str:
    """Build a text-based momentum bar."""
    filled = max(0, min(width, round(home_share * width)))
    empty = width - filled
    return "\u2588" * filled + "\u2591" * empty


def _format_event(ev: Dict[str, Any]) -> str:
    """Format a single match event for display."""
    minute = ev.get("minute", "?")
    etype = ev.get("type", ev.get("event_type", ""))
    emoji = _EVENT_EMOJI.get(etype, "\u2022")
    player = ev.get("player", "")
    detail = ev.get("detail", "")
    parts = [f"{minute}'", emoji]
    if player:
        parts.append(player)
    if detail and detail not in ("Normal Goal", "Yellow Card"):
        parts.append(f"({detail})")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# A. Live Alert Embed
# ---------------------------------------------------------------------------

def live_alert_embed(
    alert: Dict[str, Any],
    snapshot: Any,
    state: Any,
) -> discord.Embed:
    """Build a rich embed for a single live alert.

    Parameters
    ----------
    alert : dict
        Alert dict from detect_alerts() with keys: type, message, severity, minute.
    snapshot : LiveSnapshot
        Current projection snapshot for the fixture.
    state : LiveMatchState (poller version)
        Raw match state for additional context.
    """
    severity = alert.get("severity", "info")
    color, sev_emoji = _SEVERITY_MAP.get(severity, (COLOR_BLUE, "\u2139\ufe0f"))
    alert_type = alert.get("type", "alert")
    minute = alert.get("minute", "?")

    # Fixture header
    home = getattr(snapshot, "home_team", None) or getattr(state, "home_team", "?")
    away = getattr(snapshot, "away_team", None) or getattr(state, "away_team", "?")
    score = getattr(snapshot, "score", None)
    if score is None:
        h_goals = getattr(state, "home_stats", None)
        a_goals = getattr(state, "away_stats", None)
        if h_goals and a_goals:
            score = f"{h_goals.goals}-{a_goals.goals}"
        else:
            score = "?-?"
    league = getattr(snapshot, "league", None) or getattr(state, "league", "")

    title = f"{sev_emoji} LIVE ALERT \u2014 {home} {score} {away} ({minute}')"

    em = discord.Embed(
        title=title,
        color=color,
    )

    # Alert message
    em.add_field(
        name=f"\u26a0 {_alert_type_label(alert_type)}",
        value=alert.get("message", ""),
        inline=False,
    )

    # Live stats section
    stats_lines = _build_stats_section(snapshot, state)
    if stats_lines:
        em.add_field(
            name="\U0001f4ca Live Stats",
            value=stats_lines,
            inline=False,
        )

    # Action suggestion based on alert type
    action = _suggest_action(alert_type, alert, snapshot)
    if action:
        em.add_field(
            name="\U0001f4a1 Action",
            value=action,
            inline=False,
        )

    # Footer with pre-match comparison
    footer_parts = ["Matchwise Live"]
    if league:
        footer_parts.append(league)
    pre_goals = getattr(snapshot, "pre_match_total_goals", None)
    proj_goals = getattr(snapshot, "projected_total_goals", None)
    if pre_goals is not None and proj_goals is not None:
        footer_parts.append(f"Pre-match: {_f1(pre_goals)}, Now: {_f1(proj_goals)}")
    em.set_footer(text=" | ".join(footer_parts))

    return em


def _alert_type_label(alert_type: str) -> str:
    """Human-readable label for alert type."""
    labels = {
        "btts_threat": "BTTS Under Threat",
        "total_shift": "Total Probability Shift",
        "momentum_shift": "Momentum Swing",
        "red_card_impact": "Red Card Impact",
        "pace_anomaly": "Pace Anomaly",
        "value_shift": "Value Drift",
        "xg_divergence": "xG Divergence",
        "goal": "GOAL",
        "match_start": "Match Started",
        "full_time": "Full Time",
    }
    return labels.get(alert_type, alert_type.replace("_", " ").title())


def _build_stats_section(snapshot: Any, state: Any) -> str:
    """Build a formatted live stats string."""
    lines = []

    # Try to extract stats from the poller's state (which uses TeamLiveStats)
    hs = getattr(state, "home_stats", None)
    as_ = getattr(state, "away_stats", None)
    home = getattr(state, "home_team", "Home")
    away = getattr(state, "away_team", "Away")

    if hs and as_:
        sot_h = getattr(hs, "sot", 0)
        sot_a = getattr(as_, "sot", 0)
        shots_h = getattr(hs, "shots", 0)
        shots_a = getattr(as_, "shots", 0)
        poss_h = getattr(hs, "possession", 50.0)
        poss_a = getattr(as_, "possession", 50.0)
        corners_h = getattr(hs, "corners", 0)
        corners_a = getattr(as_, "corners", 0)

        lines.append(f"  {home}: {sot_h} SoT, {shots_h} shots, {poss_h:.0f}% poss, {corners_h} corners")
        lines.append(f"  {away}: {sot_a} SoT, {shots_a} shots, {poss_a:.0f}% poss, {corners_a} corners")
    else:
        # Fallback: try flat fields from live_state.LiveMatchState
        for attr_set, team_label in [
            (("sot_home", "shots_home", "possession_home", "corners_home"), home),
            (("sot_away", "shots_away", "possession_away", "corners_away"), away),
        ]:
            vals = [getattr(state, a, None) for a in attr_set]
            if any(v is not None for v in vals):
                sot, shots, poss, corners = [v or 0 for v in vals]
                lines.append(f"  {team_label}: {sot} SoT, {shots} shots, {poss:.0f}% poss, {corners} corners")

    return "\n".join(lines) if lines else ""


def _suggest_action(alert_type: str, alert: Dict, snapshot: Any) -> str:
    """Suggest an action based on alert type."""
    actions = {
        "btts_threat": "Consider cashing out BTTS Yes positions",
        "total_shift": "Review over/under positions against live probabilities",
        "momentum_shift": "Watch for live odds movement in the next 5 minutes",
        "red_card_impact": "Expect tactical adjustments; card and goal projections recalculated",
        "pace_anomaly": "Match tempo deviating from model; projections adjusted automatically",
        "value_shift": "Pre-match position at risk; evaluate cashout options",
        "xg_divergence": "Score does not reflect chances created; regression expected",
    }
    return actions.get(alert_type, "")


# ---------------------------------------------------------------------------
# B. Live Scoreboard Embed
# ---------------------------------------------------------------------------

def live_scoreboard_embed(
    snapshot: Any,
    state: Any,
) -> discord.Embed:
    """Full match overview card with all live projections.

    Parameters
    ----------
    snapshot : LiveSnapshot
        Current projection snapshot (from live_engine or poller).
    state : LiveMatchState (poller version)
        Raw match state for supplementary data.
    """
    home = getattr(snapshot, "home_team", None) or getattr(state, "home_team", "?")
    away = getattr(snapshot, "away_team", None) or getattr(state, "away_team", "?")
    score = getattr(snapshot, "score", None)
    if score is None:
        hs = getattr(state, "home_stats", None)
        as_ = getattr(state, "away_stats", None)
        if hs and as_:
            score = f"{hs.goals} - {as_.goals}"
        else:
            score = "? - ?"
    elapsed = getattr(snapshot, "elapsed_min", 0) or getattr(state, "elapsed_min", 0)
    status = getattr(snapshot, "status", "") or getattr(state, "status", "")
    league = getattr(snapshot, "league", "") or getattr(state, "league", "")

    half_label = {"1H": "1H", "HT": "HT", "2H": "2H", "ET": "ET", "FT": "FT"}.get(status, status)
    title = f"\u26bd {home} {score} {away}    {elapsed}' {half_label}"

    em = discord.Embed(title=title, color=COLOR_BLUE)

    # --- Goals ---
    proj_goals = getattr(snapshot, "projected_total_goals", None)
    pre_goals = getattr(snapshot, "pre_match_total_goals", None)
    rem_goals = getattr(snapshot, "remaining_goals", None)
    observed_goals = _observed_count(state, "goals")
    prob_goals = getattr(snapshot, "prob_over_goals", {}) or {}

    goals_lines = []
    if proj_goals is not None:
        obs_str = f"{observed_goals} observed"
        rem_str = f"+ {_f1(rem_goals)} remaining" if rem_goals is not None else ""
        total_str = f"= {_f1(proj_goals)} total"
        was_str = f"(was {_f1(pre_goals)})" if pre_goals is not None else ""
        goals_lines.append(f"{obs_str} {rem_str} {total_str} {was_str}")
    p25 = prob_goals.get("2.5")
    p15 = prob_goals.get("1.5")
    if p25 is not None or p15 is not None:
        parts = []
        if p25 is not None:
            parts.append(f"P(O2.5): {_pct(p25)}")
        if p15 is not None:
            parts.append(f"P(O1.5): {_pct(p15)}")
        goals_lines.append(" | ".join(parts))
    if goals_lines:
        em.add_field(name="Goals", value="\n".join(goals_lines), inline=False)

    # --- Corners ---
    proj_corners = getattr(snapshot, "projected_total_corners", None)
    pre_corners = getattr(snapshot, "pre_match_total_corners", None)
    rem_corners = getattr(snapshot, "remaining_corners", None)
    observed_corners = _observed_count(state, "corners")
    prob_corners = getattr(snapshot, "prob_over_corners", {}) or {}

    corners_lines = []
    if proj_corners is not None:
        obs_str = f"{observed_corners} observed"
        rem_str = f"+ {_f1(rem_corners)} remaining" if rem_corners is not None else ""
        total_str = f"= {_f1(proj_corners)} total"
        was_str = f"(was {_f1(pre_corners)})" if pre_corners is not None else ""
        corners_lines.append(f"{obs_str} {rem_str} {total_str} {was_str}")
    p105 = prob_corners.get("10.5")
    p95 = prob_corners.get("9.5")
    if p105 is not None or p95 is not None:
        parts = []
        if p105 is not None:
            parts.append(f"P(O10.5): {_pct(p105)}")
        if p95 is not None:
            parts.append(f"P(O9.5): {_pct(p95)}")
        corners_lines.append(" | ".join(parts))
    if corners_lines:
        em.add_field(name="Corners", value="\n".join(corners_lines), inline=False)

    # --- Cards ---
    proj_cards = getattr(snapshot, "projected_total_cards", None)
    pre_cards = getattr(snapshot, "pre_match_total_cards", None)
    rem_cards = getattr(snapshot, "remaining_cards", None)
    observed_cards = _observed_count(state, "cards")
    prob_cards = getattr(snapshot, "prob_over_cards", {}) or {}

    cards_lines = []
    if proj_cards is not None:
        obs_str = f"{observed_cards} observed"
        rem_str = f"+ {_f1(rem_cards)} remaining" if rem_cards is not None else ""
        total_str = f"= {_f1(proj_cards)} total"
        was_str = f"(was {_f1(pre_cards)})" if pre_cards is not None else ""
        cards_lines.append(f"{obs_str} {rem_str} {total_str} {was_str}")
    p35 = prob_cards.get("3.5")
    p45 = prob_cards.get("4.5")
    if p35 is not None or p45 is not None:
        parts = []
        if p35 is not None:
            parts.append(f"P(O3.5): {_pct(p35)}")
        if p45 is not None:
            parts.append(f"P(O4.5): {_pct(p45)}")
        cards_lines.append(" | ".join(parts))
    if cards_lines:
        em.add_field(name="Cards", value="\n".join(cards_lines), inline=False)

    # --- BTTS + Moneyline row ---
    btts_prob = getattr(snapshot, "btts_prob", None)
    pre_btts = getattr(snapshot, "pre_match_btts_prob", None)
    ml = getattr(snapshot, "moneyline", None) or {}
    btts_ml_parts = []
    if btts_prob is not None:
        btts_str = f"BTTS {_pct(btts_prob)}"
        if pre_btts is not None:
            btts_str += f" (was {_pct(pre_btts)})"
        btts_ml_parts.append(btts_str)
    if ml:
        ml_str = (
            f"Moneyline  Home {_pct(ml.get('home_win'))} | "
            f"Draw {_pct(ml.get('draw'))} | "
            f"Away {_pct(ml.get('away_win'))}"
        )
        btts_ml_parts.append(ml_str)
    if btts_ml_parts:
        em.add_field(name="Markets", value="\n".join(btts_ml_parts), inline=False)

    # --- Momentum + xG row ---
    momentum = getattr(snapshot, "momentum", None) or {}
    xg_home = getattr(snapshot, "live_xg_home", None)
    xg_away = getattr(snapshot, "live_xg_away", None)
    mom_xg_parts = []
    home_share = momentum.get("home", 0.5)
    bar = _momentum_bar(home_share)
    mom_xg_parts.append(f"Momentum  {bar} {home} {_pct(home_share)}")
    if xg_home is not None and xg_away is not None:
        mom_xg_parts.append(f"xG  {home} {_f2(xg_home)} | {away} {_f2(xg_away)}")
    if mom_xg_parts:
        em.add_field(name="Dynamics", value="\n".join(mom_xg_parts), inline=False)

    # --- Recent events ---
    recent = getattr(snapshot, "recent_events", None) or []
    if recent:
        event_strs = [_format_event(ev) for ev in recent[:8]]
        em.add_field(
            name="\U0001f4cb Events",
            value=" | ".join(event_strs),
            inline=False,
        )

    # Footer
    ts = getattr(snapshot, "timestamp", "")
    time_str = ""
    if ts:
        try:
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            time_str = f"Updated {dt.strftime('%H:%M')} UTC"
        except Exception:
            time_str = ""
    footer_parts = ["Matchwise Live"]
    if league:
        footer_parts.append(league)
    if time_str:
        footer_parts.append(time_str)
    em.set_footer(text=" | ".join(footer_parts))

    return em


def _observed_count(state: Any, stat: str) -> int:
    """Extract observed count from state (handles both poller and engine state shapes)."""
    # Poller state has home_stats/away_stats sub-objects
    hs = getattr(state, "home_stats", None)
    as_ = getattr(state, "away_stats", None)

    if hs and as_:
        if stat == "goals":
            return getattr(hs, "goals", 0) + getattr(as_, "goals", 0)
        elif stat == "corners":
            return getattr(hs, "corners", 0) + getattr(as_, "corners", 0)
        elif stat == "cards":
            return (getattr(hs, "yellow_cards", 0) + getattr(hs, "red_cards", 0)
                    + getattr(as_, "yellow_cards", 0) + getattr(as_, "red_cards", 0))
        elif stat == "sot":
            return getattr(hs, "sot", 0) + getattr(as_, "sot", 0)

    # Engine state has flat fields (total_goals, total_corners, etc.)
    prop_map = {
        "goals": "total_goals",
        "corners": "total_corners",
        "cards": "total_cards",
        "sot": "total_sot",
    }
    prop = prop_map.get(stat)
    if prop:
        val = getattr(state, prop, None)
        if val is not None:
            return int(val)
    return 0


# ---------------------------------------------------------------------------
# C. Live Fixtures List Embed
# ---------------------------------------------------------------------------

def live_fixtures_embed(fixtures: List[Dict[str, Any]]) -> discord.Embed:
    """Simple list of all currently live fixtures with scores.

    Parameters
    ----------
    fixtures : list of dict
        Each dict from LivePoller.get_active_fixtures() with keys:
        fixture_id, home, away, score, elapsed, status, league, league_id.
    """
    if not fixtures:
        em = discord.Embed(
            title="\u26bd Live Scores",
            description="No matches currently in play.",
            color=COLOR_BLUE,
        )
        em.set_footer(text="Matchwise Live")
        return em

    em = discord.Embed(
        title=f"\u26bd Live Scores ({len(fixtures)} match{'es' if len(fixtures) != 1 else ''})",
        color=COLOR_GREEN,
    )

    # Group by league
    by_league: Dict[str, List[Dict]] = {}
    for f in fixtures:
        lg = f.get("league", "Other")
        by_league.setdefault(lg, []).append(f)

    for lg, lg_fixtures in by_league.items():
        lines = []
        for f in lg_fixtures:
            status = f.get("status", "")
            elapsed = f.get("elapsed", 0)
            half_label = {"1H": "1H", "HT": "HT", "2H": "2H", "ET": "ET"}.get(status, status)
            lines.append(
                f"**{f.get('home', '?')}** {f.get('score', '?-?')} "
                f"**{f.get('away', '?')}**  {elapsed}' {half_label}"
            )
        em.add_field(
            name=f"\U0001f3df {lg}",
            value="\n".join(lines),
            inline=False,
        )

    now_str = datetime.now(timezone.utc).strftime("%H:%M")
    em.set_footer(text=f"Matchwise Live | Updated {now_str} UTC")
    return em


# ---------------------------------------------------------------------------
# D. Goal Alert Embed (special formatting)
# ---------------------------------------------------------------------------

def goal_alert_embed(
    event: Dict[str, Any],
    snapshot: Any,
    state: Any,
) -> discord.Embed:
    """Special embed for goal events with score update and market impact.

    Parameters
    ----------
    event : dict
        Goal event with minute, player, team, detail.
    snapshot : LiveSnapshot
        Updated snapshot after the goal.
    state : LiveMatchState
        Raw match state.
    """
    home = getattr(state, "home_team", "?")
    away = getattr(state, "away_team", "?")
    league = getattr(state, "league", "") or getattr(snapshot, "league", "")

    hs = getattr(state, "home_stats", None)
    as_ = getattr(state, "away_stats", None)
    if hs and as_:
        score = f"{hs.goals} - {as_.goals}"
    else:
        score = getattr(snapshot, "score", "? - ?")

    scorer = event.get("player", "")
    minute = event.get("minute", "?")
    detail = event.get("detail", "")
    detail_str = f" ({detail})" if detail and detail != "Normal Goal" else ""

    title = f"\u26bd GOAL! {home} {score} {away}"

    em = discord.Embed(title=title, color=COLOR_GREEN)
    em.add_field(
        name=f"{minute}' {scorer}{detail_str}",
        value=f"Team: **{event.get('team', '?')}**",
        inline=False,
    )

    # Updated moneyline
    ml = getattr(snapshot, "moneyline", None) or {}
    if ml:
        ml_str = (
            f"Home {_pct(ml.get('home_win'))} | "
            f"Draw {_pct(ml.get('draw'))} | "
            f"Away {_pct(ml.get('away_win'))}"
        )
        em.add_field(name="\U0001f3c6 Moneyline", value=ml_str, inline=False)

    # Updated BTTS
    btts = getattr(snapshot, "btts_prob", None)
    if btts is not None:
        em.add_field(name="\U0001f91d BTTS", value=_pct(btts), inline=True)

    # Updated goals projection
    proj = getattr(snapshot, "projected_total_goals", None)
    if proj is not None:
        em.add_field(name="\u26bd Projected Total", value=_f1(proj), inline=True)

    footer_parts = ["Matchwise Live"]
    if league:
        footer_parts.append(league)
    em.set_footer(text=" | ".join(footer_parts))
    return em


# ---------------------------------------------------------------------------
# E. Match Start / Full Time Embeds
# ---------------------------------------------------------------------------

def match_start_embed(state: Any) -> discord.Embed:
    """Embed posted when a match goes live."""
    home = getattr(state, "home_team", "?")
    away = getattr(state, "away_team", "?")
    league = getattr(state, "league", "")

    em = discord.Embed(
        title=f"\U0001f6a8 KICK OFF \u2014 {home} vs {away}",
        color=COLOR_GREEN,
    )

    # Show pre-match priors if available
    priors = []
    for label, attr in [
        ("Goals", "prior_total_goals"),
        ("Corners", "prior_total_corners"),
        ("Cards", "prior_total_cards"),
        ("SoT", "prior_total_sot"),
    ]:
        val = getattr(state, attr, None)
        if val is not None:
            priors.append(f"{label}: {val:.1f}")

    btts = getattr(state, "prior_btts_prob", None)
    if btts is not None:
        priors.append(f"BTTS: {btts:.0%}")

    if priors:
        em.add_field(
            name="\U0001f4ca Pre-Match Projections",
            value="\n".join(priors),
            inline=False,
        )

    footer_parts = ["Matchwise Live"]
    if league:
        footer_parts.append(league)
    em.set_footer(text=" | ".join(footer_parts))
    return em


def full_time_embed(snapshot: Any, state: Any) -> discord.Embed:
    """Embed posted when a match finishes. Compares projections to actuals."""
    home = getattr(state, "home_team", "?")
    away = getattr(state, "away_team", "?")
    league = getattr(state, "league", "") or getattr(snapshot, "league", "")

    hs = getattr(state, "home_stats", None)
    as_ = getattr(state, "away_stats", None)
    if hs and as_:
        score = f"{hs.goals} - {as_.goals}"
    else:
        score = getattr(snapshot, "score", "? - ?")

    em = discord.Embed(
        title=f"\U0001f3c1 FULL TIME \u2014 {home} {score} {away}",
        color=COLOR_YELLOW,
    )

    # Projection vs Actual comparison
    comparisons = []
    stat_pairs = [
        ("Goals", _observed_count(state, "goals"), getattr(state, "prior_total_goals", None)),
        ("Corners", _observed_count(state, "corners"), getattr(state, "prior_total_corners", None)),
        ("Cards", _observed_count(state, "cards"), getattr(state, "prior_total_cards", None)),
        ("SoT", _observed_count(state, "sot"), getattr(state, "prior_total_sot", None)),
    ]
    for label, actual, projected in stat_pairs:
        if projected is not None:
            diff = actual - projected
            arrow = "\u2191" if diff > 0 else "\u2193" if diff < 0 else "\u2192"
            comparisons.append(
                f"{label}: {actual} actual vs {projected:.1f} projected ({arrow} {diff:+.1f})"
            )

    if comparisons:
        em.add_field(
            name="\U0001f4ca Projection vs Actual",
            value="\n".join(comparisons),
            inline=False,
        )

    footer_parts = ["Matchwise Live"]
    if league:
        footer_parts.append(league)
    em.set_footer(text=" | ".join(footer_parts))
    return em


# ---------------------------------------------------------------------------
# F. Live Probability Embed (for /liveprob command)
# ---------------------------------------------------------------------------

def live_prob_embed(
    snapshot: Any,
    state: Any,
    market: str,
) -> discord.Embed:
    """Embed showing probabilities for a specific market.

    Parameters
    ----------
    market : str
        Market type: "goals", "corners", "cards", "sot", "btts", "moneyline".
    """
    home = getattr(snapshot, "home_team", None) or getattr(state, "home_team", "?")
    away = getattr(snapshot, "away_team", None) or getattr(state, "away_team", "?")
    score = getattr(snapshot, "score", None) or "? - ?"
    elapsed = getattr(snapshot, "elapsed_min", 0)
    league = getattr(snapshot, "league", "")

    em = discord.Embed(
        title=f"\U0001f4ca {market.upper()} \u2014 {home} {score} {away} ({elapsed}')",
        color=COLOR_BLUE,
    )

    market_lower = market.lower()

    if market_lower == "goals":
        prob_dict = getattr(snapshot, "prob_over_goals", {}) or {}
        proj = getattr(snapshot, "projected_total_goals", None)
        observed = _observed_count(state, "goals")
        lines = [f"Observed: {observed} | Projected total: {_f1(proj)}"]
        for line_val, p in sorted(prob_dict.items(), key=lambda x: float(x[0])):
            lines.append(f"P(Over {line_val}): {_pct(p)}")
        em.add_field(name="Goal Lines", value="\n".join(lines), inline=False)

    elif market_lower == "corners":
        prob_dict = getattr(snapshot, "prob_over_corners", {}) or {}
        proj = getattr(snapshot, "projected_total_corners", None)
        observed = _observed_count(state, "corners")
        lines = [f"Observed: {observed} | Projected total: {_f1(proj)}"]
        for line_val, p in sorted(prob_dict.items(), key=lambda x: float(x[0])):
            lines.append(f"P(Over {line_val}): {_pct(p)}")
        em.add_field(name="Corner Lines", value="\n".join(lines), inline=False)

    elif market_lower == "cards":
        prob_dict = getattr(snapshot, "prob_over_cards", {}) or {}
        proj = getattr(snapshot, "projected_total_cards", None)
        observed = _observed_count(state, "cards")
        lines = [f"Observed: {observed} | Projected total: {_f1(proj)}"]
        for line_val, p in sorted(prob_dict.items(), key=lambda x: float(x[0])):
            lines.append(f"P(Over {line_val}): {_pct(p)}")
        em.add_field(name="Card Lines", value="\n".join(lines), inline=False)

    elif market_lower == "sot":
        prob_dict = getattr(snapshot, "prob_over_sot", {}) or {}
        proj = getattr(snapshot, "projected_total_sot", None)
        observed = _observed_count(state, "sot")
        lines = [f"Observed: {observed} | Projected total: {_f1(proj)}"]
        for line_val, p in sorted(prob_dict.items(), key=lambda x: float(x[0])):
            lines.append(f"P(Over {line_val}): {_pct(p)}")
        em.add_field(name="SoT Lines", value="\n".join(lines), inline=False)

    elif market_lower == "btts":
        btts = getattr(snapshot, "btts_prob", None)
        pre_btts = getattr(snapshot, "pre_match_btts_prob", None)
        lines = [f"P(BTTS Yes): {_pct(btts)}"]
        if pre_btts is not None:
            lines.append(f"Pre-match: {_pct(pre_btts)}")
        em.add_field(name="BTTS", value="\n".join(lines), inline=False)

    elif market_lower == "moneyline":
        ml = getattr(snapshot, "moneyline", {}) or {}
        pre_ml = getattr(snapshot, "pre_match_moneyline", None) or {}
        lines = [
            f"Home Win: {_pct(ml.get('home_win'))}",
            f"Draw:     {_pct(ml.get('draw'))}",
            f"Away Win: {_pct(ml.get('away_win'))}",
        ]
        if pre_ml:
            lines.append("")
            lines.append("Pre-match:")
            lines.append(f"  Home: {_pct(pre_ml.get('home_win'))} | "
                         f"Draw: {_pct(pre_ml.get('draw'))} | "
                         f"Away: {_pct(pre_ml.get('away_win'))}")
        em.add_field(name="Moneyline", value="\n".join(lines), inline=False)

    else:
        em.description = f"Unknown market: {market}. Use: goals, corners, cards, sot, btts, moneyline"

    footer_parts = ["Matchwise Live"]
    if league:
        footer_parts.append(league)
    em.set_footer(text=" | ".join(footer_parts))
    return em


# ---------------------------------------------------------------------------
# G. Live Stats Embed (raw stats for /livestats)
# ---------------------------------------------------------------------------

def live_bets_embed(snapshot: Any, state: Any) -> discord.Embed:
    """Recommended live bets for a fixture based on current projections."""
    home = getattr(snapshot, "home_team", "?")
    away = getattr(snapshot, "away_team", "?")
    league = getattr(snapshot, "league", "")
    elapsed = getattr(snapshot, "elapsed_min", 0)
    score = getattr(snapshot, "score", "? - ?")

    em = discord.Embed(
        title=f"\U0001f4b0 Live Bets \u2014 {home} {score} {away} ({elapsed:.0f}')",
        color=COLOR_GREEN,
    )

    picks = []

    def _best_line_pick(prob_dict, stat_emoji, stat_name, projected, observed, remaining=None):
        """Find the single best actionable line for a stat.

        Instead of showing every line above 60%, find the TIGHTEST contested
        line — the one closest to the projection where the model has a
        directional opinion. This is where real betting value lives.
        """
        if not prob_dict or projected is None:
            return None

        best = None
        best_score = -1

        for line_str, p_over in sorted(prob_dict.items(), key=lambda x: float(x[0])):
            line = float(line_str)
            # Skip lines already cleared (e.g., Over 1.5 when 3 goals scored)
            if line < observed:
                continue

            # The actionable probability is whichever side is stronger
            if p_over >= 0.55:
                side = "Over"
                prob = p_over
            elif (1.0 - p_over) >= 0.55:
                side = "Under"
                prob = 1.0 - p_over
            else:
                continue  # too close to call

            # Score: reward lines where probability is strong but NOT trivially obvious.
            # Best picks are in the 55-80% range on a contested line.
            # 98% on Under 4.5 is useless. 62% on Under 2.5 is actionable.
            closeness_to_projection = 1.0 / (1.0 + abs(projected - line))
            actionability = prob if prob < 0.85 else (0.85 - (prob - 0.85) * 2)  # penalize trivial
            score = actionability * closeness_to_projection

            if score > best_score:
                best_score = score
                rem_str = f", {_f1(remaining)} remaining" if remaining is not None else ""
                best = {
                    "market": f"{stat_emoji} {side} {line_str} {stat_name}",
                    "prob": prob,
                    "confidence": "Strong Edge" if prob >= 0.65 else "Leaning",
                    "color": "high" if prob >= 0.65 else "medium",
                    "reason": f"Projected {_f1(projected)} total ({observed} so far{rem_str})",
                }

        return best

    # --- Goals: best single line ---
    prob_goals = getattr(snapshot, "prob_over_goals", {}) or {}
    proj_goals = getattr(snapshot, "projected_total_goals", None)
    observed_goals = _observed_count(state, "goals")
    rem_goals = getattr(snapshot, "remaining_goals", None)
    goal_pick = _best_line_pick(prob_goals, "\u26bd", "Goals", proj_goals, observed_goals, rem_goals)
    if goal_pick:
        picks.append(goal_pick)

    # --- Corners: best single line ---
    prob_corners = getattr(snapshot, "prob_over_corners", {}) or {}
    proj_corners = getattr(snapshot, "projected_total_corners", None)
    observed_corners = _observed_count(state, "corners")
    rem_corners = getattr(snapshot, "remaining_corners", None)
    corner_pick = _best_line_pick(prob_corners, "\U0001f4d0", "Corners", proj_corners, observed_corners, rem_corners)
    if corner_pick:
        picks.append(corner_pick)

    # --- Cards: best single line ---
    prob_cards = getattr(snapshot, "prob_over_cards", {}) or {}
    proj_cards = getattr(snapshot, "projected_total_cards", None)
    observed_cards = _observed_count(state, "cards")
    rem_cards = getattr(snapshot, "remaining_cards", None)
    card_pick = _best_line_pick(prob_cards, "\U0001f7e8", "Cards", proj_cards, observed_cards, rem_cards)
    if card_pick:
        picks.append(card_pick)

    # --- BTTS pick ---
    btts_prob = getattr(snapshot, "btts_prob", None)
    pre_btts = getattr(snapshot, "pre_match_btts_prob", None)
    if btts_prob is not None:
        # Check if both teams scored — handles both poller state (home_stats.goals)
        # and engine state (goals_home) formats
        hs = getattr(state, "home_stats", None)
        as_ = getattr(state, "away_stats", None)
        if hs and as_:
            goals_h = getattr(hs, "goals", 0)
            goals_a = getattr(as_, "goals", 0)
        else:
            goals_h = getattr(state, "goals_home", 0)
            goals_a = getattr(state, "goals_away", 0)
        if goals_h >= 1 and goals_a >= 1:
            pass  # BTTS already cashed — do NOT recommend
        elif btts_prob >= 0.60:
            picks.append({
                "market": "\U0001f91d BTTS Yes",
                "prob": btts_prob,
                "confidence": "Strong Edge" if btts_prob >= 0.75 else "Leaning",
                "color": "high" if btts_prob >= 0.75 else "medium",
                "reason": f"P(BTTS Yes) = {_pct(btts_prob)} (was {_pct(pre_btts)} pre-match)",
            })
        elif (1.0 - btts_prob) >= 0.65:
            picks.append({
                "market": "\U0001f91d BTTS No",
                "prob": 1.0 - btts_prob,
                "confidence": "Strong Edge" if (1.0 - btts_prob) >= 0.80 else "Leaning",
                "color": "high" if (1.0 - btts_prob) >= 0.80 else "medium",
                "reason": f"P(BTTS No) = {_pct(1.0 - btts_prob)} (was {_pct(1.0 - pre_btts if pre_btts else None)} pre-match)",
            })

    # --- Moneyline pick ---
    ml = getattr(snapshot, "moneyline", None)
    pre_ml = getattr(snapshot, "pre_match_moneyline", None)
    if ml:
        best_side = max(ml.items(), key=lambda x: x[1])
        side_label = {"home_win": f"{home} Win", "draw": "Draw", "away_win": f"{away} Win"}.get(best_side[0], best_side[0])
        pre_val = pre_ml.get(best_side[0]) if pre_ml else None
        if best_side[1] >= 0.55:
            picks.append({
                "market": f"\U0001f3c6 {side_label}",
                "prob": best_side[1],
                "confidence": "Strong Edge" if best_side[1] >= 0.70 else "Leaning",
                "color": "high" if best_side[1] >= 0.70 else "medium",
                "reason": f"Model: {_pct(best_side[1])} (was {_pct(pre_val)} pre-match)",
            })

    # --- Next Goal / No More Goals ---
    ng = getattr(snapshot, "next_goal_probs", None)
    if ng and elapsed >= 15:
        p_home_ng = ng.get("next_home", 0)
        p_away_ng = ng.get("next_away", 0)
        p_none_ng = ng.get("no_more", 0)
        p_goal = 1.0 - p_none_ng
        exp_min = ng.get("expected_min_to_next", 99)

        if p_goal > 0:
            home_share = p_home_ng / p_goal
            away_share = p_away_ng / p_goal
        else:
            home_share = away_share = 0.5

        if home_share >= 0.62 and p_goal >= 0.45:
            picks.append({
                "market": f"\u26bd Next Goal: {home}",
                "prob": p_home_ng,
                "confidence": "Strong Edge" if home_share >= 0.72 else "Leaning",
                "color": "high" if home_share >= 0.72 else "medium",
                "reason": f"{home_share:.0%} of next-goal probability. ~{exp_min:.0f} min expected",
            })
        elif away_share >= 0.62 and p_goal >= 0.45:
            picks.append({
                "market": f"\u26bd Next Goal: {away}",
                "prob": p_away_ng,
                "confidence": "Strong Edge" if away_share >= 0.72 else "Leaning",
                "color": "high" if away_share >= 0.72 else "medium",
                "reason": f"{away_share:.0%} of next-goal probability. ~{exp_min:.0f} min expected",
            })
        elif p_none_ng >= 0.55 and elapsed >= 65:
            picks.append({
                "market": "\u26bd No More Goals",
                "prob": p_none_ng,
                "confidence": "Strong Edge" if p_none_ng >= 0.70 else "Leaning",
                "color": "high" if p_none_ng >= 0.70 else "medium",
                "reason": f"P(no more goals) = {_pct(p_none_ng)} at {elapsed:.0f}'",
            })

    # --- Sort by probability descending, take top 6 ---
    picks.sort(key=lambda p: p["prob"], reverse=True)
    picks = picks[:6]

    if not picks:
        em.description = "No strong live bets identified for this fixture right now. Probabilities are too close to call."
        em.color = COLOR_BLUE
    else:
        for i, pick in enumerate(picks, 1):
            badge = "\U0001f7e2" if pick["color"] == "high" else "\U0001f7e1"
            em.add_field(
                name=f"{badge} {pick['market']}",
                value=(
                    f"**{pick['confidence']}** \u2014 {_pct(pick['prob'])}\n"
                    f"{pick['reason']}"
                ),
                inline=False,
            )

    # xG context if available
    xg_h = getattr(snapshot, "live_xg_home", None)
    xg_a = getattr(snapshot, "live_xg_away", None)
    if xg_h is not None and xg_a is not None:
        em.add_field(
            name="\U0001f4ca xG Context",
            value=f"{home} {_f2(xg_h)} \u2014 {away} {_f2(xg_a)}",
            inline=False,
        )

    # Momentum
    mom = getattr(snapshot, "momentum", None)
    if mom:
        bar = _momentum_bar(mom.get("home", 0.5))
        dominant = home if mom.get("home", 0.5) > 0.55 else (away if mom.get("away", 0.5) > 0.55 else "Even")
        em.add_field(
            name="Momentum",
            value=f"{bar} {dominant}",
            inline=False,
        )

    footer_parts = ["Matchwise Live"]
    if league:
        footer_parts.append(league)
    footer_parts.append(f"Updated {getattr(snapshot, 'timestamp', '')[:19]}")
    em.set_footer(text=" | ".join(footer_parts))
    return em


def live_bets_all_embed(fixtures_with_picks: list) -> List[discord.Embed]:
    """Summary of best live bets across ALL live fixtures."""
    embeds = []

    header = discord.Embed(
        title="\U0001f4b0 Live Bets \u2014 All Matches",
        description="Top live betting opportunities right now",
        color=COLOR_GREEN,
    )
    embeds.append(header)

    all_picks = []
    for fix_data in fixtures_with_picks:
        snapshot = fix_data["snapshot"]
        picks = fix_data["picks"]
        fixture_label = f"{getattr(snapshot, 'home_team', '?')} {getattr(snapshot, 'score', '?-?')} {getattr(snapshot, 'away_team', '?')} ({getattr(snapshot, 'elapsed_min', 0):.0f}')"
        league = getattr(snapshot, "league", "")
        for p in picks:
            all_picks.append({**p, "fixture": fixture_label, "league": league})

    all_picks.sort(key=lambda p: p["prob"], reverse=True)
    all_picks = all_picks[:10]

    if not all_picks:
        header.description = "No strong live bets across any currently live fixture."
        return embeds

    for pick in all_picks:
        badge = "\U0001f7e2" if pick["color"] == "high" else "\U0001f7e1"
        header.add_field(
            name=f"{badge} {pick['market']}",
            value=(
                f"**{pick['fixture']}** [{pick.get('league', '')}]\n"
                f"**{pick['confidence']}** \u2014 {_pct(pick['prob'])}\n"
                f"{pick['reason']}"
            ),
            inline=False,
        )

    header.set_footer(text="Matchwise Live | Sorted by model probability")
    return embeds


def live_stats_embed(snapshot: Any, state: Any) -> discord.Embed:
    """Raw live stats display for /livestats command."""
    home = getattr(state, "home_team", "?")
    away = getattr(state, "away_team", "?")
    league = getattr(state, "league", "")
    elapsed = getattr(state, "elapsed_min", 0)

    hs = getattr(state, "home_stats", None)
    as_ = getattr(state, "away_stats", None)

    if hs and as_:
        score = f"{hs.goals} - {as_.goals}"
    else:
        score = getattr(snapshot, "score", "? - ?")

    em = discord.Embed(
        title=f"\U0001f4cb Stats \u2014 {home} {score} {away} ({elapsed}')",
        color=COLOR_BLUE,
    )

    if hs and as_:
        stats_text = (
            f"```\n"
            f"{'Stat':<16} {'Home':>6} {'Away':>6}\n"
            f"{'-'*30}\n"
            f"{'Goals':<16} {hs.goals:>6} {as_.goals:>6}\n"
            f"{'Shots':<16} {hs.shots:>6} {as_.shots:>6}\n"
            f"{'SoT':<16} {hs.sot:>6} {as_.sot:>6}\n"
            f"{'Corners':<16} {hs.corners:>6} {as_.corners:>6}\n"
            f"{'Yellow Cards':<16} {hs.yellow_cards:>6} {as_.yellow_cards:>6}\n"
            f"{'Red Cards':<16} {hs.red_cards:>6} {as_.red_cards:>6}\n"
            f"{'Fouls':<16} {hs.fouls:>6} {as_.fouls:>6}\n"
            f"{'Possession %':<16} {hs.possession:>5.0f}% {as_.possession:>5.0f}%\n"
            f"```"
        )
        em.description = stats_text
    else:
        em.description = "Detailed stats not available for this fixture."

    footer_parts = ["Matchwise Live"]
    if league:
        footer_parts.append(league)
    em.set_footer(text=" | ".join(footer_parts))
    return em
