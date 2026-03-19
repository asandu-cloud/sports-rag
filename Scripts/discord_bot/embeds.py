"""Convert RAG text output into bet-card-style Discord embeds."""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

import discord

from config import COLOR_GREEN, COLOR_YELLOW, COLOR_RED, COLOR_BLUE, COLOR_PURPLE

# ── Confidence → color / rating dots ─────────────────────────────────────────

_CONF_MAP = {
    "high": (COLOR_GREEN, "●●●●●"),
    "medium": (COLOR_YELLOW, "●●●○○"),
    "low": (COLOR_RED, "●○○○○"),
}


def _conf(text: str) -> Tuple[int, str, str]:
    """Return (color, dots, label) from confidence keyword in text."""
    lower = text.lower()
    for label, (color, dots) in _CONF_MAP.items():
        if f"{label} confidence" in lower:
            return color, dots, label.title()
    return COLOR_BLUE, "●●○○○", "—"


# ── Text helpers ─────────────────────────────────────────────────────────────

def _truncate(text: str, limit: int = 1024) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _first_match(pattern: str, text: str, group: int = 1) -> Optional[str]:
    m = re.search(pattern, text)
    return m.group(group) if m else None


# ── Block splitters ──────────────────────────────────────────────────────────

def _split_fixture_blocks(text: str) -> List[str]:
    """Split RAG output into per-fixture text blocks.

    Works for both '- Home vs Away:' and '=== Home vs Away ===' formats.
    Handles single-fixture outputs too.
    """
    # Try === separator first (moneyline, correct score, team totals)
    eq_blocks = re.split(r"\n\s*===\s+", text)
    if len(eq_blocks) > 1:
        return [b.strip() for b in eq_blocks[1:] if b.strip()]

    # Try '- Home vs Away:' separator (totals, btts, spreads)
    # Match lines starting with '- ' followed by team names with vs or @
    dash_blocks = re.split(r"\n-\s+(?=[A-Z][\w\s.'()-]+(?:vs|@))", text)
    if len(dash_blocks) > 1:
        return [b.strip() for b in dash_blocks[1:] if b.strip()]

    # Single fixture: try to find the fixture content after the header
    m = re.search(r"(?:advice|Predictions?) by fixture:?\s*\n", text)
    if m:
        body = text[m.end():].strip()
        # Strip method footer
        body = re.sub(r"\nMethod:.*$", "", body, flags=re.DOTALL).strip()
        if body:
            return [body]

    return [text.strip()]


def _parse_fixture_name(block: str) -> str:
    """Extract 'Home vs Away' from a fixture block."""
    # Block starts with "Arsenal vs Chelsea ===" (from === split)
    m = re.match(r"(.+?)\s*===", block)
    if m:
        return m.group(1).strip()
    # Home vs Away: or Home @ Away:
    m = re.match(r"(.+?(?:vs|@).+?):", block)
    if m:
        return m.group(1).strip()
    # First line fallback
    return block.split("\n")[0].strip().rstrip(":").rstrip("=")


# ── Field extractors ─────────────────────────────────────────────────────────

def _extract_pick(block: str) -> Optional[str]:
    """Extract the recommended pick (e.g. 'Under 3.5', 'BTTS Yes', 'Arsenal')."""
    # >>> Recommended: Arsenal @ 2.10  (moneyline uses >>> prefix)
    m = re.search(r"(?:>>>\s*)?Recommended:\s*(.+?)(?:\s*@\s*|\s*\()", block)
    if m:
        return m.group(1).strip()
    return None


def _extract_odds(block: str) -> Optional[str]:
    """Extract odds + bookmaker, e.g. '1.45 (Bet365)'."""
    # Match "@ 1.45 (Bet365, totals)" or "@ 2.10 (Bet365)"
    # Avoid matching "@ 2.10 (high confidence)" — bookmaker names don't contain "confidence"
    m = re.search(r"@\s*([\d.]+)\s*\(([^)]+)\)", block)
    if m:
        source = m.group(2).split(",")[0].strip()
        if "confidence" in source.lower():
            return m.group(1)  # odds only, no bookmaker
        return f"{m.group(1)} ({source})"
    return None


def _extract_model_line(block: str) -> Dict[str, Optional[str]]:
    """Extract model probability, implied prob, value edge from Model: line."""
    out: Dict[str, Optional[str]] = {"model_p": None, "implied_p": None, "edge": None}
    m = re.search(r"Model:\s*P\(.+?\)\s*=\s*([\d.]+%)", block)
    if m:
        out["model_p"] = m.group(1)
    m = re.search(r"Fair implied:\s*([\d.]+%)", block)
    if m:
        out["implied_p"] = m.group(1)
    m = re.search(r"Value edge:\s*([+\-][\d.]+%)", block)
    if m:
        out["edge"] = m.group(1)
    return out


def _extract_ev(block: str) -> Optional[str]:
    m = re.search(r"Expected value:\s*([+\-][\d.]+)", block)
    return m.group(1) if m else None


def _extract_projection(block: str) -> Optional[str]:
    """Extract projection line (varies by workflow)."""
    # projected total X.XX
    m = re.search(r"projected total\s+([\d.]+)", block)
    if m:
        return m.group(1)
    # projected goal difference +X.XX
    m = re.search(r"projected goal difference\s*([+\-]?[\d.]+)", block)
    if m:
        return m.group(1)
    # P(BTTS Yes) = XX.X%
    m = re.search(r"P\(BTTS Yes\)\s*=\s*([\d.]+%)", block)
    if m:
        return m.group(1)
    return None


def _extract_confidence(block: str) -> str:
    m = re.search(r"\b(high|medium|low)\s+confidence\b", block, re.IGNORECASE)
    return m.group(1).title() if m else "—"


def _extract_moneyline_probs(block: str) -> Dict[str, str]:
    """Extract 3-way moneyline probabilities."""
    out: Dict[str, str] = {}
    m = re.search(r"P\(Home\)\s*=\s*([\d.]+%)", block)
    if m:
        out["home"] = m.group(1)
    m = re.search(r"P\(Draw\)\s*=\s*([\d.]+%)", block)
    if m:
        out["draw"] = m.group(1)
    m = re.search(r"P\(Away\)\s*=\s*([\d.]+%)", block)
    if m:
        out["away"] = m.group(1)
    return out


def _extract_correct_scores(block: str) -> List[str]:
    """Extract top scoreline lines."""
    lines = []
    for m in re.finditer(r"\d+\.\s+(.+?—\s*[\d.]+%.*)", block):
        lines.append(m.group(1).strip())
        if len(lines) >= 5:
            break
    return lines


def _extract_interval_rows(block: str) -> List[Tuple[str, str]]:
    """Extract goal interval rows: (band_label, probability)."""
    rows = []
    for m in re.finditer(r"(\d+-\d+\s+goals|\d+\+\s+goals)\s+([\d.]+%)", block):
        rows.append((m.group(1).strip(), m.group(2)))
    return rows


# ── Bet card builders ────────────────────────────────────────────────────────

def _bet_card_embed(
    fixture: str,
    pick: Optional[str],
    odds: Optional[str],
    model_p: Optional[str],
    implied_p: Optional[str],
    edge: Optional[str],
    confidence: str,
    projection: Optional[str],
    league: str = "",
    market_emoji: str = "⚽",
    extra_fields: Optional[List[Tuple[str, str, bool]]] = None,
) -> discord.Embed:
    """Build a single bet-card embed for one fixture."""
    color, dots, conf_label = _conf(f"{confidence} confidence" if confidence != "—" else "")

    em = discord.Embed(color=color)
    em.set_author(name=f"{market_emoji}  {fixture}", icon_url=None)

    if pick:
        em.add_field(name="📋 Pick", value=f"**{pick}**", inline=True)
    if odds:
        em.add_field(name="💰 Odds", value=odds, inline=True)
    if confidence and confidence != "—":
        em.add_field(name="⭐ Rating", value=f"{dots} {conf_label}", inline=True)

    if model_p:
        em.add_field(name="📊 Model", value=model_p, inline=True)
    if implied_p:
        em.add_field(name="📖 Books", value=implied_p, inline=True)
    if edge:
        sign_emoji = "📈" if edge.startswith("+") else "📉"
        em.add_field(name=f"{sign_emoji} Edge", value=edge, inline=True)

    if extra_fields:
        for name, value, inline in extra_fields:
            em.add_field(name=name, value=value, inline=inline)

    footer_parts = []
    if league:
        footer_parts.append(league)
    if projection:
        footer_parts.append(f"Projection: {projection}")
    em.set_footer(text=" | ".join(footer_parts) if footer_parts else "Betting RAG")

    return em


# ── Main public functions ────────────────────────────────────────────────────

def player_prop_embeds(
    title: str,
    text: str,
    league: str = "",
) -> List[discord.Embed]:
    """Convert player prop RAG output into styled Discord embeds.

    Parses the === fixture separator format with player recommendation blocks.
    One embed per fixture, each containing player pick cards.
    """
    _STAT_EMOJI = {
        "goalscorer": "\u26bd", "goals": "\u26bd",
        "shots on target": "\U0001f3af", "sot": "\U0001f3af",
        "cards": "\U0001f7e8", "booked": "\U0001f7e8",
        "assists": "\U0001f3c3", "assist": "\U0001f3c3",
    }

    # Detect stat type from title for emoji
    title_lower = title.lower()
    stat_emoji = "\U0001f464"
    for key, em in _STAT_EMOJI.items():
        if key in title_lower:
            stat_emoji = em
            break

    blocks = _split_fixture_blocks(text)
    embeds: List[discord.Embed] = []

    # Header embed
    header = discord.Embed(
        title=f"{stat_emoji}  {title}",
        description=f"*{league}*" if league else None,
        color=COLOR_BLUE,
    )
    embeds.append(header)

    for block in blocks:
        fixture = _parse_fixture_name(block)

        # Parse individual player recommendations from this fixture block
        players = _parse_player_prop_block(block)

        if not players:
            continue

        # Build one embed per fixture with all players as fields
        # Use highest confidence player's color for the embed
        best_conf = "low"
        for p in players:
            if p.get("confidence") == "high":
                best_conf = "high"
                break
            elif p.get("confidence") == "medium" and best_conf != "high":
                best_conf = "medium"

        color, dots, conf_label = _conf(f"{best_conf} confidence")
        em = discord.Embed(color=color)
        em.set_author(name=f"{stat_emoji}  {fixture}")

        for p in players[:5]:  # max 5 players per fixture
            # Build compact player card
            name = p.get("name", "Unknown")
            pos = p.get("position", "")
            venue = p.get("venue", "")
            conf = p.get("confidence", "low")

            conf_icon = "\U0001f7e2" if conf == "HIGH" else ("\U0001f7e1" if conf == "MEDIUM" else "\u26aa")
            pick_line = p.get("pick", "")
            role_line = p.get("role_line", "")
            proj_line = p.get("projection_line", "")
            recent_line = p.get("recent_line", "")
            opp_line = p.get("opponent_line", "")

            value_parts = []
            if pick_line:
                value_parts.append(f"**{pick_line}**")
            if role_line:
                value_parts.append(role_line)
            if proj_line:
                value_parts.append(proj_line)
            if recent_line:
                value_parts.append(recent_line)
            if opp_line:
                value_parts.append(opp_line)

            header_text = f"{conf_icon} {name}"
            if pos:
                header_text += f" [{pos}]"
            if venue:
                header_text += f" ({venue})"

            em.add_field(
                name=header_text,
                value=_truncate("\n".join(value_parts) if value_parts else "No data", 1024),
                inline=False,
            )

        em.set_footer(text=f"{league} | Model-only projections" if league else "Model-only projections")
        embeds.append(em)

    if len(embeds) > 10:
        embeds = embeds[:10]

    return embeds


def _parse_player_prop_block(block: str) -> List[Dict[str, str]]:
    """Parse player recommendation entries from a fixture block.

    Each player entry starts with an emoji + player name line.
    """
    players: List[Dict[str, str]] = []

    # Split on player header lines (emoji + name pattern)
    # Pattern: emoji(s) followed by player name, team in parens, position in brackets
    player_sections = re.split(
        r"\n(?=[\u2600-\U0001ffff]+ [A-Z])",
        block,
    )

    for section in player_sections:
        section = section.strip()
        if not section:
            continue

        p: Dict[str, str] = {}

        # Extract player name and metadata from header
        # Format: "emoji Name (team) [POS] (VENUE)"
        header_m = re.match(
            r"[\u2600-\U0001ffff]+\s+(.+?)\s+\(([^)]+)\)\s+\[([A-Z]+)\]\s+\((\w+)\)",
            section,
        )
        if header_m:
            p["name"] = header_m.group(1).strip()
            p["team"] = header_m.group(2).strip()
            p["position"] = header_m.group(3).strip()
            p["venue"] = header_m.group(4).strip()
        else:
            continue  # skip unparseable

        # Extract recommendation line
        pick_m = re.search(
            r">>>\s*Recommended:\s*(.+?)(?:\s*\((\d+%)\s+confidence\))?$",
            section, re.MULTILINE,
        )
        if pick_m:
            p["pick"] = pick_m.group(1).strip()
            if pick_m.group(2):
                p["model_prob"] = pick_m.group(2)

        # Extract role line
        role_m = re.search(r"Role:\s*(.+?)$", section, re.MULTILINE)
        if role_m:
            p["role_line"] = role_m.group(1).strip()

        # Extract projection line
        proj_m = re.search(r"Projection:\s*(.+?)$", section, re.MULTILINE)
        if proj_m:
            p["projection_line"] = proj_m.group(1).strip()

        # Extract recent form
        recent_m = re.search(r"Recent:\s*(.+?)$", section, re.MULTILINE)
        if recent_m:
            p["recent_line"] = recent_m.group(1).strip()

        # Extract opponent line
        opp_m = re.search(r"Opponent:\s*(.+?)$", section, re.MULTILINE)
        if opp_m:
            p["opponent_line"] = opp_m.group(1).strip()

        # Extract confidence
        conf_m = re.search(r"Confidence:\s*(\w+)", section)
        if conf_m:
            p["confidence"] = conf_m.group(1).strip()

        players.append(p)

    return players


def team_totals_embeds(
    title: str,
    text: str,
    league: str = "",
) -> List[discord.Embed]:
    """Convert team totals RAG output into styled Discord embeds.

    Parses the === fixture separator format, then extracts CORNERS and CARDS
    sections with per-team projections and recommendations.
    One embed per fixture with structured fields for each stat/team.
    """
    blocks = _split_fixture_blocks(text)
    embeds: List[discord.Embed] = []

    # Header embed
    header = discord.Embed(
        title=f"👥  {title}",
        description=f"*{league}*" if league else None,
        color=COLOR_BLUE,
    )
    embeds.append(header)

    for block in blocks:
        fixture = _parse_fixture_name(block)
        teams = _parse_team_totals_block(block)

        if not teams:
            continue

        # Determine best confidence across all recs for embed color
        best_conf = "low"
        for t in teams:
            conf = t.get("confidence", "low").lower()
            if conf == "high":
                best_conf = "high"
            elif conf == "medium" and best_conf != "high":
                best_conf = "medium"

        color, dots, conf_label = _conf(f"{best_conf} confidence")
        em = discord.Embed(color=color)
        em.set_author(name=f"👥  {fixture}")

        # Group teams by stat section
        corners = [t for t in teams if t.get("stat") == "corners"]
        cards = [t for t in teams if t.get("stat") == "cards"]

        # CORNERS section
        if corners:
            corner_lines = []
            for t in corners:
                team_name = t.get("team", "?")
                proj = t.get("projection", "?")
                rec = t.get("recommendation", "")
                prob = t.get("model_prob", "")
                conf = t.get("confidence", "")

                conf_icon = "🟢" if conf.lower() == "high" else (
                    "🟡" if conf.lower() == "medium" else "⚪")

                line = f"{conf_icon} **{team_name}**: {proj} corners"
                if rec:
                    line += f"\n  📋 {rec}"
                if prob:
                    line += f" ({prob})"
                corner_lines.append(line)

            # Match total
            match_total = _first_match(
                r"Match total:\s*([\d.]+)\s+projected corners", block)
            if match_total:
                corner_lines.append(f"📊 Match total: **{match_total}** projected")

            em.add_field(
                name="📐 Corners",
                value=_truncate("\n".join(corner_lines), 1024),
                inline=False,
            )

        # CARDS section
        if cards:
            card_lines = []

            # Referee info
            ref_m = re.search(
                r"Referee:\s*(.+?)(?:\s*—\s*(.+?))?$",
                block, re.MULTILINE,
            )
            if ref_m and "not yet assigned" not in (ref_m.group(0) or ""):
                ref_name = ref_m.group(1).strip()
                ref_detail = ref_m.group(2).strip() if ref_m.group(2) else ""
                ref_text = f"🧑‍⚖️ {ref_name}"
                if ref_detail:
                    ref_text += f" — {ref_detail}"
                card_lines.append(ref_text)

            for t in cards:
                team_name = t.get("team", "?")
                proj = t.get("projection", "?")
                rec = t.get("recommendation", "")
                prob = t.get("model_prob", "")
                conf = t.get("confidence", "")

                conf_icon = "🟢" if conf.lower() == "high" else (
                    "🟡" if conf.lower() == "medium" else "⚪")

                line = f"{conf_icon} **{team_name}**: {proj} cards"
                if rec:
                    line += f"\n  📋 {rec}"
                if prob:
                    line += f" ({prob})"
                card_lines.append(line)

            # Match total
            match_total = _first_match(
                r"Match total:\s*([\d.]+)\s+projected cards", block)
            if match_total:
                card_lines.append(f"📊 Match total: **{match_total}** projected")

            em.add_field(
                name="🟨 Cards",
                value=_truncate("\n".join(card_lines), 1024),
                inline=False,
            )

        em.set_footer(text=f"{league} | Per-team model projections" if league else "Per-team model projections")
        embeds.append(em)

    if len(embeds) > 10:
        embeds = embeds[:10]

    return embeds


def _parse_team_totals_block(block: str) -> List[Dict[str, str]]:
    """Parse per-team recommendations from a team totals fixture block.

    Extracts CORNERS and CARDS sections, finding team projections and
    recommended lines within each.
    """
    teams: List[Dict[str, str]] = []

    # Split block into CORNERS and CARDS sections
    corners_m = re.search(r"CORNERS:(.*?)(?=\s*CARDS:|$)", block, re.DOTALL)
    cards_m = re.search(r"CARDS:(.*?)$", block, re.DOTALL)

    for stat, section_match in [("corners", corners_m), ("cards", cards_m)]:
        if not section_match:
            continue
        section = section_match.group(1)

        # Find per-team entries: "TeamName: projected X.XX corners/match."
        # followed by optional recommendation lines
        team_pattern = re.finditer(
            r"^\s{2,6}(\S.+?):\s*projected\s+([\d.]+)\s+(?:corners|cards)/match\.",
            section, re.MULTILINE,
        )
        for tm in team_pattern:
            team_name = tm.group(1).strip()
            projection = tm.group(2)

            # Find the recommendation after this team's projection line
            # Search from this match position to the next team or section end
            rest = section[tm.end():]
            # Cut at next team projection line
            next_team = re.search(
                r"^\s{2,6}\S.+?:\s*(?:projected|insufficient)",
                rest, re.MULTILINE,
            )
            team_section = rest[:next_team.start()] if next_team else rest

            rec_text = ""
            model_prob = ""
            confidence = ""

            # Model-only recommendation
            rec_m = re.search(
                r"Recommended:\s*(\w+)\s+([\d.]+)\s+\(model-only,\s*P\(.+?\)\s*=\s*([\d.]+%),\s*(\w+)\s+confidence\)",
                team_section,
            )
            if rec_m:
                side = rec_m.group(1)
                line_val = rec_m.group(2)
                model_prob = rec_m.group(3)
                confidence = rec_m.group(4)
                rec_text = f"{side} {line_val} {stat}"
            else:
                # Bookmaker-backed recommendation
                rec_m = re.search(
                    r"Recommended:\s*(\w+)\s+([\d.]+)\s+@\s*([\d.]+)\s+\(([^)]+)\)",
                    team_section,
                )
                if rec_m:
                    side = rec_m.group(1)
                    line_val = rec_m.group(2)
                    odds = rec_m.group(3)
                    book = rec_m.group(4).split(",")[0].strip()
                    rec_text = f"{side} {line_val} {stat} @ {odds} ({book})"

                # Extract model prob from Model: line
                mp_m = re.search(r"Model:\s*P\(.+?\)\s*=\s*([\d.]+%)", team_section)
                if mp_m:
                    model_prob = mp_m.group(1)

                # Extract confidence from EV or Edge line
                conf_m = re.search(r"\b(high|medium|low)\s+confidence\b", team_section, re.I)
                if conf_m:
                    confidence = conf_m.group(1)

            teams.append({
                "stat": stat,
                "team": team_name,
                "projection": projection,
                "recommendation": rec_text,
                "model_prob": model_prob,
                "confidence": confidence,
            })

    return teams


def consolidated_fixture_embeds(
    league: str,
    fixture_data: Dict[str, Dict[str, str]],
) -> List[discord.Embed]:
    """Build compact fixture-centric embeds from pre-parsed multi-market data.

    fixture_data: {
        "Arsenal vs Chelsea": {
            "moneyline": "**Arsenal** @ 2.10 | 62% model | 🟢 High",
            "btts": "**Yes** @ 1.75 | 58% model | 🟡 Medium",
            "spreads": "**Arsenal -0.5** @ 1.85 | 61% model | 🟢 High",
            "sot": "**Over 8.5** @ 1.80 | 67% model | 🟡 Medium",
        },
        ...
    }
    """
    embeds: List[discord.Embed] = []

    # Header
    header = discord.Embed(
        title=f"📊  Match Predictions — {league}",
        color=COLOR_BLUE,
    )
    embeds.append(header)

    _SECTION_LABELS = {
        "moneyline": "🏆 Winner",
        "btts": "🤝 BTTS",
        "spreads": "📏 Handicap",
        "sot": "🎯 Shots on Target",
    }

    for fixture, markets in fixture_data.items():
        if not markets:
            continue

        # Determine best confidence for embed color
        best_conf = "low"
        for line in markets.values():
            if "🟢" in line:
                best_conf = "high"
                break
            elif "🟡" in line and best_conf != "high":
                best_conf = "medium"

        color, _, _ = _conf(f"{best_conf} confidence")
        em = discord.Embed(color=color)
        em.set_author(name=f"⚽  {fixture}")

        # Build compact market lines
        lines = []
        for key in ["moneyline", "btts", "spreads", "sot"]:
            val = markets.get(key)
            if val:
                label = _SECTION_LABELS.get(key, key.title())
                lines.append(f"{label}\n{val}")

        if lines:
            em.description = "\n\n".join(lines)

        em.set_footer(text=league)
        embeds.append(em)

    if len(embeds) > 10:
        embeds = embeds[:10]

    return embeds


def consolidated_score_embeds(
    league: str,
    fixture_data: Dict[str, Dict[str, str]],
) -> List[discord.Embed]:
    """Build compact fixture-centric embeds for correct score + intervals.

    fixture_data: {
        "Arsenal vs Chelsea": {
            "correct_score": "1. 2-1 — 14.2% | 2. 1-1 — 12.8% | ...",
            "intervals": "0-1: 22% | 2-3: 45% | 4-5: 28% | 6+: 5%",
        },
    }
    """
    embeds: List[discord.Embed] = []

    header = discord.Embed(
        title=f"🎯  Correct Score & Goal Intervals — {league}",
        color=COLOR_BLUE,
    )
    embeds.append(header)

    for fixture, data in fixture_data.items():
        if not data:
            continue

        em = discord.Embed(color=COLOR_BLUE)
        em.set_author(name=f"⚽  {fixture}")

        cs = data.get("correct_score")
        if cs:
            em.add_field(name="🎯 Most Likely Scores", value=_truncate(cs, 1024),
                         inline=False)

        iv = data.get("intervals")
        if iv:
            em.add_field(name="📊 Goal Intervals", value=_truncate(iv, 1024),
                         inline=False)

        em.set_footer(text=league)
        embeds.append(em)

    if len(embeds) > 10:
        embeds = embeds[:10]

    return embeds


def parse_fixture_picks(text: str) -> Dict[str, str]:
    """Parse RAG output into {fixture_name: compact_pick_line} dict.

    Extracts fixture name, pick, odds, model prob, confidence from each
    fixture block and formats as a single compact line.
    """
    blocks = _split_fixture_blocks(text)
    result: Dict[str, str] = {}

    for block in blocks:
        fixture = _parse_fixture_name(block)
        pick = _extract_pick(block)
        odds = _extract_odds(block)
        model = _extract_model_line(block)
        confidence = _extract_confidence(block)
        projection = _extract_projection(block)

        if not pick and not projection:
            continue

        # Build compact one-liner
        parts = []
        if pick:
            parts.append(f"**{pick}**")
        if odds:
            parts.append(f"@ {odds}")
        if model.get("model_p"):
            parts.append(f"{model['model_p']} model")
        if model.get("edge"):
            parts.append(f"edge {model['edge']}")

        conf_icon = "🟢" if confidence == "High" else (
            "🟡" if confidence == "Medium" else "⚪")
        parts.append(conf_icon)

        result[fixture] = " | ".join(parts) if parts else ""

    return result


def parse_correct_score_picks(text: str) -> Dict[str, str]:
    """Parse correct score RAG output into {fixture: top_scorelines_str}."""
    blocks = _split_fixture_blocks(text)
    result: Dict[str, str] = {}

    for block in blocks:
        fixture = _parse_fixture_name(block)
        scores = _extract_correct_scores(block)
        if scores:
            result[fixture] = "\n".join(f"`{i+1}.` {s}" for i, s in enumerate(scores[:5]))

    return result


def parse_interval_picks(text: str) -> Dict[str, str]:
    """Parse goal interval RAG output into {fixture: intervals_str}."""
    blocks = _split_fixture_blocks(text)
    result: Dict[str, str] = {}

    for block in blocks:
        fixture = _parse_fixture_name(block)
        intervals = _extract_interval_rows(block)
        if intervals:
            result[fixture] = " | ".join(f"**{band}** {prob}" for band, prob in intervals)

    return result


def rag_output_to_embeds(
    title: str,
    text: str,
    league: str = "",
    footer: Optional[str] = None,
) -> List[discord.Embed]:
    """Convert RAG text output into bet-card-style Discord embeds.

    Parses fixtures and builds one embed per fixture. Falls back to
    a plain code-block embed if parsing fails.
    """
    # Route player prop output to dedicated parser
    if "Player Prop Predictions" in text:
        return player_prop_embeds(title, text, league)

    # Route team totals output to dedicated parser
    if "Per-Team Totals Lines" in text:
        return team_totals_embeds(title, text, league)

    # Detect market type for emoji
    title_lower = title.lower()
    if "corner" in title_lower:
        emoji = "📐"
    elif "card" in title_lower:
        emoji = "🟨"
    elif "sot" in title_lower or "shot" in title_lower:
        emoji = "🎯"
    elif "btts" in title_lower:
        emoji = "🤝"
    elif "moneyline" in title_lower or "winner" in title_lower:
        emoji = "🏆"
    elif "handicap" in title_lower or "spread" in title_lower:
        emoji = "📏"
    elif "correct" in title_lower or "score" in title_lower:
        emoji = "🎯"
    elif "interval" in title_lower:
        emoji = "📊"
    elif "team" in title_lower:
        emoji = "👥"
    else:
        emoji = "⚽"

    blocks = _split_fixture_blocks(text)
    embeds: List[discord.Embed] = []

    # Header embed
    header = discord.Embed(
        title=f"{emoji}  {title}",
        description=f"*{league}*" if league else None,
        color=COLOR_BLUE,
    )
    embeds.append(header)

    for block in blocks:
        fixture = _parse_fixture_name(block)
        pick = _extract_pick(block)
        odds = _extract_odds(block)
        model = _extract_model_line(block)
        confidence = _extract_confidence(block)
        projection = _extract_projection(block)

        # Special handling for moneyline
        ml_probs = _extract_moneyline_probs(block)
        extra: List[Tuple[str, str, bool]] = []
        if ml_probs:
            prob_line = " | ".join(
                f"{'🏠' if k == 'home' else '🤝' if k == 'draw' else '✈️'} {v}"
                for k, v in ml_probs.items()
            )
            extra.append(("📊 Win Probabilities", prob_line, False))

        # Special handling for correct score
        scores = _extract_correct_scores(block)
        if scores:
            scores_text = "\n".join(f"`{i+1}.` {s}" for i, s in enumerate(scores[:5]))
            extra.append(("🎯 Top Scorelines", _truncate(scores_text, 1024), False))

        # Special handling for goal intervals
        intervals = _extract_interval_rows(block)
        if intervals:
            iv_text = "\n".join(f"`{band:<15}` {prob}" for band, prob in intervals)
            extra.append(("📊 Intervals", _truncate(iv_text, 1024), False))

        em = _bet_card_embed(
            fixture=fixture,
            pick=pick,
            odds=odds,
            model_p=model["model_p"],
            implied_p=model["implied_p"],
            edge=model["edge"],
            confidence=confidence,
            projection=projection,
            league=league,
            market_emoji=emoji,
            extra_fields=extra if extra else None,
        )
        embeds.append(em)

    # Discord limit: max 10 embeds per message
    if len(embeds) > 10:
        embeds = embeds[:10]

    return embeds


def parlay_embed(text: str, league: str = "Mixed") -> List[discord.Embed]:
    """Format parlay output as a bet-slip-style embed."""
    if not text or not text.strip():
        em = discord.Embed(
            title="🎰  Parlay Slip",
            description="No parlay data available. Try a different date or league.",
            color=COLOR_BLUE,
        )
        return [em]

    embeds: List[discord.Embed] = []

    # Parse legs — handle both old format and new format with confidence appended
    # New format: Leg 1: [EPL] Arsenal vs Chelsea | Over 2.5 (corners) @ 1.83 (Bet365) (high confidence, model 71%)
    legs: List[Dict[str, str]] = []
    for m in re.finditer(
        r"Leg\s+(\d+):\s*(?:\[([^\]]+)\]\s*)?(.+?)\s*\|\s*(.+?)\s*@\s*([\d.]+)\s*\(([^)]+?)\)"
        r"(?:\s*\((\w+)\s+confidence(?:,\s*model\s+([\d.]+%))?\))?",
        text,
    ):
        legs.append({
            "num": m.group(1),
            "league": m.group(2) or league,
            "fixture": m.group(3).strip(),
            "pick": m.group(4).strip(),
            "odds": m.group(5),
            "book": m.group(6).split(",")[0].strip(),
            "confidence": m.group(7) or None,       # "high", "medium", "low", or None
            "model_prob": m.group(8) or None,        # "71%", or None
        })

    # Parse combined odds
    combo_odds = _first_match(r"combined odds:\s*([\d.]+)", text)

    # Parse confidence breakdown
    conf_breakdown = _first_match(r"Confidence breakdown:\s*(.+?)(?:\n|$)", text)

    # Parse cross-check warnings
    cross_warnings = _first_match(r"Cross-check warnings:\s*(.+?)(?:\n|$)", text)

    if not legs:
        return _fallback_embeds("Parlay", text, league)

    # Main slip embed
    slip = discord.Embed(
        title="🎰  Parlay Slip",
        color=COLOR_PURPLE,
    )
    if league:
        slip.set_author(name=f"League: {league}")

    for leg in legs:
        # Build confidence badge for this leg
        conf = leg.get("confidence")
        model_p = leg.get("model_prob")
        conf_line = ""
        if conf:
            emoji = "🟢" if conf == "high" else ("🟡" if conf == "medium" else "⚪")
            label = {"high": "Strong Edge", "medium": "Leaning", "low": "Speculative"}.get(conf, conf.title())
            conf_line = f"\n{emoji} {label}"
            if model_p:
                conf_line += f" ({model_p})"

        slip.add_field(
            name=f"Leg {leg['num']}",
            value=(
                f"**{leg['fixture']}**\n"
                f"📋 {leg['pick']}\n"
                f"💰 {leg['odds']} ({leg['book']})"
                f"{conf_line}"
            ),
            inline=False,
        )

    # Combined odds bar
    if combo_odds:
        slip.add_field(
            name="━━━━━━━━━━━━━━━━━━━━━━",
            value=f"**Combined Odds: {combo_odds}x**",
            inline=False,
        )

    # Confidence breakdown
    if conf_breakdown:
        slip.add_field(name="⭐ Confidence", value=conf_breakdown, inline=True)

    # Cross-check warnings
    if cross_warnings:
        slip.add_field(name="⚠️ Warnings", value=cross_warnings, inline=True)

    # Parse evidence for each leg
    why = _first_match(r"Why this parlay:\s*(.+?)(?:\n-|\Z)", text, 1)
    if why:
        slip.add_field(name="💡 Why", value=_truncate(why, 1024), inline=False)

    slip.set_footer(text="Matchwise")
    embeds.append(slip)

    # Discord limit
    if len(embeds) > 10:
        embeds = embeds[:10]

    return embeds


def value_alert_embed(text: str, league: str) -> discord.Embed:
    """Format a value alert as a green bet-card embed."""
    pick = _extract_pick(text)
    odds = _extract_odds(text)
    model = _extract_model_line(text)

    em = discord.Embed(
        title="🚨  Value Alert",
        color=COLOR_GREEN,
    )
    em.set_author(name=f"League: {league}")

    if pick:
        em.add_field(name="📋 Pick", value=f"**{pick}**", inline=True)
    if odds:
        em.add_field(name="💰 Odds", value=odds, inline=True)
    if model["model_p"]:
        em.add_field(name="📊 Model", value=model["model_p"], inline=True)
    if model["edge"]:
        em.add_field(name="📈 Edge", value=model["edge"], inline=True)

    em.set_footer(text="Betting RAG | Value Alert")
    return em


# ── Fallback for unparseable output ──────────────────────────────────────────

def fixture_list_embed(
    fixtures: list,
    league: str,
    date_phrase: str,
) -> discord.Embed:
    """Build an embed listing fixtures for a given date."""
    if not fixtures:
        em = discord.Embed(
            title=f"No {league} fixtures on {date_phrase}",
            description="Try a different date or league.",
            color=COLOR_BLUE,
        )
        return em

    em = discord.Embed(
        title=f"Fixtures — {league} — {date_phrase}",
        color=COLOR_BLUE,
    )
    lines = []
    for ev in fixtures:
        home = ev.get("home_team", "?")
        away = ev.get("away_team", "?")
        kick = ev.get("commence_time", "")
        # Parse kickoff time
        time_str = ""
        if kick:
            try:
                from datetime import datetime
                dt = datetime.fromisoformat(kick.replace("Z", "+00:00"))
                time_str = f" — {dt.strftime('%H:%M')} UTC"
            except Exception:
                pass
        lines.append(f"**{home}** vs **{away}**{time_str}")

    em.description = "\n".join(lines)
    em.set_footer(text=f"{len(fixtures)} fixture(s) | Use /analyze to drill into a match")
    return em


def _fallback_embeds(
    title: str,
    text: str,
    league: str = "",
    footer: Optional[str] = None,
) -> List[discord.Embed]:
    """Plain code-block embed when structured parsing fails."""
    if not text or not text.strip():
        em = discord.Embed(
            title=title,
            description="No data available for this query. Try a different date or league.",
            color=COLOR_BLUE,
        )
        if league:
            em.set_author(name=f"League: {league}")
        em.set_footer(text=footer or "Betting RAG")
        return [em]

    color = _conf(text)[0]
    limit = 4096 - 10  # account for code fences
    chunks: List[str] = []
    current = ""
    for line in text.split("\n"):
        if len(current) + len(line) + 1 > limit:
            chunks.append(current)
            current = line
        else:
            current = f"{current}\n{line}" if current else line
    if current:
        chunks.append(current)

    embeds: List[discord.Embed] = []
    for i, chunk in enumerate(chunks):
        em = discord.Embed(
            title=title if i == 0 else f"{title} (cont.)",
            description=f"```\n{chunk}\n```",
            color=color,
        )
        if i == 0 and league:
            em.set_author(name=f"League: {league}")
        if i == len(chunks) - 1:
            em.set_footer(text=footer or "Betting RAG")
        embeds.append(em)
    return embeds[:10]
