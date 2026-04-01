"""Convert RAG text output into bet-card-style Discord embeds with team/league logos."""

from __future__ import annotations

import json
import os
import re
from typing import Dict, List, Optional, Tuple

import discord

from config import COLOR_GREEN, COLOR_YELLOW, COLOR_RED, COLOR_BLUE, COLOR_PURPLE

# ── Logo data (lazy-loaded) ─────────────────────────────────────────────────

_team_logos: Optional[Dict[str, str]] = None
_league_logos: Optional[Dict[str, str]] = None

_LOGO_DIR = os.path.join(
    os.path.dirname(__file__), os.pardir, os.pardir, "Index",
)


def _get_team_logo(team_name: str) -> Optional[str]:
    """Look up a team logo URL from team_logos.json.

    Tries exact lowercase match first, then substring containment.
    Returns None on miss so embeds degrade gracefully.
    """
    global _team_logos
    if _team_logos is None:
        path = os.path.join(_LOGO_DIR, "team_logos.json")
        try:
            with open(path) as f:
                _team_logos = json.load(f)
        except Exception:
            _team_logos = {}

    key = team_name.strip().lower()
    # Exact match
    if key in _team_logos:
        return _team_logos[key]
    # Substring match (e.g. "Arsenal" finds "arsenal")
    for stored_key, url in _team_logos.items():
        if key in stored_key or stored_key in key:
            return url
    return None


def _get_league_logo(league: str) -> Optional[str]:
    """Look up a league logo URL from league_logos.json."""
    global _league_logos
    if _league_logos is None:
        path = os.path.join(_LOGO_DIR, "league_logos.json")
        try:
            with open(path) as f:
                _league_logos = json.load(f)
        except Exception:
            _league_logos = {}

    # Try exact key first, then case-insensitive
    if league in _league_logos:
        return _league_logos[league]
    for stored_key, url in _league_logos.items():
        if stored_key.lower() == league.lower():
            return url
    return None


def _parse_teams_from_fixture(fixture: str) -> Tuple[Optional[str], Optional[str]]:
    """Extract home and away team names from a fixture string like 'Arsenal vs Chelsea'."""
    for sep in (" vs ", " @ ", " v "):
        if sep in fixture:
            parts = fixture.split(sep, 1)
            return parts[0].strip(), parts[1].strip()
    return None, None


# ── Confidence -> color / rating bar ─────────────────────────────────────────

_CONF_MAP = {
    "high": (COLOR_GREEN, "\u25cf\u25cf\u25cf\u25cf\u25cb", "Strong Edge"),
    "medium": (COLOR_YELLOW, "\u25cf\u25cf\u25cf\u25cb\u25cb", "Leaning"),
    "low": (COLOR_RED, "\u25cf\u25cb\u25cb\u25cb\u25cb", "Speculative"),
}


def _conf(text: str) -> Tuple[int, str, str]:
    """Return (color, bar, label) from confidence keyword in text."""
    lower = text.lower()
    for label, (color, bar, desc) in _CONF_MAP.items():
        if f"{label} confidence" in lower or label == lower:
            return color, bar, desc
    return COLOR_BLUE, "\u25cf\u25cf\u25cb\u25cb\u25cb", "\u2014"


def _conf_color(confidence: str) -> int:
    """Return just the embed color for a confidence level."""
    key = confidence.lower().strip()
    if key in _CONF_MAP:
        return _CONF_MAP[key][0]
    return COLOR_BLUE


def _conf_bar(confidence: str) -> str:
    """Return just the bar string for a confidence level."""
    key = confidence.lower().strip()
    if key in _CONF_MAP:
        return _CONF_MAP[key][1]
    return "\u25b0\u25b0\u25b1\u25b1\u25b1"


def _conf_label(confidence: str) -> str:
    """Return just the label for a confidence level."""
    key = confidence.lower().strip()
    if key in _CONF_MAP:
        return _CONF_MAP[key][2]
    return "\u2014"


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
    """Extract odds + bookmaker from the Recommended line, e.g. '1.45 (Bet365)'.

    Searches the >>> Recommended line first to avoid picking up odds from
    'Odds comparison' lines that list all 3 moneyline sides.
    """
    # Try Recommended line first (most reliable)
    rec_m = re.search(r"(?:>>>\s*)?Recommended:.*?@\s*([\d.]+)\s*\(([^)]+)\)", block)
    if rec_m:
        source = rec_m.group(2).split(",")[0].strip()
        if "confidence" not in source.lower():
            return f"{rec_m.group(1)} ({source})"
        return rec_m.group(1)
    # Fallback: first @ in block
    m = re.search(r"@\s*([\d.]+)\s*\(([^)]+)\)", block)
    if m:
        source = m.group(2).split(",")[0].strip()
        if "confidence" in source.lower():
            return m.group(1)
        return f"{m.group(1)} ({source})"
    return None


def _extract_odds_number(block: str) -> Optional[str]:
    """Extract just the numeric odds value."""
    m = re.search(r"@\s*([\d.]+)", block)
    return m.group(1) if m else None


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
    return m.group(1).title() if m else "\u2014"


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
    for m in re.finditer(r"\d+\.\s+(.+?\u2014\s*[\d.]+%.*)", block):
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
    market_emoji: str = "\u26bd",
    extra_fields: Optional[List[Tuple[str, str, bool]]] = None,
) -> discord.Embed:
    """Build a single bet-card embed for one fixture with logo support."""
    color, bar, conf_desc = _conf(f"{confidence} confidence" if confidence != "\u2014" else "")

    em = discord.Embed(color=color)

    # Clean fixture name as author — no emoji prefix
    league_logo = _get_league_logo(league) if league else None
    em.set_author(name=fixture)

    # Set home team logo as thumbnail
    home, away = _parse_teams_from_fixture(fixture)
    if home:
        home_logo = _get_team_logo(home)
        if home_logo:
            em.set_thumbnail(url=home_logo)

    if pick:
        em.add_field(name="\U0001f4cb Pick", value=f"**{pick}**", inline=True)
    if odds:
        em.add_field(name="\U0001f4b0 Odds", value=odds, inline=True)
    if confidence and confidence != "\u2014":
        em.add_field(name="\u2b50 Rating", value=f"{bar} {conf_desc}", inline=True)

    if model_p:
        em.add_field(name="\U0001f4ca Model", value=model_p, inline=True)
    if implied_p:
        em.add_field(name="\U0001f4d6 Books", value=implied_p, inline=True)
    if edge:
        sign_emoji = "\U0001f4c8" if edge.startswith("+") else "\U0001f4c9"
        em.add_field(name=f"{sign_emoji} Edge", value=edge, inline=True)

    if extra_fields:
        for name, value, inline in extra_fields:
            em.add_field(name=name, value=value, inline=inline)

    footer_parts = []
    if league:
        footer_parts.append(league)
    if projection:
        footer_parts.append(f"Projection: {projection}")
    footer_parts.append("Spick's Picks")
    league_logo_footer = _get_league_logo(league) if league else None
    em.set_footer(
        text=" \u2502 ".join(footer_parts),
        icon_url=league_logo_footer,
    )

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

    _STAT_FORM_ICONS: Dict[str, Tuple[str, str]] = {
        "goalscorer": ("\u26bd", "\u26aa"),
        "goals": ("\u26bd", "\u26aa"),
        "shots on target": ("\U0001f3af", "\u26aa"),
        "sot": ("\U0001f3af", "\u26aa"),
        "cards": ("\U0001f7e8", "\u26aa"),
        "booked": ("\U0001f7e8", "\u26aa"),
    }

    # Detect stat type from title for emoji
    title_lower = title.lower()
    stat_emoji = "\U0001f464"
    stat_key = ""
    for key, em in _STAT_EMOJI.items():
        if key in title_lower:
            stat_emoji = em
            stat_key = key
            break

    blocks = _split_fixture_blocks(text)
    embeds: List[discord.Embed] = []

    # Header embed
    league_logo = _get_league_logo(league) if league else None
    header = discord.Embed(
        title=f"{stat_emoji}  {title}",
        color=COLOR_BLUE,
    )
    if league:
        header.set_author(name=league, icon_url=league_logo)
    embeds.append(header)

    for block in blocks:
        fixture = _parse_fixture_name(block)
        home, away = _parse_teams_from_fixture(fixture)

        # Parse individual player recommendations from this fixture block
        players = _parse_player_prop_block(block)

        if not players:
            continue

        # Build one embed per fixture with all players as fields
        # Use highest confidence player's color for the embed
        best_conf = "low"
        for p in players:
            pc = p.get("confidence", "low").lower()
            if pc == "high":
                best_conf = "high"
                break
            elif pc == "medium" and best_conf != "high":
                best_conf = "medium"

        color = _conf_color(best_conf)
        em = discord.Embed(color=color)
        em.set_author(name=fixture)

        # Home team logo as thumbnail
        if home:
            home_logo = _get_team_logo(home)
            if home_logo:
                em.set_thumbnail(url=home_logo)

        for p in players[:5]:  # max 5 players per fixture
            name = p.get("name", "Unknown")
            team = p.get("team", "")
            pos = p.get("position", "")
            venue = p.get("venue", "")
            conf = p.get("confidence", "low").lower()
            prob = p.get("model_prob", "")

            # Build rating bar line
            bar = _conf_bar(conf)
            label = _conf_label(conf)

            # Venue icon
            venue_icon = "\U0001f3e0" if venue.upper() == "HOME" else "\u2708\ufe0f" if venue.upper() == "AWAY" else ""

            # Field name: stat emoji + player name (team) [POS] venue_icon
            field_name = f"{stat_emoji} {name}"
            if team:
                field_name += f" ({team})"
            if pos:
                field_name += f" [{pos}]"
            if venue_icon:
                field_name += f" {venue_icon}"

            # Build field value
            value_parts = []

            # Rating bar + probability + odds
            odds_str = ""
            if p.get("odds"):
                odds_str = f" @ {p['odds']}"
                if p.get("bookmaker"):
                    odds_str += f" ({p['bookmaker']})"

            if prob:
                value_parts.append(f"{bar} **{prob}** chance{odds_str}")
            else:
                value_parts.append(f"{bar} {label}{odds_str}")

            # Edge line when available
            edge = p.get("edge")
            if edge:
                value_parts.append(f"Edge: **{edge}** vs books")

            # Pick line
            pick_line = p.get("pick", "")
            if pick_line and not prob:
                value_parts.append(f"**{pick_line}**")

            # Recent form with visual icons
            recent_line = p.get("recent_line", "")
            if recent_line:
                # Try to convert recent form to visual icons
                form_icons = _format_recent_form(recent_line, stat_key, _STAT_FORM_ICONS)
                if form_icons:
                    value_parts.append(f"Recent: {form_icons}")
                else:
                    value_parts.append(f"Recent: {recent_line}")

            # Projection/role info on one line
            proj_line = p.get("projection_line", "")
            role_line = p.get("role_line", "")
            detail_parts = []
            if proj_line:
                detail_parts.append(proj_line)
            if role_line:
                detail_parts.append(role_line)
            if detail_parts:
                value_parts.append(" | ".join(detail_parts))

            # Opponent/matchup line
            opp_line = p.get("opponent_line", "")
            if opp_line:
                value_parts.append(f"vs {opp_line}")
            matchup_line = p.get("matchup_line", "")
            if matchup_line:
                value_parts.append(f"Matchup: {matchup_line}")

            em.add_field(
                name=field_name,
                value=_truncate("\n".join(value_parts) if value_parts else "No data", 1024),
                inline=False,
            )

        em.set_footer(
            text=f"{league} \u2502 Spick's Picks" if league else "Spick's Picks",
            icon_url=league_logo,
        )
        embeds.append(em)

    if len(embeds) > 10:
        embeds = embeds[:10]

    return embeds


def _format_recent_form(
    recent_text: str,
    stat_key: str,
    icon_map: Dict[str, Tuple[str, str]],
) -> str:
    """Convert a recent form string into visual icons.

    Looks for patterns like '2/6 scored', '3/6 booked', etc.
    and generates icon sequences.
    """
    # Try to parse "X/Y scored" or "X/Y booked" or similar
    m = re.search(r"(\d+)/(\d+)\s+(?:scored|goals?|hit|booked|carded|on target)", recent_text, re.IGNORECASE)
    if not m:
        return ""

    hits = int(m.group(1))
    total = int(m.group(2))
    if total <= 0 or total > 10:
        return ""

    hit_icon, miss_icon = icon_map.get(stat_key, ("\u2705", "\u26aa"))
    icons = [hit_icon] * hits + [miss_icon] * (total - hits)
    return "".join(icons)


def _parse_player_prop_block(block: str) -> List[Dict[str, str]]:
    """Parse player recommendation entries from a fixture block.

    Each player entry starts with an emoji + player name line.
    Handles both old format and new format with >>> pick lines.
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

        # NEW FORMAT: >>> Anytime Goalscorer — 34% chance @ 3.50 (Bet365)
        # Or: >>> Over 1.5 Shots on Target — 45% chance @ 2.10 (Bet365)
        pick_new = re.search(
            r">>>\s*(.+?)\s*[\u2014\-]{1,2}\s*(\d+%)\s*chance(?:\s*@\s*([\d.]+)\s*\(([^)]+)\))?",
            section,
        )
        if pick_new:
            p["pick"] = pick_new.group(1).strip()
            p["model_prob"] = pick_new.group(2)
            if pick_new.group(3):
                p["odds"] = pick_new.group(3)
                p["bookmaker"] = pick_new.group(4)
        else:
            # OLD FORMAT: >>> Recommended: Anytime Goalscorer (34% confidence)
            pick_m = re.search(
                r">>>\s*Recommended:\s*(.+?)(?:\s*\((\d+%)\s+confidence\))?$",
                section, re.MULTILINE,
            )
            if pick_m:
                p["pick"] = pick_m.group(1).strip()
                if pick_m.group(2):
                    p["model_prob"] = pick_m.group(2)

        # Extract odds comparison line: Model: 33% vs Books: 29% | Edge: +5%
        edge_m = re.search(
            r"Model:\s*(\d+%)\s*vs\s*Books:\s*(\d+%)\s*\|\s*Edge:\s*([+\-]\d+%)",
            section,
        )
        if edge_m:
            p["edge"] = edge_m.group(3)

        # Extract matchup modifier line
        matchup_m = re.search(r"Matchup:\s*(.+?)$", section, re.MULTILINE)
        if matchup_m:
            p["matchup_line"] = matchup_m.group(1).strip()

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

        # Extract confidence -- standalone line or inline
        conf_m = re.search(r"Confidence:\s*(\w+)", section)
        if conf_m:
            p["confidence"] = conf_m.group(1).strip().lower()
        else:
            # Infer from pick line or default
            conf_m2 = re.search(r"\b(high|medium|low)\s+confidence\b", section, re.IGNORECASE)
            if conf_m2:
                p["confidence"] = conf_m2.group(1).lower()

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

    league_logo = _get_league_logo(league) if league else None

    # Header embed
    header = discord.Embed(
        title=f"\U0001f465  {title}",
        color=COLOR_BLUE,
    )
    if league:
        header.set_author(name=league, icon_url=league_logo)
    embeds.append(header)

    for block in blocks:
        fixture = _parse_fixture_name(block)
        home, away = _parse_teams_from_fixture(fixture)
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

        color = _conf_color(best_conf)
        em = discord.Embed(color=color)
        em.set_author(name=fixture)

        # Home team logo as thumbnail
        if home:
            home_logo = _get_team_logo(home)
            if home_logo:
                em.set_thumbnail(url=home_logo)

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
                conf = t.get("confidence", "").lower()

                bar = _conf_bar(conf)
                label = _conf_label(conf)

                line = f"{bar} **{team_name}**: {proj} corners"
                if rec:
                    line += f"\n  \U0001f4cb {rec}"
                if prob:
                    line += f" ({prob})"
                corner_lines.append(line)

            # Match total
            match_total = _first_match(
                r"Match total:\s*([\d.]+)\s+projected corners", block)
            if match_total:
                corner_lines.append(f"\U0001f4ca Match total: **{match_total}** projected")

            em.add_field(
                name="\U0001f4d0 Corners",
                value=_truncate("\n".join(corner_lines), 1024),
                inline=False,
            )

        # CARDS section
        if cards:
            card_lines = []

            # Referee info
            ref_m = re.search(
                r"Referee:\s*(.+?)(?:\s*\u2014\s*(.+?))?$",
                block, re.MULTILINE,
            )
            if ref_m and "not yet assigned" not in (ref_m.group(0) or ""):
                ref_name = ref_m.group(1).strip()
                ref_detail = ref_m.group(2).strip() if ref_m.group(2) else ""
                ref_text = f"\U0001f9d1\u200d\u2696\ufe0f {ref_name}"
                if ref_detail:
                    ref_text += f" \u2014 {ref_detail}"
                card_lines.append(ref_text)

            for t in cards:
                team_name = t.get("team", "?")
                proj = t.get("projection", "?")
                rec = t.get("recommendation", "")
                prob = t.get("model_prob", "")
                conf = t.get("confidence", "").lower()

                bar = _conf_bar(conf)

                line = f"{bar} **{team_name}**: {proj} cards"
                if rec:
                    line += f"\n  \U0001f4cb {rec}"
                if prob:
                    line += f" ({prob})"
                card_lines.append(line)

            # Match total
            match_total = _first_match(
                r"Match total:\s*([\d.]+)\s+projected cards", block)
            if match_total:
                card_lines.append(f"\U0001f4ca Match total: **{match_total}** projected")

            em.add_field(
                name="\U0001f7e8 Cards",
                value=_truncate("\n".join(card_lines), 1024),
                inline=False,
            )

        em.set_footer(
            text=f"{league} \u2502 Spick's Picks" if league else "Spick's Picks",
            icon_url=league_logo,
        )
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
            "moneyline": "**Arsenal** @ 2.10 | 62% model | high",
            "btts": "**Yes** @ 1.75 | 58% model | medium",
            "spreads": "**Arsenal -0.5** @ 1.85 | 61% model | high",
            "sot": "**Over 8.5** @ 1.80 | 67% model | medium",
        },
        ...
    }
    """
    embeds: List[discord.Embed] = []

    league_logo = _get_league_logo(league) if league else None

    # Header
    header = discord.Embed(
        title=f"\U0001f4ca  Match Predictions \u2014 {league}",
        color=COLOR_BLUE,
    )
    if league_logo:
        header.set_author(name=league, icon_url=league_logo)
    embeds.append(header)

    _SECTION_EMOJIS = {
        "moneyline": "\U0001f3c6",
        "btts": "\U0001f91d",
        "spreads": "\U0001f4cf",
        "sot": "\U0001f3af",
        "goals": "\u26bd",
        "corners": "\U0001f4d0",
        "cards": "\U0001f7e8",
    }

    _SECTION_LABELS = {
        "moneyline": "Winner",
        "btts": "BTTS",
        "spreads": "Handicap",
        "sot": "Shots on Target",
        "goals": "Goals",
        "corners": "Corners",
        "cards": "Cards",
    }

    # Display order for markets
    _MARKET_ORDER = ["moneyline", "goals", "corners", "cards", "btts", "spreads", "sot"]

    for fixture, markets in fixture_data.items():
        if not markets:
            continue

        home, away = _parse_teams_from_fixture(fixture)

        # Format all available markets
        formatted_lines: List[Tuple[str, str]] = []  # (conf, formatted_line)
        best_conf = "low"

        for key in _MARKET_ORDER:
            val = markets.get(key)
            if not val:
                continue

            pick_m = re.search(r"\*\*(.+?)\*\*", val)
            odds_m = re.search(r"@\s*([\d.]+)", val)
            model_m = re.search(r"([\d.]+%)\s*model", val)
            edge_m = re.search(r"edge\s*([+\-][\d.]+%)", val)

            line_conf = "low"
            if "\U0001f7e2" in val or re.search(r"\bhigh\b", val, re.I):
                line_conf = "high"
            elif "\U0001f7e1" in val or re.search(r"\bmedium\b", val, re.I):
                line_conf = "medium"

            if line_conf == "high":
                best_conf = "high"
            elif line_conf == "medium" and best_conf != "high":
                best_conf = "medium"

            emoji = _SECTION_EMOJIS.get(key, "\u26bd")
            label = _SECTION_LABELS.get(key, key.title())
            bar = _conf_bar(line_conf)
            conf_desc = _conf_label(line_conf)

            pick_text = pick_m.group(1) if pick_m else "?"
            odds_text = f" @ {odds_m.group(1)}" if odds_m else ""

            line = f"{emoji} **{label}: {pick_text}**{odds_text}"
            detail_parts = [f"{bar} {conf_desc}"]
            if model_m:
                detail_parts.append(f"{model_m.group(1)} model")
            if edge_m:
                detail_parts.append(f"{edge_m.group(1)} edge")
            detail_line = " \u2022 ".join(detail_parts)
            line += f"\n{detail_line}"

            formatted_lines.append((line_conf, line))

        if not formatted_lines:
            continue

        # Also check for any markets not in _MARKET_ORDER
        for key, val in markets.items():
            if key in _MARKET_ORDER or not val:
                continue
            pick_m = re.search(r"\*\*(.+?)\*\*", val)
            if pick_m:
                emoji = _SECTION_EMOJIS.get(key, "\u26bd")
                label = _SECTION_LABELS.get(key, key.title())
                formatted_lines.append(("low", f"{emoji} **{label}: {pick_m.group(1)}**"))

        color = _conf_color(best_conf)
        em = discord.Embed(color=color)

        # Fixture name as author (clean, no emojis)
        em.set_author(name=f"{home or '?'}  vs  {away or '?'}")

        # Competition logo as thumbnail
        if league_logo:
            em.set_thumbnail(url=league_logo)

        # All markets in description
        em.description = "\n\n".join(line for _, line in formatted_lines)

        em.set_footer(text=f"{league} \u2502 Spick's Picks")
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
            "correct_score": "1. 2-1 -- 14.2% | 2. 1-1 -- 12.8% | ...",
            "intervals": "0-1: 22% | 2-3: 45% | 4-5: 28% | 6+: 5%",
        },
    }
    """
    embeds: List[discord.Embed] = []

    league_logo = _get_league_logo(league) if league else None

    header = discord.Embed(
        title=f"\U0001f3af  Correct Score & Goal Intervals \u2014 {league}",
        color=COLOR_BLUE,
    )
    if league_logo:
        header.set_author(name=league, icon_url=league_logo)
    embeds.append(header)

    for fixture, data in fixture_data.items():
        if not data:
            continue

        home, away = _parse_teams_from_fixture(fixture)

        em = discord.Embed(color=COLOR_BLUE)
        em.set_author(name=fixture)

        # Home team logo as thumbnail
        if home:
            home_logo = _get_team_logo(home)
            if home_logo:
                em.set_thumbnail(url=home_logo)

        cs = data.get("correct_score")
        if cs:
            em.add_field(name="\U0001f3af Most Likely Scores", value=_truncate(cs, 1024),
                         inline=False)

        iv = data.get("intervals")
        if iv:
            em.add_field(name="\U0001f4ca Goal Intervals", value=_truncate(iv, 1024),
                         inline=False)

        em.set_footer(
            text=f"{league} \u2502 Spick's Picks",
            icon_url=league_logo,
        )
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

        # Use confidence keyword (lowercase) for downstream parsing
        conf_lower = confidence.lower() if confidence != "\u2014" else ""
        if conf_lower:
            parts.append(conf_lower)

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
        emoji = "\U0001f4d0"
    elif "card" in title_lower:
        emoji = "\U0001f7e8"
    elif "sot" in title_lower or "shot" in title_lower:
        emoji = "\U0001f3af"
    elif "btts" in title_lower:
        emoji = "\U0001f91d"
    elif "moneyline" in title_lower or "winner" in title_lower:
        emoji = "\U0001f3c6"
    elif "handicap" in title_lower or "spread" in title_lower:
        emoji = "\U0001f4cf"
    elif "correct" in title_lower or "score" in title_lower:
        emoji = "\U0001f3af"
    elif "interval" in title_lower:
        emoji = "\U0001f4ca"
    elif "team" in title_lower:
        emoji = "\U0001f465"
    else:
        emoji = "\u26bd"

    blocks = _split_fixture_blocks(text)
    embeds: List[discord.Embed] = []

    league_logo = _get_league_logo(league) if league else None

    # Header embed
    header = discord.Embed(
        title=f"{emoji}  {title}",
        color=COLOR_BLUE,
    )
    if league:
        header.set_author(name=league, icon_url=league_logo)
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
                f"{'\U0001f3e0' if k == 'home' else '\U0001f91d' if k == 'draw' else '\u2708\ufe0f'} {v}"
                for k, v in ml_probs.items()
            )
            extra.append(("\U0001f4ca Win Probabilities", prob_line, False))

        # Special handling for correct score
        scores = _extract_correct_scores(block)
        if scores:
            scores_text = "\n".join(f"`{i+1}.` {s}" for i, s in enumerate(scores[:5]))
            extra.append(("\U0001f3af Top Scorelines", _truncate(scores_text, 1024), False))

        # Special handling for goal intervals
        intervals = _extract_interval_rows(block)
        if intervals:
            iv_text = "\n".join(f"`{band:<15}` {prob}" for band, prob in intervals)
            extra.append(("\U0001f4ca Intervals", _truncate(iv_text, 1024), False))

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
    """Format parlay output as a bet-slip-style embed with logos."""
    if not text or not text.strip():
        em = discord.Embed(
            title="\U0001f3b0  Parlay Slip",
            description="No parlay data available. Try a different date or league.",
            color=COLOR_BLUE,
        )
        return [em]

    embeds: List[discord.Embed] = []

    league_logo = _get_league_logo(league) if league else None

    # Parse legs -- handle both old format and new format with confidence appended
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
        title="\U0001f3b0  Parlay Slip",
        color=COLOR_PURPLE,
    )
    slip.set_author(name=f"League: {league}", icon_url=league_logo)

    for leg in legs:
        # Build confidence badge for this leg
        conf = leg.get("confidence")
        model_p = leg.get("model_prob")

        if conf:
            bar = _conf_bar(conf)
            label = _conf_label(conf)
            conf_line = f"\n{bar} {label}"
            if model_p:
                conf_line += f" \u2022 {model_p}"
        else:
            conf_line = ""

        # Get team logo for the first team in the fixture
        leg_home, _ = _parse_teams_from_fixture(leg["fixture"])

        # Leg league badge
        leg_league = leg.get("league", "")
        league_badge = f"[{leg_league}] " if leg_league and leg_league != league else ""

        slip.add_field(
            name=f"Leg {leg['num']}  {league_badge}",
            value=(
                f"**{leg['fixture']}**\n"
                f"\U0001f4cb {leg['pick']}\n"
                f"\U0001f4b0 {leg['odds']} ({leg['book']})"
                f"{conf_line}"
            ),
            inline=False,
        )

    # Combined odds bar
    if combo_odds:
        slip.add_field(
            name="\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500",
            value=f"**Combined Odds: {combo_odds}x**",
            inline=False,
        )

    # Confidence breakdown
    if conf_breakdown:
        slip.add_field(name="\u2b50 Confidence", value=conf_breakdown, inline=True)

    # Cross-check warnings
    if cross_warnings:
        slip.add_field(name="\u26a0\ufe0f Warnings", value=cross_warnings, inline=True)

    # Parse evidence for each leg
    why = _first_match(r"Why this parlay:\s*(.+?)(?:\n-|\Z)", text, 1)
    if why:
        slip.add_field(name="\U0001f4a1 Why", value=_truncate(why, 1024), inline=False)

    slip.set_footer(
        text="Spick's Picks",
        icon_url=league_logo,
    )
    embeds.append(slip)

    # Discord limit
    if len(embeds) > 10:
        embeds = embeds[:10]

    return embeds


def structured_parlay_embeds(result, league: Optional[str] = None) -> List[discord.Embed]:
    """Render a structured parlay result without regex reparsing."""
    selected_legs = list(getattr(result, "selected_legs", []) or [])
    request = getattr(result, "request", None)
    display_league = league or getattr(request, "display_league", None) or "Mixed"
    league_logo = _get_league_logo(display_league) if display_league else None

    if not selected_legs:
        return parlay_error_embeds("No parlay legs were selected.", [], league=display_league)

    slip = discord.Embed(
        title="\U0001f3b0  Parlay Slip",
        color=COLOR_PURPLE,
    )
    slip.set_author(name=f"League: {display_league}", icon_url=league_logo)

    for index, leg in enumerate(selected_legs, start=1):
        confidence = getattr(leg, "confidence", None)
        model_prob = getattr(leg, "model_prob", None)
        conf_line = ""
        if confidence:
            conf_line = f"\n{_conf_bar(confidence)} {_conf_label(confidence)}"
            if model_prob is not None:
                conf_line += f" \u2022 {model_prob:.0%}"

        warning = getattr(leg, "warning", None)
        warning_line = f"\n\u26a0\ufe0f {warning}" if warning else ""
        league_badge = f"[{getattr(leg, 'league', '')}] " if getattr(leg, "league", "") and getattr(leg, "league", "") != display_league else ""

        slip.add_field(
            name=f"Leg {index}  {league_badge}",
            value=(
                f"**{getattr(leg, 'fixture', '')}**\n"
                f"\U0001f4cb {getattr(leg, 'pick_display', '')}\n"
                f"\U0001f4b0 {getattr(leg, 'odds', 0.0):.2f} ({getattr(leg, 'bookmaker', 'Book') or 'Book'})"
                f"{conf_line}{warning_line}"
            ),
            inline=False,
        )

    slip.add_field(
        name="\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500",
        value=f"**Combined Odds: {getattr(result, 'combined_odds', 0.0):.2f}x**",
        inline=False,
    )

    confidence_breakdown = str(getattr(result, "confidence_breakdown", "") or "").strip()
    if confidence_breakdown:
        slip.add_field(name="\u2b50 Confidence", value=confidence_breakdown, inline=True)

    cross_warnings = list(getattr(result, "cross_warnings", []) or [])
    if cross_warnings:
        warning_text = _truncate("; ".join(cross_warnings), 1024)
        slip.add_field(name="\u26a0\ufe0f Warnings", value=warning_text, inline=True)

    summary = str(getattr(result, "summary_text", "") or "").strip()
    if summary:
        slip.add_field(name="\U0001f4a1 Why", value=_truncate(summary, 1024), inline=False)

    slip.set_footer(text="Spick's Picks", icon_url=league_logo)
    return [slip]


def parlay_error_embeds(message: str, notes: Optional[List[str]] = None, league: str = "Mixed") -> List[discord.Embed]:
    """Render a structured parlay failure state."""
    league_logo = _get_league_logo(league) if league else None
    em = discord.Embed(
        title="\U0001f3b0  Parlay Slip",
        description=message,
        color=COLOR_BLUE,
    )
    if league_logo:
        em.set_author(name=f"League: {league}", icon_url=league_logo)
    if notes:
        em.add_field(name="Notes", value=_truncate("\n".join(f"\u2022 {note}" for note in notes), 1024), inline=False)
    return [em]


def value_alert_embed(text: str, league: str) -> discord.Embed:
    """Format a value alert as a green bet-card embed with logos."""
    pick = _extract_pick(text)
    odds = _extract_odds(text)
    model = _extract_model_line(text)
    confidence = _extract_confidence(text)

    league_logo = _get_league_logo(league) if league else None

    # Try to find a fixture name for team logo
    fixture_m = re.search(r"([A-Z][\w\s.'()-]+?)\s+(?:vs|@)\s+([A-Z][\w\s.'()-]+)", text)
    home_logo = None
    if fixture_m:
        home_logo = _get_team_logo(fixture_m.group(1).strip())

    # Determine color from confidence
    if confidence and confidence != "\u2014":
        color = _conf_color(confidence.lower())
    else:
        color = COLOR_GREEN

    bar = _conf_bar(confidence.lower()) if confidence != "\u2014" else ""
    label = _conf_label(confidence.lower()) if confidence != "\u2014" else ""

    em = discord.Embed(
        title="\U0001f6a8  Value Alert",
        color=color,
    )
    em.set_author(name=f"League: {league}", icon_url=league_logo)

    if home_logo:
        em.set_thumbnail(url=home_logo)

    if pick:
        em.add_field(name="\U0001f4cb Pick", value=f"**{pick}**", inline=True)
    if odds:
        em.add_field(name="\U0001f4b0 Odds", value=odds, inline=True)
    if bar:
        em.add_field(name="\u2b50 Rating", value=f"{bar} {label}", inline=True)

    if model["model_p"]:
        em.add_field(name="\U0001f4ca Model", value=model["model_p"], inline=True)
    if model["edge"]:
        em.add_field(name="\U0001f4c8 Edge", value=model["edge"], inline=True)

    em.set_footer(
        text=f"{league} \u2502 Spick's Picks \u2502 Value Alert",
        icon_url=league_logo,
    )
    return em


# ── Fallback for unparseable output ──────────────────────────────────────────

def fixture_list_embed(
    fixtures: list,
    league: str,
    date_phrase: str,
) -> discord.Embed:
    """Build an embed listing fixtures for a given date with league logo."""
    league_logo = _get_league_logo(league) if league else None

    if not fixtures:
        em = discord.Embed(
            title=f"No {league} fixtures on {date_phrase}",
            description="Try a different date or league.",
            color=COLOR_BLUE,
        )
        if league_logo:
            em.set_author(name=league, icon_url=league_logo)
        return em

    em = discord.Embed(
        title=f"\U0001f4c5  Fixtures \u2014 {league} \u2014 {date_phrase}",
        color=COLOR_BLUE,
    )
    if league_logo:
        em.set_author(name=league, icon_url=league_logo)

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
                time_str = f" \u2014 {dt.strftime('%H:%M')} UTC"
            except Exception:
                pass
        lines.append(f"\u26bd **{home}** vs **{away}**{time_str}")

    em.description = "\n".join(lines)
    em.set_footer(
        text=f"{len(fixtures)} fixture(s) \u2502 Use /analyze to drill into a match \u2502 Spick's Picks",
        icon_url=league_logo,
    )
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
        league_logo = _get_league_logo(league) if league else None
        if league:
            em.set_author(name=f"League: {league}", icon_url=league_logo)
        em.set_footer(text=footer or "Spick's Picks")
        return [em]

    color = _conf(text)[0]
    league_logo = _get_league_logo(league) if league else None
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
            em.set_author(name=f"League: {league}", icon_url=league_logo)
        if i == len(chunks) - 1:
            em.set_footer(
                text=footer or "Spick's Picks",
                icon_url=league_logo if league else None,
            )
        embeds.append(em)
    return embeds[:10]
