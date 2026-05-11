"""
rivalry_context.py -- deterministic derby/rivalry context for fixtures.

The first version is intentionally conservative.  It detects curated rivalry
fixtures and applies market-specific modifiers where the effect is most
defensible before backtesting: cards get the largest lift, corners/SoT get a
small tempo lift, and goals stay neutral by default.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple


@dataclass(frozen=True)
class RivalryContext:
    label: str = ""
    rivalry_type: str = ""
    intensity: float = 0.0
    goals_modifier: float = 1.0
    corners_modifier: float = 1.0
    cards_modifier: float = 1.0
    sot_modifier: float = 1.0
    evidence: str = ""
    source: str = "none"
    matched_teams: Tuple[str, str] = field(default_factory=tuple)


_PROFILE_MODIFIERS: Dict[str, Dict[str, float]] = {
    "elite": {"goals": 1.00, "corners": 1.02, "cards": 1.10, "sot": 1.02},
    "major": {"goals": 1.00, "corners": 1.015, "cards": 1.08, "sot": 1.015},
    "regional": {"goals": 1.00, "corners": 1.00, "cards": 1.05, "sot": 1.00},
    "historic": {"goals": 1.00, "corners": 1.00, "cards": 1.04, "sot": 1.00},
}


_ALIASES = {
    "ac milan": "milan",
    "milan": "milan",
    "barca": "barcelona",
    "inter milan": "inter",
    "internazionale": "inter",
    "inter": "inter",
    "as roma": "roma",
    "roma": "roma",
    "ss lazio": "lazio",
    "lazio": "lazio",
    "paris saint germain": "psg",
    "paris saint-germain": "psg",
    "psg": "psg",
    "atletico madrid": "atletico madrid",
    "athletic bilbao": "athletic club",
    "athletic club": "athletic club",
    "real betis": "real betis",
    "betis": "real betis",
    "borussia moenchengladbach": "borussia monchengladbach",
    "borussia monchengladbach": "borussia monchengladbach",
    "borussia m gladbach": "borussia monchengladbach",
    "bayern munich": "bayern munich",
    "fc bayern munich": "bayern munich",
    "bayern": "bayern munich",
    "fc koln": "koln",
    "koln": "koln",
    "cologne": "koln",
    "bayer leverkusen": "bayer leverkusen",
    "leverkusen": "bayer leverkusen",
    "saint etienne": "saint etienne",
    "st etienne": "saint etienne",
    "stade rennais": "rennes",
    "rennes": "rennes",
    "man united": "manchester united",
    "man utd": "manchester united",
    "manchester utd": "manchester united",
    "man city": "manchester city",
    "spurs": "tottenham",
    "tottenham hotspur": "tottenham",
    "west ham united": "west ham",
    "newcastle united": "newcastle",
    "nottingham forest": "nottingham forest",
}


def _normalize_team(name: str) -> str:
    text = unicodedata.normalize("NFKD", str(name or ""))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower()
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    tokens = [
        t for t in text.split()
        if t not in {"fc", "cf", "afc", "sc", "club", "football", "calcio"}
    ]
    key = " ".join(tokens).strip()
    return _ALIASES.get(key, key)


def _pair(a: str, b: str) -> frozenset:
    return frozenset((_normalize_team(a), _normalize_team(b)))


_RIVALRIES: Dict[frozenset, Dict[str, object]] = {
    # Spain
    _pair("Real Madrid", "Barcelona"): {
        "label": "El Clasico", "type": "national_rivalry", "intensity": 1.00, "profile": "elite",
    },
    _pair("Real Madrid", "Atletico Madrid"): {
        "label": "Madrid Derby", "type": "city_derby", "intensity": 0.90, "profile": "major",
    },
    _pair("Barcelona", "Espanyol"): {
        "label": "Barcelona Derby", "type": "city_derby", "intensity": 0.75, "profile": "regional",
    },
    _pair("Sevilla", "Real Betis"): {
        "label": "Seville Derby", "type": "city_derby", "intensity": 0.95, "profile": "major",
    },
    _pair("Athletic Club", "Real Sociedad"): {
        "label": "Basque Derby", "type": "regional_derby", "intensity": 0.80, "profile": "regional",
    },
    _pair("Valencia", "Villarreal"): {
        "label": "Valencian Community Derby", "type": "regional_derby", "intensity": 0.55, "profile": "historic",
    },

    # England
    _pair("Manchester United", "Manchester City"): {
        "label": "Manchester Derby", "type": "city_derby", "intensity": 0.90, "profile": "major",
    },
    _pair("Liverpool", "Everton"): {
        "label": "Merseyside Derby", "type": "city_derby", "intensity": 0.90, "profile": "major",
    },
    _pair("Arsenal", "Tottenham"): {
        "label": "North London Derby", "type": "city_derby", "intensity": 0.90, "profile": "major",
    },
    _pair("Chelsea", "Tottenham"): {
        "label": "London rivalry", "type": "regional_rivalry", "intensity": 0.65, "profile": "regional",
    },
    _pair("Arsenal", "Chelsea"): {
        "label": "London rivalry", "type": "regional_rivalry", "intensity": 0.55, "profile": "historic",
    },
    _pair("West Ham", "Tottenham"): {
        "label": "London rivalry", "type": "regional_rivalry", "intensity": 0.55, "profile": "historic",
    },
    _pair("Newcastle", "Sunderland"): {
        "label": "Tyne-Wear Derby", "type": "regional_derby", "intensity": 0.95, "profile": "major",
    },
    _pair("Nottingham Forest", "Derby County"): {
        "label": "East Midlands Derby", "type": "regional_derby", "intensity": 0.85, "profile": "regional",
    },

    # Italy
    _pair("AC Milan", "Inter"): {
        "label": "Derby della Madonnina", "type": "city_derby", "intensity": 1.00, "profile": "elite",
    },
    _pair("Roma", "Lazio"): {
        "label": "Derby della Capitale", "type": "city_derby", "intensity": 1.00, "profile": "elite",
    },
    _pair("Juventus", "Torino"): {
        "label": "Turin Derby", "type": "city_derby", "intensity": 0.80, "profile": "regional",
    },
    _pair("Juventus", "Inter"): {
        "label": "Derby d'Italia", "type": "national_rivalry", "intensity": 0.90, "profile": "major",
    },
    _pair("Napoli", "Roma"): {
        "label": "Derby del Sole", "type": "regional_rivalry", "intensity": 0.65, "profile": "regional",
    },

    # Germany
    _pair("Borussia Dortmund", "Bayern Munich"): {
        "label": "Der Klassiker", "type": "national_rivalry", "intensity": 0.80, "profile": "regional",
    },
    _pair("Borussia Dortmund", "FC Schalke 04"): {
        "label": "Revierderby", "type": "regional_derby", "intensity": 1.00, "profile": "elite",
    },
    _pair("Koln", "Borussia Monchengladbach"): {
        "label": "Rhein Derby", "type": "regional_derby", "intensity": 0.85, "profile": "regional",
    },
    _pair("Koln", "Bayer Leverkusen"): {
        "label": "Rhein Derby", "type": "regional_derby", "intensity": 0.65, "profile": "regional",
    },
    _pair("Union Berlin", "Hertha Berlin"): {
        "label": "Berlin Derby", "type": "city_derby", "intensity": 0.80, "profile": "regional",
    },

    # France
    _pair("Paris Saint Germain", "Marseille"): {
        "label": "Le Classique", "type": "national_rivalry", "intensity": 0.95, "profile": "major",
    },
    _pair("Lyon", "Saint Etienne"): {
        "label": "Rhone-Alpes Derby", "type": "regional_derby", "intensity": 0.95, "profile": "major",
    },
    _pair("Lille", "Lens"): {
        "label": "Derby du Nord", "type": "regional_derby", "intensity": 0.85, "profile": "regional",
    },
    _pair("Nice", "Monaco"): {
        "label": "Cote d'Azur Derby", "type": "regional_derby", "intensity": 0.65, "profile": "regional",
    },
    _pair("Nantes", "Rennes"): {
        "label": "Breton rivalry", "type": "regional_rivalry", "intensity": 0.60, "profile": "historic",
    },

    # UK / Europe fixtures can appear in European competitions.
    _pair("Celtic", "Rangers"): {
        "label": "Old Firm", "type": "city_derby", "intensity": 1.00, "profile": "elite",
    },
}


def get_rivalry_context(home: str, away: str, league: str = "") -> RivalryContext:
    """Return rivalry/derby context for a fixture, or neutral context."""
    home_key = _normalize_team(home)
    away_key = _normalize_team(away)
    entry = _RIVALRIES.get(frozenset((home_key, away_key)))
    if not entry:
        return RivalryContext()

    profile = str(entry.get("profile") or "regional")
    modifiers = _PROFILE_MODIFIERS.get(profile, _PROFILE_MODIFIERS["regional"])
    label = str(entry.get("label") or "Rivalry fixture")
    intensity = float(entry.get("intensity") or 0.0)
    rivalry_type = str(entry.get("type") or "rivalry")
    evidence = (
        f"{label} detected ({rivalry_type.replace('_', ' ')}, intensity {intensity:.2f}); "
        f"cards {modifiers['cards']:.2f}x, corners {modifiers['corners']:.3f}x, "
        f"SoT {modifiers['sot']:.3f}x, goals neutral"
    )
    return RivalryContext(
        label=label,
        rivalry_type=rivalry_type,
        intensity=intensity,
        goals_modifier=modifiers["goals"],
        corners_modifier=modifiers["corners"],
        cards_modifier=modifiers["cards"],
        sot_modifier=modifiers["sot"],
        evidence=evidence,
        source="curated",
        matched_teams=(home_key, away_key),
    )


def rivalry_context_evidence_line(ctx: Optional[RivalryContext]) -> Optional[str]:
    if ctx is None or ctx.source == "none" or not ctx.label:
        return None
    return f"Rivalry: {ctx.evidence}."
