"""Typed view of the registry.

Dataclasses — frozen so downstream code can safely share them. All
fields carry sensible defaults so new leagues can opt into features
incrementally.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Mapping, Optional, Sequence


@dataclass(frozen=True)
class ProviderMapping:
    """How one provider talks about a given competition.

    ``team_aliases`` resolves non-standard names the provider uses
    (e.g. ``"Verona" -> "Hellas Verona"``). ``endpoint_flags`` enables
    or disables optional endpoints per-competition.
    """

    provider: str                          # 'api_football', 'odds_api', ...
    provider_id: int                       # e.g. 39 for EPL on API-Football
    team_aliases: Mapping[str, str] = field(default_factory=dict)
    player_aliases: Mapping[str, str] = field(default_factory=dict)
    endpoint_flags: Mapping[str, bool] = field(default_factory=dict)
    notes: Optional[str] = None


@dataclass(frozen=True)
class SeasonRules:
    """When does a season start / end?

    ``europe_crossyear`` is the default: season year 2025 means 2025/26.
    ``single_year`` covers summer competitions (e.g. Copa América).
    """

    style: str = "europe_crossyear"        # 'europe_crossyear' | 'single_year'
    start_month: int = 7                   # July for most European leagues
    end_month: int = 6                     # June next year
    first_matchday_offset_days: int = 7


@dataclass(frozen=True)
class CompetitionEntry:
    code: str                              # 'EPL', 'UCL', ...
    name: str                              # human-readable
    country: Optional[str]
    competition_type: str                  # 'domestic_league' | 'continental_cup' | 'international'
    tier: int = 1                          # 1 = top flight, 2 = second tier, ...
    providers: Mapping[str, ProviderMapping] = field(default_factory=dict)
    season_rules: SeasonRules = field(default_factory=SeasonRules)
    default_markets: FrozenSet[str] = frozenset({"goals", "corners", "cards", "sot", "btts", "moneyline", "spreads"})
    update_cadence_minutes: int = 30       # baseline refresh frequency
    kb_entity_types: FrozenSet[str] = frozenset({"team", "player", "fixture"})
    feature_module: Optional[str] = None   # optional path to a league-specific feature pipeline
    enabled: bool = True
    labels: Sequence[str] = field(default_factory=tuple)  # e.g. ['top5', 'uefa']

    def provider(self, name: str) -> Optional[ProviderMapping]:
        return self.providers.get(name)


@dataclass(frozen=True)
class CompetitionRegistry:
    entries: Mapping[str, CompetitionEntry]
    version: str = "1"

    def __iter__(self):
        return iter(self.entries.values())

    def codes(self) -> List[str]:
        return list(self.entries.keys())

    def get(self, code: str) -> Optional[CompetitionEntry]:
        return self.entries.get(code)

    def require(self, code: str) -> CompetitionEntry:
        entry = self.entries.get(code)
        if entry is None:
            raise KeyError(f"Unknown competition code: {code}. Known: {', '.join(self.entries)}")
        return entry

    def filter(self, *, label: Optional[str] = None, competition_type: Optional[str] = None,
               enabled_only: bool = True) -> List[CompetitionEntry]:
        out = []
        for entry in self.entries.values():
            if enabled_only and not entry.enabled:
                continue
            if label and label not in entry.labels:
                continue
            if competition_type and entry.competition_type != competition_type:
                continue
            out.append(entry)
        return out
