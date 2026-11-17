"""Game-read bench (D77, phase 2): rest, a variable for models 3–4 (D12's C1).

Each side's rest is ``(min(days, 7) − 7) / 7``: 0 with a week or more since its previous fixture, down to −6/7 after
one day. A side's log goals take two coefficients, shared by every league: one on its own rest and one on its
opponent's. A missing rest (before 2019/20, a team's first fixture we hold, an unlinked match) is 0. Days come from
the statistics export (``stats.MatchStats``), counted over the leagues and European cups we hold.
"""
from __future__ import annotations

from typing import Mapping, Optional, Sequence

from Scripts.data_platform.game_read.history import Match
from Scripts.data_platform.game_read.stats import MatchStats

FULL_REST = 7


def _scaled(days: Optional[int]) -> float:
    return 0.0 if days is None else (min(days, FULL_REST) - FULL_REST) / FULL_REST


class RestVariable:
    name, columns = "rest", 2

    def __init__(self, stats: Mapping[int, MatchStats]):
        self.stats = stats

    def features(self, match: Match) -> tuple:
        """(own rest, opponent's rest) for the home side, then for the away side."""
        stat = self.stats.get(match.id)
        home, away = (_scaled(stat.home_rest), _scaled(stat.away_rest)) if stat else (0.0, 0.0)
        return [home, away], [away, home]

    def learn(self, matches: Sequence[Match]) -> None:
        """Rest is known before each match; there is nothing to learn."""
