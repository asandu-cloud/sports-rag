"""Game-read bench (D77, phase 2): shot ratings, a variable for models 3–4 (D12's S3, and S2's stand-in).

``ShotRatings`` is model 3 fitted on a shot statistic instead of goals (no rho): each team's shot attack and shot
defence, each league's level and home advantage, with the same decay, window, ridge and promoted-team shift. It
refits at most once every ``refit_days`` (its half-life is a year, so a week's matches move it little).

``ShotVariable`` turns it into each side's value for a match: the side's attack minus the opponent's defence on the
shot scale (the log of the side's expected count, less its league's level and home advantage). Both sides are 0
until both teams have ``MIN_MATCHES`` matches with the statistic.

The statistics (``kind``):

- ``box``: shots inside the box (ours, 2019/20 on);
- ``target``: shots on target (football-data's, 2012/13 on in most leagues);
- ``xg``: the derived shot-location xG, each shot valued by where it was taken (owner, 16 Nov):
  ``LOCATION_XG[0] · inside + LOCATION_XG[1] · outside``. Shots on target are left out, since every goal is one.
"""
from __future__ import annotations

from collections import Counter
from datetime import date
from typing import Mapping, Optional, Sequence

import numpy as np
from scipy.optimize import minimize

from Scripts.data_platform.game_read.history import Match
from Scripts.data_platform.game_read.models.dixon_coles import HALF_LIFE_DAYS, DixonColes
from Scripts.data_platform.game_read.stats import MatchStats

KINDS = ("box", "target", "xg")
MIN_MATCHES = 5
REFIT_DAYS = 7
# Goals per shot inside and outside the box, fitted by Poisson maximum likelihood on the tuning seasons (2019/20 to
# 2022/23: 27,748 sides with both counts), 16 Nov, and frozen (``fit_location_xg``). On the 1,930 sides of 2022/23
# with real xG it tracks real xG at a correlation of 0.76 (average 1.43 against 1.41), and goals at 0.36 (real 0.59).
LOCATION_XG = (0.16046, 0.02885)


def location_xg(inside: Optional[int], outside: Optional[int], weights: tuple = LOCATION_XG) -> Optional[float]:
    if inside is None or outside is None:
        return None
    return weights[0] * inside + weights[1] * outside


def values(stat: Optional[MatchStats], kind: str) -> Optional[tuple]:
    """(home, away) of the statistic, or None when either side's is missing."""
    if stat is None:
        return None
    if kind == "xg":
        found = (location_xg(stat.home_box, stat.home_outside), location_xg(stat.away_box, stat.away_outside))
    else:
        found = (getattr(stat, f"home_{kind}"), getattr(stat, f"away_{kind}"))
    return None if None in found else found


def fit_location_xg(stats: Mapping[int, MatchStats], matches: Sequence[Match]) -> tuple:
    """Goals per shot inside and outside the box: goals ~ Poisson(inside · a + outside · b), every side of
    ``matches`` with both counts."""
    rows = []
    for match in matches:
        stat = stats.get(match.id)
        if stat is None:
            continue
        for inside, outside, goals in ((stat.home_box, stat.home_outside, match.home_goals),
                                       (stat.away_box, stat.away_outside, match.away_goals)):
            if inside is not None and outside is not None:
                rows.append((inside, outside, goals))
    shots, goals = np.array([r[:2] for r in rows], dtype=float), np.array([r[2] for r in rows], dtype=float)

    def objective(weights):
        mean = np.maximum(shots @ weights, 1e-12)
        return -(goals * np.log(mean) - mean).mean(), -((goals / mean - 1)[:, None] * shots).mean(axis=0)

    found = minimize(objective, np.array([0.1, 0.03]), jac=True, method="L-BFGS-B", bounds=[(1e-6, None)] * 2)
    return float(found.x[0]), float(found.x[1])


class ShotRatings(DixonColes):
    name = "shots"

    def __init__(self, kind: str, stats: Mapping[int, MatchStats], half_life_days: float = HALF_LIFE_DAYS,
                 refit_days: int = REFIT_DAYS):
        if kind not in KINDS:
            raise ValueError(f"unknown shot statistic {kind!r}; known: {', '.join(KINDS)}")
        super().__init__(half_life_days=half_life_days, low_scores=False)
        self.kind, self.stats, self.refit_days = kind, stats, refit_days
        self.name = f"shots-{kind}"
        self._counted = lambda match: values(self.stats.get(match.id), self.kind)
        self._seen = Counter()  # team -> matches with the statistic
        self._fitted_on: Optional[date] = None

    def refresh(self, today: date) -> None:
        if self._stale and (self._fitted_on is None or (today - self._fitted_on).days >= self.refit_days):
            self.refit(today)
            self._fitted_on = today

    def gap(self, match: Match) -> Optional[tuple]:
        """Each side's shot attack minus the opponent's shot defence, or None while either team has too few
        matches with the statistic."""
        if self.fitted is None or min(self._seen[match.home], self._seen[match.away]) < MIN_MATCHES:
            return None
        fitted, home_new, away_new = self.fitted, self.is_new(match, match.home), self.is_new(match, match.away)
        attack, defence = fitted.attack.get, fitted.defence.get
        return (attack(match.home, 0.0) + fitted.promoted_attack * home_new
                - defence(match.away, 0.0) - fitted.promoted_defence * away_new,
                attack(match.away, 0.0) + fitted.promoted_attack * away_new
                - defence(match.home, 0.0) - fitted.promoted_defence * home_new)

    def learn(self, matches: Sequence[Match]) -> None:
        kept = [match for match in matches if self._counted(match) is not None]
        for match in kept:
            self._seen[match.home] += 1
            self._seen[match.away] += 1
        if kept:
            super().learn(kept)


class ShotVariable:
    columns = 1

    def __init__(self, kind: str, stats: Mapping[int, MatchStats], **settings):
        self.name, self.ratings = kind, ShotRatings(kind, stats, **settings)

    def features(self, match: Match) -> tuple:
        self.ratings.refresh(match.day)
        found = self.ratings.gap(match)
        return ([0.0], [0.0]) if found is None else ([found[0]], [found[1]])

    def learn(self, matches: Sequence[Match]) -> None:
        self.ratings.learn(matches)
