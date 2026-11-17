"""Game-read bench (D77): drifting ratings, model 4 (after Rue and Salvesen 2000; Koopman and Lit 2015).

Model 3's structure (a league level and home advantage, each team's attack and defence, one rho, a promoted-team
shift), but each rating moves at its own pace. It starts from model 3's fit on the first scored day. From then on,
after each match, every rating in a side's expected goals takes one filter step on the Poisson score of that side's
goals:

    step = (goals − expected) / (expected · total uncertainty + 1)
    rating += its uncertainty · step   (defence the other way)
    uncertainty −= its uncertainty² · expected / (expected · total uncertainty + 1)

(a Kalman step on the log of the expected goals, each rating's uncertainty kept on its own). A rating with more
uncertainty moves more. Uncertainty grows by ``gain`` a day between matches, and a team's by ``summer_widening``
more at its first match of a new season. Rho and the promoted-team shift stay at model 3's values; the shift is
added once, at a promoted team's first match in its new division.

Phase 2 (D12): with ``variables``, each side's ``beta · x`` is added to the log of its expected goals, in the
forecast and in the filter step. The coefficients come from a fit of model 3 at the start and again at each new
season's first day (only the coefficients: the ratings keep drifting), since a statistic that begins with the first
scored season is still 0 everywhere at the start.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import math
from typing import Optional, Sequence

import numpy as np

from Scripts.data_platform.game_read.forecast import Forecast
from Scripts.data_platform.game_read.history import Match
from Scripts.data_platform.game_read.models.dixon_coles import PRIOR, DixonColes
from Scripts.data_platform.game_read.models.market import _result, corrected

# Chosen on the tuning seasons, 16 Nov, by the sum of the result and over 2.5 log losses (gain 1e-6 to 3e-4, summer
# widening 0 to 0.1; flat within 0.0003 for gains 1e-6 to 1e-5 with a widening of 0.003 to 0.01).
GAIN, SUMMER_WIDENING = 3e-6, 0.01
NEW_VARIANCE = 0.0625  # a newcomer's uncertainty: about the spread of attack ratings within a league (0.25 squared)
TYPICAL_GOALS, LEAGUE_MATCHES_A_WEEK = 1.3, 10


def _settled(gain: float, matches_a_week: float) -> float:
    """The uncertainty a rating settles at when it gains ``gain`` a day and is seen ``matches_a_week`` times a week
    at about 1.3 goals a match: where it starts from model 3's fit."""
    q, r = 7 * gain, 1 / (TYPICAL_GOALS * matches_a_week)
    return (-q + math.sqrt(q * q + 4 * q * r)) / 2


@dataclass
class _Rating:
    value: float
    variance: float
    day: date
    season: int


class Drifting:
    name = "drift"

    def __init__(self, gain: float = GAIN, summer_widening: float = SUMMER_WIDENING, variables: Sequence = ()):
        self.gain, self.summer_widening = gain, summer_widening
        self._base = DixonColes(variables=variables)  # keeps who is new to a division and the variables; the start
        self.name = "+".join([type(self).name, *(v.name for v in self._base.variables)])
        self.beta, self._beta_season = (), None
        self._attack, self._defence = {}, {}  # team -> _Rating
        self._level, self._home = {}, {}  # league -> _Rating
        self._where = {}  # team -> (league, season) of its latest match
        self._last = {}  # team -> (day, season, whether that match was in its first season in the division)
        self.rho = self.promoted_attack = self.promoted_defence = None

    @property
    def started(self) -> bool:
        return self.rho is not None

    # Uncertainty

    def _ahead(self, rating: _Rating, day: date, season: Optional[int]) -> float:
        variance = rating.variance + self.gain * max((day - rating.day).days, 0)
        return variance + self.summer_widening if season is not None and season != rating.season else variance

    def uncertainty(self, team: str, day: date, season: int) -> tuple:
        return self._ahead(self._attack[team], day, season), self._ahead(self._defence[team], day, season)

    # The start, from model 3

    def _start(self, day: date) -> None:
        fitted = self._base.fitted
        self.rho, self.promoted_attack, self.promoted_defence = (fitted.rho, fitted.promoted_attack,
                                                                 fitted.promoted_defence)
        self.beta, self._beta_season = fitted.beta, max(season for _, season, _ in self._last.values())
        team, league = _settled(self.gain, 1), _settled(self.gain, LEAGUE_MATCHES_A_WEEK)
        for name, attack in fitted.attack.items():
            last_day, season, new = self._last[name]
            self._attack[name] = _Rating(attack + self.promoted_attack * new, team, last_day, season)
            self._defence[name] = _Rating(fitted.defence[name] + self.promoted_defence * new, team, last_day, season)
        for name, level in fitted.level.items():
            self._level[name] = _Rating(level, league, day, 0)
            self._home[name] = _Rating(fitted.home[name], league, day, 0)

    # Ratings as of a match

    def _league(self, match: Match) -> tuple:
        if match.competition not in self._level:
            level = float(np.mean([r.value for r in self._level.values()]))
            home = float(np.mean([r.value for r in self._home.values()]))
            return _Rating(level, NEW_VARIANCE, match.day, 0), _Rating(home, NEW_VARIANCE, match.day, 0)
        return self._level[match.competition], self._home[match.competition]

    def _team(self, match: Match, team: str) -> tuple:
        """The team's attack and defence values, and their uncertainty, as of this match."""
        new = self._base.is_new(match, team)
        if team not in self._attack:
            return (self.promoted_attack * new, NEW_VARIANCE), (self.promoted_defence * new, NEW_VARIANCE)
        shifted = new and self._where.get(team) != (match.competition, match.season)
        attack, defence = self._attack[team], self._defence[team]
        return ((attack.value + self.promoted_attack * shifted, self._ahead(attack, match.day, match.season)),
                (defence.value + self.promoted_defence * shifted, self._ahead(defence, match.day, match.season)))

    def _offsets(self, match: Match) -> tuple:
        """Each side's ``beta · x``: what the variables add to the log of its expected goals."""
        if not self.beta:
            return 0.0, 0.0
        home_x, away_x = self._base.features(match)
        return float(np.dot(self.beta, home_x)), float(np.dot(self.beta, away_x))

    def means(self, match: Match) -> tuple:
        level, home = (r.value for r in self._league(match))
        (home_attack, _), (home_defence, _) = self._team(match, match.home)
        (away_attack, _), (away_defence, _) = self._team(match, match.away)
        home_offset, away_offset = self._offsets(match)
        return (math.exp(level + home + home_attack - away_defence + home_offset),
                math.exp(level + away_attack - home_defence + away_offset))

    # The model

    def predict(self, matches: Sequence[Match]) -> list:
        for match in matches:
            self._base.features(match)  # before the day is learned
        if not self.started:
            self._base.predict(matches)  # fits model 3 once it has enough history
            if self._base.fitted is not None:
                self._start(min(match.day for match in matches))
        elif self._base.variables and max(match.season for match in matches) > self._beta_season:
            self._base.predict(matches)  # refits model 3, for its coefficients only
            self.beta, self._beta_season = self._base.fitted.beta, max(match.season for match in matches)
        forecasts = []
        for match in matches:
            home, away, rho = (*self.means(match), self.rho) if self.started else (*PRIOR, 0.0)
            matrix = corrected(home, away, rho)
            forecasts.append(Forecast(result=_result(matrix), score_matrix=matrix))
        return forecasts

    def _step(self, ratings: list, signs: list, goals: int, offset: float = 0.0) -> None:
        expected = math.exp(sum(sign * r.value for r, sign in zip(ratings, signs)) + offset)
        scale = expected * sum(r.variance for r in ratings) + 1
        step = (goals - expected) / scale
        for rating, sign in zip(ratings, signs):
            rating.value += sign * rating.variance * step
            rating.variance -= rating.variance ** 2 * expected / scale

    def _update(self, match: Match) -> None:
        league = match.competition
        if league not in self._level:
            self._level[league], self._home[league] = self._league(match)
        for rating in (self._level[league], self._home[league]):
            rating.variance, rating.day = self._ahead(rating, match.day, None), match.day
        sides = {}
        for team in (match.home, match.away):
            (attack, attack_variance), (defence, defence_variance) = self._team(match, team)
            self._attack[team] = _Rating(attack, attack_variance, match.day, match.season)
            self._defence[team] = _Rating(defence, defence_variance, match.day, match.season)
            sides[team] = self._attack[team], self._defence[team]
        level, home = self._level[league], self._home[league]
        home_offset, away_offset = self._offsets(match)
        self._step([level, home, sides[match.home][0], sides[match.away][1]], [1, 1, 1, -1], match.home_goals,
                   home_offset)
        self._step([level, sides[match.away][0], sides[match.home][1]], [1, 1, -1], match.away_goals, away_offset)

    def learn(self, matches: Sequence[Match]) -> None:
        self._base.learn(matches)  # after the start, model 3 is not refitted: only its record of newcomers is used
        for match in matches:
            if self.started:
                self._update(match)
            for team in (match.home, match.away):
                self._where[team] = (match.competition, match.season)
                self._last[team] = (match.day, match.season, self._base.is_new(match, team))
