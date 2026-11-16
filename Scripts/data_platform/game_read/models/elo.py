"""Game-read bench (D77): Elo, model 1.

One rating per team, starting at 1500, updated after every match: a bigger win moves it more. Each league has its
own home advantage, learned as it goes. Each summer, ratings are pulled towards the league's mean by ``carry``.
A team new to a league starts at the level of the teams it replaces: the bottom three of last season's table
(the relegated teams), or, for a team coming down from a higher league in the data (the Premier League into the
Championship), the top three (the promoted teams). Who went up and down is known before a season starts, so this
uses no result from the future. With no table from last season, a new team starts at the league's mean plus
``new_team_offset``.

Win, draw or lose comes from the rating gap (home advantage included) through an ordered logit, so the draw is
likeliest at an even gap; the goal difference, clipped to −4…+4, through a second one. Both are refitted at each
season start on every earlier match. Elo gives no goal totals, so it is scored on the result and the handicap only.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Sequence

import numpy as np

from Scripts.data_platform import market_goals
from Scripts.data_platform.game_read import ordered
from Scripts.data_platform.game_read.forecast import Forecast, _differences
from Scripts.data_platform.game_read.history import Match

START = 1500.0
HOME_START = 60.0
SECOND_TIER = {"Championship"}
REPLACED = 3  # teams in last season's table taken as the ones a newcomer replaces
MAX_DIFFERENCE = 4
GAP_SCALE = 100.0  # the ordered logits read the gap in hundreds of points
MIN_FIT = 200  # matches before the ordered logits are fitted; until then a fixed prior

# k and carry chosen on the tuning seasons by result log loss, 16 Nov (k 10–40, carry 0.6–1.0; flat within 0.0003
# around k 12–20, carry 0.8–1.0). The home step and the new-team offset are set, not tuned.
K, HOME_STEP, CARRY, NEW_TEAM_OFFSET = 15.0, 1.0, 0.9, 0.0


def multiplier(margin: int) -> float:
    """The goal-difference multiplier: 1 for a margin of 0 or 1, 1.5 for 2, (11 + margin) / 8 from 3."""
    margin = abs(margin)
    return 1.0 if margin <= 1 else 1.5 if margin == 2 else (11 + margin) / 8


def _prior() -> Forecast:
    """A finite forecast before any fit: an average match's score model."""
    matrix = market_goals._matrix(1.45, 1.15)
    matrix = matrix / matrix.sum()
    differences = _differences(matrix)
    clipped = defaultdict(float)
    for d, p in differences.items():
        clipped[max(-MAX_DIFFERENCE, min(MAX_DIFFERENCE, d))] += p
    i, j = np.indices(matrix.shape)
    result = (float(matrix[i > j].sum()), float(matrix[i == j].sum()), float(matrix[i < j].sum()))
    return Forecast(result=result, goal_difference=dict(sorted(clipped.items())))


class Elo:
    name = "elo"

    def __init__(self, k: float = K, home_step: float = HOME_STEP, carry: float = CARRY,
                 new_team_offset: float = NEW_TEAM_OFFSET):
        self.k, self.home_step, self.carry, self.new_team_offset = k, home_step, carry, new_team_offset
        self.ratings = {}
        self.home = defaultdict(lambda: HOME_START)  # league -> home advantage in rating points
        self._season = {}  # league -> its current season
        self._members = defaultdict(set)  # (league, season) -> teams
        self._points = defaultdict(lambda: defaultdict(int))  # (league, season) -> team -> points
        self._last = {}  # team -> (league, season) of its latest match
        self._final = defaultdict(dict)  # (league, season) -> team -> rating after its latest match there
        self._starts = {}  # league -> {"down": relegated level, "up": promoted level} for this season
        self._seen = []  # (gap, result class, goal-difference class) of every learned match
        self._fit_season, self._result, self._difference = None, None, None

    # Seasons and newcomers

    def _roll(self, league: str, season: int) -> None:
        previous = self._season.get(league)
        self._season[league] = season
        if previous is None:
            return
        # last season's ratings as they ended there: a relegated team may already have started in another league
        final = {t: self._final[(league, previous)].get(t, self.ratings[t]) for t in self._members[(league, previous)]}
        mean = float(np.mean(list(final.values())))
        carried = {t: mean + self.carry * (rating - mean) for t, rating in final.items()}
        for team, rating in carried.items():
            if self._last.get(team) == (league, previous):
                self.ratings[team] = rating
        table = sorted(carried, key=lambda t: (self._points[(league, previous)][t], carried[t]))
        self._starts[league] = {"down": float(np.mean([carried[t] for t in table[:REPLACED]])),
                                "up": float(np.mean([carried[t] for t in table[-REPLACED:]]))}

    def _newcomer(self, team: str, league: str, season: int) -> float:
        starts = self._starts.get(league)
        if starts is None:
            current = [self.ratings[t] for t in self._members[(league, season)] if t != team and t in self.ratings]
            return (float(np.mean(current)) if current else START) + self.new_team_offset
        came = self._last.get(team)
        from_above = (came is not None and came[1] == season - 1 and came[0] not in SECOND_TIER
                      and league in SECOND_TIER)
        return starts["up"] if from_above else starts["down"]

    def _enter(self, match: Match) -> None:
        league, season = match.competition, match.season
        if self._season.get(league) != season and season > self._season.get(league, season - 1):
            self._roll(league, season)
        for team in (match.home, match.away):
            if team in self._members[(league, season)]:
                continue
            if team not in self._members[(league, season - 1)] or team not in self.ratings:
                self.ratings[team] = self._newcomer(team, league, season)
            self._members[(league, season)].add(team)
            self._last[team] = (league, season)

    def _gap(self, match: Match) -> float:
        return self.ratings[match.home] + self.home[match.competition] - self.ratings[match.away]

    # The model

    def _forecast(self, gap: float) -> Forecast:
        if self._result is None:
            return _prior()
        x = np.array([gap / GAP_SCALE])
        away, draw, home = self._result.probabilities(x)[0]
        differences = self._difference.probabilities(x)[0]
        return Forecast(result=(float(home), float(draw), float(away)),
                        goal_difference={d - MAX_DIFFERENCE: float(p) for d, p in enumerate(differences)})

    def _refit(self) -> None:
        if len(self._seen) < MIN_FIT:
            self._result = self._difference = None
            return
        seen = np.array(self._seen)
        x = seen[:, 0] / GAP_SCALE
        self._result = ordered.fit(x, seen[:, 1].astype(int), 3)
        self._difference = ordered.fit(x, seen[:, 2].astype(int), 2 * MAX_DIFFERENCE + 1)

    def predict(self, matches: Sequence[Match]) -> list:
        for match in matches:
            self._enter(match)
        season = max(match.season for match in matches)
        if season != self._fit_season:
            self._refit()
            self._fit_season = season
        return [self._forecast(self._gap(match)) for match in matches]

    def learn(self, matches: Sequence[Match]) -> None:
        for match in matches:
            self._enter(match)
        gaps = [self._gap(match) for match in matches]  # all before the day's updates
        for match, gap in zip(matches, gaps):
            home, away = match.home_goals, match.away_goals
            outcome = 1.0 if home > away else 0.5 if home == away else 0.0
            self._seen.append((gap, 2 if home > away else 1 if home == away else 0,
                               max(-MAX_DIFFERENCE, min(MAX_DIFFERENCE, home - away)) + MAX_DIFFERENCE))
            expected = 1 / (1 + 10 ** (-gap / 400))
            delta = self.k * multiplier(home - away) * (outcome - expected)
            self.ratings[match.home] += delta
            self.ratings[match.away] -= delta
            self.home[match.competition] += self.home_step * (outcome - expected)
            final = self._final[(match.competition, match.season)]
            final[match.home], final[match.away] = self.ratings[match.home], self.ratings[match.away]
            points = self._points[(match.competition, match.season)]
            points[match.home] += 3 if home > away else 1 if home == away else 0
            points[match.away] += 3 if away > home else 1 if home == away else 0
