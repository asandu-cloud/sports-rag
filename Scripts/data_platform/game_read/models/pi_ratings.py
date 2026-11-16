"""Game-read bench (D77): pi-ratings, model 2 (Constantinou and Fenton, 2013).

Each team has a home rating and an away rating, measured in goals and starting at 0. A rating R stands for an
expected goal difference of sign(R) · (10^(|R| / 3) − 1); a match's predicted goal difference is the home team's
home value minus the away team's away value. After the match, the error against the real goal difference is
weighed by 3 · log10(1 + |error|): the rating at the venue moves by ``learning`` times that, and the team's other
rating by ``cross`` times the move. As in the paper, a team new to a league starts at 0 and nothing is pulled
back over the summer.

The result and the goal difference come from the predicted goal difference through ordered logits refitted at each
season start, as for Elo. Pi-ratings give no goal totals, so they are scored on the result and the handicap only.
"""
from __future__ import annotations

from collections import defaultdict
import math
from typing import Sequence

from Scripts.data_platform.game_read import ordered
from Scripts.data_platform.game_read.history import Match

BASE, SCALE = 10.0, 3.0
# Chosen on the tuning seasons by result log loss, 16 Nov (learning 0.02–0.1, cross 0.2–0.9; flat within 0.0005
# around learning 0.04–0.06, cross 0.3–0.5). The paper's starting values were 0.035 and 0.7.
LEARNING, CROSS = 0.05, 0.5


def expected(rating: float) -> float:
    """The expected goal difference a rating stands for."""
    return math.copysign(BASE ** (abs(rating) / SCALE) - 1, rating)


class PiRatings:
    name = "pi"

    def __init__(self, learning: float = LEARNING, cross: float = CROSS):
        self.learning, self.cross = learning, cross
        self.home_rating = defaultdict(float)
        self.away_rating = defaultdict(float)
        self._outcomes = ordered.Outcomes()
        self._fit_season = None

    def predicted(self, match: Match) -> float:
        return expected(self.home_rating[match.home]) - expected(self.away_rating[match.away])

    def predict(self, matches: Sequence[Match]) -> list:
        season = max(match.season for match in matches)
        if season != self._fit_season:
            self._outcomes.refit()
            self._fit_season = season
        return [self._outcomes.forecast(self.predicted(match)) for match in matches]

    def learn(self, matches: Sequence[Match]) -> None:
        predictions = [self.predicted(match) for match in matches]  # all before the day's updates
        for match, prediction in zip(matches, predictions):
            self._outcomes.add(prediction, match.home_goals, match.away_goals)
            error = match.home_goals - match.away_goals - prediction
            change = self.learning * math.copysign(SCALE * math.log10(1 + abs(error)), error)
            self.home_rating[match.home] += change
            self.away_rating[match.home] += self.cross * change
            self.away_rating[match.away] -= change
            self.home_rating[match.away] -= self.cross * change
