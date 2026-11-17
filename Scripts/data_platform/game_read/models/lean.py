"""Game-read bench (D77, phase 2): the lean-away blend, model 5b.

D20's blend of the market and one member, ``logit p = a + b·logit p_market + c·logit p_member`` (``blend.fit``, a
logarithmic pool: three-way on the result, two-way on over 2.5), with ``c`` free to fall below zero: the phase 1
locked run's lead was a blend leaning away from the model. It is refitted at each season's first day on every
earlier season's recorded matches (the features pass, from 2013), so a season is always forecast by a fit that never
saw it.

``member`` names one of ``models``, or ``average`` (``average+xg``…): the members' log-linear pool, its weights per
part fitted on the same earlier seasons (``average.fit_all``). Where the market's chance is missing there is no
forecast; where the member's chance or a fit is missing, the forecast is the market's.
"""
from __future__ import annotations

from typing import Mapping, Optional, Sequence

import numpy as np

from Scripts.data_platform import blend
from Scripts.data_platform.game_read.features import FEATURES_FROM, Features, over_chance, result_chances
from Scripts.data_platform.game_read.forecast import Forecast
from Scripts.data_platform.game_read.history import Match, market_over25, market_result
from Scripts.data_platform.game_read.models import average

MIN_FIT = 500  # recorded matches before a blend is fitted
OVER, UNDER = ("goals", "over", 2.5), ("goals", "under", 2.5)


def _result(match: Match) -> int:
    return 0 if match.home_goals > match.away_goals else 1 if match.home_goals == match.away_goals else 2


def _over(match: Match) -> int:
    return 0 if match.home_goals + match.away_goals > 2.5 else 1


def _pair(chance: Optional[float]) -> Optional[list]:
    return None if chance is None else [chance, 1 - chance]


PARTS = {  # part -> (the market's chances, the member's, what happened)
    "result": (lambda m: market_result(m.pre), result_chances, _result),
    "over25": (lambda m: None if (found := market_over25(m.pre)) is None else list(found), lambda f: _pair(
        over_chance(f)), _over),
}


class LeanAway:
    def __init__(self, member: str, models: Mapping, start: int = FEATURES_FROM):
        self.member, self.name = member, f"lean-{member}"
        self.pooled = member.split("+")[0] == "average"
        self.features = Features(models, start=start)
        self.fits = {}  # season -> {"seasons", "n", "weights", "result", "over25"}

    def _said(self, match: Match, weights) -> Forecast:
        forecasts = self.features.forecasts[match.id]
        return average.combine(forecasts, weights) if self.pooled else forecasts[self.member]

    def _fit(self, season: int) -> dict:
        earlier = [m for m in self.features.history if m.season < season]
        weights = None
        if self.pooled and earlier:
            weights = average.fit_all({name: [(m, self.features.forecasts[m.id][name]) for m in earlier]
                                       for name in self.features.models})
        fitted = {"seasons": sorted({m.season for m in earlier}), "n": len(earlier), "weights": weights}
        for part, (market_of, member_of, happened) in PARTS.items():
            rows = []
            for match in earlier:
                market, said = market_of(match), member_of(self._said(match, weights))
                if market is not None and said is not None:
                    rows.append((market, said, happened(match)))
            fitted[part] = None if len(rows) < MIN_FIT else blend.fit(
                np.array([r[0] for r in rows], dtype=float), np.array([r[1] for r in rows], dtype=float),
                [r[2] for r in rows])
        return fitted

    def _part(self, match: Match, part: str, said: Forecast, fitted: dict) -> Optional[list]:
        market_of, member_of, _ = PARTS[part]
        market = market_of(match)
        if market is None:
            return None
        member = member_of(said)
        if member is None or fitted[part] is None:
            return list(market)
        return blend.predict(fitted[part], [market], [member])[0].tolist()

    def predict(self, matches: Sequence[Match]) -> list:
        self.features.predict(matches)
        for season in {m.season for m in matches} - set(self.fits):
            self.fits[season] = self._fit(season)
        forecasts = []
        for match in matches:
            fitted = self.fits[match.season]
            said = self._said(match, fitted["weights"])
            result, over = self._part(match, "result", said, fitted), self._part(match, "over25", said, fitted)
            forecasts.append(Forecast(result=None if result is None else tuple(result),
                                      lines=None if over is None else {OVER: over[0], UNDER: over[1]}))
        return forecasts

    def learn(self, matches: Sequence[Match]) -> None:
        self.features.learn(matches)
