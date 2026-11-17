"""Game-read bench (D77, phase 2): boosted trees on the market's error, model 6.

One LightGBM model per market: the result (three classes), over 2.5 and the real half-line main handicap (home covers
or not). Each starts from the market's log chances (``init_score``) and learns only corrections: its forecast is the
market's log chances plus the trees' sum, normalised. The inputs are the features pass's rows (``Features.row``);
``lead=False`` leaves out the lead's inputs (the market minus each member).

It is refitted at each season's first day. The number of rounds is set by early stopping on the last training season,
fitting on the seasons before it; the model is then refitted with that number on every earlier season from 2013.
When the trees do not beat the market on that last season, the number is 0 and the forecast is the market's. Matches
are picked by season, so a shared, finished features pass never lets a season's own results in.

The forecast carries the result and the real over 2.5 and handicap lines; derived lines get no chance.
"""
from __future__ import annotations

from typing import Mapping, Optional, Sequence

import lightgbm as lgb
import numpy as np

from Scripts.data_platform.game_read.features import Features
from Scripts.data_platform.game_read.forecast import Forecast
from Scripts.data_platform.game_read.history import Match, market_handicap, market_over25, market_result
from Scripts.data_platform.game_read.pool import _half

# The tuning grid (plan, task 6): leaves, minimum leaf, L2; the learning rate is fixed.
GRID = {"num_leaves": (7, 15), "min_data_in_leaf": (200, 500), "lambda_l2": (1, 10)}
DEFAULT = {"num_leaves": 7, "min_data_in_leaf": 200, "lambda_l2": 1}
LEARNING_RATE, MAX_ROUNDS, PATIENCE = 0.02, 3000, 100
MIN_ROWS = 1000  # training rows before a part is fitted
EPS = 1e-12


def _logit(p: float) -> float:
    return float(np.log(p / (1 - p)))


def _result(match: Match):
    market = market_result(match.pre)
    if market is None:
        return None
    won = 0 if match.home_goals > match.away_goals else 1 if match.home_goals == match.away_goals else 2
    return list(market), won


def _over25(match: Match):
    market = market_over25(match.pre)
    return None if market is None else (list(market), int(match.home_goals + match.away_goals > 2.5))


def _handicap(match: Match):
    market, line = market_handicap(match.pre), match.pre.handicap_line
    if market is None or not _half(line):
        return None
    return list(market), int(match.home_goals - match.away_goals > -float(line))


# part -> (the market's chances and what happened, None when the market has no price; classes)
PARTS = {"result": (_result, 3), "over25": (_over25, 2), "handicap": (_handicap, 2)}


def _init(markets: np.ndarray, classes: int) -> np.ndarray:
    """The market as the trees' starting score: log chances for three classes, log-odds of the first-listed side
    (over, home covers) for two."""
    logs = np.log(np.clip(markets, EPS, 1))
    return logs if classes == 3 else logs[:, 0] - logs[:, 1]


def _loss(chances: np.ndarray, won: np.ndarray, classes: int) -> float:
    """Mean log loss; for two classes ``chances`` is the first-listed side's and ``won`` is 1 when it won."""
    if classes == 3:
        return float(-np.mean(np.log(np.clip(chances[np.arange(len(won)), won], EPS, 1))))
    return float(-np.mean(np.log(np.clip(np.where(won == 1, chances, 1 - chances), EPS, 1))))


class Trees:
    def __init__(self, on: str = "error", features: Optional[Features] = None, params: Optional[Mapping] = None,
                 rounds: Optional[int] = None, lead: bool = True, name: Optional[str] = None):
        if on != "error":
            raise ValueError(f"unknown target {on!r}")
        self.on, self.lead = on, lead
        self.features = features if features is not None else Features({})
        self.params = {**DEFAULT, **(params or {})}
        self.rounds = rounds
        self.name = name or f"trees-{on}{'' if lead else '-nolead'}"
        self.fits = {}  # season -> {"train", "stop", per part: {"rounds", "stop_loss", "market_loss", "n"}}
        self._models = {}  # season -> {part: (booster or None, leagues)}
        self._vectors = {}
        self.columns = None

    # inputs

    def _vector(self, match: Match) -> np.ndarray:
        found = self._vectors.get(match.id)
        if found is None:
            row = self.features.row(match)
            if self.columns is None:
                self.columns = sorted(key for key in row if key != "league"
                                      and (self.lead or not key.startswith("lead_")))
            found = np.array([np.nan if row[key] is None else row[key] for key in self.columns], dtype=float)
            self._vectors[match.id] = found
        return found

    def _matrix(self, matches: Sequence[Match], leagues: Mapping) -> np.ndarray:
        codes = np.array([[leagues.get(m.competition, np.nan)] for m in matches], dtype=float)
        return np.hstack([np.vstack([self._vector(m) for m in matches]), codes])

    # fitting

    def _params(self, classes: int) -> dict:
        found = {"objective": "multiclass" if classes == 3 else "binary", "learning_rate": LEARNING_RATE,
                 "boost_from_average": False, "verbosity": -1, "seed": 0, "deterministic": True,
                 "force_row_wise": True, **self.params}
        if classes == 3:
            found["num_class"] = 3
        return found

    def _dataset(self, matches, part, leagues, reference=None):
        targets, classes = [PARTS[part][0](m) for m in matches], PARTS[part][1]
        kept = [(m, t) for m, t in zip(matches, targets) if t is not None]
        if not kept:
            return None, 0
        markets = np.array([t[0] for _, t in kept], dtype=float)
        won = np.array([t[1] for _, t in kept])
        x = self._matrix([m for m, _ in kept], leagues)
        found = lgb.Dataset(x, won, init_score=_init(markets, classes), categorical_feature=[x.shape[1] - 1],
                            reference=reference, free_raw_data=False)
        found.markets, found.won = markets, won
        return found, len(kept)

    def _fit_part(self, part, earlier, stop, leagues) -> tuple:
        classes = PARTS[part][1]
        full, n = self._dataset(earlier, part, leagues)
        found = {"n": n}
        rounds = self.rounds
        if rounds is None:
            before = [m for m in earlier if m.season < stop]
            last = [m for m in earlier if m.season == stop]
            train, n_train = self._dataset(before, part, leagues)
            valid, n_valid = self._dataset(last, part, leagues, reference=train) if train else (None, 0)
            rounds = 0
            if n_train >= MIN_ROWS and n_valid:
                booster = lgb.train(self._params(classes), train, num_boost_round=MAX_ROUNDS, valid_sets=[valid],
                                    callbacks=[lgb.early_stopping(PATIENCE, verbose=False)])
                market = valid.markets if classes == 3 else valid.markets[:, 0]
                found["market_loss"] = _loss(market, valid.won, classes)
                found["stop_loss"] = float(next(iter(booster.best_score["valid_0"].values())))
                if found["stop_loss"] < found["market_loss"]:
                    rounds = booster.best_iteration
        found["rounds"] = rounds
        if not rounds or n < MIN_ROWS:
            found["rounds"] = 0
            return None, found
        return lgb.train(self._params(classes), full, num_boost_round=rounds), found

    def _fit(self, season: int) -> None:
        earlier = [m for m in self.features.history if m.season < season]
        stop = max((m.season for m in earlier), default=None)
        leagues = {name: i for i, name in enumerate(sorted({m.competition for m in earlier}))}
        self.fits[season] = {"train": sorted({m.season for m in earlier}),
                             "stop": None if self.rounds is not None else stop}
        self._models[season] = {}
        for part in PARTS:
            booster, found = self._fit_part(part, earlier, stop, leagues) if earlier else (None, {"rounds": 0})
            self.fits[season][part] = found
            self._models[season][part] = (booster, leagues)

    # forecasting

    def _chances(self, match: Match, part: str, season: int) -> Optional[list]:
        target = PARTS[part][0](match)
        if target is None:
            return None
        market, classes = target[0], PARTS[part][1]
        booster, leagues = self._models[season][part]
        if booster is None or match.id not in self.features.forecasts:
            return market
        raw = booster.predict(self._matrix([match], leagues), raw_score=True)[0]
        score = _init(np.array([market], dtype=float), classes)[0] + raw
        if classes == 3:
            chances = np.exp(score - score.max())
            return (chances / chances.sum()).tolist()
        first = float(1 / (1 + np.exp(-score)))
        return [first, 1 - first]

    def predict(self, matches: Sequence[Match]) -> list:
        self.features.predict(matches)
        for season in sorted({m.season for m in matches} - set(self.fits)):
            self._fit(season)
        forecasts = []
        for match in matches:
            result = self._chances(match, "result", match.season)
            over = self._chances(match, "over25", match.season)
            handicap = self._chances(match, "handicap", match.season)
            lines = {}
            if over is not None:
                lines.update({("goals", "over", 2.5): over[0], ("goals", "under", 2.5): over[1]})
            if handicap is not None:
                line = float(match.pre.handicap_line)
                lines.update({("spreads", "home", line): handicap[0], ("spreads", "away", -line): handicap[1]})
            forecasts.append(Forecast(result=None if result is None else tuple(result), lines=lines or None))
        return forecasts

    def learn(self, matches: Sequence[Match]) -> None:
        self.features.learn(matches)
