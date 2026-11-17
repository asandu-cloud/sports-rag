"""Game-read bench (D77): Dixon–Coles ratings with time decay, model 3.

Each team has an attack and a defence, each league its own goal level and home advantage:

    log home goals = level + home advantage + home attack − away defence
    log away goals = level + away attack − home defence

with one rho for the low-score correction (as in the control line). A team in its first season in a division,
coming up from below or from outside the data, gets a shared promoted-team shift on its attack and defence. A team
coming down (from the Premier League into the Championship) keeps its ratings, on the same scale.

Phase 2 (D12): ``variables`` add fitted coefficients on statistics of each side, known before the match (a shot
rating, rest): ``log goals += beta · x``. A variable's values for a match are taken once, before the day's results
are learned, and kept for every later fit.

Matches count less as they age (half-life ``half_life_days``); only the last ``window_days`` are used. A small ridge
penalty on attack and defence pulls teams with little history towards their league. The penalty sits on the
weighted mean log-likelihood, so it weighs the same whatever the number of leagues. Fitted by L-BFGS with the exact
gradient, refitted in ``predict`` whenever matches have been learned since the last fit, starting from the last fit.

The forecast carries the score matrix, so every goal line, handicap and both teams to score.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from typing import Callable, Optional, Sequence

import numpy as np
from scipy.optimize import minimize

from Scripts.data_platform.game_read.forecast import Forecast
from Scripts.data_platform.game_read.history import Match
from Scripts.data_platform.game_read.models.market import RHO_BOUNDS, _result, corrected

# The half-life was chosen on the tuning seasons, 16 Nov, by the sum of the result and over 2.5 log losses, from 120,
# 180, 270, 365 and 540 days (270 was best on the result, 540 on over 2.5). The plan's ridge beat 1e-4 and 1e-2.
HALF_LIFE_DAYS, RIDGE, WINDOW_DAYS = 365.0, 1e-3, 1460
SECOND_TIER = {"Championship"}
MIN_FIT = 50  # matches before the first fit; until then an average match
PRIOR = (1.45, 1.15)
EPS = 1e-10


@dataclass(frozen=True)
class Design:
    leagues: list
    teams: list
    league: np.ndarray
    home: np.ndarray
    away: np.ndarray
    home_new: np.ndarray
    away_new: np.ndarray
    home_goals: np.ndarray
    away_goals: np.ndarray
    weight: np.ndarray
    home_x: np.ndarray  # matches × variables: the home side's values
    away_x: np.ndarray

    @property
    def variables(self) -> int:
        return self.home_x.shape[1]

    @property
    def size(self) -> int:
        """Levels and home advantages per league, attacks and defences per team, the two shifts, rho and one
        coefficient per variable."""
        return 2 * len(self.leagues) + 2 * len(self.teams) + 3 + self.variables


def design(matches: Sequence[Match], today: date, *, half_life_days: Optional[float], window_days: int,
           new: Callable[[Match, str], bool], goals: Optional[Callable[[Match], tuple]] = None,
           features: Optional[Callable[[Match], tuple]] = None, variables: int = 0) -> Design:
    """``goals`` gives what is counted for each side (the score by default); ``features`` each side's values of the
    ``variables``."""
    kept = [m for m in matches if (today - m.day).days <= window_days]
    counted = np.array([goals(m) if goals else (m.home_goals, m.away_goals) for m in kept],
                       dtype=float).reshape(len(kept), 2)
    sides = (np.array([features(m) for m in kept], dtype=float).reshape(len(kept), 2, variables) if features
             else np.zeros((len(kept), 2, 0)))
    leagues = sorted({m.competition for m in kept})
    teams = sorted({team for m in kept for team in (m.home, m.away)})
    league_at, team_at = {l: i for i, l in enumerate(leagues)}, {t: i for i, t in enumerate(teams)}
    ago = np.array([(today - m.day).days for m in kept], dtype=float)
    weight = np.ones(len(kept)) if half_life_days is None else np.exp(-np.log(2) * ago / half_life_days)
    return Design(leagues=leagues, teams=teams,
                  league=np.array([league_at[m.competition] for m in kept], dtype=int),
                  home=np.array([team_at[m.home] for m in kept], dtype=int),
                  away=np.array([team_at[m.away] for m in kept], dtype=int),
                  home_new=np.array([new(m, m.home) for m in kept], dtype=float),
                  away_new=np.array([new(m, m.away) for m in kept], dtype=float),
                  home_goals=counted[:, 0], away_goals=counted[:, 1], weight=weight,
                  home_x=sides[:, 0, :], away_x=sides[:, 1, :])


def _rho_at(d: Design) -> int:
    return 2 * len(d.leagues) + 2 * len(d.teams) + 2


def _unpack(theta: np.ndarray, d: Design) -> tuple:
    leagues, teams, rho = len(d.leagues), len(d.teams), _rho_at(d)
    level, home = theta[:leagues], theta[leagues:2 * leagues]
    attack = theta[2 * leagues:2 * leagues + teams]
    defence = theta[2 * leagues + teams:2 * leagues + 2 * teams]
    return level, home, attack, defence, theta[rho - 2], theta[rho - 1], theta[rho], theta[rho + 1:]


def _log_means(theta: np.ndarray, d: Design) -> tuple:
    level, home, attack, defence, shift_attack, shift_defence, _, beta = _unpack(theta, d)
    log_home = (level[d.league] + home[d.league] + attack[d.home] + shift_attack * d.home_new
                - defence[d.away] - shift_defence * d.away_new + d.home_x @ beta)
    log_away = (level[d.league] + attack[d.away] + shift_attack * d.away_new
                - defence[d.home] - shift_defence * d.home_new + d.away_x @ beta)
    return log_home, log_away


def objective(theta: np.ndarray, d: Design, ridge: float, low_scores: bool = True) -> tuple:
    """The weighted mean negative log-likelihood plus the ridge, and its exact gradient."""
    rho = theta[_rho_at(d)]
    log_home, log_away = _log_means(theta, d)
    home_mean, away_mean = np.exp(log_home), np.exp(log_away)
    y, z = d.home_goals, d.away_goals
    likelihood = y * log_home - home_mean + z * log_away - away_mean
    by_home, by_away, by_rho = y - home_mean, z - away_mean, np.zeros(len(y))
    if low_scores:
        tau, tau_home, tau_away, tau_rho = (np.ones(len(y)), np.zeros(len(y)), np.zeros(len(y)), np.zeros(len(y)))
        both = home_mean * away_mean
        cell = (y == 0) & (z == 0)
        tau[cell], tau_home[cell], tau_away[cell], tau_rho[cell] = (1 - both[cell] * rho, -both[cell] * rho,
                                                                    -both[cell] * rho, -both[cell])
        cell = (y == 0) & (z == 1)
        tau[cell], tau_home[cell], tau_rho[cell] = 1 + home_mean[cell] * rho, home_mean[cell] * rho, home_mean[cell]
        cell = (y == 1) & (z == 0)
        tau[cell], tau_away[cell], tau_rho[cell] = 1 + away_mean[cell] * rho, away_mean[cell] * rho, away_mean[cell]
        cell = (y == 1) & (z == 1)
        tau[cell], tau_rho[cell] = 1 - rho, -1.0
        tau = np.maximum(tau, EPS)
        likelihood = likelihood + np.log(tau)
        by_home, by_away, by_rho = by_home + tau_home / tau, by_away + tau_away / tau, tau_rho / tau

    w = d.weight / d.weight.sum()
    _, _, attack, defence, _, _, _, _ = _unpack(theta, d)
    value = -(w * likelihood).sum() + ridge * ((attack ** 2).sum() + (defence ** 2).sum())
    g_home, g_away = -w * by_home, -w * by_away
    leagues, teams = len(d.leagues), len(d.teams)
    grad = np.zeros_like(theta)
    grad[:leagues] = np.bincount(d.league, g_home + g_away, minlength=leagues)
    grad[leagues:2 * leagues] = np.bincount(d.league, g_home, minlength=leagues)
    grad[2 * leagues:2 * leagues + teams] = (np.bincount(d.home, g_home, minlength=teams)
                                             + np.bincount(d.away, g_away, minlength=teams) + 2 * ridge * attack)
    grad[2 * leagues + teams:2 * leagues + 2 * teams] = (-np.bincount(d.away, g_home, minlength=teams)
                                                         - np.bincount(d.home, g_away, minlength=teams)
                                                         + 2 * ridge * defence)
    at = _rho_at(d)
    grad[at - 2] = (g_home * d.home_new + g_away * d.away_new).sum()
    grad[at - 1] = -(g_home * d.away_new + g_away * d.home_new).sum()
    grad[at] = -(w * by_rho).sum()
    grad[at + 1:] = d.home_x.T @ g_home + d.away_x.T @ g_away
    return value, grad


@dataclass(frozen=True)
class Fitted:
    level: dict
    home: dict
    attack: dict
    defence: dict
    promoted_attack: float
    promoted_defence: float
    rho: float
    beta: tuple = ()  # one coefficient per variable

    def theta(self, d: Design) -> np.ndarray:
        """These values laid out for another design; a league or team not fitted here starts at the average."""
        level, home = np.mean(list(self.level.values())), np.mean(list(self.home.values()))
        return np.concatenate([[self.level.get(l, level) for l in d.leagues],
                               [self.home.get(l, home) for l in d.leagues],
                               [self.attack.get(t, 0.0) for t in d.teams],
                               [self.defence.get(t, 0.0) for t in d.teams],
                               [self.promoted_attack, self.promoted_defence, self.rho],
                               self.beta if len(self.beta) == d.variables else np.zeros(d.variables)])

    def means(self, d: Design) -> tuple:
        log_home, log_away = _log_means(self.theta(d), d)
        return np.exp(log_home), np.exp(log_away)

    def means_for(self, league: str, home: str, away: str, home_new: bool, away_new: bool,
                  home_x: Sequence = (), away_x: Sequence = ()) -> tuple:
        level = self.level.get(league, np.mean(list(self.level.values())))
        advantage = self.home.get(league, np.mean(list(self.home.values())))
        attack, defence = self.attack.get, self.defence.get
        log_home = (level + advantage + attack(home, 0.0) + self.promoted_attack * home_new
                    - defence(away, 0.0) - self.promoted_defence * away_new)
        log_away = (level + attack(away, 0.0) + self.promoted_attack * away_new
                    - defence(home, 0.0) - self.promoted_defence * home_new)
        if self.beta:
            log_home, log_away = log_home + np.dot(self.beta, home_x), log_away + np.dot(self.beta, away_x)
        return float(np.exp(log_home)), float(np.exp(log_away))


def fit(d: Design, ridge: float, low_scores: bool = True, start: Optional[Fitted] = None) -> Fitted:
    if start is None:
        theta = np.zeros(d.size)
        goals = (d.weight * (d.home_goals + d.away_goals)).sum() / (2 * d.weight.sum())
        theta[:len(d.leagues)] = np.log(max(goals, EPS))
    else:
        theta = start.theta(d)
    at = _rho_at(d)
    if not low_scores:
        theta[at] = 0.0
    bounds = [(None, None)] * at + [RHO_BOUNDS if low_scores else (0.0, 0.0)] + [(None, None)] * d.variables
    found = minimize(objective, theta, args=(d, ridge, low_scores), jac=True, method="L-BFGS-B", bounds=bounds,
                     options={"maxiter": 5000, "ftol": 1e-13, "gtol": 1e-9})
    level, home, attack, defence, shift_attack, shift_defence, rho, beta = _unpack(found.x, d)
    return Fitted(level=dict(zip(d.leagues, level.tolist())), home=dict(zip(d.leagues, home.tolist())),
                  attack=dict(zip(d.teams, attack.tolist())), defence=dict(zip(d.teams, defence.tolist())),
                  promoted_attack=float(shift_attack), promoted_defence=float(shift_defence), rho=float(rho),
                  beta=tuple(beta.tolist()))


class DixonColes:
    name = "dc"

    def __init__(self, half_life_days: float = HALF_LIFE_DAYS, ridge: float = RIDGE, window_days: int = WINDOW_DAYS,
                 variables: Sequence = (), low_scores: bool = True):
        self.half_life_days, self.ridge, self.window_days = half_life_days, ridge, window_days
        self.variables, self.low_scores = tuple(variables), low_scores
        self.name = "+".join([type(self).name, *(v.name for v in self.variables)])
        self.fitted: Optional[Fitted] = None
        self._features = {}  # match id -> (home side's values, away side's values), taken before the day is learned
        self._counted: Optional[Callable[[Match], tuple]] = None  # what is counted, when not goals
        self._history = []
        self._members = defaultdict(set)  # (league, season) -> teams
        self._where = {}  # (team, season) -> league
        self._new = {}  # (match id, team) -> in its first season in the division, not coming down
        self._stale = False

    def is_new(self, match: Match, team: str) -> bool:
        league, season = match.competition, match.season
        last = self._members.get((league, season - 1))
        if not last or team in last:
            return False
        came = self._where.get((team, season - 1))
        return not (came is not None and came not in SECOND_TIER and league in SECOND_TIER)

    def features(self, match: Match) -> tuple:
        """Each side's values of the variables for this match, from what the variables knew before its day."""
        if match.id not in self._features:
            found = [variable.features(match) for variable in self.variables]
            self._features[match.id] = ([v for home, _ in found for v in home], [v for _, away in found for v in away])
        return self._features[match.id]

    def refit(self, today: date) -> None:
        self._history = [m for m in self._history if (today - m.day).days <= self.window_days]
        if len(self._history) >= MIN_FIT:
            found = design(self._history, today, half_life_days=self.half_life_days, window_days=self.window_days,
                           new=lambda m, team: self._new[(m.id, team)], goals=self._counted,
                           features=self.features if self.variables else None,
                           variables=len(self.features(self._history[0])[0]) if self.variables else 0)
            self.fitted = fit(found, self.ridge, low_scores=self.low_scores, start=self.fitted)
        self._stale = False

    def _forecast(self, match: Match) -> Forecast:
        if self.fitted is None:
            home, away, rho = *PRIOR, 0.0
        else:
            home, away = self.fitted.means_for(match.competition, match.home, match.away,
                                               self.is_new(match, match.home), self.is_new(match, match.away),
                                               *self.features(match))
            rho = self.fitted.rho
        matrix = corrected(home, away, rho)
        return Forecast(result=_result(matrix), score_matrix=matrix)

    def predict(self, matches: Sequence[Match]) -> list:
        for match in matches:
            self.features(match)
        if self._stale:
            self.refit(min(match.day for match in matches))
        return [self._forecast(match) for match in matches]

    def learn(self, matches: Sequence[Match]) -> None:
        for match in matches:
            self.features(match)  # before the variables learn the day
            for team in (match.home, match.away):
                self._new[(match.id, team)] = self.is_new(match, team)
        for variable in self.variables:
            variable.learn(matches)
        for match in matches:
            for team in (match.home, match.away):
                self._members[(match.competition, match.season)].add(team)
                self._where[(team, match.season)] = match.competition
        self._history.extend(matches)
        self._stale = True
