"""Game-read bench (D77): run a model day by day over the history, as it would have run live.

Each day the model first predicts every match of the day, knowing only earlier days, then learns all of the day's
results. Before ``score_from`` it only learns (the warm-up seasons). Seasons after ``until`` are never shown to the
model, so a tuning run cannot see a locked season.
"""
from __future__ import annotations

from itertools import groupby
from typing import Optional, Protocol, Sequence

from Scripts.data_platform.game_read.forecast import Forecast
from Scripts.data_platform.game_read.history import Match


class Model(Protocol):
    name: str

    def predict(self, matches: Sequence[Match]) -> list:
        """One forecast per match of one day, before any of the day's results."""

    def learn(self, matches: Sequence[Match]) -> None:
        """The same day's matches, results included."""


def run(model: Model, matches: Sequence[Match], *, score_from: int = 2019, until: Optional[int] = None) -> list:
    """(match, forecast) for every match from season ``score_from`` (up to ``until``), days in order."""
    kept = sorted((m for m in matches if until is None or m.season <= until), key=lambda m: (m.day, m.id))
    rows = []
    for _, day in groupby(kept, key=lambda m: m.day):
        day = list(day)
        scored = [m for m in day if m.season >= score_from]
        if scored:
            forecasts = model.predict(scored)
            if len(forecasts) != len(scored):
                raise ValueError(f"{model.name} gave {len(forecasts)} forecasts for {len(scored)} matches")
            rows += zip(scored, forecasts)
        model.learn(day)
    return rows


__all__ = ["Forecast", "Model", "run"]
