"""Game-read bench, task 2: the margins of our saved reads (D77)."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import date, datetime, timezone

import pytest

from Scripts.data_platform import recommendations
from Scripts.data_platform.game_read import margins

HOME, AWAY = "Lyon", "Lens"


def option(key, side, point, odds, book, team=None):
    return {"market_key": key, "side": side, "team": team, "point": point, "odds": odds, "bookmaker": book}


def result(group, options):
    return {"market": {"group": group, "key": group}, "fixture": {"home_team": HOME, "away_team": AWAY},
            "projection": {"value": 0.5, "components": {}}, "decision": {"status": "no_bet"},
            "context": {"decision_audit": {"input_options": options, "candidates": []}}}


def read(over=1.90, under=1.95, *, three_and_a_half=False):
    """One saved read: two books, the same prices, so each typical price is that price."""
    goals, spreads, btts = [], [], []
    for book in ("A", "B"):
        goals += [option("totals", "over", 2.5, over, book), option("totals", "under", 2.5, under, book)]
        if three_and_a_half:
            goals += [option("totals", "over", 3.5, 3.00, book), option("totals", "under", 3.5, 1.36, book)]
        spreads += [option("spreads", None, -0.5, 1.95, book, team=HOME),
                    option("spreads", None, 0.5, 1.90, book, team=AWAY)]
        btts += [option("btts", "yes", None, 1.80, book), option("btts", "no", None, 2.00, book)]
    return [result("goals", goals), result("spreads", spreads), result("btts", btts)]


def test_booksum_adds_the_two_sides_of_each_line():
    sums = margins.booksums(recommendations.options(read()))
    assert sums[("goals", 2.5)] == pytest.approx(1 / 1.90 + 1 / 1.95)
    assert sums[("spreads", -0.5)] == pytest.approx(1 / 1.95 + 1 / 1.90)  # keyed by the home side's line
    assert ("spreads", 0.5) not in sums
    assert sums[("btts", None)] == pytest.approx(1 / 1.80 + 1 / 2.00)


def test_measure_takes_the_median_per_line_and_counts_reads():
    reads = [read(1.90, 1.95, three_and_a_half=True), read(1.80, 2.00), read(2.00, 1.80)]
    table = margins.measure(reads)
    assert table[("goals", 2.5)]["n"] == 3
    assert table[("goals", 2.5)]["booksum"] == pytest.approx(sorted(
        [1 / 1.90 + 1 / 1.95, 1 / 1.80 + 1 / 2.00, 1 / 2.00 + 1 / 1.80])[1])
    assert table[("goals", 3.5)] == {"booksum": pytest.approx(1 / 3.00 + 1 / 1.36), "n": 1}


def test_booksum_falls_back_to_the_market_then_to_105():
    table = {("goals", 2.5): 1.06, ("goals", 3.5): 1.08}
    assert margins.booksum(table, "goals", 2.5) == 1.06
    assert margins.booksum(table, "goals", 0.5) == pytest.approx(1.07)
    assert margins.booksum(table, "btts", None) == 1.05


def test_load_reads_back_what_save_wrote(tmp_path):
    path = tmp_path / "margins.json"
    margins.save({("goals", 2.5): {"booksum": 1.061, "n": 640}, ("btts", None): {"booksum": 1.072, "n": 300},
                  ("spreads", -1.5): {"booksum": 1.058, "n": 200}}, path, measured_on="2026-11-15", reads=700,
                 days={"from": "2026-09-05", "to": "2026-11-08", "count": 65})
    assert margins.load(path) == {("goals", 2.5): 1.061, ("btts", None): 1.072, ("spreads", -1.5): 1.058}


def test_export_holds_the_gate_once_per_day_over_the_domestic_leagues(monkeypatch):
    holds, asked = [], []

    @contextmanager
    def gate():
        holds.append(1)
        yield

    class Board:
        @staticmethod
        def _get_match_read_service():
            return object()

        @staticmethod
        def _load_matchday_reads(service, *, league, target_date):
            asked.append((league, target_date))
            return [{"league": league}]

        @staticmethod
        def _shown_at_kickoff(reads, *, now):
            return [({"data": {"canonical_results": read()}}, None) for _ in reads]

    monkeypatch.setattr(margins, "_board", lambda: Board)
    table = margins.export([date(2026, 11, 7), date(2026, 11, 8)], now=datetime(2026, 11, 15, tzinfo=timezone.utc),
                           gate=gate)
    assert len(holds) == 2
    assert {league for league, _ in asked} == set(margins.LEAGUES)
    assert not {"UCL", "UEL", "UECL"} & set(margins.LEAGUES) and len(margins.LEAGUES) == 10
    assert table[("goals", 2.5)]["n"] == 20
