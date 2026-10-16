"""Historical accuracy of each bookmaker against the consensus of the others (rework A1 step 4)."""
from datetime import date, timedelta
import math
import random
import sqlite3

import pytest

from Scripts.data_platform.features import bookmaker_accuracy as acc
from Scripts.ops import football_data_import as fd


def history(tmp_path, n=400, seed=7, noise=0.08):
    """Synthetic closing 1X2 history: 'Sharp' prices the true probabilities, 'Noisy' adds error."""
    rng = random.Random(seed)
    path = tmp_path / "fd.sqlite3"
    db = sqlite3.connect(path)
    db.executescript(fd.SCHEMA)
    start = date(2022, 8, 6)
    for i in range(1, n + 1):
        home = rng.uniform(0.25, 0.65)
        draw = 0.27
        away = 1 - home - draw
        outcome = rng.choices(["home", "draw", "away"], [home, draw, away])[0]
        hg, ag = {"home": (2, 0), "draw": (1, 1), "away": (0, 1)}[outcome]
        day = start + timedelta(days=7 * (i // 10))
        db.execute("INSERT INTO matches (id, fixture_api_id, season, match_date, fthg, ftag, link_status) "
                   "VALUES (?, ?, 2022, ?, ?, ?, 'matched')", (i, 1000 + i, day.isoformat(), hg, ag))
        noisy = [max(0.05, p + rng.gauss(0, noise)) for p in (home, draw, away)]
        books = {"Sharp": (home, draw, away), "Noisy": noisy, "Avg Book": (home, draw, away),
                 "Market average": (home, draw, away), "Pinnacle": (home, draw, away)}
        for book, probs in books.items():
            total = sum(probs)
            for side, p in zip(("home", "draw", "away"), probs):
                price = 1 / (p / total * 1.05)
                suspect = 1 if book == "Pinnacle" and i % 2 else 0
                db.execute("INSERT INTO odds VALUES (?, '1x2', ?, 1, ?, NULL, ?, '', ?)", (i, book, side, price, suspect))
    db.commit()
    db.close()
    return path


def test_sharp_book_beats_the_consensus_of_others_and_noisy_book_loses(tmp_path):
    rows = acc.evaluate(history(tmp_path, n=2000, noise=0.15), "winner", resamples=300)
    by_book = {r["bookmaker"]: r for r in rows}
    assert by_book["Noisy"]["mean_difference"] > 0 and by_book["Noisy"]["worse_than_consensus"]
    assert by_book["Sharp"]["mean_difference"] < by_book["Noisy"]["mean_difference"]
    assert rows[0]["bookmaker"] != "Noisy"  # sorted best first


def test_aggregates_are_not_scored_as_bookmakers_and_suspect_prices_are_dropped(tmp_path):
    by_book = {r["bookmaker"]: r for r in acc.evaluate(history(tmp_path), "winner", resamples=100)}
    assert "Market average" not in by_book
    assert by_book["Pinnacle"]["fixtures"] == 200  # the suspect half is excluded
    assert by_book["Sharp"]["fixtures"] == 400


def test_margin_and_coverage_are_reported(tmp_path):
    row = {r["bookmaker"]: r for r in acc.evaluate(history(tmp_path), "winner", resamples=50)}["Sharp"]
    assert row["mean_margin"] == pytest.approx(0.05, abs=1e-6)
    assert row["weeks"] >= 30 and row["seasons"] == [2022]


def test_a_book_needs_two_other_books_on_the_fixture_to_be_compared(tmp_path):
    path = tmp_path / "fd.sqlite3"
    db = sqlite3.connect(path)
    db.executescript(fd.SCHEMA)
    db.execute("INSERT INTO matches (id, fixture_api_id, season, match_date, fthg, ftag, link_status) "
               "VALUES (1, 1, 2022, '2022-08-06', 1, 0, 'matched')")
    for book in ("A", "B"):
        for side, price in (("home", 2.0), ("draw", 3.4), ("away", 3.8)):
            db.execute("INSERT INTO odds VALUES (1, '1x2', ?, 1, ?, NULL, ?, '', 0)", (book, side, price))
    db.commit()
    db.close()
    assert acc.evaluate(path, "winner", resamples=10) == []


def test_over_under_market_is_supported(tmp_path):
    path = tmp_path / "fd.sqlite3"
    db = sqlite3.connect(path)
    db.executescript(fd.SCHEMA)
    for i in range(1, 41):
        db.execute("INSERT INTO matches (id, fixture_api_id, season, match_date, fthg, ftag, link_status) "
                   "VALUES (?, ?, 2023, ?, ?, 0, 'matched')", (i, i, (date(2023, 1, 2) + timedelta(days=7 * i)).isoformat(), 3 if i % 2 else 1))
        for book, over in (("A", 1.9), ("B", 1.95), ("C", 2.0)):
            db.execute("INSERT INTO odds VALUES (?, 'ou25', ?, 1, 'over', NULL, ?, '', 0)", (i, book, over))
            db.execute("INSERT INTO odds VALUES (?, 'ou25', ?, 1, 'under', NULL, ?, '', 0)", (i, book, 1.9))
    db.commit()
    db.close()
    rows = acc.evaluate(path, "over2.5", resamples=50)
    assert {r["bookmaker"] for r in rows} == {"A", "B", "C"} and all(r["fixtures"] == 40 for r in rows)
    fair = (1 / 1.9) / (1 / 1.9 + 1 / 1.9)
    assert {r["bookmaker"]: r for r in rows}["A"]["book_logloss"] == pytest.approx(-math.log(fair))


def test_report_writes_both_events_and_a_readable_table(tmp_path):
    from Scripts.ops import bookmaker_accuracy_report as report
    result = report.build(history(tmp_path, n=200), resamples=50)
    assert set(result["events"]) == {"winner", "over2.5"} and result["events"]["winner"]
    text = report.markdown(result)
    assert "| Sharp |" in text and "consensus of the other bookmakers" in text
