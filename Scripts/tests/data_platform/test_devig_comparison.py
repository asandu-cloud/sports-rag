"""Comparing margin-removal methods on closing history (rework A3.1)."""
from datetime import date, timedelta
import random
import sqlite3

from Scripts.ops import devig_comparison as cmp
from Scripts.ops import football_data_import as fd


def longshot_loaded_history(tmp_path, n=3000, seed=3):
    """Books price true probabilities with a power-law margin: longshots carry more of it."""
    rng = random.Random(seed)
    path = tmp_path / "fd.sqlite3"
    db = sqlite3.connect(path)
    db.executescript(fd.SCHEMA)
    for i in range(1, n + 1):
        home = rng.uniform(0.2, 0.75)
        draw = min(0.3, 1 - home - 0.05)
        probs = [home, draw, 1 - home - draw]
        y = rng.choices(range(3), probs)[0]
        hg, ag = [(2, 0), (1, 1), (0, 1)][y]
        db.execute("INSERT INTO matches (id, fixture_api_id, season, match_date, fthg, ftag, link_status) "
                   "VALUES (?, ?, 2023, ?, ?, ?, 'matched')",
                   (i, i, (date(2023, 1, 2) + timedelta(days=7 * (i // 15))).isoformat(), hg, ag))
        for book, exponent in (("A", 0.85), ("B", 0.88), ("C", 0.86)):
            for side, p in zip(("home", "draw", "away"), probs):
                db.execute("INSERT INTO odds VALUES (?, '1x2', ?, 1, ?, NULL, ?, '', 0)", (i, book, side, 1 / p ** exponent))
    db.commit()
    db.close()
    return path


def test_power_beats_proportional_when_books_load_margin_onto_longshots(tmp_path):
    result = cmp.compare(longshot_loaded_history(tmp_path), "winner", resamples=200)
    rows = {r["method"]: r for r in result["methods"]}
    assert rows["proportional"]["difference_vs_proportional"] == 0
    assert rows["power"]["difference_vs_proportional"] < 0 and rows["power"]["interval95"][1] < 0
    assert result["fixtures"] == 3000 and result["best"] in ("power", "shin")


def test_report_lists_every_method_with_calibration(tmp_path):
    result = cmp.compare(longshot_loaded_history(tmp_path, n=300), "winner", resamples=50)
    assert {r["method"] for r in result["methods"]} == {"proportional", "additive", "power", "shin"}
    assert all("calibration_error" in r for r in result["methods"])
    assert "| power |" in cmp.markdown({"winner": result})
