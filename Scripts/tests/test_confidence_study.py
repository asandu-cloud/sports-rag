"""A5.3 study (D44): one run reports the spread cut-off, its check on the held-out week and the Spix check."""
from datetime import datetime, timedelta, timezone

from Scripts.ops import confidence_study as study

START = datetime(2026, 10, 9, 15, tzinfo=timezone.utc)


def rows():
    """Two weeks of audited lines: tight-spread picks keep value at the close, scattered ones lose it."""
    out = []
    for i in range(240):
        tight = i % 2 == 0
        out.append({"fixture_id": str(i), "stage": "pre_match", "market": "corners" if i % 3 else "goals",
                    "kickoff": (START + timedelta(hours=i * 1.4)).isoformat(), "p_ref": 0.5, "p_spix": 0.55,
                    "best_odds": 2.1, "best_book": "A", "selected": False, "y": float(i % 4 < 2), "w": 1.0,
                    "book_count": 6, "book_spread": 0.005 if tight else 0.04 + (i % 7) / 1000,
                    "p_close": 0.52 if tight else 0.46})
    return out


def test_the_study_reports_the_cutoff_its_holdout_and_spix():
    result = study.run(rows(), resamples=200)
    assert result["cutoff"] is not None and result["passes"] is True
    assert result["holdout"]["high"]["value_at_close"] > result["holdout"]["low"]["value_at_close"]
    assert set(result["spix"]) == {"corners", "goals"} and 0.2 <= result["share_high"] <= 0.8
    assert "Holdout" in study.summary(result)
