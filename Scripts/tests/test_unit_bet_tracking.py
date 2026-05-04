from __future__ import annotations

import sqlite3
import tempfile
import unittest
from datetime import date
from pathlib import Path

from Scripts.rag_ingest.prediction_tracker import (
    backfill_unit_bet_slips_from_predictions,
    get_unit_bet_daily_summary,
    get_unit_bet_track_record,
    log_prediction,
    log_unit_bet_slip,
    resolve_unit_bet_slips,
)


def _leg(event_id: str, fixture: str, pick: str, odds: float) -> dict:
    home, away = fixture.split(" vs ")
    return {
        "event_id": event_id,
        "fixture_id": event_id,
        "league": "EPL",
        "fixture": fixture,
        "home_team": home,
        "away_team": away,
        "market": "goals",
        "market_group": "totals",
        "market_key": "totals",
        "pick": pick,
        "side": "over",
        "line": 2.5,
        "odds": odds,
        "bookmaker": "Bet365",
    }


def _prediction(db_path: Path, leg: dict) -> int:
    return log_prediction(
        home_team=leg["home_team"],
        away_team=leg["away_team"],
        league=leg["league"],
        market=leg["market"],
        pick=leg["pick"],
        side=leg["side"],
        line=leg["line"],
        odds=leg["odds"],
        bookmaker=leg["bookmaker"],
        source="unit_bets",
        fixture_id=leg["event_id"],
        db_path=db_path,
    )


def _set_outcomes(db_path: Path, outcomes: list[tuple[int, str]]) -> None:
    conn = sqlite3.connect(str(db_path))
    for pred_id, outcome in outcomes:
        conn.execute(
            "UPDATE predictions SET outcome = ?, actual_result = 3, graded_at = datetime('now') WHERE id = ?",
            (outcome, pred_id),
        )
    conn.commit()
    conn.close()


class UnitBetTrackingTests(unittest.TestCase):
    def test_double_counts_as_one_winning_unit_bet(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "predictions.db"
            today = date.today().isoformat()
            legs = [
                _leg("1", "Home FC vs Away FC", "Over 2.5", 1.50),
                _leg("2", "North FC vs South FC", "Over 2.5", 1.40),
            ]
            ids = [_prediction(db_path, leg) for leg in legs]
            _set_outcomes(db_path, [(ids[0], "hit"), (ids[1], "hit")])

            slip_id = log_unit_bet_slip(
                title="Double",
                combined_odds=2.10,
                legs=legs,
                prediction_ids=ids,
                prediction_date=today,
                db_path=db_path,
            )
            result = resolve_unit_bet_slips(prediction_date=today, db_path=db_path)
            daily = get_unit_bet_daily_summary(prediction_date=today, db_path=db_path)
            record = get_unit_bet_track_record(db_path=db_path)

        self.assertGreater(slip_id, 0)
        self.assertEqual(result["graded"], 1)
        self.assertEqual(daily["wins"], 1)
        self.assertEqual(daily["losses"], 0)
        self.assertAlmostEqual(daily["units_profit"], 1.10)
        self.assertEqual(record["total_graded"], 1)
        self.assertAlmostEqual(record["units_profit"], 1.10)

    def test_double_with_one_miss_loses_one_unit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "predictions.db"
            today = date.today().isoformat()
            legs = [
                _leg("1", "Home FC vs Away FC", "Over 2.5", 1.55),
                _leg("2", "North FC vs South FC", "Over 2.5", 1.45),
            ]
            ids = [_prediction(db_path, leg) for leg in legs]
            _set_outcomes(db_path, [(ids[0], "hit"), (ids[1], "miss")])
            log_unit_bet_slip(
                title="Double",
                combined_odds=2.25,
                legs=legs,
                prediction_ids=ids,
                prediction_date=today,
                db_path=db_path,
            )

            result = resolve_unit_bet_slips(prediction_date=today, db_path=db_path)
            daily = get_unit_bet_daily_summary(prediction_date=today, db_path=db_path)

        self.assertEqual(result["miss"], 1)
        self.assertEqual(daily["losses"], 1)
        self.assertAlmostEqual(daily["units_profit"], -1.0)

    def test_pushed_leg_reduces_parlay_odds(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "predictions.db"
            today = date.today().isoformat()
            legs = [
                _leg("1", "Home FC vs Away FC", "Over 2.5", 1.80),
                _leg("2", "North FC vs South FC", "Over 2.5", 1.35),
            ]
            ids = [_prediction(db_path, leg) for leg in legs]
            _set_outcomes(db_path, [(ids[0], "hit"), (ids[1], "push")])
            log_unit_bet_slip(
                title="Double",
                combined_odds=2.43,
                legs=legs,
                prediction_ids=ids,
                prediction_date=today,
                db_path=db_path,
            )

            resolve_unit_bet_slips(prediction_date=today, db_path=db_path)
            daily = get_unit_bet_daily_summary(prediction_date=today, db_path=db_path)

        self.assertEqual(daily["wins"], 1)
        self.assertEqual(daily["pushes"], 0)
        self.assertAlmostEqual(daily["units_profit"], 0.80)

    def test_backfill_infers_legacy_single_then_double(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "predictions.db"
            today = date.today().isoformat()
            legs = [
                _leg("1", "Home FC vs Away FC", "Over 2.5", 1.90),
                _leg("2", "North FC vs South FC", "Over 2.5", 1.50),
                _leg("3", "East FC vs West FC", "Over 2.5", 1.40),
            ]
            ids = [_prediction(db_path, leg) for leg in legs]
            _set_outcomes(db_path, [(ids[0], "hit"), (ids[1], "hit"), (ids[2], "miss")])

            result = backfill_unit_bet_slips_from_predictions(db_path=db_path)
            daily = get_unit_bet_daily_summary(prediction_date=today, db_path=db_path)
            record = get_unit_bet_track_record(db_path=db_path)

        self.assertEqual(result["slips_created"], 2)
        self.assertEqual(result["slips_settled"], 2)
        self.assertEqual([s["title"] for s in daily["slips"]], ["Single", "Double"])
        self.assertEqual(daily["wins"], 1)
        self.assertEqual(daily["losses"], 1)
        self.assertAlmostEqual(record["units_profit"], -0.10)

    def test_backfill_infers_legacy_double_then_single(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "predictions.db"
            today = date.today().isoformat()
            legs = [
                _leg("1", "Home FC vs Away FC", "Over 2.5", 1.50),
                _leg("2", "North FC vs South FC", "Over 2.5", 1.40),
                _leg("3", "East FC vs West FC", "Over 2.5", 1.85),
            ]
            ids = [_prediction(db_path, leg) for leg in legs]
            _set_outcomes(db_path, [(ids[0], "hit"), (ids[1], "hit"), (ids[2], "hit")])

            result = backfill_unit_bet_slips_from_predictions(db_path=db_path)
            daily = get_unit_bet_daily_summary(prediction_date=today, db_path=db_path)

        self.assertEqual(result["slips_created"], 2)
        self.assertEqual([s["title"] for s in daily["slips"]], ["Double", "Single"])
        self.assertEqual(daily["wins"], 2)
        self.assertAlmostEqual(daily["units_profit"], 1.10 + 0.85)


if __name__ == "__main__":
    unittest.main()
