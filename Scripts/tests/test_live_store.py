from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from Scripts.live.live_store import LiveStore


class LiveStoreTests(unittest.TestCase):
    def test_store_records_fixture_state_snapshot_and_alerts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "live.db"
            store = LiveStore(db_path)
            store.ensure_fixture(1, "EPL", "Arsenal", "Chelsea", "2026-04-04T18:00:00+00:00")
            store.record_state(1, {"fixture_id": 1, "status": "1H"}, status="1H", elapsed_min=14.0)
            store.record_snapshot(
                1,
                {"fixture_id": 1, "score": "1 - 0", "alerts": [{"type": "live_bet"}]},
                snapshot_ts="2026-04-04T18:15:00+00:00",
                elapsed_min=15.0,
                score="1 - 0",
            )
            store.record_live_odds(
                1,
                {"captured_at": "2026-04-04T18:14:30+00:00", "moneyline": [], "totals": {}},
                captured_at="2026-04-04T18:14:30+00:00",
            )
            store.record_alerts(
                1,
                "2026-04-04T18:15:00+00:00",
                [{"type": "live_bet", "severity": "warning", "message": "Bet Over 2.5"}],
            )
            store.mark_finished(1)

            with sqlite3.connect(db_path) as conn:
                fixture_count = conn.execute("SELECT COUNT(*) FROM live_fixtures").fetchone()[0]
                state_count = conn.execute("SELECT COUNT(*) FROM live_state_snapshots").fetchone()[0]
                snap_count = conn.execute("SELECT COUNT(*) FROM live_model_snapshots").fetchone()[0]
                odds_count = conn.execute("SELECT COUNT(*) FROM live_odds_snapshots").fetchone()[0]
                alert_count = conn.execute("SELECT COUNT(*) FROM live_alerts").fetchone()[0]
                finished_at = conn.execute("SELECT finished_at FROM live_fixtures WHERE fixture_id = 1").fetchone()[0]

            self.assertEqual(fixture_count, 1)
            self.assertEqual(state_count, 1)
            self.assertEqual(snap_count, 1)
            self.assertEqual(odds_count, 1)
            self.assertEqual(alert_count, 1)
            self.assertIsNotNone(finished_at)


if __name__ == "__main__":
    unittest.main()
