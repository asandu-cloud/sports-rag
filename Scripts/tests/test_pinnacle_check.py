"""Pinnacle spot checks against OddsPapi's timestamped history (rework A1 step 4)."""
from datetime import datetime, timedelta, timezone
import json

import pytest

from Scripts.ops import pinnacle_check as pc

KO = datetime(2026, 10, 10, 14, tzinfo=timezone.utc)


def snap(minutes_before, price):
    return {"createdAt": (KO - timedelta(minutes=minutes_before)).isoformat().replace("+00:00", "Z"), "price": price}


def history_body():
    return {"bookmakers": {"pinnacle": {"markets": {"101": {"outcomes": {
        "101": {"players": {"0": [snap(300, 2.0), snap(60, 1.8), snap(-30, 1.2)]}},
        "102": {"players": {"0": [snap(300, 3.5), snap(60, 3.8), snap(-30, 6.0)]}},
        "103": {"players": {"0": [snap(300, 4.0), snap(60, 4.9), snap(-30, 15.0)]}},
    }}}}}}


def test_price_at_a_moment_uses_the_latest_change_before_it_and_never_after_kickoff():
    timeline = pc.timeline(history_body(), "pinnacle", pc.MARKETS["winner"], kickoff=KO)
    assert pc.state_at(timeline, KO - timedelta(minutes=120))[1] == (2.0, 3.5, 4.0)
    assert pc.state_at(timeline, KO - timedelta(minutes=10))[1] == (1.8, 3.8, 4.9)
    assert pc.state_at(timeline, KO + timedelta(minutes=40))[1] == (1.8, 3.8, 4.9)  # in-play changes ignored
    assert pc.closing(timeline) == (KO - timedelta(minutes=60), (1.8, 3.8, 4.9))
    assert pc.state_at(timeline, KO - timedelta(minutes=400)) is None


def test_lag_is_how_long_ago_pinnacle_last_showed_our_price():
    timeline = pc.timeline(history_body(), "pinnacle", pc.MARKETS["winner"], kickoff=KO)
    fresh = pc.compare((1.8, 3.8, 4.9), timeline, KO - timedelta(minutes=10))
    assert fresh["gap_pts"] == pytest.approx(0) and fresh["lag_minutes"] == 0
    stale = pc.compare((2.0, 3.5, 4.0), timeline, KO - timedelta(minutes=10))
    assert stale["lag_minutes"] == pytest.approx(50)  # Pinnacle moved away from this price 50 min earlier
    assert stale["gap_pts"] > 5
    unknown = pc.compare((2.5, 3.0, 3.0), timeline, KO - timedelta(minutes=10))
    assert unknown["lag_minutes"] is None


def test_fixtures_match_on_kickoff_and_team_names():
    ours = [{"fixture_id": 1, "kickoff": KO, "home": "Arsenal", "away": "Leeds"},
            {"fixture_id": 2, "kickoff": KO, "home": "Chelsea", "away": "Bournemouth"},
            {"fixture_id": 3, "kickoff": KO + timedelta(days=1), "home": "Everton", "away": "Fulham"}]
    theirs = [{"fixtureId": "a", "startTime": "2026-10-10T14:00:00.000Z", "participant1Name": "Arsenal FC", "participant2Name": "Leeds United"},
              {"fixtureId": "b", "startTime": "2026-10-10T14:00:00.000Z", "participant1Name": "Chelsea FC", "participant2Name": "AFC Bournemouth"},
              {"fixtureId": "c", "startTime": "2026-10-12T14:00:00.000Z", "participant1Name": "Everton FC", "participant2Name": "Fulham FC"}]
    assert pc.match_fixtures(ours, theirs) == {1: "a", 2: "b"}


def test_client_caches_and_stops_before_the_request_budget(tmp_path):
    calls = []

    class Session:
        def get(self, url, params, timeout):
            calls.append((url, dict(params)))

            class R:
                status_code = 200
                headers = {"content-type": "application/json"}

                @staticmethod
                def json():
                    return [{"ok": len(calls)}]
            return R()

    client = pc.OddsPapiClient("KEY", tmp_path, budget=2, session=Session(), sleep=lambda s: None)
    first = client.get("/v4/fixtures", tournamentId=17)
    assert client.get("/v4/fixtures", tournamentId=17) == first and len(calls) == 1  # cached
    client.get("/v4/historical-odds", fixtureId="x", bookmakers="pinnacle")  # free: not counted
    client.get("/v4/fixtures", tournamentId=8)
    with pytest.raises(pc.BudgetExceeded):
        client.get("/v4/fixtures", tournamentId=23)
    assert json.loads((tmp_path / "usage.json").read_text())["billable"] == 2
    assert all("KEY" not in p.read_text() for p in tmp_path.glob("*.json"))
