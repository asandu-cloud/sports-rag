"""A5.3 price-age check (D44): are prices seen recently still on offer at the bookmaker's next check more often?"""
from datetime import datetime, timedelta, timezone

import pytest

from Scripts.ops import confidence_age_study as age

READ = datetime(2026, 10, 18, 14, 0, tzinfo=timezone.utc)
KICKOFF = READ + timedelta(hours=1)


def at(minutes):
    return READ + timedelta(minutes=minutes)


def test_age_is_minutes_since_the_newest_record():
    assert age.seen_age_minutes([at(-50), at(-12), at(5)], READ) == 12.0


def test_no_record_before_the_read_is_unknown():
    assert age.seen_age_minutes([at(3), at(9)], READ) is None


def test_the_verdict_reads_the_first_check_after_the_read():
    points = [(at(-30), 2.10, True), (at(20), 1.95, True)]  # shortened at the second check after the read
    moments = [at(-30), at(8), at(20)]
    assert age.next_check_verdict(points, moments, READ, KICKOFF, 2.10) == "still there"  # 8 min: unchanged


def test_a_price_gone_or_suspended_at_the_next_check_is_gone():
    moments = [at(-30), at(8)]
    shorter = [(at(-30), 2.10, True), (at(8), 2.00, True)]
    suspended = [(at(-30), 2.10, True), (at(8), None, False)]
    assert age.next_check_verdict(shorter, moments, READ, KICKOFF, 2.10) == "gone"
    assert age.next_check_verdict(suspended, moments, READ, KICKOFF, 2.10) == "gone"


def test_no_check_before_kickoff():
    points = [(at(-30), 2.10, True)]
    assert age.next_check_verdict(points, [at(-30), at(70)], READ, KICKOFF, 2.10) == "no check before kickoff"


def test_fresh_prices_still_there_more_often_show_a_positive_difference():
    def record(fid, verdict):
        return {"fixture_id": str(fid), "verdict": verdict, "value_at_close": 0.01}
    fresh = [record(i, "still there" if i % 10 else "gone") for i in range(80)]           # 90% still there
    older = [record(100 + i, "still there" if i % 2 else "gone") for i in range(80)]       # 50%
    unchecked = [record(300, "no check before kickoff")]
    result = age.still_there_comparison(fresh + unchecked, older, resamples=300)
    assert result["fresh"]["still_there_share"] == pytest.approx(0.9) and result["older"]["still_there_share"] == 0.5
    assert result["fresh"]["no_check"] == 1 and result["diff"] == pytest.approx(0.4)
    assert result["diff_interval95"][0] > 0


class Lists:
    """A client whose cached fixture lists are ``cached`` [(tournament, from, to)]; anything else is billable."""
    def __init__(self, cached, listing):
        self.cached_lists, self.listing, self.asked = cached, listing, []

    def get(self, path, **params):
        key = (params["tournamentId"], params["from"], params["to"])
        self.asked.append(key)
        if key not in self.cached_lists:
            from Scripts.data_platform.oddspapi import BudgetExceeded
            raise BudgetExceeded("not cached")
        return self.listing


def ours(fid, league, kickoff):
    return {"fixture_id": fid, "kickoff": kickoff, "league": league, "home": "Arsenal", "away": "Leeds"}


def theirs(odds_id, kickoff):
    return {"fixtureId": odds_id, "startTime": kickoff.isoformat().replace("+00:00", "Z"),
            "participant1Name": "Arsenal", "participant2Name": "Leeds"}


def test_fixture_lists_come_from_the_true_closes_weekly_windows_first():
    from Scripts.data_platform import oddspapi
    from Scripts.data_platform.services.true_close import week_window
    kickoff = datetime(2026, 10, 24, 14, tzinfo=timezone.utc)
    weekly = (oddspapi.TOURNAMENTS["EPL"], *week_window(kickoff))
    client = Lists({weekly}, [theirs("op-1", kickoff)])
    mapping, skipped = age._fixture_map(client, [ours(7, "EPL", kickoff)])
    assert mapping == {"7": "op-1"} and skipped == [] and client.asked == [weekly]


def test_a_week_without_any_cached_list_is_reported_not_bought():
    kickoff = datetime(2026, 10, 24, 14, tzinfo=timezone.utc)
    client = Lists(set(), [])
    mapping, skipped = age._fixture_map(client, [ours(7, "EPL", kickoff)])
    assert mapping == {} and len(skipped) == 1 and skipped[0].startswith("EPL")


def test_by_default_only_downloaded_histories_are_read():
    class Client:
        def __init__(self):
            self.requests = []

        def cached(self, path, **params):
            return params["bookmakers"] == "pinnacle"

        def get(self, path, **params):
            self.requests.append(params["bookmakers"])
            return {"bookmakers": {params["bookmakers"]: {"markets": {}}}}
    client = Client()
    assert age._history(client, "op-1", "betano", cached_only=True) is None
    assert age._history(client, "op-1", "pinnacle", cached_only=True) == {"bookmakers": {"pinnacle": {"markets": {}}}}
    assert client.requests == ["pinnacle"]
    age._history(client, "op-1", "betano", cached_only=False)
    assert client.requests == ["pinnacle", "betano"]
