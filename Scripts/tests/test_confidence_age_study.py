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
