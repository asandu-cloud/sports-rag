"""Bookmaker coverage watch: a bookmaker dropping out of API-Football's odds raises an alert."""
from datetime import datetime, timedelta, timezone
import json

from Scripts.ops import bookmaker_coverage as bc

NOW = datetime(2026, 10, 17, 9, tzinfo=timezone.utc)


def payload(*fixtures):
    """fixtures: (fixture id, hours to kickoff, [bookmaker names]) -> an API-Football /odds body."""
    return {"get": "odds", "response": [
        {"fixture": {"id": fid, "date": (NOW + timedelta(hours=hours)).isoformat()},
         "bookmakers": [{"id": i, "name": name, "bets": [{"id": 5, "values": [{"value": "Over 2.5", "odd": "1.9"}]}]}
                        for i, name in enumerate(books)]}
        for fid, hours, books in fixtures]}


def test_each_day_counts_fixtures_kicking_off_within_two_days_once_per_bookmaker():
    captures = [(NOW, payload((1, 5, ["Betano", "Pinnacle"]), (2, 30, ["Pinnacle"]), (3, 60, ["Betano"]))),
                (NOW + timedelta(hours=3), payload((1, 5, ["Betano"]), (4, -1, ["Betano"])))]
    days = bc.day_counts(captures)
    assert days == {"2026-10-17": {"fixtures": 2, "books": {"Betano": 1, "Pinnacle": 2}}}
    # fixture 3 is 60 h away, fixture 4 had kicked off, fixture 1 is counted once across both captures


def test_a_bookmaker_without_prices_on_a_fixture_is_not_counted():
    body = payload((1, 5, ["Betano"]))
    body["response"][0]["bookmakers"].append({"id": 9, "name": "SBO", "bets": []})
    assert bc.day_counts([(NOW, body)])["2026-10-17"]["books"] == {"Betano": 1}


def history(shares, today, fixtures=100):
    """{day: counts} for the days before ``today`` with a bookmaker's share, then today's."""
    days = {}
    for back, share in enumerate(reversed(shares), start=1):
        day = (datetime.fromisoformat(today) - timedelta(days=back)).date().isoformat()
        days[day] = {"fixtures": fixtures, "books": {"Betano": round(share * fixtures), "Pinnacle": fixtures}}
    return days


def test_a_share_under_half_the_previous_weeks_median_is_an_alert():
    days = history([0.95, 0.96, 0.94, 0.95, 0.97, 0.95, 0.96], "2026-10-17")
    days["2026-10-17"] = {"fixtures": 120, "books": {"Betano": 3, "Pinnacle": 118}}
    [alert] = bc.alerts(days, "2026-10-17")
    assert (alert["bookmaker"], alert["fixtures_today"], alert["priced_today"]) == ("Betano", 120, 3)
    assert alert["usual_share"] == 0.95 and round(alert["share_today"], 3) == 0.025


def test_no_alert_for_a_normal_dip_a_rare_bookmaker_a_thin_day_or_no_history():
    days = history([0.95] * 7, "2026-10-17")
    days["2026-10-17"] = {"fixtures": 120, "books": {"Betano": 70, "Pinnacle": 120}}  # 58%: above half of 95%
    assert bc.alerts(days, "2026-10-17") == []
    rare = history([0.15] * 7, "2026-10-17")
    rare["2026-10-17"] = {"fixtures": 120, "books": {"Pinnacle": 120}}  # usually under 20%: not watched
    assert bc.alerts(rare, "2026-10-17") == []
    thin = history([0.95] * 7, "2026-10-17")
    thin["2026-10-17"] = {"fixtures": 12, "books": {"Pinnacle": 12}}  # too few fixtures so far today
    assert bc.alerts(thin, "2026-10-17") == []
    short = history([0.95] * 2, "2026-10-17")
    short["2026-10-17"] = {"fixtures": 120, "books": {"Pinnacle": 120}}  # under 3 earlier days
    assert bc.alerts(short, "2026-10-17") == []


def test_a_bookmaker_missing_today_altogether_is_an_alert():
    days = history([0.9] * 7, "2026-10-17")
    days["2026-10-17"] = {"fixtures": 80, "books": {"Pinnacle": 80}}
    assert [a["bookmaker"] for a in bc.alerts(days, "2026-10-17")] == ["Betano"]


def test_each_bookmaker_is_alerted_once_a_day():
    found = [{"bookmaker": "Betano"}, {"bookmaker": "SBO"}]
    assert [a["bookmaker"] for a in bc.unsent(found, {"2026-10-17": ["Betano"]}, "2026-10-17")] == ["SBO"]
    assert len(bc.unsent(found, {"2026-10-16": ["Betano"]}, "2026-10-17")) == 2


def test_the_message_names_the_bookmaker_and_its_coverage():
    text = bc.message([{"bookmaker": "Betano", "share_today": 0.025, "usual_share": 0.95, "fixtures_today": 120,
                        "priced_today": 3}], "2026-10-17")
    assert "Betano" in text and "3 of 120" in text and "95%" in text


def archive(path, days_with_betano, today_books):
    """A minimal odds archive: one capture a day, 30 fixtures each, for the 7 days before NOW and NOW itself."""
    import gzip
    import json
    import sqlite3
    db = sqlite3.connect(path)
    db.execute("CREATE TABLE requests(id TEXT, received_at TEXT, http_status INTEGER, raw_sha256 TEXT)")
    db.execute("CREATE TABLE raw_bodies(sha256 TEXT PRIMARY KEY, body_gzip BLOB)")
    for back in range(7, -1, -1):
        at = NOW - timedelta(days=back)
        books = today_books if back == 0 else (["Betano", "Pinnacle"] if back in days_with_betano else ["Pinnacle"])
        body = {"response": [{"fixture": {"id": back * 100 + i, "date": (at + timedelta(hours=5)).isoformat()},
                              "bookmakers": [{"name": b, "bets": [{"values": [{"odd": "2"}]}]} for b in books]}
                             for i in range(30)]}
        db.execute("INSERT INTO raw_bodies VALUES(?, ?)", (f"s{back}", gzip.compress(json.dumps(body).encode())))
        db.execute("INSERT INTO requests VALUES(?, ?, 200, ?)", (f"r{back}", at.isoformat(), f"s{back}"))
    db.commit()
    db.close()


def test_a_run_alerts_once_and_caches_the_complete_days(tmp_path):
    archive(tmp_path / "odds.sqlite3", days_with_betano=set(range(1, 8)), today_books=["Pinnacle"])
    sent = []
    report = bc.run(now=NOW, archive=tmp_path / "odds.sqlite3", out=tmp_path / "out",
                    send=lambda m: sent.append(m) or True)
    assert [a["bookmaker"] for a in report["new_alerts"]] == ["Betano"] and report["webhook_delivered"] is True
    assert len(sent) == 1 and "Betano" in sent[0]
    again = bc.run(now=NOW + timedelta(hours=1), archive=tmp_path / "odds.sqlite3", out=tmp_path / "out",
                   send=lambda m: sent.append(m) or True)
    assert again["alerts"] and again["new_alerts"] == [] and len(sent) == 1  # once a day
    cached = json.loads((tmp_path / "out" / "days.json").read_text())
    assert "2026-10-16" in cached and "2026-10-17" not in cached  # today is never cached


def test_no_alert_while_every_bookmaker_keeps_its_coverage(tmp_path):
    archive(tmp_path / "odds.sqlite3", days_with_betano=set(range(1, 8)), today_books=["Betano", "Pinnacle"])
    report = bc.run(now=NOW, archive=tmp_path / "odds.sqlite3", out=tmp_path / "out", send=lambda m: 1 / 0)
    assert report["alerts"] == [] and report["webhook_delivered"] is None


def test_a_failed_delivery_is_retried_at_the_next_run(tmp_path):
    archive(tmp_path / "odds.sqlite3", days_with_betano=set(range(1, 8)), today_books=["Pinnacle"])
    tries = []
    first = bc.run(now=NOW, archive=tmp_path / "odds.sqlite3", out=tmp_path / "out",
                   send=lambda m: tries.append(m) and False)
    assert first["webhook_delivered"] is False
    second = bc.run(now=NOW + timedelta(hours=4), archive=tmp_path / "odds.sqlite3", out=tmp_path / "out",
                    send=lambda m: tries.append(m) or True)
    assert [a["bookmaker"] for a in second["new_alerts"]] == ["Betano"] and second["webhook_delivered"] is True
    assert len(tries) == 2
