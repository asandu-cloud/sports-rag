"""Pick notes on the website's cards (D75): each pick gets the fixed wording's two lines, the teams' matches are read
once an hour per fixture, and a failed read leaves the card as it was."""
from datetime import datetime, timedelta, timezone

from Scripts.tests.data_platform.test_pick_notes import ARSENAL, BRIGHTON, block

NOW = datetime(2026, 11, 8, 12, tzinfo=timezone.utc)


def card(event_id="1001", recommendations="default"):
    return {"fixture": {"event_id": event_id, "league": "EPL", "home_team": "Arsenal", "away_team": "Brighton",
                        "kickoff": "2026-11-08T15:00:00Z"},
            "recommendations": block() if recommendations == "default" else recommendations}


def counting(asked):
    def lookup(api_id):
        asked.append(api_id)
        return {"competition": "Premier League", "home": ARSENAL, "away": BRIGHTON}
    return lookup


def notes(target):
    return [item.get("note") for kind in ("most_likely", "value") for item in target["recommendations"][kind]]


def test_each_pick_gets_its_line_for_and_its_line_against():
    from data_platform.services.pick_note_service import PickNotes

    target = card()
    PickNotes(counting([])).attach(target, now=NOW)
    over, win = notes(target)
    assert over == {"for": "3 of Arsenal's last 5 Premier League matches had 3 or more goals.",
                    "against": "3 of Brighton's last 5 Premier League matches had 2 goals or fewer."}
    assert win["for"] == "Arsenal won 3 of their last 5 Premier League matches."
    assert win["against"] == "Brighton avoided defeat in 3 of their last 5 Premier League matches."


def test_a_fixtures_matches_are_read_once_an_hour():
    from data_platform.services.pick_note_service import PickNotes

    asked = []
    service = PickNotes(counting(asked))
    for moment in (NOW, NOW + timedelta(minutes=59), NOW + timedelta(minutes=61)):
        service.attach(card(), now=moment)
    service.attach(card("1002"), now=NOW + timedelta(minutes=61))
    assert asked == [1001, 1001, 1002]


def test_a_failed_read_leaves_the_card_as_it_was_and_is_tried_again():
    from data_platform.services.pick_note_service import PickNotes

    calls = []

    def flaky(api_id):
        calls.append(api_id)
        if len(calls) == 1:
            raise RuntimeError("database is locked")
        return {"competition": "Premier League", "home": ARSENAL, "away": BRIGHTON}

    service, first, second = PickNotes(flaky), card(), card()
    service.attach(first, now=NOW)
    assert first == card()
    service.attach(second, now=NOW + timedelta(seconds=5))
    assert all(note for note in notes(second))


def test_a_fixture_with_no_matches_found_still_gets_notes_from_the_chance():
    from data_platform.services.pick_note_service import PickNotes

    for event_id, lookup in (("1001", lambda _: None), ("fixture-x", None)):
        target = card(event_id)
        PickNotes(lookup).attach(target, now=NOW)
        assert notes(target)[0] == {"for": "The likeliest option in its market that still pays at least 1.30.",
                                    "against": "It still loses about 3 times in 10."}


def test_a_card_without_picks_is_left_alone():
    from data_platform.services.pick_note_service import PickNotes

    asked = []
    service = PickNotes(counting(asked))
    for recommendations in (None, {"most_likely": [], "value": []}):
        target = card(recommendations=recommendations)
        service.attach(target, now=NOW)
        assert target == card(recommendations=recommendations)
    assert asked == []
