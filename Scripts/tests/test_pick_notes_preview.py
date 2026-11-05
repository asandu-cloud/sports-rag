"""The pick-notes preview (D75) writes notes for the cards the board serves, and only for cards with picks."""
from Scripts.ops import pick_notes_preview as preview
from Scripts.tests.data_platform.test_pick_notes import ARSENAL, BRIGHTON, block


def card(event_id, recommendations):
    return {"fixture": {"event_id": event_id, "league": "EPL", "home_team": "Arsenal", "away_team": "Brighton",
                        "kickoff": "2026-11-08T15:00:00Z"}, "recommendations": recommendations}


def test_each_card_with_picks_gets_its_facts_and_notes():
    asked = []

    def history(event_id):
        asked.append(event_id)
        return {"competition": "Premier League", "home": ARSENAL, "away": BRIGHTON}

    cards = [card("1", block()), card("2", None), card("3", {"most_likely": [], "value": []})]
    [entry] = preview.notes_for_cards(cards, history, generator=lambda _: {})
    assert asked == ["1"]
    assert entry["fixture"]["event_id"] == "1" and entry["facts"]["competition"] == "Premier League"
    assert set(entry["notes"]["notes"]) == {"most_likely-0", "value-0"}
    assert {n["source"] for n in entry["notes"]["notes"].values()} == {"fixed"}


def test_a_card_whose_fixture_has_no_history_still_gets_notes_without_form():
    [entry] = preview.notes_for_cards([card("9", block())], lambda _: None, generator=lambda _: {})
    assert entry["facts"]["competition"] is None
    assert all(p["form"] == [] for p in entry["facts"]["picks"])


def test_the_markdown_shows_each_pick_with_its_two_lines():
    [entry] = preview.notes_for_cards([card("1", block())],
                                      lambda _: {"competition": "Premier League", "home": ARSENAL, "away": BRIGHTON},
                                      generator=lambda _: {})
    text = preview.markdown([entry], "2026-11-08")
    assert "## Arsenal v Brighton (EPL, 15:00 UTC)" in text
    assert "**Over 2.5 goals** (Most likely, 72%)" in text and "**Arsenal to win** (Value, 55%)" in text
    assert "- For: At 72%, this is the likeliest option in its market that still pays at least 1.30." in text
    assert "- Against: 3 of Brighton's last 5 Premier League matches had 2 goals or fewer." in text
