"""The team's next match under a Research Assistant answer (tracker D95), from the card the match page shows."""
import copy
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from data_platform.services import next_match

CARD = json.loads((Path(__file__).resolve().parents[1] / "fixtures" / "next_match_card.json").read_text())
CORNERS = [{"market": {"group": "corners"}, "projection": {"value": 9.46}}]
BRIGHTON_BRENTFORD = {"event_id": "1557458", "league": "EPL", "home_team": "Brighton", "away_team": "Brentford",
                      "kickoff": "2026-11-08T14:00:00+00:00", "venue": "away", "opponent": "Brighton"}
BRENTFORD_EVERTON = {"event_id": "1557469", "league": "EPL", "home_team": "Brentford", "away_team": "Everton",
                     "kickoff": "2026-11-23T20:00:00+00:00", "venue": "home", "opponent": "Everton"}
BEFORE = datetime(2026, 11, 6, 12, tzinfo=timezone.utc)


def entry(fixture, read, stat, now=BEFORE):
    return next_match.entry(fixture, read, stat=stat, now=now, outlook_hours=96)


def test_a_corners_answer_gets_the_corners_spix_expects_in_the_match_and_its_pick():
    found = entry(BRIGHTON_BRENTFORD, (CARD, CORNERS), "corners")
    pick = found.pop("pick")
    assert found == {
        "fixture": {"event_id": "1557458", "league": "EPL", "home_team": "Brighton", "away_team": "Brentford",
                    "kickoff": "2026-11-08T14:00:00+00:00"},
        "venue": "away", "opponent": "Brighton", "market": "corners", "status": "read",
        "label": "Spix expects v Brighton, Sun 8 Nov",
        "expected": {"for": None, "against": None, "match": 9.5},  # Spix prices the whole match's corners only
        "note": None, "prices_as_of": "2026-11-08T13:56:26"}
    assert (pick["label"], pick["side"], pick["line"], pick["typical_price"]) == ("Over 8.5 corners", "over", 8.5, 1.36)
    assert pick["chance"] == pytest.approx(0.704, abs=5e-4)


def test_a_goals_answer_splits_the_goals_spix_expects_by_the_asked_teams_side():
    away = entry(BRIGHTON_BRENTFORD, (CARD, CORNERS), "goals")
    assert away["expected"] == {"for": 1.3, "against": 1.9, "match": 3.1}  # Brentford away: 1.28 v 1.86
    assert away["pick"]["label"] == "Over 2.5 goals"
    home = entry({**BRIGHTON_BRENTFORD, "venue": "home", "opponent": "Brentford"}, (CARD, CORNERS), "goals")
    assert home["expected"] == {"for": 1.9, "against": 1.3, "match": 3.1}
    assert home["label"] == "Spix expects v Brentford, Sun 8 Nov"


def test_a_market_without_a_pick_says_so_and_keeps_the_figures():
    card = copy.deepcopy(CARD)
    next(m for m in card["recommendations"]["markets"] if m["market"] == "corners")["suggestion"] = None
    found = entry(BRIGHTON_BRENTFORD, (card, CORNERS), "corners")
    assert found["pick"] is None and found["expected"]["match"] == 9.5
    assert found["note"] == "No pick from Spix in corners for that match."


def test_before_the_read_exists_it_says_when_the_read_comes_out():
    waiting = entry(BRENTFORD_EVERTON, None, "corners", now=datetime(2026, 11, 10, 12, tzinfo=timezone.utc))
    assert (waiting["status"], waiting["label"], waiting["expected"], waiting["pick"]) == ("waiting", None, None, None)
    assert waiting["note"] == "Spix's read of Brentford v Everton (Mon 23 Nov) comes out from Thu 19 Nov."
    late = entry(BRENTFORD_EVERTON, None, "corners", now=datetime(2026, 11, 21, 12, tzinfo=timezone.utc))
    assert late["note"] == "Spix's read of Brentford v Everton (Mon 23 Nov) isn't out yet."


def test_a_card_without_the_recommendations_block_adds_nothing():
    card = {**CARD, "recommendations": None}  # the display switch is off
    assert entry(BRIGHTON_BRENTFORD, (card, CORNERS), "corners") is None


def test_only_corners_and_goals_look_for_the_next_match():
    asked = []
    def next_fixture(team_id):
        asked.append(team_id)
        return BRIGHTON_BRENTFORD
    def load_read(event_id):
        asked.append(event_id)
        return CARD, CORNERS
    for stat in ("cards", "shots_on_target", "fouls", "possession"):  # cards and shots on target stay out (D61)
        assert next_match.find(1, stat, next_fixture=next_fixture, load_read=load_read, now=BEFORE,
                               outlook_hours=96) is None
    assert asked == []
    found = next_match.find(1, "corners", next_fixture=next_fixture, load_read=load_read, now=BEFORE, outlook_hours=96)
    assert asked == [1, "1557458"] and found["pick"]["label"] == "Over 8.5 corners"
    assert next_match.find(1, "goals", next_fixture=lambda team_id: None, load_read=load_read, now=BEFORE,
                           outlook_hours=96) is None
