"""The Research Assistant on the website (tracker D94): signed-in members only, 1 answered question a day on a free
account and 50 on a paid one, 2,000 for the whole site, behind ``SPIX_STATS_QUESTIONS``. The model is a stand-in: no
test makes a paid call."""
import json
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from Scripts.ops.stats_questions import log, model, questions
from Scripts.tests.test_stats_questions import conn  # noqa: F401  (the shared stats file fixture)
from web_app.routers import research_assistant as ra

FREE = {"id": 7, "tier": "free", "status": "active"}
NEXT_CARD = json.loads((Path(__file__).resolve().parent / "fixtures" / "next_match_card.json").read_text())
NEXT_FIXTURE = {"event_id": "1557458", "league": "EPL", "home_team": "Brighton", "away_team": "Brentford",
                "kickoff": "2026-11-08T14:00:00+00:00", "venue": "away", "opponent": "Brighton"}
PAID = {"id": 8, "tier": "pro", "status": "active"}


@pytest.fixture()
def site(conn, tmp_path, monkeypatch):
    monkeypatch.setenv("SPIX_STATS_QUESTIONS", "on")
    monkeypatch.setattr(ra, "STATS_PATH", tmp_path / "stats.db")
    monkeypatch.setattr(ra, "LOG_PATH", tmp_path / "log.db")
    asked = []

    def fake_ask(stats, question, *, previous=None):
        asked.append((question, previous))
        if question == "broken":
            raise model.ModelError("no_credit", 402, "The OpenAI account has no credit left.")
        if len(question) > 300:
            raise model.ModelError("too_long", 400, "Questions are limited to 300 characters.")
        for word, kind, params in (("Gabriel", "player", {"player": "Gabriel", "stat": "shots_on_target"}),
                                   ("Manchester", "team_average", {"team": "Manchester", "stat": "corners"}),
                                   ("corners", "team_average", {"team": "Brentford", "stat": "corners",
                                                                "venue": "away"})):
            if word in question:
                return {"question": question, "calls": [{"type": kind, "params": params}],
                        "answers": [questions.answer(stats, kind, params)], "model": "gpt-6-luna",
                        "usage": {"input_tokens": 2050, "cached_tokens": 0, "output_tokens": 40},
                        "cost_usd": 0.000225, "ms": 900, "flagged_bet": False}
        return {"question": question, "calls": [{"type": "recent_form", "params": {"team": "Brentford", "last": 3}}],
                "answers": [questions.answer(stats, "recent_form", {"team": "Brentford", "last": 3})],
                "model": "gpt-6-luna", "usage": {"input_tokens": 2050, "cached_tokens": 0, "output_tokens": 40},
                "cost_usd": 0.000225, "ms": 900, "flagged_bet": False}

    monkeypatch.setattr(ra, "ask_model", fake_ask)
    looked = []  # the team's next match (D95): the stored schedule and the match page's card, stood in for here

    def next_fixture(team_id):
        looked.append(team_id)
        return dict(NEXT_FIXTURE)

    monkeypatch.setattr(ra, "next_fixture", next_fixture)
    monkeypatch.setattr(ra, "next_read", lambda event_id: (NEXT_CARD, [{"market": {"group": "corners"},
                                                                       "projection": {"value": 9.46}}]))
    app = FastAPI()
    app.include_router(ra.router)
    who = {"user": FREE}
    app.dependency_overrides[ra.get_current_user] = lambda: who["user"]
    client = TestClient(app)
    client.who, client.asked, client.log_path, client.looked = who, asked, tmp_path / "log.db", looked
    return client


def ask(site, question="How are Brentford doing?", previous=None):
    return site.post("/api/research-assistant/ask", json={"question": question, "previous": previous})


def test_the_tab_stays_hidden_while_the_switch_is_off(site, monkeypatch):
    monkeypatch.setenv("SPIX_STATS_QUESTIONS", "")
    assert site.get("/api/research-assistant/status").json() == {"enabled": False}
    assert ask(site).status_code == 404
    assert site.post("/api/research-assistant/choose", json={"type": "recent_form", "params": {}}).status_code == 404
    assert site.asked == []


def test_visitors_see_the_tab_and_are_asked_to_sign_in(site):
    site.who["user"] = None
    assert site.get("/api/research-assistant/status").json() == {
        "enabled": True, "signed_in": False, "data_up_to": "Sun 8 Nov 2026"}
    response = ask(site)
    assert response.status_code == 401 and response.json()["detail"]["kind"] == "sign_in" and site.asked == []


def test_a_free_member_gets_one_answered_question_a_day(site):
    status = site.get("/api/research-assistant/status").json()
    assert (status["plan"], status["daily_limit"], status["remaining"]) == ("free", 1, 1)
    previous = {"question": "Brentford form", "calls": []}
    response = ask(site, previous=previous)
    body = response.json()
    assert response.status_code == 200 and body["answers"][0]["type"] == "recent_form" and body["remaining"] == 0
    assert not {"usage", "cost_usd", "model", "ms"} & set(body)  # what a question costs stays in the owner's log
    assert site.asked == [("How are Brentford doing?", previous)]
    again = ask(site)
    assert again.status_code == 429 and again.json()["detail"]["kind"] == "member_limit" and len(site.asked) == 1
    entry = log.recent(log.connect(site.log_path))[0]
    assert (entry["user_id"], entry["plan"], entry["source"], entry["cost_usd"]) == (7, "free", "website", 0.000225)


@pytest.mark.parametrize("user, limit", [(PAID, 50), ({"id": 9, "tier": "elite", "status": "trial"}, 50),
                                         ({"id": 9, "tier": "pro", "status": "cancelled"}, 1),
                                         ({"id": 9, "tier": "starter", "status": "past_due"}, 1)])
def test_paid_members_get_fifty_and_a_lapsed_subscription_counts_as_free(site, user, limit):
    site.who["user"] = user
    assert site.get("/api/research-assistant/status").json()["daily_limit"] == limit


def test_a_failed_question_says_unavailable_and_does_not_use_up_the_members_question(site):
    response = ask(site, "broken")
    detail = response.json()["detail"]
    assert response.status_code == 503 and detail["kind"] == "unavailable" and "credit" not in detail["message"]
    assert log.recent(log.connect(site.log_path))[0]["error"] == "no_credit"
    assert ask(site).status_code == 200


def test_a_too_long_question_is_refused_and_not_counted(site):
    response = ask(site, "x" * 301)
    assert response.status_code == 400 and response.json()["detail"]["kind"] == "too_long"
    assert site.get("/api/research-assistant/status").json()["remaining"] == 1


def test_the_site_stops_at_its_daily_ceiling(site, monkeypatch):
    monkeypatch.setattr(ra, "SITE_DAILY_LIMIT", 1)
    site.who["user"] = PAID
    assert ask(site).status_code == 200
    site.who["user"] = FREE
    response = ask(site)
    assert response.status_code == 429 and response.json()["detail"]["kind"] == "site_limit" and len(site.asked) == 1


def choose(site, kind, params):
    return site.post("/api/research-assistant/choose", json={"type": kind, "params": params})


def test_a_pick_answers_only_a_which_card_the_member_was_shown_today(site):
    site.who["user"] = PAID
    card = ask(site, "How often does Gabriel hit the target?").json()["answers"][0]
    assert card["type"] == "which"
    option = next(o for o in card["options"] if o["label"].startswith("Gabriel Magalhães"))
    pick = choose(site, "player", {"player": option["value"], "stat": "shots_on_target"})
    assert pick.status_code == 200 and pick.json()["answers"][0]["scope"].startswith("Gabriel Magalhães")
    assert len(site.asked) == 1 and site.get("/api/research-assistant/status").json()["remaining"] == 49
    refused = [
        {"player": "#10", "stat": "shots_on_target"},  # a real "#id", but not one the card offered
        {"player": option["value"], "stat": "goals"},  # the card's name with another question
        {"player": option["value"], "stat": "shots_on_target", "at_least": 2},
    ]
    for params in refused:
        response = choose(site, "player", params)
        assert response.status_code == 400 and response.json()["detail"]["kind"] == "not_a_pick", params
    assert choose(site, "team_average", {"team": "#3", "stat": "corners"}).status_code == 400
    site.who["user"] = FREE  # another member cannot use this member's card
    assert choose(site, "player", {"player": option["value"], "stat": "shots_on_target"}).status_code == 400
    site.who["user"] = None
    assert choose(site, "player", {"player": option["value"], "stat": "shots_on_target"}).status_code == 401


def test_a_free_member_cannot_get_answers_from_picks_alone(site):
    for _ in range(3):  # the orchestrator's check, 10 Nov
        assert choose(site, "team_average", {"team": "#42", "stat": "corners"}).status_code == 400


def test_an_oversized_previous_question_is_refused_before_the_model(site):
    previous = {"question": "Brentford form", "calls": [{"type": "recent_form", "params": {"pad": "y" * 200_000}}]}
    response = ask(site, previous=previous)
    assert response.status_code == 422 and site.asked == []


def test_without_a_stats_file_questions_are_unavailable(site, tmp_path, monkeypatch):
    monkeypatch.setattr(ra, "STATS_PATH", tmp_path / "missing.db")
    assert site.get("/api/research-assistant/status").json()["data_up_to"] is None
    response = ask(site)
    assert response.status_code == 503 and response.json()["detail"]["kind"] == "unavailable" and site.asked == []


def test_a_corners_answer_carries_the_teams_next_match_and_the_log_keeps_it(site):
    card = ask(site, "How many corners do Brentford get away?").json()["answers"][0]
    nxt = card["next_match"]
    assert site.looked == [1] and card["subject"]["team"] == "Brentford"
    assert (nxt["status"], nxt["label"], nxt["expected"]) == (
        "read", "Spix expects v Brighton, Sun 8 Nov", {"for": None, "against": None, "match": 9.5})
    assert (nxt["pick"]["label"], nxt["pick"]["typical_price"]) == ("Over 8.5 corners", 1.36)
    assert log.recent(log.connect(site.log_path))[0]["answers"][0]["next_match"] == nxt


def test_a_form_question_gets_no_next_match(site):
    card = ask(site).json()["answers"][0]
    assert card["type"] == "recent_form" and "next_match" not in card and site.looked == []


def test_the_next_match_never_breaks_an_answer(site, monkeypatch):
    def broken(team_id):
        raise RuntimeError("database is locked")
    monkeypatch.setattr(ra, "next_fixture", broken)
    response = ask(site, "How many corners do Brentford get away?")
    card = response.json()["answers"][0]
    assert response.status_code == 200 and card["type"] == "team_average" and "next_match" not in card


def test_a_pick_from_a_which_card_gets_the_next_match_too(site):
    site.who["user"] = PAID
    card = ask(site, "How many corners do Manchester get?").json()["answers"][0]
    assert card["type"] == "which" and "next_match" not in card
    option = next(o for o in card["options"] if o["label"] == "Manchester City")
    pick = choose(site, "team_average", {"team": option["value"], "stat": "corners"}).json()["answers"][0]
    assert pick["subject"]["team"] == "Manchester City" and pick["next_match"]["pick"]["label"] == "Over 8.5 corners"
    assert site.looked == [5]
