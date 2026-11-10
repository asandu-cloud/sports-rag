"""The Research Assistant on the website (tracker D94): signed-in members only, 1 answered question a day on a free
account and 50 on a paid one, 2,000 for the whole site, behind ``SPIX_STATS_QUESTIONS``. The model is a stand-in: no
test makes a paid call."""
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from Scripts.ops.stats_questions import log, model, questions
from Scripts.tests.test_stats_questions import conn  # noqa: F401  (the shared stats file fixture)
from web_app.routers import research_assistant as ra

FREE = {"id": 7, "tier": "free", "status": "active"}
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
        return {"question": question, "calls": [{"type": "recent_form", "params": {"team": "Brentford", "last": 3}}],
                "answers": [questions.answer(stats, "recent_form", {"team": "Brentford", "last": 3})],
                "model": "gpt-6-luna", "usage": {"input_tokens": 2050, "cached_tokens": 0, "output_tokens": 40},
                "cost_usd": 0.000225, "ms": 900, "flagged_bet": False}

    monkeypatch.setattr(ra, "ask_model", fake_ask)
    app = FastAPI()
    app.include_router(ra.router)
    who = {"user": FREE}
    app.dependency_overrides[ra.get_current_user] = lambda: who["user"]
    client = TestClient(app)
    client.who, client.asked, client.log_path = who, asked, tmp_path / "log.db"
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


def test_a_pick_from_a_which_card_runs_without_the_model_and_other_choices_are_refused(site):
    pick = site.post("/api/research-assistant/choose", json={
        "type": "player", "params": {"player": "#10", "stat": "shots_on_target"}})
    assert pick.status_code == 200 and pick.json()["answers"][0]["scope"].startswith("Bukayo Saka")
    other = site.post("/api/research-assistant/choose", json={
        "type": "player", "params": {"player": "Saka", "stat": "shots_on_target"}})
    assert other.status_code == 400 and other.json()["detail"]["kind"] == "not_a_pick"
    assert site.asked == [] and site.get("/api/research-assistant/status").json()["remaining"] == 1
    site.who["user"] = None
    assert site.post("/api/research-assistant/choose", json={
        "type": "player", "params": {"player": "#10", "stat": "goals"}}).status_code == 401


def test_without_a_stats_file_questions_are_unavailable(site, tmp_path, monkeypatch):
    monkeypatch.setattr(ra, "STATS_PATH", tmp_path / "missing.db")
    assert site.get("/api/research-assistant/status").json()["data_up_to"] is None
    response = ask(site)
    assert response.status_code == 503 and response.json()["detail"]["kind"] == "unavailable" and site.asked == []
