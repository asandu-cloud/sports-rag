"""The stats questions prototype's free SQL, question log and model step (tracker D92). The model is a stand-in here:
no test makes a paid call."""
import json
import sqlite3
from datetime import date, datetime, timezone
from types import SimpleNamespace

import httpx
import openai
import pytest

from Scripts.ops.stats_questions import free_sql, log, model, questions
from Scripts.tests.test_stats_questions import conn  # noqa: F401  (the shared stats file fixture)

TODAY = date(2026, 11, 10)


# -- free SQL -----------------------------------------------------------------------------------------------------

def test_free_sql_returns_the_rows_with_the_sql_it_ran(conn, tmp_path):
    out = free_sql.run(tmp_path / "stats.db", "SELECT name FROM teams WHERE league = 'EPL' ORDER BY name LIMIT 2")
    assert out["columns"] == ["name"] and out["rows"] == [["Arsenal"], ["Brentford"]]
    assert out["sql"].startswith("SELECT name") and out["truncated"] is False and "error" not in out


def test_free_sql_only_reads(conn, tmp_path):
    path = tmp_path / "stats.db"
    for sql in ("DELETE FROM matches", "UPDATE teams SET name = 'x'", "PRAGMA table_info(matches)",
                f"ATTACH DATABASE '{tmp_path / 'other.db'}' AS other", "CREATE TABLE t (x)", "DROP TABLE teams"):
        out = free_sql.run(path, sql)
        assert out["error"]["kind"] == "refused", sql
    assert free_sql.run(path, "SELECT 1; DELETE FROM matches")["error"]["kind"] == "sql"
    assert free_sql.run(path, "SELECT count(*) FROM matches")["rows"] == [[9]]
    assert not (tmp_path / "other.db").exists()


def test_free_sql_stops_at_the_row_and_time_limits(conn, tmp_path):
    path = tmp_path / "stats.db"
    many = "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c WHERE x < 600) SELECT x FROM c"
    out = free_sql.run(path, many)
    assert len(out["rows"]) == 500 and out["truncated"] is True
    endless = "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c) SELECT count(*) FROM c"
    assert free_sql.run(path, endless, seconds=0.05)["error"]["kind"] == "timeout"


# -- the question log ---------------------------------------------------------------------------------------------

def test_the_log_keeps_each_question_its_cost_and_the_owners_verdict(tmp_path):
    db = log.connect(tmp_path / "log.db")
    now = datetime(2026, 11, 10, 9, tzinfo=timezone.utc)
    first = log.record(db, mode="ask", question="How many corners do Brentford get away?",
                       calls=[{"type": "team_average", "params": {"team": "Brentford"}}], answers=[{"headline": "x"}],
                       model="gpt-6-luna", ms=812, usage={"input_tokens": 2000, "cached_tokens": 0,
                                                          "output_tokens": 90}, cost_usd=0.000245, now=now)
    log.record(db, mode="choose", calls=[{"type": "recent_form", "params": {}}], answers=[], now=now)
    log.record(db, mode="ask", question="yesterday", now=datetime(2026, 11, 9, 23, tzinfo=timezone.utc))
    assert log.asked_today(db, now=now) == 1  # model questions since midnight UTC; "Choose it yourself" is free
    assert log.set_verdict(db, first, "wrong", "should be away only") and not log.set_verdict(db, 999, "right")
    with pytest.raises(ValueError):
        log.set_verdict(db, first, "maybe")
    entry = next(e for e in log.recent(db) if e["id"] == first)
    assert entry["verdict"] == "wrong" and entry["note"] == "should be away only"
    assert entry["calls"][0]["type"] == "team_average" and entry["usage"]["output_tokens"] == 90
    assert entry["cost_usd"] == pytest.approx(0.000245) and entry["asked_at"] == "2026-11-10T09:00:00Z"


# -- the model step -----------------------------------------------------------------------------------------------

class FakeClient:
    """Stands in for OpenAI: returns the given tool calls and token counts, and keeps what it was sent."""

    def __init__(self, calls=(), *, usage=(2000, 1024, 100), error=None):
        self.calls, self.usage, self.error, self.sent = list(calls), usage, error, []
        self.responses = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.sent.append(kwargs)
        if self.error:
            raise self.error
        output = [SimpleNamespace(type="reasoning", summary=[])]
        output += [SimpleNamespace(type="function_call", name=name, arguments=json.dumps(args), call_id=f"c{i}")
                   for i, (name, args) in enumerate(self.calls)]
        tokens_in, cached, tokens_out = self.usage
        return SimpleNamespace(output=output, usage=SimpleNamespace(
            input_tokens=tokens_in, output_tokens=tokens_out,
            input_tokens_details=SimpleNamespace(cached_tokens=cached)))


def strict_args(kind, **given):
    return {name: given.get(name) for name in questions.TYPES[kind]["params"]}


def test_every_question_type_is_a_strict_tool_and_the_model_can_decline():
    tools = {t["name"]: t for t in model.tools()}
    assert set(tools) == set(questions.TYPES) | {"decline"}
    for tool in tools.values():
        params = tool["parameters"]
        assert tool["type"] == "function" and tool["strict"] is True
        assert params["required"] == list(params["properties"]) and params["additionalProperties"] is False
    assert tools["decline"]["parameters"]["properties"]["reason"]["enum"] == list(questions.DECLINE)


def test_the_model_only_chooses_and_the_code_counts(conn):
    fake = FakeClient([("team_average", strict_args("team_average", team="Brentford", stat="corners", side=None,
                                                     venue="away"))])
    out = model.ask(conn, "How many corners do Brentford get away?", client=fake, today=TODAY)
    assert out["calls"] == [{"type": "team_average",
                             "params": {"team": "Brentford", "stat": "corners", "venue": "away"}}]
    assert "4.0 corners won" in out["answers"][0]["headline"]  # side left empty: the team's own count
    sent = fake.sent[0]
    assert sent["model"] == "gpt-6-luna" and sent["tool_choice"] == "required" and sent["store"] is False
    assert "Tuesday 10 November 2026" in sent["instructions"] and "2026/27" in sent["instructions"]
    assert "EPL: Premier League" in sent["instructions"] and sent["input"].endswith(
        "How many corners do Brentford get away?")
    assert out["usage"] == {"input_tokens": 2000, "cached_tokens": 1024, "output_tokens": 100}
    assert out["cost_usd"] == pytest.approx((976 * 0.10 + 1024 * 0.01 + 100 * 0.50) / 1e6)


def test_a_bet_request_is_pointed_to_the_matches_page_and_other_questions_get_what_tab_can_do(conn):
    out = model.ask(conn, "What should I bet on tonight?", client=FakeClient([("decline", {"reason":
                                                                                           "asks_for_a_bet"})]),
                    today=TODAY)
    assert out["answers"][0]["type"] == "declined" and "matches page" in out["answers"][0]["headline"]
    assert out["flagged_bet"] is True and out["calls"] == [{"type": "decline", "params": {"reason": "asks_for_a_bet"}}]
    other = model.ask(conn, "Who will win the election?", client=FakeClient([("decline", {"reason":
                                                                                          "not_football_stats"})]),
                      today=TODAY)
    assert "recent form" in other["answers"][0]["headline"] and other["flagged_bet"] is False


def test_at_most_three_choices_are_answered(conn):
    call = ("recent_form", strict_args("recent_form", team="Brentford", venue="all", last=3))
    out = model.ask(conn, "Brentford form, four times", client=FakeClient([call] * 4), today=TODAY)
    assert len(out["answers"]) == 3


def test_a_follow_up_sees_the_previous_question_and_its_choices(conn):
    previous = {"question": "How many corners do Brentford get?",
                "calls": [{"type": "team_average", "params": {"team": "Brentford", "stat": "corners"}}]}
    fake = FakeClient([("team_average", strict_args("team_average", team="Brentford", stat="corners", side="for",
                                                     venue="home"))])
    model.ask(conn, "and at home?", previous=previous, client=fake, today=TODAY)
    sent = fake.sent[0]["input"]
    assert "Previous question: How many corners do Brentford get?" in sent and '"stat": "corners"' in sent
    assert sent.endswith("Question: and at home?")


def test_a_previous_question_carries_only_known_short_choices_into_the_prompt(conn):
    previous = {"question": "q" * 5000, "calls": [
        {"type": "team_average", "params": {"team": "Brentford", "stat": "corners", "pad": "y" * 200_000,
                                            "venue": {"nested": "x"}, "season": 2025}},
        {"type": "drop_table", "params": {"team": "x"}},
        {"type": "decline", "params": {"reason": "asks_for_a_bet", "extra": "z"}},
        {"type": "player", "params": {"player": "n" * 500, "stat": "goals"}}]}
    fake = FakeClient([("team_average", strict_args("team_average", team="Brentford", stat="corners"))])
    model.ask(conn, "and at home?", previous=previous, client=fake, today=TODAY)
    sent = fake.sent[0]["input"]
    assert "pad" not in sent and "nested" not in sent and "drop_table" not in sent and "extra" not in sent
    assert '"season": 2025' in sent and '"reason": "asks_for_a_bet"' in sent and "n" * 500 not in sent
    assert len(sent) < 1500

def test_long_or_empty_questions_and_a_missing_key_never_reach_the_model(conn, monkeypatch):
    fake = FakeClient()
    for text, kind in (("x" * 301, "too_long"), ("   ", "empty")):
        with pytest.raises(model.ModelError) as error:
            model.ask(conn, text, client=fake, today=TODAY)
        assert error.value.kind == kind and error.value.status == 400
    assert fake.sent == []
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(model.ModelError) as error:
        model.ask(conn, "Brentford corners", today=TODAY)
    assert (error.value.kind, error.value.status) == ("no_key", 503)


def test_openai_errors_become_a_plain_kind_without_their_raw_text(conn):
    response = httpx.Response(429, request=httpx.Request("POST", "https://api.openai.com/v1/responses"))
    empty = openai.RateLimitError("raw text with sk-test-123", response=response,
                                  body={"code": "insufficient_quota", "message": "raw"})
    with pytest.raises(model.ModelError) as error:
        model.ask(conn, "Brentford corners", client=FakeClient(error=empty), today=TODAY)
    assert (error.value.kind, error.value.status) == ("no_credit", 402) and "sk-test" not in error.value.message
    spent = openai.RateLimitError("raw", response=response,  # the shape OpenAI sent on 9 Nov (tracker)
                                  body={"type": "insufficient_quota", "code": "credit_balance_exhausted"})
    with pytest.raises(model.ModelError) as error:
        model.ask(conn, "Brentford corners", client=FakeClient(error=spent), today=TODAY)
    assert error.value.kind == "no_credit"
    busy = openai.RateLimitError("slow down", response=response, body={"code": "rate_limit_exceeded"})
    with pytest.raises(model.ModelError) as error:
        model.ask(conn, "Brentford corners", client=FakeClient(error=busy), today=TODAY)
    assert error.value.kind == "rate_limited"
    with pytest.raises(model.ModelError) as error:
        model.ask(conn, "Brentford corners", client=FakeClient([]), today=TODAY)
    assert error.value.kind == "bad_reply"


def test_free_sql_mode_asks_the_model_for_one_query_over_the_stats_file(conn):
    fake = FakeClient([("run_sql", {"sql": "SELECT name FROM teams LIMIT 1", "title": "One team"})])
    out = model.write_sql(conn, "Name one team", client=fake, today=TODAY)
    assert out["sql"] == "SELECT name FROM teams LIMIT 1" and out["title"] == "One team"
    assert [t["name"] for t in fake.sent[0]["tools"]] == ["run_sql"]
    assert "CREATE TABLE team_matches" in fake.sent[0]["instructions"] and out["cost_usd"] > 0


def test_the_log_counts_each_members_answered_questions_and_keeps_older_logs(tmp_path):
    path = tmp_path / "log.db"
    old = sqlite3.connect(path)  # a log from before members (10 Nov), without the member columns
    old.execute("CREATE TABLE questions (id INTEGER PRIMARY KEY, asked_at TEXT NOT NULL, mode TEXT NOT NULL, "
                "question TEXT, calls TEXT, answers TEXT, model TEXT, ms INTEGER, usage TEXT, cost_usd REAL, "
                "error TEXT, flagged_bet INTEGER, verdict TEXT, note TEXT)")
    old.execute("INSERT INTO questions (asked_at, mode, question) VALUES ('2026-11-10T08:00:00Z', 'ask', 'local')")
    old.commit(); old.close()
    db = log.connect(path)
    now = datetime(2026, 11, 10, 9, tzinfo=timezone.utc)
    log.record(db, mode="ask", question="a", user_id=7, plan="free", source="website", now=now)
    log.record(db, mode="ask", question="b", user_id=7, plan="free", source="website", error="unavailable", now=now)
    log.record(db, mode="ask", question="c", user_id=8, plan="paid", source="website", now=now)
    log.record(db, mode="choose", user_id=7, source="website", now=now)
    assert log.asked_today(db, user_id=7, now=now) == 1  # a failed question and a pick do not count against them
    assert log.asked_today(db, now=now) == 4  # the site's ceiling counts every model question, the old one too
    entry = log.recent(db)[0]
    assert (entry["user_id"], entry["plan"], entry["source"]) == (7, None, "website")
    assert log.recent(db)[-1]["source"] is None
