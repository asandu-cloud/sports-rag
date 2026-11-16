"""One match's research view (D110, the overhaul's B2): the match, Spix's read and its lines."""
from datetime import datetime, timezone

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from data_platform import recommendations
from data_platform.models import Competition, Fixture, Season, Team
from data_platform.repositories.research import ResearchRepository
from data_platform.services.research_match import ResearchMatchService, read_block, total_lines

NOW = datetime(2026, 11, 16, 12, 0, tzinfo=timezone.utc)
KEYS = {"goals": "totals", "corners": "totals_corners_over_under", "cards": "totals_cards_over_under"}
FIXTURE = {"home_team": "Chelsea", "away_team": "Leeds"}


def totals(group, prices, value):
    """A canonical result whose decision audit holds ``prices``: {line: [(over, under) per bookmaker]}."""
    options = [{"market_key": KEYS[group], "side": side, "point": line, "odds": odds, "bookmaker": f"book{b}"}
               for line, books in prices.items() for b, pair in enumerate(books)
               for side, odds in zip(("over", "under"), pair)]
    return {"market": {"group": group, "key": KEYS[group]}, "fixture": FIXTURE, "projection": {"value": value},
            "context": {"decision_audit": {"input_options": options, "candidates": []}}}


def canonical(cards=True):
    out = [totals("goals", {1.5: [(1.30, 3.60), (1.28, 3.75)], 2.5: [(1.95, 1.90), (2.0, 1.85)],
                            3.0: [(2.5, 1.55), (2.45, 1.57)]}, 2.66),
           totals("corners", {9.5: [(1.85, 1.95), (1.9, 1.9)], 10.5: [(2.3, 1.6), (2.25, 1.62)]}, 10.14)]
    if cards:
        out.append(totals("cards", {3.5: [(1.6, 2.3), (1.62, 2.25)], 4.0: [(1.9, 1.9), (1.92, 1.88)],
                                    4.5: [(2.2, 1.65), (2.25, 1.62)], 5.5: [(3.4, 1.3)]}, 4.62))
    return out


def card(block=True):
    return {"timing": {"is_preliminary": False}, "created_at": "2026-11-20T09:00:00+00:00",
            "provenance": {"evaluated_at": "2026-11-20T08:58:00+00:00"},
            "metrics": {"projected_total_goals": 2.66, "team_goals": {"home": 1.62, "away": 1.04},
                        "result_probabilities": {"home": 0.58, "draw": 0.24, "away": 0.18}},
            "recommendations": {"version": "recommendations.v1", "markets": [
                {"market": "goals", "suggestion": {"label": "Over 1.5 goals"}, "options": []},
                {"market": "corners", "suggestion": None, "options": []}]} if block else None}


def metrics_results():
    """The figures match_metrics reads: market goal figures on goals and btts, the anchored 1X2."""
    figures = {"total_goals": 2.66, "home_goals": 1.62, "away_goals": 1.04, "btts_yes": 0.5432}
    out = canonical()
    out[0]["context"]["market_anchor"] = {"goal_figures": figures}
    out.append({"market": {"group": "btts"}, "context": {"market_anchor": {"goal_figures": figures}}})
    out.append({"market": {"group": "moneyline"},
                "context": {"market_anchor": {"market": {"home": 0.58, "draw": 0.24, "away": 0.18}}}})
    return out


def test_lines_pair_over_and_under_per_half_line_with_chance_and_typical_price():
    lines = total_lines(canonical(), "goals")
    assert [line["line"] for line in lines] == [1.5, 2.5]  # whole lines are left out, as in the picks
    options = recommendations.options(canonical())["goals"]
    for line in lines:
        for side in ("over", "under"):
            option = next(o for o in options if o["line"] == line["line"] and o["side"] == side)
            assert line[side] == {"chance": option["chance"], "typical_price": option["typical_price"]}
    assert lines[1]["over"]["typical_price"] == 1.98  # the median of each bookmaker's best price
    assert lines[1]["over"]["chance"] + lines[1]["under"]["chance"] == pytest.approx(1)


def test_cards_lines_from_the_cards_result_without_a_suggestion():
    lines = total_lines(canonical(), "cards")
    assert [line["line"] for line in lines] == [3.5, 4.5]  # 4.0 is whole; 5.5 has one bookmaker
    assert lines[0]["over"]["chance"] > lines[1]["over"]["chance"]
    assert total_lines(canonical(cards=False), "cards") == []
    block = read_block(card(), canonical(), status="ready")
    assert block["suggestions"]["cards"] is None
    assert block["suggestions"]["goals"] == {"label": "Over 1.5 goals"}
    assert block["suggestions"]["corners"] is None


def test_read_block_figures_round_to_one_decimal_and_keep_chances():
    block = read_block(card(), metrics_results(), status="preliminary")
    assert block["status"] == "preliminary"
    assert block["read_at"] == "2026-11-20T09:00:00+00:00"
    assert block["prices_as_of"] == "2026-11-20T08:58:00+00:00"
    assert block["result"] == {"home": 0.58, "draw": 0.24, "away": 0.18}
    assert block["goals"] == {"home": 1.6, "away": 1.0, "total": 2.7}
    assert block["btts_yes"] == 0.5432
    assert block["corners_expected"] == 10.1 and block["cards_expected"] == 4.6
    assert set(block["lines"]) == {"cards", "goals", "corners"}
    assert [m["market"] for m in block["markets"]] == ["goals", "corners"]
    assert read_block(card(block=False), metrics_results(), status="ready")["markets"] is None
    bare = read_block(card(), [], status="ready")
    assert bare["result"] is None and bare["goals"] is None and bare["btts_yes"] is None
    assert bare["cards_expected"] is None and bare["lines"] == {"cards": [], "goals": [], "corners": []}


@pytest.fixture()
def match(session_factory):
    with session_factory() as s:
        epl = Competition(code="EPL", name="Premier League", api_football_id=39)
        s.add(epl); s.flush()
        season = Season(competition_id=epl.id, year=2026, label="2026/27")
        home, away = Team(name="Chelsea", api_football_id=49, short_code="CHE"), Team(name="Leeds", api_football_id=63)
        s.add_all([season, home, away]); s.flush()
        s.add(Fixture(api_football_id=1557601, competition_id=epl.id, season_id=season.id, home_team_id=home.id,
                      away_team_id=away.id, kickoff_utc=datetime(2026, 11, 21, 15, tzinfo=timezone.utc),
                      status="NS"))
    return session_factory


def service(session_factory, reads=None):
    reads = reads or {}
    return ResearchMatchService(ResearchRepository(session_factory), load_read=lambda event_id: reads.get(event_id),
                                now=lambda: NOW, outlook_hours=96)


def test_match_carries_the_fixture_and_a_null_read_while_waiting(match):
    out = service(match).match("1557601")
    assert out["schema_version"] == "research-match.v1"
    assert out["fixture"]["status"] == "NS"
    assert out["fixture"]["home"]["name"] == "Chelsea" and out["fixture"]["venue"] is None
    assert out["fixture"]["read"]["status"] == "waiting"
    assert out["fixture"]["read"]["opens_at"] == "2026-11-17T15:00:00+00:00"
    assert out["read"] is None and out["lineups"] is None
    ready = service(match, reads={"1557601": (card(), metrics_results())}).match("1557601")
    assert ready["read"]["status"] == "ready" and ready["read"]["cards_expected"] == 4.6


def test_match_route_404_for_unknown_event_and_422_for_a_bad_id(match, monkeypatch):
    from web_app.routers import research as router_module
    monkeypatch.setattr(router_module, "_get_match_service", lambda: service(match))
    app = FastAPI(); app.include_router(router_module.router)
    client = TestClient(app)
    assert client.get("/api/research/matches/1557601").status_code == 200
    assert client.get("/api/research/matches/999").status_code == 404
    assert client.get("/api/research/matches/abc").status_code == 422

