"""A5.2: selection policies replayed on the audited lines (D30)."""
import json

import pytest

from data_platform import disagreement as dis
from data_platform import selection_policies as sp
from Scripts.ops.selection_study import disagreements_to_check


def row(market, p_spix, p_ref, odds, *, fixture="1", stage="pre_match", side="over", y=1.0, w=1.0, close=None,
        selected=False):
    return {"fixture_id": fixture, "stage": stage, "market": market, "side": side, "p_spix": p_spix, "p_ref": p_ref,
            "best_odds": odds, "y": y, "w": w, "p_close": close, "selected": selected}


def by_name(name):
    return next(p for p in sp.policies() if p.name == name)


def test_tilt_runs_from_the_market_to_spix():
    r = row("goals", 0.70, 0.50, 2.0)
    assert sp.tilted(0.0)(r) == pytest.approx(0.50)
    assert sp.tilted(1.0)(r) == pytest.approx(0.70)
    assert 0.50 < sp.tilted(0.25)(r) < 0.70


def test_one_pick_per_market_and_three_per_fixture_by_expected_value():
    rows = [row("goals", 0.5, 0.55, 2.0), row("goals", 0.5, 0.60, 2.0, side="under"),   # EV 0.10 and 0.20
            row("corners", 0.5, 0.53, 2.0), row("cards", 0.5, 0.56, 2.0), row("btts", 0.5, 0.57, 2.0)]
    picks = sp.pick(rows, by_name("market"), min_ev=0.02)
    assert [(r["market"], r["side"]) for r, _, _ in picks] == [("goals", "under"), ("btts", "over"), ("cards", "over")]


def test_lines_below_the_minimum_value_or_without_a_price_are_not_picked():
    rows = [row("goals", 0.5, 0.505, 2.0), row("corners", 0.5, 0.6, None)]
    assert sp.pick(rows, by_name("market"), min_ev=0.02) == []


def test_current_replays_the_engines_choice_and_the_cap_drops_large_disagreements():
    rows = [row("goals", 0.80, 0.50, 2.0, selected=True), row("corners", 0.70, 0.50, 2.0)]
    assert [r["market"] for r, _, _ in sp.pick(rows, by_name("current"))] == ["goals"]
    capped = sp.pick([row("goals", 0.65, 0.50, 2.0), row("corners", 0.58, 0.50, 2.0)], by_name("spix_ev_capped_10"))
    assert [r["market"] for r, _, _ in capped] == ["corners"]


def test_evaluation_reports_value_left_at_the_close_on_priced_picks_only():
    picks = [(row("goals", 0.5, 0.5, 2.1, y=1.0, close=0.52), 0.5, 0.05),
             (row("corners", 0.5, 0.5, 2.1, y=0.0, w=0.5), 0.5, 0.05)]
    m = sp.evaluate(picks)
    assert m["ev_at_close"] == pytest.approx(0.52 * 2.1 - 1) and m["closing_picks"] == 1
    assert m["actual"] == pytest.approx(1 / 1.5)
    assert sp.stated_gap(picks) == pytest.approx(1 / 1.5 - 0.5)


def test_the_best_price_books_closing_price_is_classified():
    rows = [dict(row("goals", 0.5, 0.5, 2.0, close=0.5), best_book_close_odds=c) for c in (1.9, 2.0, 2.1, None)]
    rows.append(dict(row("goals", 0.5, 0.5, 2.0), best_book_close_odds=1.8))  # no closing snapshot: not counted
    assert sp.best_book_moves(rows) == {"shorter": 1, "same": 1, "longer": 1, "missing": 1}


def test_every_side_can_be_kept_for_the_policies():
    inputs = [{"market_key": "totals", "side": s, "point": 2.5, "odds": o, "bookmaker": b, "period": "regulation_time"}
              for b, (over, under) in {"A": (1.9, 1.95), "B": (2.0, 1.85)}.items() for s, o in (("over", over), ("under", under))]
    res = {"market": {"group": "goals"}, "fixture": {"home_team": "Lyon", "away_team": "Lens"},
           "context": {"decision_audit": {"input_options": inputs, "candidates": [
               {"option": {"market_key": "totals", "side": "over", "point": 2.5, "_model_prob": 0.4}}]}}}
    read = {"fixture_api_id": "7", "stage": "pre_match", "evaluated_at": "x", "kickoff": "y",
            "read_json": json.dumps({"canonical_results": [res]})}
    sides = {l["side"]: l for l in dis.read_lines(read, all_sides=True)}
    assert sorted(sides) == ["over", "under"] and sides["over"]["best_odds"] == 2.0 and sides["under"]["best_odds"] == 1.95
    assert sides["over"]["best_book"] == "B" and sides["under"]["best_book"] == "A"
    assert sides["over"]["p_ref"] + sides["under"]["p_ref"] == pytest.approx(1.0)


def test_disagreements_are_counted_per_read_and_market():
    rows = [row("goals", 0.70, 0.50, 2.0), row("goals", 0.65, 0.50, 2.0, side="under"), row("corners", 0.55, 0.50, 2.0),
            row("cards", 0.40, 0.50, 2.0, fixture="2")]
    assert disagreements_to_check(rows, 0.10) == {"reads": 2, "reads_with_any": 1, "markets_per_read": 0.5}
