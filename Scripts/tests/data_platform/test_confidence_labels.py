"""A5.3 (D44): a pick is labelled high when the market is firm and, where Spix has earned a vote, Spix agrees."""
import statistics

import pytest

from data_platform import confidence_labels as cl
from data_platform.confidence_labels import Cutoffs
from data_platform.devig import power
from data_platform.reference_price import audit_books, consensus_spread, devig_consensus

EVIDENCE = dict(market="goals", book_count=5, book_spread=0.01, market_value=0.03, spix_value=-0.04)
BOOKS = {"A": (1.90, 1.95), "B": (2.05, 1.80), "C": (2.00, 1.85), "Betano": (2.05, 1.82)}


def quote(market_key, side, point, odds, book, team=None):
    return {"market_key": market_key, "side": side, "team": team, "point": point, "odds": odds, "bookmaker": book,
            "period": "regulation_time"}


def totals_options(point=2.5, books=BOOKS):
    return [quote("totals", side, point, odds, book) for book, (over, under) in books.items()
            for side, odds in (("over", over), ("under", under))]


def picked(*, group="goals", market_key="totals", side="over", point=2.5, odds=2.05, book="Betano", p=0.52,
           options=None, candidates=(), labels="evidence", context=None):
    """A recommended result as the engine stores it, with the prices its decision audit kept."""
    audit = {"input_options": totals_options(point) if options is None else options, "candidates": list(candidates)}
    return {"market": {"group": group, "key": group},
            "fixture": {"event_id": "7", "home_team": "Lyon", "away_team": "Lens"},
            "decision": {"status": "recommended", "reason": None, "confidence": "medium", "model_probability": p,
                         "implied_probability": 0.48, "value_edge": 0.04, "expected_value": p * odds - 1,
                         "probability_basis": "outcome_probability",
                         "quote": {"side": side, "line": point, "odds": odds, "bookmaker": book,
                                   "market_key": market_key, "period": "regulation_time"}},
            "context": {"engine_switches": {"SPIX_DEVIG_METHOD": "power", "SPIX_CONFIDENCE_LABELS": labels},
                        "decision_audit": audit, **(context or {})}}


def test_a_firm_fresh_market_with_value_is_high():
    assert cl.label(EVIDENCE, 12, cutoffs=Cutoffs(0.02)) == "high"


@pytest.mark.parametrize("change, age, cutoffs", [
    ({"book_spread": 0.03}, 12, Cutoffs(0.02)),   # scattered bookmakers
    ({"book_count": 3}, 12, Cutoffs(0.02)),       # too few bookmakers
    ({}, 45, Cutoffs(0.02)),                      # an old price
    ({}, None, Cutoffs(0.02)),                    # an unknown price age
    ({}, 12, Cutoffs(None)),                      # no cut-off has passed its check
    ({"market_value": 0.01}, 12, Cutoffs(0.02)),  # the market sees too little value
])
def test_scattered_few_books_old_or_unknown_price_or_no_cutoff_is_low(change, age, cutoffs):
    assert cl.label({**EVIDENCE, **change}, age, cutoffs=cutoffs) == "low"
    assert cl.label(None, 12, cutoffs=Cutoffs(0.02)) == "low"


def test_spix_votes_only_in_markets_that_earned_it():
    corners = {**EVIDENCE, "market": "corners"}
    assert cl.label(corners, 12, cutoffs=Cutoffs(0.02), spix_vote=frozenset()) == "high"
    assert cl.label(corners, 12, cutoffs=Cutoffs(0.02), spix_vote=frozenset({"corners"})) == "low"
    agreeing = {**corners, "spix_value": 0.03}
    assert cl.label(agreeing, 12, cutoffs=Cutoffs(0.02), spix_vote=frozenset({"corners"})) == "high"


def test_line_evidence_reads_the_books_and_spix_from_the_result():
    chosen = {"option": {**quote("totals", "over", 2.5, 2.05, "Betano"), "_model_prob": 0.52, "_spix_prob": 0.47},
              "selector_selected": True}
    evidence = cl.line_evidence(picked(candidates=[chosen]))
    fair = [power(list(prices))[0] for prices in BOOKS.values()]
    _, consensus, _ = devig_consensus(audit_books(totals_options(), "totals"), "over", 2.5, spreads=False, method="power")
    assert evidence["book_count"] == 4 and evidence["book_spread"] == pytest.approx(consensus_spread(consensus))
    assert evidence["market_value"] == pytest.approx(statistics.median(fair) * 2.05 - 1)
    assert evidence["spix_value"] == pytest.approx(0.47 * 2.05 - 1) and evidence["method"] == "power"


def test_moneyline_spix_probability_comes_from_the_anchor_record():
    prices = {"A": (2.6, 3.4, 2.7), "B": (2.5, 3.5, 2.8), "C": (2.55, 3.3, 2.75), "D": (2.6, 3.45, 2.65)}
    options = [quote("h2h", side, None, odds, book) for book, row in prices.items()
               for side, odds in zip(("home", "draw", "away"), row)]
    anchor = {"market_anchor": {"market": {"home": 0.36, "draw": 0.28, "away": 0.36},
                                "spix": {"home": 0.42, "draw": 0.29, "away": 0.29}}}
    result = picked(group="moneyline", market_key="h2h", side="away", point=None, odds=2.8, book="B", p=0.36,
                    options=options, context=anchor)
    evidence = cl.line_evidence(result)
    assert evidence["spix_probability"] == 0.29 and evidence["book_count"] == 4


def test_a_quote_whose_own_book_is_one_sided_is_still_labelled():
    books = {"A": (1.90, 1.95), "B": (2.05, 1.80), "C": (2.00, 1.85), "D": (1.95, 1.90)}
    options = totals_options(books=books) + [quote("totals", "over", 2.5, 2.10, "Betano")]  # no under at Betano
    evidence = cl.line_evidence(picked(odds=2.10, options=options))
    assert evidence is not None and evidence["book_count"] == 4


def test_a_result_without_audited_prices_is_low():
    bare = picked()
    del bare["context"]["decision_audit"]
    assert cl.line_evidence(bare) is None
    assert cl.apply_label(bare)["decision"]["confidence"] == "low"


def test_a_quarter_line_is_labelled():
    assert cl.line_evidence(picked(point=2.25, odds=1.95)) is not None


def test_price_age_is_known_only_from_a_confirmed_price():
    def checked(status):
        return {"context": {"price_confirmation": {"status": status, "checked_at": "2026-10-24T12:30:00+00:00",
                                                   "bookmaker_seen_at": "2026-10-24T12:18:00Z"}}}
    assert cl.price_age_minutes(checked("confirmed")) == 12.0
    assert cl.price_age_minutes(checked("repriced")) == 12.0
    assert cl.price_age_minutes(checked("not_confirmed")) is None
    assert cl.price_age_minutes({"context": {}}) is None


def test_labels_change_nothing_with_the_switch_off_and_never_change_a_decision():
    off = picked(labels="edge")
    assert cl.apply_label(off) == off
    on = picked()
    labelled = cl.apply_label(on)
    assert {k: v for k, v in labelled["decision"].items() if k != "confidence"} == \
        {k: v for k, v in on["decision"].items() if k != "confidence"}
    assert labelled["decision"]["confidence"] in ("high", "low")
    assert set(labelled["context"]) - set(on["context"]) == {"confidence_evidence"}
    assert {k: v for k, v in labelled["context"].items() if k != "confidence_evidence"} == on["context"]
    record = labelled["context"]["confidence_evidence"]
    assert record["version"] == cl.VERSION and record["label"] == labelled["decision"]["confidence"]


def test_a_no_bet_is_never_labelled():
    no_bet = picked()
    no_bet["decision"]["status"] = "no_bet"
    assert cl.apply_label(no_bet) == no_bet


def test_min_value_matches_the_selection_threshold():
    import sys
    from pathlib import Path
    engine = str(Path(__file__).resolve().parents[2] / "rag_ingest")  # line_selection imports the engine's modules
    if engine not in sys.path:
        sys.path.insert(0, engine)
    from Scripts.rag_ingest.core.line_selection import MARKET_VALUE_MIN_EV
    assert cl.MIN_VALUE == MARKET_VALUE_MIN_EV


def test_the_label_it_replaced_is_kept_for_ranking_across_both_labelling_passes():
    once = cl.apply_label(picked())  # the engine's pass; the decision carried today's "medium"
    assert once["context"]["confidence_evidence"]["edge_label"] == "medium"
    twice = cl.apply_label(once)  # the Match Read worker's pass after price confirmation
    assert twice["context"]["confidence_evidence"]["edge_label"] == "medium"
