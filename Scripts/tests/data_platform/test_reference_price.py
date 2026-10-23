"""Market reference price: every bookmaker's margin removed, median across books."""
import pytest

from data_platform import reference_price as ref
from .test_safe_settlement import store, publish, payload, NOW, no_network  # noqa: F401  (fixtures)


def option(book, market_key, odds, side=None, point=None, team=None):
    o = {"bookmaker": book, "market_key": market_key, "odds": odds, "period": "regulation_time"}
    if side is not None:
        o["side"] = side
    if point is not None:
        o["point"] = point
    if team is not None:
        o["team"] = team
    return o


def fair(*odds):
    inverse = sum(1 / o for o in odds)
    return (1 / odds[0]) / inverse


def test_totals_consensus_is_the_median_of_each_books_fair_probability():
    books = {"A": [("over", 2.5, 1.8), ("under", 2.5, 2.0), ("over", 3.5, 3.0)],
             "B": [("over", 2.5, 1.9), ("under", 2.5, 1.9)],
             "C": [("over", 2.5, 2.0), ("under", 2.5, 1.8)]}
    same, consensus, issue = ref.devig_consensus(books, "over", 2.5, spreads=False, wanted_book="b")
    assert issue is None and consensus["book_count"] == 3
    assert consensus["fair_probability"] == pytest.approx(0.5)
    assert same == {"bookmaker": "B", "odds": 1.9, "fair": pytest.approx(0.5)}
    assert consensus["method"] == ref.METHOD


def test_handicap_pairs_the_team_line_with_the_opponent_at_the_opposite_line():
    books = {"A": [("lyon", -0.5, 1.9), ("palace", 0.5, 1.95), ("lyon", -1.0, 2.6), ("palace", 1.0, 1.5)]}
    _, consensus, _ = ref.devig_consensus(books, "lyon", -0.5, spreads=True)
    assert consensus["fair_probability"] == pytest.approx(fair(1.9, 1.95))


def test_three_way_result_market_uses_all_three_prices():
    books = {"A": [("home", None, 2.0), ("draw", None, 3.4), ("away", None, 3.8)]}
    _, consensus, _ = ref.devig_consensus(books, "draw", None, spreads=False)
    assert consensus["fair_probability"] == pytest.approx(fair(3.4, 2.0, 3.8))


def test_one_sided_or_duplicated_books_are_left_out():
    books = {"one_sided": [("over", 2.5, 1.8)],
             "duplicated": [("over", 2.5, 1.8), ("over", 2.5, 1.85), ("under", 2.5, 2.0)],
             "good": [("over", 2.5, 1.9), ("under", 2.5, 1.9)]}
    same, consensus, issue = ref.devig_consensus(books, "over", 2.5, spreads=False, wanted_book="one_sided")
    assert consensus["book_count"] == 1 and same is None and issue == "same_bookmaker_unavailable"


def test_a_selection_no_book_prices_has_no_reference():
    same, consensus, issue = ref.devig_consensus({"A": [("over", 3.5, 3.0), ("under", 3.5, 1.4)]},
                                                 "over", 2.5, spreads=False)
    assert (same, consensus, issue) == (None, None, "selection_unavailable")


def published_result(options, side="over", line=2.5, key="totals", group="goals", bookmaker="Book"):
    result = payload(group, key, side, line)
    result["decision"]["quote"]["bookmaker"] = bookmaker
    result["context"] = {"decision_audit": {"input_options": options}}
    return result


def test_publication_reference_reads_the_prices_stored_in_the_decision_audit():
    result = published_result([option("Book", "totals", 1.8, "over", 2.5), option("Book", "totals", 2.0, "under", 2.5),
                               option("Other", "totals", 1.9, "over", 2.5), option("Other", "totals", 1.9, "under", 2.5),
                               option("Other", "btts", 1.7, "yes")])
    reference = ref.publication_reference(result)
    assert reference["status"] == "available" and reference["source"] == "decision_audit.input_options"
    assert reference["fair_probability"] == pytest.approx((fair(1.8, 2.0) + 0.5) / 2)
    assert reference["same_book"]["odds"] == 1.8


def test_publication_reference_matches_handicap_teams_by_name():
    result = published_result([option("Book", "spreads", 1.9, point=-0.5, team="A"),
                               option("Book", "spreads", 1.95, point=0.5, team="B")],
                              side="a ", line=-0.5, key="spreads", group="spreads")
    assert ref.publication_reference(result)["fair_probability"] == pytest.approx(fair(1.9, 1.95))


def test_publication_without_stored_prices_says_so():
    reference = ref.publication_reference(published_result([]))
    assert reference == {"status": "unavailable", "reason": "no_publication_prices", "method": ref.METHOD}


def test_publishing_records_the_reference_price(store):
    result = published_result([option("Book", "totals", 2.0, "over", 2.5), option("Book", "totals", 1.8, "under", 2.5)])
    store[1].publish(result, surface="website", published_at="2026-09-18T10:00:00Z")
    (row,) = store[0].get_recent(days=3650)
    assert row["publication_reference"]["fair_probability"] == pytest.approx(fair(2.0, 1.8))


def test_backfill_fills_missing_references_once(store):
    from data_platform.services.publications import backfill_publication_references
    result = published_result([option("Book", "totals", 2.0, "over", 2.5), option("Book", "totals", 1.8, "under", 2.5)])
    pid = store[1].publish(result, surface="website", published_at="2026-09-18T10:00:00Z")["prediction_id"]
    from data_platform.models import Prediction
    with store[0]._factory() as session:  # an older publication, made before references were recorded
        row = session.get(Prediction, pid)
        row.extras = {k: v for k, v in row.extras.items() if k != "publication_reference"}
        session.commit()
    assert backfill_publication_references(store[0]._factory) == {"filled": 1, "already": 0}
    assert backfill_publication_references(store[0]._factory) == {"filled": 0, "already": 1}
    assert store[0].get_recent(days=3650)[0]["publication_reference"]["status"] == "available"


def test_power_method_is_an_option_and_the_default_is_unchanged():
    from data_platform import devig
    books = {"A": [("home", None, 1.4), ("draw", None, 4.6), ("away", None, 8.0)]}
    _, default, _ = ref.devig_consensus(books, "away", None, spreads=False)
    _, power, _ = ref.devig_consensus(books, "away", None, spreads=False, method="power")
    assert default["method"] == "proportional_devig_median.v1"
    assert power["method"] == "power_devig_median.v2"
    assert power["fair_probability"] == pytest.approx(devig.power([1.4, 4.6, 8.0])[2])
    assert power["fair_probability"] < default["fair_probability"]  # less weight on the longshot


def test_the_spread_is_the_interquartile_range_of_the_books_fair_probabilities():
    consensus = {"books": [{"fair": f} for f in (0.50, 0.52, 0.53, 0.60)]}
    assert ref.consensus_spread(consensus) == pytest.approx(0.0325)  # q1 0.515, q3 0.5475
    assert ref.consensus_spread({"books": [{"fair": 0.5}]}) is None and ref.consensus_spread(None) is None
