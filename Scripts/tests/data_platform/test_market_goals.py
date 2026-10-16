"""M1: the market's expected goals per team, solved from its fair prices (rework A3.2)."""
import pytest

from data_platform import market_goals as mg


def test_recovers_the_goal_means_behind_result_and_total_prices():
    targets = {"1x2": mg.result_probabilities(1.6, 1.1), "totals": {2.5: mg.over_probability(1.6, 1.1, 2.5)}}
    fit = mg.solve(targets)
    assert fit["home"] == pytest.approx(1.6, abs=0.01) and fit["away"] == pytest.approx(1.1, abs=0.01)
    assert fit["error"] < 1e-6 and fit["inputs"] == 4


def test_handicap_price_alone_with_a_total_is_enough():
    targets = {"handicaps": {-0.75: mg.handicap_probability(1.9, 0.9, -0.75)},
               "totals": {2.75: mg.over_probability(1.9, 0.9, 2.75)}}
    fit = mg.solve(targets)
    assert fit["home"] == pytest.approx(1.9, abs=0.02) and fit["away"] == pytest.approx(0.9, abs=0.02)


def test_handicap_probability_is_the_fair_two_way_price_including_half_results():
    # Home -0.25 is half on 0 (push on a draw) and half on -0.5. At the fair price the expected profit is zero.
    profile = mg.handicap_profile(1.4, 1.2, -0.25)
    q = mg.handicap_probability(1.4, 1.2, -0.25)
    win = profile["full_win"] + 0.5 * profile["half_win"]
    loss = profile["full_loss"] + 0.5 * profile["half_loss"]
    assert q == pytest.approx(win / (win + loss))
    assert mg.handicap_probability(1.4, 1.2, -0.5) == pytest.approx(mg.result_probabilities(1.4, 1.2)[0])


def test_over_probability_for_whole_lines_excludes_the_push():
    over = mg.over_probability(1.3, 1.1, 2.0)
    pmf = mg.total_pmf(1.3, 1.1)
    assert over == pytest.approx(sum(pmf[3:]) / (1 - pmf[2]))


def test_too_few_prices_give_no_answer():
    assert mg.solve({"totals": {2.5: 0.55}}) is None
