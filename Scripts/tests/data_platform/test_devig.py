"""Margin-removal methods for bookmaker prices (rework A3.1)."""
import pytest

from data_platform import devig

METHODS = ("proportional", "additive", "power", "shin")


@pytest.mark.parametrize("method", METHODS)
def test_every_method_returns_probabilities_that_sum_to_one(method):
    p = devig.fair([1.30, 5.50, 9.00], method)
    assert sum(p) == pytest.approx(1, abs=1e-9) and all(0 < x < 1 for x in p)


@pytest.mark.parametrize("method", METHODS)
def test_a_price_without_margin_is_left_unchanged(method):
    odds = [2.0, 4.0, 4.0]
    assert devig.fair(odds, method) == pytest.approx([0.5, 0.25, 0.25], abs=1e-9)


@pytest.mark.parametrize("method", METHODS)
def test_equal_two_way_prices_are_fifty_fifty(method):
    assert devig.fair([1.91, 1.91], method) == pytest.approx([0.5, 0.5], abs=1e-9)


@pytest.mark.parametrize("method", ("power", "shin"))
def test_favourite_longshot_methods_take_more_margin_from_the_longshot(method):
    odds = [1.25, 6.0, 15.0]
    proportional = devig.fair(odds, "proportional")
    corrected = devig.fair(odds, method)
    assert corrected[0] > proportional[0]  # favourite gets more probability
    assert corrected[2] < proportional[2]  # longshot gets less


def test_additive_gives_up_rather_than_return_a_negative_probability():
    assert devig.fair([1.01, 15.0, 30.0, 40.0], "additive") is None  # margin share 2.9 pts > the 2.5% longshot


def test_shin_reports_its_insider_share():
    p, z = devig.shin([1.25, 6.0, 15.0], with_z=True)
    assert 0 < z < 0.2 and sum(p) == pytest.approx(1, abs=1e-9)


def test_unknown_method_and_bad_prices_are_errors():
    with pytest.raises(ValueError):
        devig.fair([2.0, 2.0], "magic")
    with pytest.raises(ValueError):
        devig.fair([1.0, 2.0], "proportional")
