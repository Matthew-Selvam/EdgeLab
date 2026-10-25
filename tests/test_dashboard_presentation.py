"""Tests for chart scales and fixture presentation of saved outputs."""

import pandas as pd
import pytest

from edgelab.dashboard.app import calibration_chart, comparison_chart, fixture_detail_table


def test_comparison_chart_keeps_a_zero_origin_and_saved_values() -> None:
    scores = pd.DataFrame([
        {"Model": "baseline", "Brier score": .64, "Log loss": 1.05, "Accuracy": .46},
        {"Model": "poisson", "Brier score": .56, "Log loss": .95, "Accuracy": .55},
    ])
    brier = comparison_chart(scores, "Brier score")
    assert list(brier.data[0].x) == [.64, .56]
    assert brier.layout.xaxis.range[0] == 0
    accuracy = comparison_chart(scores, "Accuracy")
    assert list(accuracy.data[0].x) == pytest.approx([46, 55])
    assert list(accuracy.layout.xaxis.range) == [0, 100]


def test_reliability_axes_stay_within_valid_probability_range() -> None:
    bins = pd.DataFrame([
        {"Model": "poisson", "Mean predicted": .4, "Observed frequency": .5,
         "Fixtures": 23, "Bin": "0.4–0.5"},
    ])
    figure = calibration_chart(bins, "H")
    assert list(figure.layout.xaxis.range) == [0, 1]
    assert list(figure.layout.yaxis.range) == [0, 1]
    assert figure.layout.yaxis.scaleanchor is None
    assert list(figure.data[1].x) == [.4]
    assert list(figure.data[1].y) == [.5]


def test_fixture_detail_shows_saved_probabilities_and_only_available_goals() -> None:
    rows = pd.DataFrame([
        {"model": "baseline", "p_home": .45, "p_draw": .23, "p_away": .32,
         "predicted_class": "H", "expected_home_goals": None, "expected_away_goals": None},
        {"model": "poisson", "p_home": .3, "p_draw": .2, "p_away": .5,
         "predicted_class": "A", "expected_home_goals": 1.234, "expected_away_goals": 1.89},
    ])
    detail = fixture_detail_table(rows)
    assert detail["Model"].tolist() == ["Class frequency baseline", "Independent Poisson"]
    assert detail["Home win %"].tolist() == [45, 30]
    assert detail["Away win %"].tolist() == [32, 50]
    assert detail["Expected goals H / A"].tolist() == ["—", "1.23 / 1.89"]
