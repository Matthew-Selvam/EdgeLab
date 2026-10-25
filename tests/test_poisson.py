"""Synthetic, hand-checkable tests for the independent-Poisson model."""

import math

import pandas as pd
import pytest

from edgelab.models.poisson import (
    PoissonConfig,
    PoissonGoalsModel,
    poisson_outcome_probabilities,
)


def _training():
    return pd.DataFrame([
        {"home_team": "A", "away_team": "B", "home_goals": 2, "away_goals": 0},
        {"home_team": "B", "away_team": "A", "home_goals": 1, "away_goals": 1},
        {"home_team": "A", "away_team": "C", "home_goals": 4, "away_goals": 2},
        {"home_team": "C", "away_team": "B", "home_goals": 0, "away_goals": 1},
    ])


def test_hand_calculated_team_rates_and_home_advantage():
    model = PoissonGoalsModel(PoissonConfig(prior_matches=2)).fit(_training())
    assert model.league_home_rate == pytest.approx(1.75)
    assert model.league_away_rate == pytest.approx(1.0)
    home_rate, away_rate = model.predict_rates("A", "B")
    assert home_rate == pytest.approx((2.375 * 1.375) / 1.75)
    assert away_rate == pytest.approx(0.75)
    assert model.predict_rates("New Home", "New Away") == pytest.approx((1.75, 1.0))


def test_team_aliases_share_strength_and_update_changes_only_team_totals():
    model = PoissonGoalsModel().fit(_training())
    before = model.predict_rates("Arsenal FC", "Chelsea FC")
    assert before == model.predict_rates("Arsenal", "Chelsea")
    league_means = (model.league_home_rate, model.league_away_rate)
    model.update_batch([("Arsenal FC", "Chelsea FC", 3, 1)])
    assert model.predict_rates("Arsenal", "Chelsea") != before
    assert (model.league_home_rate, model.league_away_rate) == league_means


def test_completed_batch_order_does_not_change_goal_rates():
    matches = [("A", "B", 2, 0), ("A", "C", 0, 1)]
    first = PoissonGoalsModel().fit(_training())
    second = PoissonGoalsModel().fit(_training())
    first.update_batch(matches)
    second.update_batch(list(reversed(matches)))
    assert first.predict_rates("A", "B") == second.predict_rates("A", "B")


def test_known_goal_distribution_and_outcome_mapping():
    assert poisson_outcome_probabilities(0, 0) == pytest.approx((0, 1, 0))
    home, draw, away = poisson_outcome_probabilities(1, 0)
    assert (home, draw, away) == pytest.approx((1 - math.exp(-1), math.exp(-1), 0))
    home, draw, away = poisson_outcome_probabilities(0, 1)
    assert (home, draw, away) == pytest.approx((0, math.exp(-1), 1 - math.exp(-1)))
    home, draw, away = poisson_outcome_probabilities(1, 1)
    assert draw == pytest.approx(0.3085083226, abs=1e-9)
    assert home == pytest.approx(away)
    assert home + draw + away == pytest.approx(1)


def test_probability_normalization_and_truncation_guard():
    for home_rate, away_rate in ((0.2, 3.8), (4.0, 0.4), (1.75, 1.0)):
        values = poisson_outcome_probabilities(home_rate, away_rate)
        assert all(math.isfinite(value) and 0 <= value <= 1 for value in values)
        assert sum(values) == pytest.approx(1, abs=1e-12)
    with pytest.raises(ValueError, match="tail exceeds"):
        poisson_outcome_probabilities(6, 6, PoissonConfig(max_goals=2))
    with pytest.raises(ValueError, match="nonnegative"):
        poisson_outcome_probabilities(-1, 1)


def test_fit_and_config_reject_invalid_inputs():
    with pytest.raises(ValueError, match="prior_matches"):
        PoissonConfig(prior_matches=0)
    with pytest.raises(ValueError, match="tail_tolerance"):
        PoissonConfig(tail_tolerance=0)
    with pytest.raises(ValueError, match="league goal means"):
        PoissonGoalsModel().fit(pd.DataFrame([{
            "home_team": "A", "away_team": "B", "home_goals": 1, "away_goals": 0,
        }]))
    with pytest.raises(ValueError, match="before predicting"):
        PoissonGoalsModel().predict_rates("A", "B")
