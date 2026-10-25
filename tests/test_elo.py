"""Tests for the Elo model."""

from __future__ import annotations

import pytest

from edgelab.data.models import Outcome
from edgelab.models.elo import EloConfig, EloLogisticModel, EloRatings


@pytest.fixture
def ratings() -> EloRatings:
    """Return fresh EloRatings with default config."""
    config = EloConfig(initial_rating=1500.0, k_factor=20.0, home_advantage=60.0)
    return EloRatings({}, config)


@pytest.fixture
def elo_model() -> EloLogisticModel:
    """Return an EloLogisticModel with default config."""
    return EloLogisticModel()


class TestEloRatings:
    """Tests for EloRatings."""

    def test_initial_rating(self, ratings: EloRatings):
        """Unknown teams should get the initial rating."""
        assert ratings.get("unknown_team") == 1500.0

    def test_expected_score_equal_ratings(self, ratings: EloRatings):
        """Equal ratings should give expected score > 0.5 for home (due to HFA)."""
        ratings.ratings = {"team_a": 1500.0, "team_b": 1500.0}
        es = ratings.expected_score("team_a", "team_b")
        # diff = 1500 + 60 - 1500 = 60, E = 1/(1+10^(-60/400)) ≈ 0.5855
        assert es == pytest.approx(0.5855, abs=0.01)
        assert es > 0.5  # Home advantage

    def test_expected_score_stronger_home(self, ratings: EloRatings):
        """Home team with much higher rating should have high expected score."""
        ratings.ratings = {"home": 2000.0, "away": 1000.0}
        es = ratings.expected_score("home", "away")
        assert es > 0.95

    def test_expected_score_weaker_away(self, ratings: EloRatings):
        """Away team with much higher rating should have low expected score for home."""
        ratings.ratings = {"home": 1000.0, "away": 2000.0}
        es = ratings.expected_score("home", "away")
        # diff = 1000 + 60 - 2000 = -940, E ≈ very small
        assert es < 0.05

    def test_update_home_win(self, ratings: EloRatings):
        """Home win should increase home rating, decrease away rating."""
        ratings.ratings = {"home": 1500.0, "away": 1500.0}
        r_home, r_away = ratings.update("home", "away", Outcome.HOME_WIN)
        assert r_home > 1500.0
        assert r_away < 1500.0

    def test_update_away_win(self, ratings: EloRatings):
        """Away win should decrease home rating, increase away rating."""
        ratings.ratings = {"home": 1500.0, "away": 1500.0}
        r_home, r_away = ratings.update("home", "away", Outcome.AWAY_WIN)
        assert r_home < 1500.0
        assert r_away > 1500.0

    def test_update_draw(self, ratings: EloRatings):
        """Draw should adjust ratings slightly toward each other."""
        ratings.ratings = {"home": 1500.0, "away": 1500.0}
        r_home, r_away = ratings.update("home", "away", Outcome.DRAW)
        # With home advantage, home expected > 0.5, so draw (S=0.5) means
        # home loses a tiny bit, away gains a tiny bit
        assert r_home < 1500.0
        assert r_away > 1500.0

    def test_update_is_deterministic(self, ratings: EloRatings):
        """Same inputs should produce same updates."""
        ratings.ratings = {"home": 1500.0, "away": 1500.0}
        r1 = ratings.update("home", "away", Outcome.HOME_WIN)
        # Reset and do again
        ratings.ratings = {"home": 1500.0, "away": 1500.0}
        r2 = ratings.update("home", "away", Outcome.HOME_WIN)
        assert r1 == r2

    def test_regress_for_season(self, ratings: EloRatings):
        """Season regression should pull ratings toward initial."""
        config = EloConfig(initial_rating=1500.0, season_regression_factor=0.5)
        ratings = EloRatings({"team_a": 2000.0, "team_b": 1000.0}, config)
        ratings.regress_for_season()
        assert ratings.get("team_a") < 2000.0
        assert ratings.get("team_b") > 1000.0

    def test_no_regression_when_factor_zero(self, ratings: EloRatings):
        """Zero regression factor should leave ratings unchanged."""
        config = EloConfig(season_regression_factor=0.0)
        ratings = EloRatings({"team_a": 2000.0}, config)
        ratings.regress_for_season()
        assert ratings.get("team_a") == 2000.0


class TestEloLogisticModel:
    """Tests for EloLogisticModel."""

    def test_probabilities_sum_to_one(self, elo_model: EloLogisticModel):
        """Three probabilities must sum to 1."""
        ratings = EloRatings({}, EloConfig())
        ph, pd, pa = elo_model.predict_proba("Team A", "Team B", ratings)
        assert ph + pd + pa == pytest.approx(1.0, abs=1e-10)

    def test_probabilities_non_negative(self, elo_model: EloLogisticModel):
        """All probabilities must be in [0, 1]."""
        ratings = EloRatings({}, EloConfig())
        ph, pd, pa = elo_model.predict_proba("Team A", "Team B", ratings)
        assert 0 <= ph <= 1
        assert 0 <= pd <= 1
        assert 0 <= pa <= 1

    def test_equal_ratings_favor_home(self, elo_model: EloLogisticModel):
        """Equal teams should give higher P(home) than P(away) due to HFA."""
        ratings = EloRatings({"teama": 1500.0, "teamb": 1500.0}, EloConfig())
        ph, pd, pa = elo_model.predict_proba("Team A", "Team B", ratings)
        assert ph > pa

    def test_stronger_home_increases_p_home(self, elo_model: EloLogisticModel):
        """Much stronger home team should have higher P(home)."""
        ratings_weak = EloRatings({"home": 1500.0, "away": 1500.0}, EloConfig())
        ratings_strong = EloRatings({"home": 2000.0, "away": 1500.0}, EloConfig())
        ph_weak, _, _ = elo_model.predict_proba("Home", "Away", ratings_weak)
        ph_strong, _, _ = elo_model.predict_proba("Home", "Away", ratings_strong)
        assert ph_strong > ph_weak

    def test_stronger_away_decreases_p_home(self, elo_model: EloLogisticModel):
        """Much stronger away team should lower P(home)."""
        ratings_weak = EloRatings({"home": 1500.0, "away": 1500.0}, EloConfig())
        ratings_strong_away = EloRatings({"home": 1500.0, "away": 2000.0}, EloConfig())
        ph_weak, _, _ = elo_model.predict_proba("Home", "Away", ratings_weak)
        ph_strong_away, _, _ = elo_model.predict_proba("Home", "Away", ratings_strong_away)
        assert ph_strong_away < ph_weak

    def test_equal_ratings_give_draw_high_prob(self, elo_model: EloLogisticModel):
        """Equal teams should have non-trivial draw probability."""
        ratings = EloRatings({"teama": 1500.0, "teamb": 1500.0}, EloConfig())
        _, pd, _ = elo_model.predict_proba("Team A", "Team B", ratings)
        assert pd > 0.1  # Draws are common in football

    def test_extreme_elo_diff_lowers_draw(self, elo_model: EloLogisticModel):
        """Large rating difference should reduce draw probability."""
        config = EloConfig(initial_rating=1500.0, home_advantage=60.0)
        # Use real team names that exist in the alias mapping
        # "Manchester United" -> canonical_id "manchesterunited"
        # "Fulham" -> canonical_id "fulham"
        ratings_equal = EloRatings({"fulham": 1500.0, "manchesterunited": 1500.0}, config)
        ratings_diff = EloRatings({"fulham": 2200.0, "manchesterunited": 800.0}, config)
        _, pd_equal, _ = elo_model.predict_proba("Manchester United FC", "Fulham FC", ratings_equal)
        _, pd_diff, _ = elo_model.predict_proba("Manchester United FC", "Fulham FC", ratings_diff)
        assert pd_diff < pd_equal

    def test_team_name_alias_mapping(self, elo_model: EloLogisticModel):
        """Team names with different suffixes should canonicalize to same ID."""
        ratings = EloRatings({}, EloConfig())
        # "Manchester United" and "Manchester United FC" should produce same IDs
        ph1, pd1, pa1 = elo_model.predict_proba("Manchester United", "Fulham FC", ratings)
        ph2, pd2, pa2 = elo_model.predict_proba("Manchester United FC", "Fulham", ratings)
        assert ph1 == pytest.approx(ph2, abs=1e-10)
        assert pd1 == pytest.approx(pd2, abs=1e-10)
        assert pa1 == pytest.approx(pa2, abs=1e-10)
