"""Tests for the baseline model."""

from __future__ import annotations

import pandas as pd
import pytest

from edgelab.models.baseline import BaselineConfig, ClassPriorBaseline


@pytest.fixture
def balanced_training_data() -> pd.DataFrame:
    """Training data with equal H/D/A counts."""
    return pd.DataFrame({
        "outcome": ["H"] * 100 + ["D"] * 100 + ["A"] * 100,
        "match_date": ["2020-01-01"] * 300,
    })


@pytest.fixture
def skewed_training_data() -> pd.DataFrame:
    """Training data skewed toward home wins."""
    return pd.DataFrame({
        "outcome": ["H"] * 200 + ["D"] * 50 + ["A"] * 50,
        "match_date": ["2020-01-01"] * 300,
    })


class TestClassPriorBaseline:
    """Tests for the ClassPriorBaseline model."""

    def test_fit_balanced(self, balanced_training_data: pd.DataFrame):
        """Balanced data should produce approximately equal probabilities."""
        baseline = ClassPriorBaseline()
        baseline.fit(balanced_training_data)
        ph, p_draw, pa = baseline.predict_static()
        assert ph == pytest.approx(p_draw, abs=0.01)
        assert ph == pytest.approx(pa, abs=0.01)

    def test_fit_skewed(self, skewed_training_data: pd.DataFrame):
        """Skewed data should produce higher home-win probability."""
        baseline = ClassPriorBaseline()
        baseline.fit(skewed_training_data)
        ph, p_draw, pa = baseline.predict_static()
        assert ph > p_draw
        assert ph > pa

    def test_probabilities_sum_to_one(self, balanced_training_data: pd.DataFrame):
        """Probabilities must sum to 1."""
        baseline = ClassPriorBaseline()
        baseline.fit(balanced_training_data)
        ph, p_draw, pa = baseline.predict_static()
        assert ph + p_draw + pa == pytest.approx(1.0, abs=1e-10)

    def test_probabilities_non_negative(self, balanced_training_data: pd.DataFrame):
        """All probabilities must be non-negative."""
        baseline = ClassPriorBaseline()
        baseline.fit(balanced_training_data)
        ph, p_draw, pa = baseline.predict_static()
        assert ph >= 0
        assert p_draw >= 0
        assert pa >= 0

    def test_predict_proba_uses_subset(self, balanced_training_data: pd.DataFrame):
        """predict_proba should use only the provided subset."""
        baseline = ClassPriorBaseline()
        subset = balanced_training_data.iloc[:100]  # All H
        ph, p_draw, pa = baseline.predict_proba(subset)
        assert ph > p_draw
        assert ph > pa
        assert ph + p_draw + pa == pytest.approx(1.0, abs=1e-10)

    def test_smoothing_prevents_zeros(self):
        """With smoothing, no probability should be exactly zero."""
        df = pd.DataFrame({"outcome": ["H"] * 10})
        baseline = ClassPriorBaseline(BaselineConfig(smoothing_alpha=1.0))
        baseline.fit(df)
        ph, p_draw, pa = baseline.predict_static()
        assert p_draw > 0
        assert pa > 0

    def test_no_smoothing_can_give_zeros(self):
        """Without smoothing, unseen classes get zero probability."""
        df = pd.DataFrame({"outcome": ["H"] * 10})
        baseline = ClassPriorBaseline(BaselineConfig(smoothing_alpha=0.0))
        baseline.fit(df)
        ph, p_draw, pa = baseline.predict_static()
        assert ph == pytest.approx(1.0)
        assert p_draw == pytest.approx(0.0)
        assert pa == pytest.approx(0.0)
