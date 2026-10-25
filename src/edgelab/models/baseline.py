"""Baseline model: historical class-prior frequencies.

The baseline predicts H/D/A probabilities using the empirical outcome
frequencies observed in eligible training data only.  This establishes
the minimum performance benchmark against which all other models must be
compared.

For this first holdout, the baseline is fitted once on all development
seasons and remains frozen through validation and test.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from ..data.models import Outcome

logger = logging.getLogger(__name__)


@dataclass
class BaselineConfig:
    """Configuration for the class-prior baseline.

    Attributes:
        smoothing_alpha: Additive (Laplace) smoothing parameter to avoid
            zero probabilities.  Default 1.0 (add-one smoothing).
    """

    smoothing_alpha: float = 1.0


class ClassPriorBaseline:
    """Historical class-prior baseline for H/D/A prediction.

    Estimates P(H), P(D), P(A) from training data only and applies this
    same probability vector to all evaluation fixtures.

    The pipeline fits it after development and freezes its class frequencies
    throughout validation and test. ``predict_proba`` also accepts an explicit
    historical subset for future expanding-window experiments.
    """

    def __init__(self, config: Optional[BaselineConfig] = None) -> None:
        self.config = config or BaselineConfig()
        if self.config.smoothing_alpha < 0:
            raise ValueError("smoothing_alpha must be nonnegative")
        self._training_outcomes: list[str] = []
        self._is_fitted: bool = False

    def fit(self, training_data: pd.DataFrame) -> "ClassPriorBaseline":
        """Fit the baseline on training data.

        Parameters
        ----------
        training_data:
            DataFrame with at least an ``outcome`` column (values H/D/A).

        Returns
        -------
        ClassPriorBaseline
            Self (for chaining).
        """
        outcomes = training_data["outcome"].tolist()
        if not outcomes or set(outcomes) - {"H", "D", "A"}:
            raise ValueError("training outcomes must be nonempty H/D/A labels")
        self._training_outcomes = outcomes
        self._is_fitted = True

        counts = Counter(outcomes)
        total = len(outcomes)
        alpha = self.config.smoothing_alpha

        logger.info(
            "Baseline fitted on %d training matches: H=%d, D=%d, A=%d",
            total,
            counts.get("H", 0),
            counts.get("D", 0),
            counts.get("A", 0),
        )
        return self

    def predict_proba(self, training_subset: pd.DataFrame) -> tuple[float, float, float]:
        """Predict H/D/A probabilities using a training subset.

        This method allows the baseline to update with an expanding window
        by accepting the subset of training data available before the
        forecast date.

        Parameters
        ----------
        training_subset:
            DataFrame of training matches available before the forecast
            date (for expanding-window evaluation).

        Returns
        -------
        tuple
            ``(p_home, p_draw, p_away)`` — probabilities summing to 1.
        """
        outcomes = training_subset["outcome"].tolist()
        if not outcomes or set(outcomes) - {"H", "D", "A"}:
            raise ValueError("training subset must contain valid H/D/A labels")
        counts = Counter(outcomes)
        total = len(outcomes)
        alpha = self.config.smoothing_alpha

        classes = ["H", "D", "A"]
        probs = {}
        for c in classes:
            probs[c] = (counts.get(c, 0) + alpha) / (total + 3 * alpha)

        # Normalize to ensure exact sum to 1
        p_sum = probs["H"] + probs["D"] + probs["A"]
        p_home = probs["H"] / p_sum
        p_draw = probs["D"] / p_sum
        p_away = probs["A"] / p_sum

        return float(p_home), float(p_draw), float(p_away)

    def predict_static(self) -> tuple[float, float, float]:
        """Predict H/D/A probabilities using all fitted training data.

        Returns
        -------
        tuple
            ``(p_home, p_draw, p_away)`` — probabilities summing to 1.
        """
        if not self._is_fitted:
            # Use uniform prior if not fitted
            return 1.0 / 3, 1.0 / 3, 1.0 / 3

        counts = Counter(self._training_outcomes)
        total = len(self._training_outcomes)
        alpha = self.config.smoothing_alpha

        classes = ["H", "D", "A"]
        probs = {}
        for c in classes:
            probs[c] = (counts.get(c, 0) + alpha) / (total + 3 * alpha)

        p_sum = probs["H"] + probs["D"] + probs["A"]
        p_home = probs["H"] / p_sum
        p_draw = probs["D"] / p_sum
        p_away = probs["A"] / p_sum

        return float(p_home), float(p_draw), float(p_away)
