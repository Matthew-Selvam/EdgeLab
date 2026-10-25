"""Tests for evaluation metrics."""

from __future__ import annotations

import math

import pytest

from edgelab.evaluation.metrics import (
    accuracy,
    brier_score_multiclass,
    calibration_error,
    compute_all_metrics,
    log_loss_multiclass,
    rps_score,
)


class TestBrierScore:
    """Tests for multiclass Brier score."""

    def test_perfect_predictions(self):
        """Perfect predictions should give Brier score of 0."""
        y_true = ["H", "D", "A"]
        probs = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
        score = brier_score_multiclass(y_true, probs)
        assert score == pytest.approx(0.0, abs=1e-10)

    def test_uniform_predictions(self):
        """Uniform predictions should give Brier score of 2/3."""
        y_true = ["H"]
        probs = [(1/3, 1/3, 1/3)]
        score = brier_score_multiclass(y_true, probs)
        # (1/3 - 1)^2 + (1/3 - 0)^2 + (1/3 - 0)^2 = 4/9 + 1/9 + 1/9 = 6/9 = 2/3
        assert score == pytest.approx(2.0 / 3.0, abs=1e-10)

    def test_empty_input(self):
        """Empty input should return 0.0."""
        assert brier_score_multiclass([], []) == 0.0

    def test_known_value(self):
        """Test with a known computation."""
        y_true = ["H"]
        probs = [(0.5, 0.3, 0.2)]
        score = brier_score_multiclass(y_true, probs)
        # (0.5-1)^2 + (0.3-0)^2 + (0.2-0)^2 = 0.25 + 0.09 + 0.04 = 0.38
        assert score == pytest.approx(0.38, abs=1e-10)


class TestLogLoss:
    """Tests for multiclass log loss."""

    def test_perfect_prediction(self):
        """Perfect prediction should give log loss of 0."""
        y_true = ["H"]
        probs = [(1.0, 0.0, 0.0)]
        score = log_loss_multiclass(y_true, probs)
        assert score == pytest.approx(0.0, abs=1e-10)

    def test_uniform_prediction(self):
        """Uniform prediction should give log loss of log(3)."""
        y_true = ["H"]
        probs = [(1/3, 1/3, 1/3)]
        score = log_loss_multiclass(y_true, probs)
        assert score == pytest.approx(math.log(3), abs=1e-10)

    def test_worse_prediction_higher_loss(self):
        """Giving low probability to the actual class should increase loss."""
        y_true = ["H"]
        good_probs = [(0.9, 0.05, 0.05)]
        bad_probs = [(0.1, 0.45, 0.45)]
        good_loss = log_loss_multiclass(y_true, good_probs)
        bad_loss = log_loss_multiclass(y_true, bad_probs)
        assert bad_loss > good_loss

    def test_empty_input(self):
        assert log_loss_multiclass([], []) == 0.0

    def test_epsilon_clamping(self):
        """Extreme probabilities should not produce infinite loss."""
        y_true = ["H"]
        probs = [(1.0, 0.0, 0.0)]
        # Even with perfect prediction, epsilon clamping should keep it finite
        score = log_loss_multiclass(y_true, probs, epsilon=1e-15)
        assert math.isfinite(score)


class TestRPS:
    """Tests for Ranked Probability Score."""

    def test_perfect_prediction(self):
        """Perfect prediction should give RPS of 0."""
        y_true = ["H"]
        probs = [(1.0, 0.0, 0.0)]
        score = rps_score(y_true, probs)
        assert score == pytest.approx(0.0, abs=1e-10)

    def test_uniform_prediction(self):
        """Uniform prediction should give a known RPS value."""
        y_true = ["H"]
        probs = [(1/3, 1/3, 1/3)]
        score = rps_score(y_true, probs)
        # Ordering is A < D < H
        # Cumulative: p = [1/3, 2/3, 1], y = [0, 0, 1]
        # (1/3-0)^2 + (2/3-0)^2 + (1-1)^2 = 1/9 + 4/9 + 0 = 5/9
        # Divided by (K-1) = 2: 5/18
        assert score == pytest.approx(5.0 / 18.0, abs=1e-10)

    def test_empty_input(self):
        assert rps_score([], []) == 0.0

    def test_rps_range(self):
        """RPS should be in [0, 1] for 3 classes."""
        y_true = ["H", "D", "A", "H"]
        probs = [(0.5, 0.3, 0.2), (0.2, 0.6, 0.2), (0.1, 0.2, 0.7), (0.9, 0.05, 0.05)]
        score = rps_score(y_true, probs)
        assert 0 <= score <= 1


class TestAccuracy:
    """Tests for accuracy."""

    def test_perfect_accuracy(self):
        y_true = ["H", "D", "A"]
        probs = [(0.9, 0.05, 0.05), (0.05, 0.9, 0.05), (0.05, 0.05, 0.9)]
        assert accuracy(y_true, probs) == pytest.approx(1.0)

    def test_zero_accuracy(self):
        y_true = ["H", "D", "A"]
        probs = [(0.05, 0.9, 0.05), (0.9, 0.05, 0.05), (0.05, 0.9, 0.05)]
        assert accuracy(y_true, probs) == pytest.approx(0.0)

    def test_half_accuracy(self):
        y_true = ["H", "D", "A", "H"]
        # Predicted: H (argmax 0.9), A (argmax 0.9), A (argmax 0.7), H (argmax 0.9)
        # Correct: H ✓, D ✗, A ✓, H ✓ = 3/4 = 0.75
        probs = [(0.9, 0.05, 0.05), (0.05, 0.05, 0.9), (0.05, 0.25, 0.7), (0.9, 0.05, 0.05)]
        assert accuracy(y_true, probs) == pytest.approx(0.75)

    def test_empty_input(self):
        assert accuracy([], []) == 0.0


class TestCalibrationError:
    """Tests for calibration error."""

    def test_perfect_calibration(self):
        """Perfectly calibrated predictions should give ECE ≈ 0."""
        # Use 100 samples at 0.9 for class H, with 90% actually being H
        y_true = ["H"] * 90 + ["D"] * 5 + ["A"] * 5
        probs = [(0.9, 0.05, 0.05)] * 100
        ece = calibration_error(y_true, probs)
        # 90% are H, predicted 0.9 for H -> ECE ≈ |0.9 - 0.9| = 0
        assert ece["H"] == pytest.approx(0.0, abs=0.02)

    def test_miscalibrated(self):
        """Overconfident predictions should give positive ECE."""
        y_true = ["A"] * 10  # Actual is always A
        probs = [(0.9, 0.05, 0.05)] * 10  # But predicts H with 0.9
        ece = calibration_error(y_true, probs)
        assert ece["H"] > 0.0

    def test_returns_all_classes(self):
        y_true = ["H"]
        probs = [(0.4, 0.3, 0.3)]
        ece = calibration_error(y_true, probs)
        assert set(ece.keys()) == {"H", "D", "A"}


class TestComputeAllMetrics:
    """Tests for the aggregated compute_all_metrics function."""

    def test_returns_all_fields(self):
        y_true = ["H", "D", "A"]
        probs = [(0.5, 0.3, 0.2), (0.3, 0.4, 0.3), (0.2, 0.3, 0.5)]
        metrics = compute_all_metrics(y_true, probs, "test_model")
        assert metrics.model_name == "test_model"
        assert metrics.n_matches == 3
        assert 0 <= metrics.brier_score <= 2
        assert metrics.log_loss > 0
        assert 0 <= metrics.rps <= 1
        assert 0 <= metrics.accuracy <= 1
        assert set(metrics.calibration_errors.keys()) == {"H", "D", "A"}
        assert len(metrics.reliability_table) > 0
