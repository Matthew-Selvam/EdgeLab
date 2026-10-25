"""Evaluation metrics for football match prediction models.

Implements:
- Multiclass Brier score
- Multiclass log loss
- Accuracy
- Reliability/calibration summaries
- Ranked Probability Score (RPS)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Class ordering for RPS: away, draw, home (ordinal: away < draw < home)
_CLASS_ORDER = ["A", "D", "H"]


@dataclass
class ModelMetrics:
    """Container for model evaluation metrics.

    Attributes:
        model_name: Name of the model.
        n_matches: Number of evaluated matches.
        brier_score: Multiclass Brier score (lower is better).
        log_loss: Multiclass log loss (lower is better).
        rps: Ranked Probability Score (lower is better).
        accuracy: Fraction of matches where argmax prediction matches outcome.
        calibration_errors: Per-class expected calibration error (ECE).
        reliability_table: Per-bin reliability data for calibration plots.
    """

    model_name: str
    n_matches: int
    brier_score: float
    log_loss: float
    rps: float
    accuracy: float
    calibration_errors: dict[str, float]  # class -> ECE
    reliability_table: list[dict]  # bin summaries


def _to_class_vector(outcome: str) -> np.ndarray:
    """Convert an outcome label (H/D/A) to a one-hot vector [H, D, A]."""
    vec = np.zeros(3)
    idx = {"H": 0, "D": 1, "A": 2}
    vec[idx[outcome]] = 1.0
    return vec


def _proba_vector(p_home: float, p_draw: float, p_away: float) -> np.ndarray:
    """Convert probabilities to a vector [p_home, p_draw, p_away]."""
    return np.array([p_home, p_draw, p_away])


def _validate_inputs(y_true: list[str], probs: list[tuple[float, float, float]]) -> None:
    if len(y_true) != len(probs):
        raise ValueError("outcomes and probability rows must have equal length")
    if set(y_true) - {"H", "D", "A"}:
        raise ValueError("outcomes must be H, D or A")
    for values in probs:
        if len(values) != 3 or not all(math.isfinite(x) and 0 <= x <= 1 for x in values):
            raise ValueError("probabilities must be finite H/D/A values in [0, 1]")
        if not math.isclose(sum(values), 1.0, abs_tol=1e-9):
            raise ValueError("probabilities must sum to one")


def brier_score_multiclass(
    y_true: list[str],
    probs: list[tuple[float, float, float]],
) -> float:
    """Compute the multiclass Brier score.

    For each match, the Brier score is:
        sum_k (p_k - y_k)^2
    where p_k is the predicted probability for class k and y_k is 1 if
    class k is the actual outcome, 0 otherwise.

    The returned value is the average over all matches.  Range: [0, 2]
    for 3 classes.  Lower is better.

    Parameters
    ----------
    y_true:
        List of actual outcomes ('H', 'D', 'A').
    probs:
        List of (p_home, p_draw, p_away) tuples.

    Returns
    -------
    float
        Mean Brier score.
    """
    n = len(y_true)
    _validate_inputs(y_true, probs)
    if n == 0:
        return 0.0

    total = 0.0
    for actual, (ph, pd, pa) in zip(y_true, probs):
        y_vec = _to_class_vector(actual)
        p_vec = _proba_vector(ph, pd, pa)
        total += np.sum((p_vec - y_vec) ** 2)

    return float(total / n)


def log_loss_multiclass(
    y_true: list[str],
    probs: list[tuple[float, float, float]],
    epsilon: float = 1e-15,
) -> float:
    """Compute the multiclass log loss.

    For each match, the log loss is:
        -log(p_actual)
    where p_actual is the predicted probability for the actual outcome,
    clamped to [epsilon, 1 - epsilon] to prevent numerical issues.

    The returned value is the average over all matches.  Lower is better.

    Parameters
    ----------
    y_true:
        List of actual outcomes ('H', 'D', 'A').
    probs:
        List of (p_home, p_draw, p_away) tuples.
    epsilon:
        Clamping floor for numerical stability.

    Returns
    -------
    float
        Mean log loss.
    """
    n = len(y_true)
    _validate_inputs(y_true, probs)
    if n == 0:
        return 0.0

    idx_map = {"H": 0, "D": 1, "A": 2}
    total = 0.0
    for actual, (ph, pd, pa) in zip(y_true, probs):
        p_actual = [ph, pd, pa][idx_map[actual]]
        p_clamped = max(min(p_actual, 1.0 - epsilon), epsilon)
        total += -math.log(p_clamped)

    return float(total / n)


def rps_score(
    y_true: list[str],
    probs: list[tuple[float, float, float]],
) -> float:
    """Compute the Ranked Probability Score.

    RPS is a proper scoring rule for ranked (ordinal) outcomes.  We use
    the ordering A < D < H (away < draw < home).

    For cumulative probabilities:
        CP_k = sum_{j<=k} p_j  (for k=0,1,2)
        CH_k = sum_{j<=k} y_j  (for k=0,1,2)

    RPS = (1/(K-1)) * sum_k (CP_k - CH_k)^2

    where K=3 (number of classes), so divisor = 2.

    Parameters
    ----------
    y_true:
        List of actual outcomes ('H', 'D', 'A').
    probs:
        List of (p_home, p_draw, p_away) tuples.

    Returns
    -------
    float
        Mean RPS.  Range: [0, 1] for 3 classes.  Lower is better.
    """
    n = len(y_true)
    _validate_inputs(y_true, probs)
    if n == 0:
        return 0.0

    k = len(_CLASS_ORDER)  # 3
    idx_map = {c: i for i, c in enumerate(_CLASS_ORDER)}  # A=0, D=1, H=2

    total = 0.0
    for actual, (ph, pd, pa) in zip(y_true, probs):
        # Order probabilities as [p_A, p_D, p_H] to match _CLASS_ORDER
        p_vec = np.array([pa, pd, ph])
        y_vec = np.zeros(k)
        y_vec[idx_map[actual]] = 1.0

        # Cumulative sums
        cp = np.cumsum(p_vec)
        cy = np.cumsum(y_vec)

        total += np.sum((cp - cy) ** 2) / (k - 1)

    return float(total / n)


def accuracy(
    y_true: list[str],
    prob_lists: list[tuple[float, float, float]],
) -> float:
    """Compute classification accuracy.

    For each match, the predicted class is argmax(p_home, p_draw, p_away),
    mapped back to H/D/A.  Accuracy is the fraction where this matches
    the actual outcome.

    Parameters
    ----------
    y_true:
        List of actual outcomes ('H', 'D', 'A').
    prob_lists:
        List of (p_home, p_draw, p_away) tuples.

    Returns
    -------
    float
        Accuracy in [0, 1].
    """
    n = len(y_true)
    _validate_inputs(y_true, prob_lists)
    if n == 0:
        return 0.0

    correct = 0
    for actual, (ph, pd, pa) in zip(y_true, prob_lists):
        probs = np.array([ph, pd, pa])
        pred_idx = np.argmax(probs)
        pred_class = "HDA"[pred_idx]
        if pred_class == actual:
            correct += 1

    return correct / n


def calibration_error(
    y_true: list[str],
    prob_lists: list[tuple[float, float, float]],
    n_bins: int = 10,
) -> dict[str, float]:
    """Compute expected calibration error (ECE) per class.

    Partitions predicted probabilities into equal-width bins and compares
    the mean predicted probability with the observed frequency for each
    class separately (one-vs-rest).

    Parameters
    ----------
    y_true:
        List of actual outcomes ('H', 'D', 'A').
    prob_lists:
        List of (p_home, p_draw, p_away) tuples.
    n_bins:
        Number of equal-width bins on [0, 1].

    Returns
    -------
    dict
        ``{"H": ece, "D": ece, "A": ece}``
    """
    classes = ["H", "D", "A"]
    idx_map = {"H": 0, "D": 1, "A": 2}
    n = len(y_true)
    _validate_inputs(y_true, prob_lists)
    if n == 0:
        return {"H": 0.0, "D": 0.0, "A": 0.0}

    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece_results = {}

    for cls in classes:
        cls_idx = idx_map[cls]
        # Per-match predicted prob for this class and whether it's the outcome
        probs_cls = np.array([p[cls_idx] for p in prob_lists])
        outcomes_cls = np.array([1.0 if y == cls else 0.0 for y in y_true])

        ece = 0.0
        for i in range(n_bins):
            mask = (probs_cls >= bin_edges[i]) & (probs_cls < bin_edges[i + 1])
            if i == n_bins - 1:
                mask = (probs_cls >= bin_edges[i]) & (probs_cls <= bin_edges[i + 1])
            if mask.sum() > 0:
                mean_pred = probs_cls[mask].mean()
                mean_actual = outcomes_cls[mask].mean()
                ece += (mask.sum() / n) * abs(mean_pred - mean_actual)

        ece_results[cls] = float(ece)

    return ece_results


def reliability_table(
    y_true: list[str],
    prob_lists: list[tuple[float, float, float]],
    n_bins: int = 10,
) -> list[dict]:
    """Generate a reliability table for calibration analysis.

    Bins the predicted probabilities for each class and reports the
    mean predicted probability, observed frequency, and count per bin.

    Parameters
    ----------
    y_true:
        List of actual outcomes ('H', 'D', 'A').
    prob_lists:
        List of (p_home, p_draw, p_away) tuples.
    n_bins:
        Number of equal-width bins.

    Returns
    -------
    list of dict
        Each dict: ``{class, bin_low, bin_high, mean_pred, obs_freq, count}``.
    """
    _validate_inputs(y_true, prob_lists)
    classes = ["H", "D", "A"]
    idx_map = {"H": 0, "D": 1, "A": 2}
    bin_edges = np.linspace(0, 1, n_bins + 1)
    rows: list[dict] = []

    for cls in classes:
        cls_idx = idx_map[cls]
        probs_cls = np.array([p[cls_idx] for p in prob_lists])
        outcomes_cls = np.array([1.0 if y == cls else 0.0 for y in y_true])

        for i in range(n_bins):
            mask = (probs_cls >= bin_edges[i]) & (probs_cls < bin_edges[i + 1])
            if i == n_bins - 1:
                mask = (probs_cls >= bin_edges[i]) & (probs_cls <= bin_edges[i + 1])
            count = int(mask.sum())
            if count > 0:
                mean_pred = float(probs_cls[mask].mean())
                obs_freq = float(outcomes_cls[mask].mean())
            else:
                mean_pred = 0.0
                obs_freq = 0.0

            rows.append({
                "class": cls,
                "bin_low": round(float(bin_edges[i]), 3),
                "bin_high": round(float(bin_edges[i + 1]), 3),
                "mean_pred": round(mean_pred, 6),
                "obs_freq": round(obs_freq, 6),
                "count": count,
            })

    return rows


def compute_all_metrics(
    y_true: list[str],
    prob_lists: list[tuple[float, float, float]],
    model_name: str,
    n_bins: int = 10,
) -> ModelMetrics:
    """Compute all evaluation metrics for a set of predictions.

    Parameters
    ----------
    y_true:
        List of actual outcomes.
    prob_lists:
        List of (p_home, p_draw, p_away) tuples.
    model_name:
        Name of the model for the metrics object.
    n_bins:
        Number of bins for calibration analysis.

    Returns
    -------
    ModelMetrics
        Aggregated metrics.
    """
    return ModelMetrics(
        model_name=model_name,
        n_matches=len(y_true),
        brier_score=brier_score_multiclass(y_true, prob_lists),
        log_loss=log_loss_multiclass(y_true, prob_lists),
        rps=rps_score(y_true, prob_lists),
        accuracy=accuracy(y_true, prob_lists),
        calibration_errors=calibration_error(y_true, prob_lists, n_bins),
        reliability_table=reliability_table(y_true, prob_lists, n_bins),
    )
