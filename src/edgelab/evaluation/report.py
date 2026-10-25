"""Evaluation and reporting module.

Computes metrics for each model on each split (validation, test) and
generates machine-readable metrics and a human-readable comparison report.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from ..evaluation.metrics import ModelMetrics, compute_all_metrics
from ..data.ingest import PROCESSED_DATA_DIR

logger = logging.getLogger(__name__)
MODEL_NAMES = ("baseline", "elo_logistic", "poisson")


@dataclass
class ModelComparison:
    """Metrics comparison between two models on a given split.

    Attributes:
        baseline_metrics: Metrics for the baseline model.
        elo_metrics: Metrics for the Elo logistic model.
        brier_diff: Difference in Brier score (baseline - elo). Positive = Elo better.
        log_loss_diff: Difference in log loss (baseline - elo). Positive = Elo better.
        accuracy_diff: Difference in accuracy (elo - baseline). Positive = Elo better.
    """

    baseline_brier: float
    baseline_log_loss: float
    baseline_accuracy: float
    baseline_rps: float
    elo_brier: float
    elo_log_loss: float
    elo_accuracy: float
    elo_rps: float
    brier_diff: float  # baseline - elo (positive = elo better)
    log_loss_diff: float  # baseline - elo (positive = elo better)
    accuracy_diff: float  # elo - baseline (positive = elo better)
    rps_diff: float  # baseline - elo (positive = elo better)


def load_predictions(pred_path: Path) -> pd.DataFrame:
    """Load the predictions CSV."""
    return pd.read_csv(pred_path)


def evaluate_model(
    predictions: pd.DataFrame,
    split: str,
    model_name: str,
) -> ModelMetrics:
    """Compute all metrics for a single model on a single split.

    Parameters
    ----------
    predictions:
        DataFrame of prediction records.
    split:
        Evaluation split ('validation' or 'test').
    model_name:
        Model identifier ('baseline' or 'elo_logistic').

    Returns
    -------
    ModelMetrics
        Computed metrics.
    """
    subset = predictions[
        (predictions["model"] == model_name) &
        (predictions["split"] == split)
    ].copy()

    if len(subset) == 0:
        logger.warning("No predictions for model=%s, split=%s", model_name, split)

    y_true = subset["actual_outcome"].tolist()
    prob_lists = list(zip(
        subset["p_home"].tolist(),
        subset["p_draw"].tolist(),
        subset["p_away"].tolist(),
    ))

    return compute_all_metrics(y_true, prob_lists, model_name)


def compare_models(predictions: pd.DataFrame, split: str) -> tuple[ModelComparison, dict]:
    """Compare all models on exactly the same fixtures on a given split.

    Parameters
    ----------
    predictions:
        DataFrame of all prediction records.
    split:
        Evaluation split ('validation' or 'test').

    Returns
    -------
    tuple
        (ModelComparison with metric differences, detailed dict with per-class
        calibration and reliability data)
    """
    subset = predictions[predictions["split"] == split]
    if set(subset["model"]) != set(MODEL_NAMES):
        raise ValueError(f"all three models need predictions for {split}")
    by_model = {}
    for model_name in MODEL_NAMES:
        rows = subset[subset["model"] == model_name]
        if rows["match_id"].duplicated().any():
            raise ValueError("duplicate model prediction for a fixture")
        by_model[model_name] = rows.set_index("match_id").sort_index()
    reference = by_model["baseline"]
    for model_name in MODEL_NAMES[1:]:
        other = by_model[model_name]
        if set(reference.index) != set(other.index):
            raise ValueError("models must be compared on exactly the same fixtures")
        for column in ("actual_outcome", "match_date", "season"):
            if not reference[column].equals(other[column]):
                raise ValueError(f"models disagree on fixture {column}")
    scored = {name: evaluate_model(predictions, split, name) for name in MODEL_NAMES}
    baseline_metrics = scored["baseline"]
    elo_metrics = scored["elo_logistic"]

    comparison = ModelComparison(
        baseline_brier=baseline_metrics.brier_score,
        baseline_log_loss=baseline_metrics.log_loss,
        baseline_accuracy=baseline_metrics.accuracy,
        baseline_rps=baseline_metrics.rps,
        elo_brier=elo_metrics.brier_score,
        elo_log_loss=elo_metrics.log_loss,
        elo_accuracy=elo_metrics.accuracy,
        elo_rps=elo_metrics.rps,
        # Positive diff = baseline worse = elo better
        brier_diff=baseline_metrics.brier_score - elo_metrics.brier_score,
        log_loss_diff=baseline_metrics.log_loss - elo_metrics.log_loss,
        accuracy_diff=elo_metrics.accuracy - baseline_metrics.accuracy,
        rps_diff=baseline_metrics.rps - elo_metrics.rps,
    )

    detailed = {
        name: {
            "brier_score": value.brier_score,
            "log_loss": value.log_loss,
            "accuracy": value.accuracy,
            "rps": value.rps,
            "n_matches": value.n_matches,
            "calibration_errors": value.calibration_errors,
            "reliability_table": value.reliability_table,
        }
        for name, value in scored.items()
    }

    return comparison, detailed


def generate_report(
    predictions: pd.DataFrame,
    output_dir: Optional[Path] = None,
) -> str:
    """Generate a full evaluation report.

    Parameters
    ----------
    predictions:
        DataFrame of all prediction records.
    output_dir:
        Directory for output files. Defaults to data/processed/.

    Returns
    -------
    str
        Markdown report content.
    """
    if output_dir is None:
        output_dir = PROCESSED_DATA_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # Also save the predictions to a canonical location
    pred_path = output_dir / "predictions" / "predictions.csv"
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "predictions" / "backtest_summary.json"
    run_metadata = json.loads(summary_path.read_text()) if summary_path.exists() else None

    # Compute per-split comparisons
    validation_comparison, validation_detailed = compare_models(predictions, "validation")
    test_comparison, test_detailed = compare_models(predictions, "test")
    predictions.to_csv(pred_path, index=False)

    # Build machine-readable metrics
    metrics = {
        "run_metadata": run_metadata,
        "validation": validation_detailed,
        "test": test_detailed,
        "comparison": {
            "validation": {
                "brier_diff": validation_comparison.brier_diff,
                "log_loss_diff": validation_comparison.log_loss_diff,
                "accuracy_diff": validation_comparison.accuracy_diff,
                "rps_diff": validation_comparison.rps_diff,
            },
            "test": {
                "brier_diff": test_comparison.brier_diff,
                "log_loss_diff": test_comparison.log_loss_diff,
                "accuracy_diff": test_comparison.accuracy_diff,
                "rps_diff": test_comparison.rps_diff,
            },
        },
        "poisson_vs_baseline": {
            split: {
                "brier_diff": metrics_for_split["baseline"]["brier_score"] - metrics_for_split["poisson"]["brier_score"],
                "log_loss_diff": metrics_for_split["baseline"]["log_loss"] - metrics_for_split["poisson"]["log_loss"],
                "accuracy_diff": metrics_for_split["poisson"]["accuracy"] - metrics_for_split["baseline"]["accuracy"],
            }
            for split, metrics_for_split in (("validation", validation_detailed), ("test", test_detailed))
        },
    }

    metrics_path = output_dir / "predictions" / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # Build Markdown report
    lines = []
    lines.append("# EdgeLab Model Evaluation Report\n")
    lines.append("## Overview\n")
    lines.append(f"- **Total predictions:** {len(predictions)}")
    lines.append(f"- **Models compared:** {sorted(predictions['model'].unique())}")
    lines.append(f"- **Splits:** {sorted(predictions['split'].unique())}")
    lines.append(f"- **Training seasons:** 2015-16 through 2022-23")
    lines.append(f"- **Validation season:** 2023-24")
    lines.append(f"- **Test season:** 2024-25\n")
    if run_metadata:
        lines.append(f"- **Canonical CSV SHA-256:** `{run_metadata['source_sha256']}`")
        lines.append(f"- **Run settings:** `{json.dumps(run_metadata['config'], sort_keys=True)}`\n")
        for note in run_metadata.get("data_quality_notes", []):
            lines.append(f"- **Data quality:** {note}")
        correction = run_metadata.get("date_corrections")
        if correction and correction["count"]:
            lines.append(f"- **Date correction:** {correction['count']} source dates normalized; fixture-level provenance in `{correction['audit_file']}`.")
            lines.append(f"  Provider reference: {correction['reference']}")
    lines.append(f"- **Comparison coverage:** All three models forecast the same {validation_detailed['baseline']['n_matches']} validation and {test_detailed['baseline']['n_matches']} test fixtures.\n")

    for split_name in ["validation", "test"]:
        lines.append(f"## {split_name.capitalize()} Results\n")
        split_preds = predictions[predictions["split"] == split_name]
        for model_name in MODEL_NAMES:
            model_preds = split_preds[split_preds["model"] == model_name]
            if len(model_preds) == 0:
                continue
            metrics_obj = evaluate_model(predictions, split_name, model_name)
            lines.append(f"### {model_name}\n")
            lines.append(f"| Metric | Value |")
            lines.append(f"|--------|-------|")
            lines.append(f"| Matches | {metrics_obj.n_matches} |")
            lines.append(f"| Brier Score | {metrics_obj.brier_score:.6f} |")
            lines.append(f"| Log Loss | {metrics_obj.log_loss:.6f} |")
            lines.append(f"| RPS | {metrics_obj.rps:.6f} |")
            lines.append(f"| Accuracy | {metrics_obj.accuracy:.4f} |")
            lines.append("")

            # Per-class calibration
            lines.append("#### Calibration (Expected Calibration Error per class)\n")
            lines.append("| Class | ECE |")
            lines.append("|-------|-----|")
            for cls, ece in metrics_obj.calibration_errors.items():
                lines.append(f"| {cls} | {ece:.6f} |")
            lines.append("")

    # Comparison table
    lines.append("## Model Comparison\n")
    lines.append("| Split | Model | Matches | Brier ↓ | Log loss ↓ | Accuracy ↑ | RPS ↓ |")
    lines.append("|-------|-------|--------:|--------:|-----------:|-----------:|------:|")
    for split_name, details in (("validation", validation_detailed), ("test", test_detailed)):
        for model_name in MODEL_NAMES:
            value = details[model_name]
            lines.append(
                f"| {split_name} | {model_name} | {value['n_matches']} | "
                f"{value['brier_score']:.6f} | {value['log_loss']:.6f} | "
                f"{value['accuracy']:.4f} | {value['rps']:.6f} |"
            )

    lines.append("")
    lines.append("## Interpretation\n")
    brier_diff = test_comparison.brier_diff
    log_loss_diff = test_comparison.log_loss_diff
    acc_diff = test_comparison.accuracy_diff

    if brier_diff > 0:
        lines.append(f"- On the **test set**, the Elo logistic model has a **lower** Brier score "
                      f"({test_comparison.elo_brier:.6f} vs {test_comparison.baseline_brier:.6f}), "
                      f"meaning better probabilistic predictions (diff={brier_diff:+.6f}).")
    else:
        lines.append(f"- On the **test set**, the baseline has a **lower or equal** Brier score "
                      f"({test_comparison.baseline_brier:.6f} vs {test_comparison.elo_brier:.6f}), "
                      f"indicating lower squared probability error (diff={brier_diff:+.6f}).")

    if log_loss_diff > 0:
        lines.append(f"- The Elo model also has **lower** log loss ({test_comparison.elo_log_loss:.6f} "
                      f"vs {test_comparison.baseline_log_loss:.6f}), suggesting it assigns "
                      f"higher probability to actual outcomes.")
    else:
        lines.append(f"- The baseline has **lower or equal** log loss ({test_comparison.baseline_log_loss:.6f} "
                      f"vs {test_comparison.elo_log_loss:.6f}).")

    lines.append(f"- Accuracy difference (Elo - baseline): {acc_diff:+.4f}")
    poisson_test = test_detailed["poisson"]
    baseline_test = test_detailed["baseline"]
    lines.append(
        "- Poisson vs baseline on the test set: Brier "
        f"{poisson_test['brier_score']:.6f} vs {baseline_test['brier_score']:.6f}; "
        f"log loss {poisson_test['log_loss']:.6f} vs {baseline_test['log_loss']:.6f}; "
        f"accuracy {poisson_test['accuracy']:.4f} vs {baseline_test['accuracy']:.4f}."
    )
    baseline_ece = test_detailed["baseline"]["calibration_errors"]
    elo_ece = test_detailed["elo_logistic"]["calibration_errors"]
    if all(elo_ece[c] > baseline_ece[c] for c in ("H", "D", "A")):
        lines.append("- Elo's 10-bin expected calibration error is higher for H, D and A on the test set; lower Brier/log loss does not imply better calibration in every sense.")
    poisson_ece = poisson_test["calibration_errors"]
    if all(poisson_ece[c] > baseline_ece[c] for c in ("H", "D", "A")):
        lines.append("- Poisson's 10-bin expected calibration error is also higher for H, D and A on the test set, despite its lower Brier score and log loss.")
    lines.append("- These single-season differences are descriptive; no uncertainty interval or significance test is included.")
    lines.append("")

    lines.append("## Methodology Notes\n")
    settings = run_metadata["config"] if run_metadata else {}
    alpha = settings.get("baseline_smoothing_alpha", 1.0)
    initial = settings.get("elo_initial_rating", 1500.0)
    k = settings.get("elo_k_factor", 20.0)
    advantage = settings.get("elo_home_advantage", 60.0)
    draw_scale = settings.get("elo_draw_scale", 350.0)
    home_scale = settings.get("elo_home_scale", 280.0)
    lines.append(f"- **Baseline**: Development-set class frequencies with additive smoothing alpha={alpha}, frozen for both evaluation seasons.")
    lines.append(f"- **Elo**: Fixed, predeclared heuristic with initial rating {initial}, K={k}, home advantage {advantage}.")
    lines.append(f"  With d = pre-match home rating + {advantage} - away rating, p_draw = 1/(1+exp(|d|/{draw_scale}));")
    lines.append(f"  p_home = (1-p_draw)/(1+exp(-d/{home_scale})); p_away = 1-p_draw-p_home.")
    lines.append("  This conversion is not fitted logistic regression; its scales were not tuned on the test set.")
    prior = settings.get("poisson_prior_matches", 5.0)
    goal_cap = settings.get("poisson_max_goals", 30)
    tolerance = settings.get("poisson_tail_tolerance", 1e-10)
    lines.append(f"- **Poisson**: Independent home/away goal counts. Development league home/away means are frozen; team home/away scoring and conceding means receive {prior} pseudo-matches at the relevant league mean.")
    lines.append("  Expected goals multiply the relevant scoring and conceding factors with the league venue mean. Team totals expand after each completed date.")
    lines.append(f"  Joint goal probabilities are summed into H/D/A, with each marginal tail <= {tolerance:g} and a {goal_cap}-goal cap; retained mass is normalized.")
    lines.append("- **Forecast date**: date resolution only. prediction_timestamp stores the forecast date;")
    lines.append("  information_cutoff_date is the latest date whose results were eligible.")
    lines.append("- **Temporal integrity**: Elo ratings and Poisson team counts update from completed prior dates, including during evaluation.")
    lines.append("  Every match on one date is forecast before any result from that date is applied.")
    lines.append("  No model settings are selected from held-out test results.")
    lines.append("- **Metrics**: Multiclass Brier score (lower is better), log loss (lower is better),")
    lines.append("  Ranked Probability Score (lower is better), accuracy (higher is better).")
    lines.append("- **Calibration**: Expected Calibration Error computed with 10 equal-width bins per class.")

    report = "\n".join(lines)

    report_path = output_dir / "predictions" / "evaluation_report.md"
    with open(report_path, "w") as f:
        f.write(report)

    logger.info("Evaluation report written to %s", report_path)
    return report


def main() -> int:
    """Entry point for ``python -m edgelab.evaluation.report``."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    pred_path = PROCESSED_DATA_DIR / "predictions" / "predictions.csv"
    if not pred_path.exists():
        logger.error("Predictions file not found. Run backtest first.")
        return 1

    predictions = load_predictions(pred_path)
    report = generate_report(predictions)
    print(report[:2000])
    return 0


if __name__ == "__main__":
    main()
