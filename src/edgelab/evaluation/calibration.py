"""Frozen Poisson temperature-calibration experiment on saved forecasts.

Fit only on out-of-sample development walk-forward predictions. Validation
and held-out test are read and assessed after the one-parameter fit is fixed.
The original prediction and report artefacts are never modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..data.ingest import PROCESSED_DATA_DIR
from .backtest import TRAINING_SEASONS, load_matches
from .metrics import compute_all_metrics

CLASSES = ("H", "D", "A")
PROBABILITY_COLUMNS = ("p_home", "p_draw", "p_away")
DEVELOPMENT_SEASONS = tuple(sorted(TRAINING_SEASONS)[1:])
EPSILON = 1e-15
MAX_INVERSE_TEMPERATURE = 1e6
BOOTSTRAP_REPLICATES = 1000
BOOTSTRAP_SEED = 20261021


@dataclass(frozen=True)
class TemperatureFit:
    inverse_temperature: float
    temperature: float | None  # null represents the uniform-limit T=infinity
    n_matches: int
    log_loss_raw: float
    log_loss_fitted: float


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validated_probabilities(values: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(values, dtype=float)
    if probabilities.ndim != 2 or probabilities.shape[1] != 3 or len(probabilities) == 0:
        raise ValueError("expected a nonempty N×3 H/D/A probability matrix")
    if (not np.isfinite(probabilities).all() or (probabilities < 0).any()
            or (probabilities > 1).any()
            or not np.allclose(probabilities.sum(axis=1), 1, rtol=0, atol=1e-9)):
        raise ValueError("probabilities must be finite, nonnegative, and sum to one")
    return probabilities


def _outcome_indices(outcomes: list[str] | np.ndarray) -> np.ndarray:
    labels = np.asarray(outcomes)
    if labels.ndim != 1 or len(labels) == 0 or not np.isin(labels, CLASSES).all():
        raise ValueError("outcomes must be nonempty H/D/A labels")
    return np.array([CLASSES.index(label) for label in labels], dtype=int)


def temperature_probabilities(probabilities: np.ndarray, inverse_temperature: float) -> np.ndarray:
    """Return softmax(beta * log(max(p, 1e-15))) in H/D/A order."""
    values = _validated_probabilities(probabilities)
    beta = float(inverse_temperature)
    if not math.isfinite(beta) or beta < 0 or beta > MAX_INVERSE_TEMPERATURE:
        raise ValueError("inverse temperature must be finite and within the declared bounds")
    logits = beta * np.log(np.maximum(values, EPSILON))
    shifted = logits - logits.max(axis=1, keepdims=True)
    weights = np.exp(shifted)
    scaled = weights / weights.sum(axis=1, keepdims=True)
    if not np.isfinite(scaled).all():
        raise ValueError("temperature scaling produced nonfinite probabilities")
    return scaled


def fit_temperature(probabilities: np.ndarray, outcomes: list[str] | np.ndarray) -> TemperatureFit:
    """Minimize mean multiclass negative log likelihood in inverse temperature.

    The objective is convex for beta >= 0 because the input log-probabilities
    are fixed. Bisection finds its monotone derivative's zero; if a boundary
    wins, it is recorded explicitly. There is no validation/test selection.
    """
    values = _validated_probabilities(probabilities)
    y = _outcome_indices(outcomes)
    if len(values) != len(y):
        raise ValueError("probabilities and outcomes must have equal length")
    logs = np.log(np.maximum(values, EPSILON))
    actual_logs = logs[np.arange(len(y)), y]

    def objective_and_derivative(beta: float) -> tuple[float, float]:
        logits = beta * logs
        maximum = logits.max(axis=1, keepdims=True)
        shifted = logits - maximum
        log_normalizer = np.log(np.exp(shifted).sum(axis=1)) + maximum[:, 0]
        weights = np.exp(shifted)
        q = weights / weights.sum(axis=1, keepdims=True)
        loss = float(np.mean(log_normalizer - beta * actual_logs))
        derivative = float(np.mean(np.sum(q * logs, axis=1) - actual_logs))
        return loss, derivative

    raw_loss, at_identity = objective_and_derivative(1.0)
    if abs(at_identity) < 1e-13:
        beta = 1.0
    else:
        low = 0.0
        if objective_and_derivative(0.0)[1] >= 0:
            beta = 0.0
        else:
            high = 1.0
            while objective_and_derivative(high)[1] < 0 and high < MAX_INVERSE_TEMPERATURE:
                high = min(2 * high, MAX_INVERSE_TEMPERATURE)
            for _ in range(80):
                midpoint = (low + high) / 2
                if objective_and_derivative(midpoint)[1] < 0:
                    low = midpoint
                else:
                    high = midpoint
            beta = (low + high) / 2
    fitted_loss = objective_and_derivative(beta)[0]
    return TemperatureFit(beta, 1 / beta if beta > 0 else None,
                          len(y), raw_loss, fitted_loss)


def _check_rows(rows: pd.DataFrame, canonical: pd.DataFrame, seasons: tuple[str, ...],
                split: str, initial_cutoffs: dict[str, str]) -> pd.DataFrame:
    """Validate source identity, date-batched cutoffs and one row per fixture."""
    required = {"match_id", "model", "season", "match_date", "home_team", "away_team",
                "actual_outcome", "split", "prediction_timestamp", "information_cutoff_date",
                *PROBABILITY_COLUMNS}
    if required - set(rows.columns):
        raise ValueError(f"missing prediction columns: {sorted(required - set(rows.columns))}")
    if set(rows["model"]) != {"poisson"} or set(rows["split"]) != {split}:
        raise ValueError("only raw Poisson predictions from the expected split are eligible")
    if set(rows["season"]) != set(seasons) or rows["match_id"].duplicated().any():
        raise ValueError("unexpected seasons or duplicate Poisson predictions")
    expected = canonical[canonical["season"].isin(seasons)].set_index("match_id")
    if len(rows) != len(expected) or set(rows["match_id"]) != set(expected.index):
        raise ValueError("predictions do not cover exactly the expected canonical fixtures")
    ordered = rows.sort_values(["match_date", "match_id"]).reset_index(drop=True)
    source = expected.loc[ordered["match_id"]]
    for column, source_column in (("season", "season"), ("home_team", "home_team"),
                                  ("away_team", "away_team"), ("actual_outcome", "outcome")):
        if ordered[column].tolist() != source[source_column].tolist():
            raise ValueError(f"prediction {column} disagrees with canonical fixtures")
    canonical_dates = source["match_date"].dt.strftime("%Y-%m-%d").tolist()
    if ordered["match_date"].tolist() != canonical_dates or ordered["prediction_timestamp"].tolist() != canonical_dates:
        raise ValueError("prediction dates disagree with canonical fixtures")
    _validated_probabilities(ordered[list(PROBABILITY_COLUMNS)].to_numpy(dtype=float))
    for season in seasons:
        period = ordered[ordered["season"] == season]
        cutoff = initial_cutoffs[season]
        for date, batch in period.groupby("match_date", sort=True):
            if date <= cutoff or set(batch["information_cutoff_date"]) != {cutoff}:
                raise ValueError("forecast cutoff violates previous-date batching")
            cutoff = date
    return ordered


def load_development_oos(canonical: pd.DataFrame, csv_path: Path,
                         robustness_dir: Path) -> tuple[pd.DataFrame, dict]:
    """Accept only the seven audited Poisson walk-forward development folds."""
    metrics_path = robustness_dir / "metrics.json"
    prediction_path = robustness_dir / "walk_forward_predictions.csv"
    metadata = json.loads(metrics_path.read_text(encoding="utf-8"))
    if metadata["protocol"]["source_sha256"] != _sha(csv_path):
        raise ValueError("robustness predictions use another canonical source")
    if metadata["protocol"]["splits"] != {
        "development": sorted(TRAINING_SEASONS), "validation": ["2023-24"],
        "held_out_test": ["2024-25"]}:
        raise ValueError("robustness split protocol differs from the frozen definition")
    folds = metadata["development"]["folds"]
    if set(folds) != set(DEVELOPMENT_SEASONS):
        raise ValueError("incomplete development walk-forward fold metadata")
    full = pd.read_csv(prediction_path)
    if set(full["model"]) != {"baseline", "elo_logistic", "poisson"}:
        raise ValueError("walk-forward source is missing model provenance")
    poisson = full[full["model"] == "poisson"].copy()
    initial_cutoffs = {}
    for index, season in enumerate(DEVELOPMENT_SEASONS, start=1):
        fold = folds[season]
        prior = sorted(TRAINING_SEASONS)[:index]
        if (fold["training_seasons"] != prior or fold["training_matches"] != 380 * index
                or fold["scored_fixtures"] != 380):
            raise ValueError("fold metadata does not describe expanding prior-season training")
        actual_cutoff = canonical[canonical["season"].isin(prior)]["match_date"].max().strftime("%Y-%m-%d")
        if fold["training_cutoff_date"] != actual_cutoff:
            raise ValueError("fold training cutoff disagrees with canonical history")
        initial_cutoffs[season] = actual_cutoff
    checked = _check_rows(poisson, canonical, DEVELOPMENT_SEASONS,
                          "development_walk_forward", initial_cutoffs)
    if len(checked) != 2660:
        raise ValueError("expected 2,660 out-of-sample development Poisson fixtures")
    for season in DEVELOPMENT_SEASONS:
        period = checked[checked["season"] == season]
        scored = _score(period, period[list(PROBABILITY_COLUMNS)].to_numpy(dtype=float), "poisson")
        saved = metadata["development"]["seasons"][season]["poisson"]
        for name in ("brier_score", "log_loss", "accuracy", "n_matches"):
            if not np.isclose(scored[name], saved[name], atol=1e-12):
                raise ValueError(f"walk-forward {season} Poisson metrics disagree with saved study")
        for label in CLASSES:
            if not np.isclose(scored["calibration_errors"][label],
                              saved["calibration_errors"][label], atol=1e-12):
                raise ValueError(f"walk-forward {season} Poisson reliability disagrees with saved study")
    return checked, {"predictions_sha256": _sha(prediction_path),
                     "metrics_sha256": _sha(metrics_path),
                     "season_counts": {season: 380 for season in DEVELOPMENT_SEASONS}}


def load_frozen_assessment(canonical: pd.DataFrame, csv_path: Path,
                           frozen_dir: Path, robustness_dir: Path) -> tuple[dict[str, pd.DataFrame], dict]:
    """Load the unchanged validation and test rows after the fit is complete."""
    pred_path = frozen_dir / "predictions.csv"
    metrics_path = frozen_dir / "metrics.json"
    saved = json.loads(metrics_path.read_text(encoding="utf-8"))
    robustness = json.loads((robustness_dir / "metrics.json").read_text(encoding="utf-8"))
    source_sha = _sha(csv_path)
    if saved["run_metadata"]["source_sha256"] != source_sha:
        raise ValueError("frozen evaluation uses another canonical source")
    reference = robustness["frozen_reference"]
    if (reference["predictions_sha256"] != _sha(pred_path)
            or reference["metrics_sha256"] != _sha(metrics_path)):
        raise ValueError("frozen evaluation artefacts changed after the robustness study")
    all_predictions = pd.read_csv(pred_path)
    raw = all_predictions[all_predictions["model"] == "poisson"].copy()
    development_end = canonical[canonical["season"].isin(TRAINING_SEASONS)]["match_date"].max().strftime("%Y-%m-%d")
    validation = _check_rows(raw[raw["split"] == "validation"], canonical,
                             ("2023-24",), "validation", {"2023-24": development_end})
    validation_end = validation["match_date"].max()
    test = _check_rows(raw[raw["split"] == "test"], canonical,
                       ("2024-25",), "test", {"2024-25": validation_end})
    if not (validation_end < test["match_date"].min()):
        raise ValueError("validation and held-out test are not strictly ordered")
    for split, rows in (("validation", validation), ("test", test)):
        score = compute_all_metrics(rows["actual_outcome"].tolist(),
                                    list(rows[list(PROBABILITY_COLUMNS)].itertuples(index=False, name=None)),
                                    "poisson")
        for name in ("brier_score", "log_loss", "accuracy", "n_matches"):
            if not np.isclose(getattr(score, name), saved[split]["poisson"][name], atol=1e-12):
                raise ValueError(f"frozen {split} Poisson metric changed")
        for label in CLASSES:
            if not np.isclose(score.calibration_errors[label],
                              saved[split]["poisson"]["calibration_errors"][label], atol=1e-12):
                raise ValueError(f"frozen {split} Poisson reliability changed")
    return {"validation": validation, "test": test}, {
        "predictions_sha256": _sha(pred_path), "metrics_sha256": _sha(metrics_path),
        "source_sha256": source_sha}


def _score(rows: pd.DataFrame, probabilities: np.ndarray, name: str) -> dict:
    result = compute_all_metrics(rows["actual_outcome"].tolist(),
                                 [tuple(p) for p in probabilities], name)
    return asdict(result)


def _wilson(positive: int, count: int) -> tuple[float | None, float | None]:
    if count == 0:
        return None, None
    z = 1.959963984540054
    fraction = positive / count
    centre = (fraction + z*z/(2*count)) / (1 + z*z/count)
    half = z * math.sqrt(fraction*(1-fraction)/count + z*z/(4*count*count)) / (1 + z*z/count)
    return max(0.0, centre-half), min(1.0, centre+half)


def _reliability_rows(split: str, variant: str, rows: pd.DataFrame,
                      probabilities: np.ndarray, scored: dict) -> list[dict]:
    y = _outcome_indices(rows["actual_outcome"].to_numpy())
    output = []
    edges = np.linspace(0, 1, 11)
    for item in scored["reliability_table"]:
        cls = CLASSES.index(item["class"])
        index = round(item["bin_low"] * 10)
        low, high = edges[index], edges[index + 1]
        membership = (probabilities[:, cls] >= low) & (probabilities[:, cls] < high)
        if high == 1.0:
            membership = (probabilities[:, cls] >= low) & (probabilities[:, cls] <= high)
        positive = int((y[membership] == cls).sum())
        count = int(membership.sum())
        if count != item["count"]:
            raise ValueError("reliability bin counts disagree with the shared metric implementation")
        ci_low, ci_high = _wilson(positive, count)
        output.append({"split": split, "variant": variant, **item,
                       "observed_positive": positive,
                       "obs_ci95_low": ci_low, "obs_ci95_high": ci_high})
    return output


def _sample_scores(y: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    n = len(y)
    one_hot = np.eye(3)[y]
    output = {
        "brier_score": float(np.square(probabilities-one_hot).sum(axis=1).mean()),
        "log_loss": float(-np.log(np.clip(probabilities[np.arange(n), y], EPSILON,
                                             1-EPSILON)).mean()),
        "accuracy": float((probabilities.argmax(axis=1) == y).mean()),
    }
    for cls, label in enumerate(CLASSES):
        bins = np.minimum((probabilities[:, cls] * 10).astype(int), 9)
        predicted = np.bincount(bins, weights=probabilities[:, cls], minlength=10)
        observed = np.bincount(bins, weights=one_hot[:, cls], minlength=10)
        output[f"ece_{label}"] = float(np.abs(predicted-observed).sum()/n)
    return output


def paired_date_intervals(rows: pd.DataFrame, raw: np.ndarray, calibrated: np.ndarray,
                          seed: int = BOOTSTRAP_SEED,
                          replicates: int = BOOTSTRAP_REPLICATES) -> dict:
    """Predeclared paired date bootstrap for calibrated minus raw metrics."""
    if replicates < 1:
        raise ValueError("replicates must be positive")
    y = _outcome_indices(rows["actual_outcome"].to_numpy())
    dates = rows["match_date"].to_numpy()
    blocks = [np.flatnonzero(dates == date) for date in np.unique(dates)]
    generator = np.random.default_rng(seed)
    raw_point = _sample_scores(y, raw)
    calibrated_point = _sample_scores(y, calibrated)
    values = {key: [] for key in raw_point}
    for _ in range(replicates):
        sample = np.concatenate([blocks[i] for i in generator.integers(0, len(blocks), len(blocks))])
        before = _sample_scores(y[sample], raw[sample])
        after = _sample_scores(y[sample], calibrated[sample])
        for key in values:
            values[key].append(after[key] - before[key])
    return {key: {"difference": calibrated_point[key] - raw_point[key],
                  "ci95_low": float(np.quantile(samples, .025)),
                  "ci95_high": float(np.quantile(samples, .975))}
            for key, samples in values.items()}


def _plot_reliability(rows: list[dict], split: str, season: str, n: int,
                      output_path: Path) -> None:
    figure = make_subplots(rows=1, cols=3, subplot_titles=("Home win", "Draw", "Away win"))
    palette = {"raw": ("#6b7280", "circle"), "temperature": ("#087e8b", "diamond")}
    for column, label in enumerate(CLASSES, start=1):
        figure.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                    line={"color": "#b7c0c7", "dash": "dash", "width": 1},
                                    showlegend=False, hoverinfo="skip"), row=1, col=column)
        for variant in ("raw", "temperature"):
            bins = [item for item in rows if item["split"] == split and
                    item["variant"] == variant and item["class"] == label and item["count"] > 0]
            color, symbol = palette[variant]
            figure.add_trace(go.Scatter(
                x=[item["mean_pred"] for item in bins],
                y=[item["obs_freq"] for item in bins], mode="markers+lines",
                name="Raw Poisson" if variant == "raw" else "Temperature scaled",
                legendgroup=variant, showlegend=column == 1,
                marker={"color": color, "symbol": symbol, "size": 9},
                line={"color": color, "width": 2},
                error_y={"type": "data", "symmetric": False,
                         "array": [item["obs_ci95_high"] - item["obs_freq"] for item in bins],
                         "arrayminus": [item["obs_freq"] - item["obs_ci95_low"] for item in bins],
                         "color": color, "thickness": 1, "width": 2},
                customdata=[[item["count"], item["bin_low"], item["bin_high"]] for item in bins],
                hovertemplate="Mean forecast %{x:.3f}<br>Observed %{y:.3f}"
                              "<br>n=%{customdata[0]}<br>bin [%{customdata[1]:.1f}, "
                              "%{customdata[2]:.1f})<extra></extra>"), row=1, col=column)
        figure.update_xaxes(range=[0, 1], title_text="Mean predicted probability", row=1, col=column)
        figure.update_yaxes(range=[0, 1], title_text="Observed frequency" if column == 1 else None,
                            row=1, col=column)
    figure.update_layout(title=f"{split.title()} {season} reliability · {n} fixtures",
                         template="plotly_white", width=1150, height=460,
                         margin={"t": 80, "b": 70, "l": 70, "r": 30},
                         legend={"orientation": "h", "y": -0.25})
    figure.write_html(output_path, include_plotlyjs=True, full_html=True,
                      div_id=f"edgelab-calibration-{split}")


def _report(metrics: dict) -> str:
    fit = metrics["calibrator"]
    temperature = (f"{fit['temperature']:.6f}" if fit["temperature"] is not None
                   else "infinity (uniform limit)")
    lines = ["# Poisson temperature calibration experiment", "",
             "A single inverse temperature was fitted by minimizing multiclass log loss "
             "on 2,660 genuinely out-of-sample Poisson walk-forward forecasts from "
             "2016–17 through 2022–23. The fit uses no validation or held-out test results. "
             "The original prediction files and reports remain unchanged.", "",
             f"**Fitted inverse temperature:** {fit['inverse_temperature']:.6f}; "
             f"**temperature:** {temperature}. "
             "The development-fit log loss is a training diagnostic, not an unbiased "
             "estimate of calibration benefit.", "",
             "## Predeclared assessment on identical fixtures", "",
             "Lower Brier, log loss and class-wise ECE are better; higher accuracy is better. "
             "ECE uses the existing 10 equal-width one-versus-rest bins. "
             "Positive temperature scaling preserves the most likely outcome, so accuracy "
             "should remain unchanged. Reliability HTML plots and bin counts accompany this report.", "",
             "| Split | Variant | n | Brier | Log loss | Accuracy | ECE H | ECE D | ECE A |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for split in ("validation", "test"):
        for variant in ("raw", "temperature"):
            item = metrics["assessment"][split][variant]
            ece = item["calibration_errors"]
            lines.append(f"| {split} | {variant} | {item['n_matches']} | "
                         f"{item['brier_score']:.4f} | {item['log_loss']:.4f} | "
                         f"{item['accuracy']:.3f} | {ece['H']:.3f} | {ece['D']:.3f} | {ece['A']:.3f} |")
    lines += ["", "### Outcome-specific reading", ""]
    for split in ("validation", "test"):
        before = metrics["assessment"][split]["raw"]
        after = metrics["assessment"][split]["temperature"]
        parts = []
        for label, title in (("H", "home"), ("D", "draw"), ("A", "away")):
            difference = after["calibration_errors"][label] - before["calibration_errors"][label]
            movement = "lower" if difference < -1e-12 else "higher" if difference > 1e-12 else "unchanged"
            parts.append(f"{title} ECE {movement} by {abs(difference):.4f}")
        accuracy_delta = after["accuracy"] - before["accuracy"]
        lines.append(f"**{split.title()}:** " + "; ".join(parts) +
                     f". Accuracy difference: {accuracy_delta:+.4f}. "
                     "These ECE changes describe separate outcomes, not a uniform calibration gain.")
    lines += ["", "## Paired uncertainty: calibrated minus raw", "",
              "95% percentile intervals resample dates within each split, preserving all "
              "same-date fixtures and paired raw/calibrated forecasts. They do not account "
              "for serial dependence across dates or uncertainty in the fitted temperature. "
              "The test is a single season; intervals are descriptive and unadjusted for "
              "multiple outcomes/metrics.", "",
              "| Split | Δ Brier [95%] | Δ Log loss [95%] | Δ Accuracy [95%] | Δ ECE H [95%] | Δ ECE D [95%] | Δ ECE A [95%] |",
              "|---|---|---|---|---|---|---|"]
    for split in ("validation", "test"):
        differences = metrics["assessment"][split]["paired_differences"]
        def cell(key: str) -> str:
            item = differences[key]
            return f"{item['difference']:+.4f} [{item['ci95_low']:+.4f}, {item['ci95_high']:+.4f}]"
        lines.append(f"| {split} | {cell('brier_score')} | {cell('log_loss')} | "
                     f"{cell('accuracy')} | {cell('ece_H')} | {cell('ece_D')} | {cell('ece_A')} |")
    validation = metrics["assessment"]["validation"]
    held_out = metrics["assessment"]["test"]
    if (validation["temperature"]["brier_score"] < validation["raw"]["brier_score"]
            and validation["temperature"]["log_loss"] < validation["raw"]["log_loss"]
            and held_out["temperature"]["brier_score"] > held_out["raw"]["brier_score"]
            and held_out["temperature"]["log_loss"] > held_out["raw"]["log_loss"]):
        lines += ["", "The slight validation gains in Brier and log loss did not carry to "
                  "the held-out season: both scores worsened there. This experiment does not "
                  "establish a consistent calibration improvement."]
    lines += ["", "## Reading the result", "",
              "Temperature scaling changes probability sharpness without modeling home, draw, "
              "and away bias separately. A better log loss does not guarantee a better Brier "
              "score or better calibration for every class. Inspect class-wise reliability, "
              "counts, and intervals rather than treating one summary metric as proof. "
              "The 2024–25 test was assessed after the method and temperature were fixed; "
              "it was not used to choose a calibration method or setting.", ""]
    return "\n".join(lines)


def run_experiment(csv_path: Path = PROCESSED_DATA_DIR / "premier_league_matches.csv",
                   robustness_dir: Path = PROCESSED_DATA_DIR / "robustness",
                   frozen_dir: Path = PROCESSED_DATA_DIR / "predictions",
                   output_dir: Path = PROCESSED_DATA_DIR / "calibration") -> dict:
    """Fit on OOS development only; then assess validation and held-out test."""
    canonical = load_matches(csv_path)
    development, development_source = load_development_oos(canonical, csv_path, robustness_dir)
    fit = fit_temperature(development[list(PROBABILITY_COLUMNS)].to_numpy(dtype=float),
                          development["actual_outcome"].to_numpy())
    # The scalar method and parameter are now fixed; only then open assessment rows.
    assessment_rows, frozen_source = load_frozen_assessment(
        canonical, csv_path, frozen_dir, robustness_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    calibrated_frames = []
    reliability = []
    assessment = {}
    for index, split in enumerate(("validation", "test")):
        rows = assessment_rows[split]
        raw = rows[list(PROBABILITY_COLUMNS)].to_numpy(dtype=float)
        calibrated = temperature_probabilities(raw, fit.inverse_temperature)
        raw_scores = _score(rows, raw, "poisson")
        calibrated_scores = _score(rows, calibrated, "poisson_temperature")
        reliability += _reliability_rows(split, "raw", rows, raw, raw_scores)
        reliability += _reliability_rows(split, "temperature", rows, calibrated, calibrated_scores)
        assessment[split] = {
            "season": "2023-24" if split == "validation" else "2024-25",
            "raw": raw_scores,
            "temperature": calibrated_scores,
            "paired_differences": paired_date_intervals(
                rows, raw, calibrated, BOOTSTRAP_SEED + index, BOOTSTRAP_REPLICATES),
        }
        saved = rows.copy()
        for column in PROBABILITY_COLUMNS:
            saved[f"raw_{column}"] = saved[column]
        saved[list(PROBABILITY_COLUMNS)] = calibrated
        saved["predicted_class"] = [CLASSES[i] for i in calibrated.argmax(axis=1)]
        saved["model"] = "poisson_temperature"
        saved["calibration_fit_cutoff_date"] = development["match_date"].max()
        calibrated_frames.append(saved)
    output = {
        "method": {"name": "multiclass_temperature_scaling", "objective": "mean_multiclass_log_loss",
                   "probability_floor": EPSILON, "maximum_inverse_temperature": MAX_INVERSE_TEMPERATURE,
                   "reliability_bins": 10, "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                   "bootstrap_seed": BOOTSTRAP_SEED},
        "calibrator": asdict(fit),
        "training_provenance": {"split": "development_walk_forward",
                                "seasons": list(DEVELOPMENT_SEASONS),
                                "latest_match_date": development["match_date"].max(),
                                "n_matches": len(development), **development_source},
        "frozen_source": frozen_source,
        "assessment": assessment,
    }
    pd.concat(calibrated_frames, ignore_index=True).to_csv(
        output_dir / "calibrated_predictions.csv", index=False, float_format="%.17g")
    pd.DataFrame(reliability).to_csv(output_dir / "reliability_bins.csv", index=False,
                                      float_format="%.17g")
    for split in ("validation", "test"):
        _plot_reliability(reliability, split, assessment[split]["season"],
                          assessment[split]["raw"]["n_matches"],
                          output_dir / f"reliability_{split}.html")
    (output_dir / "metrics.json").write_text(json.dumps(output, indent=2, allow_nan=False) + "\n",
                                             encoding="utf-8")
    (output_dir / "REPORT.md").write_text(_report(output), encoding="utf-8")
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="Run frozen Poisson temperature-calibration experiment")
    parser.add_argument("--input", type=Path,
                        default=PROCESSED_DATA_DIR / "premier_league_matches.csv")
    parser.add_argument("--robustness-dir", type=Path, default=PROCESSED_DATA_DIR / "robustness")
    parser.add_argument("--frozen-dir", type=Path, default=PROCESSED_DATA_DIR / "predictions")
    parser.add_argument("--output-dir", type=Path, default=PROCESSED_DATA_DIR / "calibration")
    args = parser.parse_args()
    run_experiment(args.input, args.robustness_dir, args.frozen_dir, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
