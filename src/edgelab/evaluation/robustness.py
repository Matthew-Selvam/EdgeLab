"""Read-only holdout reference plus expanding-season development robustness study.

The existing validation/test pipeline and its artefacts are never rewritten here.
Every development season is a distinct fold trained on completed prior seasons.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..data.ingest import PROCESSED_DATA_DIR
from ..data.models import Outcome
from ..models.baseline import ClassPriorBaseline
from ..models.elo import EloLogisticModel, EloRatings
from ..models.poisson import PoissonGoalsModel, poisson_outcome_probabilities
from .backtest import (
    BacktestConfig, PredictionRecord, TRAINING_SEASONS, TEST_SEASONS,
    VALIDATION_SEASONS, load_matches, load_team_mapper,
)
from .metrics import compute_all_metrics
from .report import MODEL_NAMES, compare_models, load_predictions

SPLIT = "development_walk_forward"


@dataclass(frozen=True)
class RobustnessConfig:
    min_training_matches: int = 380
    bootstrap_replicates: int = 1000
    bootstrap_seed: int = 20261021
    backtest: BacktestConfig = field(default_factory=BacktestConfig)

    def __post_init__(self) -> None:
        if self.min_training_matches < 1 or self.bootstrap_replicates < 1:
            raise ValueError("minimum training matches and bootstrap replicates must be positive")
        if (self.backtest.training_seasons != TRAINING_SEASONS
                or self.backtest.validation_seasons != VALIDATION_SEASONS
                or self.backtest.test_seasons != TEST_SEASONS):
            raise ValueError("robustness study must preserve the frozen season definitions")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _season_key(season: str) -> int:
    return int(season.split("-")[0])


def _predict_season(prior: pd.DataFrame, target: pd.DataFrame,
                    settings: BacktestConfig) -> list[PredictionRecord]:
    """Use existing models with fixed settings; apply target results by date."""
    mapper = load_team_mapper()
    baseline = ClassPriorBaseline(settings.baseline_config).fit(prior)
    base_probs = baseline.predict_static()
    elo = EloLogisticModel(settings.elo_config)
    ratings = EloRatings({}, settings.elo_config)
    poisson = PoissonGoalsModel(settings.poisson_config).fit(prior)
    prior_season: str | None = None
    for _, batch in prior.groupby("match_date", sort=True):
        if batch["season"].nunique() != 1:
            raise ValueError("multiple seasons on one training date")
        season = batch.iloc[0]["season"]
        if prior_season is not None and season != prior_season:
            ratings.regress_for_season()
        ratings.update_batch([
            (mapper.canonical_id(r.home_team), mapper.canonical_id(r.away_team), Outcome(r.outcome))
            for r in batch.itertuples(index=False)
        ])
        prior_season = season
    ratings.regress_for_season()
    cutoff = prior["match_date"].max().strftime("%Y-%m-%d")
    base_cutoff = cutoff
    records: list[PredictionRecord] = []
    for date, batch in target.groupby("match_date", sort=True):
        date_text = date.strftime("%Y-%m-%d")
        for row in batch.itertuples(index=False):
            home_id = mapper.canonical_id(row.home_team)
            away_id = mapper.canonical_id(row.away_team)
            elo_probs = elo.predict_proba(row.home_team, row.away_team, ratings,
                                          home_scale=settings.home_scale,
                                          draw_scale=settings.draw_scale)
            home_rate, away_rate = poisson.predict_rates(row.home_team, row.away_team)
            poisson_probs = poisson_outcome_probabilities(
                home_rate, away_rate, settings.poisson_config)
            for model, probs in zip(MODEL_NAMES, (base_probs, elo_probs, poisson_probs)):
                records.append(PredictionRecord(
                    match_id=row.match_id, model=model, season=row.season,
                    match_date=date_text, home_team=row.home_team, away_team=row.away_team,
                    p_home=probs[0], p_draw=probs[1], p_away=probs[2],
                    predicted_class="HDA"[int(np.argmax(probs))], actual_outcome=row.outcome,
                    split=SPLIT, prediction_timestamp=date_text,
                    information_cutoff_date=base_cutoff if model == "baseline" else cutoff,
                    canonical_home_team=mapper.canonicalize(row.home_team),
                    canonical_away_team=mapper.canonicalize(row.away_team),
                    pre_match_home_rating=ratings.get(home_id) if model == "elo_logistic" else None,
                    pre_match_away_rating=ratings.get(away_id) if model == "elo_logistic" else None,
                    expected_home_goals=home_rate if model == "poisson" else None,
                    expected_away_goals=away_rate if model == "poisson" else None,
                ))
        # All forecasts for this calendar date exist before either model sees its results.
        ratings.update_batch([
            (mapper.canonical_id(r.home_team), mapper.canonical_id(r.away_team), Outcome(r.outcome))
            for r in batch.itertuples(index=False)
        ])
        poisson.update_batch([
            (r.home_team, r.away_team, int(r.home_goals), int(r.away_goals))
            for r in batch.itertuples(index=False)
        ])
        cutoff = date_text
    return records


def _aligned_arrays(predictions: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Reject incomparable fixture sets before metric or bootstrap calculation."""
    frames = {}
    for model in MODEL_NAMES:
        rows = predictions[predictions["model"] == model]
        if rows["match_id"].duplicated().any():
            raise ValueError("duplicate fixture prediction")
        frames[model] = rows.set_index("match_id").sort_index()
    reference = frames[MODEL_NAMES[0]]
    if reference.empty:
        raise ValueError("no scored fixtures")
    for model in MODEL_NAMES[1:]:
        other = frames[model]
        if not reference.index.equals(other.index):
            raise ValueError("models do not score the same fixtures")
        for column in ("actual_outcome", "match_date", "season"):
            if not reference[column].equals(other[column]):
                raise ValueError(f"models disagree on {column}")
    outcomes = reference["actual_outcome"].map({"H": 0, "D": 1, "A": 2})
    if outcomes.isna().any():
        raise ValueError("invalid outcome")
    probabilities = {model: frame[["p_home", "p_draw", "p_away"]].to_numpy(dtype=float)
                     for model, frame in frames.items()}
    for values in probabilities.values():
        if (not np.isfinite(values).all() or (values < 0).any() or (values > 1).any()
                or not np.allclose(values.sum(axis=1), 1, atol=1e-9)):
            raise ValueError("invalid probability vector")
    return outcomes.to_numpy(dtype=int), reference["match_date"].to_numpy(), probabilities


def _scores(y: np.ndarray, p: np.ndarray) -> dict[str, float | dict[str, float]]:
    n = len(y)
    one_hot = np.eye(3)[y]
    ece = {}
    for class_idx, label in enumerate("HDA"):
        bins = np.minimum((p[:, class_idx] * 10).astype(int), 9)
        count = np.bincount(bins, minlength=10)
        pred_sum = np.bincount(bins, weights=p[:, class_idx], minlength=10)
        obs_sum = np.bincount(bins, weights=one_hot[:, class_idx], minlength=10)
        ece[label] = float(np.abs(pred_sum - obs_sum).sum() / n)
    return {
        "brier_score": float(np.square(p - one_hot).sum(axis=1).mean()),
        "log_loss": float(-np.log(np.clip(p[np.arange(n), y], 1e-15, 1 - 1e-15)).mean()),
        "accuracy": float((p.argmax(axis=1) == y).mean()),
        "ece": ece,
    }


def paired_date_bootstrap(predictions: pd.DataFrame, replicates: int,
                          seed: int) -> dict[str, dict]:
    """Percentile intervals for Poisson minus comparator, resampling dates in pairs."""
    if replicates < 1:
        raise ValueError("replicates must be positive")
    y, dates, probabilities = _aligned_arrays(predictions)
    blocks = [np.flatnonzero(dates == date) for date in np.unique(dates)]
    generator = np.random.default_rng(seed)
    keys = ("brier_score", "log_loss", "accuracy", "ece_H", "ece_D", "ece_A")
    output = {}
    point = {name: _scores(y, values) for name, values in probabilities.items()}
    for comparator in ("baseline", "elo_logistic"):
        output[comparator] = {}
        for key in keys:
            if key.startswith("ece_"):
                label = key[-1]
                delta = point["poisson"]["ece"][label] - point[comparator]["ece"][label]
            else:
                delta = point["poisson"][key] - point[comparator][key]
            output[comparator][key] = {"difference": float(delta), "bootstrap_values": []}
    for _ in range(replicates):
        chosen = generator.integers(0, len(blocks), size=len(blocks))
        sample = np.concatenate([blocks[i] for i in chosen])
        scored = {name: _scores(y[sample], p[sample]) for name, p in probabilities.items()}
        for comparator in ("baseline", "elo_logistic"):
            for key in keys:
                value = (scored["poisson"]["ece"][key[-1]] - scored[comparator]["ece"][key[-1]]
                         if key.startswith("ece_") else scored["poisson"][key] - scored[comparator][key])
                output[comparator][key]["bootstrap_values"].append(float(value))
    for comparisons in output.values():
        for result in comparisons.values():
            low, high = np.quantile(result.pop("bootstrap_values"), [0.025, 0.975])
            result["ci95_low"] = float(low)
            result["ci95_high"] = float(high)
    return output


def _wilson(positive: int, count: int) -> tuple[float | None, float | None]:
    if count == 0:
        return None, None
    z = 1.959963984540054
    fraction = positive / count
    centre = (fraction + z*z/(2*count)) / (1 + z*z/count)
    half = z * np.sqrt(fraction*(1-fraction)/count + z*z/(4*count*count)) / (1 + z*z/count)
    return float(max(0, centre-half)), float(min(1, centre+half))


def _reference(reference_dir: Path, source_sha: str, settings: BacktestConfig) -> dict:
    """Verify and summarize existing frozen outputs without writing them."""
    pred_path = reference_dir / "predictions.csv"
    metrics_path = reference_dir / "metrics.json"
    predictions = load_predictions(pred_path)
    saved = json.loads(metrics_path.read_text(encoding="utf-8"))
    if saved["run_metadata"]["source_sha256"] != source_sha:
        raise ValueError("frozen results were generated from a different canonical CSV")
    original = saved["run_metadata"]["config"]
    expected = {
        "elo_initial_rating": settings.elo_config.initial_rating,
        "elo_k_factor": settings.elo_config.k_factor,
        "elo_home_advantage": settings.elo_config.home_advantage,
        "elo_season_regression_factor": settings.elo_config.season_regression_factor,
        "elo_home_scale": settings.home_scale,
        "elo_draw_scale": settings.draw_scale,
        "baseline_smoothing_alpha": (settings.baseline_config.smoothing_alpha
                                     if settings.baseline_config else 1.0),
        "poisson_prior_matches": settings.poisson_config.prior_matches,
        "poisson_tail_tolerance": settings.poisson_config.tail_tolerance,
        "poisson_max_goals": settings.poisson_config.max_goals,
    }
    if any(original.get(key) != value for key, value in expected.items()):
        raise ValueError("study model settings differ from the frozen evaluation")
    output = {"predictions_sha256": _sha(pred_path), "metrics_sha256": _sha(metrics_path),
              "source_sha256": source_sha, "splits": {}}
    for split, season in (("validation", "2023-24"), ("test", "2024-25")):
        subset = predictions[predictions["split"] == split]
        if set(subset["season"]) != {season}:
            raise ValueError(f"frozen {split} season differs from protocol")
        if len(subset) != 380 * len(MODEL_NAMES):
            raise ValueError(f"frozen {split} fixture count differs from protocol")
        _, computed = compare_models(predictions, split)
        for model in MODEL_NAMES:
            for metric in ("brier_score", "log_loss", "accuracy", "n_matches"):
                if not np.isclose(computed[model][metric], saved[split][model][metric], atol=1e-12):
                    raise ValueError(f"frozen {split} {model} {metric} disagrees with predictions")
        output["splits"][split] = {"season": season, "models": {
            model: {key: saved[split][model][key] for key in
                    ("n_matches", "brier_score", "log_loss", "accuracy", "calibration_errors")}
            for model in MODEL_NAMES}}
    return output


def _report(metrics: dict) -> str:
    lines = ["# EdgeLab calibration and season robustness", "",
             "Development walk-forward folds use fixed model settings and all completed prior seasons. "
             "The first season has no eligible prior season and is excluded. Each fold's baseline "
             "is frozen; Elo and Poisson update only after every fixture on a date has been forecast. "
             "League goal means are fitted once from prior seasons for each fold.", "",
             "## Season scores (development only)", "",
             "Multiclass Brier = mean Σ(p−y)² (0–2, lower better); log loss = mean −log(p_actual) "
             "(lower better); accuracy = argmax hit rate (higher better).", "",
             "| Season | Model | n | Brier | Log loss | Accuracy | ECE H | ECE D | ECE A |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for season, models in metrics["development"]["seasons"].items():
        for model, item in models.items():
            ece = item["calibration_errors"]
            lines.append(f"| {season} | {model} | {item['n_matches']} | {item['brier_score']:.4f} | "
                         f"{item['log_loss']:.4f} | {item['accuracy']:.3f} | "
                         f"{ece['H']:.3f} | {ece['D']:.3f} | {ece['A']:.3f} |")
    lines += ["", "## Paired uncertainty: Poisson minus comparator", "",
              "Each 95% percentile interval resamples calendar dates within that season, retaining "
              "all same-date fixtures and paired model predictions. Negative Brier, log loss, or ECE "
              "differences favour Poisson; positive accuracy differences favour Poisson. "
              "ECE is 10-bin one-versus-rest expected calibration error, a count-weighted absolute "
              "forecast versus observed gap. Bins and their Wilson observed-frequency intervals are "
              "saved in reliability_bins.csv. Sparse bins, temporal dependence across dates, "
              "multiple comparisons, and only seven development folds limit inference.", "",
              "| Season | Comparator | Δ Brier [95%] | Δ Log loss [95%] | Δ Accuracy [95%] | Δ ECE H [95%] | Δ ECE D [95%] | Δ ECE A [95%] |",
              "|---|---|---|---|---|---|---|---|"]
    for season, comparisons in metrics["development"]["paired_differences"].items():
        for comparator, values in comparisons.items():
            def cell(key: str) -> str:
                v = values[key]
                return f"{v['difference']:+.3f} [{v['ci95_low']:+.3f}, {v['ci95_high']:+.3f}]"
            lines.append(f"| {season} | {comparator} | {cell('brier_score')} | {cell('log_loss')} | {cell('accuracy')} | "
                         f"{cell('ece_H')} | {cell('ece_D')} | {cell('ece_A')} |")
    lines += ["", "## Frozen validation and held-out test (read-only reference)", "",
              "These scores come from the existing saved outputs and are not walk-forward development "
              "folds. The 2024–25 test was not used to choose settings, bins, or analysis methods. "
              "Each later split contains one season, so it does not establish long-run superiority.", "",
              "| Split | Season | Model | n | Brier | Log loss | Accuracy | ECE H | ECE D | ECE A |",
              "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for split, reference in (metrics["frozen_reference"] or {"splits": {}})["splits"].items():
        for model, item in reference["models"].items():
            ece = item["calibration_errors"]
            lines.append(f"| {split} | {reference['season']} | {model} | {item['n_matches']} | "
                         f"{item['brier_score']:.4f} | {item['log_loss']:.4f} | "
                         f"{item['accuracy']:.3f} | {ece['H']:.3f} | {ece['D']:.3f} | {ece['A']:.3f} |")
    baseline_worse = {label: sum(
        models["poisson"]["calibration_errors"][label] >
        models["baseline"]["calibration_errors"][label]
        for models in metrics["development"]["seasons"].values()) for label in "HDA"}
    supported = {label: sum(
        comparisons["baseline"][f"ece_{label}"]["ci95_low"] > 0
        for comparisons in metrics["development"]["paired_differences"].values()) for label in "HDA"}
    n_folds = len(metrics["development"]["seasons"])
    pattern = ("The positive ECE pattern recurs most clearly for home and away wins; "
               "draw evidence is weaker. " if supported["H"] >= 3 and supported["A"] >= 3
               and supported["D"] == 0 else "Outcome patterns vary by season and class. ")
    lines += ["", "## Exclusions and interpretation", "",
              f"{metrics['development']['excluded_fixtures']} development fixtures were excluded. "
              "Each excluded match and reason is in excluded_fixtures.csv. "
              f"Poisson's ECE exceeds the baseline in {baseline_worse['H']}/{n_folds} home, "
              f"{baseline_worse['D']}/{n_folds} draw, and {baseline_worse['A']}/{n_folds} away development folds. "
              "Unadjusted date-bootstrap intervals for this ECE difference are entirely above zero "
              f"in {supported['H']}/{n_folds}, {supported['D']}/{n_folds}, and "
              f"{supported['A']}/{n_folds} folds, respectively. "
              + pattern + "This is not a population-level or multiple-testing-adjusted claim. "
              "Per-class ECE depends on binning and is itself noisy; use reliability bin counts, "
              "observed-frequency intervals, and paired differences to interpret it. "
              "The study is descriptive; no calibration transform or model setting was selected.", ""]
    return "\n".join(lines)


def run_study(config: RobustnessConfig,
              csv_path: Path = PROCESSED_DATA_DIR / "premier_league_matches.csv",
              output_dir: Path = PROCESSED_DATA_DIR / "robustness",
              reference_dir: Path | None = PROCESSED_DATA_DIR / "predictions") -> dict:
    """Run development folds, then write reproducible study artefacts."""
    matches = load_matches(csv_path)
    source_sha = _sha(csv_path)
    if set(matches["season"]) != TRAINING_SEASONS | VALIDATION_SEASONS | TEST_SEASONS:
        raise ValueError("canonical seasons differ from the frozen protocol")
    reference = _reference(reference_dir, source_sha, config.backtest) if reference_dir is not None else None
    development = matches[matches["season"].isin(TRAINING_SEASONS)]
    ordered = sorted(TRAINING_SEASONS, key=_season_key)
    records: list[PredictionRecord] = []
    exclusions: list[dict] = []
    fold_metadata = {}
    for index, season in enumerate(ordered):
        prior = development[development["season"].isin(ordered[:index])]
        target = development[development["season"] == season]
        if target.empty:
            raise ValueError(f"missing development season {season}")
        if not prior.empty and prior["match_date"].max() >= target["match_date"].min():
            raise ValueError("development seasons overlap in calendar time")
        if index == 0 or len(prior) < config.min_training_matches:
            reason = "no_completed_prior_season" if index == 0 else "insufficient_prior_matches"
            exclusions.extend({"match_id": r.match_id, "season": season,
                               "match_date": r.match_date.strftime("%Y-%m-%d"),
                               "reason": reason, "eligible_prior_matches": len(prior)}
                              for r in target.itertuples(index=False))
            continue
        fold_metadata[season] = {"training_seasons": ordered[:index],
                                 "training_matches": len(prior),
                                 "scored_fixtures": len(target),
                                 "training_cutoff_date": prior["match_date"].max().strftime("%Y-%m-%d")}
        records.extend(_predict_season(prior, target, config.backtest))
    if not records:
        raise ValueError("no eligible development fixtures")
    predictions = pd.DataFrame([asdict(record) for record in records])
    season_metrics = {}
    paired = {}
    reliability_rows = []
    for index, season in enumerate(fold_metadata):
        subset = predictions[predictions["season"] == season]
        # Existing comparator enforces same fixture IDs, dates, and outcomes.
        compare_models(subset, SPLIT)
        season_metrics[season] = {}
        for model in MODEL_NAMES:
            rows = subset[subset["model"] == model]
            scored = compute_all_metrics(rows["actual_outcome"].tolist(),
                                         list(rows[["p_home", "p_draw", "p_away"]].itertuples(index=False, name=None)),
                                         model)
            season_metrics[season][model] = {key: getattr(scored, key) for key in
                ("n_matches", "brier_score", "log_loss", "accuracy", "calibration_errors")}
            for item in scored.reliability_table:
                count = item["count"]
                positive = round(item["obs_freq"] * count)
                low, high = _wilson(positive, count)
                reliability_rows.append({"season": season, "model": model, **item,
                                         "observed_positive": positive,
                                         "obs_ci95_low": low, "obs_ci95_high": high})
        paired[season] = paired_date_bootstrap(subset, config.bootstrap_replicates,
                                              config.bootstrap_seed + index)
    metrics = {
        "protocol": {
            "source_sha256": source_sha, "splits": {"development": ordered,
                "validation": sorted(VALIDATION_SEASONS), "held_out_test": sorted(TEST_SEASONS)},
            "minimum_prior_completed_seasons": 1,
            "minimum_prior_matches": config.min_training_matches,
            "bootstrap_replicates": config.bootstrap_replicates,
            "bootstrap_seed": config.bootstrap_seed,
            "reliability_bins": 10,
            "model_settings": {
                "baseline_smoothing_alpha": (config.backtest.baseline_config.smoothing_alpha
                    if config.backtest.baseline_config else 1.0),
                "elo": asdict(config.backtest.elo_config),
                "elo_home_scale": config.backtest.home_scale,
                "elo_draw_scale": config.backtest.draw_scale,
                "poisson": asdict(config.backtest.poisson_config),
            },
        },
        "development": {"folds": fold_metadata, "excluded_fixtures": len(exclusions),
                        "seasons": season_metrics, "paired_differences": paired},
        "frozen_reference": reference,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_dir / "walk_forward_predictions.csv", index=False, float_format="%.17g")
    pd.DataFrame(exclusions).to_csv(output_dir / "excluded_fixtures.csv", index=False)
    pd.DataFrame(reliability_rows).to_csv(output_dir / "reliability_bins.csv", index=False,
                                          float_format="%.17g")
    pd.DataFrame([
        {"season": season, "model": model, **{
            key: value for key, value in item.items() if key != "calibration_errors"},
         **{f"ece_{label}": item["calibration_errors"][label] for label in "HDA"}}
        for season, models in season_metrics.items() for model, item in models.items()
    ]).to_csv(output_dir / "season_metrics.csv", index=False, float_format="%.17g")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n",
                                             encoding="utf-8")
    (output_dir / "REPORT.md").write_text(_report(metrics), encoding="utf-8")
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="Run read-only EdgeLab development robustness study")
    parser.add_argument("--input", type=Path,
                        default=PROCESSED_DATA_DIR / "premier_league_matches.csv")
    parser.add_argument("--output-dir", type=Path, default=PROCESSED_DATA_DIR / "robustness")
    parser.add_argument("--reference-dir", type=Path,
                        default=PROCESSED_DATA_DIR / "predictions")
    arguments = parser.parse_args()
    run_study(RobustnessConfig(), arguments.input, arguments.output_dir,
              arguments.reference_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
