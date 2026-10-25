"""Read-only Research Explorer access to saved robustness and calibration artefacts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .data import ArtifactError, DashboardData, OUTCOME_LABELS, SPLITS

MODELS = ("baseline", "elo_logistic", "poisson")
VARIANTS = ("raw", "temperature")
SEASON_COLUMNS = {"season", "model", "n_matches", "brier_score", "log_loss", "accuracy"}
BIN_COLUMNS = {"class", "bin_low", "bin_high", "mean_pred", "obs_freq", "count"}
BOOTSTRAP_METRICS = {
    "Brier score": "brier_score", "Log loss": "log_loss", "Accuracy": "accuracy",
    "ECE · home win": "ece_H", "ECE · draw": "ece_D", "ECE · away win": "ece_A",
}


@dataclass(frozen=True)
class RobustnessArtifacts:
    scores: pd.DataFrame
    bins: pd.DataFrame
    paired: dict
    metadata: dict


@dataclass(frozen=True)
class CalibrationArtifacts:
    metrics: dict
    bins: pd.DataFrame
    predictions: pd.DataFrame


@dataclass(frozen=True)
class ResearchData:
    robustness: RobustnessArtifacts | None
    calibration: CalibrationArtifacts | None
    warnings: tuple[str, ...]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> dict:
    if not path.is_file():
        raise ArtifactError(f"Missing saved artifact: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise ArtifactError(f"Could not read {path.name}: {exc}") from exc
    if not isinstance(value, dict):
        raise ArtifactError(f"{path.name} must contain a JSON object")
    return value


def _read_csv(path: Path, columns: set[str]) -> pd.DataFrame:
    if not path.is_file():
        raise ArtifactError(f"Missing saved artifact: {path}")
    try:
        data = pd.read_csv(path)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        raise ArtifactError(f"Could not read {path.name}: {exc}") from exc
    if data.empty or columns - set(data.columns):
        raise ArtifactError(f"{path.name} is empty or missing columns: {sorted(columns - set(data.columns))}")
    return data


def _check_bins(bins: pd.DataFrame, keys: list[str], expected: pd.DataFrame) -> None:
    if set(bins["class"]) != set(OUTCOME_LABELS):
        raise ArtifactError("Reliability bins must contain H, D and A")
    numeric = ["bin_low", "bin_high", "mean_pred", "obs_freq", "count"]
    for column in numeric:
        bins[column] = pd.to_numeric(bins[column], errors="coerce")
    if (bins[numeric].isna().any().any() or
            not bins[["bin_low", "bin_high", "mean_pred", "obs_freq"]].apply(
                lambda values: values.between(0, 1).all()).all() or
            (bins["bin_high"] <= bins["bin_low"]).any() or (bins["count"] < 0).any() or
            not np.allclose(bins["count"], bins["count"].round())):
        raise ArtifactError("Reliability bins contain invalid bounds, frequencies or counts")
    if bins.duplicated(keys + ["class", "bin_low"]).any():
        raise ArtifactError("Duplicate reliability bins")
    groups = bins.groupby(keys + ["class"], sort=False).agg(
        bin_count=("count", "size"), fixtures=("count", "sum")).reset_index()
    if (groups["bin_count"] != 10).any():
        raise ArtifactError("Each class needs ten saved reliability bins")
    reference = expected[keys + ["n_matches"]].drop_duplicates().rename(columns={"n_matches": "expected"})
    checked = groups.merge(reference, on=keys, how="outer", indicator=True)
    if (checked["_merge"] != "both").any() or (checked["fixtures"] != checked["expected"]).any():
        raise ArtifactError("Reliability bin counts disagree with saved sample sizes")


def load_robustness_artifacts(root: Path, source_sha: str | None = None) -> RobustnessArtifacts:
    folder = root / "data" / "processed" / "robustness"
    metadata = _read_json(folder / "metrics.json")
    scores = _read_csv(folder / "season_metrics.csv", SEASON_COLUMNS)
    bins = _read_csv(folder / "reliability_bins.csv", BIN_COLUMNS | {"season", "model"})
    try:
        protocol = metadata["protocol"]
        development = metadata["development"]
        seasons = development["seasons"]
        paired = development["paired_differences"]
        if source_sha and protocol["source_sha256"] != source_sha:
            raise ArtifactError("Robustness results use a different canonical dataset")
        if not seasons or set(scores["season"]) != set(seasons):
            raise ArtifactError("Season score table differs from saved robustness metrics")
        if (set(scores["model"]) != set(MODELS) or
                set(map(tuple, scores[["season", "model"]].itertuples(index=False, name=None))) !=
                {(season, model) for season in seasons for model in MODELS}):
            raise ArtifactError("Season scores need one row for each saved model and season")
        for column in ("n_matches", "brier_score", "log_loss", "accuracy"):
            scores[column] = pd.to_numeric(scores[column], errors="coerce")
        if (scores[["n_matches", "brier_score", "log_loss", "accuracy"]].isna().any().any()
                or (scores["n_matches"] <= 0).any() or not scores["accuracy"].between(0, 1).all()):
            raise ArtifactError("Season scores contain invalid values")
        for row in scores.itertuples(index=False):
            saved = seasons[row.season][row.model]
            for csv_name, metric_name in (("n_matches", "n_matches"), ("brier_score", "brier_score"),
                                          ("log_loss", "log_loss"), ("accuracy", "accuracy")):
                if not np.isclose(getattr(row, csv_name), saved[metric_name], atol=1e-10):
                    raise ArtifactError("Season score CSV disagrees with saved JSON")
        for label in OUTCOME_LABELS:
            column = f"ece_{label}"
            if column not in scores:
                scores[column] = [seasons[r.season][r.model].get("calibration_errors", {}).get(label)
                                  for r in scores.itertuples(index=False)]
            else:
                scores[column] = pd.to_numeric(scores[column], errors="coerce")
            if column in scores and scores[column].notna().any():
                if not scores[column].dropna().between(0, 1).all():
                    raise ArtifactError("Season ECE values must be in [0, 1]")
                for row in scores.itertuples(index=False):
                    saved_ece = seasons[row.season][row.model]["calibration_errors"][label]
                    if not np.isclose(getattr(row, column), saved_ece, atol=1e-10):
                        raise ArtifactError("Season ECE CSV disagrees with saved JSON")
        _check_bins(bins, ["season", "model"], scores)
        if set(paired) != set(seasons):
            raise ArtifactError("Paired interval seasons disagree with the score table")
    except (KeyError, TypeError, AttributeError) as exc:
        raise ArtifactError(f"Incomplete robustness metrics: {exc}") from exc
    return RobustnessArtifacts(scores.sort_values(["season", "model"]).reset_index(drop=True),
                               bins, paired, metadata)


def load_calibration_artifacts(root: Path, dashboard: DashboardData) -> CalibrationArtifacts:
    folder = root / "data" / "processed" / "calibration"
    metrics = _read_json(folder / "metrics.json")
    bins = _read_csv(folder / "reliability_bins.csv", BIN_COLUMNS | {"split", "variant"})
    predictions = _read_csv(folder / "calibrated_predictions.csv", {
        "match_id", "model", "split", "season", "actual_outcome", "match_date",
        "p_home", "p_draw", "p_away", "raw_p_home", "raw_p_draw", "raw_p_away"})
    try:
        fit = metrics["calibrator"]
        provenance = metrics["training_provenance"]
        frozen = metrics["frozen_source"]
        assessment = metrics["assessment"]
        if (fit["n_matches"] != 2660 or provenance["n_matches"] != 2660 or
                provenance["split"] != "development_walk_forward" or
                not np.isfinite(fit["temperature"]) or fit["temperature"] <= 0):
            raise ArtifactError("Temperature fit metadata is incomplete or invalid")
        original = root / "data" / "processed" / "predictions"
        if (_sha(original / "predictions.csv") != frozen["predictions_sha256"] or
                _sha(original / "metrics.json") != frozen["metrics_sha256"]):
            raise ArtifactError("Calibration results refer to different frozen evaluation files")
        if set(assessment) != set(SPLITS) or set(predictions["split"]) != set(SPLITS):
            raise ArtifactError("Calibration assessment must keep validation and test separate")
        if set(predictions["model"]) != {"poisson_temperature"} or predictions.duplicated(["split", "match_id"]).any():
            raise ArtifactError("Calibrated predictions are duplicated or misidentified")
        raw = dashboard.predictions[dashboard.predictions["model"] == "poisson"].sort_values(
            ["split", "match_id"]).reset_index(drop=True)
        calibrated = predictions.sort_values(["split", "match_id"]).reset_index(drop=True)
        if (len(raw) != len(calibrated) or raw["match_id"].tolist() != calibrated["match_id"].tolist()
                or raw["split"].tolist() != calibrated["split"].tolist()
                or raw["actual_outcome"].tolist() != calibrated["actual_outcome"].tolist()
                or raw["season"].tolist() != calibrated["season"].tolist()
                or raw["match_date"].tolist() != calibrated["match_date"].tolist()):
            raise ArtifactError("Calibrated rows do not match original Poisson fixtures")
        for column in ("p_home", "p_draw", "p_away"):
            if not np.allclose(raw[column], calibrated[f"raw_{column}"], rtol=0, atol=1e-12):
                raise ArtifactError("Calibrated rows disagree with original raw probabilities")
            predictions[column] = pd.to_numeric(predictions[column], errors="coerce")
        if (predictions[["p_home", "p_draw", "p_away"]].isna().any().any()
                or not predictions[["p_home", "p_draw", "p_away"]].ge(0).all().all()
                or not np.allclose(predictions[["p_home", "p_draw", "p_away"]].sum(axis=1), 1,
                                   rtol=0, atol=1e-9)):
            raise ArtifactError("Calibrated probabilities are invalid")
        expected_rows = []
        for split in SPLITS:
            values = assessment[split]
            if set(values) < {"raw", "temperature", "paired_differences", "season"}:
                raise ArtifactError("Calibration assessment is missing scores or intervals")
            for variant in VARIANTS:
                item = values[variant]
                if (item["n_matches"] != len(predictions[predictions["split"] == split])
                        or item["n_matches"] != dashboard.metrics[split]["poisson"]["n_matches"]):
                    raise ArtifactError("Calibration sample sizes disagree with saved fixtures")
                if variant == "raw":
                    for key in ("brier_score", "log_loss", "accuracy"):
                        if not np.isclose(item[key], dashboard.metrics[split]["poisson"][key], atol=1e-12):
                            raise ArtifactError("Raw Poisson score differs from frozen evaluation")
                expected_rows.append({"split": split, "variant": variant, "n_matches": item["n_matches"]})
        _check_bins(bins, ["split", "variant"], pd.DataFrame(expected_rows))
    except (KeyError, TypeError, AttributeError, OSError) as exc:
        raise ArtifactError(f"Incomplete calibration artifacts: {exc}") from exc
    return CalibrationArtifacts(metrics, bins, predictions)


def load_research_data(root: Path, dashboard: DashboardData) -> ResearchData:
    """Load each optional research family independently; preserve the core dashboard."""
    warnings = []
    source_sha = (dashboard.metrics.get("run_metadata") or {}).get("source_sha256")
    try:
        robustness = load_robustness_artifacts(root, source_sha)
    except ArtifactError as exc:
        robustness = None
        warnings.append(f"Season robustness: {exc}")
    try:
        calibration = load_calibration_artifacts(root, dashboard)
    except ArtifactError as exc:
        calibration = None
        warnings.append(f"Temperature experiment: {exc}")
    return ResearchData(robustness, calibration, tuple(warnings))


def season_score_table(data: RobustnessArtifacts, seasons: list[str], models: list[str]) -> pd.DataFrame:
    """Filter saved scores; empty filters mean no displayed rows."""
    rows = data.scores[data.scores["season"].isin(seasons) & data.scores["model"].isin(models)].copy()
    return rows.rename(columns={"season": "Season", "model": "Model",
                                "n_matches": "Fixtures", "brier_score": "Brier score",
                                "log_loss": "Log loss", "accuracy": "Accuracy",
                                "ece_H": "ECE H", "ece_D": "ECE D", "ece_A": "ECE A"})


def robustness_reliability_table(data: RobustnessArtifacts, season: str,
                                 outcome: str, models: list[str]) -> pd.DataFrame:
    if outcome not in OUTCOME_LABELS:
        return pd.DataFrame()
    rows = data.bins[(data.bins["season"] == season) & (data.bins["class"] == outcome) &
                     (data.bins["model"].isin(models)) & (data.bins["count"] > 0)]
    return _chart_rows(rows, "model")


def calibration_reliability_table(data: CalibrationArtifacts, split: str,
                                  outcome: str, variants: list[str]) -> pd.DataFrame:
    if split not in SPLITS or outcome not in OUTCOME_LABELS:
        return pd.DataFrame()
    rows = data.bins[(data.bins["split"] == split) & (data.bins["class"] == outcome) &
                     (data.bins["variant"].isin(variants)) & (data.bins["count"] > 0)]
    return _chart_rows(rows, "variant")


def _chart_rows(rows: pd.DataFrame, identity: str) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame(columns=["Model", "Mean predicted", "Observed frequency", "Fixtures", "Bin"])
    return pd.DataFrame({"Model": rows[identity], "Mean predicted": rows["mean_pred"],
                         "Observed frequency": rows["obs_freq"], "Fixtures": rows["count"],
                         "Bin": rows["bin_low"].map(lambda x: f"{x:.1f}") + "–" +
                                rows["bin_high"].map(lambda x: f"{x:.1f}")})


def robustness_interval_table(data: RobustnessArtifacts, season: str, metric: str) -> pd.DataFrame:
    if season not in data.paired or metric not in BOOTSTRAP_METRICS.values():
        return pd.DataFrame()
    return pd.DataFrame([{"Comparator": model, "Difference": values[metric]["difference"],
                          "CI low": values[metric]["ci95_low"], "CI high": values[metric]["ci95_high"]}
                         for model, values in data.paired[season].items() if metric in values])


def calibration_interval_table(data: CalibrationArtifacts, split: str) -> pd.DataFrame:
    if split not in SPLITS:
        return pd.DataFrame()
    differences = data.metrics["assessment"][split]["paired_differences"]
    return pd.DataFrame([{"Metric": label, "Difference": differences[key]["difference"],
                          "CI low": differences[key]["ci95_low"],
                          "CI high": differences[key]["ci95_high"]}
                         for label, key in BOOTSTRAP_METRICS.items() if key in differences])
