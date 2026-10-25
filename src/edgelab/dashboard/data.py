"""Validated, read-only access to saved backtest outputs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

SPLITS = ("validation", "test")
OUTCOME_LABELS = {"H": "Home win", "D": "Draw", "A": "Away win"}
REQUIRED_PREDICTION_COLUMNS = {
    "match_id", "model", "season", "match_date", "home_team", "away_team",
    "p_home", "p_draw", "p_away", "actual_outcome", "split",
    "prediction_timestamp", "expected_home_goals", "expected_away_goals",
}
METRIC_COLUMNS = ("brier_score", "log_loss", "accuracy", "n_matches")


class ArtifactError(ValueError):
    """Saved artifacts are missing, empty, or inconsistent."""


@dataclass(frozen=True)
class DashboardData:
    predictions: pd.DataFrame
    metrics: dict
    scores_available: bool


def load_dashboard_data(root: Path) -> DashboardData:
    """Load generated CSV/JSON only; never run ingestion or a model."""
    output = root / "data" / "processed"
    prediction_path = output / "predictions" / "predictions.csv"
    metrics_path = output / "predictions" / "metrics.json"
    for path in (prediction_path, metrics_path):
        if not path.is_file():
            raise ArtifactError(f"Missing saved artifact: {path}. Run the evaluation pipeline first.")
    try:
        predictions = pd.read_csv(prediction_path)
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError, pd.errors.ParserError) as exc:
        raise ArtifactError(f"Could not read saved evaluation artifacts: {exc}") from exc
    if predictions.empty:
        raise ArtifactError("The saved prediction CSV is empty. Run the evaluation pipeline first.")
    missing = REQUIRED_PREDICTION_COLUMNS - set(predictions.columns)
    if missing:
        raise ArtifactError(f"Prediction CSV is missing columns: {sorted(missing)}")
    if not isinstance(metrics, dict):
        raise ArtifactError("Metrics JSON must contain an object.")
    if predictions[list(REQUIRED_PREDICTION_COLUMNS - {"expected_home_goals", "expected_away_goals"})].isna().any().any():
        raise ArtifactError("Prediction CSV has missing required values.")
    if not set(predictions["split"]).issubset(SPLITS):
        raise ArtifactError("Prediction CSV contains an unknown evaluation split.")
    if not set(predictions["actual_outcome"]).issubset(OUTCOME_LABELS):
        raise ArtifactError("Prediction CSV contains an unknown actual outcome.")
    for col in ("p_home", "p_draw", "p_away"):
        predictions[col] = pd.to_numeric(predictions[col], errors="coerce")
        if predictions[col].isna().any() or not predictions[col].between(0, 1).all():
            raise ArtifactError(f"Invalid saved probabilities in {col}.")
    if not (predictions[["p_home", "p_draw", "p_away"]].sum(axis=1) - 1).abs().le(1e-8).all():
        raise ArtifactError("Saved H/D/A probabilities do not sum to one.")
    if predictions.duplicated(["split", "model", "match_id"]).any():
        raise ArtifactError("Duplicate model predictions exist for a fixture.")
    fixture_fields = predictions.groupby(["split", "match_id"])[
        ["season", "match_date", "actual_outcome"]
    ].nunique()
    if fixture_fields.gt(1).any().any():
        raise ArtifactError("Models disagree on saved fixture season, date, or actual outcome.")
    try:
        pd.to_datetime(predictions["match_date"], format="%Y-%m-%d", errors="raise")
        pd.to_datetime(predictions["prediction_timestamp"], format="%Y-%m-%d", errors="raise")
    except ValueError as exc:
        raise ArtifactError("Saved forecast dates are invalid.") from exc
    for split in SPLITS:
        split_metrics = metrics.get(split)
        if not isinstance(split_metrics, dict):
            raise ArtifactError(f"Metrics JSON has no {split} results.")
        split_rows = predictions.loc[predictions["split"].eq(split)]
        if set(split_rows["model"]) != set(split_metrics):
            raise ArtifactError(f"Models differ between saved predictions and {split} metrics.")
        fixture_sets = []
        for model, values in split_metrics.items():
            if not isinstance(values, dict) or any(key not in values for key in METRIC_COLUMNS):
                raise ArtifactError(f"Incomplete {split} metrics for {model}.")
            rows = split_rows.loc[split_rows["model"].eq(model)]
            if len(rows) != values["n_matches"]:
                raise ArtifactError(f"Saved {split} metrics count differs from {model} predictions.")
            fixture_sets.append(set(rows["match_id"]))
            if "calibration_errors" not in values or "reliability_table" not in values:
                raise ArtifactError(f"Missing {split} calibration output for {model}.")
        if fixture_sets and any(items != fixture_sets[0] for items in fixture_sets[1:]):
            raise ArtifactError(f"{split} models do not cover the same fixtures.")

    scores_available = False
    canonical_path = output / "premier_league_matches.csv"
    if canonical_path.is_file():
        try:
            canonical = pd.read_csv(canonical_path, usecols=["match_id", "home_goals", "away_goals"])
            if canonical["match_id"].duplicated().any():
                raise ArtifactError("Canonical match CSV contains duplicate IDs.")
            predictions = predictions.merge(canonical, on="match_id", how="left", validate="many_to_one")
            scores_available = predictions[["home_goals", "away_goals"]].notna().all().all()
        except (OSError, ValueError, pd.errors.ParserError) as exc:
            raise ArtifactError(f"Could not read canonical scores: {exc}") from exc
    return DashboardData(predictions=predictions, metrics=metrics, scores_available=scores_available)


def filtered_predictions(data: DashboardData, split: str, season: str, models: list[str]) -> pd.DataFrame:
    """Select stored prediction rows without recalculating forecasts or metrics."""
    if split not in SPLITS or not models:
        return data.predictions.iloc[0:0].copy()
    rows = data.predictions
    return rows.loc[rows["split"].eq(split) & rows["season"].eq(season) & rows["model"].isin(models)].copy()


def filter_team_rows(rows: pd.DataFrame, query: str) -> pd.DataFrame:
    """Search saved source team names case-insensitively and literally."""
    query = query.strip()
    if not query:
        return rows.copy()
    return rows.loc[
        rows["home_team"].str.contains(query, case=False, regex=False)
        | rows["away_team"].str.contains(query, case=False, regex=False)
    ].copy()


def metric_table(data: DashboardData, split: str, models: list[str]) -> pd.DataFrame:
    """Return saved full-split metrics for selected models."""
    if split not in SPLITS:
        return pd.DataFrame()
    records = [
        {"Model": model, "Brier score": values["brier_score"],
         "Log loss": values["log_loss"], "Accuracy": values["accuracy"],
         "Fixtures": values["n_matches"]}
        for model, values in data.metrics[split].items() if model in models
    ]
    return pd.DataFrame.from_records(records)


def reliability_table(data: DashboardData, split: str, models: list[str], outcome: str) -> pd.DataFrame:
    """Return nonempty saved reliability bins for a single outcome."""
    if split not in SPLITS or outcome not in OUTCOME_LABELS:
        return pd.DataFrame()
    records = [
        {"Model": model, "Mean predicted": row["mean_pred"],
         "Observed frequency": row["obs_freq"], "Fixtures": row["count"],
         "Bin": f"{row['bin_low']:.1f}–{row['bin_high']:.1f}"}
        for model, values in data.metrics[split].items() if model in models
        for row in values["reliability_table"]
        if row["class"] == outcome and row["count"] > 0
    ]
    return pd.DataFrame.from_records(records)


def poisson_calibration_note(data: DashboardData) -> str:
    """Describe the saved held-out Poisson calibration relative to baseline."""
    test = data.metrics.get("test", {})
    if "poisson" not in test or "baseline" not in test:
        return "Inspect per-class calibration alongside aggregate scores; one test season is limited evidence."
    poisson = test["poisson"]["calibration_errors"]
    baseline = test["baseline"]["calibration_errors"]
    worse = [OUTCOME_LABELS[outcome].lower() for outcome in OUTCOME_LABELS
             if poisson[outcome] > baseline[outcome]]
    if len(worse) == 3:
        return ("In the saved held-out test, Poisson's calibration error exceeds the "
                "baseline's for home wins, draws, and away wins.")
    if worse:
        return ("In the saved held-out test, Poisson's calibration error exceeds the "
                f"baseline's for {', '.join(worse)}.")
    return "Poisson's held-out calibration should be read with the sparse-bin and one-season limits in mind."


def history_table(rows: pd.DataFrame, scores_available: bool) -> pd.DataFrame:
    """Make a readable fixture history, retaining saved probability precision."""
    columns = ["match_date", "home_team", "away_team", "model", "actual_outcome",
               "p_home", "p_draw", "p_away", "expected_home_goals", "expected_away_goals"]
    if rows.empty:
        return pd.DataFrame()
    history = rows[columns].copy()
    history["actual_outcome"] = history["actual_outcome"].map(OUTCOME_LABELS)
    if scores_available:
        score = rows["home_goals"].astype(int).astype(str) + "–" + rows["away_goals"].astype(int).astype(str)
        history.insert(4, "Score", score)
    history = history.rename(columns={
        "match_date": "Date", "home_team": "Home", "away_team": "Away",
        "model": "Model", "actual_outcome": "Actual result",
        "p_home": "P(Home)", "p_draw": "P(Draw)", "p_away": "P(Away)",
        "expected_home_goals": "Expected home goals", "expected_away_goals": "Expected away goals",
    })
    return history.sort_values(["Date", "Home", "Away", "Model"], ascending=[False, True, True, True])
