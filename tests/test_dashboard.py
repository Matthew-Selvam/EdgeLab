"""Tests for read-only dashboard artifact handling and presentation helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from edgelab.dashboard.data import (
    ArtifactError,
    filter_team_rows,
    filtered_predictions,
    history_table,
    load_dashboard_data,
    metric_table,
    poisson_calibration_note,
    reliability_table,
)


@pytest.fixture
def dashboard_root(tmp_path: Path) -> Path:
    output = tmp_path / "data" / "processed"
    generated = output / "predictions"
    generated.mkdir(parents=True)
    rows = []
    for split, season, date, fixture in (
        ("validation", "2023-24", "2023-08-01", "v1"),
        ("test", "2024-25", "2024-08-01", "t1"),
    ):
        for model in ("baseline", "poisson"):
            rows.append({
                "match_id": fixture, "model": model, "season": season,
                "match_date": date, "home_team": "Home FC", "away_team": "Away FC",
                "p_home": 0.5, "p_draw": 0.3, "p_away": 0.2,
                "actual_outcome": "H", "split": split, "prediction_timestamp": date,
                "expected_home_goals": 1.7 if model == "poisson" else None,
                "expected_away_goals": 0.9 if model == "poisson" else None,
            })
    pd.DataFrame(rows).to_csv(generated / "predictions.csv", index=False)
    model_metrics = {
        model: {"brier_score": 0.4, "log_loss": 0.7, "accuracy": 1.0,
                "n_matches": 1, "calibration_errors": {"H": 0.5, "D": 0.3, "A": 0.2},
                "reliability_table": [
                    {"class": "H", "bin_low": 0.0, "bin_high": 0.1,
                     "mean_pred": 0.0, "obs_freq": 0.0, "count": 0},
                    {"class": "H", "bin_low": 0.5, "bin_high": 0.6,
                     "mean_pred": 0.5, "obs_freq": 1.0, "count": 1},
                ]}
        for model in ("baseline", "poisson")
    }
    (generated / "metrics.json").write_text(json.dumps({
        "validation": model_metrics, "test": model_metrics,
    }), encoding="utf-8")
    pd.DataFrame([
        {"match_id": "v1", "home_goals": 2, "away_goals": 1},
        {"match_id": "t1", "home_goals": 1, "away_goals": 0},
    ]).to_csv(output / "premier_league_matches.csv", index=False)
    return tmp_path


def test_dashboard_loads_saved_artifacts_and_scores(dashboard_root: Path) -> None:
    data = load_dashboard_data(dashboard_root)
    assert len(data.predictions) == 4
    assert data.scores_available
    assert data.predictions.loc[data.predictions.model.eq("poisson"), "expected_home_goals"].eq(1.7).all()
    assert data.predictions.loc[data.predictions.model.eq("baseline"), "expected_home_goals"].isna().all()


def test_filters_keep_splits_separate_and_use_saved_metrics(dashboard_root: Path) -> None:
    data = load_dashboard_data(dashboard_root)
    selected = filtered_predictions(data, "test", "2024-25", ["poisson"])
    assert selected["match_id"].tolist() == ["t1"]
    assert filtered_predictions(data, "test", "2023-24", ["poisson"]).empty
    assert filtered_predictions(data, "test", "2024-25", []).empty
    assert metric_table(data, "test", ["poisson"])["Brier score"].tolist() == [0.4]


def test_team_search_is_literal_and_does_not_mix_splits(dashboard_root: Path) -> None:
    data = load_dashboard_data(dashboard_root)
    rows = filtered_predictions(data, "test", "2024-25", ["baseline", "poisson"])
    assert len(filter_team_rows(rows, " home ")) == 2
    assert filter_team_rows(rows, ".*").empty
    assert set(filter_team_rows(rows, "")["split"]) == {"test"}


def test_reliability_and_history_transformations(dashboard_root: Path) -> None:
    data = load_dashboard_data(dashboard_root)
    bins = reliability_table(data, "validation", ["poisson"], "H")
    assert bins["Fixtures"].tolist() == [1]
    assert bins["Mean predicted"].tolist() == [0.5]
    rows = filtered_predictions(data, "validation", "2023-24", ["poisson"])
    history = history_table(rows, data.scores_available)
    assert history.iloc[0]["Actual result"] == "Home win"
    assert history.iloc[0]["Score"] == "2–1"
    assert history.iloc[0]["Expected home goals"] == 1.7


def test_calibration_note_follows_saved_test_metrics(dashboard_root: Path) -> None:
    data = load_dashboard_data(dashboard_root)
    assert "home wins, draws, and away wins" not in poisson_calibration_note(data)
    metrics_path = dashboard_root / "data" / "processed" / "predictions" / "metrics.json"
    metrics = json.loads(metrics_path.read_text())
    metrics["test"]["poisson"]["calibration_errors"] = {"H": 0.6, "D": 0.4, "A": 0.3}
    metrics_path.write_text(json.dumps(metrics))
    assert "home wins, draws, and away wins" in poisson_calibration_note(load_dashboard_data(dashboard_root))


def test_missing_empty_and_inconsistent_artifacts_raise_clear_errors(dashboard_root: Path) -> None:
    generated = dashboard_root / "data" / "processed" / "predictions"
    predictions = generated / "predictions.csv"
    original = predictions.read_bytes()
    predictions.unlink()
    with pytest.raises(ArtifactError, match="Missing saved artifact"):
        load_dashboard_data(dashboard_root)
    predictions.write_text("match_id,model\n", encoding="utf-8")
    with pytest.raises(ArtifactError, match="empty"):
        load_dashboard_data(dashboard_root)
    predictions.write_bytes(original)
    rows = pd.read_csv(predictions)
    rows.loc[0, "p_home"] = 0.8
    rows.to_csv(predictions, index=False)
    with pytest.raises(ArtifactError, match="do not sum to one"):
        load_dashboard_data(dashboard_root)
    predictions.write_bytes(original)
    metrics_path = generated / "metrics.json"
    metrics = json.loads(metrics_path.read_text())
    metrics["test"]["poisson"]["n_matches"] = 2
    metrics_path.write_text(json.dumps(metrics))
    with pytest.raises(ArtifactError, match="count differs"):
        load_dashboard_data(dashboard_root)


def test_disagreeing_actual_result_is_rejected(dashboard_root: Path) -> None:
    predictions = dashboard_root / "data" / "processed" / "predictions" / "predictions.csv"
    rows = pd.read_csv(predictions)
    rows.loc[(rows["split"] == "validation") & (rows["model"] == "poisson"), "actual_outcome"] = "D"
    rows.to_csv(predictions, index=False)
    with pytest.raises(ArtifactError, match="Models disagree"):
        load_dashboard_data(dashboard_root)
