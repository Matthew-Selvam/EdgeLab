"""Read-only Research Explorer checks against the saved study artefacts."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from edgelab.dashboard.app import calibration_chart, season_trend_chart
from edgelab.dashboard.data import filtered_predictions, load_dashboard_data
from edgelab.dashboard.research import (
    calibration_interval_table,
    calibration_reliability_table,
    load_research_data,
    robustness_interval_table,
    robustness_reliability_table,
    season_score_table,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def saved_research():
    dashboard = load_dashboard_data(ROOT)
    research = load_research_data(ROOT, dashboard)
    assert research.warnings == ()
    assert research.robustness is not None
    assert research.calibration is not None
    return dashboard, research


def test_saved_season_scores_and_filters(saved_research) -> None:
    _, research = saved_research
    robustness = research.robustness
    assert len(robustness.scores) == 21
    assert set(robustness.scores["n_matches"]) == {380}
    selected = season_score_table(robustness, ["2022-23"], ["poisson"])
    assert selected[["Season", "Model", "Fixtures"]].iloc[0].tolist() == ["2022-23", "poisson", 380]
    saved = json.loads((ROOT / "data/processed/robustness/metrics.json").read_text())
    expected = saved["development"]["seasons"]["2022-23"]["poisson"]
    assert selected.iloc[0]["Brier score"] == pytest.approx(expected["brier_score"])
    assert selected.iloc[0]["ECE D"] == pytest.approx(expected["calibration_errors"]["D"])
    assert season_score_table(robustness, [], ["poisson"]).empty
    assert season_score_table(robustness, ["2022-23"], []).empty
    trend = season_trend_chart(season_score_table(robustness, sorted(robustness.scores.season.unique()),
                                                 ["baseline", "elo_logistic", "poisson"]), "Brier score")
    assert len(trend.data) == 3
    assert all(len(trace.x) == 7 for trace in trend.data)


def test_saved_reliability_and_intervals(saved_research) -> None:
    _, research = saved_research
    robustness, calibration = research.robustness, research.calibration
    dev_bins = robustness_reliability_table(robustness, "2022-23", "H", ["poisson"])
    assert dev_bins["Fixtures"].sum() == 380
    assert robustness_reliability_table(robustness, "2022-23", "invalid", ["poisson"]).empty
    plot = calibration_chart(dev_bins, "H")
    assert list(plot.layout.xaxis.range) == [0, 1]
    assert list(plot.layout.yaxis.range) == [0, 1]
    assert plot.data[1].name == "Poisson"

    bins = calibration_reliability_table(calibration, "test", "D", ["raw", "temperature"])
    assert set(bins["Model"]) == {"raw", "temperature"}
    assert bins.groupby("Model")["Fixtures"].sum().to_dict() == {"raw": 380, "temperature": 380}
    assert [trace.name for trace in calibration_chart(bins, "D").data] == [
        "Ideal calibration", "Raw Poisson", "Temperature scaled"]
    assert calibration_reliability_table(calibration, "unknown", "D", ["raw"]).empty

    dev_interval = robustness_interval_table(robustness, "2022-23", "brier_score")
    saved_dev = robustness.paired["2022-23"]["baseline"]["brier_score"]
    assert dev_interval.loc[dev_interval.Comparator.eq("baseline"), "CI low"].iloc[0] == pytest.approx(
        saved_dev["ci95_low"])
    assert robustness_interval_table(robustness, "2022-23", "unknown").empty
    test_interval = calibration_interval_table(calibration, "test")
    saved_test = calibration.metrics["assessment"]["test"]["paired_differences"]["ece_H"]
    assert test_interval.loc[test_interval.Metric.eq("ECE · home win"), "CI high"].iloc[0] == pytest.approx(
        saved_test["ci95_high"])
    assert calibration_interval_table(calibration, "unknown").empty


def test_calibration_provenance_and_original_explorer(saved_research) -> None:
    dashboard, research = saved_research
    calibration = research.calibration
    assert calibration.metrics["calibrator"]["temperature"] == pytest.approx(0.981548, abs=5e-7)
    assert calibration.metrics["training_provenance"]["n_matches"] == 2660
    assert calibration.metrics["training_provenance"]["split"] == "development_walk_forward"
    assert set(calibration.predictions["split"]) == {"validation", "test"}
    assert calibration.predictions.groupby("split").size().to_dict() == {"test": 380, "validation": 380}
    original = filtered_predictions(dashboard, "test", "2024-25", ["baseline", "elo_logistic", "poisson"])
    assert len(original) == 1140
    assert set(original.model) == {"baseline", "elo_logistic", "poisson"}
    assert "poisson_temperature" not in set(dashboard.predictions.model)
    assert calibration.metrics["assessment"]["test"]["temperature"]["brier_score"] > \
        calibration.metrics["assessment"]["test"]["raw"]["brier_score"]


def test_missing_and_inconsistent_research_files_leave_core_dashboard_available(tmp_path: Path) -> None:
    output = tmp_path / "data/processed"
    shutil.copytree(ROOT / "data/processed/predictions", output / "predictions")
    shutil.copy(ROOT / "data/processed/premier_league_matches.csv", output)
    dashboard = load_dashboard_data(tmp_path)
    missing = load_research_data(tmp_path, dashboard)
    assert missing.robustness is None and missing.calibration is None
    assert len(missing.warnings) == 2
    assert len(filtered_predictions(dashboard, "validation", "2023-24", ["poisson"])) == 380

    shutil.copytree(ROOT / "data/processed/robustness", output / "robustness")
    partial = load_research_data(tmp_path, dashboard)
    assert partial.robustness is not None and partial.calibration is None
    assert len(partial.warnings) == 1
    shutil.copytree(ROOT / "data/processed/calibration", output / "calibration")
    scores = output / "robustness/season_metrics.csv"
    scores.write_text(scores.read_text().replace("2016-17,baseline,380", "2016-17,baseline,379", 1))
    inconsistent = load_research_data(tmp_path, dashboard)
    assert inconsistent.robustness is None and inconsistent.calibration is not None
    assert "disagrees" in inconsistent.warnings[0]


def test_streamlit_research_explorer_with_actual_outputs() -> None:
    app = AppTest.from_file(str(ROOT / "src/edgelab/dashboard/app.py")).run(timeout=30)
    assert not app.exception
    assert [tab.label for tab in app.tabs] == [
        "Scoreboard", "Calibration", "Fixture explorer", "Research explorer", "Methodology"]
    season_table = next(item.value for item in app.dataframe if "ECE H" in item.value.columns and
                        "Season" in item.value.columns)
    assert len(season_table) == 21
    assert set(season_table["Fixtures"]) == {380}
    calibration_table = next(item.value for item in app.dataframe if "Variant" in item.value.columns)
    assert calibration_table["Variant"].tolist() == ["Raw Poisson", "Temperature scaled"]
    assert calibration_table["Brier score"].iloc[0] == pytest.approx(
        json.loads((ROOT / "data/processed/calibration/metrics.json").read_text())
        ["assessment"]["validation"]["raw"]["brier_score"])
    assert any("2,660" in item.value and "0.981548" in item.value for item in app.markdown)
    assert any("Fixture explorer" in tab.label for tab in app.tabs)

    app.multiselect[1].set_value(["2022-23"]).run(timeout=30)
    season_table = next(item.value for item in app.dataframe if "ECE H" in item.value.columns and
                        "Season" in item.value.columns)
    assert len(season_table) == 3
    app.radio[2].set_value("test").run(timeout=30)
    calibration_table = next(item.value for item in app.dataframe if "Variant" in item.value.columns)
    assert calibration_table["Brier score"].iloc[1] > calibration_table["Brier score"].iloc[0]
