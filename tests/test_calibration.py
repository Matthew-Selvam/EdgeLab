"""Temperature scaling and saved-forecast provenance tests."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from edgelab.evaluation import calibration
from edgelab.evaluation.backtest import load_matches

ROOT = Path(__file__).resolve().parents[1] / "data" / "processed"


def test_known_temperature_transform_and_class_order():
    raw = np.array([[.25, .25, .5], [.5, .25, .25]])
    np.testing.assert_allclose(calibration.temperature_probabilities(raw, 1), raw)
    np.testing.assert_allclose(calibration.temperature_probabilities(raw[:1], 2),
                               [[1/6, 1/6, 2/3]])
    np.testing.assert_allclose(calibration.temperature_probabilities(raw[:1], 0),
                               [[1/3, 1/3, 1/3]])
    assert np.array_equal(calibration.temperature_probabilities(raw, 2).argmax(axis=1),
                          raw.argmax(axis=1))


def test_extreme_probabilities_are_stable_and_normalized():
    raw = np.array([[1.0, 0.0, 0.0], [1e-300, 0.0, 1.0],
                    [1e-15, 1e-15, 1-2e-15]])
    for beta in (0.0, .5, 1.0, 1000.0, 1e6):
        scaled = calibration.temperature_probabilities(raw, beta)
        assert np.isfinite(scaled).all()
        assert (scaled >= 0).all() and (scaled <= 1).all()
        np.testing.assert_allclose(scaled.sum(axis=1), 1)
    for bad in (np.array([[.4, .4, .4]]), np.array([[np.nan, .5, .5]]),
                np.array([[-.1, .5, .6]])):
        with pytest.raises(ValueError, match="probabilities"):
            calibration.temperature_probabilities(bad, 1)
    with pytest.raises(ValueError, match="inverse temperature"):
        calibration.temperature_probabilities(raw, -1)


def test_fit_is_reproducible_and_uses_multiclass_log_loss():
    probabilities = np.array([[.9, .05, .05]] * 6)
    outcomes = ["H", "H", "H", "D", "D", "A"]
    fitted = calibration.fit_temperature(probabilities, outcomes)
    assert fitted == calibration.fit_temperature(probabilities, outcomes)
    assert 0 <= fitted.inverse_temperature < 1
    assert fitted.log_loss_fitted < fitted.log_loss_raw
    scaled = calibration.temperature_probabilities(probabilities, fitted.inverse_temperature)
    assert scaled[0, 0] < probabilities[0, 0]
    with pytest.raises(ValueError, match="equal length"):
        calibration.fit_temperature(probabilities, ["H"])
    with pytest.raises(ValueError, match="H/D/A"):
        calibration.fit_temperature(probabilities, ["X"] * 6)


def test_toy_scores_keep_accuracy_distinct_from_probability_quality():
    rows = pd.DataFrame({"match_date": ["2023-01-01"], "actual_outcome": ["A"]})
    raw = np.array([[.25, .25, .5]])
    scaled = calibration.temperature_probabilities(raw, 2)
    before = calibration._score(rows, raw, "poisson")
    after = calibration._score(rows, scaled, "poisson_temperature")
    assert before["accuracy"] == after["accuracy"] == 1
    assert after["brier_score"] < before["brier_score"]
    assert after["log_loss"] < before["log_loss"]
    assert before["calibration_errors"] == {"H": .25, "D": .25, "A": .5}
    assert after["calibration_errors"]["A"] == pytest.approx(1/3)
    intervals = calibration.paired_date_intervals(rows, raw, scaled, replicates=20)
    assert intervals["accuracy"]["difference"] == 0
    assert intervals["brier_score"]["ci95_high"] < 0


def test_development_fit_rows_are_audited_walk_forward_forecasts():
    csv = ROOT / "premier_league_matches.csv"
    canonical = load_matches(csv)
    rows, provenance = calibration.load_development_oos(canonical, csv, ROOT / "robustness")
    assert len(rows) == 2660
    assert set(rows["split"]) == {"development_walk_forward"}
    assert rows["match_date"].max() < "2023-24"
    assert provenance["season_counts"] == {season: 380 for season in calibration.DEVELOPMENT_SEASONS}
    metadata = json.loads((ROOT / "robustness" / "metrics.json").read_text())
    cutoffs = {season: metadata["development"]["folds"][season]["training_cutoff_date"]
               for season in calibration.DEVELOPMENT_SEASONS}
    tampered = rows.copy()
    first_date = tampered.loc[tampered.index[0], "match_date"]
    same_date = tampered.index[tampered["match_date"] == first_date]
    assert len(same_date) > 1
    tampered.loc[same_date[1], "information_cutoff_date"] = first_date
    with pytest.raises(ValueError, match="previous-date batching"):
        calibration._check_rows(tampered, canonical, calibration.DEVELOPMENT_SEASONS,
                                "development_walk_forward", cutoffs)
    tampered = rows.copy()
    tampered.loc[tampered.index[0], "actual_outcome"] = "X"
    with pytest.raises(ValueError, match="disagrees with canonical"):
        calibration._check_rows(tampered, canonical, calibration.DEVELOPMENT_SEASONS,
                                "development_walk_forward", cutoffs)


def test_in_sample_or_future_rows_cannot_enter_calibration_fit(tmp_path: Path):
    csv = ROOT / "premier_league_matches.csv"
    canonical = load_matches(csv)
    source = ROOT / "robustness"
    shutil.copy(source / "metrics.json", tmp_path / "metrics.json")
    rows = pd.read_csv(source / "walk_forward_predictions.csv")
    rows.loc[rows["model"] == "poisson", "split"] = "train"
    rows.to_csv(tmp_path / "walk_forward_predictions.csv", index=False)
    with pytest.raises(ValueError, match="expected split"):
        calibration.load_development_oos(canonical, csv, tmp_path)
    rows.loc[rows["model"] == "poisson", "split"] = "development_walk_forward"
    first = rows.index[rows["model"] == "poisson"][0]
    rows.loc[first, "season"] = "2024-25"
    rows.to_csv(tmp_path / "walk_forward_predictions.csv", index=False)
    with pytest.raises(ValueError, match="unexpected seasons"):
        calibration.load_development_oos(canonical, csv, tmp_path)


def test_frozen_test_file_tampering_is_rejected_before_assessment(tmp_path: Path):
    csv = ROOT / "premier_league_matches.csv"
    canonical = load_matches(csv)
    source = ROOT / "predictions"
    shutil.copy(source / "metrics.json", tmp_path / "metrics.json")
    rows = pd.read_csv(source / "predictions.csv")
    first = rows.index[(rows["model"] == "poisson") & (rows["split"] == "test")][0]
    rows.loc[first, "actual_outcome"] = "X"
    rows.to_csv(tmp_path / "predictions.csv", index=False)
    with pytest.raises(ValueError, match="artefacts changed"):
        calibration.load_frozen_assessment(canonical, csv, tmp_path, ROOT / "robustness")


def test_calibrator_is_fitted_before_assessment_files_are_opened(monkeypatch, tmp_path: Path):
    events = []
    original_fit = calibration.fit_temperature

    def fit_spy(probabilities, outcomes):
        events.append(("fit", len(outcomes)))
        return original_fit(probabilities, outcomes)

    def assessment_spy(*args):
        events.append(("assessment", None))
        raise RuntimeError("stopped before held-out assessment")

    monkeypatch.setattr(calibration, "fit_temperature", fit_spy)
    monkeypatch.setattr(calibration, "load_frozen_assessment", assessment_spy)
    with pytest.raises(RuntimeError, match="stopped before"):
        calibration.run_experiment(ROOT / "premier_league_matches.csv", ROOT / "robustness",
                                   ROOT / "predictions", tmp_path)
    assert events == [("fit", 2660), ("assessment", None)]


def test_reliability_rows_and_plot_are_reproducible(tmp_path: Path):
    rows = pd.DataFrame({"match_date": ["2023-01-01", "2023-01-02"],
                         "actual_outcome": ["H", "A"]})
    raw = np.array([[.7, .2, .1], [.1, .2, .7]])
    scored = calibration._score(rows, raw, "poisson")
    bins = calibration._reliability_rows("validation", "raw", rows, raw, scored)
    assert len(bins) == 30
    assert all(sum(item["count"] for item in bins if item["class"] == label) == 2
               for label in "HDA")
    assert all(item["obs_ci95_low"] is None for item in bins if item["count"] == 0)
    plot_rows = bins + calibration._reliability_rows("validation", "temperature", rows, raw, scored)
    one, two = tmp_path / "one.html", tmp_path / "two.html"
    calibration._plot_reliability(plot_rows, "validation", "2023-24", 2, one)
    calibration._plot_reliability(plot_rows, "validation", "2023-24", 2, two)
    assert one.read_bytes() == two.read_bytes()
    assert b"Home win" in one.read_bytes() and b"Temperature scaled" in one.read_bytes()
