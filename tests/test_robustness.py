"""Development robustness protocol tests with small, known chronological fixtures."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import pytest

from edgelab.evaluation.backtest import BacktestConfig
from edgelab.evaluation.robustness import (
    RobustnessConfig, _aligned_arrays, _predict_season, _reference, _scores,
    paired_date_bootstrap, run_study,
)


def fixture(season: str, date: str, number: int, home: str, away: str,
            home_goals: int, away_goals: int) -> dict:
    return {
        "match_id": f"{season}-{number}", "competition": "Premier League",
        "season": season, "match_date": date, "home_team": home, "away_team": away,
        "home_goals": home_goals, "away_goals": away_goals,
        "outcome": "H" if home_goals > away_goals else "A" if home_goals < away_goals else "D",
        "source": "test",
    }


def sample_season(season: str) -> list[dict]:
    year = season[:4]
    return [
        fixture(season, f"{year}-08-01", 1, "Arsenal FC", "Chelsea FC", 2, 1),
        fixture(season, f"{year}-08-01", 2, "Liverpool FC", "Everton FC", 1, 1),
        fixture(season, f"{year}-08-02", 3, "Arsenal FC", "Chelsea FC", 0, 2),
    ]


def _frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    prior = pd.DataFrame(sample_season("2015-16"))
    target = pd.DataFrame(sample_season("2016-17"))
    for frame in (prior, target):
        frame["match_date"] = pd.to_datetime(frame["match_date"])
    return prior, target


def _probabilities(records, match_id, model):
    row = next(record for record in records if record.match_id == match_id and record.model == model)
    return row.p_home, row.p_draw, row.p_away


def test_same_date_results_are_unavailable_to_other_forecasts():
    prior, target = _frames()
    original = _predict_season(prior, target, BacktestConfig())
    changed = target.copy()
    changed.loc[changed["match_id"] == "2016-17-1", ["home_goals", "away_goals", "outcome"]] = [0, 3, "A"]
    replay = _predict_season(prior, changed, BacktestConfig())
    for model in ("baseline", "elo_logistic", "poisson"):
        assert _probabilities(original, "2016-17-2", model) == _probabilities(replay, "2016-17-2", model)
    for model in ("elo_logistic", "poisson"):
        assert _probabilities(original, "2016-17-3", model) != _probabilities(replay, "2016-17-3", model)


def test_future_result_changes_cannot_affect_earlier_date():
    prior, target = _frames()
    original = _predict_season(prior, target, BacktestConfig())
    changed = target.copy()
    changed.loc[changed["match_id"] == "2016-17-3", ["home_goals", "away_goals", "outcome"]] = [4, 0, "H"]
    replay = _predict_season(prior, changed, BacktestConfig())
    for model in ("baseline", "elo_logistic", "poisson"):
        assert _probabilities(original, "2016-17-1", model) == _probabilities(replay, "2016-17-1", model)
        assert _probabilities(original, "2016-17-2", model) == _probabilities(replay, "2016-17-2", model)
    assert all(record.information_cutoff_date < record.match_date for record in original)


def test_known_metrics_and_paired_bootstrap_are_reproducible():
    actual = pd.DataFrame([
        {"match_id": "a", "match_date": "2020-01-01", "season": "2019-20", "actual_outcome": "H"},
        {"match_id": "b", "match_date": "2020-01-01", "season": "2019-20", "actual_outcome": "A"},
    ])
    frames = []
    for model, vectors in {
        "baseline": [(1/3, 1/3, 1/3)] * 2,
        "elo_logistic": [(0.5, 0.25, 0.25)] * 2,
        "poisson": [(1.0, 0.0, 0.0), (0.0, 0.0, 1.0)],
    }.items():
        rows = actual.copy()
        rows["model"] = model
        rows[["p_home", "p_draw", "p_away"]] = pd.DataFrame(vectors)
        frames.append(rows)
    predictions = pd.concat(frames, ignore_index=True)
    y, _, probabilities = _aligned_arrays(predictions)
    perfect = _scores(y, probabilities["poisson"])
    assert perfect["brier_score"] == 0
    assert perfect["accuracy"] == 1
    assert perfect["ece"] == {"H": 0.0, "D": 0.0, "A": 0.0}
    uniform = _scores(y, probabilities["baseline"])
    assert uniform["brier_score"] == pytest.approx(2 / 3)
    assert uniform["log_loss"] == pytest.approx(1.0986122886681098)
    first = paired_date_bootstrap(predictions, 20, 42)
    assert first == paired_date_bootstrap(predictions, 20, 42)
    assert first["baseline"]["brier_score"]["difference"] == pytest.approx(-2 / 3)


def test_incomparable_fixture_samples_are_rejected():
    prior, target = _frames()
    rows = pd.DataFrame([r.__dict__ for r in _predict_season(prior, target, BacktestConfig())])
    rows = rows[~((rows["model"] == "poisson") & (rows["match_id"] == "2016-17-2"))]
    with pytest.raises(ValueError, match="same fixtures"):
        _aligned_arrays(rows)


def test_fold_exclusion_reproducibility_and_held_out_separation(tmp_path: Path):
    seasons = [f"{year}-{str((year + 1) % 100).zfill(2)}" for year in range(2015, 2025)]
    csv = tmp_path / "matches.csv"
    all_rows = [row for season in seasons for row in sample_season(season)]
    pd.DataFrame(all_rows).to_csv(csv, index=False)
    settings = RobustnessConfig(min_training_matches=3, bootstrap_replicates=20)
    out_a, out_b = tmp_path / "a", tmp_path / "b"
    first = run_study(settings, csv, out_a, reference_dir=None)
    second = run_study(settings, csv, out_b, reference_dir=None)
    assert first == second
    for name in ("walk_forward_predictions.csv", "excluded_fixtures.csv", "reliability_bins.csv",
                 "season_metrics.csv", "metrics.json", "REPORT.md"):
        assert (out_a / name).read_bytes() == (out_b / name).read_bytes()
    predictions = pd.read_csv(out_a / "walk_forward_predictions.csv")
    excluded = pd.read_csv(out_a / "excluded_fixtures.csv")
    assert len(excluded) == 3
    assert set(excluded["season"]) == {"2015-16"}
    assert set(excluded["reason"]) == {"no_completed_prior_season"}
    assert len(predictions) == 7 * 3 * 3
    assert set(predictions["season"]) == set(seasons[1:8])
    assert not set(predictions["season"]) & {"2023-24", "2024-25"}
    assert first["frozen_reference"] is None
    assert first["development"]["folds"]["2022-23"]["training_matches"] == 7 * 3
    changed = pd.DataFrame(all_rows)
    changed.loc[changed["match_id"] == "2024-25-1", ["home_goals", "away_goals", "outcome"]] = [0, 3, "A"]
    changed_csv = tmp_path / "changed_test.csv"
    changed.to_csv(changed_csv, index=False)
    out_c = tmp_path / "c"
    run_study(settings, changed_csv, out_c, reference_dir=None)
    assert (out_a / "walk_forward_predictions.csv").read_bytes() == (out_c / "walk_forward_predictions.csv").read_bytes()


def test_reference_checks_source_identity_and_keeps_files_unchanged():
    root = Path(__file__).resolve().parents[1] / "data" / "processed"
    canonical = root / "premier_league_matches.csv"
    reference = root / "predictions"
    source_sha = hashlib.sha256(canonical.read_bytes()).hexdigest()
    hashes_before = {name: hashlib.sha256((reference / name).read_bytes()).hexdigest()
                     for name in ("predictions.csv", "metrics.json")}
    result = _reference(reference, source_sha, BacktestConfig())
    assert result["splits"]["validation"]["season"] == "2023-24"
    assert result["splits"]["test"]["season"] == "2024-25"
    with pytest.raises(ValueError, match="different canonical CSV"):
        _reference(reference, "0" * 64, BacktestConfig())
    assert hashes_before == {name: hashlib.sha256((reference / name).read_bytes()).hexdigest()
                             for name in hashes_before}
