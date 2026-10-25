"""Synthetic boundary cases for the first chronological holdout."""

import math

import pandas as pd
import pytest

from edgelab.data.models import Outcome
from edgelab.evaluation.backtest import BacktestConfig, run_backtest
from edgelab.evaluation.metrics import brier_score_multiclass, log_loss_multiclass
from edgelab.evaluation.report import compare_models
from edgelab.models.elo import EloConfig, EloRatings
from edgelab.models.teams import TeamAliasMapper


def _fixture(match_id, season, date, home, away, hg, ag):
    return dict(match_id=match_id, competition="Premier League", season=season,
                match_date=date, home_team=home, away_team=away,
                home_goals=hg, away_goals=ag,
                outcome="H" if hg > ag else "A" if hg < ag else "D",
                source="synthetic")


def _run(tmp_path, rows, name):
    csv_path = tmp_path / f"{name}.csv"
    output_dir = tmp_path / name
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    run_backtest(BacktestConfig(training_seasons={"2022-23"},
                                validation_seasons={"2023-24"},
                                test_seasons={"2024-25"}), csv_path, output_dir)
    return pd.read_csv(output_dir / "predictions" / "predictions.csv")


def _sample():
    return [
        _fixture("train", "2022-23", "2023-05-01", "Arsenal FC", "Chelsea FC", 2, 1),
        _fixture("v1", "2023-24", "2023-08-01", "Arsenal FC", "Chelsea FC", 2, 0),
        _fixture("v2", "2023-24", "2023-08-01", "Arsenal FC", "Fulham FC", 0, 0),
        _fixture("test", "2024-25", "2024-08-01", "Arsenal FC", "Chelsea FC", 0, 1),
    ]


def _probs(df, split, model):
    rows = df[(df.split == split) & (df.model == model)].sort_values("match_id")
    return rows[["p_home", "p_draw", "p_away"]].to_numpy().tolist()


def test_same_date_results_cannot_affect_any_forecast_on_that_date(tmp_path):
    rows = _sample()
    original = _run(tmp_path, rows, "original")
    changed = [row.copy() for row in rows]
    changed[1]["home_goals"], changed[1]["away_goals"], changed[1]["outcome"] = 0, 2, "A"
    alternate = _run(tmp_path, changed, "alternate")
    for model in ("baseline", "elo_logistic", "poisson"):
        assert _probs(original, "validation", model) == _probs(alternate, "validation", model)
    assert _probs(original, "test", "elo_logistic") != _probs(alternate, "test", "elo_logistic")
    assert _probs(original, "test", "poisson") != _probs(alternate, "test", "poisson")
    assert _probs(original, "test", "baseline") == _probs(alternate, "test", "baseline")
    assert set(original["prediction_timestamp"]) == {"2023-08-01", "2024-08-01"}
    assert set(original[original.split == "validation"]["information_cutoff_date"]) == {"2023-05-01"}


def test_held_out_result_does_not_change_held_out_forecast(tmp_path):
    rows = _sample()
    original = _run(tmp_path, rows, "original")
    changed = [row.copy() for row in rows]
    changed[-1]["home_goals"], changed[-1]["away_goals"], changed[-1]["outcome"] = 3, 0, "H"
    alternate = _run(tmp_path, changed, "alternate")
    for model in ("baseline", "elo_logistic", "poisson"):
        assert _probs(original, "test", model) == _probs(alternate, "test", model)
    assert original[original.model == "elo_logistic"]["pre_match_home_rating"].notna().all()
    assert original[original.model == "baseline"]["pre_match_home_rating"].isna().all()
    assert original["home_team"].iloc[0] == "Arsenal FC"
    assert original["canonical_home_team"].iloc[0] == "Arsenal"


def test_same_date_elo_updates_use_pre_date_snapshot():
    config = EloConfig(home_advantage=0, k_factor=20)
    matches = [("a", "b", Outcome.HOME_WIN), ("a", "c", Outcome.AWAY_WIN)]
    first = EloRatings({}, config)
    second = EloRatings({}, config)
    first.update_batch(matches)
    second.update_batch(list(reversed(matches)))
    assert first.ratings == second.ratings
    assert first.get("a") == pytest.approx(1500)
    assert first.get("b") == pytest.approx(1490)
    assert first.get("c") == pytest.approx(1510)


def test_matched_fixture_guard_and_metric_inputs(tmp_path):
    predictions = _run(tmp_path, _sample(), "complete")
    compare_models(predictions, "validation")
    # Remove only the Elo row for one fixture.
    incomplete = predictions[~((predictions.match_id == "v2") &
                               (predictions.model == "elo_logistic"))]
    with pytest.raises(ValueError, match="same fixtures"):
        compare_models(incomplete, "validation")
    missing_poisson = predictions[~((predictions.match_id == "v2") &
                                    (predictions.model == "poisson"))]
    with pytest.raises(ValueError, match="same fixtures"):
        compare_models(missing_poisson, "validation")
    with pytest.raises(ValueError, match="sum to one"):
        brier_score_multiclass(["H"], [(0.2, 0.2, 0.2)])
    with pytest.raises(ValueError, match="equal length"):
        log_loss_multiclass(["H", "D"], [(1.0, 0.0, 0.0)])
    assert brier_score_multiclass(["H"], [(0.5, 0.25, 0.25)]) == pytest.approx(0.375)
    assert log_loss_multiclass(["H"], [(0.5, 0.25, 0.25)]) == pytest.approx(math.log(2))


def test_every_observed_team_alias_is_mapped():
    from edgelab.data.ingest import PROCESSED_DATA_DIR
    from edgelab.evaluation.backtest import load_matches

    matches = load_matches(PROCESSED_DATA_DIR / "premier_league_matches.csv")
    mapper = TeamAliasMapper()
    names = set(matches["home_team"]) | set(matches["away_team"])
    assert len(names) == 47
    assert names <= set(mapper._name_to_canonical)
    assert len({mapper.canonical_id(name) for name in names}) == 34
