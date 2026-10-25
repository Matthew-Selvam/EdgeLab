"""Chronological backtesting engine for EdgeLab.

Implements expanding-window evaluation with strict temporal integrity:
- Training/development: 2015-16 through 2022-23
- Validation: 2023-24
- Held-out test: 2024-25

Key design: all matches on the same date are predicted before any results
from that date update ratings or features.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

from ..data.ingest import RAW_DATA_DIR, PROCESSED_DATA_DIR
from ..data.models import Outcome
from ..models.baseline import ClassPriorBaseline, BaselineConfig
from ..models.elo import EloConfig, EloLogisticModel, EloRatings
from ..models.poisson import PoissonConfig, PoissonGoalsModel, poisson_outcome_probabilities
from ..models.teams import TeamAliasMapper

logger = logging.getLogger(__name__)

# Chronological split: train through 2022-23, validate 2023-24, test 2024-25
TRAINING_SEASONS = {"2015-16", "2016-17", "2017-18", "2018-19",
                    "2019-20", "2020-21", "2021-22", "2022-23"}
VALIDATION_SEASONS = {"2023-24"}
TEST_SEASONS = {"2024-25"}


@dataclass
class PredictionRecord:
    """A single model prediction for a match.

    Attributes:
        match_id: Canonical match identifier.
        model: Model name/identifier.
        season: Season of the match.
        match_date: ISO date string (forecast date).
        home_team: Source team name.
        away_team: Source team name.
        p_home: Predicted P(home win).
        p_draw: Predicted P(draw).
        p_away: Predicted P(away win).
        predicted_class: 'H', 'D', or 'A' (argmax).
        actual_outcome: 'H', 'D', or 'A' from full-time result.
        split: 'train', 'validation', or 'test'.
        prediction_timestamp: Date-resolution forecast label; no kickoff time is implied.
    """

    match_id: str
    model: str
    season: str
    match_date: str
    home_team: str
    away_team: str
    p_home: float
    p_draw: float
    p_away: float
    predicted_class: str
    actual_outcome: str
    split: str
    prediction_timestamp: str
    information_cutoff_date: str | None = None
    canonical_home_team: str | None = None
    canonical_away_team: str | None = None
    pre_match_home_rating: float | None = None
    pre_match_away_rating: float | None = None
    expected_home_goals: float | None = None
    expected_away_goals: float | None = None


@dataclass
class BacktestConfig:
    """Configuration for a backtest run.

    Attributes:
        training_seasons: Set of season strings for training/development.
        validation_seasons: Set of season strings for validation.
        test_seasons: Set of season strings for held-out test.
        elo_config: Elo rating parameters.
        baseline_config: Baseline model parameters.
        home_scale: Elo logistic scaling for home-win probability.
        draw_scale: Elo draw probability scaling.
        poisson_config: Fixed smoothing and numerical controls for Poisson goals.
    """

    training_seasons: set[str] = field(default_factory=lambda: TRAINING_SEASONS)
    validation_seasons: set[str] = field(default_factory=lambda: VALIDATION_SEASONS)
    test_seasons: set[str] = field(default_factory=lambda: TEST_SEASONS)
    elo_config: EloConfig = field(default_factory=EloConfig)
    baseline_config: Optional[BaselineConfig] = None
    home_scale: float = 280.0
    draw_scale: float = 350.0
    poisson_config: PoissonConfig = field(default_factory=PoissonConfig)


def load_matches(csv_path: Path) -> pd.DataFrame:
    """Load the canonical match CSV into a DataFrame.

    Parses match_date as datetime and sorts chronologically by date.
    """
    df = pd.read_csv(csv_path)
    required = {"match_id", "season", "match_date", "home_team", "away_team",
                "home_goals", "away_goals", "outcome"}
    if required - set(df.columns):
        raise ValueError(f"missing canonical columns: {sorted(required - set(df.columns))}")
    if df.empty or df[list(required)].isna().any().any():
        raise ValueError("empty or incomplete canonical match data")
    df["match_date"] = pd.to_datetime(df["match_date"], format="%Y-%m-%d", errors="raise")
    for column in ("home_goals", "away_goals"):
        if not pd.api.types.is_integer_dtype(df[column]) or (df[column] < 0).any():
            raise ValueError(f"{column} must contain nonnegative integers")
    expected = df.apply(lambda r: "H" if r.home_goals > r.away_goals else
                        "A" if r.home_goals < r.away_goals else "D", axis=1)
    if not df["outcome"].eq(expected).all():
        raise ValueError("outcomes disagree with scores")
    if df["match_id"].duplicated().any() or df.duplicated(
            ["season", "match_date", "home_team", "away_team"]).any():
        raise ValueError("duplicate matches")
    df = df.sort_values(["match_date", "match_id"]).reset_index(drop=True)
    return df


def load_team_mapper(raw_dir: Path = RAW_DATA_DIR) -> TeamAliasMapper:
    """Load the team alias mapper.

    The mapper is self-contained with hand-curated aliases.
    """
    return TeamAliasMapper()


def get_split(season: str) -> str:
    """Return the split name for a given season string."""
    if season in TRAINING_SEASONS:
        return "train"
    if season in VALIDATION_SEASONS:
        return "validation"
    if season in TEST_SEASONS:
        return "test"
    return "unknown"


def run_backtest(
    config: BacktestConfig,
    csv_path: Path = PROCESSED_DATA_DIR / "premier_league_matches.csv",
    output_dir: Optional[Path] = None,
) -> dict:
    """Run the full backtest pipeline.

    Generates predictions for baseline, Elo and Poisson on all
    non-training fixtures, with chronological integrity.

    Parameters
    ----------
    config:
        Backtest configuration.
    csv_path:
        Path to the canonical CSV.
    output_dir:
        Directory for output files. Defaults to data/processed/.

    Returns
    -------
    dict
        Summary with prediction counts per model and split.
    """
    if output_dir is None:
        output_dir = PROCESSED_DATA_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_matches(csv_path)
    mapper = load_team_mapper()
    groups = (config.training_seasons, config.validation_seasons, config.test_seasons)
    if any(not group for group in groups) or len(set.union(*groups)) != sum(map(len, groups)):
        raise ValueError("season splits must be nonempty and disjoint")
    if set(df["season"]) - set.union(*groups):
        raise ValueError("unassigned seasons in canonical data")
    bounds = []
    for seasons in groups:
        dates = df.loc[df["season"].isin(seasons), "match_date"]
        if dates.empty:
            raise ValueError("every split must contain fixtures")
        bounds.append((dates.min(), dates.max()))
    if not (bounds[0][1] < bounds[1][0] and bounds[1][1] < bounds[2][0]):
        raise ValueError("training, validation and test dates must be strictly ordered")
    split_for = {season: split for split, seasons in zip(
        ("train", "validation", "test"), groups) for season in seasons}

    # --- Baseline model (class prior) ---
    baseline = ClassPriorBaseline(config.baseline_config or BaselineConfig())

    # Fit baseline on training data
    train_df = df[df["season"].isin(config.training_seasons)].copy()
    baseline.fit(train_df)

    predictions: list[PredictionRecord] = []
    # Frozen development-only class prior: validation/test labels never change it.
    ph, pd_val, pa = baseline.predict_static()
    eval_df = df[df["season"].isin(config.validation_seasons | config.test_seasons)].copy()
    last_training_date = train_df["match_date"].max().strftime("%Y-%m-%d")

    for _, row in eval_df.iterrows():

        pred_class = max(
            [("H", ph), ("D", pd_val), ("A", pa)],
            key=lambda x: x[1]
        )[0]

        split = split_for[row["season"]]
        predictions.append(PredictionRecord(
            match_id=row["match_id"],
            model="baseline",
            season=row["season"],
            match_date=row["match_date"].strftime("%Y-%m-%d"),
            home_team=row["home_team"],
            away_team=row["away_team"],
            p_home=ph,
            p_draw=pd_val,
            p_away=pa,
            predicted_class=pred_class,
            actual_outcome=row["outcome"],
            split=split,
            prediction_timestamp=row["match_date"].strftime("%Y-%m-%d"),
            information_cutoff_date=last_training_date,
            canonical_home_team=mapper.canonicalize(row["home_team"]),
            canonical_away_team=mapper.canonicalize(row["away_team"]),
        ))

    # --- Elo model ---
    elo_model = EloLogisticModel(config.elo_config)
    ratings = EloRatings({}, config.elo_config)
    poisson = PoissonGoalsModel(config.poisson_config).fit(train_df)

    # First, process all training matches to build up ratings
    train_df_sorted = train_df.sort_values(["match_date", "match_id"]).reset_index(drop=True)
    prior_training_season = None
    for _, date_group in train_df_sorted.groupby("match_date", sort=True):
        if date_group["season"].nunique() != 1:
            raise ValueError("multiple seasons on one date")
        season = date_group.iloc[0]["season"]
        if prior_training_season is not None and season != prior_training_season:
            ratings.regress_for_season()
        prior_training_season = season
        ratings.update_batch([
            (mapper.canonical_id(row["home_team"]), mapper.canonical_id(row["away_team"]),
             Outcome(row["outcome"])) for _, row in date_group.iterrows()
        ])

    # Now evaluate on non-training data with same-date batch handling
    # Group by date to ensure same-date matches are predicted before updates
    eval_df_sorted = eval_df.sort_values(["match_date", "match_id"]).reset_index(drop=True)

    last_completed_date = last_training_date
    last_season = train_df_sorted.iloc[-1]["season"]
    for match_date, date_group in eval_df_sorted.groupby("match_date"):
        if date_group["season"].nunique() != 1:
            raise ValueError("multiple seasons on one date")
        season = date_group.iloc[0]["season"]
        if season != last_season:
            ratings.regress_for_season()
            last_season = season
        # First: generate predictions for ALL matches on this date
        for _, row in date_group.iterrows():
            home_id = mapper.canonical_id(row["home_team"])
            away_id = mapper.canonical_id(row["away_team"])

            ph, pd_val, pa = elo_model.predict_proba(
                row["home_team"], row["away_team"], ratings,
                home_scale=config.home_scale,
                draw_scale=config.draw_scale,
            )

            pred_class = max(
                [("H", ph), ("D", pd_val), ("A", pa)],
                key=lambda x: x[1]
            )[0]

            split = split_for[row["season"]]
            predictions.append(PredictionRecord(
                match_id=row["match_id"],
                model="elo_logistic",
                season=row["season"],
                match_date=match_date.strftime("%Y-%m-%d"),
                home_team=row["home_team"],
                away_team=row["away_team"],
                p_home=ph,
                p_draw=pd_val,
                p_away=pa,
                predicted_class=pred_class,
                actual_outcome=row["outcome"],
                split=split,
                prediction_timestamp=match_date.strftime("%Y-%m-%d"),
                information_cutoff_date=last_completed_date,
                canonical_home_team=mapper.canonicalize(row["home_team"]),
                canonical_away_team=mapper.canonicalize(row["away_team"]),
                pre_match_home_rating=ratings.get(home_id),
                pre_match_away_rating=ratings.get(away_id),
            ))

            home_rate, away_rate = poisson.predict_rates(row["home_team"], row["away_team"])
            po_h, po_d, po_a = poisson_outcome_probabilities(
                home_rate, away_rate, config.poisson_config)
            po_class = max(zip(("H", "D", "A"), (po_h, po_d, po_a)),
                           key=lambda item: item[1])[0]
            predictions.append(PredictionRecord(
                match_id=row["match_id"],
                model="poisson",
                season=row["season"],
                match_date=match_date.strftime("%Y-%m-%d"),
                home_team=row["home_team"],
                away_team=row["away_team"],
                p_home=po_h,
                p_draw=po_d,
                p_away=po_a,
                predicted_class=po_class,
                actual_outcome=row["outcome"],
                split=split,
                prediction_timestamp=match_date.strftime("%Y-%m-%d"),
                information_cutoff_date=last_completed_date,
                canonical_home_team=mapper.canonicalize(row["home_team"]),
                canonical_away_team=mapper.canonicalize(row["away_team"]),
                expected_home_goals=home_rate,
                expected_away_goals=away_rate,
            ))

        # Then: update ratings with all matches from this date
        ratings.update_batch([
            (mapper.canonical_id(row["home_team"]), mapper.canonical_id(row["away_team"]),
             Outcome(row["outcome"])) for _, row in date_group.iterrows()
        ])
        poisson.update_batch([
            (row["home_team"], row["away_team"],
             int(row["home_goals"]), int(row["away_goals"]))
            for _, row in date_group.iterrows()
        ])
        last_completed_date = match_date.strftime("%Y-%m-%d")

    # Save predictions
    pred_df = pd.DataFrame([p.__dict__ for p in predictions])
    (output_dir / "predictions").mkdir(parents=True, exist_ok=True)

    # Save as CSV (parquet may not have pyarrow engine available)
    pred_csv_path = output_dir / "predictions" / "predictions.csv"
    pred_df.to_csv(pred_csv_path, index=False)
    logger.info("Saved %d predictions to %s", len(pred_df), pred_csv_path)

    # Summary
    summary = {}
    for model in ["baseline", "elo_logistic", "poisson"]:
        model_preds = pred_df[pred_df["model"] == model]
        summary[model] = {
            "total_predictions": len(model_preds),
            "train": len(model_preds[model_preds["split"] == "train"]),
            "validation": len(model_preds[model_preds["split"] == "validation"]),
            "test": len(model_preds[model_preds["split"] == "test"]),
        }

    summary["config"] = {
        "training_seasons": sorted(config.training_seasons),
        "validation_seasons": sorted(config.validation_seasons),
        "test_seasons": sorted(config.test_seasons),
        "elo_initial_rating": config.elo_config.initial_rating,
        "elo_k_factor": config.elo_config.k_factor,
        "elo_home_advantage": config.elo_config.home_advantage,
        "elo_season_regression_factor": config.elo_config.season_regression_factor,
        "elo_home_scale": config.home_scale,
        "elo_draw_scale": config.draw_scale,
        "baseline_smoothing_alpha": baseline.config.smoothing_alpha,
        "poisson_prior_matches": config.poisson_config.prior_matches,
        "poisson_tail_tolerance": config.poisson_config.tail_tolerance,
        "poisson_max_goals": config.poisson_config.max_goals,
        "poisson_league_home_rate": poisson.league_home_rate,
        "poisson_league_away_rate": poisson.league_away_rate,
    }
    summary["source_sha256"] = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    correction_path = csv_path.parent / "date_corrections.json"
    if correction_path.exists():
        correction_audit = json.loads(correction_path.read_text(encoding="utf-8"))
        summary["date_corrections"] = {
            "count": correction_audit["count"],
            "reference": correction_audit["reference"],
            "audit_file": correction_path.name,
        }
    anomalous = df[df["season"] == "2019-20"]
    july_2019 = int(anomalous["match_date"].dt.strftime("%Y-%m").eq("2019-07").sum())
    july_2020 = int(anomalous["match_date"].dt.strftime("%Y-%m").eq("2020-07").sum())
    summary["data_quality_notes"] = []
    if july_2019 and not july_2020:
        summary["data_quality_notes"].append(
            f"2019-20 source dates include {july_2019} fixtures in July 2019 and "
            "none in July 2020; development Elo update order may be unreliable."
        )

    summary_path = output_dir / "predictions" / "backtest_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info(
        "Backtest complete: baseline=%d, elo_logistic=%d, poisson=%d predictions",
        summary["baseline"]["total_predictions"],
        summary["elo_logistic"]["total_predictions"],
        summary["poisson"]["total_predictions"],
    )

    return summary


def main() -> int:
    """Entry point for ``python -m edgelab.evaluation``."""
    parser = argparse.ArgumentParser(description="Run date-batched EdgeLab evaluation")
    parser.add_argument("--input", type=Path,
                        default=PROCESSED_DATA_DIR / "premier_league_matches.csv")
    parser.add_argument("--output-dir", type=Path, default=PROCESSED_DATA_DIR)
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    run_backtest(BacktestConfig(), csv_path=args.input, output_dir=args.output_dir)
    from .report import generate_report, load_predictions
    generate_report(load_predictions(args.output_dir / "predictions" / "predictions.csv"),
                    output_dir=args.output_dir)
    return 0


if __name__ == "__main__":
    main()
