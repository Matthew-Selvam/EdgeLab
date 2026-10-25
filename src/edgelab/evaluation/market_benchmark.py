"""Retrospective Pinnacle closing-line reference, using saved forecasts only.

No model is fitted here. Derived fixture-level market probabilities stay under
the gitignored processed directory because redistribution permission is unclear.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from ..data.ingest import PROJECT_ROOT, PROCESSED_DATA_DIR
from ..models.teams import TeamAliasMapper
from .backtest import load_matches
from .metrics import compute_all_metrics
from .report import MODEL_NAMES

MARKET_NAME = "pinnacle_closing"
PRICE_FIELDS = ("PSCH", "PSCD", "PSCA")
PROB_FIELDS = ("p_home", "p_draw", "p_away")
OUTCOMES = "HDA"
FOLDS = tuple(f"{year}-{str(year + 1)[-2:]}" for year in range(2016, 2023))
PERIODS = tuple((season, "development_walk_forward") for season in FOLDS) + (
    ("2023-24", "validation"), ("2024-25", "test"))
DEFAULT_RAW = PROCESSED_DATA_DIR.parent / "raw/football-data"
DEFAULT_AUDIT = PROCESSED_DATA_DIR / "odds_audit/source_audit.json"
DEFAULT_CANONICAL = PROCESSED_DATA_DIR / "premier_league_matches.csv"
DEFAULT_ROBUSTNESS = PROCESSED_DATA_DIR / "robustness"
DEFAULT_FROZEN = PROCESSED_DATA_DIR / "predictions"
DEFAULT_CALIBRATION = PROCESSED_DATA_DIR / "calibration"
DEFAULT_OUTPUT = PROCESSED_DATA_DIR / "market_benchmark"
OUTPUT_FILENAMES = ("market_predictions.csv", "reliability_bins.csv", "metrics.json", "REPORT.md")


def validate_output_directory(output_dir: Path) -> Path:
    """Confine restricted derivatives to this checkout's ignored benchmark tree.

    Resolve the requested directory and individual output targets, including
    existing symlinks. The allowed anchor deliberately stays lexical beneath the
    resolved project root: redirecting data/ or market_benchmark/ through an
    external symlink must not expand the allowed area. No external override is
    provided. This protects normal local use, not concurrent hostile filesystem
    mutation during a run.
    """
    allowed = PROJECT_ROOT.resolve() / "data" / "processed" / "market_benchmark"
    resolved = output_dir.resolve()
    targets = [resolved, *(resolved / name for name in OUTPUT_FILENAMES)]
    if any(not target.resolve().is_relative_to(allowed) for target in targets):
        raise ValueError(
            "restricted market output must remain within data/processed/market_benchmark; "
            "external paths and escaping symlinks are prohibited"
        )
    return resolved


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def closing_probabilities(prices: tuple[str, str, str]) -> tuple[tuple[float, ...],
                                                                   tuple[float, ...], float]:
    """Return q=1/odds, proportional de-vig p=q/sum(q), and sum(q)-1."""
    if len(prices) != 3:
        raise ValueError("closing 1X2 needs exactly three prices")
    try:
        odds = tuple(float(value) for value in prices)
    except (TypeError, ValueError) as exc:
        raise ValueError("closing prices must be numeric") from exc
    if not all(math.isfinite(value) and value > 1 for value in odds):
        raise ValueError("closing decimal prices must be finite and greater than one")
    q = tuple(1.0 / value for value in odds)
    total = math.fsum(q)
    if not math.isfinite(total) or total <= 0:
        raise ValueError("invalid implied-probability sum")
    p = tuple(value / total for value in q)
    if not all(math.isfinite(value) and value > 0 for value in p) or not math.isclose(
            math.fsum(p), 1.0, abs_tol=1e-12):
        raise ValueError("invalid normalized closing probabilities")
    return q, p, total - 1.0


def _parse_date(value: str) -> str:
    year = value.rsplit("/", 1)[-1]
    fmt = "%d/%m/%Y" if len(year) == 4 else "%d/%m/%y"
    return datetime.strptime(value, fmt).date().isoformat()


def _index_canonical(canonical: pd.DataFrame, mapper: TeamAliasMapper) -> dict[tuple, dict]:
    index: dict[tuple, dict] = {}
    for row in canonical.itertuples(index=False):
        key = (row.season, row.match_date.strftime("%Y-%m-%d"),
               mapper.canonicalize(row.home_team), mapper.canonicalize(row.away_team))
        if key in index:
            raise ValueError(f"ambiguous canonical fixture: {key}")
        index[key] = row._asdict()
    return index


def load_market_rows(canonical_path: Path, raw_dir: Path, audit_path: Path) -> pd.DataFrame:
    """Strictly reconcile every audited source row to one canonical fixture."""
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("exceptions") or audit.get("duplicate_canonical_match_assignments"):
        raise ValueError("odds source audit contains unresolved fixture exceptions")
    if sha256(canonical_path) != audit["canonical"]["sha256"]:
        raise ValueError("canonical file differs from verified odds audit")
    canonical = load_matches(canonical_path)
    mapper = TeamAliasMapper()
    aliases = {name: item["canonical_team"] for name, item in audit["aliases"].items()}
    index = _index_canonical(canonical, mapper)
    seen_keys: set[tuple] = set()
    seen_ids: set[str] = set()
    records: list[dict] = []
    if not audit["files"] or len({item["season"] for item in audit["files"]}) != len(
            audit["files"]):
        raise ValueError("odds audit has no files or duplicate season entries")
    for item in audit["files"]:
        season = item["season"]
        folder = Path(item["relative_path"]).parent.name
        path = raw_dir / folder / "E0.csv"
        if not path.is_file() or sha256(path) != item["sha256"]:
            raise ValueError(f"source hash differs from verified audit: {path}")
        with path.open("r", encoding=item["encoding"], newline="") as source:
            reader = csv.DictReader(source)
            required = {"Div", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR",
                        *PRICE_FIELDS}
            if not reader.fieldnames or required - set(reader.fieldnames):
                raise ValueError(f"missing closing-market fields: {path}")
            count = 0
            for line, row in enumerate(reader, 2):
                count += 1
                if None in row or any(value is None for value in row.values()):
                    raise ValueError(f"malformed CSV row: {path}:{line}")
                if row["Div"] != "E0":
                    raise ValueError(f"unexpected division: {path}:{line}")
                date = _parse_date(row["Date"])
                home = mapper.canonicalize(aliases.get(row["HomeTeam"], row["HomeTeam"]))
                away = mapper.canonicalize(aliases.get(row["AwayTeam"], row["AwayTeam"]))
                key = (season, date, home, away)
                if key in seen_keys:
                    raise ValueError(f"duplicate odds fixture: {key}")
                seen_keys.add(key)
                match = index.get(key)
                if match is None:
                    raise ValueError(f"unmatched odds fixture: {key}")
                match_id = match["match_id"]
                if match_id in seen_ids:
                    raise ValueError(f"duplicate canonical assignment: {match_id}")
                seen_ids.add(match_id)
                if (int(row["FTHG"]), int(row["FTAG"]), row["FTR"]) != (
                        match["home_goals"], match["away_goals"], match["outcome"]):
                    raise ValueError(f"result mismatch: {path}:{line} {match_id}")
                q, p, overround = closing_probabilities(
                    tuple(row[field] for field in PRICE_FIELDS))
                records.append({
                    "match_id": match_id, "season": season, "match_date": date,
                    "actual_outcome": match["outcome"], "source_file": item["relative_path"],
                    "source_row": line, "q_home": q[0], "q_draw": q[1], "q_away": q[2],
                    "p_home": p[0], "p_draw": p[1], "p_away": p[2],
                    "overround": overround,
                })
            if count != item["rows_parsed"]:
                raise ValueError(f"source row count differs from verified audit: {path}")
    if seen_ids != set(canonical["match_id"]):
        missing = set(canonical["match_id"]) - seen_ids
        raise ValueError(f"unmatched canonical fixtures: {len(missing)}")
    return pd.DataFrame(records).sort_values(["season", "match_date", "match_id"]).reset_index(
        drop=True)


def load_model_rows(canonical: pd.DataFrame, frozen_dir: Path,
                    robustness_dir: Path) -> pd.DataFrame:
    """Validate saved out-of-sample rows and exact fixture sets, never reforecast."""
    frozen = pd.read_csv(frozen_dir / "predictions.csv", keep_default_na=False)
    development = pd.read_csv(robustness_dir / "walk_forward_predictions.csv",
                              keep_default_na=False)
    rows = pd.concat([development, frozen], ignore_index=True)
    if set(rows["model"]) != set(MODEL_NAMES):
        raise ValueError("unexpected model identifiers in saved predictions")
    if set(zip(rows["season"], rows["split"])) != set(PERIODS):
        raise ValueError("saved prediction periods differ from fixed protocol")
    canonical_by_id = canonical.set_index("match_id")
    for season, split in PERIODS:
        subset = rows[(rows["season"] == season) & (rows["split"] == split)]
        expected = set(canonical[canonical["season"] == season]["match_id"])
        if len(expected) != 380:
            raise ValueError(f"canonical season count differs from protocol: {season}")
        for model in MODEL_NAMES:
            part = subset[subset["model"] == model]
            if len(part) != len(expected) or part["match_id"].duplicated().any() or set(
                    part["match_id"]) != expected:
                raise ValueError(f"saved model sample differs from canonical: {season} {model}")
            for row in part.itertuples(index=False):
                match = canonical_by_id.loc[row.match_id]
                if (row.match_date != match.match_date.strftime("%Y-%m-%d")
                        or row.actual_outcome != match.outcome):
                    raise ValueError(f"saved forecast fixture disagreement: {row.match_id}")
                if row.information_cutoff_date >= row.match_date:
                    raise ValueError(f"nonhistorical model information cutoff: {row.match_id}")
                probabilities = [getattr(row, key) for key in PROB_FIELDS]
                if (not all(math.isfinite(value) and 0 <= value <= 1
                            for value in probabilities)
                        or not math.isclose(math.fsum(probabilities), 1.0, abs_tol=1e-9)):
                    raise ValueError(f"invalid saved model probability: {row.match_id}")
            if (part.groupby("match_date")["information_cutoff_date"].nunique() != 1).any():
                raise ValueError(f"same-date cutoff disagreement: {season} {model}")
    return rows


def _fixture_scores(outcomes: np.ndarray, probabilities: np.ndarray) -> dict[str, np.ndarray]:
    one_hot = np.eye(3)[outcomes]
    return {
        "brier_score": np.square(probabilities - one_hot).sum(axis=1),
        "log_loss": -np.log(np.clip(probabilities[np.arange(len(outcomes)), outcomes],
                                    1e-15, 1 - 1e-15)),
        "accuracy": (probabilities.argmax(axis=1) == outcomes).astype(float),
    }


def assert_matched_sample(market_part: pd.DataFrame, model_part: pd.DataFrame,
                          season: str, model: str) -> None:
    """Require identical IDs, dates, and outcomes before any score comparison."""
    if model_part.index.has_duplicates or not model_part.index.equals(market_part.index):
        raise ValueError(f"market/model fixture sample differs: {season} {model}")
    for column in ("match_date", "actual_outcome"):
        if not model_part[column].equals(market_part[column]):
            raise ValueError(f"market/model {column} differs: {season} {model}")


def paired_date_intervals(outcomes: np.ndarray, dates: np.ndarray,
                          probabilities: dict[str, np.ndarray],
                          replicates: int = 1000, seed: int = 20261022) -> dict:
    """Market-minus-model paired 95% percentile intervals, by calendar date."""
    if replicates < 1 or len(outcomes) == 0:
        raise ValueError("nonempty sample and positive replicate count required")
    unique_dates, date_ids = np.unique(dates, return_inverse=True)
    generator = np.random.default_rng(seed)
    draws = generator.integers(0, len(unique_dates),
                              size=(replicates, len(unique_dates)))
    counts = np.bincount(date_ids, minlength=len(unique_dates))
    denominators = counts[draws].sum(axis=1)
    market_scores = _fixture_scores(outcomes, probabilities[MARKET_NAME])
    output: dict[str, dict] = {}
    for model in MODEL_NAMES:
        model_scores = _fixture_scores(outcomes, probabilities[model])
        output[model] = {}
        for metric in ("brier_score", "log_loss", "accuracy"):
            paired = market_scores[metric] - model_scores[metric]
            block_sums = np.bincount(date_ids, weights=paired, minlength=len(unique_dates))
            estimates = block_sums[draws].sum(axis=1) / denominators
            low, high = np.quantile(estimates, [0.025, 0.975])
            output[model][metric] = {
                "difference": float(paired.mean()),
                "ci95_low": float(low), "ci95_high": float(high),
            }
    return output


def _input_hashes(canonical_path: Path, audit_path: Path, raw_dir: Path,
                  frozen_dir: Path, robustness_dir: Path,
                  calibration_dir: Path) -> dict[str, str]:
    paths = [canonical_path, audit_path]
    paths += sorted(raw_dir.glob("*/E0.csv"))
    for folder in (frozen_dir, robustness_dir, calibration_dir):
        paths += sorted(path for path in folder.rglob("*") if path.is_file())
    return {str(path.resolve()): sha256(path) for path in paths}


def _report(payload: dict) -> str:
    lines = [
        "# Retrospective Pinnacle closing-market benchmark", "",
        "[Football-Data.co.uk](https://football-data.co.uk/notes.txt)'s "
        "PSCH/PSCD/PSCA columns supply Pinnacle closing H/D/A "
        "decimal odds. This is a historical closing-line reference, **not** a strictly "
        "time-aligned forecast contest. Odds quote timestamps are unavailable; match "
        "kickoff time is absent through 2018–19, and EdgeLab forecasts use date-level "
        "information cutoffs.", "",
        "For each price o, raw implied q=1/o. Proportional margin removal gives "
        "p=q/sum(q), with overround=sum(q)-1. This is one margin-removal convention, "
        "not the uniquely correct one. No model was refitted or tuned.", "",
        "Multiclass Brier = mean sum of three squared probability errors (lower better); "
        "log loss = mean -log probability of the actual class (lower better); accuracy "
        "= argmax hit rate (higher better). ECE is 10 equal-width-bin, one-versus-rest "
        "expected calibration error for each H/D/A outcome. Its value depends on "
        "binning and sparse bins can be unstable.", "",
        "## Coverage", "",
        "All 3,800 audited odds fixtures matched canonical results. The 2015–16 "
        "season has 380 closing triplets but no eligible prior-season model forecasts "
        "and is excluded from scoring. Each of the next seven development folds and "
        "the separate validation and test seasons scores the same 380 fixtures "
        "for the market and all three models.", "",
        "Mean closing overround by source season (sum of raw implied probabilities minus one):", "",
        "| Season | n | Mean | Min | Max | Negative count |",
        "|:--|--:|--:|--:|--:|--:|",
    ]
    for season, item in payload["coverage"]["overround_by_season"].items():
        lines.append(f"| {season} | {item['n']} | {item['mean']:.4f} | "
                     f"{item['minimum']:.4f} | {item['maximum']:.4f} | "
                     f"{item['negative_count']} |")
    lines += [
        "", "## Scores on identical fixtures", "",
        "| Period | Season | n | Model | Brier | Log loss | Accuracy | ECE H | ECE D | ECE A |",
        "|:--|:--|--:|:--|--:|--:|--:|--:|--:|--:|",
    ]
    for period in payload["periods"]:
        for model, item in period["models"].items():
            ece = item["calibration_errors"]
            lines.append(
                f"| {period['split']} | {period['season']} | {item['n_matches']} | {model} | "
                f"{item['brier_score']:.4f} | {item['log_loss']:.4f} | "
                f"{item['accuracy']:.3f} | {ece['H']:.3f} | {ece['D']:.3f} | {ece['A']:.3f} |"
            )
    lines += [
        "", "## Paired differences", "",
        "Sign convention is **market minus model**. Negative Brier/log-loss differences "
        "and positive accuracy differences favour the market descriptively. "
        "Intervals are 95% percentile intervals from 1,000 paired bootstrap resamples "
        "of calendar dates within each period, retaining all same-date fixtures. "
        "They do not adjust for serial dependence across dates, season-to-season "
        "variation, or the many comparisons shown. They are not a proof of "
        "out-of-sample long-run superiority or achievable betting returns.", "",
        "| Period | Season | Comparator | Δ Brier [95%] | Δ Log loss [95%] | Δ Accuracy [95%] |",
        "|:--|:--|:--|:--|:--|:--|",
    ]
    for period in payload["periods"]:
        for model, metrics in period["paired_market_minus_model"].items():
            def cell(key: str) -> str:
                value = metrics[key]
                return (f"{value['difference']:+.4f} "
                        f"[{value['ci95_low']:+.4f}, {value['ci95_high']:+.4f}]")
            lines.append(f"| {period['split']} | {period['season']} | {model} | "
                         f"{cell('brier_score')} | {cell('log_loss')} | "
                         f"{cell('accuracy')} |")
    lines += [
        "", "Fixture-level derived q, p and overround are saved locally under the "
        "gitignored market_benchmark directory. No raw odds or row-level prices "
        "are redistributed. Attribution: Football-Data.co.uk and Pinnacle. "
        "The [free-data policy](https://football-data.co.uk/contact.php) "
        "permits private personal/academic research via manual "
        "or light download; separate permission would be needed for other uses. "
        "See docs/ODDS_SOURCE_AUDIT.md for source and field definitions.", "",
        "Validation is 2023–24 and held-out test is 2024–25. Test covers one season "
        "only; no market result was used to select model settings or benchmark method. "
        "The source does not establish when closing quotes were observed relative "
        "to EdgeLab's date-batched forecasts. Do not infer tradeability.", "",
    ]
    return "\n".join(lines)


def run_benchmark(canonical_path: Path = DEFAULT_CANONICAL,
                  raw_dir: Path = DEFAULT_RAW, audit_path: Path = DEFAULT_AUDIT,
                  frozen_dir: Path = DEFAULT_FROZEN,
                  robustness_dir: Path = DEFAULT_ROBUSTNESS,
                  calibration_dir: Path = DEFAULT_CALIBRATION,
                  output_dir: Path = DEFAULT_OUTPUT,
                  bootstrap_replicates: int = 1000,
                  bootstrap_seed: int = 20261022) -> dict:
    """Build and score a separate local retrospective market benchmark."""
    output_resolved = output_dir.resolve()
    protected = (raw_dir, frozen_dir, robustness_dir, calibration_dir)
    if any(output_resolved == folder.resolve() or output_resolved.is_relative_to(
            folder.resolve()) for folder in protected) or output_resolved in {
            canonical_path.resolve(), audit_path.resolve(), audit_path.parent.resolve()}:
        raise ValueError("benchmark output directory must be separate from inputs")
    output_dir = validate_output_directory(output_dir)
    before = _input_hashes(canonical_path, audit_path, raw_dir, frozen_dir,
                           robustness_dir, calibration_dir)
    robustness = json.loads((robustness_dir / "metrics.json").read_text(encoding="utf-8"))
    if robustness["protocol"]["source_sha256"] != before[str(canonical_path.resolve())]:
        raise ValueError("robustness study and canonical source differ")
    reference = robustness["frozen_reference"]
    if (reference["predictions_sha256"] != sha256(frozen_dir / "predictions.csv")
            or reference["metrics_sha256"] != sha256(frozen_dir / "metrics.json")):
        raise ValueError("frozen forecasts differ from robustness reference")
    frozen_metrics = json.loads((frozen_dir / "metrics.json").read_text(encoding="utf-8"))
    canonical = load_matches(canonical_path)
    market = load_market_rows(canonical_path, raw_dir, audit_path)
    models = load_model_rows(canonical, frozen_dir, robustness_dir)
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if ({item["season"] for item in audit["files"]} != {
            "2015-16", *FOLDS, "2023-24", "2024-25"}
            or len(market) != 3800 or len(models) != 3420 * len(MODEL_NAMES)):
        raise ValueError("source or out-of-sample coverage differs from frozen protocol")
    periods = []
    reliability_rows = []
    for season, split in PERIODS:
        market_part = market[market["season"] == season].set_index("match_id").sort_index()
        model_part = models[(models["season"] == season) & (models["split"] == split)]
        if len(market_part) != 380:
            raise ValueError(f"market coverage incomplete: {season}")
        y_labels = market_part["actual_outcome"].tolist()
        y = np.array([OUTCOMES.index(label) for label in y_labels], dtype=int)
        dates = market_part["match_date"].to_numpy()
        probabilities = {MARKET_NAME: market_part[list(PROB_FIELDS)].to_numpy(dtype=float)}
        scored = {}
        for name in (MARKET_NAME, *MODEL_NAMES):
            if name == MARKET_NAME:
                frame = market_part
            else:
                frame = model_part[model_part["model"] == name].set_index("match_id").sort_index()
                assert_matched_sample(market_part, frame, season, name)
                probabilities[name] = frame[list(PROB_FIELDS)].to_numpy(dtype=float)
            values = [tuple(row) for row in probabilities[name]]
            metrics = compute_all_metrics(y_labels, values, name)
            scored[name] = {key: value for key, value in asdict(metrics).items()
                            if key != "reliability_table" and key != "model_name"}
            if name != MARKET_NAME:
                saved = (robustness["development"]["seasons"][season][name]
                         if split == "development_walk_forward" else
                         frozen_metrics[split][name])
                if (saved["n_matches"] != metrics.n_matches or any(
                        not math.isclose(getattr(metrics, key), saved[key], abs_tol=1e-12)
                        for key in ("brier_score", "log_loss", "accuracy"))):
                    raise ValueError(f"saved model score disagreement: {season} {name}")
            for entry in metrics.reliability_table:
                reliability_rows.append({"split": split, "season": season, "model": name,
                                         "n_matches": 380, **entry})
        intervals = paired_date_intervals(y, dates, probabilities,
                                          bootstrap_replicates, bootstrap_seed)
        periods.append({"season": season, "split": split, "n_fixtures": 380,
                        "models": scored, "paired_market_minus_model": intervals})
    overround_by_season = {}
    for season, group in market.groupby("season", sort=True):
        values = group["overround"]
        overround_by_season[season] = {
            "n": len(group), "mean": float(values.mean()),
            "minimum": float(values.min()), "maximum": float(values.max()),
            "negative_count": int((values < 0).sum()),
        }
    coverage = {
        "canonical_fixtures": len(canonical), "market_matched": len(market),
        "scored_fixtures": 3420, "per_scored_period": 380,
        "excluded": {"season": "2015-16", "fixtures": 380,
                     "reason": "no eligible completed prior season for out-of-sample model forecasts"},
        "unmatched_market": 0, "unmatched_canonical": 0,
        "overround_by_season": overround_by_season,
    }
    payload = {
        "protocol": {
            "name": "retrospective_pinnacle_closing_reference",
            "odds_fields": list(PRICE_FIELDS),
            "margin_removal": "q_i=1/o_i; p_i=q_i/sum(q); overround=sum(q)-1",
            "class_order": "HDA", "bootstrap_unit": "match_date",
            "bootstrap_replicates": bootstrap_replicates, "bootstrap_seed": bootstrap_seed,
            "difference_convention": "market minus model",
            "reliability_bins": 10,
            "periods": [{"season": season, "split": split} for season, split in PERIODS],
            "information_limit": "odds quote timestamps unavailable; model forecasts have date-level cutoffs",
        },
        "input_sha256": before, "coverage": coverage, "periods": periods,
    }
    if before != _input_hashes(canonical_path, audit_path, raw_dir, frozen_dir,
                                robustness_dir, calibration_dir):
        raise ValueError("an input or frozen artifact changed during benchmark")
    # Recheck before writes in case an output target changed while scoring.
    output_dir = validate_output_directory(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    market.to_csv(output_dir / "market_predictions.csv", index=False, float_format="%.17g")
    pd.DataFrame(reliability_rows).to_csv(output_dir / "reliability_bins.csv", index=False)
    (output_dir / "metrics.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "REPORT.md").write_text(_report(payload), encoding="utf-8")
    if before != _input_hashes(canonical_path, audit_path, raw_dir, frozen_dir,
                                robustness_dir, calibration_dir):
        raise ValueError("an input or frozen artifact changed during output write")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT,
                        help="local directory within data/processed/market_benchmark only")
    args = parser.parse_args()
    try:
        validate_output_directory(args.output_dir)
    except ValueError as exc:
        parser.error(str(exc))
    result = run_benchmark(output_dir=args.output_dir)
    print(f"Retrospective closing-market benchmark: "
          f"{result['coverage']['scored_fixtures']} matched scored fixtures, "
          f"{len(result['periods'])} periods; outputs in {args.output_dir}")


if __name__ == "__main__":
    main()
