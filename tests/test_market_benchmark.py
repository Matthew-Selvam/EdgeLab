"""Strict source and paired-sample checks for the separate market reference."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from edgelab.evaluation import market_benchmark
from edgelab.evaluation.market_benchmark import (
    MARKET_NAME, _fixture_scores, assert_matched_sample, closing_probabilities,
    load_market_rows, paired_date_intervals, run_benchmark, sha256,
)
from edgelab.evaluation.metrics import compute_all_metrics


def _tiny_sources(tmp_path: Path, *, duplicate: bool = False,
                  bad_date: bool = False, bad_result: bool = False) -> tuple[Path, Path, Path]:
    canonical_path = tmp_path / "canonical.csv"
    canonical = [
        ["id1", "Premier League", "2016-17", "2016-08-13", "Manchester City FC",
         "Arsenal FC", 2, 1, "H", "test"],
        ["id2", "Premier League", "2016-17", "2016-08-14", "Everton FC",
         "Liverpool FC", 0, 0, "D", "test"],
    ]
    with canonical_path.open("w", newline="") as destination:
        writer = csv.writer(destination)
        writer.writerow(["match_id", "competition", "season", "match_date", "home_team",
                         "away_team", "home_goals", "away_goals", "outcome", "source"])
        writer.writerows(canonical)
    raw_dir = tmp_path / "raw"
    source_path = raw_dir / "1617/E0.csv"
    source_path.parent.mkdir(parents=True)
    source_rows = [
        ["E0", "13/08/16", "Man City", "Arsenal", "2", "1", "H",
         "bad opening price", "2", "3", "4"],
        ["E0", "15/08/16" if bad_date else "14/08/16", "Everton", "Liverpool",
         "0", "0", "H" if bad_result else "D", "", "3", "3", "3"],
    ]
    if duplicate:
        source_rows.append(source_rows[0])
    with source_path.open("w", newline="") as destination:
        writer = csv.writer(destination)
        writer.writerow(["Div", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG",
                         "FTR", "PSH", "PSCH", "PSCD", "PSCA"])
        writer.writerows(source_rows)
    audit_path = tmp_path / "audit.json"
    audit_path.write_text(json.dumps({
        "canonical": {"sha256": sha256(canonical_path)},
        "exceptions": [], "duplicate_canonical_match_assignments": 0,
        "aliases": {"Man City": {"canonical_team": "Manchester City"}},
        "files": [{"season": "2016-17", "relative_path": "data/raw/football-data/1617/E0.csv",
                   "sha256": sha256(source_path), "encoding": "utf-8",
                   "rows_parsed": len(source_rows)}],
    }))
    return canonical_path, raw_dir, audit_path


def test_closing_overround_and_normalization_known_example() -> None:
    q, p, overround = closing_probabilities(("2", "3", "4"))
    assert q == pytest.approx((0.5, 1 / 3, 0.25))
    assert overround == pytest.approx(1 / 12)
    assert p == pytest.approx(tuple(value / (13 / 12) for value in q))
    assert sum(p) == pytest.approx(1)
    assert closing_probabilities(("2", "4", "4"))[2] == pytest.approx(0)


@pytest.mark.parametrize("bad", ["", "abc", "0", "-2", "1", "nan", "inf"])
def test_closing_prices_reject_invalid_values(bad: str) -> None:
    with pytest.raises(ValueError):
        closing_probabilities(("2", bad, "4"))


def test_only_closing_fields_are_read_and_aliases_match(tmp_path: Path) -> None:
    canonical, raw, audit = _tiny_sources(tmp_path)
    rows = load_market_rows(canonical, raw, audit)
    assert rows["match_id"].tolist() == ["id1", "id2"]
    assert rows.loc[0, "q_home"] == pytest.approx(0.5)
    assert rows.loc[0, "p_home"] == pytest.approx(6 / 13)
    assert rows["p_home"].add(rows["p_draw"]).add(rows["p_away"]).eq(1).all() or np.allclose(
        rows[["p_home", "p_draw", "p_away"]].sum(axis=1), 1)
    assert set(rows["actual_outcome"]) == {"H", "D"}


@pytest.mark.parametrize(("change", "message"), [
    ({"duplicate": True}, "duplicate odds fixture"),
    ({"bad_date": True}, "unmatched odds fixture"),
    ({"bad_result": True}, "result mismatch"),
])
def test_fixture_mismatches_fail_loudly(tmp_path: Path, change: dict, message: str) -> None:
    canonical, raw, audit = _tiny_sources(tmp_path, **change)
    with pytest.raises(ValueError, match=message):
        load_market_rows(canonical, raw, audit)


def test_source_hash_change_is_rejected(tmp_path: Path) -> None:
    canonical, raw, audit = _tiny_sources(tmp_path)
    path = raw / "1617/E0.csv"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="source hash differs"):
        load_market_rows(canonical, raw, audit)


def test_matched_sample_equality_checks_ids_dates_and_outcomes() -> None:
    market = pd.DataFrame({"match_date": ["2023-08-11"], "actual_outcome": ["H"]},
                          index=pd.Index(["id1"], name="match_id"))
    assert_matched_sample(market, market.copy(), "2023-24", "poisson")
    with pytest.raises(ValueError, match="fixture sample"):
        assert_matched_sample(market, market.rename(index={"id1": "id2"}),
                              "2023-24", "poisson")
    with pytest.raises(ValueError, match="actual_outcome"):
        assert_matched_sample(market, market.assign(actual_outcome="A"),
                              "2023-24", "poisson")


def test_metric_scores_and_paired_interval_toy_example() -> None:
    y = np.array([0, 2])
    dates = np.array(["2023-08-11", "2023-08-12"])
    market = np.array([[0.5, 0.25, 0.25], [0.1, 0.2, 0.7]])
    weaker = np.array([[0.4, 0.3, 0.3], [0.2, 0.3, 0.5]])
    metrics = compute_all_metrics(["H", "A"], [tuple(row) for row in market], MARKET_NAME)
    assert metrics.brier_score == pytest.approx(0.2575)
    assert metrics.log_loss == pytest.approx((-math.log(0.5) - math.log(0.7)) / 2)
    assert metrics.accuracy == 1
    scores = _fixture_scores(y, market)
    assert scores["brier_score"].tolist() == pytest.approx([0.375, 0.14])
    probabilities = {MARKET_NAME: market, "baseline": weaker,
                     "elo_logistic": weaker, "poisson": weaker}
    first = paired_date_intervals(y, dates, probabilities, replicates=50, seed=7)
    second = paired_date_intervals(y, dates, probabilities, replicates=50, seed=7)
    assert first == second
    assert first["baseline"]["brier_score"]["difference"] < 0


def test_frozen_reference_mismatch_stops_before_writing(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(market_benchmark, "PROJECT_ROOT", tmp_path)
    canonical, raw, audit = _tiny_sources(tmp_path)
    frozen = tmp_path / "frozen"
    robustness = tmp_path / "robustness"
    calibration = tmp_path / "calibration"
    for folder in (frozen, robustness, calibration):
        folder.mkdir()
    (frozen / "predictions.csv").write_text("frozen predictions")
    (frozen / "metrics.json").write_text("frozen metrics")
    (robustness / "metrics.json").write_text(json.dumps({
        "protocol": {"source_sha256": sha256(canonical)},
        "frozen_reference": {"predictions_sha256": "wrong", "metrics_sha256": "wrong"},
    }))
    before = {path: sha256(path) for path in frozen.iterdir()}
    output = tmp_path / "data/processed/market_benchmark"
    with pytest.raises(ValueError, match="frozen forecasts differ"):
        run_benchmark(canonical, raw, audit, frozen, robustness, calibration, output)
    assert not output.exists()
    assert before == {path: sha256(path) for path in frozen.iterdir()}


def test_output_cannot_overwrite_frozen_directory(tmp_path: Path) -> None:
    canonical, raw, audit = _tiny_sources(tmp_path)
    frozen = tmp_path / "frozen"
    robustness = tmp_path / "robustness"
    calibration = tmp_path / "calibration"
    with pytest.raises(ValueError, match="output directory must be separate"):
        run_benchmark(canonical, raw, audit, frozen, robustness, calibration,
                      frozen / "market")


def test_default_output_and_cli_stay_in_ignored_tree(monkeypatch, capsys) -> None:
    default = market_benchmark.DEFAULT_OUTPUT
    assert market_benchmark.validate_output_directory(default) == default.resolve()
    calls = []

    def fake_run(**kwargs):
        calls.append(kwargs["output_dir"])
        return {"coverage": {"scored_fixtures": 3420}, "periods": [None] * 9}

    monkeypatch.setattr(market_benchmark, "run_benchmark", fake_run)
    monkeypatch.setattr("sys.argv", ["edgelab-market-benchmark"])
    market_benchmark.main()
    assert calls == [default]
    assert "3420 matched scored fixtures" in capsys.readouterr().out


@pytest.mark.parametrize("suffix", ["", "rerun", "a/../rerun"])
def test_safe_output_subdirectories(tmp_path: Path, monkeypatch, suffix: str) -> None:
    monkeypatch.setattr(market_benchmark, "PROJECT_ROOT", tmp_path)
    allowed = tmp_path / "data/processed/market_benchmark"
    assert market_benchmark.validate_output_directory(allowed / suffix).is_relative_to(allowed)


@pytest.mark.parametrize("path", ["public", "data/processed/market_benchmark/../public",
                                   "data/processed/market_benchmark/../../../public",
                                   "data/processed/market_benchmark-copy"])
def test_external_output_rejected_before_input_reads(tmp_path: Path, monkeypatch, path: str) -> None:
    monkeypatch.setattr(market_benchmark, "PROJECT_ROOT", tmp_path)
    monkeypatch.chdir(tmp_path)
    target = Path(path)
    with pytest.raises(ValueError, match="restricted market output"):
        run_benchmark(output_dir=target)
    with pytest.raises(ValueError, match="restricted market output"):
        run_benchmark(output_dir=target.absolute())
    assert not (tmp_path / "public").exists()


@pytest.mark.parametrize("link_at", ["data", "data/processed/market_benchmark",
                                   "data/processed/market_benchmark/escape"])
def test_directory_symlink_cannot_redirect_outputs(tmp_path: Path, monkeypatch,
                                                  link_at: str) -> None:
    monkeypatch.setattr(market_benchmark, "PROJECT_ROOT", tmp_path)
    external = tmp_path / "public"
    external.mkdir()
    link = tmp_path / link_at
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(external, target_is_directory=True)
    target = (link if link_at.endswith("escape") else
              tmp_path / "data/processed/market_benchmark")
    with pytest.raises(ValueError, match="restricted market output"):
        market_benchmark.validate_output_directory(target)
    assert not list(external.iterdir())


@pytest.mark.parametrize("filename", market_benchmark.OUTPUT_FILENAMES)
def test_existing_output_file_symlink_cannot_escape(tmp_path: Path, monkeypatch,
                                                   filename: str) -> None:
    monkeypatch.setattr(market_benchmark, "PROJECT_ROOT", tmp_path)
    allowed = tmp_path / "data/processed/market_benchmark"
    allowed.mkdir(parents=True)
    external = tmp_path / "public.csv"
    external.write_text("preserve")
    (allowed / filename).symlink_to(external)
    with pytest.raises(ValueError, match="restricted market output"):
        market_benchmark.validate_output_directory(allowed)
    assert external.read_text() == "preserve"


def test_cli_rejects_external_output_without_reading_data(tmp_path: Path, monkeypatch,
                                                        capsys) -> None:
    monkeypatch.setattr("sys.argv", ["edgelab-market-benchmark", "--output-dir", str(tmp_path)])
    with pytest.raises(SystemExit) as error:
        market_benchmark.main()
    assert error.value.code == 2
    assert "restricted market output" in capsys.readouterr().err
