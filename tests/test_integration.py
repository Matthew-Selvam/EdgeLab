"""Integration tests that run the ingestion against the real downloaded data."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from edgelab.data.ingest import run_ingestion


# --------------------------------------------------------------------------- #
# These tests only run when the real raw data files are present
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def real_report(tmp_path_factory):
    """Run the full ingestion pipeline against the real data directory.

    Returns the IngestionReport.  Uses a temporary processed dir so we don't
    pollute the actual output during testing.
    """
    raw_dir = Path(__file__).resolve().parent.parent / "data" / "raw" / "openfootball"
    if not raw_dir.exists():
        pytest.skip("Real raw data directory not found")

    processed_dir = tmp_path_factory.mktemp("processed")
    return run_ingestion(raw_dir=raw_dir, processed_dir=processed_dir)


class TestRealData:
    """Tests against the real downloaded Premier League season files."""

    def test_all_seasons_present(self, raw_data_dir):
        """All 10 seasons (2015-16 through 2024-25) should exist."""
        expected = {
            f"{y}-{str(y - 2000 + 1).zfill(2)}.json" for y in range(2015, 2025)
        }
        # More explicitly:
        expected = {
            "2015-16.json", "2016-17.json", "2017-18.json", "2018-19.json",
            "2019-20.json", "2020-21.json", "2021-22.json", "2022-23.json",
            "2023-24.json", "2024-25.json",
        }
        actual = {f.name for f in raw_data_dir.glob("*.json")}
        missing = expected - actual
        assert not missing, f"Missing season files: {missing}"

    def test_total_match_count(self, real_report):
        """Should import 3800 matches (10 seasons x 380 matches)."""
        assert real_report.total_raw_matches == 3800

    def test_all_seasons_imported(self, real_report):
        """All 10 seasons should be in the season_breakdown."""
        assert len(real_report.season_breakdown) == 10
        for season in ["2015-16", "2016-17", "2017-18", "2018-19",
                       "2019-20", "2020-21", "2021-22", "2022-23",
                       "2023-24", "2024-25"]:
            assert season in real_report.season_breakdown
            assert real_report.season_breakdown[season]["raw"] == 380

    def test_all_matches_imported(self, real_report):
        """All 3800 matches should be imported (no skips or rejections)."""
        assert real_report.imported == 3800
        assert real_report.skipped == 0
        assert real_report.rejected == 0

    def test_no_duplicate_match_ids(self, real_report):
        """No duplicate match_ids should exist in the final output."""
        assert real_report.rejected == 0
        """The final CSV should have exactly the canonical columns."""
        processed_dir = Path(__file__).resolve().parent.parent / "data" / "processed"
        csv_path = processed_dir / "premier_league_matches.csv"
        if csv_path.exists():
            with open(csv_path, newline="") as f:
                reader = csv.DictReader(f)
                assert set(reader.fieldnames or []) == {
                    "match_id", "competition", "season", "match_date",
                    "home_team", "away_team", "home_goals", "away_goals",
                    "outcome", "source",
                }

    def test_outcome_values_are_valid(self):
        """All outcome values in the CSV should be H, D, or A."""
        processed_dir = Path(__file__).resolve().parent.parent / "data" / "processed"
        csv_path = processed_dir / "premier_league_matches.csv"
        if not csv_path.exists():
            pytest.skip("CSV not generated yet; run ingestion first")
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            outcomes = {row["outcome"] for row in reader}
        assert outcomes.issubset({"H", "D", "A"})

    def test_report_json_exists(self):
        """The ingestion report JSON should exist after running."""
        processed_dir = Path(__file__).resolve().parent.parent / "data" / "processed"
        report_path = processed_dir / "ingestion_report.json"
        if not report_path.exists():
            pytest.skip("Report not generated yet; run ingestion first")
        with open(report_path) as f:
            data = json.load(f)
        assert "imported" in data
        assert "skipped" in data
        assert "rejected" in data
        assert "season_breakdown" in data

    def test_competition_uniform(self):
        """All matches should have 'Premier League' as competition."""
        processed_dir = Path(__file__).resolve().parent.parent / "data" / "processed"
        csv_path = processed_dir / "premier_league_matches.csv"
        if not csv_path.exists():
            pytest.skip("CSV not generated yet")
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            competitions = {row["competition"] for row in reader}
        assert competitions == {"Premier League"}

    def test_season_provenance_preserved(self):
        """Each row should have a source filename matching its season."""
        processed_dir = Path(__file__).resolve().parent.parent / "data" / "processed"
        csv_path = processed_dir / "premier_league_matches.csv"
        if not csv_path.exists():
            pytest.skip("CSV not generated yet")
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                season = row["season"]
                source = row["source"]
                assert source == f"{season}.json", (
                    f"Source {source} does not match season {season}"
                )
