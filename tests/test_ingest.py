"""Tests for the ingestion pipeline."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from edgelab.data.ingest import (
    CSV_COLUMNS,
    _build_match_id,
    _extract_season,
    convert_match,
    deduplicate_records,
    ingest_file,
    run_ingestion,
    write_csv,
    write_report,
)
from edgelab.data.models import IngestionReport, MatchRecord, Outcome


# --------------------------------------------------------------------------- #
# Helper / parser tests
# --------------------------------------------------------------------------- #


class TestExtractSeason:
    """Tests for _extract_season."""

    def test_valid_filename(self):
        assert _extract_season("2024-25.json") == "2024-25"

    def test_invalid_filename(self):
        assert _extract_season("premier_league.csv") is None

    def test_partial_filename(self):
        assert _extract_season("2024.json") is None

    def test_none_filename(self):
        assert _extract_season(None) is None  # type: ignore[arg-type]


class TestBuildMatchId:
    """Tests for _build_match_id."""

    def test_basic_id(self):
        mid = _build_match_id("2024-25", "2024-08-16", "Manchester United FC", "Fulham FC")
        assert mid == "EPL_2024-25_2024-08-16_manchesterunitedfc_vs_fulhamfc"

    def test_id_is_deterministic(self):
        """Same inputs should produce the same match_id."""
        id1 = _build_match_id("2024-25", "2024-08-16", "Man Utd", "Fulham")
        id2 = _build_match_id("2024-25", "2024-08-16", "Man Utd", "Fulham")
        assert id1 == id2

    def test_id_differs_for_different_teams(self):
        id1 = _build_match_id("2024-25", "2024-08-16", "TeamA", "TeamB")
        id2 = _build_match_id("2024-25", "2024-08-16", "TeamB", "TeamA")
        assert id1 != id2


# --------------------------------------------------------------------------- #
# convert_match tests
# --------------------------------------------------------------------------- #


class TestConvertMatch:
    """Tests for convert_match."""

    def test_valid_home_win(self, sample_match_data: dict):
        rec = convert_match(sample_match_data, "2024-25", "2024-25.json", 0)
        assert rec is not None
        assert rec.home_goals == 1
        assert rec.away_goals == 0
        assert rec.outcome == Outcome.HOME_WIN
        assert rec.competition == "Premier League"
        assert rec.season == "2024-25"
        assert rec.source == "2024-25.json"

    def test_valid_away_win(self):
        match = {
            "round": "Matchday 1",
            "date": "2024-08-17",
            "time": "12:30",
            "team1": "Ipswich Town FC",
            "team2": "Liverpool FC",
            "score": {"ht": [0, 0], "ft": [0, 2]},
        }
        rec = convert_match(match, "2024-25", "2024-25.json", 1)
        assert rec is not None
        assert rec.home_goals == 0
        assert rec.away_goals == 2
        assert rec.outcome == Outcome.AWAY_WIN

    def test_valid_draw(self):
        match = {
            "round": "Matchday 1",
            "date": "2024-08-17",
            "time": "15:00",
            "team1": "Arsenal FC",
            "team2": "Wolverhampton Wanderers FC",
            "score": {"ht": [1, 0], "ft": [1, 1]},
        }
        rec = convert_match(match, "2024-25", "2024-25.json", 2)
        assert rec is not None
        assert rec.home_goals == 1
        assert rec.away_goals == 1
        assert rec.outcome == Outcome.DRAW

    def test_missing_ft_score(self):
        match = {
            "round": "Matchday 1",
            "date": "2024-08-16",
            "team1": "Team A",
            "team2": "Team B",
            "score": {"ht": [0, 0]},
        }
        rec = convert_match(match, "2024-25", "2024-25.json", 0)
        assert rec is None

    def test_missing_ht_is_ok(self):
        """Matches without ht should still convert if ft is present."""
        match = {
            "round": "Matchday 1",
            "date": "2024-08-16",
            "team1": "Team A",
            "team2": "Team B",
            "score": {"ft": [1, 0]},
        }
        rec = convert_match(match, "2024-25", "2024-25.json", 0)
        assert rec is not None
        assert rec.home_goals == 1

    def test_missing_team1(self):
        match = {
            "round": "Matchday 1",
            "date": "2024-08-16",
            "team2": "Team B",
            "score": {"ft": [1, 0]},
        }
        rec = convert_match(match, "2024-25", "2024-25.json", 0)
        assert rec is None

    def test_missing_date(self):
        match = {
            "team1": "Team A",
            "team2": "Team B",
            "score": {"ft": [1, 0]},
        }
        rec = convert_match(match, "2024-25", "2024-25.json", 0)
        assert rec is None

    def test_malformed_date(self):
        match = {
            "date": "08/16/2024",
            "team1": "Team A",
            "team2": "Team B",
            "score": {"ft": [1, 0]},
        }
        rec = convert_match(match, "2024-25", "2024-25.json", 0)
        assert rec is None

    def test_negative_goals_rejected(self):
        match = {
            "date": "2024-08-16",
            "team1": "Team A",
            "team2": "Team B",
            "score": {"ft": [-1, 0]},
        }
        rec = convert_match(match, "2024-25", "2024-25.json", 0)
        assert rec is None

    def test_empty_team_name_rejected(self):
        match = {
            "date": "2024-08-16",
            "team1": "",
            "team2": "Team B",
            "score": {"ft": [1, 0]},
        }
        rec = convert_match(match, "2024-25", "2024-25.json", 0)
        assert rec is None

    def test_ft_not_list_rejected(self):
        match = {
            "date": "2024-08-16",
            "team1": "Team A",
            "team2": "Team B",
            "score": {"ft": "2-1"},
        }
        rec = convert_match(match, "2024-25", "2024-25.json", 0)
        assert rec is None

    def test_ft_wrong_length_rejected(self):
        match = {
            "date": "2024-08-16",
            "team1": "Team A",
            "team2": "Team B",
            "score": {"ft": [1]},
        }
        rec = convert_match(match, "2024-25", "2024-25.json", 0)
        assert rec is None


# --------------------------------------------------------------------------- #
# ingest_file tests
# --------------------------------------------------------------------------- #


class TestIngestFile:
    """Tests for ingest_file."""

    def test_valid_file(self, sample_season_file: Path):
        records, skips, rejections = ingest_file(sample_season_file)
        # All 3 matches have valid ft scores; missing ht is acceptable
        assert len(records) == 3
        assert len(skips) == 0
        assert len(rejections) == 0

    def test_missing_ft_is_skipped(self, tmp_path: Path):
        """A match missing ft score should be skipped, not rejected."""
        data = {
            "name": "Test",
            "matches": [
                {
                    "date": "2024-08-16",
                    "team1": "A",
                    "team2": "B",
                    "score": {"ht": [0, 0]},  # missing ft
                },
            ],
        }
        path = tmp_path / "2024-25.json"
        with open(path, "w") as f:
            json.dump(data, f)
        records, skips, rejections = ingest_file(path)
        assert len(records) == 0
        assert len(skips) == 1
        assert len(rejections) == 0
        assert "ft" in skips[0]["reason"].lower() or "score" in skips[0]["reason"].lower()

    def test_nonexistent_file(self, tmp_path: Path):
        path = tmp_path / "does_not_exist.json"
        records, skips, rejections = ingest_file(path)
        assert len(records) == 0
        assert len(rejections) == 1
        assert "read error" in rejections[0]["reason"].lower() or "parse" in rejections[0]["reason"].lower()

    def test_invalid_json(self, tmp_path: Path):
        path = tmp_path / "2024-25.json"
        path.write_text("{ this is not valid json }")
        records, skips, rejections = ingest_file(path)
        assert len(records) == 0
        assert len(rejections) == 1
        assert "parse" in rejections[0]["reason"].lower()


# --------------------------------------------------------------------------- #
# deduplication tests
# --------------------------------------------------------------------------- #


class TestDeduplicate:
    """Tests for deduplicate_records."""

    def test_no_duplicates(self):
        recs = [
            MatchRecord(
                match_id="a",
                competition="PL",
                season="2024-25",
                match_date="2024-08-16",
                home_team="A",
                away_team="B",
                home_goals=1,
                away_goals=0,
                outcome=Outcome.HOME_WIN,
                source="f.json",
            ),
            MatchRecord(
                match_id="b",
                competition="PL",
                season="2024-25",
                match_date="2024-08-17",
                home_team="C",
                away_team="D",
                home_goals=2,
                away_goals=1,
                outcome=Outcome.HOME_WIN,
                source="f.json",
            ),
        ]
        unique, dups = deduplicate_records(recs)
        assert len(unique) == 2
        assert len(dups) == 0

    def test_removes_duplicate(self):
        recs = [
            MatchRecord(
                match_id="dup",
                competition="PL",
                season="2024-25",
                match_date="2024-08-16",
                home_team="A",
                away_team="B",
                home_goals=1,
                away_goals=0,
                outcome=Outcome.HOME_WIN,
                source="f1.json",
            ),
            MatchRecord(
                match_id="dup",
                competition="PL",
                season="2024-25",
                match_date="2024-08-16",
                home_team="A",
                away_team="B",
                home_goals=1,
                away_goals=0,
                outcome=Outcome.HOME_WIN,
                source="f2.json",
            ),
        ]
        unique, dups = deduplicate_records(recs)
        assert len(unique) == 1
        assert len(dups) == 1


# --------------------------------------------------------------------------- #
# CSV output tests
# --------------------------------------------------------------------------- #


class TestWriteCSV:
    """Tests for write_csv."""

    def test_write_csv(self, tmp_path: Path):
        recs = [
            MatchRecord(
                match_id="EPL_2024-25_2024-08-16_a_vs_b",
                competition="Premier League",
                season="2024-25",
                match_date="2024-08-16",
                home_team="Team A",
                away_team="Team B",
                home_goals=1,
                away_goals=0,
                outcome=Outcome.HOME_WIN,
                source="2024-25.json",
            ),
        ]
        path = tmp_path / "output.csv"
        write_csv(recs, path)
        assert path.exists()

        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 1
        assert rows[0]["match_id"] == "EPL_2024-25_2024-08-16_a_vs_b"
        assert rows[0]["outcome"] == "H"
        assert rows[0]["home_goals"] == "1"
        assert rows[0]["source"] == "2024-25.json"

    def test_csv_columns(self):
        """CSV_COLUMNS should contain all required canonical fields."""
        required = {
            "match_id", "competition", "season", "match_date",
            "home_team", "away_team", "home_goals", "away_goals",
            "outcome", "source",
        }
        assert set(CSV_COLUMNS) == required


# --------------------------------------------------------------------------- #
# Report tests
# --------------------------------------------------------------------------- #


class TestWriteReport:
    """Tests for write_report."""

    def test_write_report(self, tmp_path: Path):
        report = IngestionReport(
            total_files=1,
            total_raw_matches=3,
            imported=2,
            skipped=1,
            rejected=0,
        )
        path = tmp_path / "report.json"
        write_report(report, path)
        assert path.exists()

        with open(path) as f:
            data = json.load(f)
        assert data["total_files"] == 1
        assert data["imported"] == 2
        assert data["skipped"] == 1


# --------------------------------------------------------------------------- #
# Full integration tests
# --------------------------------------------------------------------------- #


class TestRunIngestion:
    """Integration tests for the full ingestion pipeline."""

    def test_full_pipeline(self, sample_raw_dir: Path, processed_data_dir: Path):
        report = run_ingestion(
            raw_dir=sample_raw_dir,
            processed_dir=processed_data_dir,
        )
        assert report.total_files == 1
        assert report.imported == 2
        assert report.skipped == 0  # no missing ft in this fixture

        csv_path = processed_data_dir / "premier_league_matches.csv"
        assert csv_path.exists()

        report_path = processed_data_dir / "ingestion_report.json"
        assert report_path.exists()

        with open(report_path) as f:
            report_data = json.load(f)
        assert report_data["imported"] == 2
        assert report_data["total_files"] == 1

    def test_pipeline_with_bad_file(self, tmp_path: Path, processed_data_dir: Path):
        raw_dir = tmp_path / "raw" / "openfootball"
        raw_dir.mkdir(parents=True)

        # Valid file
        valid = {
            "name": "Test",
            "matches": [
                {
                    "date": "2024-08-16",
                    "team1": "A",
                    "team2": "B",
                    "score": {"ft": [1, 0]},
                },
            ],
        }
        with open(raw_dir / "2024-25.json", "w") as f:
            json.dump(valid, f)

        # Invalid JSON file with bad name (should be skipped)
        with open(raw_dir / "bad-file.json", "w") as f:
            f.write("not json")

        report = run_ingestion(
            raw_dir=raw_dir,
            processed_dir=processed_data_dir,
        )
        assert report.imported == 1
        assert report.total_files == 1  # bad-file.json is not a season file
