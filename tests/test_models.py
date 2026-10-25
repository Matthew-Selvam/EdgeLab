"""Tests for the canonical data models."""

from __future__ import annotations

import pytest

from edgelab.data.models import IngestionReport, MatchRecord, Outcome


class TestOutcome:
    """Tests for the Outcome enumeration."""

    def test_outcome_values(self):
        """Outcome should have exactly H, D, A."""
        assert Outcome.HOME_WIN.value == "H"
        assert Outcome.DRAW.value == "D"
        assert Outcome.AWAY_WIN.value == "A"

    def test_outcome_is_str_enum(self):
        """Outcome should be a string-based enum."""
        assert isinstance(Outcome.HOME_WIN, Outcome)
        assert isinstance(Outcome.HOME_WIN.value, str)


class TestMatchRecord:
    """Tests for the MatchRecord dataclass."""

    def test_create_valid_record(self):
        """A valid MatchRecord should be created with all fields."""
        rec = MatchRecord(
            match_id="EPL_2024-25_2024-08-16_manunitedfc_vs_fulhamfc",
            competition="Premier League",
            season="2024-25",
            match_date="2024-08-16",
            home_team="Manchester United FC",
            away_team="Fulham FC",
            home_goals=1,
            away_goals=0,
            outcome=Outcome.HOME_WIN,
            source="2024-25.json",
        )
        assert rec.match_id == "EPL_2024-25_2024-08-16_manunitedfc_vs_fulhamfc"
        assert rec.competition == "Premier League"
        assert rec.season == "2024-25"
        assert rec.match_date == "2024-08-16"
        assert rec.home_goals == 1
        assert rec.away_goals == 0
        assert rec.outcome == Outcome.HOME_WIN
        assert rec.source == "2024-25.json"

    def test_record_is_frozen(self):
        """MatchRecord should be immutable (frozen dataclass)."""
        rec = MatchRecord(
            match_id="test",
            competition="PL",
            season="2024-25",
            match_date="2024-08-16",
            home_team="A",
            away_team="B",
            home_goals=1,
            away_goals=0,
            outcome=Outcome.HOME_WIN,
            source="file.json",
        )
        with pytest.raises(AttributeError):
            rec.home_goals = 5


class TestIngestionReport:
    """Tests for the IngestionReport dataclass."""

    def test_default_values(self):
        """IngestionReport should have sensible defaults."""
        report = IngestionReport()
        assert report.total_files == 0
        assert report.total_raw_matches == 0
        assert report.imported == 0
        assert report.skipped == 0
        assert report.rejected == 0
        assert report.skips == []
        assert report.rejections == []
        assert report.season_breakdown == {}

    def test_populated_report(self):
        """An IngestionReport should store all provided values."""
        report = IngestionReport(
            total_files=2,
            total_raw_matches=760,
            imported=758,
            skipped=1,
            rejected=1,
            skips=[{"file": "2024-25.json", "reason": "missing ft"}],
            rejections=[{"file": "2023-24.json", "reason": "malformed"}],
            season_breakdown={"2024-25": {"raw": 380, "imported": 379, "skipped": 1, "rejected": 0}},
        )
        assert report.total_files == 2
        assert report.imported == 758
        assert len(report.skips) == 1
        assert len(report.rejections) == 1
