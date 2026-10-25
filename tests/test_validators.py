"""Tests for the validators module."""

from __future__ import annotations

from edgelab.data.models import Outcome
from edgelab.data.validators import (
    derive_outcome,
    validate_date,
    validate_score,
    validate_team,
)


class TestValidateDate:
    """Tests for validate_date."""

    def test_valid_date(self):
        ok, reason = validate_date("2024-08-16")
        assert ok is True
        assert reason == ""

    def test_invalid_format(self):
        ok, reason = validate_date("16/08/2024")
        assert ok is False
        assert "malformed" in reason

    def test_empty_string(self):
        ok, reason = validate_date("")
        assert ok is False
        assert "missing" in reason

    def test_none_input(self):
        ok, reason = validate_date(None)  # type: ignore[arg-type]
        assert ok is False
        assert "missing" in reason

    def test_invalid_month(self):
        ok, reason = validate_date("2024-13-01")
        assert ok is False
        assert "malformed" in reason

    def test_invalid_day(self):
        ok, reason = validate_date("2024-02-30")
        assert ok is False
        assert "malformed" in reason


class TestValidateScore:
    """Tests for validate_score."""

    def test_valid_score(self):
        ok, reason = validate_score(3, "home_goals")
        assert ok is True
        assert reason == ""

    def test_zero_is_valid(self):
        ok, reason = validate_score(0, "goals")
        assert ok is True

    def test_none_score(self):
        ok, reason = validate_score(None, "ft_goals")
        assert ok is False
        assert "missing" in reason

    def test_negative_score(self):
        ok, reason = validate_score(-1, "goals")
        assert ok is False
        assert "negative" in reason

    def test_non_integer_score(self):
        ok, reason = validate_score("two", "goals")
        assert ok is False
        assert "not an integer" in reason


class TestValidateTeam:
    """Tests for validate_team."""

    def test_valid_team(self):
        ok, reason = validate_team("Manchester United")
        assert ok is True
        assert reason == ""

    def test_empty_string(self):
        ok, reason = validate_team("")
        assert ok is False
        assert "missing or empty" in reason

    def test_whitespace_only(self):
        ok, reason = validate_team("   ")
        assert ok is False
        assert "missing or empty" in reason

    def test_none_team(self):
        ok, reason = validate_team(None)  # type: ignore[arg-type]
        assert ok is False
        assert "missing or empty" in reason


class TestDeriveOutcome:
    """Tests for derive_outcome."""

    def test_home_win(self):
        assert derive_outcome(3, 1) == Outcome.HOME_WIN

    def test_away_win(self):
        assert derive_outcome(0, 2) == Outcome.AWAY_WIN

    def test_draw(self):
        assert derive_outcome(1, 1) == Outcome.DRAW

    def test_zero_zero_draw(self):
        assert derive_outcome(0, 0) == Outcome.DRAW

    def test_large_home_win(self):
        assert derive_outcome(10, 0) == Outcome.HOME_WIN
