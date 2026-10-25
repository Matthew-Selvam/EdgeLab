"""Validation utilities for the EdgeLab football ingestion pipeline.

Provides functions to validate individual match records from OpenFootball JSON
files against the canonical schema requirements.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

from .models import Outcome

logger = logging.getLogger(__name__)


def validate_date(date_str: str) -> tuple[bool, str]:
    """Validate that *date_str* is a well-formed ISO 8601 date (YYYY-MM-DD).

    Returns a tuple of (is_valid, reason).  When *is_valid* is True, reason
    is an empty string.
    """
    if not date_str or not isinstance(date_str, str):
        return False, "missing or non-string date"
    try:
        datetime.strptime(date_str, "%Y-%m-%d")
        return True, ""
    except ValueError:
        return False, f"malformed date: {date_str}"


def validate_score(goals: Optional[int], label: str = "score") -> tuple[bool, str]:
    """Validate a goal count.

    Accepts non-negative integers.  ``None`` is treated as missing and
    therefore invalid.  Returns ``(is_valid, reason)``.
    """
    if goals is None:
        return False, f"missing {label}"
    if not isinstance(goals, int):
        return False, f"{label} is not an integer: {goals!r}"
    if goals < 0:
        return False, f"{label} is negative: {goals}"
    return True, ""


def validate_team(team: Optional[str]) -> tuple[bool, str]:
    """Validate that a team name is present and non-empty."""
    if not team or not isinstance(team, str) or not team.strip():
        return False, "missing or empty team name"
    return True, ""


def derive_outcome(home_goals: int, away_goals: int) -> Outcome:
    """Derive the match outcome ('H', 'D', or 'A') from full-time goals.

    Parameters
    ----------
    home_goals:
        Goals scored by the home team.
    away_goals:
        Goals scored by the away team.

    Returns
    -------
    Outcome
        ``Outcome.HOME_WIN`` if home > away, ``Outcome.DRAW`` if equal,
        ``Outcome.AWAY_WIN`` if away > home.
    """
    if home_goals > away_goals:
        return Outcome.HOME_WIN
    if home_goals < away_goals:
        return Outcome.AWAY_WIN
    return Outcome.DRAW
