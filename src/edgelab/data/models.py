"""Canonical data models for the EdgeLab ingestion pipeline.

This module defines the canonical schema for a normalized football match record
and the enumeration of possible match outcomes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Outcome(str, Enum):
    """Full-time match outcome from the home team's perspective."""

    HOME_WIN = "H"
    DRAW = "D"
    AWAY_WIN = "A"


@dataclass(frozen=True)
class MatchRecord:
    """A single normalized football match record.

    Attributes:
        match_id: Deterministic unique identifier derived from season + date + teams.
        competition: The competition name (e.g. "Premier League").
        season: The season string as it appears in source (e.g. "2024-25").
        match_date: ISO 8601 date string (YYYY-MM-DD).  No time component here
            because sources only provide a date; kickoff times are preserved
            separately only when present.
        home_team: Name of the home team (team1 in the source).
        away_team: Name of the away team (team2 in the source).
        home_goals: Full-time goals scored by the home team.
        away_goals: Full-time goals scored by the away team.
        outcome: 'H', 'D', or 'A' derived from full-time goals.
        source: The filename of the JSON file this record originated from.
    """

    match_id: str
    competition: str
    season: str
    match_date: str
    home_team: str
    away_team: str
    home_goals: int
    away_goals: int
    outcome: Outcome
    source: str


@dataclass
class IngestionReport:
    """Machine-readable report of an ingestion run.

    Attributes:
        total_files: Number of source files processed.
        total_raw_matches: Total matches found across all files.
        imported: Number of matches that passed all validation and were written.
        skipped: Number of matches skipped for non-critical reasons
            (e.g. missing data that cannot be recovered).
        rejected: Number of matches rejected due to fatal validation errors.
        skips: Detailed list of skipped records with reasons.
        rejections: Detailed list of rejected records with reasons.
        season_breakdown: Per-season counts of imported/skipped/rejected.
    """

    total_files: int = 0
    total_raw_matches: int = 0
    imported: int = 0
    skipped: int = 0
    rejected: int = 0
    skips: list[dict] = field(default_factory=list)
    rejections: list[dict] = field(default_factory=list)
    season_breakdown: dict[str, dict] = field(default_factory=dict)
    correction_count: int = 0
