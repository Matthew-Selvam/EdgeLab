"""Pytest configuration and shared fixtures for EdgeLab tests."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

# Ensure the src directory is on the path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


@pytest.fixture
def raw_data_dir() -> Path:
    """Return the path to the raw OpenFootball data directory."""
    return PROJECT_ROOT / "data" / "raw" / "openfootball"


@pytest.fixture
def processed_data_dir(tmp_path: Path) -> Path:
    """Return a temporary directory for processed output."""
    return tmp_path / "processed"


@pytest.fixture
def sample_match_data() -> dict:
    """Return a single valid match dict for testing."""
    return {
        "round": "Matchday 1",
        "date": "2024-08-16",
        "time": "20:00",
        "team1": "Manchester United FC",
        "team2": "Fulham FC",
        "score": {
            "ht": [0, 0],
            "ft": [1, 0],
        },
    }


@pytest.fixture
def sample_season_file(tmp_path: Path, sample_match_data: dict) -> Path:
    """Create a small sample season JSON file in a tmp directory."""
    data = {
        "name": "English Premier League 2024/25",
        "matches": [
            sample_match_data,
            {
                "round": "Matchday 1",
                "date": "2024-08-17",
                "time": "12:30",
                "team1": "Arsenal FC",
                "team2": "Ipswich Town FC",
                "score": {"ht": [0, 0], "ft": [2, 0]},
            },
            {
                "round": "Matchday 1",
                "date": "2024-08-17",
                "time": "15:00",
                "team1": "Everton FC",
                "team2": "Brighton & Hove Albion FC",
                "score": {"ft": [0, 3]},
            },
        ],
    }
    path = tmp_path / "2024-25.json"
    with open(path, "w") as f:
        json.dump(data, f)
    return path


@pytest.fixture
def sample_raw_dir(tmp_path: Path, sample_match_data: dict) -> Path:
    """Create a temporary raw data directory with a sample season file."""
    raw_dir = tmp_path / "raw" / "openfootball"
    raw_dir.mkdir(parents=True)
    data = {
        "name": "English Premier League 2024/25",
        "matches": [
            sample_match_data,
            {
                "round": "Matchday 1",
                "date": "2024-08-17",
                "time": "12:30",
                "team1": "Arsenal FC",
                "team2": "Ipswich Town FC",
                "score": {"ht": [0, 0], "ft": [2, 0]},
            },
        ],
    }
    with open(raw_dir / "2024-25.json", "w") as f:
        json.dump(data, f)
    return raw_dir
