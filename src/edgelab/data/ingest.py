"""Ingestion of OpenFootball JSON match data into canonical EdgeLab records.

The :mod:`ingest` module reads all downloaded Premier League season files from
``data/raw/openfootball/``, normalizes each match into a
:class:`~edgelab.data.models.MatchRecord`, validates data quality, and writes
the results to ``data/processed/premier_league_matches.csv`` together with a
machine-readable JSON report.

Reproduce::

    python -m edgelab.data.ingest
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .models import IngestionReport, MatchRecord, Outcome
from .validators import (
    derive_outcome,
    validate_date,
    validate_score,
    validate_team,
)

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent  # edgelab/
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "openfootball"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
CSV_OUTPUT = PROCESSED_DATA_DIR / "premier_league_matches.csv"
REPORT_OUTPUT = PROCESSED_DATA_DIR / "ingestion_report.json"
CORRECTIONS_OUTPUT = PROCESSED_DATA_DIR / "date_corrections.json"
DATE_CORRECTION_REFERENCE = "https://github.com/openfootball/football.json/issues/54"
EXPECTED_2019_20_CORRECTIONS = 66
# SHA-256 of JSON-serialized [source index, round, original date, home, away,
# full-time score] for the 66 locally audited source rows, in source order.
EXPECTED_2019_20_AFFECTED_SHA256 = (
    "99963e6caeb8c198f7e7d5928d81d8880344773881549cd82a9fd291d8599f8d"
)

CSV_COLUMNS = [
    "match_id",
    "competition",
    "season",
    "match_date",
    "home_team",
    "away_team",
    "home_goals",
    "away_goals",
    "outcome",
    "source",
]

# Canonical competition name for all files in this dataset.
COMPETITION_NAME = "Premier League"

# Season identifier extracted from filenames like ``2024-25.json``.
SEASON_PATTERN = re.compile(r"^(\d{4}-\d{2})\.json$")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _extract_season(filename: str) -> str | None:
    """Extract the season string (e.g. ``2024-25``) from a filename.

    Returns ``None`` if the filename does not match the expected pattern.
    """
    if not filename:
        return None
    match = SEASON_PATTERN.match(filename)
    return match.group(1) if match else None


def _build_match_id(season: str, match_date: str, home_team: str, away_team: str) -> str:
    """Build a deterministic match_id from season, date and team names.

    The ID is designed to be stable across re-ingestion runs so that the same
    match is always assigned the same identifier.
    """
    # Clean team names for use in an identifier: lowercase, strip non-alphanum
    clean_home = re.sub(r"[^a-z0-9]", "", home_team.lower())
    clean_away = re.sub(r"[^a-z0-9]", "", away_team.lower())
    return f"EPL_{season}_{match_date}_{clean_home}_vs_{clean_away}"


def _normalize_date(date_str: str) -> str:
    """Return a normalized ``YYYY-MM-DD`` date string.

    OpenFootball dates are already in ISO format; this function exists to
    centralise format handling and to reject anything unexpected.
    """
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    return dt.strftime("%Y-%m-%d")


def _correct_known_date_error(
    date_str: str, season: str, source_file: str, match_round: str | None
) -> str:
    """Correct only the confirmed July 2019 year error in 2019–20.

    OpenFootball issue #54 identifies the July fixtures in Matchdays 32–38
    as played in July 2020. Month and day are preserved. The raw dictionary
    and source file remain untouched.
    """
    affected_rounds = {f"Matchday {number}" for number in range(32, 39)}
    if (season == "2019-20" and source_file == "2019-20.json"
            and match_round in affected_rounds and date_str.startswith("2019-07-")):
        return "2020" + date_str[4:]
    return date_str


def _extract_season_from_date(date_str: str) -> str:
    """Derive an approximate season label from a match date.

    The Premier League season ``2024-25`` starts in August 2024 and ends in
    May 2025.  We use the August-start convention: if the match month is
    October or later, the season is ``YYYY-(YY+1)``; otherwise it is
    ``(YYYY-1)-YY``.

    This is used only as a cross-check against the filename-derived season;
    the filename season takes priority.
    """
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    year = dt.year
    if dt.month >= 8:
        next_year = year + 1
        return f"{year}-{str(next_year)[-2:]}"
    else:
        prev_year = year - 1
        return f"{prev_year}-{str(year)[-2:]}"


# --------------------------------------------------------------------------- #
# Core conversion
# --------------------------------------------------------------------------- #

def convert_match(
    raw_match: dict[str, Any],
    season: str,
    source_file: str,
    match_index: int,
    corrections: list[dict[str, Any]] | None = None,
) -> MatchRecord | None:
    """Convert a single raw match dict into a :class:`MatchRecord`.

    Returns ``None`` if the match cannot be converted (it is the caller's
    responsibility to record the skip/reject reason).  Validation errors are
    logged via ``logger.warning``.
    """
    # --- required fields -------------------------------------------------- #
    date_raw = raw_match.get("date")
    team1 = raw_match.get("team1")
    team2 = raw_match.get("team2")
    score = raw_match.get("score") or {}

    ft = score.get("ft")
    if ft is None:
        logger.warning(
            "Skipped %s match #%d: missing full-time score", source_file, match_index
        )
        return None

    # ft is a list [home_goals, away_goals]
    if not isinstance(ft, list) or len(ft) != 2:
        logger.warning(
            "Rejected %s match #%d: ft score is not a 2-element list: %r",
            source_file,
            match_index,
            ft,
        )
        return None

    home_goals_raw, away_goals_raw = ft[0], ft[1]

    # --- validate date ---------------------------------------------------- #
    if date_raw is None:
        logger.warning(
            "Skipped %s match #%d: missing date", source_file, match_index
        )
        return None
    date_ok, date_reason = validate_date(date_raw)
    if not date_ok:
        logger.warning("Rejected %s match #%d: %s", source_file, match_index, date_reason)
        return None
    original_date = _normalize_date(date_raw)
    match_date = _correct_known_date_error(
        original_date, season, source_file, raw_match.get("round")
    )

    # --- validate teams --------------------------------------------------- #
    team1_ok, team1_reason = validate_team(team1)
    if not team1_ok:
        logger.warning("Rejected %s match #%d: %s", source_file, match_index, team1_reason)
        return None
    team2_ok, team2_reason = validate_team(team2)
    if not team2_ok:
        logger.warning("Rejected %s match #%d: %s", source_file, match_index, team2_reason)
        return None

    # --- validate scores -------------------------------------------------- #
    hg_ok, hg_reason = validate_score(home_goals_raw, "home_goals")
    if not hg_ok:
        logger.warning("Rejected %s match #%d: %s", source_file, match_index, hg_reason)
        return None

    ag_ok, ag_reason = validate_score(away_goals_raw, "away_goals")
    if not ag_ok:
        logger.warning("Rejected %s match #%d: %s", source_file, match_index, ag_reason)
        return None

    home_goals = int(home_goals_raw)
    away_goals = int(away_goals_raw)
    outcome = derive_outcome(home_goals, away_goals)

    match_id = _build_match_id(season, match_date, str(team1), str(team2))

    record = MatchRecord(
        match_id=match_id,
        competition=COMPETITION_NAME,
        season=season,
        match_date=match_date,
        home_team=str(team1),
        away_team=str(team2),
        home_goals=home_goals,
        away_goals=away_goals,
        outcome=outcome,
        source=source_file,
    )
    if corrections is not None and match_date != original_date:
        corrections.append({
            "source": source_file,
            "source_index": match_index,
            "round": raw_match.get("round"),
            "home_team": str(team1),
            "away_team": str(team2),
            "home_goals": home_goals,
            "away_goals": away_goals,
            "original_date": original_date,
            "corrected_date": match_date,
            "original_match_id": _build_match_id(season, original_date, str(team1), str(team2)),
            "corrected_match_id": match_id,
            "reference": DATE_CORRECTION_REFERENCE,
        })
    return record


def ingest_file(
    path: Path, corrections: list[dict[str, Any]] | None = None
) -> tuple[list[MatchRecord], list[dict], list[dict]]:
    """Ingest a single OpenFootball JSON file.

    Parameters
    ----------
    path:
        Filesystem path to a season JSON file.

    Returns
    -------
    tuple
        ``(records, skips, rejections)`` where *records* is a list of
        valid :class:`MatchRecord` objects, *skips* is a list of dicts
        describing skipped matches, and *rejections* is a list of dicts
        describing rejected matches.
    """
    source_file = path.name
    season = _extract_season(source_file)

    if season is None:
        logger.error("Could not parse season from filename: %s", source_file)
        return [], [], [{"file": source_file, "reason": "unparseable season from filename"}]

    logger.info("Ingesting %s (season %s)", source_file, season)

    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except json.JSONDecodeError as exc:
        logger.error("Failed to parse JSON in %s: %s", source_file, exc)
        return [], [], [{"file": source_file, "reason": f"JSON parse error: {exc}"}]
    except OSError as exc:
        logger.error("Failed to read file %s: %s", source_file, exc)
        return [], [], [{"file": source_file, "reason": f"file read error: {exc}"}]

    matches = data.get("matches", [])
    if not isinstance(matches, list):
        logger.error("Expected 'matches' to be a list in %s", source_file)
        return [], [], [{"file": source_file, "reason": "'matches' is not a list"}]

    records: list[MatchRecord] = []
    skips: list[dict] = []
    rejections: list[dict] = []

    for idx, raw_match in enumerate(matches):
        if not isinstance(raw_match, dict):
            reason = f"unexpected non-dict match entry at index {idx}"
            logger.warning("%s: %s", source_file, reason)
            rejections.append({"file": source_file, "index": idx, "reason": reason})
            continue

        record = convert_match(raw_match, season, source_file, idx, corrections)
        if record is None:
            # Determine whether this was a skip (missing data) or a reject
            # (malformed data).  We classify based on what's missing.
            ft = (raw_match.get("score") or {}).get("ft")
            if ft is None:
                skips.append({
                    "file": source_file,
                    "index": idx,
                    "season": season,
                    "date": raw_match.get("date"),
                    "teams": [raw_match.get("team1"), raw_match.get("team2")],
                    "reason": "missing full-time score or halftime data incomplete",
                })
            else:
                rejections.append({
                    "file": source_file,
                    "index": idx,
                    "season": season,
                    "date": raw_match.get("date"),
                    "teams": [raw_match.get("team1"), raw_match.get("team2")],
                    "reason": "malformed score or team fields",
                })
            continue

        records.append(record)

    return records, skips, rejections


def deduplicate_records(
    records: list[MatchRecord],
) -> tuple[list[MatchRecord], list[dict]]:
    """Remove duplicate records that share the same ``match_id``.

    Returns ``(unique_records, duplicate_entries)``.
    """
    seen: dict[str, MatchRecord] = {}
    duplicates: list[dict] = []

    for rec in records:
        if rec.match_id in seen:
            duplicates.append({
                "match_id": rec.match_id,
                "season": rec.season,
                "date": rec.match_date,
                "teams": [rec.home_team, rec.away_team],
                "source": rec.source,
                "reason": "duplicate match_id (same season + date + teams)",
            })
        else:
            seen[rec.match_id] = rec

    return list(seen.values()), duplicates


# --------------------------------------------------------------------------- #
# Output writers
# --------------------------------------------------------------------------- #

def write_csv(records: list[MatchRecord], path: Path) -> None:
    """Write normalized match records to a CSV file at *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for rec in records:
            row = {
                "match_id": rec.match_id,
                "competition": rec.competition,
                "season": rec.season,
                "match_date": rec.match_date,
                "home_team": rec.home_team,
                "away_team": rec.away_team,
                "home_goals": rec.home_goals,
                "away_goals": rec.away_goals,
                "outcome": rec.outcome.value,
                "source": rec.source,
            }
            writer.writerow(row)
    logger.info("Wrote %d records to %s", len(records), path)


def write_report(report: IngestionReport, path: Path) -> None:
    """Write the ingestion report to a JSON file at *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)

    report_dict = {
        "total_files": report.total_files,
        "total_raw_matches": report.total_raw_matches,
        "imported": report.imported,
        "skipped": report.skipped,
        "rejected": report.rejected,
        "correction_count": report.correction_count,
        "skips": report.skips,
        "rejections": report.rejections,
        "season_breakdown": report.season_breakdown,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report_dict, fh, indent=2, ensure_ascii=False)
    logger.info("Wrote ingestion report to %s", path)


def write_corrections(
    corrections: list[dict[str, Any]], source_hashes: dict[str, str], path: Path
) -> None:
    """Write a deterministic, fixture-level audit of source date corrections."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "reason": "OpenFootball 2019-20 Matchdays 32-38 July dates have the wrong year",
        "reference": DATE_CORRECTION_REFERENCE,
        "count": len(corrections),
        "source_sha256": source_hashes,
        "corrections": sorted(corrections, key=lambda row: (row["source"], row["source_index"])),
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- #
# Main pipeline
# --------------------------------------------------------------------------- #

def run_ingestion(
    raw_dir: Path = RAW_DATA_DIR,
    processed_dir: Path = PROCESSED_DATA_DIR,
) -> IngestionReport:
    """Run the full ingestion pipeline.

    Reads every ``YYYY-MM.json`` file in *raw_dir*, normalizes matches, validates
    them, deduplicates, writes the canonical CSV and a JSON report.

    Parameters
    ----------
    raw_dir:
        Directory containing OpenFootball JSON files.
    processed_dir:
        Directory where output CSV and report will be written.

    Returns
    -------
    IngestionReport
        Summary of the ingestion run.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    processed_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(raw_dir.glob("*.json"))
    # Filter to only valid season filenames
    season_files = [f for f in files if _extract_season(f.name)]

    logger.info("Found %d season files in %s", len(season_files), raw_dir)

    all_records: list[MatchRecord] = []
    all_skips: list[dict] = []
    all_rejections: list[dict] = []
    all_corrections: list[dict[str, Any]] = []
    corrected_source_hashes: dict[str, str] = {}
    total_raw = 0
    season_breakdown: dict[str, dict] = defaultdict(
        lambda: {"raw": 0, "imported": 0, "skipped": 0, "rejected": 0}
    )

    for path in season_files:
        file_corrections: list[dict[str, Any]] = []
        records, skips, rejections = ingest_file(path, file_corrections)
        season = _extract_season(path.name) or "unknown"

        file_raw = len(records) + len(skips) + len(rejections)
        if path.name == "2019-20.json":
            signature_rows = [
                [entry["source_index"], entry["round"], entry["original_date"],
                 entry["home_team"], entry["away_team"],
                 [entry["home_goals"], entry["away_goals"]]]
                for entry in file_corrections
            ]
            signature = hashlib.sha256(json.dumps(
                signature_rows, separators=(",", ":"), ensure_ascii=False
            ).encode("utf-8")).hexdigest()
            if (file_raw != 380 or len(file_corrections) != EXPECTED_2019_20_CORRECTIONS
                    or signature != EXPECTED_2019_20_AFFECTED_SHA256):
                raise ValueError(
                    "2019-20 source differs from the 66 audited corrections; "
                    "review the provider data before regenerating canonical matches"
                )
            corrected_source_hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        all_corrections.extend(file_corrections)
        total_raw += file_raw
        all_records.extend(records)
        all_skips.extend(skips)
        all_rejections.extend(rejections)

        season_breakdown[season]["raw"] += file_raw
        season_breakdown[season]["imported"] += len(records)
        season_breakdown[season]["skipped"] += len(skips)
        season_breakdown[season]["rejected"] += len(rejections)

    # Deduplicate
    unique_records, duplicates = deduplicate_records(all_records)
    if duplicates:
        logger.warning("Found %d duplicate matches; keeping first occurrence", len(duplicates))
        all_rejections.extend(duplicates)

    # Sort by season then date for deterministic output
    unique_records.sort(key=lambda r: (r.season, r.match_date, r.home_team, r.away_team))

    write_csv(unique_records, processed_dir / "premier_league_matches.csv")
    write_corrections(all_corrections, corrected_source_hashes,
                      processed_dir / "date_corrections.json")

    report = IngestionReport(
        total_files=len(season_files),
        total_raw_matches=total_raw,
        imported=len(unique_records),
        skipped=len(all_skips),
        rejected=len(all_rejections),
        skips=all_skips,
        rejections=all_rejections,
        season_breakdown=dict(season_breakdown),
        correction_count=len(all_corrections),
    )
    write_report(report, processed_dir / "ingestion_report.json")

    logger.info(
        "Ingestion complete: %d imported, %d skipped, %d rejected from %d files",
        report.imported,
        report.skipped,
        report.rejected,
        report.total_files,
    )
    return report


def main() -> int:
    """Entry point for ``python -m edgelab.data.ingest``."""
    report = run_ingestion()
    return 0


if __name__ == "__main__":
    sys.exit(main())
