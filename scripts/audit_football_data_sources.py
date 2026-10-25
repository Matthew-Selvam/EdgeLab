"""Read-only audit of local Football-Data EPL CSVs against canonical results.

Writes aggregate checks and fixture-match exceptions only. It never exports
prices, computes market probabilities, or changes source/evaluation files.
Run from the repository root: python scripts/audit_football_data_sources.py
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from edgelab.models.teams import TeamAliasMapper

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/football-data"
CANONICAL = ROOT / "data/processed/premier_league_matches.csv"
OUTPUT = ROOT / "data/processed/odds_audit/source_audit.json"
SEASONS = {
    "2015-16": "1516", "2016-17": "1617", "2017-18": "1718",
    "2018-19": "1819", "2019-20": "1920", "2020-21": "2021",
    "2021-22": "2122", "2022-23": "2223", "2023-24": "2324",
    "2024-25": "2425",
}
ODDS_FIELDS = ("PSH", "PSD", "PSA", "PSCH", "PSCD", "PSCA")
TRIPLETS = {"pinnacle_preclosing": ODDS_FIELDS[:3],
            "pinnacle_closing": ODDS_FIELDS[3:]}

# Explicit Football-Data abbreviations, inspected against EdgeLab's team set.
# Already-known names are handled by TeamAliasMapper. Unknown names are errors.
FOOTBALL_DATA_ALIASES = {
    "Bournemouth": "AFC Bournemouth", "Cardiff": "Cardiff City",
    "Huddersfield": "Huddersfield Town", "Hull": "Hull City",
    "Ipswich": "Ipswich Town", "Leeds": "Leeds United",
    "Leicester": "Leicester City", "Luton": "Luton Town",
    "Man City": "Manchester City", "Man United": "Manchester United",
    "Newcastle": "Newcastle United", "Norwich": "Norwich City",
    "Nott'm Forest": "Nottingham Forest", "Stoke": "Stoke City",
    "Swansea": "Swansea City", "Tottenham": "Tottenham Hotspur",
    "West Brom": "West Bromwich Albion", "West Ham": "West Ham United",
    "Wolves": "Wolverhampton Wanderers",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_date(value: str) -> str:
    fmt = "%d/%m/%Y" if len(value.rsplit("/", 1)[-1]) == 4 else "%d/%m/%y"
    return datetime.strptime(value, fmt).date().isoformat()


def price_status(value: str) -> str:
    if not value.strip():
        return "missing"
    try:
        number = float(value)
    except ValueError:
        return "malformed"
    if not math.isfinite(number):
        return "nonfinite"
    if number <= 0:
        return "nonpositive"
    if number <= 1:
        return "equal_to_one"
    return "valid"


def load_csv(path: Path) -> tuple[list[str], list[tuple[int, dict[str, str]]], list[dict], str]:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Required nonempty CSV is missing: {path}")
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            content = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError(f"Could not decode {path}")
    reader = csv.reader(content.splitlines())
    header = next(reader, None)
    if not header or len(header) != len(set(header)) or not all(header):
        raise ValueError(f"Missing or duplicate header names: {path}")
    rows: list[tuple[int, dict[str, str]]] = []
    malformed: list[dict] = []
    for line, values in enumerate(reader, 2):
        if len(values) != len(header):
            malformed.append({"line": line, "reason": "column_count",
                              "observed_columns": len(values), "expected_columns": len(header)})
            continue
        rows.append((line, dict(zip(header, values))))
    return header, rows, malformed, encoding


def main() -> None:
    mapper = TeamAliasMapper()
    known_aliases = {name for aliases in mapper._ALIASES.values() for name in aliases}
    canonical_header, canonical_rows, canonical_malformed, _ = load_csv(CANONICAL)
    if canonical_malformed or len(canonical_rows) != 3800:
        raise ValueError("Canonical results differ from the frozen 3,800-row schema")
    canonical = [row for _, row in canonical_rows]
    index_exact: dict[tuple, list[dict]] = defaultdict(list)
    index_normalized: dict[tuple, list[dict]] = defaultdict(list)
    index_pair: dict[tuple, list[dict]] = defaultdict(list)
    for row in canonical:
        raw_key = (row["season"], row["match_date"], row["home_team"], row["away_team"])
        home = mapper.canonicalize(row["home_team"])
        away = mapper.canonicalize(row["away_team"])
        index_exact[raw_key].append(row)
        index_normalized[(row["season"], row["match_date"], home, away)].append(row)
        index_pair[(row["season"], home, away)].append(row)

    files = []
    all_exceptions: list[dict] = []
    alias_counts: Counter[str] = Counter()
    all_matched: Counter[str] = Counter()
    for season, folder in SEASONS.items():
        path = RAW / folder / "E0.csv"
        header, rows, malformed, encoding = load_csv(path)
        required = {"Div", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR", *ODDS_FIELDS}
        if required - set(header):
            raise ValueError(f"Missing audit columns in {path}: {sorted(required - set(header))}")
        field_counts = {field: Counter() for field in ODDS_FIELDS}
        triplet_counts = Counter()
        match_counts = Counter()
        date_values: list[str] = []
        time_counts = Counter()
        division_counts = Counter()
        source_exact_keys: Counter[tuple] = Counter()
        source_normalized_keys: Counter[tuple] = Counter()
        source_pair_keys: Counter[tuple] = Counter()
        matched_ids: set[str] = set()
        exceptions = list(malformed)
        for line, row in rows:
            division_counts[row["Div"]] += 1
            statuses = {field: price_status(row[field]) for field in ODDS_FIELDS}
            for field, status in statuses.items():
                field_counts[field][status] += 1
            for label, fields in TRIPLETS.items():
                if all(statuses[field] == "valid" for field in fields):
                    triplet_counts[label] += 1
            if "Time" in header:
                value = row["Time"].strip()
                if not value:
                    time_counts["missing"] += 1
                else:
                    try:
                        datetime.strptime(value, "%H:%M")
                        time_counts["valid"] += 1
                    except ValueError:
                        time_counts["malformed"] += 1
                        exceptions.append({"line": line, "reason": "malformed_kickoff_time"})
            try:
                date = parse_date(row["Date"])
            except ValueError:
                match_counts["invalid_date"] += 1
                exceptions.append({"line": line, "reason": "invalid_date"})
                continue
            date_values.append(date)
            if not (int(season[:4]) <= int(date[:4]) <= int(season[-2:]) + 2000):
                exceptions.append({"line": line, "reason": "date_outside_season"})
            raw_key = (season, date, row["HomeTeam"], row["AwayTeam"])
            source_exact_keys[raw_key] += 1
            unknown = [name for name in (row["HomeTeam"], row["AwayTeam"])
                       if name not in known_aliases and name not in FOOTBALL_DATA_ALIASES]
            if unknown:
                match_counts["unknown_team_name"] += 1
                exceptions.append({"line": line, "reason": "unknown_team_name", "names": unknown})
                continue
            home = mapper.canonicalize(FOOTBALL_DATA_ALIASES.get(row["HomeTeam"], row["HomeTeam"]))
            away = mapper.canonicalize(FOOTBALL_DATA_ALIASES.get(row["AwayTeam"], row["AwayTeam"]))
            for name in (row["HomeTeam"], row["AwayTeam"]):
                if name in FOOTBALL_DATA_ALIASES:
                    alias_counts[name] += 1
            if home != row["HomeTeam"] or away != row["AwayTeam"]:
                match_counts["rows_requiring_alias"] += 1
            source_normalized_keys[(season, date, home, away)] += 1
            source_pair_keys[(season, home, away)] += 1
            if len(index_exact[raw_key]) == 1:
                match_counts["raw_name_date_matches"] += 1
            candidates = index_normalized[(season, date, home, away)]
            if len(candidates) == 0:
                pair_candidates = index_pair[(season, home, away)]
                if len(pair_candidates) == 1:
                    match_counts["date_discrepancies"] += 1
                    exceptions.append({"line": line, "reason": "date_discrepancy",
                                       "source_date": date,
                                       "canonical_date": pair_candidates[0]["match_date"],
                                       "canonical_match_id": pair_candidates[0]["match_id"]})
                elif len(pair_candidates) > 1:
                    match_counts["ambiguous_matches"] += 1
                    exceptions.append({"line": line, "reason": "ambiguous_team_pair"})
                else:
                    match_counts["unmatched_source"] += 1
                    exceptions.append({"line": line, "reason": "unmatched_source"})
                continue
            if len(candidates) > 1:
                match_counts["ambiguous_matches"] += 1
                exceptions.append({"line": line, "reason": "ambiguous_exact_key"})
                continue
            match = candidates[0]
            if (row["FTHG"], row["FTAG"], row["FTR"]) != (
                    match["home_goals"], match["away_goals"], match["outcome"]):
                match_counts["result_discrepancies"] += 1
                exceptions.append({"line": line, "reason": "result_discrepancy",
                                   "canonical_match_id": match["match_id"]})
                continue
            match_counts["exact_after_alias_and_result"] += 1
            matched_ids.add(match["match_id"])
            all_matched[match["match_id"]] += 1
        for key, count in source_exact_keys.items():
            if count > 1:
                exceptions.append({"reason": "duplicate_raw_fixture", "count": count})
        for key, count in source_normalized_keys.items():
            if count > 1:
                exceptions.append({"reason": "duplicate_normalized_fixture", "count": count})
        for key, count in source_pair_keys.items():
            if count > 1:
                exceptions.append({"reason": "duplicate_season_home_away", "count": count})
        match_counts["duplicate_raw_fixture_keys"] = sum(count - 1 for count in source_exact_keys.values() if count > 1)
        match_counts["duplicate_normalized_fixture_keys"] = sum(count - 1 for count in source_normalized_keys.values() if count > 1)
        match_counts["duplicate_season_home_away_keys"] = sum(count - 1 for count in source_pair_keys.values() if count > 1)
        match_counts["unmatched_canonical"] = sum(
            row["match_id"] not in matched_ids for row in canonical if row["season"] == season)
        files.append({
            "season": season, "relative_path": str(path.relative_to(ROOT)),
            "bytes": path.stat().st_size, "sha256": sha256(path), "encoding": encoding,
            "header": header, "header_column_count": len(header),
            "rows_parsed": len(rows), "malformed_row_count": len(malformed),
            "date_min": min(date_values) if date_values else None,
            "date_max": max(date_values) if date_values else None,
            "unique_match_dates": len(set(date_values)),
            "division_counts": dict(sorted(division_counts.items())),
            "kickoff_time_column_present": "Time" in header,
            "kickoff_time_counts": dict(sorted(time_counts.items())),
            "odds_fields": {field: {status: field_counts[field][status] for status in
                                    ("valid", "missing", "malformed", "nonfinite",
                                     "nonpositive", "equal_to_one")}
                            for field in ODDS_FIELDS},
            "complete_triplets": dict(sorted(triplet_counts.items())),
            "matching": {key: match_counts[key] for key in (
                "raw_name_date_matches", "rows_requiring_alias",
                "exact_after_alias_and_result", "invalid_date", "unknown_team_name",
                "date_discrepancies", "ambiguous_matches", "unmatched_source",
                "result_discrepancies", "duplicate_raw_fixture_keys",
                "duplicate_normalized_fixture_keys", "duplicate_season_home_away_keys",
                "unmatched_canonical")},
            "exceptions": exceptions,
        })
        all_exceptions.extend({"season": season, **item} for item in exceptions)
    result = {
        "audit_version": 1,
        "source_documentation": {
            "field_key": "https://football-data.co.uk/notes.txt",
            "download_page": "https://football-data.co.uk/englandm.php",
            "fair_use": "https://football-data.co.uk/contact.php",
            "field_interpretation": "PSH/PSD/PSA are Pinnacle pre-closing H/D/A; inserted C denotes closing, hence PSCH/PSCD/PSCA",
        },
        "canonical": {"relative_path": str(CANONICAL.relative_to(ROOT)),
                      "sha256": sha256(CANONICAL), "rows": len(canonical),
                      "header": canonical_header},
        "match_strategy": "Season + parsed source date + explicit Football-Data team aliases + existing EdgeLab team aliases + home/away role; require same full-time goals and result. Unique season/home/away fallback diagnoses date discrepancies only.",
        "aliases": {name: {"canonical_team": canonical_name,
                            "team_occurrences": alias_counts[name]}
                    for name, canonical_name in sorted(FOOTBALL_DATA_ALIASES.items())},
        "files": files,
        "total_exact_after_alias_and_result": sum(f["matching"].get("exact_after_alias_and_result", 0) for f in files),
        "duplicate_canonical_match_assignments": sum(count - 1 for count in all_matched.values() if count > 1),
        "exceptions": all_exceptions,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT.relative_to(ROOT)}: {len(files)} files, "
          f"{result['total_exact_after_alias_and_result']} exact fixture matches, "
          f"{len(all_exceptions)} exceptions")


if __name__ == "__main__":
    main()
