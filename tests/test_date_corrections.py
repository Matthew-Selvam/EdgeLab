"""Regression tests for the confirmed OpenFootball 2019–20 year error."""

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest

from edgelab.data.ingest import (
    DATE_CORRECTION_REFERENCE,
    EXPECTED_2019_20_AFFECTED_SHA256,
    RAW_DATA_DIR,
    _build_match_id,
    convert_match,
    run_ingestion,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _raw_matches():
    return json.loads((RAW_DATA_DIR / "2019-20.json").read_text())["matches"]


def test_confirmed_source_manifest_is_exact():
    assert _sha(RAW_DATA_DIR / "2019-20.json") == (
        "9e1695f85ce5553f762e809dea357ec0a95838939be89569060a83c4e98a302d"
    )
    affected = [
        [index, match["round"], match["date"], match["team1"], match["team2"],
         match["score"]["ft"]]
        for index, match in enumerate(_raw_matches())
        if match["date"].startswith("2019-07-")
    ]
    digest = hashlib.sha256(json.dumps(
        affected, separators=(",", ":"), ensure_ascii=False
    ).encode()).hexdigest()
    assert len(affected) == 66
    assert digest == EXPECTED_2019_20_AFFECTED_SHA256
    assert min(row[2] for row in affected) == "2019-07-01"
    assert max(row[2] for row in affected) == "2019-07-26"
    assert Counter(row[1] for row in affected) == {
        "Matchday 32": 6, **{f"Matchday {n}": 10 for n in range(33, 39)}
    }


def test_conversion_changes_only_confirmed_year_and_keeps_raw_dict():
    raw = next(match for match in _raw_matches() if match["date"].startswith("2019-07-"))
    original = json.loads(json.dumps(raw))
    changes = []
    record = convert_match(raw, "2019-20", "2019-20.json", 314, changes)
    assert raw == original
    assert record.match_date == "2020" + raw["date"][4:]
    assert (record.home_team, record.away_team) == (raw["team1"], raw["team2"])
    assert [record.home_goals, record.away_goals] == raw["score"]["ft"]
    assert len(changes) == 1
    assert changes[0]["reference"] == DATE_CORRECTION_REFERENCE

    june = next(match for match in _raw_matches() if match["date"].startswith("2020-06-"))
    assert convert_match(june, "2019-20", "2019-20.json", 0).match_date == june["date"]
    assert convert_match(raw, "2018-19", "2018-19.json", 0).match_date == raw["date"]
    altered_round = {**raw, "round": "Matchday 31"}
    assert convert_match(altered_round, "2019-20", "2019-20.json", 0).match_date == raw["date"]


def test_ingestion_is_reproducible_and_changes_only_intended_dates(tmp_path):
    source_hashes = {path.name: _sha(path) for path in RAW_DATA_DIR.glob("*.json")}
    first_dir, second_dir = tmp_path / "first", tmp_path / "second"
    first = run_ingestion(RAW_DATA_DIR, first_dir)
    second = run_ingestion(RAW_DATA_DIR, second_dir)
    assert first.imported == second.imported == 3800
    assert first.correction_count == second.correction_count == 66
    for filename in ("premier_league_matches.csv", "date_corrections.json"):
        assert (first_dir / filename).read_bytes() == (second_dir / filename).read_bytes()
    assert source_hashes == {path.name: _sha(path) for path in RAW_DATA_DIR.glob("*.json")}

    with (first_dir / "premier_league_matches.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    canonical = {(row["source"], row["home_team"], row["away_team"]): row for row in rows}
    assert len(canonical) == 3800
    for source_path in RAW_DATA_DIR.glob("*.json"):
        season = source_path.stem
        for raw in json.loads(source_path.read_text())["matches"]:
            row = canonical[(source_path.name, raw["team1"], raw["team2"])]
            corrected = (season == "2019-20" and raw["date"].startswith("2019-07-")
                         and raw["round"] in {f"Matchday {n}" for n in range(32, 39)})
            expected_date = "2020" + raw["date"][4:] if corrected else raw["date"]
            assert row["match_date"] == expected_date
            assert row["match_id"] == _build_match_id(
                season, expected_date, raw["team1"], raw["team2"])
            assert row["season"] == season
            assert row["competition"] == "Premier League"
            assert [int(row["home_goals"]), int(row["away_goals"])] == raw["score"]["ft"]
            expected_outcome = ("H" if raw["score"]["ft"][0] > raw["score"]["ft"][1]
                                else "A" if raw["score"]["ft"][0] < raw["score"]["ft"][1]
                                else "D")
            assert row["outcome"] == expected_outcome

    audit = json.loads((first_dir / "date_corrections.json").read_text())
    assert audit["count"] == len(audit["corrections"]) == 66
    assert audit["source_sha256"] == {"2019-20.json": source_hashes["2019-20.json"]}
    for entry in audit["corrections"]:
        assert entry["source"] == "2019-20.json"
        assert entry["corrected_date"] == "2020" + entry["original_date"][4:]
        assert entry["reference"] == DATE_CORRECTION_REFERENCE
    corrected_season = [row for row in rows if row["season"] == "2019-20"]
    assert len(corrected_season) == 380
    assert min(row["match_date"] for row in corrected_season) == "2019-08-09"
    assert max(row["match_date"] for row in corrected_season) == "2020-07-26"
    assert sum(row["match_date"].startswith("2020-07-") for row in corrected_season) == 66
    assert sum(row["match_date"].startswith("2020-06-") for row in corrected_season) == 26


def test_changed_affected_fixture_fails_manifest_guard(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    data = json.loads((RAW_DATA_DIR / "2019-20.json").read_text())
    index = next(i for i, match in enumerate(data["matches"])
                 if match["date"].startswith("2019-07-"))
    data["matches"][index]["score"]["ft"][0] += 1
    (raw_dir / "2019-20.json").write_text(json.dumps(data))
    with pytest.raises(ValueError, match="66 audited corrections"):
        run_ingestion(raw_dir, tmp_path / "processed")
    assert not (tmp_path / "processed" / "premier_league_matches.csv").exists()
