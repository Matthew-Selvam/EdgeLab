# Football-Data Premier League odds: source audit

Audited 2026-10-21. This is a **source and fixture audit only**. No market
probabilities, model scores, or benchmark dataset were produced.
This dated audit predates the later retrospective benchmark; current aggregate
results and limitations are in the [research report](EDGELAB_RESEARCH_REPORT.md).

## Provenance and use

The ten Premier League CSVs were manually downloaded from
[Football-Data.co.uk](https://football-data.co.uk/englandm.php) and supplied as
`E0.csv` and `E0 (1).csv` through `E0 (9).csv` in Downloads. They were assigned
to seasons by their parsed date coverage and copied byte-for-byte into
`data/raw/football-data/{1516,1617,1718,1819,1920,2021,2122,2223,2324,2425}/E0.csv`.
The attachment-to-folder order was `E0 (9)` to `1516`, `(8)` to `1617`,
`(7)` to `1718`, `(6)` to `1819`, `(5)` to `1920`, `(4)` to `2021`,
`(3)` to `2122`, `(2)` to `2223`, `(1)` to `2324`, and `E0.csv` to `2425`.
The Downloads originals and local raw copies have identical SHA-256 hashes.
Both remain unchanged; the audit reads them without rewriting them. The local
raw files are excluded by `.gitignore` (the entire `/data/` tree). The complete per-file
hashes, headers and checks are in the local, gitignored
`data/processed/odds_audit/source_audit.json`.

Football-Data's [column key](https://football-data.co.uk/notes.txt) defines
`PSH`, `PSD`, `PSA` as Pinnacle home, draw and away prices. It says a `C`
inserted after the bookmaker abbreviation identifies **closing** odds.
Consequently `PSCH`, `PSCD`, `PSCA` are the documented Pinnacle closing 1X2
fields, and `PSH`, `PSD`, `PSA` are the separate non-closing triplet. The
[historical download page](https://football-data.co.uk/downloadm.php) also
describes Pinnacle closing 1X2 availability from 2012–13. We verified field
presence and values in every local file rather than inferring coverage from
that site-level statement.

The [fair-data-use policy](https://football-data.co.uk/contact.php) describes
free data for private personal betting and/or academic interest and research
through manual or at most light automated download; it excludes commercial
use and heavy repeated automated data-training use. This audit makes no
downloads and does not redistribute raw odds or row-level prices. The
[source page](https://football-data.co.uk/data.php) attributes compiled odds
to Betbrain and Oddsportal. A third-party pipeline's software license is not
a license for underlying odds. Any later use must stay within the source
terms or obtain suitable permission.

## File and date audit

Every file is non-empty, readable, has a unique nonblank header, and parses
to exactly 380 data rows with `Div=E0`. No row has an incorrect CSV column
count, invalid date, or malformed available kickoff time. Minimum and maximum
dates and the number of distinct match dates per season are recorded in the
machine-readable audit. The complete ordered header array for **each** season is recorded in the machine-readable
audit. There are four observed header layouts: 65 columns in 2015–16 through
2017–18, 62 in 2018–19, 106 in 2019–20 through 2023–24, and 120 in
2024–25. All contain `Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR` and all six
Pinnacle columns.

| Season | Local folder | Bytes | Header columns | Rows | First date | Last date | Kickoff `Time` | Complete closing triplets |
|:--|:--|--:|--:|--:|:--|:--|:--:|--:|
| 2015–16 | 1516 | 100,656 | 65 | 380 | 2015-08-08 | 2016-05-17 | absent | 380 |
| 2016–17 | 1617 | 99,990 | 65 | 380 | 2016-08-13 | 2017-05-21 | absent | 380 |
| 2017–18 | 1718 | 100,734 | 65 | 380 | 2017-08-11 | 2018-05-13 | absent | 380 |
| 2018–19 | 1819 | 96,144 | 62 | 380 | 2018-08-10 | 2019-05-12 | absent | 380 |
| 2019–20 | 1920 | 175,475 | 106 | 380 | 2019-08-09 | 2020-07-26 | 380 valid | 380 |
| 2020–21 | 2021 | 175,815 | 106 | 380 | 2020-09-12 | 2021-05-23 | 380 valid | 380 |
| 2021–22 | 2122 | 175,363 | 106 | 380 | 2021-08-13 | 2022-05-22 | 380 valid | 380 |
| 2022–23 | 2223 | 176,180 | 106 | 380 | 2022-08-05 | 2023-05-28 | 380 valid | 380 |
| 2023–24 | 2324 | 172,196 | 106 | 380 | 2023-08-11 | 2024-05-19 | 380 valid | 380 |
| 2024–25 | 2425 | 197,110 | 120 | 380 | 2024-08-16 | 2025-05-25 | 380 valid | 380 |

For **each season and each of** `PSH, PSD, PSA, PSCH, PSCD, PSCA`: 380 valid
finite decimal prices greater than 1; zero missing, malformed, nonfinite,
nonpositive, or equal-to-one values. Both pre-closing and closing triplets
are complete on 380/380 rows in every season. There are zero duplicate raw
fixture keys, normalized fixture keys, or season/home/away pairs in each
file. The JSON gives all six field-status counts and both triplet counts
separately for every season.

## Reconciliation with corrected EdgeLab results

Matching used the **stated season**, parsed match date, home/away role and
explicit Football-Data team-name aliases, followed by the existing EdgeLab
team canonicalizer. A match was accepted only when exactly one canonical
fixture had the same normalized key and full-time home goals, away goals and
H/D/A result. A unique same-season team pair without a date match was used
only to *diagnose* a date discrepancy, never to force a match.

Nineteen source abbreviations needed an explicit mapping: Bournemouth,
Cardiff, Huddersfield, Hull, Ipswich, Leeds, Leicester, Luton, Man City,
Man United, Newcastle, Norwich, Nott'm Forest, Stoke, Swansea, Tottenham,
West Brom, West Ham and Wolves. The exact target names and occurrence counts
are in the JSON. In total 2,856 source rows required at least one such alias;
eight fixtures matched on the raw names and date alone.

**Result:** 3,800/3,800 source records uniquely reconcile to the corrected
3,800-match canonical dataset, 380/380 in every season. There are zero
unmatched source or canonical fixtures, ambiguous matches, date discrepancies,
result disagreements or duplicate canonical assignments. The 2019–20 odds
source records July matches in **2020**, consistent with the corrected
canonical dates. The canonical file itself was neither edited nor regenerated.

## Temporal limitation and decision

The `Time` column is absent through 2018–19 and present with valid `HH:MM`
kickoff values from 2019–20 onward. It gives **match kickoff time**, not a
timestamp for when any price was observed, and the files provide no odds quote
timestamp or timezone. EdgeLab's saved model forecasts are date-level:
fixtures on a date are forecast before applying any results from that date.
A closing price observed near kickoff could incorporate information from
earlier matches on the same date. Thus price and model information cutoffs
cannot be proven equal from these files.

**Go for a later, explicitly labelled retrospective closing-line reference
study on fixture identity and data completeness. No-go for claiming a
strictly time-aligned pre-match contest or a trading simulation on this
evidence.** Define and freeze the comparison protocol before examining
market-vs-model scores, retain the existing development/validation/held-out
splits, and do not silently substitute non-closing prices. This audit does
not authorize a benchmark implementation or report a market result.

## Reproduce

From the repository root with the existing environment:

```bash
.venv/bin/python scripts/audit_football_data_sources.py
```

The script reads the ten local raw CSVs and canonical results, and writes
only the deterministic aggregate/exception audit JSON at
`data/processed/odds_audit/source_audit.json`. It does not export prices,
calculate implied probabilities, train models, change frozen evaluation
artifacts, or copy/modify any raw CSV.
