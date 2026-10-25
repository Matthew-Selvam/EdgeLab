# Reproducing EdgeLab

Run commands from the repository root in a fresh, isolated checkout. Generation
commands write local ignored outputs; do not run them over an artifact snapshot
you intend to preserve. EdgeLab supports Python **3.12** for this audited
workflow (tested on Python 3.12.13, macOS arm64). Other Python versions and
platforms have not been verified. `requirements.txt` pins the audited environment;
`pyproject.toml` declares package dependencies and console commands.

No raw datasets or historical generated outputs are included in the public file
allowlist. The complete test suite includes real-data and saved-artifact checks,
so generate the core artifacts before running it. A bare checkout without those
inputs is not expected to pass the full suite.

## 1. Environment

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install -e . --no-deps
.venv/bin/python -m pip check
```

Editable installation is intentional: the CLI and dashboard use this checkout's
`data/` tree and dashboard stylesheet. These instructions do not claim a portable
wheel installation or offline dependency installation. The requirements snapshot
includes packages from the original research environment beyond the minimal
runtime; no new dependency is needed for release hardening.

## 2. Acquire the permitted results source

[OpenFootball football.json](https://github.com/openfootball/football.json#license)
dedicates its data to the public domain under CC0. The ten EPL source files for
2015–16 through 2024–25 are pinned to revision
`e6744429ee395bc86f247348c6184bb08d4eb361`. The exact URLs and SHA-256 hashes
are in `docs/OPENFOOTBALL_SOURCES.json`.

```bash
.venv/bin/python scripts/fetch_openfootball.py
```

This fixed, one-time acquisition reads only those ten public results URLs. It
downloads missing files into `data/raw/openfootball/YYYY-YY.json`, verifies hashes
before writing, reuses matching existing files, and refuses to replace differing
local sources. It does not access Football-Data.co.uk or fetch odds. Alternatively,
download the manifest's files manually and rename `en.1.json` to each stated
season filename. Do not substitute a moving upstream revision: the 2019–20 source
manifest intentionally contains the documented 66 July date errors corrected by
the ingestion pipeline. Hash drift requires review, not a relaxed test.

## 3. Generate the core historical research artifacts

```bash
.venv/bin/python -m edgelab.data
.venv/bin/python -m edgelab.evaluation --input data/processed/premier_league_matches.csv --output-dir data/processed
.venv/bin/python -m edgelab.evaluation.robustness --input data/processed/premier_league_matches.csv --reference-dir data/processed/predictions --output-dir data/processed/robustness
.venv/bin/python -m edgelab.evaluation.calibration --input data/processed/premier_league_matches.csv --robustness-dir data/processed/robustness --frozen-dir data/processed/predictions --output-dir data/processed/calibration
.venv/bin/python -m pytest -q
```

These commands reproduce the existing fixed methods for verification. They do
not authorise selecting new methods against the inspected 2024–25 test season.
Installed equivalents are `edgelab-ingest`, `edgelab-backtest`,
`edgelab-robustness` and `edgelab-calibrate`. `edgelab-evaluate` regenerates
metrics/report from the default saved predictions and is optional after backtest.

| Local output | Expected check |
|:--|:--|
| `data/processed/premier_league_matches.csv` | 3,800 fixtures, ten seasons × 380; no rejected/skipped results |
| `data/processed/date_corrections.json` | 66 audited July 2019 → July 2020 corrections; raw inputs unchanged |
| `data/processed/ingestion_report.json` | Counts/provenance; `generated_at` changes between runs |
| `data/processed/predictions/` | 2,280 forecast rows = three models × 760 validation/test fixtures; summary, metrics and evaluation report |
| `data/processed/robustness/` | 7,980 forecast rows = three models × 2,660 development fixtures; 380 first-season exclusions; season metrics, reliability bins, JSON and report |
| `data/processed/calibration/` | 760 assessment rows, temperature fitted from 2,660 development OOS Poisson forecasts; metrics, reliability bins, two HTML plots and report |

The [research report](EDGELAB_RESEARCH_REPORT.md#reproduction-and-artifact-identity)
records the original canonical/prediction/metric hashes and aggregate scores.
Canonical CSV and frozen forecast/metric hashes should match in the pinned
environment. Ingestion timestamps, absolute provenance paths in study metadata,
and regenerated HTML identifiers can differ without a scientific change; compare
numeric scores, probabilities, sample sizes, fixed settings and bootstrap
intervals as well as the relevant content hashes. Do not rewrite historical
snapshots merely to make provenance paths identical.

## 4. Launch the saved-results dashboard

```bash
.venv/bin/streamlit run src/edgelab/dashboard/app.py
```

Open the local URL printed by Streamlit (normally `http://localhost:8501`). The
dashboard reads generated artifacts and never trains models. The Research
Explorer requires the robustness and calibration outputs. Missing files produce
an explanatory message. Market derivatives are not dashboard inputs. The full
suite includes Streamlit AppTest; a health endpoint alone is not visual QA.

## 5. Optional retrospective closing-market study: manual inputs

The core workflow and full test suite do **not** require real bookmaker files.
The market study does. Check the current
[Football-Data.co.uk fair-use policy](https://football-data.co.uk/contact.php)
and [England download page](https://football-data.co.uk/englandm.php). Eligible
private academic researchers must manually download the Premier League CSV for
each listed season. There is no odds downloader, scraper or repeated job here.

| Season | Save manually downloaded file as |
|:--|:--|
| 2015–16 | `data/raw/football-data/1516/E0.csv` |
| 2016–17 | `data/raw/football-data/1617/E0.csv` |
| 2017–18 | `data/raw/football-data/1718/E0.csv` |
| 2018–19 | `data/raw/football-data/1819/E0.csv` |
| 2019–20 | `data/raw/football-data/1920/E0.csv` |
| 2020–21 | `data/raw/football-data/2021/E0.csv` |
| 2021–22 | `data/raw/football-data/2122/E0.csv` |
| 2022–23 | `data/raw/football-data/2223/E0.csv` |
| 2023–24 | `data/raw/football-data/2324/E0.csv` |
| 2024–25 | `data/raw/football-data/2425/E0.csv` |

Then, after the core outputs exist:

```bash
.venv/bin/python scripts/audit_football_data_sources.py
.venv/bin/python -m edgelab.evaluation.market_benchmark
```

The audit writes `data/processed/odds_audit/source_audit.json`. Review its hashes,
headers, missingness and exceptions: expected coverage is 3,800 unique matched
fixtures, 380 complete `PSCH/PSCD/PSCA` triplets in each season, zero unresolved
exceptions. The benchmark rejects changed audited inputs, invalid prices,
unmatched/ambiguous fixtures or result disagreements. Updated provider files may
need a new reviewed audit; do not assume identical downloads forever.

The benchmark writes `market_predictions.csv`, `metrics.json`,
`reliability_bins.csv` and `REPORT.md` under
`data/processed/market_benchmark/`. There are 3,800 matched market rows and
3,420 scored fixtures across nine periods; the 380 first-season fixtures are
explicit exclusions. `--output-dir` may select a subdirectory of this ignored
tree, but resolved external paths, traversal escapes and escaping output
symlinks are rejected. There is no external-output override. Preserve an existing
snapshot by using a new subdirectory. Do not publish raw odds or fixture-level
market derivatives. See [licensing boundaries](LICENSING.md).

This optional workflow is **partially reproducible from public contents**: users
must acquire their own eligible manual inputs. Local verification with previously
manually acquired files cannot establish universal access or redistribution
rights. Closing quote timestamps are unavailable, early kickoff times are
incomplete, and model cutoffs are date based; this remains a retrospective
reference, with no claim of executable or profitable forecasts.

## Verification status

See `RELEASE_READINESS.md` for the actual isolated-checkout run, test results,
frozen-artifact checks and outstanding publication decisions. No commit, remote
creation or publication is part of these instructions.
