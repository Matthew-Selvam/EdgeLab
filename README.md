<div align="center">

# ⚽ EdgeLab

### Premier League prediction research, built for honest evaluation.

Baseline · Elo · Independent Poisson · Calibration · Market benchmark

[![Python 3.12](https://img.shields.io/badge/python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![MIT License](https://img.shields.io/badge/license-MIT-8A2BE2.svg)](LICENSE)
[![Premier League](https://img.shields.io/badge/data-Premier%20League-3800%20matches-5B2C83)](docs/EDGELAB_RESEARCH_REPORT.md)

</div>

---

EdgeLab is a reproducible research project for forecasting historical English Premier League match outcomes. It compares three transparent models using chronological, out-of-sample evaluation and keeps validation, held-out testing, calibration studies and a retrospective market benchmark clearly separated.

> **Research only.** The dashboard displays saved forecasts for played matches. It does not predict upcoming fixtures or simulate betting. One held-out season is limited evidence and does not establish long-run superiority.

## At a glance

| Dataset | Evaluation | Models | Saved forecasts |
|:--|:--|:--|:--|
| 3,800 matches · 10 seasons | Development through 2022–23 | Baseline, Elo, Poisson | 2,280 validation/test rows |

### Held-out results · 2024–25

Scores use the same 380 fixtures for every model. Lower Brier score and log loss are better; higher accuracy is better.

| Model | Brier ↓ | Log loss ↓ | Accuracy ↑ |
|:--|--:|--:|--:|
| Class-frequency baseline | 0.6556 | 1.0809 | 40.79% |
| Elo | 0.6484 | 1.0682 | 39.74% |
| Independent Poisson | 0.6393 | 1.0614 | 46.05% |

These are observed results from one season, not proof of future performance. Poisson has better aggregate probability scores here, while calibration remains a limitation. The separate temperature-scaling experiment did not improve held-out Brier score or log loss; raw Poisson remains the default.

<details>
<summary>See validation results · 2023–24</summary>

| Model | Brier ↓ | Log loss ↓ | Accuracy ↑ |
|:--|--:|--:|--:|
| Class-frequency baseline | 0.6371 | 1.0541 | 46.05% |
| Elo | 0.6325 | 1.0453 | 44.21% |
| Independent Poisson | 0.5609 | 0.9509 | 55.00% |

</details>

## What’s inside

- **Three models:** a historical outcome-frequency baseline, Elo ratings with an explicit three-outcome conversion, and independent Poisson goal rates.
- **Chronological evaluation:** development (2015–16–2022–23), validation (2023–24), and held-out test (2024–25). All fixtures on a date are forecast before that date’s results update model state.
- **Robustness study:** seven expanding-window development folds with season-level metrics, reliability summaries and paired date-bootstrap intervals.
- **Calibration experiment:** temperature scaling fitted only on development out-of-sample forecasts, then assessed separately on validation and test.
- **Research Explorer:** read-only dashboard views of saved comparisons, reliability bins, and calibration results.
- **Retrospective closing-market benchmark:** aggregate comparison with Pinnacle closing 1X2 odds on matched fixtures. Quote timestamps are unavailable, so this is not a strictly time-aligned forecast contest or evidence of executable betting returns.

## Quick start

Requires Python 3.12. From a fresh checkout:

```bash
git clone https://github.com/Matthew-Selvam/EdgeLab.git
cd EdgeLab
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

Download the ten pinned OpenFootball results files and generate the local dataset and forecasts:

```bash
.venv/bin/python scripts/fetch_openfootball.py
.venv/bin/python -m edgelab.data
.venv/bin/python -m edgelab.evaluation --input data/processed/premier_league_matches.csv --output-dir data/processed
```

The ingestion pipeline applies the documented 66-match 2019–20 date correction (July 2019 to July 2020) and writes a fixture-level audit log. It preserves raw source files. The evaluation command writes saved predictions, metrics and a readable report under `data/processed/predictions/`.

Run the test suite and open the saved-results dashboard:

```bash
.venv/bin/python -m pytest -q
.venv/bin/streamlit run src/edgelab/dashboard/app.py
```

The dashboard does not train models or recompute historical forecasts. For exact environment pins, artifact checks and all study commands, see the [reproducibility guide](docs/REPRODUCIBILITY.md).

## Reproduce the research studies

After generating the core dataset and evaluation files:

```bash
.venv/bin/python -m edgelab.evaluation.robustness --input data/processed/premier_league_matches.csv --reference-dir data/processed/predictions --output-dir data/processed/robustness
.venv/bin/python -m edgelab.evaluation.calibration --input data/processed/premier_league_matches.csv --robustness-dir data/processed/robustness --frozen-dir data/processed/predictions --output-dir data/processed/calibration
```

The full report contains the methods, season-level robustness results, calibration findings and limitations: [EdgeLab research report](docs/EDGELAB_RESEARCH_REPORT.md). See [methodology](docs/METHODOLOGY.md) for model assumptions, metrics and temporal rules.

## Optional market benchmark

The closing-market study requires ten **manually downloaded** Football-Data.co.uk season CSVs. EdgeLab has no odds scraper or repeated download job. Check the source’s current [fair-use policy](https://football-data.co.uk/contact.php) and the [local input instructions](docs/REPRODUCIBILITY.md#5-optional-retrospective-closing-market-study-manual-inputs) before use.

The benchmark uses only Pinnacle closing `PSCH/PSCD/PSCA` fields and proportional overround normalization. Raw odds and fixture-level market derivatives are excluded from Git and must remain local. Public summary results are retrospective; quote timestamps are unavailable and date-level model cutoffs prevent strict timing alignment. See the [source audit](docs/ODDS_SOURCE_AUDIT.md) and [licensing boundaries](docs/LICENSING.md).

## Data, licensing and attribution

- The public repository contains original EdgeLab code and documentation, not the local match dataset, raw odds or generated fixture-level outputs. See [the public file allowlist](docs/PUBLIC_FILES.txt).
- OpenFootball results are sourced from a pinned revision and dedicated upstream under CC0. See the [source manifest](docs/OPENFOOTBALL_SOURCES.json) and [OpenFootball license](https://github.com/openfootball/football.json#license).
- Football-Data.co.uk odds retain their source terms. Their availability does not grant redistribution rights, and EdgeLab’s [MIT License](LICENSE) does not cover third-party data or content.
- The project’s licensing scope and source-use boundaries are documented in [LICENSING.md](docs/LICENSING.md).

## Project activity

If you find EdgeLab useful, you can star the repository. The chart below is generated by Star History and updates with public GitHub star counts.

[![Star History Chart](https://api.star-history.com/svg?repos=Matthew-Selvam/EdgeLab&type=Date)](https://star-history.com/#Matthew-Selvam/EdgeLab&Date)

---

<div align="center">

Made for careful, reproducible football research.

</div>
