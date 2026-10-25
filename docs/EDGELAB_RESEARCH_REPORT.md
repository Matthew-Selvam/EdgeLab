# EdgeLab: Historical Premier League Probability Forecasting

**Research report and release audit · 24 October 2026**

## Abstract

EdgeLab asks whether transparent, reproducible models can assign useful
home-win, draw and away-win probabilities to Premier League fixtures using
only results available before each forecast date. It compares a class-frequency
baseline, an Elo rating model and an independent-Poisson goals model. The
primary evaluation uses fixed development, validation and held-out test
seasons. Separate studies examine seven earlier walk-forward seasons, a
Poisson calibration attempt and a retrospective Pinnacle closing-odds
reference.

Poisson has the lowest observed Brier score and log loss of the three EdgeLab
models in both saved evaluation seasons, but its class-wise calibration is
uneven and its test scores are materially worse than its validation scores.
A single temperature fitted on out-of-sample development predictions gave
tiny validation gains that did not carry to the held-out season. Closing-market
probabilities have lower observed Brier score and log loss than EdgeLab's
models on the matched historical fixtures. Closing quotes cannot be placed at
the same information cutoff as EdgeLab forecasts, so that comparison does not
establish a fair contemporaneous contest or a betting opportunity.

This report is a read-only synthesis of the saved artifacts listed below. It
does not fit a model, alter an evaluation split or publish fixture-level odds.

## Question, scope and evidence

The practical question is how well simple, inspectable historical methods
forecast three football outcomes **out of sample**, and where probability
quality breaks down. A useful model should be assessed with proper
probability scores and calibration as well as winner-picking accuracy.
EdgeLab is a retrospective research pipeline and a saved-results dashboard.
It does not make live forecasts or simulate wagers.

Numbers in this report come from the local canonical match file and the
saved evaluation, robustness, calibration and market-benchmark metrics.
During this audit, the frozen model scores (six model/split cells), 21
development model/fold cells and nine market/season cells were recomputed
from their saved prediction rows and agreed with the corresponding metrics
JSON. Canonical and saved-forecast SHA-256 values also agreed across the
artifact provenance records. Scores below are rounded for readability;
the machine-readable files retain their full precision.

## Data and provenance

[OpenFootball's football.json project](https://github.com/openfootball/football.json)
supplies the historical Premier League results. The corrected canonical
`data/processed/premier_league_matches.csv` contains **3,800 matches**,
exactly 380 in each season from **2015–16 through 2024–25**. Match dates span
**2015-08-08 to 2025-05-25** across 1,138 distinct calendar dates. Outcomes
are 1,691 home wins, 886 draws and 1,223 away wins. The ingestion report
records 3,800 imported, zero skipped and zero rejected. Original source team
names remain in the canonical data; the model layer maps observed aliases to
stable club identities.

The OpenFootball 2019–20 source misdated **66 matches** from Matchdays
32–38 as **1–26 July 2019** rather than **1–26 July 2020**. The documented
normalizer changes only the year for the audited rows, preserving month,
day, teams and results. Valid June 2020 fixtures are untouched. Raw JSON
remains unchanged, and `data/processed/date_corrections.json` records each
correction and source hash. See the
[provider issue](https://github.com/openfootball/football.json/issues/54)
and [correction impact audit](DATE_CORRECTION_IMPACT.md). The corrected
canonical CSV SHA-256 is
`db975a7f31eafe2848b5e7e8a56437d47ce6d5aa260462c52ee21964a5e1ccb9`.

The separate bookmaker source is ten **manually downloaded**
[Football-Data.co.uk](https://football-data.co.uk/englandm.php) EPL CSVs.
The [source audit](ODDS_SOURCE_AUDIT.md) found 380 complete Pinnacle closing
1X2 triplets per season and uniquely reconciled all 3,800 source fixtures to
the corrected results. Football-Data's
[column key](https://football-data.co.uk/notes.txt) identifies
`PSCH/PSCD/PSCA` as closing home/draw/away odds; the distinct
`PSH/PSD/PSA` fields were not mixed into the benchmark. Raw odds and
fixture-level derived market probabilities remain local and gitignored.
Only aggregate market scores appear here. Football-Data's
[fair-data-use policy](https://football-data.co.uk/contact.php) describes
manual or light download for private personal or academic research and
restricts commercial and heavy repeated automated use; it does not clarify
redistribution of derived fixture-level data.

## Forecast protocol and models

**Development** is 2015–16 through 2022–23 (3,040 fixtures).
**Validation** is 2023–24 (380); the **held-out test** is 2024–25 (380).
The frozen evaluation stores 2,280 rows: three models on each of the 760
validation/test fixtures. Settings were fixed before the held-out assessment.
The seven-fold robustness study separately uses 2016–17 through 2022–23 as
2,660 out-of-sample development fixtures; 2015–16's 380 fixtures have no
completed prior season and are explicitly excluded from that study.

The canonical CSV has dates but no reliable kickoff timestamps. All forecasts
for a calendar date are generated **before any result from that date updates**
Elo ratings or Poisson team totals. Saved `information_cutoff_date` values
are earlier than each match date and consistent within model/date batches.
The baseline remains frozen after its training period. Elo and Poisson may
incorporate **earlier completed dates within validation or test** when
forecasting later dates; “held out” protects parameter choice and the
same-date cutoff, rather than forbidding sequential updates. The
`prediction_timestamp` field is a date-resolution label, not an observed
clock time.

- **Class-frequency baseline.** Count H/D/A outcomes in eligible prior
  seasons, add one pseudo-count to each class and divide by `N+3`. The
  primary validation/test vector is fitted on all 3,040 development fixtures
  and then frozen.
- **Elo.** Each club begins at rating 1,500. A home advantage of 60 points
  enters the pre-match difference; completed results update ratings with
  `K=20`. Elo's expected match score is not itself a three-outcome
  probability. A fixed heuristic converts the difference `d` to
  `p_D=1/(1+exp(|d|/350))`,
  `p_H=(1-p_D)/(1+exp(-d/280))`, and `p_A=1-p_D-p_H`,
  followed by probability validation/normalization. The saved identifier
  `elo_logistic` does not mean fitted logistic regression.
- **Independent Poisson.** Development results estimate league home and
  away goal means. Team scoring and conceding means are venue-specific
  and smoothed toward those league means with five pseudo-matches.
  Multiplying the relevant attack and defence factors produces expected
  home and away goals. Independent Poisson goal masses are summed into H/D/A.
  Each marginal omitted tail must be at most `1e-10` with a 30-goal cap;
  retained joint mass is normalized. League means and smoothing strength
  stay fixed, while completed team results update after each full date.

The [methodology](METHODOLOGY.md) gives implementation details and limitations
of these assumptions. No model uses market odds as an input.

## Measures

For actual one-hot outcome `y` and H/D/A vector `p`, **multiclass Brier**
is the mean `Σ(p_k-y_k)²` (range 0–2; lower is better). **Log loss** is
mean `−log(p_actual)`, with a `1e-15` numerical floor (lower is better).
**Accuracy** is the fraction where the largest probability names the actual
outcome (higher is better; ties follow H/D/A order). Scores compare the same
fixtures within each season or split.

Each class also has a one-versus-rest reliability table with 10 equal-width
probability bins. **Expected calibration error (ECE)** is the
count-weighted mean absolute difference between bin-average probability and
observed class frequency (lower is better). ECE depends on bin edges and can
be unstable with sparse bins; it is not a complete calibration test.
The frozen evaluation also saves ranked probability score, but the main
comparison below emphasizes Brier, log loss, accuracy and class-wise ECE.

## Frozen validation and held-out results

All rows in each table refer to the **same 380 fixtures**. Values are from
`data/processed/predictions/metrics.json`; the 2024–25 season was not used
to choose the model settings.

| 2023–24 validation model | n | Brier ↓ | Log loss ↓ | Accuracy ↑ | ECE H ↓ | ECE D ↓ | ECE A ↓ |
|:--|--:|--:|--:|--:|--:|--:|--:|
| Baseline | 380 | 0.637105 | 1.054054 | 0.4605 | 0.0129 | 0.0182 | 0.0052 |
| Elo heuristic | 380 | 0.632454 | 1.045291 | 0.4421 | 0.1365 | 0.1904 | 0.0904 |
| Independent Poisson | 380 | 0.560905 | 0.950870 | 0.5500 | 0.0439 | 0.0272 | 0.0397 |

| 2024–25 held-out model | n | Brier ↓ | Log loss ↓ | Accuracy ↑ | ECE H ↓ | ECE D ↓ | ECE A ↓ |
|:--|--:|--:|--:|--:|--:|--:|--:|
| Baseline | 380 | 0.655589 | 1.080894 | 0.4079 | 0.0397 | 0.0108 | 0.0289 |
| Elo heuristic | 380 | 0.648356 | 1.068165 | 0.3974 | 0.0839 | 0.1621 | 0.1111 |
| Independent Poisson | 380 | 0.639256 | 1.061357 | 0.4605 | 0.0663 | 0.0205 | 0.0700 |

Poisson has the lowest observed Brier and log loss among the three EdgeLab
models in both seasons. Its observed Brier advantage over the baseline
narrows from 0.076200 in validation to 0.016333 in test. Its test ECE
exceeds the baseline's for **all three classes**, despite better aggregate
probability scores. Elo's draw ECE is 0.1904 in validation and 0.1621 in
test. These are one-season comparisons, not estimates of long-run ranking.
The original frozen report includes no uncertainty interval for these
model-only split scores.

## Seven-fold development robustness

Each fold fits on completed earlier development seasons with unchanged
settings, then scores 380 later fixtures. The table retains the
**Poisson-minus-baseline** Brier difference and the saved paired 95%
percentile interval. Negative differences favour Poisson on Brier. The
1,000 bootstrap replicates resample whole calendar dates within a season,
preserving paired predictions and same-date fixture groups.

| Scored season | n | Baseline Brier | Elo Brier | Poisson Brier | Poisson − baseline Brier [95%] |
|:--|--:|--:|--:|--:|:--|
| 2016–17 | 380 | 0.6371 | 0.6718 | 0.5680 | −0.0691 [−0.0934, −0.0447] |
| 2017–18 | 380 | 0.6443 | 0.6382 | 0.5816 | −0.0628 [−0.0886, −0.0372] |
| 2018–19 | 380 | 0.6319 | 0.6369 | 0.5359 | −0.0960 [−0.1194, −0.0719] |
| 2019–20 | 380 | 0.6434 | 0.6323 | 0.5780 | −0.0654 [−0.0891, −0.0416] |
| 2020–21 | 380 | 0.6630 | 0.6738 | 0.6149 | −0.0481 [−0.0754, −0.0210] |
| 2021–22 | 380 | 0.6478 | 0.6369 | 0.5637 | −0.0841 [−0.1098, −0.0584] |
| 2022–23 | 380 | 0.6339 | 0.6527 | 0.6128 | −0.0211 [−0.0479, +0.0073] |

Poisson's observed Brier and log loss are below both comparators in all
seven folds, but the **2022–23 Poisson-minus-baseline Brier and log-loss
intervals include zero**. Its 10-bin ECE exceeds the baseline in 5/7 folds
for H, 5/7 for D and 6/7 for A. The unadjusted paired ECE intervals are
entirely above zero in 4/7, 0/7 and 3/7 folds, respectively. Thus the
class-wise pattern varies by season and outcome. These within-season
bootstrap intervals do not capture dependence across dates or variation
across future seasons, and the many metrics and comparisons are not adjusted
for multiplicity. They support careful description, not a population-level
significance claim. Full scores, log-loss/accuracy intervals, reliability
bin counts and the 380 excluded 2015–16 rows are saved under
`data/processed/robustness/`.

## Poisson temperature-scaling experiment

The calibrator fitted **one shared inverse temperature** by minimizing
multiclass log loss on **2,660 genuinely out-of-sample development Poisson
forecasts** from the seven folds. It did not fit to in-sample goals-model
predictions or use validation/test labels for its parameter. The saved fit
is inverse temperature **1.018799** (temperature **0.981548**). A positive
temperature preserves the most likely class, so accuracy cannot improve
under this transform.

| Assessment | Variant | n | Brier ↓ | Log loss ↓ | Accuracy ↑ | ECE H ↓ | ECE D ↓ | ECE A ↓ |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|
| Validation 2023–24 | Raw Poisson | 380 | 0.560905 | 0.950870 | 0.5500 | 0.0439 | 0.0272 | 0.0397 |
| Validation 2023–24 | Temperature | 380 | 0.560678 | 0.950499 | 0.5500 | 0.0485 | 0.0299 | 0.0368 |
| Held-out 2024–25 | Raw Poisson | 380 | 0.639256 | 1.061357 | 0.4605 | 0.0663 | 0.0205 | 0.0700 |
| Held-out 2024–25 | Temperature | 380 | 0.640188 | 1.062836 | 0.4605 | 0.0679 | 0.0229 | 0.0743 |

The validation Brier change (temperature minus raw) is **−0.000227** with
paired date-bootstrap interval **[−0.000718, +0.000317]**: it does not
resolve a gain. On the held-out season Brier worsens by **+0.000932**
**[+0.000352, +0.001496]**, and log loss worsens by **+0.001479**
**[+0.000601, +0.002323]**. Test ECE also increases for H, D and A;
accuracy is unchanged. These intervals are descriptive and unadjusted for
multiple comparisons or uncertainty in the fitted temperature. The
experiment **did not demonstrate a durable calibration improvement**.
Raw Poisson remains EdgeLab's default; the calibrated version is labelled
experimental in the dashboard.

## Retrospective closing-market reference

For each audited Pinnacle closing decimal price `o_i`, the benchmark
computes raw implied `q_i=1/o_i` and removes the overround by
**proportional normalization**, `p_i=q_i/Σq`. Overround is `Σq−1`.
This is one margin-removal convention, not the only defensible one. The
benchmark reads saved EdgeLab forecasts rather than training a new model.
All **3,420 scored fixtures** (seven development folds plus validation and
test) have complete closing triplets and matched model rows; the first
season's 380 closing triplets are unscored for the stated historical-data
reason. Public documentation contains only aggregate results.

| Matched season | Source | n | Brier ↓ | Log loss ↓ | Accuracy ↑ | ECE H ↓ | ECE D ↓ | ECE A ↓ |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|
| Validation 2023–24 | Pinnacle closing | 380 | 0.525848 | 0.899550 | 0.5974 | 0.0416 | 0.0088 | 0.0464 |
| Validation 2023–24 | Raw Poisson | 380 | 0.560905 | 0.950870 | 0.5500 | 0.0439 | 0.0272 | 0.0397 |
| Held-out 2024–25 | Pinnacle closing | 380 | 0.575079 | 0.966412 | 0.5526 | 0.0514 | 0.0185 | 0.0584 |
| Held-out 2024–25 | Raw Poisson | 380 | 0.639256 | 1.061357 | 0.4605 | 0.0663 | 0.0205 | 0.0700 |

Across the nine matched seasons, the closing-market vector has lower
observed Brier and log loss than baseline, Elo and raw Poisson. The closest
Poisson Brier comparison is 2019–20: market minus Poisson is **−0.0036**
with within-season paired date-bootstrap interval **[−0.0203, +0.0137]**.
On validation the market-minus-Poisson Brier difference is **−0.0351**
**[−0.0543, −0.0167]**; on test it is **−0.0642**
**[−0.0907, −0.0391]**. These intervals are unadjusted, describe fixed
historical seasons and do not settle long-run ranking.

This reference is **not a strictly time-aligned forecast comparison**.
Football-Data supplies kickoff `Time` only from 2019–20 onward and no
odds-quote timestamp or timezone in any audited season. A closing price
near kickoff could incorporate information unavailable at EdgeLab's
before-date cutoff, including earlier same-day results. It cannot establish
that EdgeLab “beat the market,” nor that either vector was executable at
these prices. No profitability conclusion follows.

## Limitations, unresolved questions and next experiment

The models are deliberately simple. The baseline ignores teams and form;
the Elo H/D/A conversion is a fixed heuristic, especially weak for draw
calibration; the independent-Poisson model cannot express goal dependence
or changing tactical conditions. One validation and one test season cannot
establish stable future performance. ECE bins are sparse in places.
Within-season date bootstrap leaves cross-date dependence, season
variation and multiple comparisons unresolved. Corrected source dates
are documented, but fixture-level kickoff and odds quote timing remain
insufficient for point-in-time market alignment.

The failed shared-temperature experiment suggests a **pre-registered,
class-specific calibration test** as the next scientific step. The
hypothesis is that a small H/D/A bias adjustment fitted only to earlier
out-of-sample development forecasts could reduce class-wise error without
worsening proper scores. It should be assessed with rolling development
folds, then a fixed validation protocol, and finally a **new, previously
unexamined complete season** after the method is frozen. Reusing the
already-inspected 2024–25 test to select or claim success for a new
calibrator would invalidate that test. This experiment has **not** been
implemented.

## Reproduction and artifact identity

Use a **fresh, isolated checkout** for regeneration: the commands below
write ignored outputs, while this audit intentionally left the existing
frozen files untouched. Python 3.12 and the manually obtained ten
OpenFootball JSONs under `data/raw/openfootball/` are prerequisites.
The separate ten Football-Data EPL `E0.csv` files must be placed in the
season folders named in [the source audit](ODDS_SOURCE_AUDIT.md); there is
no scraper or repeat-download job. Source access and terms remain the
reader's responsibility. The pinned `requirements.txt` describes the
audited environment; `pyproject.toml` defines the local package.

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install -e . --no-deps
.venv/bin/python -m edgelab.data
.venv/bin/python -m edgelab.evaluation --input data/processed/premier_league_matches.csv --output-dir data/processed
.venv/bin/python -m edgelab.evaluation.robustness --input data/processed/premier_league_matches.csv --reference-dir data/processed/predictions --output-dir data/processed/robustness
.venv/bin/python -m edgelab.evaluation.calibration --input data/processed/premier_league_matches.csv --robustness-dir data/processed/robustness --frozen-dir data/processed/predictions --output-dir data/processed/calibration
.venv/bin/python scripts/audit_football_data_sources.py
.venv/bin/python -m edgelab.evaluation.market_benchmark
.venv/bin/python -m pytest -q
.venv/bin/streamlit run src/edgelab/dashboard/app.py
```

The dashboard reads saved historical predictions, robustness and calibration
artifacts; it does not refit models or display a live market benchmark.
For checking a local artifact set without regeneration, compare SHA-256
values below and run the tests. The following saved hashes were verified
during this audit:

| Local artifact | SHA-256 |
|:--|:--|
| `data/processed/premier_league_matches.csv` | `db975a7f31eafe2848b5e7e8a56437d47ce6d5aa260462c52ee21964a5e1ccb9` |
| `data/processed/predictions/predictions.csv` | `22fb97231cdcb4e8b9f60500c9ed0b65cee62d2e04b604617fe4c88f6e0f4927` |
| `data/processed/predictions/metrics.json` | `591319922d52dbb6aef7121620bcc3d14f2075e47797467576e39fc70e289ad4` |
| `data/processed/robustness/metrics.json` | `ad6dc39eacf52217c999664ad3e35951683eda4e2a927f3e2bb392f695be0ccd` |
| `data/processed/calibration/metrics.json` | `17a6dcc65e994540b9607a329c925839c8e64bf981fce556fb35a3a4cce58958` |
| `data/processed/market_benchmark/metrics.json` | `006b96e6f0e3a916ba2ef8cb88d4e8278330d9a54eab0cc87baaa07aed1233b1` |

## Release-readiness audit

This section records the original research-report audit before release
hardening. See [the subsequent release audit](RELEASE_READINESS.md) for the
curated staging set, new safeguards, isolated reproduction results and remaining
licensing/commit decisions. Scientific findings and historical artifact hashes
in this report remain unchanged.

The research **content** is coherent with the saved files: the code's
three-outcome models and split definitions match the methodology; saved
scores independently recompute; the 66 corrected fixtures, 380-match
seasons and market coverage agree with their audits. The full suite
passed **207 tests** during this documentation audit, including
same-date, provenance, metric, market-source and Streamlit AppTest checks.
No numerical disagreement was found in the audited saved metrics. The
README's obsolete claim that team-name mapping was deferred, and the
dated odds audit's lack of a link to the later benchmark, were corrected.
This is not a claim of exhaustive validation.

The **repository is not yet ready for a reproducible public release**.
On this checkout, `git ls-files` returns **zero tracked files** and
`git rev-parse --verify HEAD` finds **no commit**; there is no remote
configured. Raw and processed data are covered by `.gitignore`, but the
source, documentation and tests have no committed release snapshot.
The 5.7 MB generated `graphify-out/` tree is currently unignored and
would be included by a broad add. No repository `LICENSE` file is
present, and a fresh reader must manually obtain source data because
raw and processed files are intentionally excluded. Before publication,
create a curated initial commit, decide what generated graph material
belongs in Git, declare code licensing, document the manual source
acquisition and validate a clean-clone reproduction. Keep odds and
fixture-level derived market data outside the public repository;
clarify permissions for any broader use. The market-benchmark CLI accepts
an alternate output directory, so a user could otherwise write derived
fixture-level probabilities to a trackable location; a public release
should guard or clearly restrict that path. The initial commit must include
the ignore rules. No staging, commit, publication or data export was
performed in this audit.

## Portfolio conclusion

EdgeLab demonstrates careful chronological evaluation, explicit
source correction, paired historical comparisons and an honest negative
calibration result. Independent Poisson is the strongest of its three
simple models by observed Brier and log loss in the frozen validation and
test seasons, while draw and other class-wise calibration remain imperfect.
The retrospective closing-line reference supplies useful context but
cannot support a contemporaneous market or wagering claim. The work is
suitable for a student research narrative **once the repository release
blockers above are resolved**; the current checkout should not yet be
presented as a reproducible public release.
