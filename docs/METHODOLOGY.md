# EdgeLab prediction and evaluation methodology

For the verified aggregate findings, caveats, reproduction commands and
release audit, see the [research report](EDGELAB_RESEARCH_REPORT.md).

Pinned source acquisition and clean-checkout commands are documented in
[REPRODUCIBILITY.md](REPRODUCIBILITY.md). Release hardening does not change the
models, splits or saved research results. The market benchmark confines resolved
output directories and file targets to `data/processed/market_benchmark/`;
external destinations and escaping symlinks are rejected without an override.
The entire data tree and generated outputs are excluded from public Git contents.

## Canonical data audit (2026-10-21)

The existing `data/processed/premier_league_matches.csv` has 3,800 rows and
10 fields: `match_id`, `competition`, `season`, `match_date`, `home_team`,
`away_team`, `home_goals`, `away_goals`, `outcome`, `source`. IDs, competition,
season, date, teams, outcome and source are text fields; goals parse as
nonnegative integers; dates parse as ISO calendar dates. All rows have all
required fields. There are 10 seasons from 2015–16 through 2024–25, exactly
380 matches in each. Date coverage is 2015-08-08 to 2025-05-25 over 1,138
distinct dates, with as many as 10 matches on one date. Outcomes are 1,691 H,
886 D and 1,223 A, and all agree with full-time goals. There are no duplicate
match IDs or duplicate `(season, date, home team, away team)` fixtures. The
ingestion report records 3,800 imported, zero skipped and zero rejected.

There are 47 distinct source team names. Several clubs have both plain and
`FC` forms (for example `Manchester United` / `Manchester United FC` and
`Brighton & Hove Albion` / `Brighton & Hove Albion FC`). The curated
`TeamAliasMapper` maps these to stable identities for Elo while prediction
records retain the original names. All 47 observed aliases map to 34 canonical
clubs. The source's kickoff time is not present
in the canonical CSV, so the pipeline makes no within-day ordering claim.

Before correction, the raw 2019–20 JSON and canonical CSV both placed 66
fixtures in July 2019 (1–26 July) and none in July 2020. They are the six July
fixtures in Matchday 32 and all ten in each of Matchdays 33–38. The
[OpenFootball issue #54](https://github.com/openfootball/football.json/issues/54)
documents that only their year is wrong. During ingestion, the normalizer
changes the year to 2020 only when the source is `2019-20.json`, the season is
`2019-20`, the round is 32–38, and the original date begins `2019-07-`.
Month/day, teams, scores and outcomes are preserved. Because match IDs contain
the date, the 66 corresponding IDs are regenerated. The source file stays
untouched; `data/processed/date_corrections.json` records every original and
corrected fixture ID and date, source row index, raw source SHA-256 and issue
reference. Ingestion checks the audited 66-row fixture signature before
writing the canonical CSV, failing if the provider rows change unexpectedly.

## Temporal policy

Development: 2015–16 through 2022–23. Validation: 2023–24. Held-out test:
2024–25. The pipeline rejects overlapping or chronologically out-of-order
splits. Model parameters are fixed in code and not chosen using test results.
The test remains a held-out evaluation period for parameter selection; as in
sequential forecasting, earlier test **results** may update Elo ratings for
later test dates. Each day's forecasts use only ratings through the previous
completed date. Same-date Elo rating changes are computed from a pre-date
snapshot and applied together, including when a team appears more than once
on a date. Source outcomes are attached to saved prediction rows after the
probabilities have been computed. `prediction_timestamp` is a date-resolution
forecast label, not a fabricated kickoff time; `information_cutoff_date`
records the latest eligible completed match date. The baseline's cutoff stays
at the final development date.

## Models

The **baseline** counts H/D/A across all 3,040 development fixtures and adds
one to each class. Thus `p(c) = (n_c + 1) / (N + 3)`. The resulting vector is
frozen for validation and test. It uses no evaluation outcomes.

The **Elo model** starts each club at 1,500. A new club begins at 1,500.
Before a match, `d = R_home + 60 - R_away` and Elo's expected match score is
`E = 1 / (1 + 10^(-d/400))`. The match result score is 1 for H, 0.5 for D,
and 0 for A. With `K = 20`, the home rating changes by `K(S-E)` and the away
rating by its negative. Ratings carry across seasons by default. Initial
rating, K, home advantage and optional regression toward 1,500 are
configurable. This expected score is **not** a 1X2 home-win probability.

The fixed three-outcome conversion uses `p_draw = 1/(1+exp(|d|/350))`,
`p_home = (1-p_draw)/(1+exp(-d/280))`, and
`p_away = 1-p_draw-p_home`. Values are clamped and normalized to sum to one.
The conversion's scales (350 and 280) are predeclared defaults; they are not
coefficients fitted by logistic regression. The saved model identifier
`elo_logistic` is retained for compatibility with earlier outputs, but the
implementation is a logistic-shaped **heuristic**. Its draw formula can
overpredict draws; inspect calibration as well as aggregate scores.

The **Poisson goals model** assumes independent home and away goal counts:
`G_home ~ Poisson(lambda_home)` and `G_away ~ Poisson(lambda_away)`. Development
fixtures set fixed league venue means `mu_home` and `mu_away`. These encode
league-wide home scoring advantage and are not re-fitted on validation or
test. For a team's home fixtures, its smoothed goals-scored mean is
`(goals_scored_home + s*mu_home)/(home_matches + s)`; its home goals-conceded
mean uses `mu_away`. Away scoring and conceding use the analogous away counts
and league priors. Default `s=5` supplies five pseudo-matches, shrinking
sparse or promoted teams toward the league mean. Unseen teams have factors
of one.

For home team `i` and away team `j`, `lambda_home` is `i`'s smoothed home
scoring mean times `j`'s smoothed away goals-conceded mean, divided by
`mu_home`. `lambda_away` is `j`'s smoothed away scoring mean times `i`'s
smoothed home goals-conceded mean, divided by `mu_away`. The model fits league
means and initializes team totals on development only. After forecasting
*every* match on an evaluation date, it adds that date's scores to team totals.
These totals inform later dates, while league means and the pseudo-match
setting stay fixed. Earlier test results may inform later test-date forecasts
but never change settings or same-date forecasts.

The joint score probability for `(h,a)` is
`Poisson(h; lambda_home) * Poisson(a; lambda_away)`. Summing cells with `h>a`,
`h=a`, and `h<a` gives H/D/A. Each marginal is extended by recurrence until
its omitted tail is at most `1e-10`, subject to a 30-goal cap. If either tail
is still too large at the cap, prediction fails rather than silently using a
materially truncated distribution. The retained joint H/D/A mass is
normalized to one. Invalid or nonfinite rates are rejected. These numerical
limits were fixed before held-out evaluation.

## Evaluation

All three models are scored on exactly the same 380 fixtures in each evaluation
split. The report generator rejects missing or duplicate model rows and
disagreement about fixture date, season, or actual outcome. Predictions save
source and canonical team names, pre-match Elo ratings, Poisson expected goals,
H/D/A probabilities,
actual outcome, forecast date, information cutoff date, model ID, and split.
`backtest_summary.json` saves configuration and SHA-256 of the canonical CSV.

Multiclass Brier is the per-fixture sum of three squared probability errors,
averaged across fixtures (range 0–2). Log loss is the mean `-log(p_actual)`
with probability clipped at `1e-15` for numerical stability. Accuracy is
the fraction for which the highest-probability class equals the outcome, with
ties broken in H/D/A order. Per-class reliability uses 10 equal-width bins;
each bin records count, mean forecast probability, and observed class
frequency. Per-class expected calibration error is the count-weighted mean
absolute gap between those latter two values. Empty bins have count zero.
RPS is additionally reported with ordinal A/D/H order and divisor 2.

## Season robustness and calibration study

`python -m edgelab.evaluation.robustness` runs a separate, read-only study of
the development period. It creates seven expanding-season folds: 2016–17 is
trained on 2015–16, then each later season uses all **completed earlier
development seasons**. The first season has no prior completed season; its
380 fixtures are explicitly listed as excluded. A fold also excludes fixtures
if fewer than 380 prior matches exist. Fold dates must be strictly after all
prior-season dates. Model settings match the original frozen evaluation and
are checked against its saved metadata; no parameter is tuned by this study.

Each fold fits the class prior once on prior seasons and freezes it through
the target season. Elo ratings process prior seasons by date, with the same
fixed rating and probability conversion settings; target-season results update
ratings only after all fixtures on that date have been forecast. Poisson fits
its fixed league home/away goal means and initial team venue totals from the
same prior seasons. Target-season team totals update after each entire date,
while its league means and smoothing strength remain fixed. Prediction rows
record source/canonical team names, rates or ratings where applicable, actual
outcome, date-resolution forecast label, and latest information cutoff date.
All three models must score identical fixture IDs, dates, and outcomes.

The study reuses the existing multiclass Brier, log-loss, accuracy, and 10-bin
class-wise reliability definitions. `season_metrics.csv` and
`reliability_bins.csv` are suitable for optional plotting in the dashboard;
each reliability bin includes its sample count, observed positive count and
an approximate 95% Wilson interval for observed frequency. Empty bins have
null intervals. For model comparisons, 1,000 paired percentile bootstrap
replicates resample *calendar dates* within each season, preserving same-date
fixtures and paired predictions; the seed is fixed at 20261021. Intervals
cover Poisson minus baseline or Elo differences in Brier, log loss, accuracy,
and one-versus-rest expected calibration error (ECE) for H/D/A. These are
approximate within-season intervals, not evidence of generalization across
future seasons: dates remain serially dependent, binning is coarse, some bins
are small, and the many displayed comparisons are not adjusted for multiple
testing. A difference interval that spans zero does not resolve its sign.

The existing 2023–24 validation and 2024–25 held-out test predictions and
metrics are verified against their canonical CSV hash, fixed settings, split
identity, fixture counts and recomputed aggregate scores, then included as
**read-only references** in the study report. They are not added to the
development walk-forward sample and their saved files are not rewritten. The
held-out test remains one season and was not used to choose this study's
settings, bins, or uncertainty method.

## Poisson temperature-calibration experiment

`python -m edgelab.evaluation.calibration` reads the saved Poisson rows from
the seven-fold robustness walk-forward predictions. They are genuinely out of
sample with respect to the goals model: each fold used only completed prior
seasons for its initial fit and prior dates for within-season updates. The
calibrator's 2,660 fit rows span 2016–17 through 2022–23. It never fits to
Poisson forecasts made on matches used to estimate that goals model. The
loader verifies canonical source hash, fold training cutoffs, fixture IDs,
dates, outcomes, season counts, and the previous-date information cutoff for
every date batch. It also verifies the original frozen validation/test file
hashes against the robustness study before assessment.

For each valid raw H/D/A vector `p`, the calibrated vector is
`q_k(beta) = exp(beta * log(max(p_k, 1e-15))) /
sum_j exp(beta * log(max(p_j, 1e-15)))`, where `beta = 1/T >= 0` is one shared
inverse temperature. The probability floor is solely for finite logarithms.
A shifted log-sum-exp calculation avoids overflow and underflow. `beta=1`
reproduces the raw vector to floating-point precision; positive beta
preserves its class ranking; `beta=0` is the uniform limit (`T=∞`).

The fitting objective is mean multiclass negative log likelihood on the
development walk-forward rows, not Brier, ECE, accuracy, or a held-out score.
The objective is convex in `beta` for fixed input log probabilities. A
deterministic bisection of its derivative searches `0 <= beta <= 1e6`, with
explicit boundary handling. There is one parameter for all classes. No
class-specific bias, second method, feature, or hyperparameter is selected
from validation or test. The development-fit loss is a training diagnostic,
not an unbiased estimate of the calibrator's benefit.

Once `beta` is fitted, the experiment assesses the unchanged 2023–24
validation fixtures, then the unchanged 2024–25 held-out test fixtures. Raw
and calibrated forecasts are compared on the exact same 380 fixtures per
season using the original Brier, log-loss, accuracy and 10-bin one-versus-rest
reliability/ECE definitions. Reliability-bin CSV rows include sample counts,
observed positives, and approximate Wilson intervals for observed frequency;
the corresponding HTML plots show both variants and a perfect-calibration
diagonal. A fixed-seed, 1,000-replicate paired date bootstrap supplies
descriptive intervals for calibrated-minus-raw differences. This resamples
same-date fixtures together, but does not account for serial dependence
across dates or uncertainty in the fitted temperature. Multiple displayed
metrics and classes are not adjusted for multiplicity.

The experiment writes only under `data/processed/calibration/`. The original
uncalibrated predictions, metrics, robustness study, and dashboard inputs
remain unchanged. Temperature scaling can change probability quality without
changing predicted classes. One shared scalar may not fix a draw-specific or
other class-specific bias; the one-season held-out test cannot establish
long-run benefit.

## Limits

Date-only ordering means forecasts are conservatively treated as if made
before every match on that date. The class-frequency baseline ignores teams
and recent form. The Elo probability conversion is a fixed heuristic and is
not statistically calibrated on development data. The Poisson model assumes
independent scores and does not account for low-score dependence or tactical
changes. Its venue strengths are smoothed averages, not a fitted
maximum-likelihood attack/defence system. One validation and one
test season cannot establish stable superiority. Calibration bins are sparse
in places. The original evaluation report has no uncertainty intervals; the
separate robustness study provides approximate paired intervals. The original
evaluation did not use odds; the separate closing-market study below is a
retrospective reference using saved forecasts.

## Retrospective closing-market benchmark

The [odds source audit](ODDS_SOURCE_AUDIT.md) records Football-Data.co.uk's
field definitions, fair-use terms, attribution and local-only storage. Ten
manually downloaded Premier League CSVs were audited against all 3,800
corrected fixtures. The documented Pinnacle closing columns `PSCH`, `PSCD`,
`PSCA` contain complete, valid triplets for every season; the distinct
non-closing `PSH`, `PSD`, `PSA` columns must not be substituted silently.
`python -m edgelab.evaluation.market_benchmark` reads these local CSVs and
the audited alias mapping, requires their SHA-256 hashes to match the source
audit, and matches on season, date, normalized home and away team, full-time
scores and outcome. Every source row must match exactly one canonical fixture;
missing, duplicate or disagreeing records cause failure. All 3,800 match.
The first season's 380 fixtures have no eligible earlier development season
for out-of-sample model forecasts and are excluded from scoring. The next
seven expanding-window seasons plus validation and test each have 380 matched
fixtures for the closing market, baseline, Elo and raw Poisson. The new study
does not refit, update or tune any model, and confirms its input hashes and
saved model scores before reporting results.

For closing decimal odds `o_H, o_D, o_A`, raw implied values are `q_i=1/o_i`.
The reported market vector is `p_i=q_i/sum(q)`, and overround is `sum(q)-1`.
This proportional normalization is one possible margin-removal method, not
a unique fair-price estimate. Prices must be finite and greater than one;
probabilities must be finite, positive and sum to one. The local
`data/processed/market_benchmark/market_predictions.csv` retains raw implied
`q`, normalized `p`, and overround without copying prices. The local metrics
JSON, reliability-bin CSV and report contain 10-bin class-wise ECE, multiclass
Brier, log loss and accuracy for every scored period. Brier is the mean sum
of three squared errors; log loss is the mean negative log probability of the
actual outcome; accuracy uses H/D/A argmax. Binning and sparse bins limit ECE.

Market-minus-model paired Brier, log-loss and accuracy differences have 95%
percentile intervals from 1,000 fixed-seed resamples of calendar dates within
each period. Resampling whole date batches retains the paired and same-date
structure. Negative Brier/log-loss differences and positive accuracy
differences descriptively favour the market. The intervals do not handle
serial dependence across dates, season-to-season variation, or multiple
comparisons. The market has lower observed Brier and log loss than each model
in each scored season, including validation and test; this is a descriptive
finding under the closing-line protocol, not a claim about future seasons.
On validation the market's Brier/log-loss/accuracy are
`0.5258 / 0.8995 / 0.597`; on the single held-out season they are
`0.5751 / 0.9664 / 0.553`. Full class-wise calibration and paired differences
are in the separate local report and machine-readable outputs.

This is expressly a **retrospective closing-line reference**. Kickoff times
are absent through 2018–19 and odds quote timestamps are absent throughout,
so a strictly time-aligned comparison with EdgeLab's date-level forecasts
cannot be established. Later same-date information may be reflected in a
closing price. The study supports no execution, bankroll or profitability
claim. The [source's fair-use policy](https://football-data.co.uk/contact.php)
permits private personal/academic research by manual or light download and
does not clarify redistribution of derived fixture-level data; all such
artifacts remain local under gitignored `data/processed/`. The existing
frozen predictions, robustness study, calibration experiment and dashboard
remain byte-for-byte unchanged. Validation and test split definitions are
unchanged, and test results were not used to select this method.

## Dashboard provenance

The Streamlit dashboard is a read-only view of the generated predictions CSV
and metrics JSON. It uses the canonical match CSV only to add full-time scores
to history rows. The page does not refit models or generate new forecasts.
Model comparison and reliability panels display saved full-split metrics and
bins; period, season, model and team filters select visible rows, not a new
scoring sample. The fixture explorer displays each selected match's stored
model probabilities and Poisson goal means. Validation and held-out test remain separate. The test
contains one season, so its aggregate ranking and calibration estimates are
limited evidence about future performance.

The Research explorer is a separate read-only view of
`data/processed/robustness/season_metrics.csv`, its reliability-bin CSV and
metrics JSON, plus the corresponding saved calibration experiment files. It
shows all seven development fold scores and counts, selectable H/D/A
reliability plots on 0–1 axes, and the saved paired date-bootstrap intervals.
These intervals describe within-season or within-split differences and retain
the serial-dependence, fit-uncertainty and multiple-comparison limitations
described above. ECE changes with the bin definition, and observed frequencies
in sparse bins are unstable. The calibration view labels its temperature
scaling as experimental, shows the fitted temperature and 2,660 development
fit rows, and keeps 2023–24 validation and 2024–25 held-out test assessments
separate. The saved results do not demonstrate a consistent calibration gain.
No dashboard filter refits temperature, updates a model, or modifies the
frozen predictions or metrics; raw Poisson remains the main dashboard default.
If research files are absent or inconsistent with the frozen files, that study
view reports the problem while the original panels remain usable.
