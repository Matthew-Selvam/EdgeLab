# Public release hardening audit

This audit follows the historical readiness section in
`EDGELAB_RESEARCH_REPORT.md`. Research conclusions, model settings and original
artifacts are unchanged. No new model, feature, dependency or dashboard change
was introduced.

## Public Git contents

The starting checkout had zero tracked files, branch `main`, no `HEAD` commit,
no history and no remote. There were no previously tracked restricted files to
remove. `docs/PUBLIC_FILES.txt` is the explicit, reviewed initial staging
allowlist: source, synthetic and integration tests, configuration, dependencies,
the source audit script, the pinned OpenFootball acquisition helper and manifest,
README and relevant research/reproduction/licensing documentation.

No directory wildcard or broad staging command is needed. Review the exact
allowlist and staged contents before an initial commit:

```bash
cat docs/PUBLIC_FILES.txt
git diff --cached --name-only
git diff --cached --stat
git diff --cached --check
git diff --cached
git status --short --ignored
```

The preparation stages only the individual paths in the allowlist. Owner review
of the staged snapshot remains pending. No commit, remote, push or publication
has been created.

Excluded: all of `data/` (including raw odds, corrected results, historical
forecasts, market derivatives and pre-correction archives), `artifacts/`,
`reports/`, generated `graphify-out/`, local stale `HANDOFF.md`, environments,
build/egg metadata, caches, logs, machine/editor files and secrets. Root-level
`E0*.csv` and `market_predictions*.csv` are additionally ignored to guard common
accidental copies. There are no binary files or symlinks in the public allowlist.
Source provenance hashes and aggregate research tables are public; actual odds
and fixture-level market probabilities are absent. Synthetic toy prices in tests
are not bookmaker records. A pattern scan found no credentials/private keys or
machine-specific absolute paths in the allowlist; pattern scanning is not an
exhaustive security guarantee.

`git check-ignore -v` confirmed coverage of actual raw odds, market predictions,
ordinary historical predictions, Graphify output and environment/build files,
plus representative `.env.local`, `.streamlit/secrets.toml`, reports and root
odds-copy paths. `git status --short --ignored` and the exact prospective Git
list were inspected. Empty local `configs/` and `data/demo/` contain no releasable
files. `.git/` is Git metadata, not public project content. Graphify was updated
after code changes and its generated tree remains ignored.

## Market output safeguard

The API and CLI now allow outputs only beneath this checkout's
`data/processed/market_benchmark/`. The requested directory and all four output
file targets are resolved before input reads and rechecked before writes.
External absolute/relative destinations, `..` escapes, similarly named sibling
directories, redirected data/output directories and escaping file symlinks are
rejected. Safe subdirectories remain supported. No external override exists.
The original independent input-directory protection remains in place. This is
protection for ordinary local use, not a sandbox against concurrent hostile
filesystem mutation. Output logs expose counts and paths, not odds prices.

## Actual isolated-checkout reproduction

An equivalent clean checkout was made by copying only public-allowlisted
source/configuration/tests/documentation to a new temporary directory; no existing
raw or processed data, virtual environment or egg metadata was copied. There is
no source commit to clone yet. These checks used a fresh Python 3.12.13 virtual
environment on macOS arm64:

1. Installed `requirements.txt` and editable package with the documented commands;
   `pip check` reported no broken requirements. Cached wheels were available;
   this does not establish offline or cross-platform reproduction.
2. Downloaded ten public-domain OpenFootball JSONs from the pinned revision.
   All ten hashes matched the original local raw files. A repeat acquisition
   downloaded zero files and preserved existing sources.
3. Ran ingestion, fixed baseline/Elo/Poisson evaluation, seven-fold robustness
   and temperature calibration, in the order in `REPRODUCIBILITY.md`.
4. The regenerated canonical CSV, correction audit, frozen predictions and
   metrics, robustness predictions and metrics, and calibration predictions and
   metrics were all byte-identical to the original snapshots. Sample counts were
   3,800 canonical fixtures, 2,280 frozen forecast rows, 7,980 development forecast
   rows, 380 first-season exclusions and 760 calibration assessment rows.
5. The full suite passed **226 tests in 7.20 seconds** in this clean environment,
   without real odds files. Existing Streamlit AppTest checks passed. A real
   Streamlit server also launched with the generated data and returned `ok` from
   its health endpoint. No new rendered screenshot/visual inspection is claimed.
6. To verify the optional market workflow, the ten previously manually downloaded
   odds files were copied locally into the isolated ignored data tree. No odds
   were downloaded automatically. The source audit found **3,800 unique matches,
   zero exceptions**; the benchmark scored **3,420 fixtures across nine periods**.
   Market predictions, reliability bins and report reproduced byte-for-byte.
   All aggregate metrics, paired intervals, coverage and protocol matched. The
   market JSON differs only in input-hash provenance: absolute checkout paths and
   the absence of the original historical `pre_correction/` archive in the clean
   checkout. That archive is not used for scoring.
7. The optional `edgelab-evaluate` command also ran successfully in the isolated
   checkout; resulting frozen predictions and metrics remained byte-identical.

The ingestion report's `generated_at` is expected to change. No numerical
disagreement was found. The original checkout's **47 data/artifact files**,
including all raw sources, frozen forecasts/metrics, robustness, calibration,
market outputs and pre-correction archives, retained their original SHA-256
hashes. They were neither regenerated nor staged here.

The original full-suite baseline was **207 passed in 6.89 seconds**. After
hardening it passed **226 tests in 6.81 seconds** in the original checkout:
16 additional market-path cases and three source-preservation tests. No test was
removed or weakened. Commands and run order are in `REPRODUCIBILITY.md`.

## Final staged review and remaining publication decisions

The final review on **2026-10-25** checked the complete staged text snapshot,
including source, tests, configuration, documentation and ignore rules, using
content/secret/path checks and review of data acquisition and output boundaries.
The final index matches `PUBLIC_FILES.txt` exactly: **62 files**, with no binary
files, symlinks, raw odds, fixture-level market derivatives, personal session
notes, machine-specific absolute paths, generated bulk outputs or detected secret
patterns. Existing source attribution and third-party data restrictions remain.
`git diff --cached --name-status`, `--stat` and `--check` were inspected; the
whitespace check passed. The full suite, `.venv/bin/python -m pytest -q`, passed
**226 tests in 7.97 seconds**. All **47 raw/research artifact files** retained
their pre-review SHA-256 hashes; the six artifact hashes published in the
research report also still match. The new licence matches the standard MIT text
with the authorised notice. Only `LICENSE`, README, licensing documentation,
allowlist and this release audit changed in this final step; implementation,
tests and research results did not change. No blocking discrepancy was found.
Secret scanning and source review do not independently establish ownership or
guarantee the absence of every possible security issue.

- **Code licensing resolved (2026-10-25):** the owner selected MIT for original
  EdgeLab source code and original documentation, Copyright (c) 2026 Matthew
  Selvam, on the stated assumption of authority to license that material. The
  root `LICENSE`, README and `LICENSING.md` record that scope; third-party content
  and data remain subject to their own terms. The final allowlist contains **62
  files**, adding only `LICENSE` to the previous 61-file snapshot.
- **Initial snapshot:** owner must review the staged allowlist/diff and then
  authorise the initial commit. A remote/publication destination is a later,
  separate decision. No remote is required for the local reproduction checks.
- **Market access:** the public repository cannot supply restricted inputs.
  This optional study remains partially reproducible from public files until
  each researcher obtains eligible manual source data. Broader distribution or
  commercial use needs its own permissions; a code licence does not confer them.

The core documented workflow has now been reproduced in isolation. The public
release remains pending owner review/commit; the optional odds
workflow retains its explicit manual-access limitation. Research limitations
and the negative calibration result remain as stated in the research report.
