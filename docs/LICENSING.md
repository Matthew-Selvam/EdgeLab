# Code licence and data-use boundaries

At the owner's instruction, **original EdgeLab source code and original
documentation are licensed under the [MIT License](../LICENSE)**, with the notice:

> Copyright (c) 2026 Matthew Selvam

The root `LICENSE` contains the standard MIT terms. The grant covers the original
project code (including original tests, scripts and stylesheet) and original
authored documentation. It relies on the owner's stated assumption of authority
to license that original material; this review does not independently establish
ownership of every contribution.

MIT permits reuse of the covered material, subject to preservation of its
copyright and licence notice, and includes a warranty disclaimer. See the
[Open Source Initiative's MIT text](https://opensource.org/license/mit).
**It does not license third-party content or datasets, or grant rights in
excluded raw data or data-derived outputs.** Original explanatory prose and
analysis in the research documentation are covered; underlying third-party
material retains its own terms and attribution.

## Separate source and output rights

- **Original EdgeLab code and documentation:** governed by the root MIT licence.
  Third-party quotations, source-format examples and other third-party material
  are excluded from that grant. Preserve their attribution and applicable terms.
- **OpenFootball results:** the [upstream project](https://github.com/openfootball/football.json#license)
  dedicates its schema, data and scripts to the public domain under CC0. The
  pinned source URLs and hashes are in `OPENFOOTBALL_SOURCES.json`. EdgeLab keeps
  these downloaded inputs and generated datasets outside Git as a release policy.
- **Football-Data.co.uk / Pinnacle odds:** subject to the source's
  [fair-use policy](https://football-data.co.uk/contact.php), checked during this
  hardening audit. It permits private personal/academic use with manual or light
  downloads and restricts heavy repeated automated and commercial uses. It does
  not supply a general redistribution licence for the odds. No code licence on
  EdgeLab or another pipeline overrides those conditions.
- **Derived outputs:** rights depend on the source and use. Raw odds and
  fixture-level market probabilities, raw implied probabilities and overround
  remain local and ignored. Only aggregate research findings with attribution
  appear in public documentation. Do not infer permission to distribute richer
  derived datasets or commercialise a service; obtain appropriate permission
  before changing this boundary. Dependencies retain their own licences;
  installing them does not relicense them under EdgeLab's MIT terms. No vendored
  dependency code is included in the staged snapshot.

The core reproduction workflow uses public-domain results. The optional market
workflow depends on the researcher's eligibility and manual source acquisition.
See `REPRODUCIBILITY.md` and `ODDS_SOURCE_AUDIT.md`.
