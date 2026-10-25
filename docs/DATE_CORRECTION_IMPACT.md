# Impact of the 2019–20 date correction

The pre-correction evaluation used canonical CSV SHA-256
`4104b87dde6fcc16221dc0eaddc26ebbc3e3fa8e048d6d156415f5435c18282c`.
The corrected evaluation uses
`db975a7f31eafe2848b5e7e8a56437d47ce6d5aa260462c52ee21964a5e1ccb9`.
Both runs used the same baseline and Elo settings and the same chronological
development, validation and test seasons. Each model scored the same 380
fixtures per evaluation season. The pre-correction metrics and predictions
were copied to `data/processed/predictions/pre_correction/` before regeneration.

| Split | Model | Brier before | Brier after | Log loss before | Log loss after | Accuracy before | Accuracy after |
|---|---|---:|---:|---:|---:|---:|---:|
| Validation | Baseline | 0.637105 | 0.637105 | 1.054054 | 1.054054 | 0.4605 | 0.4605 |
| Validation | Elo | 0.632508 | 0.632454 | 1.045379 | 1.045291 | 0.4421 | 0.4421 |
| Test | Baseline | 0.655589 | 0.655589 | 1.080894 | 1.080894 | 0.4079 | 0.4079 |
| Test | Elo | 0.648321 | 0.648356 | 1.068119 | 1.068165 | 0.3974 | 0.3974 |

All 760 baseline probability vectors are identical across the two runs.
All 760 Elo vectors changed because moving the 66 2019–20 results to their
correct dates changes the sequence of rating updates and thus later ratings.
The largest absolute change to one Elo probability was about 0.00365.
Validation Elo Brier and log loss decreased by about 0.000054 and 0.000088;
test Elo Brier and log loss increased by about 0.000035 and 0.000047.
Accuracy was unchanged. These are data-correction effects, not model tuning.
Earlier test results still update ratings only for later test dates.

The raw source remains at SHA-256
`9e1695f85ce5553f762e809dea357ec0a95838939be89569060a83c4e98a302d`.
See `data/processed/date_corrections.json` for all 66 fixture-level changes
and the [provider issue](https://github.com/openfootball/football.json/issues/54)
for the source-date evidence.
