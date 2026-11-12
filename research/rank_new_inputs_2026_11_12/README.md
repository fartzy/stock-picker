# Frozen Rank input experiment

This research run compares the current 151 active raw inputs with the same inputs
plus 15 prior-pattern evidence statistics, four Kronos forecasts, and all seven
chronological SVM outputs (177 head inputs; 170 raw source-SVM inputs). It retains
all 95 existing exclusions, including `dip_recovery_fraction` and
`dip_volume_change_1d`. No ticker receives a special exclusion. Assumed trading
costs are zero. This is a retrospective experiment, not a production promotion.

The runner is `python/stock_picker/training/rank_input_experiment.py`. Before any
fit, it freezes the active 2,000-stock universe, deterministic 200-stock holdout,
exclusions, feature names, source-file hashes, package versions, historical prices,
and a raw matrix. Data and labels stop on November 11, 2026. Later source refreshes
cannot change these snapshots. The protocol is saved in the ignored runtime
directory; a compact public copy records its hash. All artifacts and source hashes
must match on restart. The runner does not write to production feature, model,
training-run, or configuration stores.

The 15 pattern inputs are `seq{2,3,4}_open3_{count,std,age,shrunk_mean,recent_mean}`.
They use the existing `build_pattern_evidence` builder and join on the decision
date after the ordinary lag-safe training frame is built. Kronos also joins on the
decision date, using exact forecast request hashes from the frozen price history.
Missing, invalid, incompatible, or unmatched forecasts remain NaN; their stocks
remain eligible. The exact Kronos execution contract stays with every frame and
model that consumes these inputs.

Each stage uses four chronological outer folds and the existing three-block OOF
seed. The candidate SVMs consume all 170 unpruned raw inputs and are fitted once
per seed/outer block. The baseline head uses exactly `fold.baseline_train` rows;
the candidate uses the matching `fold.stacked_train` rows. Both heads use unchanged
production Rank/ListFold parameters. Each candidate archive includes its seven
paired source estimators. Models, matrices, and OOF score rows stay under
`.runtime/research/rank_new_inputs_2026_11_12/`.

Evaluation reports separate (a) chronological OOF test rows for training tickers
and (b) the 200 held-out tickers on those same test dates, scored by the corresponding
earlier-date fold models. Within either cohort, baseline and candidate evaluate
the identical stock/date rows. The final archived heads use the last fold's OOF
training and test rows together. One additional seven-SVM fit uses the full
training-ticker panel for the final candidate archive. Final refits are not scored
against their own fitting dates and presented as out-of-time results.

The primary metric is equal-weight daily top-one gross open-to-close return.
Reports also include top-five/top-twenty return and hit rate, daily Rank IC,
sample/date coverage, and paired daily differences with a 2,000-draw ISO-week
cluster bootstrap (seed 42). The uncertainty intervals are exploratory. Gains for
each of the 26 additions are descriptive split gains, not independent feature
contributions. This design tests the combined candidate and makes no family
ablation claim. Prior use of these dates also limits fresh-discovery claims.

`pilot` freezes the contiguous completed Kronos prefix available when its stage
starts and labels every result provisional. It must have at least 20 sessions and
cannot bridge a completion hole. Its winter-only evaluation does not describe
full-year behavior. `full` requires the existing `ready_contract` gate for the
entire frozen universe through November 11, plus at least one exact-source valid
Kronos row per eligible panel date. The 251-session collection readiness window
can exceed the eligible matrix by one day because the existing lag-safe builder
drops the first feature row. Reports disclose that date separately; no row is
fabricated. Per-date and per-ticker CSVs report finite-K coverage.

Preparation and launch (from the repository root):

```bash
bazelisk test //python/stock_picker/training/tests:test_rank_input_experiment
bazelisk build //python/stock_picker/training:rank_input_experiment
bazel-bin/python/stock_picker/training/rank_input_experiment --mode prepare
nohup /usr/bin/caffeinate -i bazel-bin/python/stock_picker/training/rank_input_experiment \
  --mode pilot-then-wait-full \
  > .runtime/research/rank_new_inputs_2026_11_12/process.log 2>&1 < /dev/null &
```

The durable process writes `state.json` and logs each preparation/fold/stage
boundary. Between stages it polls full readiness every 60 seconds, without
touching the existing backfill. It fails on an exited backfill, unresolved worker
errors, changed contract, changed source code, or changed snapshots. The
backfill's process start time and command identify its PID; zombie/reused PIDs
fail closed. Waiting also stops at its configured six-hour limit plus five minutes.
Full seven-SVM training previously took about 2h12m; this is not a quick smoke
test. There are eight source-fitting passes per stage: three seed, four outer,
one final (seven outputs in each pass).

Completed stages resume only after artifact/hash validation. A partial training
stage is retained and fails closed on restart, rather than silently overwriting
expensive intermediate work. Inspect `state.json`, logs, fold reports and model
archives before using both a fresh `--runtime` directory and a fresh `--reports`
directory for a retry. A different frozen run cannot overwrite an existing report
directory. An exclusive lock
prevents duplicate runs in the same directory. No experiment command promotes a
model or registers a production training run.
