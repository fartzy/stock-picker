# Architecture Decision Records

One short record per decision that shaped this codebase: what was decided,
why, and what it costs. Records are append-only -- when a decision is
reversed, the old record's Status points at the new one instead of being
rewritten.

Records 0001-0017 were backfilled on 2026-09-26 by synthesizing the commit
history, README, and CLAUDE.md. Dates are the relevant commits' dates;
context and rejected alternatives are reconstructed after the fact, so treat
them as approximate. Records from 0018 on are written at decision time.

| # | Date | Decision |
|---|---|---|
| [0001](0001-bazel-monorepo.md) | 2026-09-02 | Bazel monorepo, pinned via bazelisk, gazelle-generated BUILD files |
| [0002](0002-layered-architecture.md) | 2026-09-02 | Five layers, each talking only to the one below |
| [0003](0003-open-close-label-lookahead-safety.md) | 2026-09-02 | Same-day open→close label; shift close-known features, never open-known |
| [0004](0004-walk-forward-and-ticker-holdout.md) | 2026-09-03 | Walk-forward validation plus held-out tickers; never k-fold |
| [0005](0005-repository-stores.md) | 2026-09-04 | Repository-pattern stores: parquet for series, SQLite for event logs |
| [0006](0006-registry-as-metadata-only.md) | 2026-09-04 | Feast-style registry is metadata over the pipeline, not a second backend |
| [0007](0007-feature-transparency-views.md) | 2026-09-04 | Every feature ships prose, formula, and worked example, completeness-tested |
| [0008](0008-empirical-ensemble-membership.md) | 2026-09-06 | Model families earn ensemble weight empirically; solos always searched |
| [0009](0009-seeded-reproducible-training.md) | 2026-09-07 | Every trainer is seeded; run-to-run variance is a bug, not noise |
| [0010](0010-data-tracked-in-git.md) | 2026-09-07 | All of data/ is committed; a fresh clone is a working, trained app |
| [0011](0011-confidence-gated-buy-list.md) | 2026-09-07 | Fit trades only above a 0.5% predicted-return confidence gate |
| [0012](0012-top-2000-universe.md) | 2026-09-08 | Universe = top ~2000 US names by market cap |
| [0013](0013-archived-runs-live-model-pick.md) | 2026-09-08 | Every training run's model is archived; the live model is a user pick |
| [0014](0014-scheduled-jobs.md) | 2026-09-11 | Nightly 3:30 PM CT retrain; morning scoring 8:30:15 CT, click takes priority |
| [0015](0015-rank-as-parallel-pickle.md) | 2026-09-16 | Rank is its own top-K pickle, never blended with Fit |
| [0016](0016-news-skip-rules-and-local-tracing.md) | 2026-09-20 | News judged locally (Langfuse-traced); skip rules are conditional on the gap |
| [0017](0017-quote-provider-chain.md) | 2026-09-23 | Morning quotes: Polygon, then Yahoo, then Finnhub; blacklist store |
| [0018](0018-listfold-rank-objective.md) | 2026-09-26 | Rank objective is ListFold (custom listwise loss), replacing lambdarank |
| [0019](0019-consensus-gate-proposed.md) | 2026-09-26 | Proposed: consensus gate (Fit 0.5% ∩ Rank top-20) as a stricter buy list |
