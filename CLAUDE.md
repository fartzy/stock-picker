# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A stock-picker web app: ~500-ticker universe, ~100 engineered features across
11 categories, a composable LightGBM/RandomForest/NeuralNet/Ridge ensemble
predicting day-session (open->close) return, a live confidence-gated "buy
signal" feature, and a real trading log with live P&L. FastAPI backend +
React/Vite/TS frontend, both Bazel-integrated.

## Commands

Bazel version is pinned in `.bazelversion` (9.0.1) -- install
[bazelisk](https://github.com/bazelbuild/bazelisk) (`brew install bazelisk`)
and invoke it as `bazelisk`, not plain `bazel`.

```
bazelisk build //...
bazelisk test //...                                          # whole stack, Python + TS
bazelisk test //python/stock_picker/training/tests:test_model # a single test target
bazelisk run //python/stock_picker/api:main                  # FastAPI backend on :8000
bazelisk run //typescript:dev                                 # Vite dev server on :5173, proxies /api -> :8000
bazelisk run //python/stock_picker/ingestion:main
bazelisk run //python/stock_picker/features:main
bazelisk run //python/stock_picker/features:catalog           # every feature: description, formula, example, coverage
bazelisk run //python/stock_picker/training:main               # CLI training entrypoint (same path the UI's "Run training" button uses)
bazelisk run //python/stock_picker/training:tune_experiment    # research script: pruning + hyperparameter + ensemble-weight search
bazelisk run //python/stock_picker/features:log_trade -- --ticker AAPL --side buy --shares 10 --price 230.00
bazelisk run //:gazelle                                        # regenerate BUILD.bazel srcs/deps after adding/removing an import
```

**The API server has no auto-reload.** After changing any Python file under
`python/stock_picker/`, kill and re-run `//python/stock_picker/api:main` for
the change to take effect -- the frontend dev server (Vite) does hot-reload
on its own.

**Gazelle py_binary/py_library ambiguity**: a new `py_binary` inside a package
that already has a same-named-module `py_binary` (e.g. `training/main.py` is
both a source in the `:training` `py_library` and its own `:main` `py_binary`)
will fail to resolve if it imports from that module. Simplest fix used
throughout this codebase: don't import the constant, hardcode it locally with
a comment explaining why (see `training/tune_experiment.py`,
`training/recent_picks_demo.py`).

Price/feature/model data lives in `data/{prices,features,universe,models}/`
(gitignored) -- nothing in the app ships with pretrained data; run the
ingestion -> features -> training pipeline once to populate it.

## Architecture

Five layers, each only talking to the one below it:

```
ingestion/  -> storage/  -> features/  -> training/  -> api/ -> typescript/
```

- **`storage/`**: Repository pattern over Parquet/pickle files (`UniverseStore`,
  `PriceStore`, `FeatureStore`, `ModelStore`, `TradeStore`,
  `PrunedFeatureStore`, `TrainingRunStore`, `TrainingConfigStore`). Every
  store constructor takes an optional `data_dir` for test isolation
  (`Store(data_dir=tmp_path)`) -- this is the established DI pattern for
  testing anything that touches persistence.
- **`features/`**: the ~100-column pipeline (`features/pipeline.py`), plus
  three parallel pattern-matched, column-name-driven views that must all stay
  in sync for every real feature: `descriptions.py` (plain-English prose),
  `formulas.py` (the actual pandas expression), `examples.py` (a worked
  numeric example). Each has its own completeness test that fails if any real
  feature column falls through to an "unknown" placeholder. `registry.py`
  layers Feast-style metadata (Entity/FeatureView/FeatureService) over the
  same pipeline -- metadata only, not a second backend.
- **`training/`**: `model.py` has one trainer function per model family
  (`train_lightgbm`, `train_random_forest`, `train_neural_net`, `train_ridge`,
  plus the standalone `train_logistic_regression` diagnostic), all returning
  a common `TrainedModel` shape so `ensemble.py`'s `Ensemble` can blend any
  mix of them via weighted average. `PREDICTIVE_MODEL_TYPES` lists which
  types can be an ensemble member; `logistic_regression` is deliberately
  excluded because it predicts a binarized direction, not the continuous
  return the others do, so it's fit and persisted standalone as an
  importance lens instead. `dataset.py` builds the lookahead-safe label --
  read it before touching anything else in this module. Validation is
  date-based walk-forward (never k-fold -- k-fold's shuffling would leak
  future rows into past predictions) plus a held-out set of entire tickers
  never seen in training. `inference.py` builds one lookahead-safe row for
  live scoring from a ticker's last persisted feature snapshot + today's live
  open, raising rather than silently scoring on stale/implausible data.
- **`tune_experiment.py`** (a permanent research tool, not wired into any
  production path) is where a new model family's ensemble weight gets
  decided *empirically*: it searches every solo/pairwise/all-combo weight
  split, always including the pure solos, so the search can never recommend
  a blend worse than the best solo model already in production. This is how
  RandomForest and NeuralNet were both found to add zero value and excluded
  from `DEFAULT_MODEL_SPECS` -- decided by running the numbers, not assumed.
- **`api/routes.py`**: every endpoint wraps an already-tested pure function
  from `features/`/`training/`/`storage/` -- no new business logic lives
  here.
- **`typescript/`**: Trading / Feature Store / Models / Data tabs. Shared
  fetch helper `getJson<T>()` in `api.ts`; `useFetchData()` hook for
  on-mount-plus-optional-polling data loading (manual/click-triggered fetches
  just call the `api.ts` function directly instead).

## Conventions

- One feature branch + one PR per logical change, squash-merged; check
  `git rev-list --left-right --count main...origin/main` and pull before
  branching, since other sessions/worktrees may be working in this repo
  concurrently.
- Never guess at a library's current API -- this codebase's own patterns
  (Repository stores, pattern-matched description/formula/example modules,
  the empirical weight-search methodology) are the reference for "how things
  are done here."
