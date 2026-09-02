# stock-picker

Builds a universe of the top 500 US companies by market cap and pulls daily
OHLCV price history for each via `yfinance`.

## Structure

```
python/
└── stock_picker/
    ├── tickers/     # top-500-by-market-cap universe + manual_additions.py
    ├── ingestion/   # yfinance pull -> raw OHLCV
    ├── storage/     # Parquet/LightGBM persistence (Repository pattern):
    │                 #   PriceStore, UniverseStore, FeatureStore, ModelStore
    ├── features/    # ~80-column feature pipeline (the "F" in FTI):
    │                 #   momentum, volatility, trend, oscillators, volume,
    │                 #   candle/gap shape, distributional, cross-sectional,
    │                 #   calendar -- see pipeline.py for the orchestrator
    └── training/    # the "T"/"I" in FTI: LightGBM on the day-session
                      # (open->close) return, walk-forward validated,
                      # tracked in MLflow -- see dataset.py for the
                      # lookahead-bias fix, the single most important
                      # correctness property of this module
```

## Build / test / run

```
bazel build //...
bazel test //...
bazel run //python/stock_picker/ingestion:main
bazel run //python/stock_picker/features:main
bazel run //python/stock_picker/training:main
```

Price history is written to `data/prices/<TICKER>.parquet` (gitignored).
Computed features are written to `data/features/<TICKER>.parquet` (gitignored),
one row per date, ~80 columns (see `python/stock_picker/features/pipeline.py`).
Features needing more trailing history than is available (e.g. a 120-day
return early in the series) are left `NaN`, not dropped or filled --
trimming/imputation is a training-time decision.

The cross-sectional `sector_relative_return` feature is defined but currently
always omitted: it needs a per-ticker sector label we don't persist yet
(would mean adding a `sector` column to `UniverseStore`, populated via one
`yfinance` `.info` call per newly-added ticker) -- a follow-up, not required
for `pipeline.py` itself.

Universe membership is tracked over time in `data/universe/registry.parquet`
(gitignored) via `UniverseStore`. Tracking is monotonic: the top-500 cutoff
only governs which new tickers get added on a given sync -- once a ticker
has ever qualified, it stays tracked even if it later falls out of the top
500. Add tickers outside the market-cap ranking via `MANUAL_ADDITIONS` in
`python/stock_picker/tickers/manual_additions.py`.

## Training

The model predicts the day-session (open->close) return -- buy at today's open, sell
at today's close. Trained pooled across all tracked tickers (no per-ticker models, no
ticker-identity feature, to avoid memorizing per-ticker quirks with so little data),
validated with date-based walk-forward splits (never a random shuffle -- see
`training/splits.py`).

Read `training/dataset.py` before touching anything else in this module: it fixes the
lookahead bias that would otherwise leak day t's own close into the features used to
predict day t's return. `training/inference.py` reuses that exact same row-construction
logic for live "this morning" scoring, so training and serving can't drift apart.

Trained models persist via `ModelStore` (`data/models/<name>.txt`, LightGBM's native
format) -- that's what `inference.py` reads from. MLflow (`data/mlruns/mlflow.db`,
local SQLite backend, no server) is experiment tracking only, not the model registry.

With only 3 tickers and 128 days of history, walk-forward directional accuracy is
currently noise-level (~45-65%) -- expected at this data scale. This milestone is
about the pipeline being mechanically correct end-to-end, not about having a model
with real signal yet.

## Roadmap

- Expand ingestion to the real top-500 universe -- current results are a 3-ticker,
  6-month pipeline smoke test, not a meaningful backtest.
- Persist sector labels to unlock `sector_relative_return`.
- Feature selection / importance analysis once there's enough data for it to be
  meaningful (currently ~79 features over ~380 pooled rows).
