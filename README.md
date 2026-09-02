# stock-picker

Builds a universe of the top 500 US companies by market cap and pulls daily
OHLCV price history for each via `yfinance`.

## Structure

```
python/
└── stock_picker/
    ├── tickers/     # top-500-by-market-cap universe + manual_additions.py
    ├── ingestion/   # yfinance pull -> raw OHLCV
    ├── storage/     # Parquet persistence (Repository pattern):
    │                 #   PriceStore, UniverseStore, FeatureStore
    └── features/    # ~80-column feature pipeline (the "F" in FTI):
                      #   momentum, volatility, trend, oscillators, volume,
                      #   candle/gap shape, distributional, cross-sectional,
                      #   calendar -- see pipeline.py for the orchestrator
```

## Build / test / run

```
bazel build //...
bazel test //...
bazel run //python/stock_picker/ingestion:main
bazel run //python/stock_picker/features:main
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

## Roadmap

- Persist sector labels to unlock `sector_relative_return`.
- Baseline factor-ranking model (momentum + vol-adjusted return), then
  gradient-boosted trees (LightGBM) with walk-forward validation.
- MLflow for experiment tracking once model training starts.
