"""Shared history horizons: longer feature warmup, unchanged training span."""

# Keep the existing Yahoo OHLCV adjustment convention; never splice the
# separate Massive research cache into the ordinary price store.
PRICE_HISTORY_PERIOD = "2y"
FEATURE_SNAPSHOT_YEARS = 1
