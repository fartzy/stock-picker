"""Historically-conditioned bucketed-average features: "other times this ticker had a
similar (overnight gap, prior-day return) setup, how much did it move that day on
average?" -- the same mechanism as calendar.py's day_of_week_seasonality (expanding mean
of daily return, grouped by a bucket key), except the bucket key here is a discretized
pair of continuous-valued setup signals instead of a simple day-of-week integer.

Design notes (first pass, see README Roadmap for status):

- Bucketing: each axis (gap, prior-day return) is split into 3 buckets -- down / flat /
  up -- using a fixed +/-0.5% dead-zone threshold (GAP_FLAT_THRESHOLD /
  PRIOR_RETURN_FLAT_THRESHOLD) rather than per-ticker quantiles. Fixed thresholds are
  simpler and don't need a warm-up period to calibrate, at the cost of not adapting to a
  given ticker's typical volatility (a 0.5% gap is a big deal for a low-vol utility
  stock, background noise for a high-beta momentum name). 3x3 = 9 combined buckets keeps
  per-bucket sample counts workable; a finer grid (e.g. 5x5) would sharpen the
  conditioning but fragments an already-scarce history even further -- see the
  per-ticker sparsity note below.
- Per-ticker vs. pooled-across-universe: this module's primary function,
  `setup_seasonality`, is per-ticker only -- it mirrors day_of_week_seasonality exactly
  (single history in, expanding mean grouped by bucket, no cross-ticker inputs needed) so
  it's always available and trivially safe to reason about. The tradeoff Mike's prompt
  flagged is real: a new listing with 60 days of history has very few prior occurrences
  in any given one of 9 buckets, so early rows are mostly NaN and even mature tickers
  will have noisy averages in the rarer buckets (big-gap-down after a big-down-day is a
  tail combination). A pooled-across-universe variant (bucket keys shared, historical
  average computed over every tracked ticker's prior occurrences, not just this one's)
  would have much richer density sooner and is the natural next step -- see
  `pooled_setup_seasonality`, wired the same optional way cross_sectional.py's peer/
  benchmark inputs are (own to the caller to assemble the pooled input; the pooled
  average itself is a pure function of it). Both variants are lookahead-safe by the same
  discipline as day_of_week_seasonality: expanding().mean() is inclusive of the current
  row's own same-day return, so on its own this is NOT safe to use as a same-day
  feature -- it only becomes safe via training/dataset.py's blanket features.shift(1),
  exactly like day_of_week_seasonality. Do not special-case this column into
  dataset.GAP_COLUMN.
"""

from __future__ import annotations

import pandas as pd

GAP_FLAT_THRESHOLD = 0.005
PRIOR_RETURN_FLAT_THRESHOLD = 0.005

_DOWN, _FLAT, _UP = -1, 0, 1


def bucket_signal(signal: pd.Series, flat_threshold: float) -> pd.Series:
    """Discretize a continuous-valued return-like signal into -1 (down), 0 (flat),
    +1 (up) using a fixed dead-zone threshold around zero."""
    bucket = pd.Series(_FLAT, index=signal.index, dtype="float64")
    bucket[signal > flat_threshold] = _UP
    bucket[signal < -flat_threshold] = _DOWN
    bucket[signal.isna()] = pd.NA
    return bucket


def setup_bucket(gap: pd.Series, prior_return: pd.Series) -> pd.Series:
    """Combine the gap bucket and prior-day-return bucket into a single categorical
    key, e.g. "gap_up_prior_down" -- one of 9 combinations (3 gap buckets x 3
    prior-return buckets). NaN (missing gap or prior-return, e.g. the first row of
    history) propagates to NaN so it's excluded from the groupby average rather than
    silently landing in a spurious "flat_flat" bucket.
    """
    gap_bucket = bucket_signal(gap, GAP_FLAT_THRESHOLD)
    prior_bucket = bucket_signal(prior_return, PRIOR_RETURN_FLAT_THRESHOLD)
    labels = {_DOWN: "down", _FLAT: "flat", _UP: "up"}

    key = pd.Series(pd.NA, index=gap.index, dtype="object")
    valid = gap_bucket.notna() & prior_bucket.notna()
    key[valid] = [
        f"gap_{labels[g]}_prior_{labels[p]}"
        for g, p in zip(gap_bucket[valid], prior_bucket[valid])
    ]
    return key


def setup_seasonality(history: pd.DataFrame) -> pd.Series:
    """Expanding mean same-day return for every prior (and current) occurrence of
    today's (gap, prior-day-return) setup bucket -- "other times this ticker opened
    this much higher/lower after a day that was up/down/flat this much, how much did
    it move that day, on average?" Per-ticker only; see module docstring for the
    pooled-across-universe alternative and the sparsity tradeoff.
    """
    close = history["Close"]
    open_ = history["Open"]
    daily_return = close.pct_change()
    gap = (open_ - close.shift(1)) / close.shift(1)
    prior_return = daily_return.shift(1)

    bucket = setup_bucket(gap, prior_return)
    return daily_return.groupby(bucket).transform(lambda s: s.expanding().mean())


def pooled_setup_seasonality(
    daily_returns_by_ticker: dict[str, pd.Series], buckets_by_ticker: dict[str, pd.Series]
) -> dict[str, pd.Series]:
    """Same mechanism as setup_seasonality, but the expanding average for a given
    bucket is computed across every ticker's prior occurrences pooled together, not
    just the current ticker's own history -- much richer density for young tickers
    at the cost of mixing in other names' idiosyncratic behavior. Ticker -> its own
    setup_bucket() output must already be computed by the caller (see
    pipeline.build_features_for_universe for how the equivalent return_rank pooling
    is assembled) since bucket keys need each ticker's own gap/prior-return, but the
    averaging happens jointly across the whole pooled long-format table.

    Same-date rows across different tickers are treated as simultaneous, not
    ordered -- ticker X's value on date t only ever averages over (bucket-matching)
    rows with date STRICTLY BEFORE t, from any ticker including X itself. A naive
    "concat everything, sort by date, expanding().mean()" is NOT safe here: pandas'
    expanding() is inclusive of the row itself, so when two tickers share a date,
    whichever one sorts second within that date would (wrongly) see the other's
    same-day return baked into its own same-day value -- a real trap this
    implementation deliberately avoids by computing same-day pooled bucket sums
    once per date and taking a strictly-lagged cumulative sum/count, so every
    ticker's date-t value depends only on dates < t.

    Rough first pass, not yet wired into build_features_for_universe -- see README
    Roadmap.
    """
    long_frame = pd.concat(
        {
            ticker: pd.DataFrame(
                {"return": daily_returns_by_ticker[ticker], "bucket": buckets_by_ticker[ticker]}
            )
            for ticker in daily_returns_by_ticker
        },
        names=["ticker", "date"],
    )

    # One row per (date, bucket): that date's pooled sum/count across every ticker
    # occupying that bucket -- the atomic unit of "what happened on this date."
    per_date_bucket = (
        long_frame.dropna(subset=["bucket"])
        .groupby(["date", "bucket"])["return"]
        .agg(["sum", "count"])
    )

    # Cumulative sum/count up through (and including) each date, per bucket, then
    # shifted by one date-position so date t only sees dates strictly before it --
    # this is the piece that makes cross-ticker pooling lookahead-safe.
    cumulative = per_date_bucket.groupby(level="bucket").cumsum()
    prior_cumulative = cumulative.groupby(level="bucket").shift(1)
    prior_average = (prior_cumulative["sum"] / prior_cumulative["count"]).rename("seasonality")

    result = {}
    for ticker in daily_returns_by_ticker:
        bucket = buckets_by_ticker[ticker]
        lookup_key = pd.MultiIndex.from_arrays([bucket.index, bucket], names=["date", "bucket"])
        result[ticker] = prior_average.reindex(lookup_key).set_axis(bucket.index)
    return result


def build_conditional_seasonality_features(history: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {"setup_seasonality": setup_seasonality(history)},
        index=history.index,
    )
