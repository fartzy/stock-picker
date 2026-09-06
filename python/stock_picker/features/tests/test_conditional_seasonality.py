import pandas as pd
import pytest

from stock_picker.features.conditional_seasonality import (
    bucket_signal,
    build_conditional_seasonality_features,
    pooled_setup_seasonality,
    setup_bucket,
    setup_seasonality,
    ticker_setup_bucket,
)
from stock_picker.features.tests.fixtures import synthetic_history


def test_bucket_signal_splits_down_flat_up_with_dead_zone():
    signal = pd.Series([0.01, -0.01, 0.001, -0.001, 0.0, float("nan")])

    bucket = bucket_signal(signal, flat_threshold=0.005)

    assert bucket.iloc[0] == 1  # up
    assert bucket.iloc[1] == -1  # down
    assert bucket.iloc[2] == 0  # flat (inside dead zone)
    assert bucket.iloc[3] == 0  # flat (inside dead zone)
    assert bucket.iloc[4] == 0  # flat (exactly zero)
    assert pd.isna(bucket.iloc[5])


def test_setup_bucket_combines_gap_and_prior_return_and_propagates_nan():
    gap = pd.Series([0.01, -0.01, float("nan")])
    prior_return = pd.Series([0.01, 0.0, 0.01])

    bucket = setup_bucket(gap, prior_return)

    assert bucket.iloc[0] == "gap_up_prior_up"
    assert bucket.iloc[1] == "gap_down_prior_flat"
    assert pd.isna(bucket.iloc[2])


def test_build_conditional_seasonality_features_has_setup_seasonality_column():
    history = synthetic_history(n=60)

    features = build_conditional_seasonality_features(history)

    assert list(features.columns) == ["setup_seasonality"]
    # first two rows have no gap/prior-return (need two prior closes), rest fill in
    assert features["setup_seasonality"].iloc[:2].isna().all()
    assert features["setup_seasonality"].iloc[2:].notna().any()


def test_build_conditional_seasonality_features_includes_pooled_column_when_supplied():
    history = synthetic_history(n=60)
    dummy_pooled = history["Close"].pct_change()

    features = build_conditional_seasonality_features(history, pooled_seasonality=dummy_pooled)

    assert list(features.columns) == ["setup_seasonality", "pooled_setup_seasonality"]
    pd.testing.assert_series_equal(
        features["pooled_setup_seasonality"], dummy_pooled, check_names=False
    )


def test_ticker_setup_bucket_matches_setup_bucket_on_the_same_gap_and_prior_return():
    history = synthetic_history(n=60)
    close = history["Close"]
    gap = (history["Open"] - close.shift(1)) / close.shift(1)
    prior_return = close.pct_change().shift(1)

    bucket = ticker_setup_bucket(history)

    pd.testing.assert_series_equal(bucket, setup_bucket(gap, prior_return), check_names=False)


def test_setup_seasonality_matches_hand_computed_expanding_mean_within_a_bucket():
    # Construct a history where rows 2 and 4 land in the identical setup bucket
    # (same gap direction, same prior-day-return direction) so we can hand-verify
    # the expanding mean picks up exactly those occurrences.
    dates = pd.bdate_range("2026-01-02", periods=6)
    opens = [100.0, 100.0, 102.0, 100.0, 104.0, 100.0]
    closes = [100.0, 101.0, 100.0, 103.0, 100.0, 106.0]
    history = pd.DataFrame({"Open": opens, "Close": closes}, index=dates)

    seasonality = setup_seasonality(history)

    daily_return = history["Close"].pct_change()
    gap = (history["Open"] - history["Close"].shift(1)) / history["Close"].shift(1)
    prior_return = daily_return.shift(1)

    # row index 2 (2026-01-06): gap = (102-101)/101 > 0 (up), prior_return = row1's
    # return = (101-100)/100 > 0 (up) -> bucket "gap_up_prior_up", first occurrence,
    # so its own expanding mean is just its own same-day return.
    assert gap.iloc[2] > 0 and prior_return.iloc[2] > 0
    assert seasonality.iloc[2] == pytest.approx(daily_return.iloc[2])

    # row index 4 (2026-01-08): gap = (104-103)/103 > 0 (up), prior_return = row3's
    # return = (103-100)/100 > 0 (up) -> same bucket as row 2 -> expanding mean of
    # both occurrences' same-day returns.
    assert gap.iloc[4] > 0 and prior_return.iloc[4] > 0
    expected = (daily_return.iloc[2] + daily_return.iloc[4]) / 2
    assert seasonality.iloc[4] == pytest.approx(expected)


def test_setup_seasonality_is_not_affected_by_changing_a_later_row():
    """Lookahead-safety proof: mutating a future row's price must not change an
    earlier row's computed setup_seasonality value."""
    history = synthetic_history(n=40)
    before = setup_seasonality(history)

    mutated = history.copy()
    mutated.loc[mutated.index[-1], "Close"] *= 5.0
    mutated.loc[mutated.index[-1], "Open"] *= 5.0
    after = setup_seasonality(mutated)

    early = before.iloc[: len(before) - 5]
    early_after = after.iloc[: len(after) - 5]
    pd.testing.assert_series_equal(early, early_after)


def test_pooled_setup_seasonality_shares_the_average_across_tickers_in_the_same_bucket():
    dates = pd.bdate_range("2026-01-02", periods=4)
    bucket = pd.Series(["x", "x", "x", "x"], index=dates)
    returns_a = pd.Series([0.01, 0.02, 0.03, 0.04], index=dates)
    returns_b = pd.Series([0.10, 0.20, 0.30, 0.40], index=dates)

    result = pooled_setup_seasonality(
        {"AAA": returns_a, "BBB": returns_b}, {"AAA": bucket, "BBB": bucket}
    )

    # first date: no prior dates exist at all yet for either ticker
    assert pd.isna(result["AAA"].iloc[0])
    assert pd.isna(result["BBB"].iloc[0])

    # second date: pooled average of both tickers' first-date returns
    expected_second = (0.01 + 0.10) / 2
    assert result["AAA"].iloc[1] == pytest.approx(expected_second)
    assert result["BBB"].iloc[1] == pytest.approx(expected_second)

    # both tickers see the identical pooled value on a shared date/bucket
    pd.testing.assert_series_equal(result["AAA"], result["BBB"], check_names=False)


def test_pooled_setup_seasonality_excludes_same_date_cross_ticker_return():
    """The trap this implementation guards against: a naive concat-and-sort
    expanding mean would let one ticker's same-day return leak into another
    ticker's same-day value. Prove that doesn't happen."""
    dates = pd.bdate_range("2026-01-02", periods=2)
    bucket = pd.Series(["x", "x"], index=dates)
    returns_a = pd.Series([0.01, 0.02], index=dates)
    returns_b = pd.Series([100.0, 200.0], index=dates)  # extreme, would be obvious if leaked

    result = pooled_setup_seasonality(
        {"AAA": returns_a, "BBB": returns_b}, {"AAA": bucket, "BBB": bucket}
    )

    # AAA's first-date value must not be influenced by BBB's first-date 100.0 return
    assert pd.isna(result["AAA"].iloc[0])
    assert pd.isna(result["BBB"].iloc[0])


def test_pooled_setup_seasonality_not_affected_by_changing_a_later_row():
    dates = pd.bdate_range("2026-01-02", periods=5)
    bucket_a = pd.Series(["x", "x", "y", "x", "x"], index=dates)
    bucket_b = pd.Series(["x", "y", "x", "x", "x"], index=dates)
    returns_a = pd.Series([0.01, 0.02, 0.03, 0.04, 0.05], index=dates)
    returns_b = pd.Series([0.10, 0.20, 0.30, 0.40, 0.50], index=dates)

    before = pooled_setup_seasonality(
        {"AAA": returns_a, "BBB": returns_b}, {"AAA": bucket_a, "BBB": bucket_b}
    )["AAA"].iloc[:3]

    mutated_returns_a = returns_a.copy()
    mutated_returns_a.iloc[-1] = 999.0
    mutated_bucket_a = bucket_a.copy()
    mutated_bucket_a.iloc[-1] = "y"
    after = pooled_setup_seasonality(
        {"AAA": mutated_returns_a, "BBB": returns_b}, {"AAA": mutated_bucket_a, "BBB": bucket_b}
    )["AAA"].iloc[:3]

    pd.testing.assert_series_equal(before, after)
