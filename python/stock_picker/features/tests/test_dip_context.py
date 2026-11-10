from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from stock_picker.features.catalog import experimental_features, list_feature_columns
from stock_picker.features.descriptions import describe_feature
from stock_picker.features.dip_context import (
    DIP_CLOSE_KNOWN_COLUMNS,
    DIP_COLUMNS,
    DIP_FAMILIES,
    DIP_FLAG_COLUMNS,
    DIP_OPEN_KNOWN_COLUMNS,
    build_dip_context_features,
    complete_dip_history,
    morning_dip_features,
    morning_dip_row,
)
from stock_picker.features.dip_context_metadata import DIP_FEATURE_METADATA
from stock_picker.features.examples import feature_example
from stock_picker.features.formulas import describe_computation


def history(n=300, flat=False):
    dates = pd.bdate_range("2024-01-02", periods=n)
    close = np.full(n, 110.0) if flat else 110 + np.arange(n) * .02 + np.sin(np.arange(n))
    return pd.DataFrame({
        "Open": close, "High": close + 1, "Low": close - 1, "Close": close,
        "Volume": np.full(n, 1e6) if flat else 1e6 + np.arange(n) * 1000,
    }, index=dates)


def selloff_history():
    bars = history()
    bars.iloc[-4] = [100, 101, 99, 100, 1e6]
    bars.iloc[-3] = [95, 96, 89, 90, 2e6]
    bars.iloc[-2] = [85.3, 86, 85, 85.5, 1.5e6]
    bars.iloc[-1] = [87, 88, 86, 87.5, 1e6]
    return bars


def test_selloff_and_pressure_measurements_match_hand_calculation():
    bars = selloff_history()
    row = morning_dip_features(bars).iloc[-1]
    assert row.dip_close_to_low_252d == pytest.approx(85.5 / 85 - 1)
    assert row.dip_open_to_low_252d == pytest.approx(87 / 85 - 1)
    assert row.dip_recovery_fraction == pytest.approx(1.5 / 14.5)
    assert row.dip_low_age_252d == 1
    assert row.dip_prior_new_low_252d == 1
    assert row.dip_deep_unrecovered == 1
    assert row.dip_pressure_easing == 1
    assert row.dip_range_change_1d == pytest.approx(5 / 11 - 1)
    assert row.dip_volume_change_1d == pytest.approx(-.25)
    prior_volumes = bars.Volume.iloc[-21:-1].to_numpy()
    expected_z = (prior_volumes[-1] - prior_volumes.mean()) / prior_volumes.std(ddof=1)
    assert row.dip_gap_previous_volume_zscore == pytest.approx((87 / 85.5 - 1) * expected_z)


def test_volatility_normalizer_ends_before_both_shock_days():
    bars = selloff_history()
    t = len(bars) - 1
    earlier = bars.Close.iloc[t - 23:t - 2].to_numpy()
    prior_returns = earlier[1:] / earlier[:-1] - 1
    assert len(prior_returns) == 20
    sigma = prior_returns.std(ddof=1)
    row = morning_dip_features(bars).iloc[t]
    assert row.dip_loss_2d_sigma == pytest.approx((85.5 / 100 - 1) / (sigma * np.sqrt(2)))
    assert row.dip_loss_change_sigma == pytest.approx((.05 - .10) / sigma)


def test_latest_tied_low_is_used_for_age():
    bars = selloff_history()
    bars.iloc[-6, bars.columns.get_loc("Low")] = 85
    assert morning_dip_features(bars).iloc[-1].dip_low_age_252d == 1
    bars.iloc[-2, bars.columns.get_loc("Low")] = 85.1
    assert morning_dip_features(bars).iloc[-1].dip_low_age_252d == 5


def test_full_252_completed_sessions_are_required():
    features = morning_dip_features(history())
    assert features.iloc[251].dip_low_history_valid == 0
    assert np.isnan(features.iloc[251].dip_open_to_low_252d)
    assert np.isnan(features.iloc[251].dip_prior_new_low_252d)
    assert features.iloc[252].dip_low_history_valid == 1
    assert np.isfinite(features.iloc[252].dip_open_to_low_252d)


@pytest.mark.parametrize("bad_value", [np.nan, 0, -1, np.inf])
def test_missing_or_invalid_prior_bar_does_not_shorten_low_window(bad_value):
    bars = selloff_history()
    bars.iloc[100, bars.columns.get_loc("Volume")] = bad_value
    row = morning_dip_features(bars).iloc[-1]
    assert row.dip_low_history_valid == 0
    assert np.isnan(row.dip_close_to_low_252d)
    assert np.isnan(row.dip_deep_unrecovered)


def test_invalid_ohlc_range_is_an_unknown_bar():
    bars = selloff_history()
    bars.iloc[-2, bars.columns.get_loc("High")] = 80
    row = morning_dip_features(bars).iloc[-1]
    assert row.dip_low_history_valid == 0
    assert np.isnan(row.dip_pressure_easing)
    assert np.isnan(row.dip_recovery_fraction)


def test_current_and_future_hlcv_cannot_change_current_decision():
    bars = selloff_history()
    t = 275
    expected = morning_dip_features(bars).iloc[t]
    changed = bars.copy()
    changed.loc[changed.index[t:], ["High", "Low", "Close", "Volume"]] = -999
    changed.iloc[t + 1:, changed.columns.get_loc("Open")] = np.nan
    pd.testing.assert_series_equal(morning_dip_features(changed).iloc[t], expected)


def test_today_open_changes_only_open_known_columns():
    bars = selloff_history()
    expected = morning_dip_features(bars).iloc[-1]
    bars.iloc[-1, bars.columns.get_loc("Open")] = 84
    actual = morning_dip_features(bars).iloc[-1]
    preopen = [c for c in DIP_COLUMNS if c not in DIP_OPEN_KNOWN_COLUMNS]
    pd.testing.assert_series_equal(actual[preopen], expected[preopen])
    assert actual.dip_open_below_low_252d == 1
    assert actual.dip_recovery_fraction < 0
    assert actual.dip_deep_unrecovered == 0


@pytest.mark.parametrize("timezone", [None, "America/New_York"])
def test_live_style_helper_matches_historical_decision_row(timezone):
    bars = selloff_history()
    if timezone:
        bars.index = bars.index.tz_localize(timezone)
    expected = morning_dip_features(bars).iloc[-1]
    actual = morning_dip_row(bars.iloc[:-1], bars.Open.iloc[-1], bars.index[-1].date())
    pd.testing.assert_series_equal(actual, expected)


def test_helper_rejects_current_or_future_history():
    bars = history()
    with pytest.raises(ValueError, match="strictly before"):
        morning_dip_row(bars, 100, bars.index[-1].date())
    with pytest.raises(ValueError, match="strictly before"):
        morning_dip_row(bars, 100, bars.index[-2].date())


@pytest.mark.parametrize("opening", [0, -5, np.nan, np.inf])
def test_helper_rejects_invalid_open(opening):
    with pytest.raises(ValueError, match="finite and positive"):
        morning_dip_row(history(), opening, date(2026, 1, 1))


def test_helper_allows_empty_history_but_reports_unknowns():
    empty = history().iloc[:0]
    row = morning_dip_row(empty, 100, date(2026, 1, 1))
    assert row.dip_low_history_valid == 0
    assert row.dip_recovery_valid == 0
    assert row[list(DIP_FLAG_COLUMNS)].isna().all()


@pytest.mark.parametrize("kind", ["duplicate", "descending", "intraday", "integer", "missing_column"])
def test_invalid_index_or_schema_fails_explicitly(kind):
    bars = history()
    if kind == "duplicate":
        bars = pd.concat([bars, bars.iloc[-1:]])
    elif kind == "descending":
        bars = bars.iloc[::-1]
    elif kind == "intraday":
        bars.index = bars.index + pd.Timedelta(hours=1)
    elif kind == "integer":
        bars = bars.reset_index(drop=True)
    else:
        bars = bars.drop(columns="Volume")
    with pytest.raises(ValueError):
        morning_dip_features(bars)


def test_no_loss_zero_variance_and_zero_range_stay_finite_or_unknown():
    bars = history(flat=True)
    bars.iloc[-2] = [110, 110, 110, 110, 1e6]
    row = morning_dip_features(bars).iloc[-1]
    assert row.dip_recovery_valid == 0
    assert np.isnan(row.dip_recovery_fraction)
    assert np.isnan(row.dip_loss_2d_sigma)
    assert np.isnan(row.dip_pressure_easing)
    assert not np.isinf(row.to_numpy()).any()


@pytest.mark.parametrize("outside", [False, True])
def test_inclusive_two_percent_drop_boundary(outside):
    bars = history()
    first_close = 3.43 + (3.5 * 1e-8 if outside else 0)
    bars.iloc[-4] = [3.5, 3.51, 3.49, 3.5, 1e6]
    bars.iloc[-3] = [first_close, 3.44, 3.42, first_close, 1e6]
    bars.iloc[-2] = [3.3614, 3.37, 3.33, 3.3614, 1e6]
    bars.iloc[-1] = [3.37, 3.38, 3.36, 3.37, 1e6]
    assert morning_dip_features(bars).iloc[-1].dip_deep_unrecovered == float(not outside)


@pytest.mark.parametrize("outside", [False, True])
def test_inclusive_two_percent_low_distance_boundary(outside):
    bars = history()
    close = 102 + (100 * 1e-8 if outside else 0)
    bars.iloc[-4] = [110, 111, 109, 110, 1e6]
    bars.iloc[-3] = [105, 106, 104, 105, 1e6]
    bars.iloc[-2] = [close, 103, 100, close, 1e6]
    bars.iloc[-1] = [103, 104, 102, 103, 1e6]
    assert morning_dip_features(bars).iloc[-1].dip_deep_unrecovered == float(not outside)


@pytest.mark.parametrize("outside", [False, True])
def test_inclusive_quarter_recovery_boundary(outside):
    bars = history()
    opening = 11.23 + (.92 * 1e-8 if outside else 0)
    bars.iloc[-4] = [11.92, 12, 11.9, 11.92, 1e6]
    bars.iloc[-3] = [11.5, 11.6, 11.4, 11.5, 1e6]
    bars.iloc[-2] = [11, 11.1, 10.9, 11, 1e6]
    bars.iloc[-1] = [opening, 11.4, 11.2, opening, 1e6]
    assert morning_dip_features(bars).iloc[-1].dip_deep_unrecovered == float(not outside)


@pytest.mark.parametrize("outside", [False, True])
def test_inclusive_half_range_close_boundary(outside):
    bars = history()
    close = 1.45 - (.3 * 1e-8 if outside else 0)
    bars.iloc[-4] = [1.6, 1.7, 1.5, 1.6, 1e6]
    bars.iloc[-3] = [1.5, 1.6, 1.4, 1.5, 2e6]
    bars.iloc[-2] = [1.5, 1.6, 1.3, close, 1e6]
    assert morning_dip_features(bars).iloc[-1].dip_pressure_easing == float(not outside)


@pytest.mark.parametrize("outside", [False, True])
def test_inclusive_trend_pullback_return_boundary(outside):
    bars = history()
    close = np.linspace(1, 3.55, len(bars))
    close[-4:] = [3.5, 3.47, 3.43 + (3.5 * 1e-8 if outside else 0), 3.45]
    bars.loc[:, "Open"] = close
    bars.loc[:, "Close"] = close
    bars.loc[:, "High"] = close + .01
    bars.loc[:, "Low"] = close - .01
    assert morning_dip_features(bars).iloc[-1].dip_trend_pullback == float(not outside)


def test_support_reference_excludes_yesterdays_failed_break():
    bars = history(flat=True)
    bars.iloc[-2] = [109, 111, 108, 110, 1e6]
    bars.iloc[-1] = [109.5, 111, 109, 110, 1e6]
    row = morning_dip_features(bars).iloc[-1]
    # The 20 earlier lows are109; yesterday's108 must not lower the reference.
    assert row.dip_support_break_depth_20d == pytest.approx(1 / 109)
    assert row.dip_support_close_distance_20d == pytest.approx(1 / 109)
    assert row.dip_support_open_distance_20d == pytest.approx(.5 / 109)
    assert row.dip_support_reclaim_20d == 1
    bars.iloc[-1, bars.columns.get_loc("Open")] = 108.5
    assert morning_dip_features(bars).iloc[-1].dip_support_reclaim_20d == 0


def test_pullback_flag_distinguishes_rising_trend_context():
    bars = history()
    close = np.linspace(100, 200, len(bars))
    close[-2] = close[-4] * .975
    bars.loc[:, "Open"] = close
    bars.loc[:, "Close"] = close
    bars.loc[:, "High"] = close + 1
    bars.loc[:, "Low"] = close - 1
    row = morning_dip_features(bars).iloc[-1]
    assert row.dip_trend_pullback == 1
    expected_slope = close[-51:-1].mean() / close[-61:-11].mean() - 1
    assert row.dip_sma50_slope_10d == pytest.approx(expected_slope)
    bars.iloc[-2] = [90, 91, 89, 90, 1e6]
    assert morning_dip_features(bars).iloc[-1].dip_trend_pullback == 0


def test_consistent_split_basis_rescaling_preserves_dimensionless_features():
    bars = selloff_history()
    adjusted = bars.copy()
    adjusted[["Open", "High", "Low", "Close"]] *= .5
    adjusted.Volume *= 2
    pd.testing.assert_frame_equal(morning_dip_features(adjusted), morning_dip_features(bars))


def test_input_is_not_mutated_and_all_columns_are_numeric():
    bars = selloff_history()
    original = bars.copy(deep=True)
    result = morning_dip_features(bars)
    pd.testing.assert_frame_equal(bars, original)
    assert tuple(result.columns) == DIP_COLUMNS
    assert len(DIP_COLUMNS) == len(set(DIP_COLUMNS)) == 21
    assert set(DIP_COLUMNS) == {c for family in DIP_FAMILIES.values() for c in family}
    assert all(dtype == np.dtype(float) for dtype in result.dtypes)


def test_dip_features_have_complete_ordinary_metadata():
    bars = history(n=140)
    assert list_feature_columns(bars)["dip_context"] == list(DIP_COLUMNS)
    assert not set(DIP_COLUMNS) & set(experimental_features())
    metadata = DIP_FEATURE_METADATA
    for name in DIP_COLUMNS:
        assert name in metadata
        assert describe_feature(name) == metadata[name].description
        assert describe_computation(name) == metadata[name].computation
        assert feature_example(name) == metadata[name].example
        assert "earlier" in metadata[name].computation.lower()


def test_snapshot_adapter_advances_only_close_known_values():
    bars = selloff_history()
    original = bars.copy(deep=True)
    snapshots = build_dip_context_features(bars)
    decisions = morning_dip_features(bars)
    close_columns = list(DIP_CLOSE_KNOWN_COLUMNS)
    open_columns = sorted(DIP_OPEN_KNOWN_COLUMNS)
    pd.testing.assert_frame_equal(snapshots[open_columns], decisions[open_columns])
    pd.testing.assert_frame_equal(
        snapshots[close_columns].iloc[:-1].reset_index(drop=True),
        decisions[close_columns].iloc[1:].reset_index(drop=True),
    )
    assert len(close_columns) == 13 and len(open_columns) == 8
    pd.testing.assert_frame_equal(bars, original)


def test_final_snapshot_needs_no_future_prices_and_ignores_every_future_field():
    bars = selloff_history()
    cutoff = 270
    expected = build_dip_context_features(bars.iloc[:cutoff]).iloc[-1]
    changed = bars.copy()
    changed.iloc[cutoff:] = [1e9, 1e10, .01, 1e8, 9e12]
    actual = build_dip_context_features(changed).iloc[cutoff - 1]
    pd.testing.assert_series_equal(actual, expected)
    # Changing the placeholder's Open also cannot change close-known values.
    next_morning = morning_dip_row(bars.iloc[:cutoff], .01, bars.index[cutoff].date())
    pd.testing.assert_series_equal(
        expected[list(DIP_CLOSE_KNOWN_COLUMNS)],
        next_morning[list(DIP_CLOSE_KNOWN_COLUMNS)], check_names=False,
    )


@pytest.mark.parametrize("end", ["2026-03-08", "2026-11-01"])
def test_snapshot_placeholder_keeps_midnight_across_dst(end):
    bars = history()
    bars.index = pd.date_range(end=end, periods=len(bars), tz="America/New_York")
    actual = build_dip_context_features(bars)
    assert actual.index.equals(bars.index)
    tomorrow = bars.index[-1].date() + timedelta(days=1)
    expected = morning_dip_row(bars, 100, tomorrow)
    pd.testing.assert_series_equal(
        actual.iloc[-1][list(DIP_CLOSE_KNOWN_COLUMNS)],
        expected[list(DIP_CLOSE_KNOWN_COLUMNS)], check_names=False,
    )


def test_snapshot_adapter_empty_history_preserves_schema():
    empty = history().iloc[:0]
    actual = build_dip_context_features(empty)
    assert actual.empty and tuple(actual.columns) == DIP_COLUMNS
    assert actual.index.equals(empty.index)


@pytest.mark.parametrize("invalid", ["duplicate", "descending", "intraday"])
def test_snapshot_adapter_rejects_invalid_index(invalid):
    bars = history()
    if invalid == "duplicate":
        bars.index = pd.DatetimeIndex([bars.index[1], *bars.index[1:]])
    elif invalid == "descending":
        bars = bars.iloc[::-1]
    else:
        bars.index += pd.Timedelta(hours=1)
    with pytest.raises(ValueError):
        build_dip_context_features(bars)


def test_scalar_bounded_history_matches_full_history_with_missing_bar():
    bars = history(n=1000)
    bars.iloc[-150] = np.nan
    expected = morning_dip_features(bars).iloc[-1]
    actual = morning_dip_row(bars.iloc[:-1], bars.Open.iloc[-1], bars.index[-1].date())
    pd.testing.assert_series_equal(actual, expected)


def test_scalar_validates_dates_outside_retained_tail():
    bars = history(n=1000)
    bars.index = pd.DatetimeIndex([bars.index[1], *bars.index[1:]])
    with pytest.raises(ValueError, match="unique and ascending"):
        morning_dip_row(bars.iloc[:-1], bars.Open.iloc[-1], bars.index[-1].date())


@pytest.mark.parametrize("tz", [None, "America/New_York"])
def test_session_completion_inserts_exchange_gaps_but_not_holidays(tz):
    bars = history(n=4)
    bars.index = pd.DatetimeIndex(["2026-11-24", "2026-11-25", "2026-11-27", "2026-12-01"], tz=tz)
    actual = complete_dip_history(bars)
    absent_session = pd.Timestamp("2026-11-30", tz=tz)
    assert actual.loc[absent_session].isna().all()
    assert pd.Timestamp("2026-11-26", tz=tz) not in actual.index  # Thanksgiving.
    assert pd.Timestamp("2026-11-28", tz=tz) not in actual.index
    assert actual.index.tz == bars.index.tz
    pd.testing.assert_frame_equal(actual.loc[bars.index], bars)
