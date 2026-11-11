import numpy as np
import pandas as pd
import pytest

from stock_picker.features.pattern_evidence import (
    build_pattern_evidence, pattern_evidence_row, prior_pattern_evidence,
)


def history():
    dates = pd.bdate_range("2026-02-02", periods=100)
    opening = 100 + np.arange(100) * 0.1
    close = opening * (1 + np.tile([0.01, -0.02, 0.005, -0.01], 25))
    return pd.DataFrame({"Open": opening, "High": np.maximum(opening, close) + 1,
                         "Low": np.minimum(opening, close) - 1, "Close": close,
                         "Volume": 1000.0}, index=dates)


def test_hand_calculated_counts_dispersion_shrinkage_and_missing_outcomes():
    values = pd.Series([0.1, -0.1, np.nan, 0.3, 999.0])
    keys = pd.Series(["a", "b", "a", "a", "a"])
    result = prior_pattern_evidence(values, keys)
    assert result.loc[0, "count"] == 0
    assert result.loc[2, "count"] == 1
    assert result.loc[3, "count"] == 1  # missing outcome did not add an observation
    assert result.loc[3, "age"] == 3
    assert result.loc[3, "shrunk_mean"] == pytest.approx(0.1 / 21)
    assert result.loc[4, "count"] == 2
    assert result.loc[4, "std"] == pytest.approx(np.std([0.1, 0.3], ddof=1))
    assert result.loc[4, "shrunk_mean"] == pytest.approx((0.4 + 20 * 0.1) / 22)
    decay = 2 ** (-3 / 63)
    assert result.loc[4, "recent_mean"] == pytest.approx((0.1 * decay + 0.3) / (decay + 1))


def test_current_and_future_outcomes_never_change_an_earlier_feature():
    original = history()
    expected = build_pattern_evidence(original)
    changed = original.copy()
    changed.loc[changed.index[70]:, ["High", "Low", "Close", "Volume"]] *= 100
    actual = build_pattern_evidence(changed)
    pd.testing.assert_frame_equal(actual.iloc[:71], expected.iloc[:71])


def test_live_row_equals_historical_row_and_uses_current_open():
    original = history()
    expected = build_pattern_evidence(original).iloc[70]
    actual = pattern_evidence_row(original.iloc[:70], original["Open"].iloc[70], str(original.index[70].date()))
    pd.testing.assert_series_equal(actual, expected)
    # Force a down gap; the original row already belongs to the up-gap bucket.
    changed = pattern_evidence_row(original.iloc[:70], 50.0, str(original.index[70].date()))
    assert not changed.equals(expected)


def test_missing_exchange_session_breaks_sequence_instead_of_compressing_it():
    original = history()
    # A Tuesday in the middle of the history is a real exchange session.
    missing = pd.Timestamp("2026-03-10")
    observed = original.drop(missing)
    result = build_pattern_evidence(observed)
    assert result.loc["2026-03-11"].isna().all()
    assert pd.isna(result.loc["2026-03-12", "seq2_open3_count"])


def test_unknown_pattern_is_not_a_zero_count_pattern():
    result = prior_pattern_evidence(pd.Series([0.1, 0.2]), pd.Series([pd.NA, "new"]))
    assert result.iloc[0].isna().all()
    assert result.loc[1, "count"] == 0
    assert result.loc[1, "shrunk_mean"] == pytest.approx(0.1)
    assert np.isnan(result.loc[1, "std"])
