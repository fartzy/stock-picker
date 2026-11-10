import pandas as pd

from stock_picker.features.dip_context import (
    DIP_COLUMNS, build_dip_context_features, complete_dip_history, morning_dip_features,
)
from stock_picker.training.dataset import (
    LABEL_COLUMN,
    build_pooled_dataset,
    build_training_frame,
)


def _make_history_and_features(n=10):
    dates = pd.date_range("2026-01-01", periods=n, freq="B")
    history = pd.DataFrame(
        {
            "Open": [100.0 + i for i in range(n)],
            "High": [101.0 + i for i in range(n)],
            "Low": [99.0 + i for i in range(n)],
            "Close": [100.5 + i for i in range(n)],
            "Volume": [1000 + i for i in range(n)],
        },
        index=dates,
    )
    # a feature that obviously depends on today's close -- must NOT leak into row t
    features = pd.DataFrame(
        {
            "leaky_feature": history["Close"] * 2,
            "overnight_gap": [float(i) for i in range(n)],
        },
        index=dates,
    )
    return history, features


def test_dip_snapshot_dataset_alignment_matches_decision_helper_for_all_rows():
    history, _ = _make_history_and_features(n=320)
    history.iloc[275] = [150, 151, 148, 149, 2000]
    history.iloc[276] = [147, 148, 144, 145, 1500]
    history.iloc[110] = float("nan")  # Missing bars must remain missing after alignment.
    snapshots = build_dip_context_features(history)
    actual = build_training_frame(history, snapshots).loc[:, list(DIP_COLUMNS)]
    expected = morning_dip_features(history).loc[actual.index]
    pd.testing.assert_frame_equal(actual, expected)


def test_dip_training_inputs_are_invariant_to_current_and_future_outcomes():
    history, _ = _make_history_and_features(n=320)
    cutoff = history.index[290]
    expected = build_training_frame(history, build_dip_context_features(history)).loc[cutoff, list(DIP_COLUMNS)]
    changed = history.copy()
    changed.loc[cutoff:, ["High", "Low", "Close", "Volume"]] = [1e8, .01, 1e6, 1e12]
    changed.loc[changed.index[291]:, "Open"] = 1e7
    actual = build_training_frame(changed, build_dip_context_features(changed)).loc[cutoff, list(DIP_COLUMNS)]
    pd.testing.assert_series_equal(actual, expected)


def test_missing_exchange_date_preserves_labels_and_old_features_but_not_dip_shift():
    history, old_features = _make_history_and_features(n=320)
    missing = pd.Timestamp("2026-12-01")  # An ordinary Tuesday session.
    history = history.drop(index=missing)
    old_features = old_features.loc[history.index]
    features = pd.concat([old_features, build_dip_context_features(history)], axis=1)
    actual = build_training_frame(history, features)
    original = build_training_frame(history, old_features)
    pd.testing.assert_frame_equal(actual[original.columns], original)
    expected = morning_dip_features(complete_dip_history(history)).loc[actual.index]
    pd.testing.assert_frame_equal(actual[list(DIP_COLUMNS)], expected)
    assert actual.loc["2026-12-02", "dip_low_history_valid"] == 0
    assert pd.isna(actual.loc["2026-12-02", "dip_loss_2d_sigma"])


def test_label_matches_day_session_return():
    history, features = _make_history_and_features()

    frame = build_training_frame(history, features)

    expected = (history["Close"] - history["Open"]) / history["Open"]
    pd.testing.assert_series_equal(frame[LABEL_COLUMN], expected.iloc[1:], check_names=False)


def test_zero_open_is_not_a_training_label():
    history, features = _make_history_and_features()
    bad_date = history.index[3]
    history.loc[bad_date, "Open"] = 0.0

    frame = build_training_frame(history, features)

    assert bad_date not in frame.index
    assert frame[LABEL_COLUMN].notna().all()


def test_shifted_feature_uses_prior_day_not_same_day():
    history, features = _make_history_and_features()

    frame = build_training_frame(history, features)

    for date, row in frame.iterrows():
        prior_date_pos = features.index.get_loc(date) - 1
        expected_value = features["leaky_feature"].iloc[prior_date_pos]
        assert row["leaky_feature"] == expected_value
        assert row["leaky_feature"] != features.loc[date, "leaky_feature"]


def test_overnight_gap_is_not_shifted():
    history, features = _make_history_and_features()

    frame = build_training_frame(history, features)

    for date, row in frame.iterrows():
        assert row["overnight_gap"] == features.loc[date, "overnight_gap"]


def test_open_known_recency_columns_are_not_shifted():
    history, features = _make_history_and_features()
    features["seq3_open3_seasonality"] = [float(i) for i in range(len(history))]
    features["open_in_yday_range_seasonality"] = [float(i) + 0.5 for i in range(len(history))]

    frame = build_training_frame(history, features)

    for date, row in frame.iterrows():
        assert row["seq3_open3_seasonality"] == features.loc[date, "seq3_open3_seasonality"]
        assert row["open_in_yday_range_seasonality"] == features.loc[date, "open_in_yday_range_seasonality"]


def test_first_row_is_dropped():
    history, features = _make_history_and_features()

    frame = build_training_frame(history, features)

    assert history.index[0] not in frame.index


def test_build_pooled_dataset_combines_tickers_with_metadata_columns():
    history, features = _make_history_and_features()
    histories = {"AAA": history, "BBB": history}
    features_by_ticker = {"AAA": features, "BBB": features}

    pooled = build_pooled_dataset(histories, features_by_ticker)

    assert set(pooled["ticker"]) == {"AAA", "BBB"}
    assert "date" in pooled.columns
    assert len(pooled) == 2 * (len(history) - 1)



def test_news_columns_are_shifted_with_close_known_features():
    history, features = _make_history_and_features()
    features["news_has_material_1d"] = [float(i) for i in range(len(history))]
    features["news_article_count_3d"] = [float(i) + 10 for i in range(len(history))]

    frame = build_training_frame(history, features)

    for date, row in frame.iterrows():
        prior_date_pos = features.index.get_loc(date) - 1
        assert row["news_has_material_1d"] == features["news_has_material_1d"].iloc[prior_date_pos]
        assert row["news_article_count_3d"] == features["news_article_count_3d"].iloc[prior_date_pos]
