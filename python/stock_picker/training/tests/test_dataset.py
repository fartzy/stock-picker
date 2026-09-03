import pandas as pd

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


def test_label_matches_day_session_return():
    history, features = _make_history_and_features()

    frame = build_training_frame(history, features)

    expected = (history["Close"] - history["Open"]) / history["Open"]
    pd.testing.assert_series_equal(frame[LABEL_COLUMN], expected.iloc[1:], check_names=False)


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
