from datetime import date

import pandas as pd
import pytest

from stock_picker.training.dataset import GAP_COLUMN, LABEL_COLUMN, build_training_frame
from stock_picker.training.ensemble import Ensemble
from stock_picker.training.inference import (
    ImplausibleGapError,
    StaleFeatureSnapshotError,
    build_inference_row,
    compute_overnight_gap,
    predict_signal,
)
from stock_picker.training.model import train_lightgbm

# Arbitrary but fixed dates a day apart, so "fresh" is the default across every
# test unless a test is specifically about staleness.
SNAPSHOT_DATE = date(2026, 1, 1)
AS_OF_DATE = date(2026, 1, 2)


def _make_history_and_features(n=15):
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
    # gap computed with the real formula so it agrees with compute_overnight_gap
    gap = (history["Open"] - history["Close"].shift(1)) / history["Close"].shift(1)
    features = pd.DataFrame(
        {"some_feature": [float(i) for i in range(n)], GAP_COLUMN: gap},
        index=dates,
    )
    return history, features


def test_compute_overnight_gap_matches_hand_computed_value():
    assert compute_overnight_gap(today_open=102.0, yesterday_close=100.0) == pytest.approx(0.02)


def test_build_inference_row_matches_what_dataset_would_have_produced():
    history, features = _make_history_and_features()
    frame = build_training_frame(history, features)

    # "this morning" at the open of the last date, holding yesterday's snapshot
    prior_day_features = features.iloc[-2]
    today_open = history["Open"].iloc[-1]
    yesterday_close = history["Close"].iloc[-2]

    inference_row = build_inference_row(
        prior_day_features, today_open, yesterday_close, SNAPSHOT_DATE, AS_OF_DATE
    )

    # only the feature values need to match -- build_inference_row's row is labeled
    # by the input snapshot's date, not the date being predicted for
    expected_row = frame.drop(columns=[LABEL_COLUMN]).iloc[[-1]]
    pd.testing.assert_frame_equal(
        inference_row.reset_index(drop=True),
        expected_row.reset_index(drop=True),
        check_names=False,
    )


def test_build_inference_row_raises_when_snapshot_is_stale():
    history, features = _make_history_and_features()
    prior_day_features = features.iloc[-2]
    today_open = history["Open"].iloc[-1]
    yesterday_close = history["Close"].iloc[-2]
    stale_as_of = date(2026, 1, 10)  # well beyond the snapshot's 1-day ttl

    with pytest.raises(StaleFeatureSnapshotError):
        build_inference_row(prior_day_features, today_open, yesterday_close, SNAPSHOT_DATE, stale_as_of)


def test_build_inference_row_raises_on_an_implausible_gap():
    history, features = _make_history_and_features()
    prior_day_features = features.iloc[-2]
    yesterday_close = history["Close"].iloc[-2]
    # A 50% overnight "gap" -- e.g. a 2-for-1 stock split misread as a real move.
    split_like_open = yesterday_close * 1.5

    with pytest.raises(ImplausibleGapError):
        build_inference_row(prior_day_features, split_like_open, yesterday_close, SNAPSHOT_DATE, AS_OF_DATE)


def test_predict_signal_returns_a_float_using_a_trained_model():
    history, features = _make_history_and_features(n=30)
    frame = build_training_frame(history, features)
    model = train_lightgbm(frame, params={"min_data_in_leaf": 2}, num_boost_round=5)
    ensemble = Ensemble(members=[model], weights=[1.0])

    prior_day_features = features.iloc[-2]
    today_open = history["Open"].iloc[-1]
    yesterday_close = history["Close"].iloc[-2]
    inference_row = build_inference_row(
        prior_day_features, today_open, yesterday_close, SNAPSHOT_DATE, AS_OF_DATE
    )

    signal = predict_signal(ensemble, inference_row)

    assert isinstance(signal, float)
