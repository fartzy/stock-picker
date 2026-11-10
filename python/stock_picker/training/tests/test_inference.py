from datetime import date

import pandas as pd
import pytest

from stock_picker.features.dip_context import (
    DIP_CLOSE_KNOWN_COLUMNS, DIP_COLUMNS, DIP_OPEN_KNOWN_COLUMNS,
    build_dip_context_features, complete_dip_history, morning_dip_features,
)
from stock_picker.features.open_pattern_seasonality import OPEN_KNOWN_COLUMNS, build_open_pattern_features
from stock_picker.features.tests.fixtures import synthetic_history
from stock_picker.training.dataset import GAP_COLUMN, LABEL_COLUMN, build_training_frame
from stock_picker.training.ensemble import Ensemble
from stock_picker.training.inference import (
    StaleFeatureSnapshotError,
    build_inference_row,
    compute_overnight_gap,
    predict_signal,
)
from stock_picker.training.model import feature_columns, train_lightgbm


def _dip_live_row(history, snapshot, *, excluded=None, prior_history=None):
    return build_inference_row(
        snapshot, float(history.Open.iloc[-1]), float(history.Close.iloc[-2]),
        history.index[-2].date(), history.index[-1].date(),
        prior_history=prior_history, excluded_features=excluded,
    )


def test_dip_live_row_matches_dataset_and_decision_helper_from_final_snapshot():
    history = synthetic_history(n=320)
    snapshots = build_dip_context_features(history)
    # Build yesterday's final snapshot without today's price row present.
    snapshot = build_dip_context_features(history.iloc[:-1]).iloc[-1]
    actual = _dip_live_row(history, snapshot, prior_history=history.iloc[:-1]).iloc[0][list(DIP_COLUMNS)]
    expected = build_training_frame(history, snapshots).iloc[-1][list(DIP_COLUMNS)]
    pd.testing.assert_series_equal(actual, expected, check_names=False)
    pd.testing.assert_series_equal(actual, morning_dip_features(history).iloc[-1], check_names=False)


@pytest.mark.parametrize("empty", [False, True])
def test_dip_live_missing_history_clears_stale_open_values_and_keeps_close_fields(empty):
    history = synthetic_history(n=320)
    snapshot = build_dip_context_features(history.iloc[:-1]).iloc[-1]
    snapshot.loc[list(DIP_OPEN_KNOWN_COLUMNS)] = 123.0
    row = _dip_live_row(history, snapshot, prior_history=history.iloc[:0] if empty else None).iloc[0]
    assert row[list(DIP_OPEN_KNOWN_COLUMNS)].isna().all()
    pd.testing.assert_series_equal(row[list(DIP_CLOSE_KNOWN_COLUMNS)], snapshot[list(DIP_CLOSE_KNOWN_COLUMNS)])


def test_dip_live_all_pruned_skips_helper_and_does_not_add_absent_fields(monkeypatch):
    from stock_picker.training import inference

    def forbidden(*args, **kwargs):
        raise AssertionError("pruned dip inputs must not be recomputed")

    monkeypatch.setattr(inference, "morning_dip_row", forbidden)
    history = synthetic_history(n=320)
    snapshot = build_dip_context_features(history.iloc[:-1]).iloc[-1]
    actual = _dip_live_row(history, snapshot, excluded=frozenset(DIP_COLUMNS), prior_history=history.iloc[:-1])
    assert actual[list(DIP_OPEN_KNOWN_COLUMNS)].isna().all().all()
    assert feature_columns(actual, excluded_features=set(DIP_COLUMNS)) == [GAP_COLUMN]
    actual = _dip_live_row(history, pd.Series({"ordinary_feature": 1.}), prior_history=history.iloc[:-1])
    assert not set(DIP_COLUMNS) & set(actual.columns)


def test_dip_live_partial_prune_recomputes_only_selected_open_fields():
    history = synthetic_history(n=320)
    snapshot = build_dip_context_features(history.iloc[:-1]).iloc[-1]
    excluded = frozenset({"dip_recovery_fraction", "dip_deep_unrecovered"})
    actual = _dip_live_row(history, snapshot, excluded=excluded, prior_history=history.iloc[:-1]).iloc[0]
    expected = morning_dip_features(history).iloc[-1]
    assert actual[list(excluded)].isna().all()
    included = sorted(DIP_OPEN_KNOWN_COLUMNS - excluded)
    pd.testing.assert_series_equal(actual[included], expected[included], check_names=False)


def test_dip_columns_follow_ordinary_default_selection_and_explicit_pruning():
    frame = build_dip_context_features(synthetic_history(n=320))
    assert feature_columns(frame) == list(DIP_COLUMNS)
    excluded = {"dip_deep_unrecovered", "dip_open_to_low_252d"}
    assert feature_columns(frame, excluded_features=excluded) == [c for c in DIP_COLUMNS if c not in excluded]
    selected = {"dip_deep_unrecovered", "dip_pressure_easing"}
    assert feature_columns(frame, included_features=selected, excluded_features=excluded) == ["dip_pressure_easing"]


@pytest.mark.parametrize("missing_offset", [1, 12])
def test_dip_live_and_dataset_agree_when_an_exchange_session_is_absent(missing_offset):
    history = synthetic_history(n=320)
    # Use a known run of ordinary exchange sessions ending Dec 3, 2026.
    history.index = pd.bdate_range(end="2026-12-03", periods=len(history))
    missing_date = history.index[-1 - missing_offset]
    history = history.drop(index=missing_date)
    snapshots = build_dip_context_features(history)
    snapshot = build_dip_context_features(history.iloc[:-1]).iloc[-1]
    actual = _dip_live_row(history, snapshot, prior_history=history.iloc[:-1]).iloc[0][list(DIP_COLUMNS)]
    expected = morning_dip_features(complete_dip_history(history)).iloc[-1]
    training = build_training_frame(history, snapshots).iloc[-1][list(DIP_COLUMNS)]
    pd.testing.assert_series_equal(actual, expected, check_names=False)
    pd.testing.assert_series_equal(actual, training, check_names=False)
    assert actual.dip_low_history_valid == 0

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


def test_build_inference_row_keeps_a_large_overnight_gap():
    history, features = _make_history_and_features()
    prior_day_features = features.iloc[-2]
    yesterday_close = history["Close"].iloc[-2]
    # A 50% overnight move is a real open (earnings, crash) -- the quote
    # fetcher already required it to be dated today, so scoring it is correct.
    large_open = yesterday_close * 1.5

    row = build_inference_row(prior_day_features, large_open, yesterday_close, SNAPSHOT_DATE, AS_OF_DATE)

    assert row.iloc[0][GAP_COLUMN] == pytest.approx(0.5)


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


def test_build_inference_row_recomputes_open_known_columns_from_today_open():
    history = synthetic_history(n=80)
    features = build_open_pattern_features(history)
    features[GAP_COLUMN] = (history["Open"] - history["Close"].shift(1)) / history["Close"].shift(1)
    prior_history = history.iloc[:-1]
    today_open = float(history["Open"].iloc[-1])
    yesterday_close = float(history["Close"].iloc[-2])

    row = build_inference_row(
        prior_day_features=features.iloc[-2],
        today_open=today_open,
        yesterday_close=yesterday_close,
        snapshot_date=SNAPSHOT_DATE,
        as_of_date=AS_OF_DATE,
        prior_history=prior_history,
    )

    expected = features.iloc[-1]
    for column in OPEN_KNOWN_COLUMNS:
        actual = row.iloc[0][column]
        target = expected[column]
        if pd.isna(target):
            assert pd.isna(actual)
        else:
            assert actual == pytest.approx(target)
