from datetime import date

import numpy as np
import pandas as pd

from stock_picker.storage.feature_store import FeatureStore
from stock_picker.storage.model_store import ModelStore
from stock_picker.storage.universe_store import UniverseStore
from stock_picker.training.buy_signal import NO_MODEL_SENTINEL, compute_buy_signals
from stock_picker.training.dataset import LABEL_COLUMN
from stock_picker.training.ensemble import Ensemble
from stock_picker.training.main import MODEL_NAME
from stock_picker.training.model import train_lightgbm

_AS_OF = date(2026, 9, 5)
_FRESH_SNAPSHOT_DATE = date(2026, 9, 4)  # 1 day old -- within DEFAULT_TTL_DAYS
_STALE_SNAPSHOT_DATE = date(2026, 9, 2)  # 3 days old -- past DEFAULT_TTL_DAYS


def _trained_ensemble():
    # Proportional (not sign-based) label: a tree regressor's predictions
    # for out-of-training-range inputs just clamp to whichever leaf the
    # split boundary puts them in, so a *magnitude*-ordering test (see
    # test_signals_are_sorted_by_predicted_return_descending below) needs
    # the model to have actually learned "bigger signal -> bigger return,"
    # not just "positive signal -> the same +0.02 regardless of size."
    rng = np.random.default_rng(0)
    signal = rng.normal(size=200)
    label = 0.01 * signal
    train_frame = pd.DataFrame({"signal": signal, LABEL_COLUMN: label})
    model = train_lightgbm(train_frame, params={"min_data_in_leaf": 5}, num_boost_round=20)
    return Ensemble(members=[model], weights=[1.0])


def _stores(tmp_path):
    return (
        UniverseStore(data_dir=tmp_path / "universe"),
        FeatureStore(data_dir=tmp_path / "features"),
        ModelStore(data_dir=tmp_path / "models"),
    )


def _seed_ticker(feature_store, universe_store, ticker, snapshot_date, signal_value):
    universe_store.sync({ticker: "manual"})
    frame = pd.DataFrame({"signal": [signal_value]}, index=pd.DatetimeIndex([snapshot_date], name="date"))
    feature_store.write(ticker, frame)


def _quote(open_price=101.0, prev_close=100.0):
    return {"open": open_price, "last": open_price + 1, "prev_close": prev_close}


def test_no_persisted_model_returns_empty_result_with_one_skip_entry(tmp_path):
    universe_store, feature_store, model_store = _stores(tmp_path)

    result = compute_buy_signals(
        as_of=_AS_OF,
        universe_store=universe_store,
        feature_store=feature_store,
        model_store=model_store,
        quote_fetcher=lambda tickers: {},
    )

    assert result.signals == []
    assert result.skipped == [{"ticker": NO_MODEL_SENTINEL, "reason": "no trained model persisted yet"}]


def test_ticker_clearing_threshold_is_returned_as_a_signal(tmp_path):
    universe_store, feature_store, model_store = _stores(tmp_path)
    model_store.write(MODEL_NAME, _trained_ensemble())
    _seed_ticker(feature_store, universe_store, "AAPL", _FRESH_SNAPSHOT_DATE, signal_value=5.0)

    result = compute_buy_signals(
        threshold=0.005,
        as_of=_AS_OF,
        universe_store=universe_store,
        feature_store=feature_store,
        model_store=model_store,
        quote_fetcher=lambda tickers: {"AAPL": _quote()},
    )

    assert result.scored_count == 1
    assert result.skipped == []
    [signal] = result.signals
    assert signal.ticker == "AAPL"
    assert signal.predicted_return > 0.005
    assert signal.open_price == 101.0
    assert signal.snapshot_date == _FRESH_SNAPSHOT_DATE.isoformat()


def test_ticker_scored_but_below_threshold_is_excluded_from_signals(tmp_path):
    universe_store, feature_store, model_store = _stores(tmp_path)
    model_store.write(MODEL_NAME, _trained_ensemble())
    _seed_ticker(feature_store, universe_store, "MSFT", _FRESH_SNAPSHOT_DATE, signal_value=-5.0)

    result = compute_buy_signals(
        threshold=0.005,
        as_of=_AS_OF,
        universe_store=universe_store,
        feature_store=feature_store,
        model_store=model_store,
        quote_fetcher=lambda tickers: {"MSFT": _quote()},
    )

    assert result.scored_count == 1
    assert result.signals == []
    assert result.skipped == []


def test_ticker_missing_a_live_quote_is_skipped_with_a_reason(tmp_path):
    universe_store, feature_store, model_store = _stores(tmp_path)
    model_store.write(MODEL_NAME, _trained_ensemble())
    _seed_ticker(feature_store, universe_store, "AAPL", _FRESH_SNAPSHOT_DATE, signal_value=5.0)

    result = compute_buy_signals(
        as_of=_AS_OF,
        universe_store=universe_store,
        feature_store=feature_store,
        model_store=model_store,
        quote_fetcher=lambda tickers: {},
    )

    assert result.signals == []
    assert result.scored_count == 0
    assert result.skipped == [{"ticker": "AAPL", "reason": "no live quote available"}]


def test_ticker_with_no_previous_close_is_skipped_with_a_reason(tmp_path):
    universe_store, feature_store, model_store = _stores(tmp_path)
    model_store.write(MODEL_NAME, _trained_ensemble())
    _seed_ticker(feature_store, universe_store, "AAPL", _FRESH_SNAPSHOT_DATE, signal_value=5.0)

    result = compute_buy_signals(
        as_of=_AS_OF,
        universe_store=universe_store,
        feature_store=feature_store,
        model_store=model_store,
        quote_fetcher=lambda tickers: {"AAPL": {"open": 101.0, "last": 102.0, "prev_close": None}},
    )

    assert result.signals == []
    assert result.skipped == [{"ticker": "AAPL", "reason": "no previous close available"}]


def test_ticker_with_no_feature_history_is_skipped_with_a_reason(tmp_path):
    universe_store, feature_store, model_store = _stores(tmp_path)
    model_store.write(MODEL_NAME, _trained_ensemble())
    universe_store.sync({"AAPL": "manual"})  # active, but never had a feature table written

    result = compute_buy_signals(
        as_of=_AS_OF,
        universe_store=universe_store,
        feature_store=feature_store,
        model_store=model_store,
        quote_fetcher=lambda tickers: {"AAPL": _quote()},
    )

    assert result.signals == []
    assert result.skipped == [{"ticker": "AAPL", "reason": "no feature history"}]


def test_stale_snapshot_is_skipped_with_the_freshness_error_message(tmp_path):
    universe_store, feature_store, model_store = _stores(tmp_path)
    model_store.write(MODEL_NAME, _trained_ensemble())
    _seed_ticker(feature_store, universe_store, "AAPL", _STALE_SNAPSHOT_DATE, signal_value=5.0)

    result = compute_buy_signals(
        as_of=_AS_OF,
        universe_store=universe_store,
        feature_store=feature_store,
        model_store=model_store,
        quote_fetcher=lambda tickers: {"AAPL": _quote()},
    )

    assert result.signals == []
    assert result.scored_count == 0
    [skip] = result.skipped
    assert skip["ticker"] == "AAPL"
    assert "stale" not in skip["reason"]  # message names the actual snapshot age, not the word "stale"
    assert str(_STALE_SNAPSHOT_DATE) in skip["reason"]


def test_signals_are_sorted_by_predicted_return_descending(tmp_path):
    universe_store, feature_store, model_store = _stores(tmp_path)
    model_store.write(MODEL_NAME, _trained_ensemble())
    _seed_ticker(feature_store, universe_store, "WEAK", _FRESH_SNAPSHOT_DATE, signal_value=1.0)
    _seed_ticker(feature_store, universe_store, "STRONG", _FRESH_SNAPSHOT_DATE, signal_value=9.0)

    result = compute_buy_signals(
        threshold=0.0,
        as_of=_AS_OF,
        universe_store=universe_store,
        feature_store=feature_store,
        model_store=model_store,
        quote_fetcher=lambda tickers: {"WEAK": _quote(), "STRONG": _quote()},
    )

    assert [signal.ticker for signal in result.signals] == ["STRONG", "WEAK"]


def test_top_drivers_reflects_the_ensembles_blended_feature_importance(tmp_path):
    universe_store, feature_store, model_store = _stores(tmp_path)
    model_store.write(MODEL_NAME, _trained_ensemble())
    _seed_ticker(feature_store, universe_store, "AAPL", _FRESH_SNAPSHOT_DATE, signal_value=5.0)

    result = compute_buy_signals(
        as_of=_AS_OF,
        universe_store=universe_store,
        feature_store=feature_store,
        model_store=model_store,
        quote_fetcher=lambda tickers: {"AAPL": _quote()},
    )

    # The only feature the fixture ensemble was trained on -- it must
    # dominate its own blended importance.
    assert result.top_drivers[0][0] == "signal"
