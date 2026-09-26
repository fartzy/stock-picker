"""Time live-row rebuild. Measure first -- do not treat spawn as the 4-minute bill.

Run: bazelisk run //python/stock_picker/training:profile_live_rows
"""

from __future__ import annotations

import time
from datetime import date

import pandas as pd

from stock_picker.features.open_pattern_seasonality import (
    _parts,
    build_open_pattern_features,
    open_known_feature_row,
)
from stock_picker.storage.feature_store import FeatureStore
from stock_picker.storage.price_store import PriceStore
from stock_picker.storage.universe_store import UniverseStore
from stock_picker.training.inference import build_inference_row
from stock_picker.training.live_rows import _index_dates, prepare_live_rows, prepare_one


def _ms(seconds: float) -> str:
    return f"{seconds * 1000:.1f}ms"


def main() -> None:
    as_of = date(2026, 9, 25)
    feature_store = FeatureStore()
    price_store = PriceStore()
    tickers = UniverseStore().active_tickers()
    print(f"universe {len(tickers)}")

    ticker = "AAPL"
    t0 = time.perf_counter()
    features = feature_store.read(ticker)
    t_feat = time.perf_counter() - t0
    t0 = time.perf_counter()
    prices = price_store.read(ticker)
    t_price = time.perf_counter() - t0
    print(
        f"AAPL parquet features={_ms(t_feat)} n={len(features)} "
        f"prices={_ms(t_price)} n={len(prices)}"
    )

    t0 = time.perf_counter()
    fdates = _index_dates(features.index)
    prior_features = features.iloc[fdates < as_of]
    pdates = _index_dates(prices.index)
    prior_history = prices.iloc[pdates < as_of]
    t_slice = time.perf_counter() - t0
    print(
        f"AAPL date slice {_ms(t_slice)} feat_rows={len(prior_features)} "
        f"price_rows={len(prior_history)}"
    )

    today_open = float(prior_history["Open"].iloc[-1])
    yesterday_close = float(prior_history["Close"].iloc[-1])
    snapshot = fdates[fdates < as_of][-1]

    t0 = time.perf_counter()
    build_inference_row(
        prior_day_features=prior_features.iloc[-1],
        today_open=today_open,
        yesterday_close=yesterday_close,
        snapshot_date=snapshot,
        as_of_date=as_of,
        prior_history=None,
    )
    t_gap_only = time.perf_counter() - t0

    t0 = time.perf_counter()
    build_inference_row(
        prior_day_features=prior_features.iloc[-1],
        today_open=today_open,
        yesterday_close=yesterday_close,
        snapshot_date=snapshot,
        as_of_date=as_of,
        prior_history=prior_history,
    )
    t_with_open_known = time.perf_counter() - t0

    t0 = time.perf_counter()
    live_row = open_known_feature_row(prior_history, today_open)
    t_open_known = time.perf_counter() - t0
    _ = live_row

    t0 = time.perf_counter()
    build_open_pattern_features(prior_history)
    t_full_hist = time.perf_counter() - t0

    dummy_index = prior_history.index[-1] + pd.Timedelta(days=1)
    dummy = pd.DataFrame(
        {
            "Open": [today_open],
            "High": [today_open],
            "Low": [today_open],
            "Close": [today_open],
            "Volume": [0.0],
        },
        index=pd.DatetimeIndex([dummy_index]),
    )
    for column in prior_history.columns:
        if column not in dummy.columns:
            dummy[column] = float("nan")
    extended = pd.concat([prior_history, dummy[prior_history.columns]])

    t0 = time.perf_counter()
    _parts(extended)
    t_parts = time.perf_counter() - t0

    t0 = time.perf_counter()
    build_open_pattern_features(extended)
    t_build_ext = time.perf_counter() - t0

    print(f"AAPL build_inference_row gap-only {_ms(t_gap_only)}")
    print(f"AAPL build_inference_row + open_known {_ms(t_with_open_known)}")
    print(f"AAPL open_known_feature_row {_ms(t_open_known)}")
    print(f"AAPL build_open_pattern_features whole history {_ms(t_full_hist)}")
    print(f"AAPL _parts(extended) {_ms(t_parts)}")
    print(f"AAPL build_open_pattern_features(extended) {_ms(t_build_ext)}")
    print(f"AAPL implied buckets+means {_ms(t_build_ext - t_parts)}")

    sample = tickers[:50]
    quotes = {}
    for name in sample:
        try:
            hist = price_store.read(name)
            hist = hist.iloc[_index_dates(hist.index) < as_of]
            if hist.empty:
                continue
            quotes[name] = {
                "open": float(hist["Open"].iloc[-1]),
                "prev_close": float(hist["Close"].iloc[-1]),
                "last": float(hist["Close"].iloc[-1]),
            }
        except FileNotFoundError:
            continue

    t0 = time.perf_counter()
    n_ok = 0
    for name in sample:
        item = prepare_one(
            name,
            quotes=quotes,
            earnings=set(),
            feature_store=feature_store,
            price_store=price_store,
            as_of=as_of,
            spy_open=None,
            spy_prev_close=None,
            blocked=set(),
        )
        if item.row is not None:
            n_ok += 1
    elapsed50 = time.perf_counter() - t0
    print(
        f"prepare_one sequential 50 names {elapsed50:.2f}s scored={n_ok} "
        f"per={_ms(elapsed50 / 50)}"
    )
    print(f"extrapolate {len(tickers)} sequential {elapsed50 / 50 * len(tickers):.0f}s")

    t0 = time.perf_counter()
    for name in sample:
        try:
            feature_store.read(name)
            price_store.read(name)
        except FileNotFoundError:
            pass
    t_io50 = time.perf_counter() - t0
    print(f"parquet read only 50 names {t_io50:.2f}s extrapolate {t_io50 / 50 * len(tickers):.1f}s")

    t0 = time.perf_counter()
    prepare_live_rows(
        sample,
        quotes=quotes,
        earnings=set(),
        feature_store=feature_store,
        price_store=price_store,
        as_of=as_of,
        spy_open=None,
        spy_prev_close=None,
        bucket_size=25,
        workers=2,
    )
    t_threads = time.perf_counter() - t0
    print(f"prepare_live_rows 50 names 2 buckets {t_threads:.2f}s")


if __name__ == "__main__":
    main()
