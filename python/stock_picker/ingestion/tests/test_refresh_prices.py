from datetime import date

import pandas as pd

from stock_picker.ingestion.refresh_prices import (
    backfill_missing_sessions,
    history_from_finnhub_bars,
    tickers_missing_session,
)
from stock_picker.storage.price_store import PriceStore


def _bar(day, open_price=10.0, close_price=10.5):
    return {
        "date": day,
        "Open": open_price,
        "High": close_price + 0.2,
        "Low": open_price - 0.2,
        "Close": close_price,
        "Volume": 1000.0,
    }


def test_tickers_missing_session_includes_absent_and_stale(tmp_path):
    store = PriceStore(data_dir=tmp_path)
    store.write(
        "AAPL",
        pd.DataFrame(
            {"Open": [1.0], "High": [1.1], "Low": [0.9], "Close": [1.05], "Adj Close": [1.05], "Volume": [1.0]},
            index=pd.DatetimeIndex(["2026-09-22"]),
        ),
    )

    missing = tickers_missing_session(["AAPL", "OMC"], store, date(2026, 9, 23))

    assert missing == ["AAPL", "OMC"]


def test_backfill_writes_finnhub_bars_for_names_yahoo_missed(tmp_path):
    store = PriceStore(data_dir=tmp_path)

    def fetch_candles(ticker, start, cutoff):
        if ticker != "OMC":
            return []
        return [_bar(date(2026, 9, 23))]

    stats = backfill_missing_sessions(
        ["OMC"],
        store,
        date(2026, 9, 23),
        fetch_candles=fetch_candles,
        sleep_seconds=0,
    )

    assert stats["written"] == 1
    history = store.read("OMC")
    assert history.index[-1].date() == date(2026, 9, 23)
    assert history.iloc[-1]["Open"] == 10.0


def test_backfill_skips_a_none_fetch_so_a_later_nightly_can_retry(tmp_path):
    store = PriceStore(data_dir=tmp_path)

    stats = backfill_missing_sessions(
        ["BAD"],
        store,
        date(2026, 9, 23),
        fetch_candles=lambda *a, **k: None,
        sleep_seconds=0,
    )

    assert stats["written"] == 0
    assert tickers_missing_session(["BAD"], store, date(2026, 9, 23)) == ["BAD"]


def test_history_from_finnhub_bars_sets_adj_close():
    frame = history_from_finnhub_bars([_bar(date(2026, 9, 23), open_price=4.0, close_price=4.5)])

    assert list(frame.columns) == ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    assert frame.iloc[0]["Adj Close"] == 4.5
