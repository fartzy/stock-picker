from datetime import date

import pandas as pd

from stock_picker.features.hold_to_close import apply_hold_to_close
from stock_picker.storage.price_store import PriceStore


def _prices(tmp_path, ticker, day, open_px, close):
    store = PriceStore(data_dir=tmp_path)
    history = pd.DataFrame(
        {"Open": [open_px], "High": [close], "Low": [close - 0.2], "Close": [close], "Volume": [1.0]},
        index=pd.DatetimeIndex([day]),
    )
    store.write(ticker, history)
    return store


def test_hold_to_close_is_open_to_close_not_the_actual_fill(tmp_path):
    store = _prices(tmp_path, "XNDU", "2026-09-17", 7.67, 7.66)
    lots = [
        {
            "ticker": "XNDU",
            "shares": 10800,
            "buy_time": "2026-09-17T11:26:00-04:00",
            "buy_price": 7.65,
            "sell_price": 7.65,
            "pnl": 11.91,
        }
    ]

    annotated = apply_hold_to_close(lots, price_store=store, completed_through=date(2026, 9, 17))

    assert annotated[0]["hold_open_price"] == 7.67
    assert annotated[0]["hold_close_price"] == 7.66
    assert annotated[0]["hold_close_pnl"] == round((7.66 - 7.67) * 10800, 2)
    assert annotated[0]["pnl"] == 11.91


def test_in_progress_session_stays_blank(tmp_path):
    store = _prices(tmp_path, "BHVN", "2026-09-18", 13.50, 14.00)
    lots = [
        {
            "ticker": "BHVN",
            "shares": 5790,
            "buy_time": "2026-09-18T10:11:00-04:00",
            "buy_price": 13.67,
            "pnl": 427.23,
        }
    ]

    annotated = apply_hold_to_close(lots, price_store=store, completed_through=date(2026, 9, 17))

    assert annotated[0]["hold_close_price"] is None
    assert annotated[0]["hold_close_pnl"] is None


def test_missing_price_file_is_blank_not_an_error(tmp_path):
    store = PriceStore(data_dir=tmp_path)
    lots = [
        {
            "ticker": "ZZZZ",
            "shares": 10,
            "buy_time": "2026-09-17T10:00:00-04:00",
            "buy_price": 1.0,
            "pnl": 0.0,
        }
    ]

    annotated = apply_hold_to_close(lots, price_store=store, completed_through=date(2026, 9, 17))

    assert annotated[0]["hold_close_price"] is None
    assert annotated[0]["hold_close_pnl"] is None
