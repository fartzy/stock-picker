from datetime import date
from math import isclose

import pandas as pd

from stock_picker.features.hold_to_close import apply_hold_to_close
from stock_picker.features.trades import position_summaries
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
            "buy_time": "2026-09-17T09:40:00-04:00",
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
            "buy_time": "2026-09-18T09:41:00-04:00",
            "buy_price": 13.67,
            "pnl": 427.23,
        }
    ]

    annotated = apply_hold_to_close(lots, price_store=store, completed_through=date(2026, 9, 17))

    assert annotated[0]["hold_close_price"] is None
    assert annotated[0]["hold_close_pnl"] is None


def test_stored_bar_wins_over_live_quotes(tmp_path):
    store = _prices(tmp_path, "ZS", "2026-09-25", 205.50, 193.05)
    lots = [
        {
            "ticker": "ZS",
            "shares": 25,
            "buy_time": "2026-09-25T09:35:00-04:00",
            "buy_price": 194.52,
            "pnl": 55.0,
        }
    ]

    annotated = apply_hold_to_close(
        lots,
        price_store=store,
        completed_through=date(2026, 9, 25),
        live_quotes={"ZS": {"open": 1.0, "last": 2.0}},
    )

    assert annotated[0]["hold_open_price"] == 205.5
    assert annotated[0]["hold_close_price"] == 193.05


def test_live_quotes_fill_after_close_when_todays_bar_is_missing(tmp_path):
    store = PriceStore(data_dir=tmp_path)
    lots = [
        {
            "ticker": "ZS",
            "shares": 25,
            "buy_time": "2026-09-28T09:35:00-04:00",
            "buy_price": 194.52,
            "pnl": 55.0,
        }
    ]

    annotated = apply_hold_to_close(
        lots,
        price_store=store,
        completed_through=date(2026, 9, 28),
        live_quotes={"ZS": {"open": 188.89, "last": 199.01}},
    )

    assert annotated[0]["hold_open_price"] == 188.89
    assert annotated[0]["hold_close_price"] == 199.01
    assert annotated[0]["hold_close_pnl"] == round((199.01 - 188.89) * 25, 2)


def test_missing_price_file_is_blank_not_an_error(tmp_path):
    store = PriceStore(data_dir=tmp_path)
    lots = [
        {
            "ticker": "ZZZZ",
            "shares": 10,
            "buy_time": "2026-09-17T09:59:00-04:00",
            "buy_price": 1.0,
            "pnl": 0.0,
        }
    ]

    annotated = apply_hold_to_close(lots, price_store=store, completed_through=date(2026, 9, 17))

    assert annotated[0]["hold_close_price"] is None
    assert annotated[0]["hold_close_pnl"] is None


def test_cutoff_is_strictly_before_nine_chicago_in_any_offset(tmp_path):
    store = _prices(tmp_path, "AAA", "2026-09-29", 10.0, 11.0)
    cases = [
        ("2026-09-29T09:59:59-04:00", True),
        ("2026-09-29T08:59:59-05:00", True),
        ("2026-09-29T13:59:59Z", True),
        ("2026-09-29T10:00:00-04:00", False),
        ("2026-09-29T09:00:00-05:00", False),
        ("2026-09-29T14:00:00Z", False),
        ("2026-09-29T12:25:19-04:00", False),
    ]
    for stamp, eligible in cases:
        lot = {"ticker": "AAA", "shares": 10, "invested": 105, "buy_time": stamp, "pnl": 7.25}
        row = apply_hold_to_close([lot], store, date(2026, 9, 29))[0]
        assert row["hold_close_pnl"] == (10 if eligible else None), stamp
        assert row["hold_eligible_shares"] == (10 if eligible else 0), stamp
        assert row["hold_eligible_invested"] == (105 if eligible else 0), stamp
        assert row["hold_close_price"] == 11  # Exclusion doesn't hide the Close column.
        assert row["pnl"] == lot["pnl"]
        assert row["invested"] == lot["invested"]
        assert "hold_close_pnl" not in lot  # Annotation does not mutate the input.


def test_cutoff_observes_winter_central_time(tmp_path):
    store = _prices(tmp_path, "AAA", "2026-01-15", 10.0, 11.0)
    for stamp, expected in [("2026-01-15T14:59:59Z", 10), ("2026-01-15T15:00:00Z", None)]:
        lot = {"ticker": "AAA", "shares": 10, "buy_time": stamp}
        row = apply_hold_to_close([lot], store, date(2026, 1, 15))[0]
        assert row["hold_close_pnl"] == expected


def test_late_overnight_entry_is_not_eligible_on_the_sell_day(tmp_path):
    store = _prices(tmp_path, "VCYT", "2026-09-28", 45.04, 45.23)
    lot = {
        "ticker": "VCYT", "shares": 120, "buy_time": "2026-09-28T12:00:00-04:00",
        "sell_time": "2026-09-29T09:36:49-04:00", "pnl": 179.4,
    }
    row = apply_hold_to_close([lot], store, date(2026, 9, 29))[0]
    assert row["hold_close_pnl"] is None
    assert row["hold_eligible_shares"] == 0
    assert row["pnl"] == 179.4


def test_late_add_on_preserves_early_shares_across_partial_exits(tmp_path):
    store = _prices(tmp_path, "AAA", "2026-09-29", 10.0, 12.0)
    trades = pd.DataFrame([
        {"ticker": "AAA", "side": "buy", "shares": 100, "price": 10, "executed_at": "2026-09-29T09:35:00-04:00"},
        {"ticker": "AAA", "side": "buy", "shares": 100, "price": 8, "executed_at": "2026-09-29T12:30:00-04:00"},
        {"ticker": "AAA", "side": "sell", "shares": 50, "price": 11, "executed_at": "2026-09-29T13:00:00-04:00"},
        {"ticker": "AAA", "side": "sell", "shares": 50, "price": 12, "executed_at": "2026-09-29T14:00:00-04:00"},
    ])
    positions = position_summaries(trades, {"AAA": {"last": 12}})
    rows = apply_hold_to_close(positions, store, date(2026, 9, 29))
    assert all(isclose(r["hold_eligible_shares"], expected) for r, expected in zip(rows, [25, 25, 50]))
    assert [r["hold_eligible_invested"] for r in rows] == [250, 250, 500]
    assert [r["hold_close_pnl"] for r in rows] == [50, 50, 100]
    assert [r["pnl"] for r in rows] == [100, 150, 300]
    for position, row in zip(positions, rows):
        assert {key: row[key] for key in position if key != "_buy_allocations"} == {
            key: value for key, value in position.items() if key != "_buy_allocations"
        }
        assert "_buy_allocations" not in row


def test_late_reentry_after_flat_is_excluded(tmp_path):
    store = _prices(tmp_path, "GRDN", "2026-09-29", 40.48, 40.68)
    trades = pd.DataFrame([
        {"ticker": "GRDN", "side": "buy", "shares": 200, "price": 40.51, "executed_at": "2026-09-29T09:35:07-04:00"},
        {"ticker": "GRDN", "side": "sell", "shares": 200, "price": 40.765, "executed_at": "2026-09-29T10:21:13-04:00"},
        {"ticker": "GRDN", "side": "buy", "shares": 125, "price": 40.505, "executed_at": "2026-09-29T12:25:19-04:00"},
        {"ticker": "GRDN", "side": "sell", "shares": 125, "price": 40.71, "executed_at": "2026-09-29T12:56:42-04:00"},
    ])
    rows = apply_hold_to_close(position_summaries(trades, {}), store, date(2026, 9, 29))
    assert [r["hold_close_pnl"] for r in rows] == [40, None]
    assert [r["hold_eligible_shares"] for r in rows] == [200, 0]


def test_unknown_entry_time_is_excluded(tmp_path):
    lot = {"ticker": "AAA", "shares": 10, "buy_time": None}
    row = apply_hold_to_close([lot], PriceStore(tmp_path), date(2026, 9, 29))[0]
    assert row["hold_close_pnl"] is None
    assert row["hold_eligible_shares"] == 0
