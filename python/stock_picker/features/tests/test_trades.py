import pandas as pd

from stock_picker.features.trades import position_summaries, trade_history


def test_buy_then_sell_computes_realized_pnl():
    trades = pd.DataFrame(
        [
            {"ticker": "CIEN", "side": "buy", "shares": 30, "price": 320.73, "executed_at": "2026-09-04T10:09:39-04:00"},
            {"ticker": "CIEN", "side": "sell", "shares": 30, "price": 320.93, "executed_at": "2026-09-04T15:57:30-04:00"},
        ]
    )

    history = trade_history(trades)

    buy_row = next(t for t in history if t["side"] == "buy")
    sell_row = next(t for t in history if t["side"] == "sell")
    assert buy_row["realized_pnl"] is None
    assert sell_row["realized_pnl"] == round((320.93 - 320.73) * 30, 2)


def test_partial_sell_uses_average_cost_basis():
    trades = pd.DataFrame(
        [
            {"ticker": "AAA", "side": "buy", "shares": 10, "price": 100.0, "executed_at": "2026-09-04T09:00:00-04:00"},
            {"ticker": "AAA", "side": "buy", "shares": 10, "price": 110.0, "executed_at": "2026-09-04T10:00:00-04:00"},
            {"ticker": "AAA", "side": "sell", "shares": 5, "price": 120.0, "executed_at": "2026-09-04T11:00:00-04:00"},
        ]
    )

    history = trade_history(trades)

    sell_row = next(t for t in history if t["side"] == "sell")
    # avg cost basis over 20 shares @ (100+110)/2 = 105 -> pnl = (120-105)*5 = 75
    assert sell_row["realized_pnl"] == 75.0


def test_trade_history_handles_empty_input():
    assert trade_history(pd.DataFrame()) == []


def test_position_summaries_merges_closed_position_into_one_row():
    trades = pd.DataFrame(
        [
            {"ticker": "CIEN", "side": "buy", "shares": 30, "price": 320.73, "executed_at": "2026-09-04T10:09:39-04:00"},
            {"ticker": "CIEN", "side": "sell", "shares": 30, "price": 320.93, "executed_at": "2026-09-04T15:57:30-04:00"},
        ]
    )
    quotes = {"CIEN": {"open": 321.67, "last": 321.0, "prev_close": 322.0, "gap": -0.33, "gap_pct": -0.001}}

    positions = position_summaries(trades, quotes)

    assert len(positions) == 1
    pos = positions[0]
    assert pos["ticker"] == "CIEN"
    assert pos["shares"] == 30
    assert pos["closed"] is True
    assert pos["buy_time"] == "2026-09-04T10:09:39-04:00"
    assert pos["sell_time"] == "2026-09-04T15:57:30-04:00"
    assert pos["invested"] == round(30 * 320.73, 2)
    assert pos["pnl"] == round((320.93 - 320.73) * 30, 2)
    assert pos["day"] == "2026-09-04"


def test_position_summaries_computes_unrealized_pnl_for_open_position():
    trades = pd.DataFrame(
        [{"ticker": "AAPL", "side": "buy", "shares": 10, "price": 100.0, "executed_at": "2026-09-04T09:30:00-04:00"}]
    )
    quotes = {"AAPL": {"open": 99.0, "last": 105.0}}

    positions = position_summaries(trades, quotes)

    assert positions[0]["closed"] is False
    assert positions[0]["sell_time"] is None
    assert positions[0]["pnl"] == round((105.0 - 100.0) * 10, 2)


def test_position_summaries_handles_empty_input():
    assert position_summaries(pd.DataFrame(), {}) == []


def test_overnight_buy_sold_next_day_is_closed_not_an_empty_open():
    trades = pd.DataFrame(
        [
            {"ticker": "TBBK", "side": "buy", "shares": 200, "price": 51.24, "executed_at": "2026-09-09T10:00:00-05:00"},
            {"ticker": "TBBK", "side": "sell", "shares": 200, "price": 50.32, "executed_at": "2026-09-10T10:00:00-05:00"},
            {"ticker": "NVTS", "side": "buy", "shares": 900, "price": 11.78, "executed_at": "2026-09-11T09:40:00-05:00"},
        ]
    )
    quotes = {"NVTS": {"open": 11.29, "last": 11.50, "prev_close": 11.00}}

    positions = position_summaries(trades, quotes)
    by_ticker = {row["ticker"]: row for row in positions}

    assert by_ticker["TBBK"]["closed"] is True
    assert by_ticker["TBBK"]["shares"] == 200
    assert by_ticker["TBBK"]["day"] == "2026-09-10"
    assert by_ticker["NVTS"]["closed"] is False
    assert by_ticker["NVTS"]["shares"] == 900
    assert by_ticker["NVTS"]["day"] == "2026-09-11"


def test_position_summaries_accepts_mixed_utc_offsets():
    trades = pd.DataFrame(
        [
            {"ticker": "HOOD", "side": "buy", "shares": 50, "price": 121.88, "executed_at": "2026-09-04T10:08:10-04:00"},
            {"ticker": "FLY", "side": "buy", "shares": 500, "price": 21.36, "executed_at": "2026-09-11T09:40:00-05:00"},
        ]
    )

    positions = position_summaries(trades, {})

    days = {row["ticker"]: row["day"] for row in positions}
    assert days["HOOD"] == "2026-09-04"
    assert days["FLY"] == "2026-09-11"
