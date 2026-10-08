import pandas as pd
import pytest

from stock_picker.features.trades import position_summaries, trade_history
from stock_picker.storage.trade_store import Trade, TradeStore


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


def test_peak_working_does_not_double_count_recycled_cash():
    from stock_picker.features.trades import peak_working_by_day

    trades = pd.DataFrame(
        [
            {"ticker": "TTAN", "side": "buy", "shares": 465, "price": 58.465, "executed_at": "2026-09-16T10:02:00-04:00"},
            {"ticker": "TTAN", "side": "sell", "shares": 930, "price": 58.465, "executed_at": "2026-09-16T10:02:00-04:00"},
            {"ticker": "XNDU", "side": "buy", "shares": 1300, "price": 7.81, "executed_at": "2026-09-16T10:01:00-04:00"},
            {"ticker": "XNDU", "side": "buy", "shares": 1350, "price": 7.65, "executed_at": "2026-09-16T11:00:00-04:00"},
        ]
    )

    peaks = peak_working_by_day(trades)

    # XNDU 10k then TTAN 27k then sell TTAN -- peak is ~37k, not 10k+27k+10k.
    assert peaks["2026-09-16"] < 40000
    assert peaks["2026-09-16"] > 25000


def test_time_weighted_working_is_less_than_a_lunch_spike():
    from stock_picker.features.trades import time_weighted_working_by_day

    trades = pd.DataFrame(
        [
            {"ticker": "XNDU", "side": "buy", "shares": 1300, "price": 10.0, "executed_at": "2026-09-16T09:31:00-04:00"},
            {"ticker": "XNDU", "side": "sell", "shares": 1300, "price": 10.0, "executed_at": "2026-09-16T09:41:00-04:00"},
        ]
    )
    averages = time_weighted_working_by_day(trades)
    # 10 minutes of $13k then flat: in-market average is the $13k, not a
    # 6.5h smear and not diluted by the empty afternoon.
    assert averages["2026-09-16"] > 12000
    assert averages["2026-09-16"] < 14000


def test_full_exit_does_not_carry_fractional_cost_into_next_session():
    from stock_picker.features.trades import time_weighted_working_by_day

    trades = pd.DataFrame(
        [
            {"ticker": "OLD", "side": "buy", "shares": 1, "price": 0.492,
             "executed_at": "2026-10-01T09:31:00-04:00"},
            {"ticker": "OLD", "side": "buy", "shares": 2, "price": 0.509,
             "executed_at": "2026-10-01T09:31:00-04:00"},
            {"ticker": "OLD", "side": "sell", "shares": 3, "price": 0.50,
             "executed_at": "2026-10-01T09:32:00-04:00"},
            {"ticker": "NEW", "side": "buy", "shares": 1, "price": 40000.0,
             "executed_at": "2026-10-02T09:31:00-04:00"},
            {"ticker": "NEW", "side": "sell", "shares": 1, "price": 40001.0,
             "executed_at": "2026-10-02T09:32:00-04:00"},
        ]
    )

    assert time_weighted_working_by_day(trades)["2026-10-02"] == 40000.0


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


def test_same_timestamp_buy_then_sell_closes_the_full_position():
    trades = pd.DataFrame(
        [
            {"ticker": "TTAN", "side": "buy", "shares": 465, "price": 56.80, "executed_at": "2026-09-11T10:30:00-05:00"},
            {"ticker": "TTAN", "side": "sell", "shares": 930, "price": 58.465, "executed_at": "2026-09-16T10:02:00-04:00"},
            {"ticker": "TTAN", "side": "buy", "shares": 465, "price": 58.465, "executed_at": "2026-09-16T10:02:00-04:00"},
        ]
    )

    positions = position_summaries(trades, {})

    assert len(positions) == 1
    assert positions[0]["closed"] is True
    assert positions[0]["shares"] == 930
    assert positions[0]["ticker"] == "TTAN"


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


def test_manual_entry_fee_follows_overnight_partial_exits(tmp_path):
    store = TradeStore(data_dir=tmp_path)
    store.append(Trade("AAA", "buy", 10, 10, "2026-09-04T10:00:00-04:00", 1.0))
    store.append(Trade("AAA", "sell", 4, 12, "2026-09-05T10:00:00-04:00", 0.2))

    first = position_summaries(store.read(), {"AAA": {"last": 11}})
    closed = next(row for row in first if row["closed"])
    open_row = next(row for row in first if not row["closed"])
    assert (closed["day"], closed["pnl"]) == ("2026-09-05", 8.0)
    assert closed["manual_fee"] == pytest.approx(0.6)
    assert open_row["shares"] == 6.0
    assert open_row["manual_fee"] == pytest.approx(0.6)

    store.append(Trade("AAA", "sell", 6, 11, "2026-09-06T10:00:00-04:00", 0.3))
    exits = [row for row in position_summaries(store.read(), {}) if row["closed"]]
    assert [row["day"] for row in exits] == ["2026-09-06", "2026-09-05"]
    assert [row["manual_fee"] for row in exits] == pytest.approx([0.9, 0.6])
    assert round(sum(row["pnl"] - row["manual_fee"] for row in exits), 2) == 12.5


def test_manual_fees_follow_average_cost_buy_allocations(tmp_path):
    store = TradeStore(data_dir=tmp_path)
    store.append(Trade("AAA", "buy", 10, 10, "2026-09-04T10:00:00-04:00", 1.0))
    store.append(Trade("AAA", "buy", 10, 20, "2026-09-04T11:00:00-04:00", 3.0))
    store.append(Trade("AAA", "sell", 5, 25, "2026-09-05T10:00:00-04:00", 0.5))

    rows = position_summaries(store.read(), {"AAA": {"last": 25}})
    closed = next(row for row in rows if row["closed"])
    open_row = next(row for row in rows if not row["closed"])
    assert closed["pnl"] == 50.0  # Five shares at the $15 average basis.
    assert [buy["shares"] for buy in closed["_buy_allocations"]] == [2.5, 2.5]
    assert closed["manual_fee"] == pytest.approx(1.5)
    assert open_row["manual_fee"] == pytest.approx(3.0)
