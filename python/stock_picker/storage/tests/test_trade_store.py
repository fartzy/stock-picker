from stock_picker.storage.trade_store import Trade, TradeStore


def test_read_returns_empty_frame_with_columns_before_any_append(tmp_path):
    store = TradeStore(data_dir=tmp_path)

    trades = store.read()

    assert trades.empty
    assert list(trades.columns) == ["ticker", "side", "shares", "price", "executed_at"]


def test_append_then_read_round_trips_a_trade(tmp_path):
    store = TradeStore(data_dir=tmp_path)

    store.append(Trade(ticker="HOOD", side="buy", shares=50, price=121.88, executed_at="2026-09-04T10:08:10-04:00"))
    trades = store.read()

    assert len(trades) == 1
    row = trades.iloc[0]
    assert row["ticker"] == "HOOD"
    assert row["side"] == "buy"
    assert row["shares"] == 50
    assert row["price"] == 121.88
    assert row["executed_at"] == "2026-09-04T10:08:10-04:00"


def test_append_accumulates_across_calls(tmp_path):
    store = TradeStore(data_dir=tmp_path)

    store.append(Trade(ticker="HOOD", side="buy", shares=50, price=121.88, executed_at="2026-09-04T10:08:10-04:00"))
    store.append(Trade(ticker="CIEN", side="buy", shares=30, price=320.73, executed_at="2026-09-04T10:09:39-04:00"))
    trades = store.read()

    assert list(trades["ticker"]) == ["HOOD", "CIEN"]
