import sqlite3

import pytest

from stock_picker.storage.trade_store import ParquetTradeLog, Trade, TradeStore


def test_read_returns_empty_frame_with_columns_before_any_append(tmp_path):
    store = TradeStore(data_dir=tmp_path)

    trades = store.read()

    assert trades.empty
    assert list(trades.columns) == ["ticker", "side", "shares", "price", "executed_at", "manual_fee"]


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


def test_append_also_writes_a_readable_csv(tmp_path):
    store = TradeStore(data_dir=tmp_path)

    store.append(Trade(ticker="HOOD", side="buy", shares=50, price=121.88, executed_at="2026-09-04T10:08:10-04:00"))

    csv_path = tmp_path / "trades.csv"
    assert csv_path.is_file()
    text = csv_path.read_text()
    assert "ticker,side,shares,price,executed_at,manual_fee" in text
    assert "HOOD,buy," in text
    assert "121.88" in text
    assert "2026-09-04T10:08:10-04:00" in text


def test_duplicate_append_is_ignored(tmp_path):
    store = TradeStore(data_dir=tmp_path)
    trade = Trade(ticker="HOOD", side="buy", shares=50, price=121.88, executed_at="2026-09-04T10:08:10-04:00")

    store.append(trade)
    store.append(trade)

    assert len(store.read()) == 1


def test_manual_fee_is_atomic_with_fill_and_conflicting_retry_is_rejected(tmp_path):
    store = TradeStore(data_dir=tmp_path)
    fill = Trade("HOOD", "buy", 50, 121.88, "2026-09-04T10:08:10-04:00", 0.17)
    with sqlite3.connect(tmp_path / "stockpicker.db") as conn:
        conn.execute(
            "CREATE TRIGGER fail_fill BEFORE INSERT ON trades "
            "BEGIN SELECT RAISE(ABORT, 'simulated write failure'); END"
        )

    with pytest.raises(sqlite3.IntegrityError, match="simulated write failure"):
        store.append(fill)
    assert store.read().empty

    with sqlite3.connect(tmp_path / "stockpicker.db") as conn:
        conn.execute("DROP TRIGGER fail_fill")
    store.append(fill)
    store.append(fill)
    assert store.read()["manual_fee"].tolist() == [0.17]

    with pytest.raises(ValueError, match="different manual fee"):
        store.append(Trade("HOOD", "buy", 50, 121.88, fill.executed_at, 0.25))
    assert store.read()["manual_fee"].tolist() == [0.17]


def test_existing_sqlite_schema_migrates_without_changing_fills(tmp_path):
    with sqlite3.connect(tmp_path / "stockpicker.db") as conn:
        conn.execute(
            "CREATE TABLE trades (id INTEGER PRIMARY KEY AUTOINCREMENT, ticker TEXT NOT NULL, "
            "side TEXT NOT NULL, shares REAL NOT NULL, price REAL NOT NULL, "
            "executed_at TEXT NOT NULL, UNIQUE (ticker, side, shares, price, executed_at))"
        )
        conn.execute(
            "INSERT INTO trades (ticker, side, shares, price, executed_at) VALUES (?, ?, ?, ?, ?)",
            ("OLD", "buy", 5, 10, "2026-09-03T10:00:00-04:00"),
        )

    store = TradeStore(data_dir=tmp_path)
    assert store.read()[["ticker", "manual_fee"]].to_dict(orient="records") == [
        {"ticker": "OLD", "manual_fee": 0.0}
    ]
    store.append(Trade("NEW", "buy", 2, 12, "2026-09-04T10:00:00-04:00", 0.05))
    assert store.read()["manual_fee"].tolist() == [0.0, 0.05]


def test_sqlite_imports_existing_parquet_once(tmp_path):
    parquet = ParquetTradeLog(tmp_path)
    parquet.append(Trade(ticker="HOOD", side="buy", shares=50, price=121.88, executed_at="2026-09-04T10:08:10-04:00"))

    store = TradeStore(data_dir=tmp_path)

    trades = store.read()
    assert len(trades) == 1
    assert trades.iloc[0]["ticker"] == "HOOD"


def test_parquet_backend_still_round_trips_when_injected(tmp_path):
    store = TradeStore(data_dir=tmp_path, backend=ParquetTradeLog(tmp_path))

    store.append(Trade(ticker="HOOD", side="buy", shares=50, price=121.88, executed_at="2026-09-04T10:08:10-04:00"))

    assert (tmp_path / "trades.parquet").is_file()
    assert store.read().iloc[0]["ticker"] == "HOOD"
