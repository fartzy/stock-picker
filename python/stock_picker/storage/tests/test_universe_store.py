from datetime import date

from stock_picker.storage.universe_store import UniverseStore


def test_sync_adds_new_tickers_as_active(tmp_path):
    store = UniverseStore(data_dir=tmp_path)

    registry = store.sync(
        {"AAPL": "market_cap", "ZZZ": "manual"}, as_of=date(2026, 1, 1)
    ).set_index("ticker")

    assert registry.loc["AAPL", "active"]
    assert registry.loc["AAPL", "source"] == "market_cap"
    assert registry.loc["AAPL", "first_seen"] == "2026-01-01"
    assert registry.loc["ZZZ", "source"] == "manual"


def test_sync_keeps_previously_seen_tickers_active_when_absent(tmp_path):
    store = UniverseStore(data_dir=tmp_path)
    store.sync({"AAPL": "market_cap", "MSFT": "market_cap"}, as_of=date(2026, 1, 1))

    registry = store.sync({"AAPL": "market_cap"}, as_of=date(2026, 1, 2)).set_index("ticker")

    assert registry.loc["AAPL", "active"]
    assert registry.loc["MSFT", "active"]
    assert registry.loc["MSFT", "last_seen"] == "2026-01-01"
    assert registry.loc["MSFT", "first_seen"] == "2026-01-01"


def test_sync_bumps_last_seen_on_reappearance(tmp_path):
    store = UniverseStore(data_dir=tmp_path)
    store.sync({"AAPL": "market_cap"}, as_of=date(2026, 1, 1))
    store.sync({}, as_of=date(2026, 1, 2))

    registry = store.sync({"AAPL": "market_cap"}, as_of=date(2026, 1, 3)).set_index("ticker")

    assert registry.loc["AAPL", "active"]
    assert registry.loc["AAPL", "last_seen"] == "2026-01-03"
    assert registry.loc["AAPL", "first_seen"] == "2026-01-01"


def test_active_tickers_includes_ones_dropped_from_a_later_sync(tmp_path):
    store = UniverseStore(data_dir=tmp_path)
    store.sync({"AAPL": "market_cap", "MSFT": "market_cap"}, as_of=date(2026, 1, 1))
    store.sync({"AAPL": "market_cap"}, as_of=date(2026, 1, 2))

    assert set(store.active_tickers()) == {"AAPL", "MSFT"}


def test_all_tickers_includes_full_history(tmp_path):
    store = UniverseStore(data_dir=tmp_path)
    store.sync({"AAPL": "market_cap", "MSFT": "market_cap"}, as_of=date(2026, 1, 1))
    store.sync({"AAPL": "market_cap"}, as_of=date(2026, 1, 2))

    all_tickers = set(store.all_tickers()["ticker"])
    assert all_tickers == {"AAPL", "MSFT"}
