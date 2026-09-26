from stock_picker.storage.ticker_blacklist_store import TickerBlacklistStore


def test_read_is_empty_before_any_add(tmp_path):
    assert TickerBlacklistStore(data_dir=tmp_path).read() == set()


def test_add_then_read_contains_the_ticker(tmp_path):
    store = TickerBlacklistStore(data_dir=tmp_path)
    store.add("tdth")
    assert store.read() == {"TDTH"}


def test_add_is_idempotent(tmp_path):
    store = TickerBlacklistStore(data_dir=tmp_path)
    store.add("TDTH")
    store.add("TDTH")
    assert store.read() == {"TDTH"}


def test_remove_unblocks(tmp_path):
    store = TickerBlacklistStore(data_dir=tmp_path)
    store.add("TDTH")
    store.add("XNDU")
    store.remove("TDTH")
    assert store.read() == {"XNDU"}
