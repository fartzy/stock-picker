from stock_picker.storage.fee_store import Fee, FeeStore


def test_read_is_empty_before_any_append(tmp_path):
    assert FeeStore(data_dir=tmp_path).read() == []


def test_append_then_read_round_trips(tmp_path):
    store = FeeStore(data_dir=tmp_path)
    store.append(Fee(ticker="IONQ", day="2026-09-25", side="sell", amount=0.19, note="ticket"))
    rows = store.read()
    assert len(rows) == 1
    assert rows[0].ticker == "IONQ"
    assert rows[0].amount == 0.19


def test_append_is_idempotent(tmp_path):
    store = FeeStore(data_dir=tmp_path)
    fee = Fee(ticker="IONQ", day="2026-09-25", side="sell", amount=0.19)
    store.append(fee)
    store.append(fee)
    assert len(store.read()) == 1
