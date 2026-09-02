import pandas as pd

from stock_picker.storage.price_store import PriceStore


def test_write_then_read_round_trips(tmp_path):
    store = PriceStore(data_dir=tmp_path)
    history = pd.DataFrame(
        {"Close": [1.0, 2.0]},
        index=pd.date_range("2026-01-01", periods=2),
    )

    store.write("AAPL", history)
    result = store.read("AAPL")

    pd.testing.assert_frame_equal(result, history, check_freq=False)
