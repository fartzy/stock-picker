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


def test_invalid_open_is_missing_but_valid_close_and_caller_are_preserved(tmp_path):
    store = PriceStore(data_dir=tmp_path)
    history = pd.DataFrame(
        {
            "Open": [0.0, float("inf"), -1.0, 10.0],
            "High": [11.0] * 4,
            "Low": [9.0] * 4,
            "Close": [10.5] * 4,
        },
        index=pd.date_range("2026-10-01", periods=4),
    )

    store.write("BAD", history)
    actual = store.read("BAD")

    assert actual["Open"].iloc[:3].isna().all()
    assert actual["Open"].iloc[3] == 10.0
    assert (actual["Close"] == 10.5).all()
    assert history["Open"].iloc[0] == 0.0
