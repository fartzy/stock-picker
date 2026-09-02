import pandas as pd

from stock_picker.storage.feature_store import FeatureStore


def test_write_then_read_round_trips(tmp_path):
    store = FeatureStore(data_dir=tmp_path)
    features = pd.DataFrame(
        {"return_1d": [0.01, -0.02], "rsi_14": [55.0, 48.0]},
        index=pd.date_range("2026-01-01", periods=2),
    )

    store.write("AAPL", features)
    result = store.read("AAPL")

    pd.testing.assert_frame_equal(result, features, check_freq=False)
