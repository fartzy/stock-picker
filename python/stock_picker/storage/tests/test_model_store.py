import lightgbm as lgb
import numpy as np

from stock_picker.storage.model_store import ModelStore


def test_write_then_read_round_trips(tmp_path):
    x = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([1.0, 2.0, 3.0, 4.0])
    model = lgb.train(
        {"objective": "regression", "verbosity": -1, "min_data_in_leaf": 1},
        lgb.Dataset(x, label=y),
        num_boost_round=2,
    )

    store = ModelStore(data_dir=tmp_path)
    store.write("test_model", model)
    loaded = store.read("test_model")

    assert np.allclose(model.predict(x), loaded.predict(x))
