import numpy as np
import pandas as pd

from stock_picker.storage.model_store import ModelStore
from stock_picker.training.dataset import LABEL_COLUMN
from stock_picker.training.ensemble import ModelSpec, predict_ensemble, train_ensemble


def test_write_then_read_round_trips_a_mixed_ensemble(tmp_path):
    train_frame = pd.DataFrame(
        {"x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], LABEL_COLUMN: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]}
    )
    ensemble = train_ensemble(
        train_frame,
        [
            ModelSpec("lightgbm", params={"min_data_in_leaf": 1}),
            ModelSpec("random_forest", params={"n_estimators": 5, "min_samples_leaf": 1}),
        ],
    )

    store = ModelStore(data_dir=tmp_path)
    store.write("test_model", ensemble)
    loaded = store.read("test_model")

    assert np.allclose(predict_ensemble(ensemble, train_frame), predict_ensemble(loaded, train_frame))


def test_exists_reflects_whether_a_model_has_been_written(tmp_path):
    store = ModelStore(data_dir=tmp_path)

    assert store.exists("never_written") is False

    store.write("never_written", {"anything": "picklable"})

    assert store.exists("never_written") is True
