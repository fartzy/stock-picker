import numpy as np
import pandas as pd

from stock_picker.training.dataset import LABEL_COLUMN
from stock_picker.training.model import evaluate, feature_columns, train_lightgbm


def _make_learnable_frame(n, seed):
    rng = np.random.default_rng(seed)
    signal = rng.normal(size=n)
    noise = rng.normal(scale=0.005, size=n)
    label = 0.05 * np.sign(signal) + noise
    return pd.DataFrame({"signal": signal, "noise_feature": rng.normal(size=n), LABEL_COLUMN: label})


def test_train_lightgbm_learns_a_clear_signal():
    train_frame = _make_learnable_frame(400, seed=1)
    test_frame = _make_learnable_frame(200, seed=2)

    model = train_lightgbm(train_frame, params={"min_data_in_leaf": 10}, num_boost_round=50)
    metrics = evaluate(model, test_frame)

    assert metrics["directional_accuracy"] > 0.9


def test_feature_columns_excludes_metadata_and_label():
    frame = pd.DataFrame({"ticker": ["A"], "date": [1], "signal": [0.1], LABEL_COLUMN: [0.01]})

    assert feature_columns(frame) == ["signal"]


def test_feature_columns_also_excludes_pruned_features():
    frame = pd.DataFrame(
        {"ticker": ["A"], "date": [1], "signal": [0.1], "noise_feature": [0.2], LABEL_COLUMN: [0.01]}
    )

    assert feature_columns(frame, excluded_features={"noise_feature"}) == ["signal"]
