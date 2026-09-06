import numpy as np
import pandas as pd
import pytest

from stock_picker.training.dataset import LABEL_COLUMN
from stock_picker.training.importance import gain_importance
from stock_picker.training.model import train_lightgbm


def _make_learnable_frame(n, seed):
    rng = np.random.default_rng(seed)
    signal = rng.normal(size=n)
    label = 0.05 * np.sign(signal) + rng.normal(scale=0.005, size=n)
    return pd.DataFrame(
        {"signal": signal, "noise_feature": rng.normal(size=n), LABEL_COLUMN: label}
    )


def _trained_model():
    train_frame = _make_learnable_frame(400, seed=1)
    return train_lightgbm(train_frame, params={"min_data_in_leaf": 10}, num_boost_round=50)


def test_gain_importance_sums_to_roughly_100():
    importance = gain_importance(_trained_model())

    assert set(importance) == {"signal", "noise_feature"}
    assert sum(importance.values()) == pytest.approx(100, abs=1.0)


def test_gain_importance_ranks_the_real_signal_above_pure_noise():
    importance = gain_importance(_trained_model())

    assert importance["signal"] > importance["noise_feature"]
