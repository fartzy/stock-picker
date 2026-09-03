import numpy as np
import pandas as pd

from stock_picker.training.dataset import LABEL_COLUMN
from stock_picker.training.train import run_walk_forward


def _make_pooled_dataset(n_dates=20, n_tickers=2):
    dates = pd.date_range("2026-01-01", periods=n_dates, freq="B")
    rng = np.random.default_rng(0)
    frames = []
    for i in range(n_tickers):
        signal = rng.normal(size=n_dates)
        label = 0.02 * np.sign(signal)
        frames.append(
            pd.DataFrame(
                {"date": dates, "ticker": f"T{i}", "signal": signal, LABEL_COLUMN: label}
            )
        )
    return pd.concat(frames, ignore_index=True)


def test_run_walk_forward_returns_one_result_per_fold(tmp_path):
    pooled = _make_pooled_dataset()

    results = run_walk_forward(
        pooled, n_splits=3, params={"min_data_in_leaf": 2}, tracking_dir=tmp_path
    )

    assert len(results) == 3
    assert results[-1]["train_rows"] > results[0]["train_rows"]
    for result in results:
        assert "directional_accuracy" in result["metrics"]
