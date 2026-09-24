import pandas as pd

from stock_picker.training.dataset import LABEL_COLUMN
from stock_picker.training.story_prune import (
    columns_to_prune,
    experiment_columns_present,
    rank_experiment_columns,
)


def test_experiment_columns_present_keeps_only_named_stories():
    assert experiment_columns_present(
        ["return_5d", "story_week_run_open5_seasonality", "week_run_open3_seasonality"]
    ) == ["week_run_open3_seasonality", "story_week_run_open5_seasonality"]


def test_rank_drops_only_when_rf_and_ridge_agree_it_is_weak():
    n = 400
    signal = pd.Series(range(n), dtype="float64")
    rng = pd.Series(range(n), dtype="float64")
    noise = (rng * 17 % 13).astype("float64")
    frame = pd.DataFrame(
        {
            "week_run_open3_seasonality": signal,
            "story_seq2_open5_seasonality": noise,
            "story_seq2_loc3_seasonality": noise + 1.0,
            "story_seq4_open5_seasonality": -noise,
            "story_seq4_loc3_seasonality": noise * 0.01,
            LABEL_COLUMN: 0.01 * signal,
        }
    )
    columns = [c for c in frame.columns if c != LABEL_COLUMN]
    table = rank_experiment_columns(frame, columns)
    drops = columns_to_prune(table)
    assert "week_run_open3_seasonality" not in drops
    assert "story_seq2_open5_seasonality" in drops
