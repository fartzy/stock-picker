import numpy as np
import pandas as pd

from stock_picker.training.feature_family_search import _evaluate, variants


def test_variants_hold_baseline_fixed_and_add_only_requested_family():
    columns = ["old_feature", "volatility_32d", "market_advance_fraction", "path_cluster_distance"]

    full = variants(columns, "full")
    recent = variants(columns, "recent")

    assert full["baseline"] == {"old_feature"}
    assert full["volatility"] == {"old_feature", "volatility_32d"}
    assert full["volatility_shape"] == {"old_feature", "volatility_32d"}
    assert full["volatility_direction"] == {"old_feature"}
    assert recent["breadth"] == {"old_feature", "market_advance_fraction"}
    assert recent["trajectory"] == {"old_feature", "path_cluster_distance"}
    assert recent["breadth_price"] == {"old_feature", "market_advance_fraction"}
    assert recent["trajectory_shape"] == {"old_feature", "path_cluster_distance"}
    assert recent["all"] == set(columns)


def test_rank_screen_does_not_report_uninterpretable_regression_metrics():
    frame = pd.DataFrame({
        "date": pd.to_datetime(["2026-01-01", "2026-01-01", "2026-01-01"]),
        "label": [0.03, -0.01, 0.02],
    })
    from stock_picker.training.dataset import LABEL_COLUMN
    frame = frame.rename(columns={"label": LABEL_COLUMN})

    metrics = _evaluate(frame, np.array([0.9, 0.1, 0.7]), "rank")

    assert "mae" not in metrics
    assert "directional_accuracy" not in metrics
    assert metrics["top5_hit_rate"] == 2 / 3
