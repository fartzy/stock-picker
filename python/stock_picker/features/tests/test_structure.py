import pandas as pd

from stock_picker.features.structure import (
    CLUSTER_GAP_COLUMN,
    CLUSTER_ID_COLUMN,
    build_structure_features,
    fill_cluster_overnight_gaps,
)
from stock_picker.training.live_rows import LiveRow


def _ticker_frame(n=30, seed=0):
    rng = pd.RangeIndex(n)
    index = pd.bdate_range("2026-01-05", periods=n)
    base = 10.0 + seed
    close = pd.Series([base + 0.01 * i for i in rng], index=index)
    return pd.DataFrame(
        {
            "Open": close.shift(1).fillna(base),
            "High": close,
            "Low": close,
            "Close": close,
            "Volume": 1.0,
            "return_1d": close.pct_change(),
            "return_5d": close.pct_change(5),
            "return_20d": close.pct_change(20),
            "rsi_14": 50.0 + seed,
            "volatility_20d": 0.2 + 0.001 * seed,
            "atr_14": 0.5,
        },
        index=index,
    )


def test_pca_and_cluster_columns_appear_on_each_ticker():
    histories = {}
    features = {}
    for i, name in enumerate(["AAA", "BBB", "CCC", "DDD"] * 12):
        ticker = f"{name}{i}"
        table = _ticker_frame(seed=i)
        histories[ticker] = table[["Open", "High", "Low", "Close", "Volume"]]
        features[ticker] = table
    extra = build_structure_features(histories, features)
    sample = extra["AAA0"]
    assert "pca_1" in sample.columns
    assert CLUSTER_ID_COLUMN in sample.columns
    assert CLUSTER_GAP_COLUMN in sample.columns
    last = sample.dropna(how="all")
    assert not last.empty
    assert pd.notna(last.iloc[-1][CLUSTER_ID_COLUMN])


def test_fill_cluster_gap_uses_peer_opens_not_self():
    row_a = pd.DataFrame({"cluster_id": [1.0], "cluster_overnight_gap": [pd.NA]})
    row_b = pd.DataFrame({"cluster_id": [1.0], "cluster_overnight_gap": [pd.NA]})
    rows = [
        LiveRow("AAA", open_price=11.0, row=row_a),
        LiveRow("BBB", open_price=10.5, row=row_b),
    ]
    quotes = {
        "AAA": {"open": 11.0, "prev_close": 10.0},
        "BBB": {"open": 10.5, "prev_close": 10.0},
    }
    fill_cluster_overnight_gaps(rows, quotes)
    assert rows[0].row.iloc[0][CLUSTER_GAP_COLUMN] == 0.05
    assert rows[1].row.iloc[0][CLUSTER_GAP_COLUMN] == 0.10
