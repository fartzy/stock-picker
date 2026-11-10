import numpy as np
import pandas as pd
import pytest

from stock_picker.features.structure import (
    CLUSTER_GAP_COLUMN,
    CLUSTER_ID_COLUMN,
    build_structure_features,
    fill_cluster_overnight_gaps,
    lagged_cluster_overnight_gaps,
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
    assert rows[0].row.iloc[0][CLUSTER_GAP_COLUMN] == pytest.approx(0.05)
    assert rows[1].row.iloc[0][CLUSTER_GAP_COLUMN] == pytest.approx(0.10)


def test_historical_cluster_gap_uses_prior_membership_and_matches_live():
    dates = pd.bdate_range("2026-01-05", periods=3)
    histories = {
        ticker: pd.DataFrame({"Open": [100, 100, opening], "Close": [100, 100, 100]}, index=dates)
        for ticker, opening in {"A": 110, "B": 105, "C": 98}.items()
    }
    structures = {
        "A": pd.DataFrame({CLUSTER_ID_COLUMN: [0, 0, 1]}, index=dates),
        "B": pd.DataFrame({CLUSTER_ID_COLUMN: [0, 0, 2]}, index=dates),
        "C": pd.DataFrame({CLUSTER_ID_COLUMN: [1, 1, 0]}, index=dates),
    }
    actual = lagged_cluster_overnight_gaps(histories, structures)
    assert actual["A"].iloc[-1] == pytest.approx(0.05)
    assert actual["B"].iloc[-1] == pytest.approx(0.10)
    assert pd.isna(actual["C"].iloc[-1])
    quotes = {ticker: {"open": table.Open.iloc[-1], "prev_close": table.Close.iloc[-2]} for ticker, table in histories.items()}
    rows = [LiveRow(ticker, row=pd.DataFrame({CLUSTER_ID_COLUMN: [table.iloc[-2, 0]], CLUSTER_GAP_COLUMN: [999.0]})) for ticker, table in structures.items()]
    fill_cluster_overnight_gaps(rows, quotes)
    for item in rows:
        assert item.row.iloc[0][CLUSTER_GAP_COLUMN] == pytest.approx(actual[item.ticker].iloc[-1], nan_ok=True)


def test_lagged_membership_preserves_missing_labels_and_ticker_calendar():
    dates = pd.bdate_range("2026-01-05", periods=3)
    structures = {
        "A": pd.DataFrame({CLUSTER_ID_COLUMN: [1, np.nan]}, index=dates[[0, 2]]),
        "B": pd.DataFrame({CLUSTER_ID_COLUMN: [1, 1, 7]}, index=dates),
        "C": pd.DataFrame({CLUSTER_ID_COLUMN: [1, np.nan, 1]}, index=dates),
    }
    histories = {
        ticker: pd.DataFrame({"Open": 110.0, "Close": 100.0}, index=table.index)
        for ticker, table in structures.items()
    }
    actual = lagged_cluster_overnight_gaps(histories, structures)
    assert actual["A"].index.equals(dates[[0, 2]])
    assert actual["A"].iloc[-1] == pytest.approx(0.10)
    assert actual["B"].iloc[-1] == pytest.approx(0.10)
    assert pd.isna(actual["C"].iloc[-1])
    assert pd.isna(actual["A"].iloc[0])


def test_current_close_derived_clusters_cannot_change_current_opening_gap():
    dates = pd.bdate_range("2026-01-05", periods=2)
    rng = np.random.default_rng(42)
    columns = ["return_1d", "return_5d", "return_20d", "rsi_14", "volatility_20d", "atr_14"]
    source = rng.normal(size=(48, len(columns)))
    features = {f"T{i:02}": pd.DataFrame([source[i], source[i]], columns=columns, index=dates) for i in range(48)}
    histories = {ticker: pd.DataFrame({"Open": [100, 100 + i], "Close": [100, 100]}, index=dates) for i, ticker in enumerate(features)}
    before = build_structure_features(histories, features)
    changed = {ticker: table.copy() for ticker, table in features.items()}
    for i, ticker in enumerate(changed):
        changed[ticker].loc[dates[-1], columns] = source[47 - i]
    after = build_structure_features(histories, changed)
    for ticker in before:
        assert before[ticker][CLUSTER_GAP_COLUMN].iloc[-1] == pytest.approx(after[ticker][CLUSTER_GAP_COLUMN].iloc[-1], nan_ok=True)
