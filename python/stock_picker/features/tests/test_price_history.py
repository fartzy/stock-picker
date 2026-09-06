import pandas as pd

from stock_picker.features.price_history import price_series


def test_price_series_sorts_ascending_and_shapes_each_row():
    index = pd.to_datetime(["2026-01-03", "2026-01-02", "2026-01-01"])
    history = pd.DataFrame(
        {
            "Open": [12.0, 11.0, 10.0],
            "High": [12.5, 11.5, 10.5],
            "Low": [11.5, 10.5, 9.5],
            "Close": [12.2, 11.2, 10.2],
            "Volume": [300.0, 200.0, 100.0],
        },
        index=index,
    )

    rows = price_series(history)

    assert [row["date"][:10] for row in rows] == ["2026-01-01", "2026-01-02", "2026-01-03"]
    assert rows[0] == {
        "date": rows[0]["date"],
        "open": 10.0,
        "high": 10.5,
        "low": 9.5,
        "close": 10.2,
        "volume": 100.0,
    }


def test_price_series_returns_empty_list_for_empty_history():
    history = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])

    assert price_series(history) == []
