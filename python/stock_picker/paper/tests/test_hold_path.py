import pandas as pd

from stock_picker.paper.hold_path import later_sessions
from stock_picker.storage.price_store import PriceStore


def test_later_sessions_start_the_day_after_the_pick(tmp_path):
    store = PriceStore(data_dir=tmp_path)
    store.write("ACHR", pd.DataFrame(
        {
            "Open": [4.32, 4.01, 4.30],
            "High": [4.37, 4.29, 4.31],
            "Low": [4.06, 3.97, 4.01],
            "Close": [4.07, 4.27, 4.07],
        },
        index=pd.to_datetime(["2026-10-14", "2026-10-15", "2026-10-16"]),
    ))

    rows = later_sessions("ACHR", pd.Timestamp("2026-10-14").date(), 5, store)

    assert [row["date"] for row in rows] == ["2026-10-15", "2026-10-16"]
    assert rows[0]["high"] == 4.29
    assert rows[0]["low"] == 3.97


def test_later_sessions_missing_ticker_is_empty(tmp_path):
    assert later_sessions("NOPE", pd.Timestamp("2026-10-14").date(), 2, PriceStore(data_dir=tmp_path)) == []
