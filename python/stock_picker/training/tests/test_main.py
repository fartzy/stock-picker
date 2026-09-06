import pandas as pd

from stock_picker.training.main import _date_range


def test_date_range_returns_min_and_max_date_as_strings():
    frame = pd.DataFrame({"date": pd.to_datetime(["2026-01-15", "2026-01-01", "2026-01-31"])})

    assert _date_range(frame) == ("2026-01-01", "2026-01-31")


def test_date_range_handles_a_single_row():
    frame = pd.DataFrame({"date": pd.to_datetime(["2026-01-15"])})

    assert _date_range(frame) == ("2026-01-15", "2026-01-15")
