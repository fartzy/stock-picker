"""Reproduce the saved-open valuation regression with frozen market evidence.

Run explicitly (three failures are expected on the current production code):
  PYTHONPATH=python .venv/bin/python -m pytest -q <this-file>

For the historical control, set PAPER_BOOK_SOURCE to book.py exported from
37357d2465^. All six cases pass there. No real scan, price, or paper stores
are written: every store is injected under pytest's temporary directory.
"""

from copy import deepcopy
from datetime import date
import importlib.util
import json
import os
from pathlib import Path

import pandas as pd
import pytest

from stock_picker.paper import book as current_book
from stock_picker.storage.paper_book_store import PaperBookStore
from stock_picker.storage.price_store import PriceStore
from stock_picker.storage.scan_store import ScanStore

ROOT = Path(__file__).resolve().parent
CASES = [("2026-09-21", "PS"), ("2026-09-22", "HHH"), ("2026-09-23", "TRLV")]


def implementation():
    source = os.environ.get("PAPER_BOOK_SOURCE")
    if not source:
        return current_book
    spec = importlib.util.spec_from_file_location("historical_paper_book", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("day,ticker", CASES)
@pytest.mark.parametrize("saved_open", ["stale", "verified"])
def test_historical_entry_cannot_be_below_the_entire_sessions_range(
    tmp_path, day, ticker, saved_open,
):
    daily = next(
        row["payload"] for row in json.loads((ROOT / "daily_price_snapshot.json").read_text())
        if row["date"] == day and row["ticker"] == ticker
    )
    original = json.loads((ROOT / "rank_scan_snapshot.json").read_text())[day]
    signal = deepcopy(next(row for row in original["signals"] if row["ticker"] == ticker))
    assert daily["status"] == "OK" and daily["from"] == day and daily["symbol"] == ticker
    assert signal["open_price"] < daily["low"], "Fixture must reproduce an impossible entry"
    if saved_open == "verified":
        signal["open_price"] = daily["open"]

    scans = ScanStore(data_dir=tmp_path / "scans")
    payload = {"as_of": day, "kind": "rank", "signals": [signal]}
    scans.write(day, "rank", payload)
    prices = PriceStore(data_dir=tmp_path / "prices")
    prices.write(ticker, pd.DataFrame({
        name: [daily[name.lower()]] for name in ("Open", "High", "Low", "Close", "Volume")
    }, index=pd.DatetimeIndex([day])))
    paper = PaperBookStore(data_dir=tmp_path / "paper")
    picks = implementation().rebuild_paper_book(
        completed_through=date.fromisoformat(day), scan_store=scans, price_store=prices,
        paper_store=paper, quote_fetcher=lambda *_: {}, news_fetcher=lambda *_: {},
    )

    assert len(picks) == 1
    assert scans.read(day, "rank") == payload, "Valuation must not rewrite the saved signal"
    assert paper.read()[0].open_price == picks[0].open_price
    assert daily["low"] <= picks[0].open_price <= daily["high"], (
        f"{day} {ticker}: book uses ${picks[0].open_price:.4f}; "
        f"the entire session traded ${daily['low']:.4f}–${daily['high']:.4f}, "
        f"with a verified open of ${daily['open']:.4f}"
    )
    assert picks[0].open_price == pytest.approx(daily["open"])
    assert picks[0].session_return == pytest.approx(daily["close"] / daily["open"] - 1)
