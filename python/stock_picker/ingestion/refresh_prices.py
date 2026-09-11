"""Refresh OHLCV for the already-active universe -- no recap / market-cap rebuild."""

from __future__ import annotations

from stock_picker.ingestion.session import completed_sessions, last_completed_session_date
from stock_picker.ingestion.yfinance_client import download_price_history
from stock_picker.storage.price_store import PriceStore
from stock_picker.storage.universe_store import UniverseStore


def main() -> None:
    tickers = UniverseStore().active_tickers()
    print(f"refreshing prices for {len(tickers)} tickers", flush=True)
    history = download_price_history(tickers, period="1y", interval="1d")
    print(f"downloaded {len(history)}", flush=True)
    store = PriceStore()
    cutoff = last_completed_session_date()
    written = 0
    for ticker, frame in history.items():
        finished = completed_sessions(frame)
        if finished.empty:
            continue
        store.write(ticker, finished)
        written += 1
    print(f"wrote {written} through {cutoff}", flush=True)
    sample = store.read("AAPL")
    print(f"AAPL last {sample.index[-1].date()} n={len(sample)}", flush=True)


if __name__ == "__main__":
    main()
