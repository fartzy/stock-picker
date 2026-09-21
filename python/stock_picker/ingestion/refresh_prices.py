"""Refresh OHLCV for the already-active universe -- no recap / market-cap rebuild."""

from __future__ import annotations

from stock_picker.ingestion.session import completed_sessions, last_completed_session_date
from stock_picker.ingestion.yfinance_client import download_price_history
from stock_picker.log import get_logger
from stock_picker.storage.price_store import PriceStore
from stock_picker.storage.universe_store import UniverseStore

logger = get_logger(__name__)


def main() -> None:
    tickers = UniverseStore().active_tickers()
    logger.info("refreshing prices for %s tickers", len(tickers))
    history = download_price_history(tickers, period="1y", interval="1d")
    logger.info("downloaded %s", len(history))
    store = PriceStore()
    cutoff = last_completed_session_date()
    written = 0
    for ticker, frame in history.items():
        finished = completed_sessions(frame)
        if finished.empty:
            continue
        store.write(ticker, finished)
        written += 1
    logger.info("wrote %s through %s", written, cutoff)
    sample = store.read("AAPL")
    logger.info("AAPL last %s n=%s", sample.index[-1].date(), len(sample))


if __name__ == "__main__":
    main()
