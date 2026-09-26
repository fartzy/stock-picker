"""One-shot: which provider fills WRBY (or --tickers) today.

  bazelisk run //python/stock_picker/ingestion:probe_quotes
  bazelisk run //python/stock_picker/ingestion:probe_quotes -- --tickers WRBY,PS,PL
"""

from __future__ import annotations

import argparse

from stock_picker.ingestion.quote_providers import quote_providers
from stock_picker.ingestion.session import cash_session_date
from stock_picker.log import configure_logging, get_logger

logger = get_logger(__name__)


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", default="WRBY", help="Comma-separated tickers")
    args = parser.parse_args()
    tickers = [part.strip().upper() for part in args.tickers.split(",") if part.strip()]
    as_of = cash_session_date()
    logger.info("as_of=%s tickers=%s", as_of, tickers)
    for provider in quote_providers():
        quotes = provider.fetch(tickers, as_of)
        for ticker in tickers:
            quote = quotes.get(ticker)
            if quote:
                logger.info(
                    "%s %s open=%s last=%s prev_close=%s",
                    provider.name,
                    ticker,
                    quote.get("open"),
                    quote.get("last"),
                    quote.get("prev_close"),
                )
            else:
                logger.info("%s %s missing", provider.name, ticker)


if __name__ == "__main__":
    main()
