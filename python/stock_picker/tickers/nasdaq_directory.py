"""Fetch and filter NASDAQ Trader's official symbol directories.

NASDAQ publishes two free, no-auth, plain-text symbol directories covering
essentially all US-exchange-listed symbols (NASDAQ-listed + NYSE/AMEX/other-
listed). Combined, these are a genuinely broad ~13,000-symbol pool, used as
the candidate universe for top-N-by-market-cap ranking in `universe.py` --
unlike the S&P 500 list, this isn't pre-limited to ~500 constituents.
"""

from __future__ import annotations

import re
from io import StringIO

import pandas as pd
import requests

NASDAQ_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/symdir/nasdaqlisted.txt"
OTHER_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/symdir/otherlisted.txt"

_REQUEST_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; stock-picker/1.0)"}

# Deliberately not excluding "Trust"/"Fund" broadly -- that would also kill
# legitimate REITs, exactly the kind of mid/large-cap operating company this
# broader universe wants. SPAC detection here is a best-effort text match,
# not a guarantee -- a SPAC that doesn't say "Acquisition"/"Blank Check"
# slips through, but pre-merger SPAC trusts are small and mostly get pushed
# out by the top-N-by-market-cap cut anyway.
_EXCLUDE_NAME_PATTERN = re.compile(
    r"\bWarrants?\b|\bWts?\b|\bUnits?\b|\bRights?\b|\bPreferred\b|\bPfd\b"
    r"|\bDepositary Shares?\b|\bAcquisition (Corp|Company|Holdings)\b|\bBlank Check\b",
    re.IGNORECASE,
)


def _fetch_directory(source_url: str) -> pd.DataFrame:
    response = requests.get(source_url, headers=_REQUEST_HEADERS, timeout=30)
    response.raise_for_status()
    # StringIO, not the raw string -- same reasoning as fetch_sp500_constituents.
    df = pd.read_csv(StringIO(response.text), sep="|")
    # Last line is a "File Creation Time: ..." footer, not a data row -- it has
    # a value in the first column but NaN everywhere else, so dropping on a
    # column every real row always has (Test Issue) removes exactly that line.
    return df.dropna(subset=["Test Issue"])


def _is_common_stock(security_name: str) -> bool:
    return not _EXCLUDE_NAME_PATTERN.search(security_name)


def fetch_nasdaq_listed_symbols(source_url: str = NASDAQ_LISTED_URL) -> list[str]:
    """Return common-stock ticker symbols from NASDAQ's own listed directory."""
    table = _fetch_directory(source_url)
    table = table[
        (table["Test Issue"] == "N")
        & (table["ETF"] == "N")
        & (table["NextShares"] == "N")
        & table["Security Name"].apply(_is_common_stock)
    ]
    return table["Symbol"].str.replace(".", "-", regex=False).tolist()


def fetch_other_listed_symbols(source_url: str = OTHER_LISTED_URL) -> list[str]:
    """Return common-stock ticker symbols from NASDAQ Trader's "other listed"
    directory (NYSE/AMEX/etc.)."""
    table = _fetch_directory(source_url)
    table = table[
        (table["Test Issue"] == "N")
        & (table["ETF"] == "N")
        & table["Security Name"].apply(_is_common_stock)
    ]
    return table["ACT Symbol"].str.replace(".", "-", regex=False).tolist()


def fetch_candidate_tickers() -> list[str]:
    """Combine both NASDAQ Trader directories into one deduplicated, common-
    stock-only candidate pool -- the broad list `top_n_by_market_cap` ranks
    down to the final universe."""
    symbols = set(fetch_nasdaq_listed_symbols()) | set(fetch_other_listed_symbols())
    return sorted(symbols)
