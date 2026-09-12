"""Weekly-ish Yahoo profile pull: sector labels for the universe.

Not a morning-open feed -- Ticker.info is one request per name. Nightly can
refresh names that still have no sector. Features use whatever is already
on UniverseStore.
"""

from __future__ import annotations

from stock_picker.storage.universe_store import UniverseStore

try:
    import yfinance as yf
except ImportError:  # pragma: no cover
    yf = None


def sector_from_info(info: dict) -> str | None:
    sector = info.get("sector") or info.get("sectorDisp")
    if not sector or not str(sector).strip():
        return None
    return str(sector).strip()


def fetch_sectors(tickers: list[str]) -> dict[str, str]:
    if yf is None or not tickers:
        return {}
    found = {}
    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).info or {}
        except Exception:
            continue
        sector = sector_from_info(info)
        if sector:
            found[ticker] = sector
    return found


def refresh_missing_sectors(universe_store: UniverseStore | None = None, limit: int = 80) -> int:
    """Fill sector on active names that don't have one yet. Capped so a
    nightly run doesn't hammer Yahoo for all 2000."""
    store = universe_store or UniverseStore()
    have = store.sector_by_ticker()
    missing = [ticker for ticker in store.active_tickers() if ticker not in have]
    if not missing:
        return 0
    written = fetch_sectors(missing[:limit])
    if written:
        store.write_sectors(written)
    return len(written)


def main() -> None:
    n = refresh_missing_sectors()
    print(f"wrote sectors for {n} tickers", flush=True)


if __name__ == "__main__":
    main()
