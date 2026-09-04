"""Trade log wiring and shaping for the API/CLI.

Combined in one file (unlike catalog_loader.py/catalog.py's split) since
there's no ticker-selection branching to share across call sites -- just one
store and one read, mirroring price_store.py's single-purpose simplicity.
"""

from __future__ import annotations

import pandas as pd

from stock_picker.storage.trade_store import TradeStore

NOTIONAL_DECIMAL_PLACES = 2


def trade_log() -> pd.DataFrame:
    return TradeStore().read()


def trade_history(trades: pd.DataFrame) -> list[dict]:
    """Trades newest-first, each with a computed notional (shares * price) and,
    for sells, a realized_pnl against the average cost basis of prior buys.

    Average-cost method (not FIFO lots) -- simplest correct approach at this
    volume. Walks chronologically since cost basis only makes sense forward
    in time, then re-sorts newest-first for display, matching the prior
    behavior of this function.
    """
    if trades.empty:
        return []
    chronological = trades.sort_values("executed_at", ascending=True)

    position_shares: dict[str, float] = {}
    position_cost: dict[str, float] = {}
    enriched = []
    for row in chronological.itertuples():
        ticker = row.ticker
        realized_pnl = None
        if row.side == "buy":
            position_shares[ticker] = position_shares.get(ticker, 0.0) + row.shares
            position_cost[ticker] = position_cost.get(ticker, 0.0) + row.shares * row.price
        else:
            prior_shares = position_shares.get(ticker, 0.0)
            avg_cost_basis = (position_cost.get(ticker, 0.0) / prior_shares) if prior_shares else 0.0
            realized_pnl = round((row.price - avg_cost_basis) * row.shares, NOTIONAL_DECIMAL_PLACES)
            position_shares[ticker] = prior_shares - row.shares
            position_cost[ticker] = position_cost.get(ticker, 0.0) - avg_cost_basis * row.shares

        enriched.append(
            {
                "ticker": ticker,
                "side": row.side,
                "shares": row.shares,
                "price": row.price,
                "notional": round(row.shares * row.price, NOTIONAL_DECIMAL_PLACES),
                "executed_at": row.executed_at,
                "realized_pnl": realized_pnl,
            }
        )
    enriched.sort(key=lambda t: t["executed_at"], reverse=True)
    return enriched


def position_summaries(trades: pd.DataFrame, quotes: dict[str, dict]) -> list[dict]:
    """One row per (ticker, day): that day's buy(s) and sell(s) for the
    ticker merged into a single round-trip view, instead of one row per raw
    transaction.

    day_open/prev_close/current_price come from live quotes, which only ever
    reflect *today's* session -- correct for same-day positions (the only
    kind that exist so far); a position dated on a prior day would need a
    historical lookup (features/price_history.py) instead, not built here
    since there's no multi-day trade history yet to need it.
    """
    if trades.empty:
        return []
    trades = trades.copy()
    trades["executed_dt"] = pd.to_datetime(trades["executed_at"])
    trades["day"] = trades["executed_dt"].dt.date

    rows = []
    for (ticker, day), group in trades.groupby(["ticker", "day"]):
        buys = group[group["side"] == "buy"].sort_values("executed_dt")
        sells = group[group["side"] == "sell"].sort_values("executed_dt")

        buy_shares = float(buys["shares"].sum())
        buy_cost = float((buys["shares"] * buys["price"]).sum())
        avg_buy_price = buy_cost / buy_shares if buy_shares else None

        sell_shares = float(sells["shares"].sum())
        sell_proceeds = float((sells["shares"] * sells["price"]).sum())
        avg_sell_price = sell_proceeds / sell_shares if sell_shares else None

        quote = quotes.get(ticker, {})
        current_price = quote.get("last")
        day_open = quote.get("open")
        prev_close = quote.get("prev_close")
        gap = round(day_open - prev_close, NOTIONAL_DECIMAL_PLACES) if day_open is not None and prev_close else None
        gap_pct = round(gap / prev_close, 4) if gap is not None else None
        is_closed = buy_shares > 0 and sell_shares >= buy_shares

        if is_closed:
            pnl = round(sell_proceeds - buy_cost, NOTIONAL_DECIMAL_PLACES)
        elif current_price is not None and buy_shares:
            pnl = round((current_price - avg_buy_price) * buy_shares, NOTIONAL_DECIMAL_PLACES)
        else:
            pnl = None

        rows.append(
            {
                "ticker": ticker,
                "day": day.isoformat(),
                "shares": buy_shares,
                "invested": round(buy_cost, NOTIONAL_DECIMAL_PLACES),
                "buy_time": buys["executed_at"].iloc[0] if not buys.empty else None,
                "buy_price": round(avg_buy_price, NOTIONAL_DECIMAL_PLACES) if avg_buy_price is not None else None,
                "day_open": day_open,
                "prev_close": prev_close,
                "gap": gap,
                "gap_pct": gap_pct,
                "sell_time": sells["executed_at"].iloc[-1] if not sells.empty else None,
                "sell_price": round(avg_sell_price, NOTIONAL_DECIMAL_PLACES) if avg_sell_price is not None else None,
                "current_price": current_price,
                "closed": is_closed,
                "pnl": pnl,
            }
        )
    rows.sort(key=lambda r: (r["day"], r["ticker"]), reverse=True)
    return rows
