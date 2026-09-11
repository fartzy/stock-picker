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
    """One row per closed lot plus leftover open shares, walking the log in
    time so a buy on Monday sold Tuesday is closed -- not an empty "open"
    row on the sell day.

    Open leftover shares are dated on the last buy that added to them.
    Live quotes only apply to leftover open lots (today's mark); closed lots
    use fill prices.
    """
    if trades.empty:
        return []
    trades = trades.copy()
    # utc=True is required when the log mixes offsets (EDT -04:00 and CDT
    # -05:00): naive to_datetime then yields object dtype and .dt blows up.
    trades["executed_dt"] = pd.to_datetime(trades["executed_at"], utc=True)
    chronological = trades.sort_values("executed_dt", ascending=True)

    open_lots: dict[str, dict] = {}
    rows: list[dict] = []

    def _ny_day(timestamp: pd.Timestamp) -> str:
        return timestamp.tz_convert("America/New_York").date().isoformat()

    def _quote_fields(ticker: str) -> dict:
        quote = quotes.get(ticker, {})
        day_open = quote.get("open")
        prev_close = quote.get("prev_close")
        gap = (
            round(day_open - prev_close, NOTIONAL_DECIMAL_PLACES)
            if day_open is not None and prev_close
            else None
        )
        return {
            "day_open": day_open,
            "prev_close": prev_close,
            "gap": gap,
            "gap_pct": round(gap / prev_close, 4) if gap is not None else None,
            "current_price": quote.get("last"),
        }

    for row in chronological.itertuples():
        ticker = row.ticker
        if row.side == "buy":
            lot = open_lots.get(ticker)
            if lot is None:
                open_lots[ticker] = {
                    "shares": float(row.shares),
                    "cost": float(row.shares) * float(row.price),
                    "buy_time": row.executed_at,
                    "buy_dt": row.executed_dt,
                }
            else:
                lot["shares"] += float(row.shares)
                lot["cost"] += float(row.shares) * float(row.price)
                lot["buy_time"] = row.executed_at
                lot["buy_dt"] = row.executed_dt
            continue

        lot = open_lots.get(ticker)
        sell_shares = float(row.shares)
        if lot is None or lot["shares"] <= 0:
            quotes_for = _quote_fields(ticker)
            rows.append(
                {
                    "ticker": ticker,
                    "day": _ny_day(row.executed_dt),
                    "shares": 0.0,
                    "invested": 0.0,
                    "buy_time": None,
                    "buy_price": None,
                    **quotes_for,
                    "sell_time": row.executed_at,
                    "sell_price": round(float(row.price), NOTIONAL_DECIMAL_PLACES),
                    "closed": True,
                    "pnl": None,
                }
            )
            continue

        avg_cost = lot["cost"] / lot["shares"]
        matched = min(sell_shares, lot["shares"])
        invested = avg_cost * matched
        pnl = round((float(row.price) - avg_cost) * matched, NOTIONAL_DECIMAL_PLACES)
        rows.append(
            {
                "ticker": ticker,
                "day": _ny_day(row.executed_dt),
                "shares": matched,
                "invested": round(invested, NOTIONAL_DECIMAL_PLACES),
                "buy_time": lot["buy_time"],
                "buy_price": round(avg_cost, NOTIONAL_DECIMAL_PLACES),
                "day_open": None,
                "prev_close": None,
                "gap": None,
                "gap_pct": None,
                "sell_time": row.executed_at,
                "sell_price": round(float(row.price), NOTIONAL_DECIMAL_PLACES),
                "current_price": None,
                "closed": True,
                "pnl": pnl,
            }
        )
        lot["shares"] -= matched
        lot["cost"] -= invested
        if lot["shares"] <= 1e-9:
            open_lots.pop(ticker, None)

    for ticker, lot in open_lots.items():
        if lot["shares"] <= 1e-9:
            continue
        avg_buy = lot["cost"] / lot["shares"]
        quotes_for = _quote_fields(ticker)
        current = quotes_for["current_price"]
        pnl = (
            round((current - avg_buy) * lot["shares"], NOTIONAL_DECIMAL_PLACES)
            if current is not None
            else None
        )
        rows.append(
            {
                "ticker": ticker,
                "day": _ny_day(lot["buy_dt"]),
                "shares": lot["shares"],
                "invested": round(lot["cost"], NOTIONAL_DECIMAL_PLACES),
                "buy_time": lot["buy_time"],
                "buy_price": round(avg_buy, NOTIONAL_DECIMAL_PLACES),
                **quotes_for,
                "sell_time": None,
                "sell_price": None,
                "closed": False,
                "pnl": pnl,
            }
        )

    rows.sort(key=lambda r: (r["day"], r["ticker"]), reverse=True)
    return rows
