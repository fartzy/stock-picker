"""Trade log wiring and shaping for the API/CLI.

Combined in one file (unlike catalog_loader.py/catalog.py's split) since
there's no ticker-selection branching to share across call sites -- just one
store and one read, mirroring price_store.py's single-purpose simplicity.
"""

from __future__ import annotations

from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

from stock_picker.storage.trade_store import TradeStore

_NY = ZoneInfo("America/New_York")
_RTH_OPEN = time(9, 30)
_RTH_CLOSE = time(16, 0)

NOTIONAL_DECIMAL_PLACES = 2


def _fills_in_time_order(trades: pd.DataFrame) -> pd.DataFrame:
    """Chronological fills; at the same timestamp, buys before sells.

    TTAN's 465 buy and 930 sell share 10:02 — if the sell is applied first
    the earlier 465 leftover looks open even though the day was flat.
    """
    frame = trades.copy()
    frame["executed_dt"] = pd.to_datetime(frame["executed_at"], utc=True, format="ISO8601")
    frame["_side_order"] = (frame["side"] != "buy").astype(int)
    frame["_row"] = range(len(frame))
    return frame.sort_values(["executed_dt", "_side_order", "_row"], kind="mergesort")


def trade_log() -> pd.DataFrame:
    return TradeStore().read()


def peak_working_by_day(trades: pd.DataFrame) -> dict[str, float]:
    """Max cost basis actually out during each NY session.

    Walks the whole log so overnight inventory carries in. Sells reduce by
    average cost of the shares sold -- not sell proceeds -- so taking TTAN
    off does not wipe XNDU that is still on until later that day.
    """
    if trades.empty:
        return {}
    frame = _fills_in_time_order(trades)
    shares: dict[str, float] = {}
    cost: dict[str, float] = {}
    peaks: dict[str, float] = {}
    for row in frame.itertuples():
        ticker = row.ticker
        qty = float(row.shares)
        notional = qty * float(row.price)
        if row.side == "buy":
            shares[ticker] = shares.get(ticker, 0.0) + qty
            cost[ticker] = cost.get(ticker, 0.0) + notional
        else:
            held = shares.get(ticker, 0.0)
            avg = (cost.get(ticker, 0.0) / held) if held else 0.0
            sold = min(qty, held) if held else 0.0
            cost[ticker] = cost.get(ticker, 0.0) - avg * sold
            shares[ticker] = held - sold
        working = sum(value for value in cost.values() if value > 0)
        day = row.executed_dt.tz_convert("America/New_York").date().isoformat()
        peaks[day] = round(max(peaks.get(day, 0.0), working), NOTIONAL_DECIMAL_PLACES)
    return peaks


def _apply_fill(shares: dict[str, float], cost: dict[str, float], row) -> float:
    ticker = row.ticker
    qty = float(row.shares)
    notional = qty * float(row.price)
    if row.side == "buy":
        shares[ticker] = shares.get(ticker, 0.0) + qty
        cost[ticker] = cost.get(ticker, 0.0) + notional
    else:
        held = shares.get(ticker, 0.0)
        avg = (cost.get(ticker, 0.0) / held) if held else 0.0
        sold = min(qty, held) if held else 0.0
        cost[ticker] = cost.get(ticker, 0.0) - avg * sold
        shares[ticker] = held - sold
    return sum(value for value in cost.values() if value > 0)


def time_weighted_working_by_day(trades: pd.DataFrame) -> dict[str, float]:
    """Average dollars on the book during 9:30–16:00 ET, weighted by time.

    Peak of the day is not the average -- 20 minutes at $60k then flat is
    not a $60k day. Overnight inventory is on the book at 9:30 until sold.
    """
    if trades.empty:
        return {}
    frame = _fills_in_time_order(trades)
    frame["executed_dt"] = frame["executed_dt"].dt.tz_convert(_NY)
    shares: dict[str, float] = {}
    cost: dict[str, float] = {}
    averages: dict[str, float] = {}
    first_day = frame["executed_dt"].iloc[0].date()
    last_day = frame["executed_dt"].iloc[-1].date()
    fills = list(frame.itertuples())
    fill_i = 0
    day = first_day
    while day <= last_day:
        if day.weekday() < 5:
            session_open = datetime.combine(day, _RTH_OPEN, tzinfo=_NY)
            session_close = datetime.combine(day, _RTH_CLOSE, tzinfo=_NY)
            while fill_i < len(fills) and fills[fill_i].executed_dt < session_open:
                _apply_fill(shares, cost, fills[fill_i])
                fill_i += 1
            working = sum(value for value in cost.values() if value > 0)
            last_t = session_open
            integral = 0.0
            cursor = fill_i
            while cursor < len(fills) and fills[cursor].executed_dt <= session_close:
                fill = fills[cursor]
                dt = (fill.executed_dt - last_t).total_seconds()
                if dt > 0:
                    integral += working * dt
                working = _apply_fill(shares, cost, fill)
                last_t = fill.executed_dt
                cursor += 1
            dt = (session_close - last_t).total_seconds()
            if dt > 0:
                integral += working * dt
            duration = (session_close - session_open).total_seconds()
            averages[day.isoformat()] = round(integral / duration, NOTIONAL_DECIMAL_PLACES) if duration else 0.0
            fill_i = cursor
        day += timedelta(days=1)
    return averages


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
    chronological = _fills_in_time_order(trades)

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
    # utc=True is required when the log mixes offsets (EDT -04:00 and CDT
    # -05:00): naive to_datetime then yields object dtype and .dt blows up.
    chronological = _fills_in_time_order(trades)

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
