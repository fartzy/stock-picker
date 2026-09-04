"""CLI to log an executed trade.

bazel run //python/stock_picker/features:log_trade -- \\
  --ticker HOOD --side buy --shares 50 --price 121.88 \\
  --executed-at 2026-09-04T10:08:10-04:00
"""

from __future__ import annotations

import argparse
from datetime import datetime

from stock_picker.storage.trade_store import Trade, TradeStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Log an executed trade")
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--side", choices=["buy", "sell"], required=True)
    parser.add_argument("--shares", type=float, required=True)
    parser.add_argument("--price", type=float, required=True)
    parser.add_argument(
        "--executed-at",
        default=None,
        help="ISO 8601 timestamp with offset, e.g. 2026-09-04T10:08:10-04:00. Defaults to now.",
    )
    args = parser.parse_args()

    executed_at = args.executed_at or datetime.now().astimezone().isoformat()

    TradeStore().append(
        Trade(
            ticker=args.ticker,
            side=args.side,
            shares=args.shares,
            price=args.price,
            executed_at=executed_at,
        )
    )
    print(f"Logged {args.side} {args.shares} {args.ticker} @ {args.price} at {executed_at}")


if __name__ == "__main__":
    main()
