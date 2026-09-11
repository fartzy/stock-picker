"""One-shot log of Fidelity Individual TOD fills (Sep 9-11 2026).

529 contributions, gifting deposit, and SPY cash/margin journals are omitted.
"""

from __future__ import annotations

from stock_picker.storage.trade_store import Trade, TradeStore

# Mid-session stamps so they sort as that trading day, not midnight.
FILLS = [
    ("ABM", "buy", 200, 50.32, "2026-09-09T10:00:00-05:00"),
    ("TBBK", "buy", 200, 51.24, "2026-09-09T10:00:00-05:00"),
    ("BRZE", "buy", 400, 25.13, "2026-09-09T10:00:00-05:00"),
    ("CASY", "buy", 16, 610.59, "2026-09-09T10:00:00-05:00"),
    ("COHR", "buy", 33, 310.65, "2026-09-09T10:00:00-05:00"),
    ("PLTR", "buy", 60, 170.09, "2026-09-09T10:00:00-05:00"),
    ("TTAN", "buy", 165, 61.71, "2026-09-09T10:00:00-05:00"),
    ("SFD", "buy", 500, 20.70, "2026-09-09T10:00:00-05:00"),
    ("ABM", "sell", 200, 49.52, "2026-09-10T10:00:00-05:00"),
    ("TBBK", "sell", 200, 50.32, "2026-09-10T10:00:00-05:00"),
    ("BRZE", "sell", 400, 24.17, "2026-09-10T10:00:00-05:00"),
    ("CASY", "sell", 16, 650.90, "2026-09-10T10:00:00-05:00"),
    ("COHR", "sell", 33, 305.52, "2026-09-10T10:00:00-05:00"),
    ("FTNT", "sell", 15, 160.54, "2026-09-10T10:00:00-05:00"),
    ("PLTR", "sell", 60, 167.46, "2026-09-10T10:00:00-05:00"),
    ("SFD", "sell", 500, 20.78, "2026-09-10T10:00:00-05:00"),
    ("FLY", "buy", 500, 21.36, "2026-09-11T09:40:00-05:00"),
    ("IREN", "buy", 250, 43.90, "2026-09-11T09:40:00-05:00"),
    ("NBIS", "buy", 50, 228.24, "2026-09-11T09:40:00-05:00"),
    ("UCTT", "buy", 130, 73.74, "2026-09-11T09:40:00-05:00"),
    ("NVTS", "buy", 900, 11.78, "2026-09-11T09:40:00-05:00"),
    ("UCTT", "sell", 130, 74.86, "2026-09-11T10:30:00-05:00"),
    ("TTAN", "buy", 200, 54.12, "2026-09-11T10:30:00-05:00"),
    ("SHOP", "sell", 25, 131.15, "2026-09-11T10:30:00-05:00"),
    ("TTAN", "buy", 100, 54.07, "2026-09-11T10:30:00-05:00"),
    ("FLY", "sell", 300, 21.48, "2026-09-11T11:00:00-05:00"),
    ("FLY", "sell", 200, 21.45, "2026-09-11T11:00:00-05:00"),
    ("IREN", "sell", 250, 44.21, "2026-09-11T11:00:00-05:00"),
]


def main() -> None:
    store = TradeStore()
    for ticker, side, shares, price, executed_at in FILLS:
        store.append(
            Trade(ticker=ticker, side=side, shares=shares, price=price, executed_at=executed_at)
        )
        print(f"logged {side} {shares} {ticker} @ {price} on {executed_at[:10]}")
    print(f"done {len(FILLS)} fills")


if __name__ == "__main__":
    main()
