---
name: import-broker-fills
description: Import new Fidelity fills into the stock-picker trade log. Use when the user pastes order confirmations, mentions a new account CSV or saved activity page, or asks to update the app, docket, or trade log with today's or yesterday's fills.
---

# Import broker fills

Log new fills into `data/trades/stockpicker.db`. Skip anything already stored. Do not commit or push unless the user asks.

## Find the source

Search `~/Downloads` for the newest files, by modification time:

- `Portfolio Activity & Orders*.html`
- `Accounts_History*.csv`
- `History_for_Account*.csv`

Also parse confirmation text the user pasted in the same turn. A pasted confirmation with an order number is the authority for that order. A saved page only contains details for rows the user expanded before saving. A collapsed row is a summary: side, shares, symbol, displayed price, and amount. It has no clock time and no order number.

The CSV has the fee and a rounded price. It has no clock time. The saved page has the exact fill price and the execution time for expanded orders. The confirmation net is before the fee.

## Do not log

- `SPYM`
- `SPAXX`, `FCNTX`, and other cash or fund rows
- orders marked `Verified Canceled` or `Open`
- an order number already present in `data/trades/broker_orders_YYYY-MM-DD.json`
- a fill with no order number that already matches ticker, side, shares, net amount, and fee

## Parse the page

Read the HTML as text. Strip tags only after finding the expanded panels. For each expanded order keep:

- order number, matching `2E` plus six letters or digits
- ticker, side, shares, and market session
- the execution time under "Executions for this trade", not the order date, when both exist
- the exact fill price and the confirmation net

Convert `09:21:00 AM ET` on the fill date to `YYYY-MM-DDTHH:MM:SS-04:00`. A premarket fill stays at its real time. The trade list adds the sun when the Eastern time is before 9:30.

## Parse the CSV

Find the header row that starts with `Run Date`. Do not assume column positions. Read `Action`, `Symbol`, `Price ($)`, `Quantity`, `Fees ($)`, and `Amount ($)`.

Keep rows whose action starts with `YOU BOUGHT` or `YOU SOLD`. `YOU BOUGHT` is a buy. A negative share count in the activity list is a sell.

When the same fill exists in the page and the CSV, use the page price and time, and use the CSV fee. Match them by ticker, side, shares, and cash amount. The CSV price is rounded, so do not match on price alone.

If the CSV splits one page order into several rows, log the page order total once. Note the split in the broker record. Do not log both.

## Price and fee

Store the exact confirmation price. The displayed activity price can be rounded.

For a sell, the fee is shares times price minus the confirmation net. For a buy, it is the confirmation net minus shares times price. Use the CSV `Fees ($)` value when the fill came from the CSV. A fee under $0.02 can round to zero.

## Write

Use the existing store. Do not hand-edit the database, and do not rewrite `trades.csv`. The store rewrites that file as a sidecar. The app reads the database.

```python
from stock_picker.storage.trade_store import Trade, TradeStore

TradeStore().append(Trade(ticker, side, shares, price, executed_at, manual_fee))
```

Run it with `PYTHONPATH=python`. Append one record to `data/trades/broker_orders_YYYY-MM-DD.json` for the fill date. Create that file if the day has no record yet. Keep the existing order shape: `order_number`, `ticker`, `side`, `shares`, `price`, `executed_at`, `broker_net_amount`, `manual_fee`, `timestamp_source`, and `market_session`. Do not store account numbers.

If a fill has no execution time, store a clearly marked placeholder and say which fill it is. Do not present the placeholder as a confirmed time.

## Report

Say how many fills were added and how many order numbers were already in the log. For each new fill, give ticker, side, shares, price, time, order number, and any fee. Mention canceled and open orders that were left out.
