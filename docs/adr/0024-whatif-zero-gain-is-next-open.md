# 0024: What If +0% / 1-day wait is the Next morning open

Date: 2026-10-17
Status: Implemented

## Problem

On the What If tab, these two settings showed different Rank and Fit totals:

- After the close → Next morning open, All included, 100% held
- After the close → Wait for a gain, +0%, any stop, Wait 1d

They should match. The next morning's open is the first price after the
close, and it already meets a 0% target, so the stop never gets a turn.

PR #241 tried to patch the +0% branch of `summarizeHold` to use the
next-open formula. That still disagreed because the two paths differed in
three ways, not one:

| | Next morning open | Hold path |
|---|---|---|
| Price source | verified Massive series, action-checked (`training/overnight_hold.py`) | `PriceStore` parquet bars (`paper/hold_path.py`) |
| Population | scored/held picks; skips a pick whose verified close disagrees with the book close | every pick with a close |
| Rounding | cents per pick | float sum |

## Decision

Don't keep two formulas. `resolveAfterClose` (`typescript/src/whatIfHold.ts`)
maps the toolbar choice to one of three calculations, and sends +0% / 1d to
the Next morning open calculation (rule `all`, 100% held). Both settings now
go through `summarizeSelectedOvernightHistory` with the same actuals, so the
dollars, average and priced count are the same by construction.

All other gain/wait combinations go through `holdExit`, and the total is
the whole trade: the day's open-to-close result plus the after-close move on
the same shares, over the same scored/held picks as the close-only baseline.
Each later session checks its open first, so a gap through the target or the
stop fills at the open. Within a session the stop wins when both levels
print. If neither prints, the pick sells at the deadline close, and it stays
pending only until it exits.

Before this, the hold total counted only the close-to-exit leg, let the stop
override an open that had already filled the target, filled gap-downs at the
stop price, and counted skipped picks. On 2026-10-17 that showed Rank +0% / 5d
at -$7,298 against +$2,692 close-only. The corrected figure is +$2,157.

## Update 2026-10-18: minute bars break same-session ties

When a session's high reaches the target and its low reaches the stop after
an open between them, daily bars cannot say which came first. The What If
page now sends only those sessions to `POST /api/paper-book/first-touch`.
That route reads regular-session (9:30-16:00 ET) Massive one-minute bars,
caches each (ticker, session) under `data/prices/minute/`, and answers
`target`, `stop`, `same_minute`, `neither` or `unavailable`.

Only a `target` answer changes the exit. `stop`, `same_minute`, and a
missing answer all keep the stop-first rule, so missing data never makes a
result look better.

Rank 5 / Fit 5, +1% / -2% / 1d on the real book: 32 ties (13 target first,
18 stop first, 1 same minute). Rank moves from +$2,576 to +$4,080, Fit from
-$12,851 to -$10,449.
