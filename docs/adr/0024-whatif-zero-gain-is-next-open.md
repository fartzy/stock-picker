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

All other gain/wait combinations keep the hold rule in `holdExit`: sell at
the target if the high reaches it, sell at the stop if the low reaches it,
the stop wins when both print on the same day, and otherwise sell at the
deadline close.

## Known gap

For gains above 0%, `summarizeHold` counts only the close-to-exit return. It
leaves out the day's own open-to-close trade, so those totals can't be
compared directly with Sell at the close or Next morning open.
