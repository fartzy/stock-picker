# Intraday buy and sell timing research

Status: TODO, requested September 29, 2026. Not implemented by the What if dollar-sizing change.

Investigate when a given stock has historically offered favorable intraday entry and exit prices. This may become its own panel or a ticker-level section in Data. Historical extrema must be presented as hindsight observations, not promises about the next session.

## Requested views

- Show each of the last five completed trading sessions separately: T−1, T−2, T−3, T−4, T−5, with dates and support for more sessions. Define T as the selected analysis session; exclude T from its historical inputs.
- For each day, show the best observed sell price and its time/window prominently, alongside the best observed buy price and its time/window. Keep all tied windows rather than choosing an arbitrary timestamp.
- Show separate weekly windows (W−1, W−2, W−3, etc.) and monthly windows (M−1, M−2, etc.), not just one aggregate of all recent history. Also offer cumulative past 1/2/3 weeks, past 1/2/3 months, and past year summaries so changes over time can be compared.
- Label every window with its actual date range, usable trading-session count, and coverage for that ticker. Keep completed calendar periods distinct from rolling lookbacks and label a current partial period explicitly if offered.
- Compare candidate buy and sell windows and show the price movement between them.
- Offer a ticker drilldown from the existing picks and trade views if useful.

## Proposed statistics to evaluate

- Normalize prices to that session's open or another explicitly labeled reference before comparing different days. An average raw dollar price across a year is not a useful entry level on its own.
- For each intraday time bucket, report the median normalized price, interquartile range, sample count, and frequency of containing the session low/high. Keep a mean available for comparison rather than silently deciding it is the right summary.
- Use distributions of time windows rather than only averaging the clock times of daily extrema; two distinct morning/afternoon clusters should remain visible.
- Proposed default for each weekly/monthly/yearly view: the most frequent sell-high and buy-low time buckets, their frequencies, and median time with a middle-50% range. Show multiple common windows when the distribution has separate peaks; do not label the median alone as the best time. Evaluate bucket size and stability before choosing defaults.
- Separate independent daily low/high observations from the best possible long-only round trip: a valid buy must precede its sell. Explain that even a printed low/high may not have been executable.
- Evaluate repeatable entry/exit windows using walk-forward holdouts, fees, spread/slippage assumptions, and a simple baseline. A window selected using the same days being scored is hindsight, not an out-of-sample result.

## Data and implementation TODOs

- Confirm provider access, retention, and cost for timestamped intraday bars, especially a full year. Daily OHLC cannot reveal the time of a low/high.
- Decide bar resolution and regular-session boundaries; handle exchange holidays, early closes, DST, missing bars, low volume, new listings, and splits consistently.
- Define week/month/year boundaries and the analysis-session anchor explicitly. The five-day detail view means five trading sessions, not five calendar dates. Use the same reusable window definitions for separate prior periods and cumulative lookbacks; never include data after the selected analysis time.
- Keep ingestion and cached intraday storage separate from pure statistical calculations, API responses, and the panel. Reuse the same result objects for per-day detail and aggregate summaries.
- Do not fabricate unavailable history or convert daily lows/highs into invented intraday timestamps. Show coverage and insufficient-sample states.
- Test time normalization, buy-before-sell ordering, tied extrema, missing data, split normalization, and no future-data leakage.

## Open decisions

Choose a standalone Intraday timing panel versus a Data ticker subsection; minute bars versus wider windows; the reference price for normalization; and the minimum sample count before displaying a timing pattern. No training or live-model changes are authorized by this TODO.
