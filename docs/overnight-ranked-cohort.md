# Overnight training cohort

The overnight training job reconstructs each session's Rank top 20 from
same-day models fitted strictly before that session. It ranks the entire
available cross-section first, then joins verified raw close → next-open
labels. Fit-only names do not enter the cohort; a missing overnight bar cannot
advance a lower-ranked ticker. Equal Rank scores use ticker ascending as a
deterministic tie-break. The saved model records
`cohort_source=reconstructed_prior_fit_rank_oof`.
Each run also writes a content-addressed CSV manifest under
`mlruns/overnight_cohorts/` (or the requested tracking directory). It lists
date, ticker, Rank position/score, the Rank fitting cutoff, and one of
`verified_label`, `no_verified_label`, `outside_requested_range`, or
`outside_ticker_whitelist`. The model records the manifest's path **relative to
the run's tracking directory** and its SHA-256 digest, not an absolute path
specific to one machine.

This is **not an exact replay of the historical morning scans**. The current
implementation builds its candidate cross-section from today's active ticker
registry and the pooled same-day training rows. Those rows require available
price/feature files and an observed same-day close. Historically inactive or
delisted names, names missing local history, and names with an open but no close
are absent. The saved artifact records this limitation as
`cohort_universe_limit=current_active_tickers_with_observed_day_close_and_features`.
It must not be described as "the stocks actually ranked that morning" until
dated universe/quote snapshots can reconstruct those exact candidates.

For requested label dates `[start, end]`, the Massive fetch begins six XNYS
sessions before `start` so the first date can have six prior completed closes.
Only labels dated within the requested window enter training; the last fetched
bar still has no next-open label. Missing or incomplete same-day score
cross-sections stop the run rather than changing the reconstructed top 20.

The CLI reports `label_rows` for selected, verified Rank ticker-days only.
`provider_labeled_rows`, `provider_candidate_rows`, and
`provider_excluded_by_reason` instead cover **all fetched bars for all fetched
tickers**, including six-session history, dates outside the selected Rank
cohort, and the final unlabeled bar. Their denominator is intentionally
different from `label_rows`; do not calculate a cohort exclusion rate from
the provider-wide counts.
