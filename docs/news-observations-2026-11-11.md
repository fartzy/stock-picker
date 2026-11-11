# News observation history

News collection now keeps immutable fetch snapshots: raw response, actual observation time, provider, classifier source hash/package version, and each classified article version. Successful empty responses and failures are distinct. Old articles are not assigned invented first-seen dates.

`NewsStore.observed_articles(ticker, at)` returns only versions observed and published by the cutoff. A later correction cannot overwrite what an earlier replay sees. Legacy news-feature behavior is unchanged; these observations are the source for future event-conditioned features.

Existing morning collection records observations automatically. The scheduled nightly pipeline additionally refreshes up to 20 saved Rank and 20 saved Fit candidates, including untraded names, even if morning ingestion already completed. It then resumes the remaining universe ingestion normally. This adds at most 40 candidate requests, at the existing rate limit; it does not add a full-universe second sweep or UI controls.

Manual incremental refresh:

```sh
bazelisk run //python/stock_picker/training:news_ingest -- --refresh --tickers MRCY,TTMI
```

Tests verify first-seen preservation, corrections, availability cutoffs, coverage failures, incremental refresh and candidate selection. The initial live refresh succeeded for all **39 candidates**, preserving **518 article observations**. Its verified compressed export is recorded in [the snapshot manifest](../data/snapshots/2026-11-11/news-observations-initial-manifest.json).

This is morning/post-close collection, not continuous intraday news coverage. Event-category features and a regular intraday polling service remain separate follow-ups.
