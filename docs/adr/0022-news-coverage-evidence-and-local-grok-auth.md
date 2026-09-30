# 0022: Preserve news coverage evidence and use local Grok authentication

Date: 2026-09-30
Status: Accepted

## Context

The September 30 TRLV scan reported a completed news check without a flag.
Its saved archive omitted a regulatory catalyst later returned by Finnhub;
the exact morning shortlist response was not retained. Replaying the public
headline through the fallback classifier also missed it. The local Grok
configuration contained an API key, but the stock-picker loader only read
its environment-variable setting.

The UI conflated no flag with a successful check. API response models dropped
`news_checked`, and later morning batches were persisted only if they flagged
a name. These independent gaps made incomplete checks appear clear.

## Decision

- Read the existing local `~/.grok/config.toml` model `api_key` in memory,
  after an explicit `LLM_PROXY_API_KEY` override. Also support Grok's ordered
  `env_key` list. Do not copy keys into the checkout, saved scans, or logs.
  Bind config keys to their configured endpoint and do not use a personal
  xAI key for an unrelated proxy.
- Query both Finnhub and Polygon/Massive for each shortlisted name, using
  existing credential readers. A nonempty primary response does not establish
  complete coverage. Preserve empty success separately from failure.
- Normalize publication times, exclude stale/future articles, deduplicate
  headlines/URLs, and save the exact normalized inputs with the scan. This
  evidence is separate from the universe-news training corpus.
- Review headlines and summaries, including company and directly relevant
  sector/regulatory catalysts. The LLM receives up to 20 articles. Any cap,
  feed failure, or classifier fallback yields an explicit limited check.
  A narrow regulatory-action fallback covers the reported missed event
  without changing the text model used to produce historical ML features.
- Keep the existing 40-name per-batch request limit; return `not_checked` for
  overflow. Requests and judge concurrency remain bounded.
- Carry status, counts, sources, issues, judge, timestamp, and article evidence
  through saved scans, API models, Test run, and the paper book. Save every
  completed morning batch, including checks with no flags. Existing paper
  databases get a nullable JSON evidence column; old scans remain unknown.
- Display reviewed, no articles found, limited check, failed, pending, or
  not checked. Historical data without evidence says coverage not recorded.

## Consequences

Two feeds improve coverage but cannot guarantee every catalyst was indexed.
Both providers may rate-limit or omit stories; partial coverage remains visible.
Live validation confirmed the local Grok key works and flags the public TRLV
rescheduling headline. Polygon returned HTTP 200 but no articles for that
specific window, so it is supplemental coverage, not proof it caught this story.

The gap-conditional trading policy in ADR 0016 remains unchanged. Replays do
not claim a news check when it was disabled. Historical scans are not rewritten
using articles obtained after their original decision time.

## References

- Grok local user guide: `~/.grok/docs/user-guide/05-configuration.md`, Custom Models.
- [Massive news endpoint](https://massive.com/docs/rest/stocks/news)
- [Requests timeouts and HTTP errors](https://docs.python-requests.org/en/latest/user/quickstart/)
