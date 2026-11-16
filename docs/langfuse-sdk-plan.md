# Langfuse SDK upgrade plan (news judge)

Status: **Planned, not started.** Owner decision pending on step 0 (server
version). Written 2026-11-16 from a disk-cleanup session; nothing below is
implemented yet.

Depends on: PR #304 (ADR 0031, ClickHouse log caps for the local Langfuse
stack). Merge that first so `docker compose up` does not regrow ~3 GB of
ClickHouse self-logs.

## Why

Langfuse is wired in (ADR 0016, #94) but only as a flat event log:
`grok_judge()` (`python/stock_picker/training/news_day_judge.py`) calls xAI
with a raw `requests.post`, and `trace_news_judge()`
(`python/stock_picker/training/langfuse_local.py`) posts one `trace-create`
per ticker with `{ticker, headlines, judge}` -> `{avoid, reason}`.

As of 2026-11-13 the local instance held **803 traces** (2026-09-26 ..
2026-11-03), **0 observations, 0 scores**. Missing: the prompt, model, tokens,
cost, latency, any link from a verdict to what the stock actually did, and
prompt versioning. The stack has been stopped since 2026-11-03, and the client
no-ops silently when it is down, so nothing has been traced since.

Goal: make the judge measurable -- "was skipping this name right?" -- and use
the work to learn the Langfuse SDK properly.

## Step 0 -- pick server + SDK version (decide before coding)

Current Langfuse Python SDK is **v5**, which requires a **Langfuse v4 server
(>= v4.59.0)**. `langfuse/docker-compose.yml` pins `langfuse:3` /
`langfuse-worker:3`; for v3 servers the docs say stay on **SDK v4**.

| Option | Work | Trade-off |
|---|---|---|
| **A (recommended)** Upgrade local server to v4, use SDK v5 | Bump images, back up `langfuse_langfuse_postgres_data` + `langfuse_langfuse_clickhouse_data` first, let migrations run | Learn the current API; local-only, 803 traces, low risk |
| B Stay on server v3, use SDK v4 | Add dependency only | Less work, but learning an API that is already a major version behind |

Before writing each phase, confirm exact signatures against
https://python.reference.langfuse.com -- do not code from memory
(repo convention: never guess a library's current API).

## Phase 1 -- real tracing (one PR)

- Replace the ingestion `requests.post` with the SDK client (`get_client()`,
  `LANGFUSE_BASE_URL`). **Keep the existing guarantees**: localhost-only base
  URL check, `LANGFUSE_DISABLED`, and morning scoring never fails or slows
  when Langfuse is down (SDK exports async and swallows its own errors; keep a
  short flush timeout).
- One root span per morning run; per ticker a **generation**
  (`start_as_current_observation(as_type="generation", model=...,
  model_parameters={"temperature": 0})`) recording system + user messages, raw
  completion, `usage_details` from the xAI response, latency.
- Fallback paths (`tfidf`, `high_impact_event_flag`, regulatory) as plain spans
  so the UI shows *which* path decided.
- `propagate_attributes(session_id=<scan date>, tags=[...])` so each morning is
  one Langfuse session.
- `grok_judge()` currently returns only `(avoid, reason)`; it will need to
  surface usage/raw content to the tracer without changing its callers'
  contract (`flag_from_articles`, `check_news_coverage`).
- Deps: add `langfuse` to `requirements.txt` / the Bazel `pip.parse` lock; run
  `bazelisk run //:gazelle`. Note the open P3 gazelle trap in
  `docs/delegation-briefing.md` lists "langfuse additions" as a suspect.
- Tests: keep the "Langfuse down -> no-op" behaviour covered; stub the client
  rather than requiring a running server.

## Phase 2 -- outcome scores (one PR)

- After the close, score each judged ticker's trace: `judge_correct`
  (boolean) and `realized_return` (numeric, Open->Close).
- Reuse the existing Open->Close scoring in `paper/` (What if book) -- do not
  re-derive returns.
- Result: a running accuracy view of the judge in Langfuse dashboards.

## Phase 3 -- prompt management (one PR)

- Move `_SYSTEM` in `news_day_judge.py` to a versioned Langfuse prompt
  (`get_prompt()`), **with the in-code text as the fallback** so the 8:30
  morning path never depends on Langfuse being up.
- Link each generation to its prompt version to compare accuracy across
  versions.

## Phase 4 -- datasets + experiments (one PR)

- Build a dataset from past mornings (headlines -> realized outcome).
- Experiment script that re-runs the judge against it on prompt/model changes;
  same "decide empirically" methodology as `tune_experiment.py`.

## Process

- One branch + PR per phase, squash-merged (CLAUDE.md conventions).
- One ADR per phase; point ADR 0016 at the new record rather than rewriting it.
- Delete or mark this plan `Done` as phases land.
