# Session handoff — 2026-09-29

This note is for resuming the stock-picker work. **Do not reset, clean, or bulk-stage the original worktree.** The user expects the remaining local files to be accounted for and pushed, but that has not happened yet.

## Original worktree: outstanding work

- Directory: `/Users/michael.artz/dev/stock-picker`; branch: `experiment/svm-stacked-features`; HEAD: `c7bae2cf` (`Morning picks 2026-09-29`).
- At handoff: **4,024 modified tracked files and 7 untracked files**, before this note was added. Most tracked changes are generated `data/features/*.parquet` and other market data, mixed with source/UI changes. The untracked set includes a model pickle, Sep 29 broker/open-order CSVs, an intraday-timing TODO, an SVM test, and What If TypeScript files. This note adds one more untracked file.
- None of those original-worktree changes were reset or bulk-committed. **Audit and group them before staging/pushing**; do not assume every generated data or model artifact belongs in a source PR. The user explicitly called out that this dirty branch had not been pushed.

## Already merged

- What If Rank/Fit S&P intraday and buy-and-hold comparison: [GitHub PR #147](https://github.com/fartzy/stock-picker/pull/147), squash merge `b05bb632f1612b2604b8c8135f45f2af80c1aa4d` on `main`. Source was implemented/tested in a separate worktree; it did **not** clear the original worktree.

## Separate branches not merged

- `fix/spy-benchmark-history`, worktree `/private/tmp/stock-picker-spy-hold-20260929`, commit `a85781f9`, **pushed** to origin. The API now uses one SPY download for intraday, prior-close, and hold returns. Focused benchmark and full API route Bazel tests passed. A `gh pr create` attempt failed under sandbox network access; an escalated attempt was interrupted. **Check whether a PR exists before retrying**, then create/merge if desired. Do not claim the Tue Sep 29 blank is fixed live without checking the response: if Yahoo omits Sep 28/29 valid closes, the cell correctly stays unavailable. See `python/stock_picker/api/routes.py:328` and `python/stock_picker/features/benchmark.py:17`.
- `feat/svm-derived-features`, worktree `/private/tmp/stock-picker-svm.Oevew9/worktree`, remote HEAD `c68f7074f261cc83b1c7b0f56758e0d483004850`, **pushed but no PR/merge**. Two verified commits add the seven experimental SVM feature definitions/catalog UI and a chronological out-of-fold research comparison runner. This does **not** wire SVM into nightly/live training, and the long performance trial has not run.
- The SVM worktree additionally has **uncommitted** checkpoint/resume changes (`python/stock_picker/training/svr_stack_run.py` and related edits). Tests passed, but review found a blocker: `svr_stack_run.py:207` resumes via `pd.read_pickle` from a user-supplied run directory. A checksum in the same SQLite DB does not authenticate it. Replace with numeric-only serialization (`np.load(..., allow_pickle=False)`), validate shape/finite values, bump checkpoint schema, rerun tests, then review before committing or running the trial. Do not merge or run this uncommitted code as-is.

## News diagnosis (no code fix yet)

- FICO was in the Sep 29 morning Fit scan and was marked `news_checked=true`, `news_flag=null` despite about a 19.7% opening gap down. The local news archive contains related FICO mortgage-scoring competition headlines, but marked all archived FICO articles non-material. The morning log did not persist per-headline judgments, so it cannot prove which exact article the morning checker saw. See `python/stock_picker/news_skip.py:39`, `python/stock_picker/training/headline_sentiment.py:20`, and `python/stock_picker/training/news_day_judge.py:35`. A separate paper-book path ignores the scan's checked flag (`python/stock_picker/paper/book.py:192`). Recommended follow-up: regulatory-competition regression test, per-ticker decision trace, distinguish failed/unavailable checks from clear, and propagate checked state.

## Next safe steps

1. Confirm remote/PR state for `fix/spy-benchmark-history`; do not duplicate a possibly-created PR.
2. Audit the original dirty worktree by file category and size, separate generated data/model artifacts from source, and agree on what should be committed. Preserve all user changes. The user specifically wants an accounting and push of that worktree.
3. Fix the SVM checkpoint deserialization blocker before a long experiment. Only promote SVM features into nightly training after the agreed walk-forward/holdout performance gate; merging research code alone is not promotion.

