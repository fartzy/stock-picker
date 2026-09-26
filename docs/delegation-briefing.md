# Delegation briefing: running cheaper models on this repo

Instructions for delegating stock-picker work to a less expensive model
(Sonnet/Haiku-class). Written 2026-09-26 by the session that shipped ListFold
(PR #129) and the SVR evaluation. Read CLAUDE.md first -- this file does not
repeat it; it adds the judgment layer: what to do, in what order, in what
style, and when to stop and escalate.

## The one-paragraph mental model

This app answers "what do I buy at today's open and sell at today's close."
Two models score every morning: **Fit** (LightGBM MAE regression, only shows
names predicted above +0.5%) and **Rank** (LightGBM with the ListFold
listwise objective, shows the top 20 by relative score). Everything else --
stores, features, jobs, UI -- exists to feed and audit those two lists. Every
architectural decision has a numbered record in `docs/adr/`; skim the index
before proposing structural change.

## Non-negotiables (violating any of these is a stop-the-line bug)

1. **Lookahead safety.** Day t is predicted from features through t-1 plus
   open-known columns only (ADR 0003, `training/dataset.py`). Any new feature
   must be classified close-known vs open-known. If you cannot argue the
   classification in one sentence, escalate.
2. **Never k-fold; never tune on holdout.** Validation is date-based
   walk-forward; the ~200 held-out tickers are scored once, after a winner is
   chosen (ADR 0004). A search loop that reads holdout metrics is invalid by
   construction, even if its conclusion looks right.
3. **Empirical promotion bars.** Nothing joins production because it's newer
   or fancier. A new objective must beat the incumbent on Rank IC AND top-20
   session return (see `objective_search.py`); a new model family must beat
   solo LightGBM in the solo-vs-blend search (ADR 0008). Report losing
   numbers just as prominently as winning ones.
4. **Don't touch live serving state during market hours.** `data/models/*.pkl`
   is read live by the API. Code changes ride branches; pickle refreshes
   happen via the nightly 3:30 PM CT retrain or an explicit user-clicked
   training run. Never regenerate/commit pickles as a side effect.
5. **Every change through a branch + PR**, squash-merged, with `bazelisk test
   //...` green. Pull main first; other sessions work this repo concurrently.

## Style: how code is written here

- **Comments state constraints and reasons, not narration.** The codebase's
  docstrings say *why the design is what it is* ("k-fold would leak future
  rows") -- match that. Never write comments that describe the diff.
- **Magic numbers get names.** Thresholds, window sizes, grade counts --
  module-level constants with a comment saying why that value.
- **Stores are the only persistence path**, and every store takes
  `data_dir` for test isolation (ADR 0005). Tests construct
  `Store(data_dir=tmp_path)`; never monkeypatch paths.
- **New features update three parallel views** (descriptions, formulas,
  examples) or their completeness tests fail (ADR 0007). This is intentional
  friction; do not weaken the tests.
- **Rendered UI text**: no em-dashes, no "--" placeholders -- write a word
  that says why a value is absent ("pending", "no model yet").
- **Research scripts are permanent tools**, not throwaway: docstring at top
  states what question the script answers, what it deliberately does NOT do
  (e.g. "never touches holdout"), and the exact `bazelisk run` line.

## Known traps (each has burned a session already)

- **gazelle is broken repo-wide** (phantom `stock_picker.log` ambiguity, since
  ~2026-09-26). Edit BUILD.bazel files manually, mirroring an existing target
  in the same package. Do not "fix" this in passing; it's a scoped task.
- **A py_binary whose module is also a py_library source cannot be imported
  across that boundary** (gazelle limitation, documented in CLAUDE.md). Copy
  the `_load_pooled` pattern from `objective_search.py` instead of importing
  `training.main`.
- **The API server has no auto-reload** -- restart it after any Python change.
- **LightGBM is 4.x**: custom objectives go in `params["objective"]` as a
  callable `(preds, dataset) -> (grad, hess)`; the old `fobj=` argument is
  gone. Hessians must be positive -- floor them (see `listwise.py`).
- **Holdout metrics look inflated** vs live (dates overlap training even
  though tickers don't). Only compare candidates to each other on the same
  split, never to live hit rates.
- **LinearSVR at ~450k rows takes tens of minutes per fold.** Anything
  slower than LightGBM by 10x should run as a background job with progress
  logging per fold, or be down-scoped first.

## Work queue (in priority order)

**P0 -- verify the ListFold rollout (first nightly retrain after PR #129
merges).** Check the Models tab run history recorded the retrain, the Rank
list renders ~20 names the next morning, and Rank IC on the run's holdout
metrics looks like the bake-off (>0.15), not like the old lambdarank
(~0.03). If it regressed, revert is: retrain from the previous selected run
(ADR 0013) -- not a code revert.

**P1 -- surface consensus in the morning UI (ADR 0019).** Mark Fit names
that are also in Rank's top-20 (a badge/pill on both lists). Do NOT change
any default gating -- evidence gathering only; the What if book measures it
live for a few weeks. Acceptance: marker visible on Trading + What if replay,
scan persistence unchanged, tests for the intersection helper reused from
`backtest.simulate_consensus` (don't re-derive it in TS; expose it on the
scan payload from the API).

**P1 -- decide SVR from the numbers** in `training/svr_search.py`'s run (or
rerun it). Rule: if no blend beats solo LightGBM on fold directional
accuracy AND rank IC, close it out -- keep the trainer + script, don't add
"svr" to PREDICTIVE_MODEL_TYPES; record a short ADR ("evaluated, rejected,
numbers"). Precedent: RandomForest/NeuralNet/Ridge all ended there.

**P2 -- bottom-of-list evidence.** ListFold orders the bottom too (lambdarank
never did). Add bottom-20 tracking to the What if paper book (do the day's
lowest-ranked names actually underperform?). Strictly observational; no
shorting UI. This is the cheapest test of whether a short leg ever becomes
real (the ListFold paper's actual use case).

**P2 -- volatility-normalized gate, revisited** (ADR 0011 left it open). The
diagnostic exists in `tune_experiment.py`; extend the comparison to include
the consensus gate, since consensus may already capture what vol-normalization
was for. Promotion bar: better stability across holdout halves at matched
trade count.

**P3 (nice to have) -- ListFold-specific tuning.** Current Rank uses
regression-tuned LightGBM params (lr 0.03, 100 rounds, leaves 31). One
bounded pass over lr x rounds x leaves for the ListFold objective, walk-forward
only, same promotion bar as always. Also worth one candidate each:
position-weighted ListFold (emphasize the top-20 steps), and NDCG@±k as an
extra reported metric (paper Sec 4.3.2). Expect modest gains; time-box it.

**P3 (nice to have) -- root-cause gazelle.** Reproduce on clean main, bisect
recent commits (suspects: `python/stock_picker/paper/__init__.py`, langfuse
additions), fix, delete the manual-BUILD memory note and the trap above.

## When to stop and escalate to a more expensive model

- Anything that changes `dataset.py` label/shift semantics, or where you
  find yourself unsure whether information is knowable at the open.
- Deriving new loss gradients/hessians or debugging why a custom objective
  trains worse than its metrics imply (subtle-math territory; the
  finite-difference test pattern in `test_listwise.py` is the floor, not the
  ceiling).
- Empirical results that contradict each other across splits (e.g. folds
  say A, holdout says B) -- do not average away a contradiction; escalate it.
- Any trade-affecting behavior change (gates, skip rules, scheduling) beyond
  what a P-item above explicitly authorizes.

## Definition of done, every time

Branch -> code + tests -> `bazelisk test //...` green -> for model claims: a
research-script run whose log shows walk-forward numbers (and holdout only
for a chosen winner) -> PR whose description contains the actual numbers ->
if the change decides or reverses anything architectural, an ADR in the same
PR.
