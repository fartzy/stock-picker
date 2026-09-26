# 0018: Rank objective is ListFold (custom listwise loss), replacing lambdarank

Date: 2026-09-26
Status: Accepted (PR #129)

## Context

The Rank model (0015) used LightGBM's built-in lambdarank with ups-only
relevance grades: every down session was grade 0, so the model was never
taught to order the bottom of the day, and pair generation truncated near
the top. Its measured quality was weak -- mean walk-forward Rank IC 0.017,
below even the MAE regression model's 0.045, and unstable across folds.
The ListFold paper (Zhang, Wu, Chen, arXiv:2104.12484) proposes a listwise
loss built from stepwise (top, bottom) pair selection -- shift-invariant,
Plackett-Luce-interpretable, and attentive to both ends of the list.

## Decision

Implement ListMLE and ListFold-exp as vectorized custom LightGBM objectives
(`training/listwise.py`, gradients/hessians finite-difference-verified) and
bake them off in `objective_search.py` against mae, huber, lambdarank, a
tuned lambdarank (deeper truncation, finer grades), and rank_xendcg. The
documented promotion bar: beat the incumbent on Rank IC AND top-20 session
return. ListFold cleared it everywhere -- fold Rank IC 0.0995 vs 0.0166,
holdout (200 never-seen tickers) 0.2206 vs 0.0275, top-20 hit 0.634 vs
0.501 -- and was the only candidate IC-positive in every fold. The tuned
lambdarank control shows the gap is the objective, not its knobs.
`train_lightgbm_rank` now trains ListFold; model_type and pickle name are
unchanged, so storage/serving/UI paths are untouched.

## Consequences

- Rank scores remain relative (top-K serving unchanged); the pickle
  refreshes through the normal nightly retrain.
- ListFold orders the bottom of the list too, which makes a future short
  leg / "what to avoid" list measurable for the first time.
- Custom-objective training adds a per-round Python callback (~equal fit
  time at current scale); the objective callable pickles safely (tested).
