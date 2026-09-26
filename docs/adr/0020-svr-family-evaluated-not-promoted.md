# 0020: LinearSVR evaluated for the Fit ensemble — promising, not promoted

Date: 2026-09-26
Status: Proposed (evaluation run once; promotion decision open — the addendum
below is the handoff brief for the agent continuing this work)

## Context

An SVM primer conversation raised the question: can Support Vector Machines
help this app? The same session also implemented the ListFold paper
(arXiv:2104.12484, see ADR 0018) — note carefully: **that paper does not use
SVMs anywhere.** Its scoring function is a small MLP; its contribution is the
listwise loss. The SVM thread is a separate, Fit-side (return-regression)
question, orthogonal to Rank/ListFold.

The governing precedent is ADR 0008: a new model family earns ensemble weight
empirically through a solo-vs-blend search where the solos are always
candidates, or it doesn't ship. RandomForest, NeuralNet, and Ridge all went
through that gate and all lost to solo LightGBM.

## Decision (so far)

`train_svr` (LinearSVR in the impute→scale→regress Pipeline, `model.py`) and
`svr_search.py` (the cached-predictions weight search) landed in PR #131.
LinearSVR, not kernel SVR — kernel fitting is ~quadratic in rows and the
pooled dataset is 443k rows; the linear formulation is the only SVM variant
that scales here. Its epsilon-insensitive loss is the family's actual point of
difference vs ridge (MAE-like: errors inside the tube are free, outside grow
linearly — the same robustness rationale as production Fit's MAE objective).

**`svr` is registered in `MODEL_TRAINERS` but deliberately NOT in
`PREDICTIVE_MODEL_TYPES`** — no Models-tab or production exposure until the
follow-up below settles it. Unlike RF/NN/Ridge, the first search run was NOT
a clean reject, which is why this record is Proposed rather than Rejected.

## Evidence — the one search run (2026-09-26, 94 min)

443,009 pooled rows, 1,800 train tickers, 200 holdout tickers untouched,
4 walk-forward folds. Fold fit times (lightgbm+svr+ridge together, SVR
dominates): 400s / 1,002s / 1,649s / 2,552s — LinearSVR is 7–43 min per fold
and grows with rows. Mean across folds, sorted by directional accuracy:

| weights (lgbm,svr,ridge) | mae | acc | rank_ic | gated_n | gated_hit | gated_avg |
|---|---|---|---|---|---|---|
| (0,1,0) solo SVR | 0.01929 | **0.5076** | 0.0344 | 15463 | 0.5163 | 0.0020 |
| (1,3,0) | 0.01871 | 0.5066 | 0.0396 | 12847 | 0.5145 | 0.0021 |
| (1,1,1) | 0.01922 | 0.5061 | 0.0469 | 9110 | 0.5177 | 0.0027 |
| (1,1,0) | 0.01822 | 0.5061 | 0.0456 | 10582 | 0.5142 | 0.0022 |
| (3,1,0) | 0.01787 | 0.5047 | **0.0490** | 9627 | 0.5151 | 0.0021 |
| (3,1,1) | 0.01848 | 0.5046 | 0.0496 | 8621 | 0.5125 | 0.0022 |
| (3,0,1) | 0.01840 | 0.5040 | 0.0437 | 9539 | 0.5048 | 0.0015 |
| (0,0,1) solo ridge | 0.02161 | 0.5039 | 0.0249 | 15319 | **0.5247** | **0.0028** |
| (1,0,0) solo LightGBM (prod) | **0.01781** | 0.5025 | 0.0421 | 11851 | 0.5033 | 0.0012 |

How to read this honestly:

- **The encouraging signal**: solo SVR tops fold accuracy, and the (3,1,0)
  lgbm+svr blend beats solo LightGBM on accuracy, Rank IC, gated hit, AND
  gated avg simultaneously. Compare (3,1,0) against (3,0,1) — swapping SVR
  for ridge in the same blend shape is better on every one of those — which
  is the evidence that epsilon-insensitive loss adds something ridge (already
  evaluated and rejected) does not.
- **The caution**: margins are small (~0.2–0.5pp), the run logged only fold
  MEANS (per-fold logging was added right after, in the same PR, but never
  re-run), and one hot fold can carry a blend. Solo ridge topping gated_hit /
  gated_avg while being nearly worst on IC is a hint these gated stats are
  noisy at this sample.
- Solo LightGBM still wins MAE — expected; MAE is its objective.

---

# Addendum: SVM handoff brief

Written for the (cheaper) agent continuing this work. Read
`docs/delegation-briefing.md` first for the repo's non-negotiables, style,
and traps — this addendum is the SVM-specific state on top of it.

## A. What was read / learned this session (the knowledge, so you don't re-derive it)

1. **SVM primer takeaways** (from the conversation that started this):
   SVC classifies direction, SVR regresses the return — this app's Fit
   predicts a continuous return, so SVR is the fit-for-purpose variant.
   Kernel trick = implicit projection to a higher-dimensional space where a
   linear separator exists (nonlinear boundary in the original space).
   Feature scaling is mandatory (margin geometry is scale-sensitive) — the
   Pipeline's StandardScaler handles it. Kernel methods get painful in the
   hundreds-of-thousands of rows; we measured exactly that constraint.
2. **The ListFold paper does NOT use SVMs.** If anyone says the "paper we
   read" used SVM heavily — no. Its scorer is a 68×136×272×34×1 MLP. Do not
   burn time looking for SVM ideas in it.
3. **SVMs cannot cluster.** They are supervised. The "SVM-flavored
   PCA/cluster" idea is **Kernel PCA** (the kernel trick applied to
   unsupervised PCA). This repo already runs linear PCA (5 components) +
   MiniBatchKMeans (k=8) per day in `features/structure.py` → `pca_1..5`,
   `cluster_id`, `cluster_overnight_gap`. Kernel PCA would be a drop-in
   nonlinear upgrade of that step (O(n²) on the ~1,800-name daily
   cross-section — feasible; it's rows that kill kernels, and a day is small).
4. **RankSVM (Joachims 2002)** is the classic pairwise learn-to-rank SVM:
   classify score differences on pairs (x_i − x_j). It is the pairwise
   counterpart to ListFold's listwise loss — a legitimate future
   `objective_search.py` candidate for Rank, but it needs pair sampling
   (~2,000 names/day → don't generate all pairs).
5. **Where SVMs can contribute without being the final model**: a fitted
   SVM's `decision_function` (signed distance to boundary) as one more
   engineered feature for LightGBM (stacking). See the leakage trap in
   section D before attempting.

## B. Exact current state (what exists, where)

- `python/stock_picker/training/model.py` — `SVR_DEFAULT_PARAMS`
  (C=1.0, epsilon=0.001 ≈ 0.1% return, loss=epsilon_insensitive,
  max_iter=10000, random_state=0; each value's why is commented) and
  `train_svr` (ridge's exact Pipeline pattern). In `MODEL_TRAINERS`; NOT in
  `PREDICTIVE_MODEL_TYPES`.
- `python/stock_picker/training/svr_search.py` — the search harness. Solos
  always included; walk-forward folds only; holdout never touched. Since the
  run above, it also logs **per-fold acc and rank IC per weight candidate**
  ("acc by fold" / "rank_ic by fold" lines) — the stability evidence the
  first run lacked.
- `python/stock_picker/training/tests/test_model.py` — two `train_svr` tests
  (learns a clear signal; imputes NaNs).
- `docs/delegation-briefing.md` — has the P1 "finish the SVR decision" item;
  this ADR supersedes it as the detailed brief.
- The run log above lived in /tmp (gone after reboot); every number that
  matters is in the table in this record.

## C. The work queue, in order

1. **Rerun the search for stability** —
   `bazelisk run //python/stock_picker/training:svr_search` as a background
   job (expect ~1.5–2h; it pins all cores — don't run alongside a retrain).
   Decision input: does the (3,1,0)-or-(1,1,1) edge over (1,0,0) hold in
   **≥3 of 4 folds** on both acc and rank IC, or was it one hot fold?
2. **If stable → one holdout confirmation.** Pick THE one winning blend and
   score it once against solo LightGBM on the 200 never-seen tickers (fit on
   full pooled train; mirror the holdout block pattern at the end of
   `objective_search.py`). Both the fold stability AND the holdout must favor
   the blend. Holdout is scored once, at the end, never inside a loop.
3. **If both pass → promote**: add "svr" to `PREDICTIVE_MODEL_TYPES`, set
   `DEFAULT_MODEL_SPECS` to the winning weights, retrain through
   `run_training` (Models tab or `//python/stock_picker/training:main`), and
   flip this ADR to Accepted with the holdout numbers. Cost to state in the
   PR: nightly 3:30 CT retrain grows by ~35–45 min (full-train LinearSVR
   fit); live scoring cost is negligible (linear predict). Note
   `ensemble_feature_names` (ensemble.py) already unions members' features,
   so live-row building automatically accounts for an SVR member.
4. **If either fails → close it out**: keep trainer + script (RF/NN/Ridge
   precedent), flip this ADR to Rejected with the numbers. Done.
5. **Only after 1–4 settles, optional SVM explorations** (each its own
   research script + the same empirical bar; time-box them):
   a. Bounded C × epsilon grid for LinearSVR (2×3 grid max — fold time is
      the budget constraint, not ideas).
   b. Nystroem kernel approximation + LinearSVR/SGDRegressor — the only
      scalable way to test an "RBF-ish" nonlinear SVM lens at 443k rows.
   c. Kernel PCA swap for `features/structure.py`'s linear PCA (per-day
      cross-section, so O(n²) is fine); evaluate like any feature change —
      walk-forward IC/acc delta, holdout once.
   d. RankSVM as an `objective_search.py` candidate (sampled pairs).
   e. Out-of-fold SVM `decision_function` as a stacked feature (see trap D3).

## D. Traps specific to this work

1. **Never kernel-SVR the pooled rows.** 443k rows × O(n²) does not finish.
   Rows kill kernels; the per-day cross-section (~1,800) does not.
2. **LinearSVR wall-clock grows superlinearly with rows** (400s → 2,552s
   across folds 1→4). Always background the search; never let it collide
   with the 3:30 CT retrain or an 8:30 morning window.
3. **Stacked-feature leakage**: an SVM feature fed to LightGBM must be
   produced out-of-fold (fit on data strictly before the rows it scores).
   Fitting once on all rows and appending its predictions as a column leaks
   the label. This is the single easiest way to produce a fake win here.
4. **Don't compare across different fold windows.** Numbers in this ADR came
   from one dataset snapshot (2026-09-26, 443,009 rows). Nightly ingestion
   grows the data; rerun baselines in the same run as candidates (svr_search
   already does — just don't hand-mix old and new tables).
5. **ConvergenceWarnings / imputer warnings during fits are known**: the
   all-NaN early-window columns (`momentum_spread_20_120d`, `volatility_120d`)
   warning is benign; max_iter is already raised to 10000.
6. The generic repo traps (holdout discipline, gazelle broken → hand-edit
   BUILD files, no API auto-reload) are in `docs/delegation-briefing.md`.

## E. Reflections (what the 12 hours taught us about the app)

- **The empirical gate keeps paying for itself.** The instinct "SVMs seem
  like they could help" was worth exactly one search run — and unlike RF/NN,
  it survived it. Neither hype nor dismissal; the numbers earned SVR a
  follow-up and nothing more.
- **SVR's value hypothesis is diversity, not supremacy.** Nobody expects
  LinearSVR to beat LightGBM solo (nonlinear trees on tabular data are the
  stronger scorer). The question is whether its epsilon-insensitive *linear*
  view is different enough from the trees' view that blending reduces
  correlated errors. The (3,1,0)-vs-(3,0,1) comparison is the cleanest
  evidence yes — ridge, the incumbent linear lens, was already rejected, so
  "different from ridge" is the bar that matters.
- **Fold means without per-fold spread nearly caused a wrong call** in both
  directions — the fix (per-fold logging) is one line of discipline every
  future search here should copy.
- **Small margins on 500k rows are still small margins.** Directional
  accuracy differences of 0.2–0.5pp sit inside regime noise (fold 3 was bad
  for everything, all session). The gated (0.5%-threshold) stats — the ones
  production actually trades on — are noisier still. Hence: stability rerun
  + holdout, not enthusiasm.
- **Compute is a real design input for this family.** A ~40-min full-train
  fit changes nightly operations; that cost belongs in the promotion
  decision, stated in the PR, not discovered after.
