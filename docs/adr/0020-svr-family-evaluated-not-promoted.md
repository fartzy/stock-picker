# 0020: LinearSVR evaluated for the Fit ensemble — rejected as a blender member

Date: 2026-09-26 (closed 2026-09-27)
Status: Rejected as an ensemble member (ADR 0008 bar). Remaining SVM uses —
stacked features, Kernel PCA, RankSVM — live in ADR 0021.

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

## Decision

`train_svr` (LinearSVR in the impute→scale→regress Pipeline, `model.py`) and
`svr_search.py` (the cached-predictions weight search) landed in PR #131.
LinearSVR, not kernel SVR — kernel fitting is ~quadratic in rows and the
pooled dataset is 443k rows; the linear formulation is the only SVM variant
that scales here. Its epsilon-insensitive loss is the family's actual point of
difference vs ridge (MAE-like: errors inside the tube are free, outside grow
linearly — the same robustness rationale as production Fit's MAE objective).

**`svr` stays in `MODEL_TRAINERS` and stays out of `PREDICTIVE_MODEL_TYPES`.**
Same close-out as RandomForest, NeuralNet, and Ridge: trainer + search script
kept, no Models-tab or production blend. The overnight stability rerun
(2026-09-26 21:12–22:44 CDT) failed the promotion bar, so holdout was never
run.

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
- **The caution**: margins are small (~0.2–0.5pp), the first run logged only
  fold MEANS, and one hot fold can carry a blend. Solo ridge topping gated_hit
  / gated_avg while being nearly worst on IC is a hint these gated stats are
  noisy at this sample.
- Solo LightGBM still wins MAE — expected; MAE is its objective.

## Evidence — overnight stability rerun (2026-09-26 21:12–22:44 CDT)

Same 443,009 rows, same 1,800/200 split, same 4 folds. Fold fit times
427s / 929s / 1,596s / 2,540s. Mean table is a copy of the first run
(same seed, same data). The new evidence is per-fold acc and rank IC,
which the first run lacked.

Bar (this record's original queue): `(3,1,0)` or `(1,1,1)` beats solo
LightGBM `(1,0,0)` in **≥3 of 4 folds on both acc and rank IC**.

Per-fold acc | rank IC:

| weights | fold 1 | fold 2 | fold 3 | fold 4 |
|---|---|---|---|---|
| (1,0,0) LightGBM | 0.4877 / 0.0117 | 0.4934 / 0.0651 | 0.5152 / 0.0262 | 0.5136 / 0.0655 |
| (0,1,0) solo SVR | 0.4976 / 0.0399 | 0.4977 / 0.0495 | 0.5196 / −0.0013 | 0.5156 / 0.0497 |
| (3,1,0) | 0.4941 / 0.0294 | 0.4914 / 0.0754 | 0.5204 / 0.0235 | 0.5128 / 0.0676 |
| (1,1,0) | 0.4975 / 0.0368 | 0.4917 / 0.0704 | 0.5214 / 0.0143 | 0.5139 / 0.0609 |
| (1,3,0) | 0.4973 / 0.0391 | 0.4937 / 0.0597 | 0.5205 / 0.0049 | 0.5147 / 0.0548 |
| (1,1,1) | 0.5029 / 0.0488 | 0.4897 / 0.0614 | 0.5211 / 0.0143 | 0.5108 / 0.0632 |

Wins vs LightGBM (acc / rank IC / both ≥3/4):

| blend | acc | rank IC | both ≥3/4 |
|---|---|---|---|
| (3,1,0) | 2/4 | 3/4 | no |
| (1,1,1) | 2/4 | 1/4 | no |
| (1,1,0) | 3/4 | 2/4 | no |
| (1,3,0) | 4/4 | 1/4 | no |

Fold 2 and fold 4 flip by metric. The mean edge was one hot fold carrying
the average. Holdout was never scored — the bar required fold stability
first.

Log: `~/Library/Logs/stock-picker/svr_search.log`. Liblinear
ConvergenceWarnings on every fold are known (max_iter already 10000).

## Consequences

- Production Fit stays `DEFAULT_MODEL_SPECS = [ModelSpec("lightgbm")]`.
- `train_svr` and `svr_search.py` stay as research tools, same as RF/NN/Ridge.
- A ~40-min LinearSVR fit does **not** join the 3:30 CT nightly retrain.
- SVM as extra LightGBM columns (stacking), Kernel PCA, and RankSVM are a
  different question — ADR 0021.

## Reflections

- Fold means without per-fold spread nearly caused a wrong call. The
  per-fold lists are the discipline every future search here should copy.
- SVR's value hypothesis was diversity, not supremacy. The (3,1,0)-vs-(3,0,1)
  comparison still says epsilon-insensitive loss is a different linear view
  from ridge; that difference was not stable enough to pay for a second
  nightly fit. It is exactly the kind of difference stacking can use
  (ADR 0021) without putting SVR on the blender.
