# Overnight open forecast — implementation plan

Status: **proposed for review; no application code implemented**  
Scope: a separate model and Overnight tab first; historical What if replay later.

## Decision this feature supports

During the trading day, answer: **"I normally sell by end of day, but if this stock closes around its current price, might holding it until the next open be better?"** This is a conditional scenario, not a claim that today's final close is known. The default comparison is against exiting by today's close at the assumed price, with explicit uncertainty about both that exit price and tomorrow's executable open. It must remain separate from Rank and Fit, whose target is today's open-to-close return (`python/stock_picker/training/dataset.py:21-48`, `README.md:157-164`).

For session *t*, define the training label as `overnight_gap_t = Open_(t+1) / Close_t - 1`, where *t+1* is the **next actual exchange session**, not necessarily the next calendar day. At inference, substitute an explicitly labeled `assumed_close` for `Close_t` and calculate `projected_open = assumed_close × (1 + predicted_gap)`. For example, previous close $10, today's open $9, and assumed close $9.50 yield a known opening gap of −10% and a hypothetical open-to-close move of +5.56%; the next open is **not** derivable from those three prices without a trained model. A predicted +2% overnight gap would imply $9.69, purely as an illustration.

This first model estimates a **close-conditioned overnight gap**. It does *not* predict the remaining intraday move. If the price changes before the real close, the user changes or refreshes the assumption and reruns the forecast. The tab should also rerun the model for **assumed close − $0.05, assumed close, and assumed close + $0.05** (with an editable step for other price scales); each scenario has its own model inputs, predicted gap, projected open, and implied hold-versus-exit difference. A future direct intraday-price→next-open model would require timestamped historical intraday observations and a different training/evaluation contract.

## Existing architecture and allowed patterns (Phase 0: documentation discovery)

- `PriceStore.read(ticker)` supplies per-ticker daily OHLCV; stores accept an injectable `data_dir` for isolated tests (`python/stock_picker/storage/price_store.py:14-28`, `docs/adr/0005-repository-stores.md:13-21`).
- `ModelStore.write/read/exists(name)` can persist a **separately named** overnight artifact atomically (`python/stock_picker/storage/model_store.py:24-63`). The current training path archives a run before replacing its latest artifact (`python/stock_picker/training/main.py:259-266`).
- `select_holdout_tickers` and `walk_forward_splits` are the established validation primitives (`python/stock_picker/training/splits.py:19-56`, `docs/adr/0004-walk-forward-and-ticker-holdout.md:13-26`).
- `GET /api/positions` exposes open lots and `current_price`; its response does **not** include a quote timestamp (`python/stock_picker/api/routes.py:400-425`, `python/stock_picker/api/models.py:106-130`). `fetchPositions` and `fetchQuotes` are the existing UI clients (`typescript/src/api.ts:508`, `typescript/src/api.ts:629-643`).
- The tab switch and panel pattern are in `typescript/src/App.tsx:12-21,40-76`; the historical What if UI and pure calculation pattern are in `typescript/src/components/WhatIf.tsx:366-534` and `typescript/src/whatIf.ts:81-165`.
- Keep the API thin and business rules in tested lower layers (`docs/adr/0002-layered-architecture.md:8-23`).

Before implementation, verify the installed LightGBM API and the paid market-data provider's historical/open, timestamp, corporate-action, and calendar semantics against their **current official documentation**. Web documentation was unavailable during this planning pass, so no new vendor method or parameter is asserted here. The existing repo's LightGBM `Dataset`/`train` calls at `python/stock_picker/training/model.py:205-217` are the local copy-ready pattern, subject to that verification.

## Phase 1 — Build a lookahead-safe daily overnight dataset

**Implement:** Add a dedicated dataset builder in `training/` that aligns each session's actual close with the following valid session's open from `PriceStore`. Build a small, explicitly enumerated feature set available in both historical training and the live scenario: prior completed-session features, today's already-observed open, and today's **assumed** close and derived returns. Store the feature order and definitions with the model. Use the same pure feature builder for actual historical closes and hypothetical live closes; changing the assumed close must recompute every close-derived input, not merely rescale a fixed output. Follow `training/dataset.py:33-48` for an explicit label/feature contract, but create a *different* target and builder.

**Data quality:** Validate positive finite prices, unique ordered sessions, missing bars, and the next exchange session. Verify raw-versus-adjusted price consistency and handle split/ex-dividend discontinuities explicitly. Yahoo history is requested with `auto_adjust=False` (`python/stock_picker/ingestion/yfinance_client.py:48-55`), while the Finnhub backfill sets `Adj Close = Close` (`python/stock_picker/ingestion/refresh_prices.py:55-63`); those sources must not be silently mixed across an overnight label. Log excluded rows and reasons rather than silently clipping real large moves.

**Verify:** Unit-test the $10→$9→$9.50 example, Friday→Monday/holiday alignment, missing-session rejection, split/price-basis cases, and feature equality between historical and scenario construction.

**Do not:** Use the next session's open as a feature; reuse morning `overnight_gap` (it already requires that open); feed today's final high/low/volume or a full-day feature row into a forecast made before the close (`training/inference.py:32-78`). In particular, a same-day news feature can include after-close articles (`python/stock_picker/features/news.py:3-7,91-118`); exclude it unless timestamp cutoffs are implemented and tested.

## Phase 2 — Train, save, and evaluate a separate overnight model

**Implement:** Train a standalone LightGBM return regressor using the Phase 1 frame and a separate model name, artifact metadata, and command. Copy the existing fitted-booster and atomic store patterns (`python/stock_picker/training/model.py:205-217`, `python/stock_picker/storage/model_store.py:34-60`). Do not call the existing `train_lightgbm` or `evaluate` unchanged: they select the day-session `LABEL_COLUMN` (`python/stock_picker/training/model.py:21-29,205-217`). Training must not overwrite Rank/Fit. Arrange refresh only after the needed daily bars are complete; do not retrain on each forecast request.

**Evaluate:** Use chronological out-of-time folds and the existing ticker holdout. Report sample/coverage counts, MAE for gap and implied opening price, directional accuracy, and performance by weekday/weekend and large-gap bucket. Compare with at least `next_open = assumed_close` and a historical-gap baseline. If displaying a range or probability, calibrate it using **out-of-fold residuals**, not in-sample fit. Keep the model visible even if its signal is small; show the measured uncertainty rather than introducing an arbitrary promotion gate.

**Verify:** Reproduce a saved model's predictions after reload; prove every train date precedes its test dates; test a missing/stale artifact and a run that fails before publish.

**Do not:** Blend overnight percentages with open→close Rank/Fit scores, tune on the ticker holdout, or report a backtest of actual end-of-day closes as proof that a *midday* hold decision would have worked (`docs/adr/0004-walk-forward-and-ticker-holdout.md:13-26`).

## Phase 3 — One reusable forecast service and thin API

**Implement:** A pure scenario function accepts ticker history through the last completed session, today's observed open, `assumed_close > 0`, model artifact, and an as-of timestamp. It returns predicted overnight gap, implied next open, model/version metadata, next session, and data-freshness/coverage status. Add a proposed `POST /api/overnight/forecast` request/response in `api/models.py` and a thin wrapper in `api/routes.py` (existing route pattern: `python/stock_picker/api/routes.py:180-211,400-425`). Define the forecast result independently of the UI so What if can later replay it. If an interval/probability is supported by Phase 2 calibration, include its methodology and sample count.

**Verify:** API tests for a valid scenario, invalid/nonfinite assumed close, missing open/price history, stale current quote, missing model, weekend/holiday next-session date, and fixed response serialization.

**Do not:** Label `current_price` as a confirmed close or silently use a stale quote. Existing quote summaries lack timestamps (`python/stock_picker/features/quotes.py:28-57`); add trustworthy source/as-of metadata or require the user to enter the assumed close manually when freshness cannot be established.

## Phase 4 — Overnight tab, clearly framed as a scenario

**Implement:** Add `Overnight` to `typescript/src/App.tsx:12-21` and mount a focused component using the established panel and API-client patterns (`typescript/src/App.tsx:70-76`, `typescript/src/api.ts:476-508`). Show open positions first, with ticker search for any covered name. Default the editable **Assumed close** field from a fresh current quote when available; otherwise leave it blank. Place three compact scenarios side by side: −$0.05, base, +$0.05, with a configurable step. Recompute the model's close-derived features and prediction for **each** price, rather than adjusting one predicted open mechanically. Show next-session date, predicted open, expected dollar/% change versus each assumed close, uncertainty if calibrated, quote/model as-of times, and a plain statement: “Assumes this price is today's close; remaining intraday movement is not predicted.” Keep buying cost basis separate from the incremental exit-today-versus-hold-overnight comparison.

**Verify:** Type-check/build the UI; test all three scenario prices, independent model calls/feature recomputation, positive-price validation, and formatting; check empty holdings, missing quote, stale model, small screens, and clear error states. Follow the existing open-lot filter (`typescript/src/components/TradeHistory.tsx:475-483`) and numeric-input handling (`typescript/src/components/AddTradeForm.tsx:33-42,88-103`).

**Do not:** Show an exact-looking forecast without its assumption and time, imply an automatic trade recommendation, or replace the morning Trading view.

## Phase 5 — Historical What if replay (follow-on, designed now)

**Implement later:** Reuse the exact Phase 1 feature builder and Phase 3 forecast result to replay each historical close→next-open prediction in What if. Persist or select model provenance for each replay; a model trained on later dates must **not** be used to claim historical out-of-sample performance. Show actual close, predicted next open, actual next open, error, and hypothetical hold-versus-sell result. Keep these overnight rows distinct from the current morning open→close paper book (`typescript/src/whatIf.ts:81-165`, `python/stock_picker/paper/book.py:22-62`).

**Verify:** Fixture replay equals the standalone forecast for identical model/features/as-of inputs; an out-of-time test refuses future-trained artifacts; weekend/holiday and missing-next-open rows show explicit status.

**Do not:** Retrofit overnight returns into existing morning P&L totals or call an after-close replay a validated intraday strategy. Honest historical *intraday* replay is a separate future phase requiring archived timestamped intraday prices/quotes.

## Final verification and review questions

Run focused Python/API tests, TypeScript build/tests, Bazel build/tests, and an end-to-end scenario against the saved model. Check LSP/type diagnostics after edits. Review the exact historical feature cutoff, price adjustment basis, next-session calendar, quote timestamp, out-of-fold interval calibration, and model artifact provenance before accepting displayed probabilities or What if performance.

Questions for Claude/Grok review:

1. Does the close-conditioned scenario remain lookahead-safe at every proposed feature boundary, especially news and same-day OHLCV?
2. Which provider fields and adjustment rules reliably pair `Close_t` with the **actual next-session** open across splits, dividends, missing sessions, and holidays?
3. Are the baselines, chronological evaluation, and uncertainty display sufficient to prevent one anecdotal STLA outcome from being mistaken for repeatable edge?
4. Does the API/UI make “assumed close” unmistakable while the market is open, and can What if later replay the same model without hindsight?
