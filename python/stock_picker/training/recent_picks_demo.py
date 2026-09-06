"""Illustrates the confidence-gated strategy concretely: for the most recent
trading days on the held-out (never-trained-on) tickers, shows which picks
the threshold gate would have made and what actually happened.

Held-out tickers are the honest choice for this, not just "the last two
weeks" of any ticker -- the persisted model has never seen these stocks at
all, regardless of date, so a prediction here is a genuine "would this have
worked" test, not a look at rows the model was fit on.

Throwaway demo script, not wired into any BUILD-permanent workflow -- run
directly via `bazel run //python/stock_picker/training:recent_picks_demo`.
"""

from __future__ import annotations

from stock_picker.storage.feature_store import FeatureStore
from stock_picker.storage.model_store import ModelStore
from stock_picker.storage.price_store import PriceStore
from stock_picker.storage.universe_store import UniverseStore
from stock_picker.training.dataset import LABEL_COLUMN, build_pooled_dataset
from stock_picker.training.ensemble import predict_ensemble
from stock_picker.training.splits import select_holdout_tickers

# Matches training/main.py's MODEL_NAME -- not imported directly to avoid
# that module's existing py_binary/py_library ambiguity for gazelle (see
# api/BUILD.bazel's own gazelle:resolve comment for the same collision).
MODEL_NAME = "day_session_return"
THRESHOLD = 0.005
LOOKBACK_TRADING_DAYS = 10  # ~2 trading weeks


def main() -> None:
    tickers = UniverseStore().active_tickers()
    holdout_tickers = sorted(select_holdout_tickers(tickers))

    price_store, feature_store = PriceStore(), FeatureStore()
    histories, features_by_ticker = {}, {}
    for t in holdout_tickers:
        try:
            histories[t] = price_store.read(t)
            features_by_ticker[t] = feature_store.read(t)
        except FileNotFoundError:
            continue
    pooled = build_pooled_dataset(histories, features_by_ticker)

    recent_dates = sorted(pooled["date"].unique())[-LOOKBACK_TRADING_DAYS:]
    recent = pooled[pooled["date"].isin(recent_dates)].copy()

    ensemble = ModelStore().read(MODEL_NAME)
    recent["predicted"] = predict_ensemble(ensemble, recent)

    print(
        f"Over the last {len(recent_dates)} trading days "
        f"({recent_dates[0].date()} to {recent_dates[-1].date()}) across "
        f"{len(holdout_tickers)} tickers this model has never been trained on:\n"
    )

    picks = recent[recent["predicted"] > THRESHOLD].sort_values("date")
    print(f"{len(picks)} of {len(recent)} (ticker, day) rows cleared the {THRESHOLD:.1%} confidence gate:\n")
    print(f"{'DATE':<12}{'TICKER':<8}{'PREDICTED':>12}{'ACTUAL':>12}  RESULT")
    hits = 0
    for _, row in picks.iterrows():
        actual = row[LABEL_COLUMN]
        hit = actual > 0
        hits += hit
        mark = "hit" if hit else "miss"
        print(f"{row['date'].date()!s:<12}{row['ticker']:<8}{row['predicted'] * 100:>+11.2f}%{actual * 100:>+11.2f}%  {mark}")

    if len(picks):
        print(f"\nHit rate: {hits}/{len(picks)} = {hits / len(picks) * 100:.1f}%")
        total_return = picks[LABEL_COLUMN].sum()
        print(f"Sum of realized returns across all picks: {total_return * 100:+.2f}% (equal-weighted, no compounding)")
    else:
        print("\nNo picks cleared the threshold in this window.")


if __name__ == "__main__":
    main()
