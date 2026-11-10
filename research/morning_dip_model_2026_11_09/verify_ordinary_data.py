"""Read-only checks after ordinary2y prices and1y feature snapshots rebuild.

Use the current ordinary pipeline, deliberately not prepare_baseline's frozen
research-module resolver. Reports are written only under this research folder.
"""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[variable] = "1"
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUNFILES = (ROOT / "bazel-bin/python/stock_picker/api/main.runfiles").resolve()
sys.path[:0] = [str(ROOT / "python"), *map(str, RUNFILES.glob("rules_python++pip+*/site-packages"))]

import numpy as np
import pandas as pd

from stock_picker.features.dip_context import DIP_COLUMNS, complete_dip_history, morning_dip_features
from stock_picker.ingestion.session import last_completed_session_date
from stock_picker.training.dataset import build_training_frame
from stock_picker.training.inference import build_inference_row
from stock_picker.training.splits import select_holdout_tickers


def main():
    cutoff = last_completed_session_date()
    lower = (pd.Timestamp(cutoff) - pd.DateOffset(years=1)).date()
    registry = pd.read_parquet(ROOT / "data/universe/registry.parquet")
    names = sorted(registry.index[registry.active].astype(str))
    held = set(select_holdout_tickers(names))
    records, failures = [], []
    total_rows = 0
    training_rows = 0
    training_dates = set()
    nonmissing = {column: 0 for column in DIP_COLUMNS}
    latest_valid = 0
    for ticker in names:
        prices_path = ROOT / "data/prices" / f"{ticker}.parquet"
        features_path = ROOT / "data/features" / f"{ticker}.parquet"
        if not prices_path.exists() or not features_path.exists():
            records.append({"ticker": ticker, "missing_store": True})
            continue
        prices, features = pd.read_parquet(prices_path), pd.read_parquet(features_path)
        missing = set(DIP_COLUMNS) - set(features.columns)
        if missing:
            failures.append({"ticker": ticker, "missing_columns": sorted(missing)})
            continue
        dates = pd.DatetimeIndex(features.index).date
        if features.index.has_duplicates or np.any(dates <= lower) or np.any(dates > cutoff):
            failures.append({"ticker": ticker, "invalid_snapshot_dates": True})
        values = features[list(DIP_COLUMNS)]
        if np.isinf(values.to_numpy()).any():
            failures.append({"ticker": ticker, "infinite_dip_values": True})
        total_rows += len(features)
        training_frame = build_training_frame(prices, features)
        if ticker not in held:
            training_rows += len(training_frame)
            training_dates.update(training_frame.index.date)
        for column in DIP_COLUMNS:
            nonmissing[column] += int(values[column].notna().sum())
        if len(features):
            latest_valid += int(features.iloc[-1].dip_low_history_valid == 1)
        ohlc = prices[["Open", "High", "Low", "Close"]]
        records.append({
            "ticker": ticker, "price_rows": len(prices), "feature_rows": len(features),
            "eligible_training_rows": len(training_frame), "heldout_ticker": ticker in held,
            "last_price_date": str(prices.index.max().date()) if len(prices) else None,
            "first_feature_date": str(features.index.min().date()) if len(features) else None,
            "last_feature_date": str(features.index.max().date()) if len(features) else None,
            "invalid_ohlc_rows": int((~np.isfinite(ohlc).all(axis=1) | ~ohlc.gt(0).all(axis=1)).sum()),
            "latest_low_history_valid": float(features.iloc[-1].dip_low_history_valid) if len(features) else None,
        })
    mrcy_prices = pd.read_parquet(ROOT / "data/prices/MRCY.parquet")
    mrcy_features = pd.read_parquet(ROOT / "data/features/MRCY.parquet")
    if len(mrcy_prices) < 253 or mrcy_features.iloc[-1].dip_low_history_valid != 1:
        failures.append({"ticker": "MRCY", "long_history_not_ready": True})
    aligned = build_training_frame(mrcy_prices, mrcy_features)
    reference = morning_dip_features(complete_dip_history(mrcy_prices))
    recent = aligned.index[-5:]
    np.testing.assert_allclose(aligned.loc[recent, list(DIP_COLUMNS)], reference.loc[recent, list(DIP_COLUMNS)], equal_nan=True, rtol=1e-10, atol=1e-12)
    entry = recent[-1]
    prior_prices = mrcy_prices.loc[mrcy_prices.index < entry]
    prior_features = mrcy_features.loc[mrcy_features.index < entry]
    served = build_inference_row(
        prior_features.iloc[-1], float(mrcy_prices.loc[entry, "Open"]),
        float(prior_prices.iloc[-1].Close), prior_features.index[-1].date(),
        entry.date(), prior_history=prior_prices,
    )
    np.testing.assert_allclose(served[list(DIP_COLUMNS)].iloc[0], aligned.loc[entry, list(DIP_COLUMNS)], equal_nan=True, rtol=1e-10, atol=1e-12)
    pruned = json.loads((ROOT / "data/pruned_features/pruned.json").read_text())
    pruned_dip_columns = sorted(set(DIP_COLUMNS) & {record["feature"] for record in pruned})
    result = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(), "cutoff": str(cutoff),
        "feature_lower_exclusive": str(lower), "active_tickers": len(names),
        "feature_rows": total_rows, "dip_columns": list(DIP_COLUMNS), "nonmissing_counts": nonmissing,
        "normal_training_rows_excluding_holdout": training_rows,
        "normal_training_dates": len(training_dates),
        "normal_training_from": str(min(training_dates)) if training_dates else None,
        "normal_training_through": str(max(training_dates)) if training_dates else None,
        "heldout_tickers": len(held), "mrcy_heldout": "MRCY" in held,
        "latest_valid_low_history_tickers": latest_valid, "failures": failures,
        "ordinary_pruned_dip_columns": pruned_dip_columns,
        "mrcy": {"price_rows": len(mrcy_prices), "feature_rows": len(mrcy_features),
                 "latest_low_history_valid": float(mrcy_features.iloc[-1].dip_low_history_valid),
                 "training_historical_parity_last5": True, "live_training_parity_date": str(entry.date())},
        "ticker_coverage": records,
    }
    (HERE / "ordinary_data_verification.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "ticker_coverage"}), flush=True)
    if failures:
        raise RuntimeError(f"Ordinary data verification failed: {failures[:5]}")


if __name__ == "__main__":
    main()
