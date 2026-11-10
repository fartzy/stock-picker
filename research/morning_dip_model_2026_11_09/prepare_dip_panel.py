"""Join already decision-aligned dip candidates to the immutable baseline keys.

Run after ``prepare_baseline.py --phase assemble`` and independent review of
the final candidate module. This does not fit models or change row eligibility.
The canonical SPY-session histories retain unavailable internal bars as NaN.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import time

# This import sets the one-thread numerical limits before loading pandas/numpy.
import prepare_baseline as baseline

import numpy as np
import pandas as pd
import stock_picker.features.dip_context as dip_module

from stock_picker.features.dip_context import (
    DIP_COLUMNS,
    DIP_FAMILIES,
    DIP_FLAG_COLUMNS,
    morning_dip_features,
)

HERE = baseline.HERE
ROOT = baseline.ROOT
DIP_CACHE = baseline.CACHE_DIR / "dip_features"
DIP_SOURCE = "python/stock_picker/features/dip_context.py"


def prepare_one(ticker: str, dates: pd.DatetimeIndex) -> dict:
    """Compute from the full preceding history; select common model rows last."""
    history = baseline._WORKER_CONTEXT["histories"][ticker]
    frame = morning_dip_features(history).reindex(dates)
    if tuple(frame.columns) != DIP_COLUMNS or len(frame) != len(dates):
        raise RuntimeError(f"Candidate shape mismatch for {ticker}")
    if np.isinf(frame.to_numpy()).any():
        raise RuntimeError(f"Infinite candidate input for {ticker}")
    path = DIP_CACHE / f"{ticker}.parquet"
    frame.rename_axis("date").to_parquet(path, compression="zstd")
    return {
        "ticker": ticker, "rows": len(frame),
        "path": str(path.relative_to(ROOT)), "sha256": baseline.sha256(path),
    }


def prepare_panel(workers: int) -> None:
    started = time.monotonic()
    input_path = HERE / "input_manifest.json"
    input_manifest = json.loads(input_path.read_text())
    baseline_manifest_path = HERE / "baseline_manifest.json"
    baseline_manifest = json.loads(baseline_manifest_path.read_text())
    implementation_path = baseline.CONTEXT_DIR / "dip_implementation.json"
    implementation = json.loads(implementation_path.read_text())
    source_path = baseline.Path(dip_module.__file__)
    source_hash = baseline.sha256(source_path)
    if implementation["source_sha256"][DIP_SOURCE] != source_hash:
        raise RuntimeError("Candidate source differs from its reviewed implementation artifact")
    if baseline_manifest["input_manifest_sha256"] != baseline.sha256(input_path):
        raise RuntimeError("Baseline input manifest changed")
    price_manifest = json.loads((HERE / "price_feature_manifest.json").read_text())
    baseline.verified_read_bytes(baseline.CONTEXT_FILE, price_manifest["context_sha256"])
    if len(DIP_COLUMNS) != 21 or len(set(DIP_COLUMNS)) != 21:
        raise RuntimeError("Unexpected candidate schema")
    before = baseline.verify_production_inputs(input_manifest, require_unchanged=False)
    baseline.verify_frozen_context(input_manifest, price_manifest)
    baseline_path = ROOT / baseline_manifest["path"]
    baseline.verified_read_bytes(baseline_path, baseline_manifest["sha256"])
    keys = pd.read_parquet(baseline_path, columns=["date", "ticker"])
    keys["ticker"] = keys.ticker.astype(str)
    if keys.duplicated(["date", "ticker"]).any():
        raise RuntimeError("Baseline keys are not unique")
    held = set(input_manifest["holdout_tickers"])
    if keys.ticker.isin(held).any() or not keys.ticker.eq("MRCY").any():
        raise RuntimeError("Unexpected universe or MRCY exclusion")
    grouped_dates = {
        str(ticker): pd.DatetimeIndex(group.date)
        for ticker, group in keys.groupby("ticker", sort=True, observed=True)
    }
    DIP_CACHE.mkdir(exist_ok=True)
    baseline.log("dip_panel_started", tickers=len(grouped_dates), rows=len(keys), workers=workers)
    records = []
    with ProcessPoolExecutor(
        max_workers=workers, initializer=baseline.initialize_worker,
        initargs=(str(baseline.CONTEXT_FILE),),
    ) as executor:
        futures = [executor.submit(prepare_one, ticker, dates) for ticker, dates in grouped_dates.items()]
        for count, future in enumerate(as_completed(futures), 1):
            records.append(future.result())
            if count % 200 == 0 or count == len(futures):
                baseline.log("dip_panel_progress", completed=count, total=len(futures))
    pieces = []
    for record in sorted(records, key=lambda value: value["ticker"]):
        path = ROOT / record["path"]
        baseline.verified_read_bytes(path, record["sha256"])
        frame = pd.read_parquet(path).reset_index()
        frame.insert(0, "ticker", record["ticker"])
        pieces.append(frame)
    panel = pd.concat(pieces, ignore_index=True).sort_values(["date", "ticker"]).reset_index(drop=True)
    if not panel[["date", "ticker"]].equals(keys):
        raise RuntimeError("Candidate rows do not exactly match baseline row order and keys")
    panel["ticker"] = panel.ticker.astype("category")
    target = HERE / "dip_panel.parquet"
    panel.to_parquet(target, compression="zstd", index=False)
    coverage = {
        column: {
            "nonmissing": int(panel[column].notna().sum()),
            "fraction": float(panel[column].notna().mean()),
            "infinite": int(np.isinf(panel[column]).sum()),
        }
        for column in DIP_COLUMNS
    }
    flags = {
        column: {
            "true_rows": int(panel[column].eq(1).sum()),
            "false_rows": int(panel[column].eq(0).sum()),
            "unknown_rows": int(panel[column].isna().sum()),
        }
        for column in DIP_FLAG_COLUMNS
    }
    per_date = panel.groupby("date", observed=True).agg(
        rows=("ticker", "size"),
        valid_low_history=("dip_low_history_valid", "sum"),
        valid_recovery=("dip_recovery_valid", "sum"),
    )
    per_date_path = HERE / "panel_date_coverage.json"
    baseline.write_json(per_date_path, [
        {"date": str(date.date()), **{column: int(value) for column, value in row.items()}}
        for date, row in per_date.iterrows()
    ])
    after = baseline.verify_production_inputs(input_manifest, require_unchanged=False)
    if baseline.sha256(source_path) != source_hash:
        raise RuntimeError("Candidate source changed during preparation")
    baseline.write_json(HERE / "dip_panel_manifest.json", {
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.monotonic() - started, "workers": workers, "blas_threads": 1,
        "path": str(target.relative_to(ROOT)), "sha256": baseline.sha256(target),
        "bytes": target.stat().st_size, "rows": len(panel), "tickers": int(panel.ticker.nunique()),
        "dates": int(panel.date.nunique()), "mrcy_rows": int(panel.ticker.eq("MRCY").sum()),
        "columns": DIP_COLUMNS, "families": DIP_FAMILIES, "coverage": coverage, "flags": flags,
        "baseline_sha256": baseline_manifest["sha256"],
        "context_sha256": price_manifest["context_sha256"],
        "input_manifest_sha256": baseline.sha256(input_path),
        "implementation_artifact_sha256": baseline.sha256(implementation_path),
        "source_sha256": {DIP_SOURCE: source_hash},
        "source_location": str(source_path.relative_to(ROOT)),
        "per_date_coverage": str(per_date_path.relative_to(ROOT)),
        "timing": "Direct join after baseline build_training_frame; no additional shift. Current Open and prior complete bars only.",
        "eligibility": "Exact baseline row keys and order; no candidate-validity, price, volume or event filter.",
        "missing_history": "Canonical SPY dates retained after first observed close; invalid prior bars invalidate windows, producing NaN values and explicit validity indicators.",
        "production_before": before, "production_after": after,
        "feature_files": sorted(records, key=lambda value: value["ticker"]),
    })
    baseline.log("dip_panel_complete", rows=len(panel), tickers=int(panel.ticker.nunique()), bytes=target.stat().st_size)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        raise ValueError("Use one to four workers")
    prepare_panel(args.workers)


if __name__ == "__main__":
    main()
