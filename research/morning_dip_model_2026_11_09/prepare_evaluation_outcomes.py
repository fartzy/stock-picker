"""Create future-session outcome sidecars for evaluation, never model inputs.

Keys exactly match baseline.parquet. Low/high outcomes are unknowable at entry
and must be joined only after predictions are cached. No new rows or filters are
applied to the common model dataset.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json

import prepare_baseline as baseline

import numpy as np
import pandas as pd


def main() -> None:
    here = baseline.HERE
    metadata = json.loads((here / "baseline_manifest.json").read_text())
    source = baseline.ROOT / metadata["path"]
    baseline.verified_read_bytes(source, metadata["sha256"])
    keys = pd.read_parquet(source, columns=["date", "ticker"])
    keys["ticker"] = keys.ticker.astype(str)
    price_manifest = json.loads((here / "price_feature_manifest.json").read_text())
    baseline.verified_read_bytes(baseline.CONTEXT_FILE, price_manifest["context_sha256"])
    context = pd.read_pickle(baseline.CONTEXT_FILE)
    if context["input_manifest_sha256"] != metadata["input_manifest_sha256"]:
        raise RuntimeError("Outcome source differs from baseline source")
    pieces = []
    for ticker, rows in keys.groupby("ticker", sort=True, observed=True):
        bars = context["histories"][ticker].reindex(pd.DatetimeIndex(rows.date))
        open_valid = np.isfinite(bars.Open) & bars.Open.gt(0)
        low_valid = open_valid & np.isfinite(bars.Low) & bars.Low.gt(0) & bars.Low.le(bars.Open)
        high_valid = open_valid & np.isfinite(bars.High) & bars.High.gt(0) & bars.High.ge(bars.Open)
        ohlc = bars[["Open", "High", "Low", "Close"]]
        consistent = (
            np.isfinite(ohlc).all(axis=1) & ohlc.gt(0).all(axis=1)
            & bars.High.ge(bars[["Open", "Close"]].max(axis=1))
            & bars.Low.le(bars[["Open", "Close"]].min(axis=1))
            & bars.High.ge(bars.Low)
        )
        frame = pd.DataFrame({
            "open_to_low_return": (bars.Low / bars.Open - 1).where(low_valid),
            "open_to_high_return": (bars.High / bars.Open - 1).where(high_valid),
            "low_outcome_valid": low_valid,
            "high_outcome_valid": high_valid,
            "source_ohlc_consistent": consistent,
        }, index=bars.index).rename_axis("date").reset_index()
        frame.insert(0, "ticker", ticker)
        pieces.append(frame)
    panel = pd.concat(pieces, ignore_index=True).sort_values(["date", "ticker"]).reset_index(drop=True)
    if not panel[["date", "ticker"]].equals(keys):
        raise RuntimeError("Outcome keys differ from baseline keys")
    if panel.open_to_low_return.gt(0).any() or panel.open_to_high_return.lt(0).any():
        raise RuntimeError("Impossible signed high/low outcome")
    panel["ticker"] = panel.ticker.astype("category")
    target = here / "evaluation_outcomes.parquet"
    panel.to_parquet(target, compression="zstd", index=False)
    baseline.write_json(here / "evaluation_outcomes_manifest.json", {
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "path": str(target.relative_to(baseline.ROOT)),
        "sha256": baseline.sha256(target), "bytes": target.stat().st_size,
        "baseline_sha256": metadata["sha256"],
        "input_manifest_sha256": metadata["input_manifest_sha256"],
        "context_sha256": price_manifest["context_sha256"],
        "rows": len(panel), "mrcy_rows": int(panel.ticker.eq("MRCY").sum()),
        "valid_low_rows": int(panel.low_outcome_valid.sum()),
        "valid_high_rows": int(panel.high_outcome_valid.sum()),
        "source_consistent_ohlc_rows": int(panel.source_ohlc_consistent.sum()),
        "columns": list(panel.columns),
        "use": "Evaluation only: join after predictions are cached; these realized future-session outcomes must never become training inputs.",
        "formulas": {
            "open_to_low_return": "Low/Open - 1 when finite positive Open and Low with Low <= Open; otherwise NaN.",
            "open_to_high_return": "High/Open - 1 when finite positive Open and High with High >= Open; otherwise NaN.",
            "source_ohlc_consistent": "All OHLC finite positive, High >= max(Open,Close), Low <= min(Open,Close), High >= Low; strict comparisons.",
        },
    })
    baseline.log("evaluation_outcomes_complete", rows=len(panel), bytes=target.stat().st_size)


if __name__ == "__main__":
    main()
