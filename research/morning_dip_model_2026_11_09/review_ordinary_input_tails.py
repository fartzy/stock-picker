"""Read-only representation check on refreshed ordinary prices, without outcomes."""
from __future__ import annotations

from datetime import datetime, timezone
import argparse
import json
from pathlib import Path
import sys

from review_input_tails import ROOT, HERE, RATIOS, describe, digest, np, pd

sys.path.insert(0, str(ROOT / "python"))
from stock_picker.features.dip_context import DIP_COLUMNS, complete_dip_history, morning_dip_features
from stock_picker.storage.price_store import sanitize_price_history
from stock_picker.training.dataset import LABEL_COLUMN, build_training_frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--persisted", action="store_true", help="Inspect exact persisted ordinary feature rows after normal dataset alignment")
    args = parser.parse_args()
    inputs = json.loads((HERE / "input_manifest.json").read_text())
    registry = pd.read_parquet(ROOT / "data/universe/registry.parquet")
    active = set(registry.index[registry.active])
    if active != set(inputs["registry_tickers"]):
        raise ValueError("Registry changed: review training membership before proceeding")
    tickers = sorted(active - set(inputs["holdout_tickers"]))
    cutoff = pd.Timestamp("2026-11-09")
    lower = cutoff - pd.DateOffset(years=1)
    pieces, sources = [], {}
    for n, ticker in enumerate(tickers, 1):
        path = ROOT / "data/prices" / f"{ticker}.parquet"
        before = digest(path)
        raw = sanitize_price_history(pd.read_parquet(path).sort_index())
        if digest(path) != before:
            raise ValueError(f"Price changed during read: {ticker}")
        raw = raw.loc[raw.index <= cutoff]
        feature_hash = None
        if args.persisted:
            feature_path = ROOT / "data/features" / f"{ticker}.parquet"
            feature_hash = digest(feature_path)
            features = pd.read_parquet(feature_path, columns=list(DIP_COLUMNS))
            if digest(feature_path) != feature_hash:
                raise ValueError(f"Features changed during read: {ticker}")
            # Use the ordinary dataset alignment and finite-label eligibility;
            # immediately discard labels without examining return relationships.
            values = build_training_frame(raw, features).drop(columns=[LABEL_COLUMN])
        else:
            retained = raw.index[(raw.index > lower) & (raw.index <= cutoff)][1:]
            values = morning_dip_features(complete_dip_history(raw)).reindex(retained).loc[:, list(RATIOS)]
            eligible = raw.reindex(retained)
            values = values.loc[np.isfinite(eligible.Open) & eligible.Open.gt(0) & np.isfinite(eligible.Close)]
        values = values.reset_index(names="date")
        values.insert(0, "ticker", ticker)
        pieces.append(values)
        sources[ticker] = {"sha256": before, "price_rows": len(raw), "first": str(raw.index.min().date()), "last": str(raw.index.max().date()), "eligible_input_rows": len(values)}
        if feature_hash:
            sources[ticker]["feature_sha256"] = feature_hash
        if n % 400 == 0:
            print(f"checked {n}/{len(tickers)} price files", flush=True)
    frame = pd.concat(pieces, ignore_index=True)
    extremes = {}
    columns = list(DIP_COLUMNS) if args.persisted else list(RATIOS)
    for name in columns:
        extremes[name] = []
        for row in frame.loc[frame[name].abs().nlargest(5).index, ["ticker", "date", name]].itertuples(index=False, name=None):
            ticker, when, value = row
            raw = pd.read_parquet(ROOT / "data/prices" / f"{ticker}.parquet").sort_index()
            i = raw.index.get_loc(when)
            c1, opening = float(raw.Close.iloc[i-1]), float(raw.Open.iloc[i])
            c3 = float(raw.Close.iloc[i-3]) if i >= 3 else None
            extremes[name].append({"ticker": ticker, "date": str(when.date()), "value": float(value), "open_t": opening, "close_t_minus_1": c1, "close_t_minus_3": c3, "closing_loss_dollars": c3-c1 if c3 is not None else None,
                                   "volume_t_minus_1": float(raw.Volume.iloc[i-1]), "volume_t_minus_2": float(raw.Volume.iloc[i-2]) if i >= 2 else None})
    result = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Input-only review of the upcoming ordinary training population from refreshed Yahoo prices; normal ticker holdout and one-year feature retention. No target/outcome analysis or model fitting. No live data writes.",
        "source_basis": "Yahoo refreshed prices, distinct from the frozen Massive research panel",
        "persisted_feature_alignment": args.persisted,
        "rows": len(frame), "tickers": len(tickers), "from": str(frame.date.min().date()), "through": str(frame.date.max().date()),
        "ratios": {name: describe(frame[name]) for name in columns},
        "largest_absolute_ratio_rows": extremes,
        "frozen_research_spcx_example_present": bool(((frame.ticker == "SPCX") & (frame.date == pd.Timestamp("2026-02-09"))).any()),
        "sources": sources,
    }
    destination = HERE / ("ordinary_persisted_input_representation_review.json" if args.persisted else "ordinary_input_representation_review.json")
    destination.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"artifact": str(destination), "rows": len(frame), "tickers": len(tickers), "recovery": result["ratios"]["dip_recovery_fraction"]}, indent=2))


if __name__ == "__main__":
    main()
