"""Frozen four-way context experiment. Never writes the active model store."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import date, datetime, timedelta
import gzip
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from research.intraday_context_2026_11_11.collect import atomic_json, read_archive
from stock_picker.features.intraday_context import (
    CONTEXT_DESCRIPTIONS, MARKET_DESCRIPTIONS, VOLUME_DESCRIPTIONS, build_intraday_context,
)
from stock_picker.features.intraday_entry import FEATURE_COLUMNS
from stock_picker.ingestion.session import cash_session_window
from stock_picker.training.intraday_entry_model import train_intraday_entry

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent.parent
ET = ZoneInfo("America/New_York")


def load_ticker(ticker: str, manifest: list[dict]) -> dict[str, list[dict]]:
    days = defaultdict(list)
    seen = set()
    for source in manifest:
        if source["ticker"] != ticker or "error_type" in source:
            continue
        document = read_archive(ROOT / source["pointer"])
        if document["ticker"] != ticker:
            raise ValueError("source ticker mismatch")
        for bar in document["bars"]:
            if bar["t"] in seen:
                raise ValueError("overlapping range identity")
            seen.add(bar["t"])
            day = datetime.fromtimestamp(bar["t"]/1000, ET).date().isoformat()
            days[day].append(bar)
    return dict(days)


def build_frame() -> pd.DataFrame:
    base = pd.read_parquet(ROOT / "base.parquet")
    collection = json.loads((ROOT / "collection_manifest.json").read_text())
    specs = json.loads((ROOT / "collection_plan.json").read_text())["ranges"]
    if len(collection) != len(specs):
        raise ValueError("collection still in progress")
    spy = load_ticker("SPY", collection)
    original = json.loads((REPO / "research/rank_dip_timing_2026_11_09/minute_manifest.json").read_text())
    original = {(row["ticker"], row["session"]): row for row in original}
    sessions = {}
    day = date.fromisoformat(base.session.min()) - timedelta(days=45)
    while day <= date.fromisoformat(base.session.max()):
        window = cash_session_window(datetime.combine(day, datetime.min.time(), tzinfo=ET) + timedelta(hours=12))
        if window:
            sessions[day.isoformat()] = window
        day += timedelta(days=1)
    ordered_sessions = sorted(sessions)
    context = []
    for count, (ticker, group) in enumerate(base.groupby("ticker"), 1):
        previous_bars = load_ticker(ticker, collection)
        for session, rows in group.groupby("session"):
            window = sessions[session]
            previous_days = [day for day in ordered_sessions if day < session][-20:]
            history = [(sessions[day].open_at, sessions[day].close_at, previous_bars.get(day, []))
                       for day in previous_days]
            source = original[ticker, session]
            raw = (REPO / source["path"]).read_bytes()
            if hashlib.sha256(raw).hexdigest() != source["sha256"]:
                raise ValueError("frozen current-stock source hash mismatch")
            current = json.loads(gzip.decompress(raw))["bars"]
            for row in rows.itertuples():
                values, coverage = build_intraday_context(
                    stock_bars=current, spy_bars=spy.get(session, []), history=history,
                    observed_at=datetime.fromisoformat(row.observed_at), session_open_at=window.open_at,
                    stock_return_from_open=row.return_from_open, stock_return_15m=row.return_15m,
                )
                context.append({"row_id": row.Index, **values, **coverage})
        if count % 40 == 0:
            print(f"Built context for {count} / {base.ticker.nunique()} tickers", flush=True)
    additions = pd.DataFrame(context).set_index("row_id").reindex(base.index)
    frame = pd.concat([base, additions], axis=1)
    assert len(frame) == len(base) and not additions.index.duplicated().any()
    frame.to_parquet(ROOT / "context.parquet", index=False)
    atomic_json(ROOT / "coverage.json", {
        "rows": len(frame), "sessions": int(frame.session.nunique()),
        "stock_days": len(frame[["ticker", "session"]].drop_duplicates()),
        "missing": {name: int(frame[name].isna().sum()) for name in CONTEXT_DESCRIPTIONS},
        "failed_ranges": [row for row in collection if "error_type" in row],
        "MRCY_rows": int((frame.ticker == "MRCY").sum()),
        "MRCY_explanation": "No ticker exclusion. MRCY is in the separate live cohort on Nov 4, not this frozen baseline model's cached top-20 cohort. Never silently pool different rank systems.",
        "protocol_sha256": hashlib.sha256((ROOT / "protocol.json").read_bytes()).hexdigest(),
        "collection_manifest_sha256": hashlib.sha256((ROOT / "collection_manifest.json").read_bytes()).hexdigest(),
    })
    return frame


def paired_difference(test: pd.DataFrame, prediction: dict, baseline: dict, metric: str) -> dict:
    """Five-date moving-block bootstrap; preserve all repeated observations per date."""
    if test.empty:
        return {"sessions": 0}
    target = test.target_return.to_numpy(float)
    if metric == "brier":
        truth = (target > 0).astype(float)
        difference = (prediction["probability_profit"]-truth)**2 - (baseline["probability_profit"]-truth)**2
    else:
        difference = abs(prediction["expected_return"]-target) - abs(baseline["expected_return"]-target)
    rows = test[["session", "ticker"]].copy()
    rows["difference"] = difference
    daily = rows.groupby(["session", "ticker"])["difference"].mean().groupby("session").mean().to_numpy()
    rng = np.random.default_rng(1729)
    size, block = len(daily), min(5, len(daily))
    draws = []
    for _ in range(2000):
        starts = rng.integers(0, size-block+1, size=(size+block-1)//block)
        indices = np.concatenate([np.arange(start, start+block) for start in starts])[:size]
        draws.append(float(daily[indices].mean()))
    return {"sessions": size, "difference": float(daily.mean()),
            "ci95": [float(x) for x in np.quantile(draws, [.025, .975])],
            "negative_is_better": True}


def screen_individual_features(frame: pd.DataFrame) -> tuple[str, ...]:
    """Select at most two candidates without accessing outer calibration/test labels."""
    sessions = sorted(frame.session.unique())
    fit_sessions = sessions[:int(len(sessions)*.6)]
    development = frame[frame.session.isin(fit_sessions)]
    trials = {"existing": (), **{name: (name,) for name in CONTEXT_DESCRIPTIONS}}
    scored = {name: [] for name in trials}
    already_evaluated = set()
    splits = []
    for fraction in (.65, .82, 1.0):
        selected_days = fit_sessions[:int(len(fit_sessions)*fraction)]
        train = development[development.session.isin(selected_days)]
        evaluation_days = set(selected_days[int(len(selected_days)*.8):]) - already_evaluated
        test = train[train.session.isin(evaluation_days) & (train["rank"] == 1)]
        already_evaluated.update(evaluation_days)
        splits.append({"data_through": selected_days[-1], "evaluation_sessions": sorted(evaluation_days)})
        for name, additions in trials.items():
            model = train_intraday_entry(train, feature_columns=(*FEATURE_COLUMNS, *additions),
                                         feature_descriptions=CONTEXT_DESCRIPTIONS)
            probability = model.predict(test)["probability_profit"]
            rows = test[["session", "ticker", "observed_at"]].copy()
            rows["error"] = (probability - (test.target_return.to_numpy() > 0)) ** 2
            scored[name].append(rows)
    scores = {name: pd.concat(parts).groupby("session").error.mean() for name, parts in scored.items()}
    differences = {name: float((scores[name] - scores["existing"]).mean())
                   for name in CONTEXT_DESCRIPTIONS}
    selected = tuple(sorted((name for name, value in differences.items() if value < 0),
                            key=differences.get)[:2])
    atomic_json(ROOT / "individual_feature_screen.json", {
        "outer_fit_through": fit_sessions[-1], "inner_splits": splits,
        "inner_rank1_evaluation_sessions": len(scores["existing"]),
        "existing_brier": float(scores["existing"].mean()),
        "individual_brier_difference": differences, "selected": selected,
        "selection_rule": "At most two individually better Rank1 Brier results; exploratory, not a significance test",
    })
    print("Training-only screen selected:", selected, flush=True)
    return selected


def evaluate(frame: pd.DataFrame, *, screen: bool = False) -> None:
    original_metadata = json.loads((ROOT / "base_metadata.json").read_text())
    splits = sorted(frame.session.unique())
    test_start = splits[int(len(splits)*.8)]
    test = frame[frame.session >= test_start].sort_values(["session", "ticker", "observed_at"])
    variants = {"existing": (), "market": tuple(MARKET_DESCRIPTIONS),
                "volume": tuple(VOLUME_DESCRIPTIONS), "market_and_volume": tuple(CONTEXT_DESCRIPTIONS)}
    if screen:
        variants["training_selected"] = screen_individual_features(frame)
    predictions, results = {}, {}
    for name, additions in variants.items():
        model = train_intraday_entry(frame, {**original_metadata, "context_experiment": name,
            "evaluation_status": "Retrospective reused evaluation; not fresh confirmation"},
            feature_columns=(*FEATURE_COLUMNS, *additions), feature_descriptions=CONTEXT_DESCRIPTIONS)
        model.metadata["notes"] = [note.replace("untouched evaluation", "evaluation already inspected in earlier research")
                                   for note in model.metadata["notes"]]
        predictions[name] = model.predict(test)
        results[name] = model.metadata
        atomic_json(ROOT / f"{name}_metadata.json", model.metadata)
        print(name, json.dumps({k: model.metadata["validation"]["rank1"][k]
                               for k in ("brier", "baseline_brier", "return_mae", "sessions")}), flush=True)
    comparison = {}
    for name in variants:
        comparison[name] = {}
        for subset, mask in {"all": np.ones(len(test), dtype=bool),
                             "rank1": (test["rank"] == 1).to_numpy(),
                             "rank1_dips": ((test["rank"] == 1) & (test.return_from_open < 0)).to_numpy(),
                             "rank1_full_context": ((test["rank"] == 1) & test[list(CONTEXT_DESCRIPTIONS)].notna().all(axis=1)).to_numpy()}.items():
            actual, reference = ({k: v[mask] for k, v in item.items()}
                                  for item in (predictions[name], predictions["existing"]))
            comparison[name][subset] = {metric: paired_difference(test.loc[mask], actual, reference, metric)
                                       for metric in ("brier", "return_mae")}
    for name, prediction in predictions.items():
        for key, values in prediction.items():
            test[name + "_" + key] = values
    test.to_parquet(ROOT / "evaluation_predictions.parquet", index=False)
    atomic_json(ROOT / "comparisons.json", comparison)
    atomic_json(ROOT / "run_manifest.json", {
        "protocol_sha256": hashlib.sha256((ROOT / "protocol.json").read_bytes()).hexdigest(),
        "context_sha256": hashlib.sha256((ROOT / "context.parquet").read_bytes()).hexdigest(),
        "source_hashes": {str(path.relative_to(REPO)): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in (Path(__file__).resolve(), ROOT / "collect.py",
                            REPO / "python/stock_picker/features/intraday_context.py",
                            REPO / "python/stock_picker/training/intraday_entry_model.py")},
        "expense_fraction": 0.0, "active_model_changed": False,
        "models": list(results), "test_start": test_start, "test_end": test.session.max(),
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reuse-frame", action="store_true")
    parser.add_argument("--screen", action="store_true", help="Add the prespecified training-only individual feature screen")
    args = parser.parse_args()
    frame = pd.read_parquet(ROOT / "context.parquet") if args.reuse_frame else build_frame()
    evaluate(frame, screen=args.screen)


if __name__ == "__main__":
    main()
