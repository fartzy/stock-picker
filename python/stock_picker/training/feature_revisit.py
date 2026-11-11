"""Repeatable, isolated morning-feature review with a frozen later-date check.

Never changes exclusions, live models, source data, or the UI. Rank and raw Fit
are screened separately; raw Fit is explicitly NOT the deployed seven-SVM stack.
Run prepare once, then screen/confirm per model. Reports are resumable and tied
to the same input, protocol and source hashes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from stock_picker.features.pattern_evidence import (
    HALF_LIFE_SESSIONS, PATTERN_EVIDENCE_COLUMNS, PRIOR_OBSERVATIONS,
    STATISTICS, build_pattern_evidence,
)
from stock_picker.storage.feature_exclusion_store import PrunedFeatureStore
from stock_picker.storage.feature_store import FeatureStore
from stock_picker.storage.paths import data_root
from stock_picker.storage.price_store import PriceStore
from stock_picker.storage.universe_store import UniverseStore
from stock_picker.training.dataset import LABEL_COLUMN, build_training_frame
from stock_picker.training.model import LIGHTGBM_DEFAULT_PARAMS, feature_columns, predict, train_lightgbm, train_lightgbm_rank
from stock_picker.training.splits import select_holdout_tickers

BLOCK_SESSIONS = 21
SCREEN_FOLDS = 3
ROUNDS = 100
QUALITY_EXCLUSIONS = frozenset({"dip_recovery_fraction", "dip_volume_change_1d"})


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def date_windows(dates) -> tuple[list[tuple[pd.Timestamp, pd.Timestamp]], tuple[pd.Timestamp, pd.Timestamp]]:
    ordered = sorted(pd.to_datetime(pd.Series(dates)).unique())
    required = (SCREEN_FOLDS + 1) * BLOCK_SESSIONS
    if len(ordered) < required + 100:
        raise ValueError("need at least 100 fitting sessions before four 21-session blocks")
    # Explicit positive positions avoid Python's -0 slice boundary.
    blocks = [ordered[len(ordered) - required + i * BLOCK_SESSIONS:
                      len(ordered) - required + (i + 1) * BLOCK_SESSIONS]
              for i in range(SCREEN_FOLDS + 1)]
    return [(pd.Timestamp(b[0]), pd.Timestamp(b[-1])) for b in blocks[:-1]], (
        pd.Timestamp(blocks[-1][0]), pd.Timestamp(blocks[-1][-1]))


def candidate_variants(baseline: set[str], eligible: set[str], gains: dict[str, float]) -> dict[str, set[str]]:
    """Review every eligible prune's fitting gain; test chosen additions later."""
    ranked = sorted((name for name in eligible if gains.get(name, 0) > 0),
                    key=lambda name: (-gains[name], name))
    variants = {"baseline": baseline, "restore_all_eligible": baseline | eligible}
    patterns = {name for name in eligible if name.endswith("_seasonality")}
    if patterns:
        variants["restore_patterns"] = baseline | patterns
    for count in (5, 10):
        if ranked:
            variants[f"restore_top{count}"] = baseline | set(ranked[:count])
    for slot, name in enumerate(ranked[:5], 1):
        variants[f"restore_single{slot}"] = baseline | {name}
    for name in PATTERN_EVIDENCE_COLUMNS:
        variants[f"add_{name}"] = baseline | {name}
    for statistic in STATISTICS:
        variants[f"add_pattern_{statistic}"] = baseline | {
            name for name in PATTERN_EVIDENCE_COLUMNS if name.endswith(f"_{statistic}")
        }
    variants["add_all_pattern_evidence"] = baseline | set(PATTERN_EVIDENCE_COLUMNS)
    return variants


def daily_results(frame: pd.DataFrame, predictions: np.ndarray) -> pd.DataFrame:
    """Equal-weight daily top-K; deterministic ticker tie breaks; no expense haircut."""
    if len(predictions) != len(frame) or not np.isfinite(predictions).all():
        raise ValueError("nonfinite or misaligned predictions")
    scored = frame[["date", "ticker", LABEL_COLUMN]].copy()
    scored["prediction"] = predictions
    scored = scored.sort_values(["date", "prediction", "ticker"], ascending=[True, False, True])
    records = []
    for date, day in scored.groupby("date", sort=True):
        record = {"date": str(pd.Timestamp(date).date()), "stocks": len(day)}
        for count in (1, 5, 20):
            outcomes = day[LABEL_COLUMN].iloc[:count]
            record[f"top{count}"] = float(outcomes.mean())
            record[f"hit{count}"] = float(outcomes.gt(0).mean())
        record["rank_ic"] = float(day["prediction"].rank().corr(day[LABEL_COLUMN].rank()))
        if not np.isfinite(record["rank_ic"]):
            record["rank_ic"] = None
        record["mae"] = float(np.abs(day["prediction"] - day[LABEL_COLUMN]).mean())
        records.append(record)
    return pd.DataFrame(records)


def choose_variant(results: dict[str, list[dict]]) -> str:
    """Frozen rule: positive mean Rank #1 lift, 2/3 wins, no top-five deterioration."""
    baseline = results["baseline"]
    eligible = []
    for name, folds in results.items():
        if name == "baseline" or len(folds) != SCREEN_FOLDS:
            continue
        one = [f["top1"] - b["top1"] for f, b in zip(folds, baseline)]
        five = [f["top5"] - b["top5"] for f, b in zip(folds, baseline)]
        if np.mean(one) > 0 and sum(d > 0 for d in one) >= 2 and np.mean(five) >= 0:
            eligible.append((float(np.mean(one)), name))
    return sorted(eligible, key=lambda item: (-item[0], item[1]))[0][1] if eligible else "baseline"


def prepare(output: Path) -> None:
    if (output / "protocol.json").exists():
        raise FileExistsError("protocol already frozen; use screen or confirm")
    output.mkdir(parents=True, exist_ok=True)
    tickers = sorted(UniverseStore().active_tickers())
    holdout = select_holdout_tickers(tickers)
    if "MRCY" in holdout:
        raise ValueError("MRCY unexpectedly falls in ticker holdout; resolve cohort before proceeding")
    frames, inputs, missing = [], [], []
    prices, features = PriceStore(), FeatureStore()
    for ticker in sorted(set(tickers) - holdout):
        paths = [data_root() / family / f"{ticker}.parquet" for family in ("prices", "features")]
        if not all(path.exists() for path in paths):
            missing.append(ticker)
            continue
        before = [sha256(path) for path in paths]
        history, stored = prices.read(ticker), features.read(ticker)
        frame = build_training_frame(history, stored)
        evidence = build_pattern_evidence(history).reindex(frame.index)
        frame = frame.join(evidence)
        if before != [sha256(path) for path in paths]:
            raise RuntimeError(f"source changed during read: {ticker}")
        inputs.extend({"path": str(path.relative_to(data_root().parent)), "sha256": digest}
                      for path, digest in zip(paths, before))
        frame = frame.reset_index(names="date")
        frame.insert(0, "ticker", ticker)
        frames.append(frame)
        if len(frames) % 200 == 0:
            print(f"prepared {len(frames)} stocks", flush=True)
    panel = pd.concat(frames, ignore_index=True).sort_values(["date", "ticker"]).reset_index(drop=True)
    if panel.duplicated(["date", "ticker"]).any():
        raise ValueError("duplicate stock-days")
    # Any nonfinite candidate input becomes explicitly unknown for every variant.
    numeric = feature_columns(panel)
    infinite = int(np.isinf(panel[numeric].to_numpy(dtype=float)).sum())
    panel[numeric] = panel[numeric].replace([np.inf, -np.inf], np.nan)
    pruned = PrunedFeatureStore().read()
    raw = set(numeric) - set(PATTERN_EVIDENCE_COLUMNS)
    baseline = raw - pruned
    eligible = raw & pruned - QUALITY_EXCLUSIONS
    screen, confirmation = date_windows(panel["date"])
    panel.to_parquet(output / "panel.parquet", index=False)
    protocol = {
        "created_at": datetime.now(timezone.utc).isoformat(), "rows": len(panel),
        "stocks": int(panel.ticker.nunique()), "sessions": int(panel.date.nunique()),
        "source_inputs": inputs, "panel_sha256": sha256(output / "panel.parquet"),
        "source_sha256": {str(path): sha256(path) for path in (
            Path(__file__).resolve(),
            *(Path(__file__).resolve().parents[1] / relative for relative in (
                "features/pattern_evidence.py", "features/open_pattern_seasonality.py",
                "features/dip_context.py", "features/conditional_seasonality.py",
                "storage/price_store.py", "training/dataset.py", "training/model.py",
                "training/listwise.py", "training/splits.py")))},
        "baseline": sorted(baseline), "eligible_prunes": sorted(eligible),
        "quality_exclusions": sorted(QUALITY_EXCLUSIONS), "prune_records": PrunedFeatureStore().read_all(),
        "holdout_tickers": sorted(holdout), "missing_sources": missing,
        "mrcy_rows": int(panel.ticker.eq("MRCY").sum()), "infinities_marked_missing": infinite,
        "screen": [[str(start.date()), str(end.date())] for start, end in screen],
        "confirmation": [str(date.date()) for date in confirmation],
        "rounds": ROUNDS, "threads": 2, "seed": 0,
        "tree_params": {**LIGHTGBM_DEFAULT_PARAMS, "num_threads": 2},
        "packages": {name: version(name) for name in ("lightgbm", "numpy", "pandas", "exchange-calendars")},
        "pattern_prior_observations": PRIOR_OBSERVATIONS, "pattern_half_life_sessions": HALF_LIFE_SESSIONS,
        "target": "same-session open-to-close", "assumed_expenses": 0,
        "selection": "mean top1 lift >0, positive in >=2/3 folds, mean top5 lift >=0; largest top1 lift",
        "status": "retrospective research; dates were previously studied; no automatic promotion",
        "fit_raw": "LightGBM regression using raw inputs; not production Fit's seven-SVM stack",
    }
    save_json(output / "protocol.json", protocol)
    print(f"frozen {len(panel)} rows; {len(baseline)} baseline inputs; {len(eligible)} eligible prunes", flush=True)


def checked_panel(output: Path) -> tuple[pd.DataFrame, dict]:
    protocol = json.loads((output / "protocol.json").read_text())
    if sha256(output / "panel.parquet") != protocol["panel_sha256"]:
        raise ValueError("panel hash mismatch")
    for path, expected in protocol["source_sha256"].items():
        if sha256(Path(path)) != expected:
            raise ValueError(f"research source changed after protocol freeze: {path}")
    return pd.read_parquet(output / "panel.parquet"), protocol


def run_period(panel: pd.DataFrame, protocol: dict, directory: Path, model_type: str,
               start: str, end: str, selected: str | None = None) -> dict[str, dict]:
    directory.mkdir(parents=True, exist_ok=True)
    train = panel[panel.date < pd.Timestamp(start)]
    test = panel[panel.date.between(pd.Timestamp(start), pd.Timestamp(end))]
    if train.empty or test.empty or train.date.max() >= test.date.min():
        raise ValueError("empty or overlapping fitting/evaluation dates")
    baseline, eligible = set(protocol["baseline"]), set(protocol["eligible_prunes"])
    trainer = train_lightgbm_rank if model_type == "rank" else train_lightgbm
    params = {"num_threads": protocol["threads"], "seed": protocol["seed"]}
    all_model = trainer(train, params=params, num_boost_round=protocol["rounds"],
                        included_features=baseline | eligible)
    gains = dict(zip(all_model.feature_names, map(float, all_model.estimator.feature_importance("gain"))))
    save_json(directory / "training_gain.json", {name: gains.get(name, 0.0) for name in sorted(eligible)})
    variants = candidate_variants(baseline, eligible, gains)
    if selected is not None:
        variants = {name: variants[name] for name in dict.fromkeys(["baseline", selected])}
    reports = {}
    for name, columns in variants.items():
        destination = directory / f"{name}.json"
        if destination.exists():
            report = json.loads(destination.read_text())
            if report["columns"] != sorted(columns):
                raise ValueError("resumed feature selection differs")
            reports[name] = report
            continue
        model = all_model if name == "restore_all_eligible" else trainer(
            train, params=params, num_boost_round=protocol["rounds"], included_features=columns)
        predictions = predict(model, test)
        daily = daily_results(test, predictions)
        scored = test[["date", "ticker", LABEL_COLUMN]].copy()
        scored["prediction"] = predictions
        scored.to_parquet(directory / f"{name}.parquet", index=False)
        daily.to_json(directory / f"{name}.daily.json", orient="records", indent=2)
        report = {"model_type": model_type, "variant": name, "columns": sorted(columns),
                  "train_rows": len(train), "train_through": str(train.date.max().date()),
                  "test_from": start, "test_through": end, "test_rows": len(test),
                  "days": len(daily), **{key: float(daily[key].mean()) for key in
                                        ("top1", "top5", "top20", "hit1", "hit5", "hit20")}}
        if model_type == "fit_raw":
            report["mae"] = float(daily.mae.mean())
        save_json(destination, report)
        reports[name] = report
        print(f"{model_type} {start} {name}: top1={report['top1']:.5f} top5={report['top5']:.5f}", flush=True)
    return reports


def screen(output: Path, model_type: str) -> None:
    panel, protocol = checked_panel(output)
    directory = output / model_type
    selection_path = directory / "selection.json"
    if selection_path.exists():
        raise FileExistsError("selection already frozen; run confirm")
    results: dict[str, list[dict]] = {}
    for number, (start, end) in enumerate(protocol["screen"]):
        reports = run_period(panel, protocol, directory / f"screen_{number}", model_type, start, end)
        for name, report in reports.items():
            results.setdefault(name, []).append(report)
    chosen = choose_variant(results)
    save_json(selection_path, {"created_at": datetime.now(timezone.utc).isoformat(),
                              "protocol_sha256": sha256(output / "protocol.json"),
                              "chosen": chosen, "folds": results})
    print(f"{model_type}: frozen selection {chosen}", flush=True)


def confirm(output: Path, model_type: str) -> None:
    selection = json.loads((output / model_type / "selection.json").read_text())
    if selection["protocol_sha256"] != sha256(output / "protocol.json"):
        raise ValueError("protocol changed after selection")
    panel, protocol = checked_panel(output)
    results = run_period(panel, protocol, output / model_type / "confirmation", model_type,
                         *protocol["confirmation"], selected=selection["chosen"])
    save_json(output / model_type / "confirmation.json", results)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("prepare", "screen", "confirm"), required=True)
    parser.add_argument("--model", choices=("rank", "fit_raw"), default="rank")
    args = parser.parse_args()
    output = args.output.resolve()
    if args.phase == "prepare":
        prepare(output)
    elif args.phase == "screen":
        screen(output, args.model)
    else:
        confirm(output, args.model)


if __name__ == "__main__":
    main()
