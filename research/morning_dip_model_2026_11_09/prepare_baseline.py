"""Prepare a source-consistent historical baseline for the dip-feature experiment.

This reproduces current raw feature definitions on one immutable five-year
price cache. It never fits a prediction model, evaluates ticker holdout labels,
or writes production stores. MRCY receives the same eligibility as other names.

Run from the repository root (four processes, one BLAS thread each):
    bazel-bin/python/stock_picker/api/main.runfiles/rules_python++python+python_3_12_aarch64-apple-darwin/bin/python3 research/morning_dip_model_2026_11_09/prepare_baseline.py --phase prices

The subsequent assembly phase adds cross-sectional structure, corrects the
historical opening cluster gap to use prior membership, and applies the existing
training-row shift before writing the isolated panel.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

# Set before importing numerical libraries; spawned workers inherit these limits.
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[variable] = "1"

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RUNFILES = (ROOT / "bazel-bin/python/stock_picker/api/main.runfiles").resolve()
sys.path[:0] = [str(ROOT / "python"), *map(str, RUNFILES.glob("rules_python++pip+*/site-packages"))]

# Replays must not pick up the ordinary-pipeline integration being developed
# after this experiment was frozen. Keep unchanged support modules available
# from the repository, while resolving copied numerical modules first.
BASELINE_SOURCE_ROOT = HERE / "context_inputs/baseline_source"
if BASELINE_SOURCE_ROOT.exists():
    import stock_picker.features
    import stock_picker.storage
    import stock_picker.training

    for package in (stock_picker.features, stock_picker.storage, stock_picker.training):
        frozen = BASELINE_SOURCE_ROOT / "python" / package.__name__.replace(".", "/")
        package.__path__ = [str(frozen), *package.__path__]

import numpy as np
import pandas as pd

from stock_picker.features.breadth import build_breadth_features
from stock_picker.features.conditional_seasonality import pooled_setup_seasonality, ticker_setup_bucket
from stock_picker.features.cross_sectional import RETURN_RANK_WINDOWS, return_rank
from stock_picker.features.news import NEWS_FEATURE_COLUMNS
from stock_picker.features.pipeline import build_features
from stock_picker.features.structure import build_structure_features
from stock_picker.features.trajectory import build_trajectory_features
from stock_picker.features.weather import build_weather_features
from stock_picker.storage.price_store import sanitize_price_history
from stock_picker.training.dataset import LABEL_COLUMN, build_training_frame

CACHE_DIR = HERE / "cache"
FEATURE_CACHE = CACHE_DIR / "features"
CONTEXT_DIR = HERE / "context_inputs"
CONTEXT_FILE = CACHE_DIR / "context.pkl"
GLOBAL_WARMUP_SESSIONS = 252
_WORKER_CONTEXT: dict = {}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def log(event: str, **fields) -> None:
    print(json.dumps({"time": datetime.now(timezone.utc).isoformat(), "event": event, **fields}), flush=True)


def verified_read_bytes(path: Path, expected: str) -> bytes:
    value = path.read_bytes()
    if hashlib.sha256(value).hexdigest() != expected:
        raise RuntimeError(f"Frozen input changed: {path}")
    return value


def verify_production_inputs(manifest: dict, require_unchanged: bool = True) -> dict:
    """Audit live references; only the initial freeze requires their equality.

    A scheduled refresh may legitimately change live files after context and
    features have been frozen. Subsequent phases verify actual frozen copies
    and retain every live-reference divergence instead of refreshing the pin.
    The live Jev database is deliberately not an input.
    """
    changed = []
    checked = 0
    for relative, record in manifest["files"].items():
        # The source news DB is only a provenance reference. Frozen feature
        # columns below, not new database ingestion, feed this experiment.
        if relative == "data/news/news.db":
            continue
        path = ROOT / relative
        observed = sha256(path)
        if observed != record["sha256"]:
            changed.append({
                "path": relative, "frozen_sha256": record["sha256"],
                "observed_live_sha256": observed,
                "observed_live_mtime_utc": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
            })
        checked += 1
    if changed and require_unchanged:
        raise RuntimeError(f"Production input changed during preparation: {changed[:20]}")
    return {"checked_files": checked, "changed_files": changed,
            "policy": "Initial freeze equality required" if require_unchanged else "Live references audited only; verified immutable copies are the inputs"}


def verify_frozen_context(manifest: dict, price_manifest: dict) -> dict:
    """Check all shared experiment copies without consulting refreshed stores."""
    verified_read_bytes(CONTEXT_FILE, price_manifest["context_sha256"])
    source = manifest["five_year_cache"]
    verified_read_bytes(ROOT / source["path"], source["sha256"])
    frozen_context = json.loads((CONTEXT_DIR / "manifest.json").read_text())
    for record in frozen_context["files"].values():
        verified_read_bytes(ROOT / record["path"], record["sha256"])
    frozen_source = json.loads((HERE / "baseline_source_manifest.json").read_text())
    for record in frozen_source["files"].values():
        verified_read_bytes(ROOT / record["copy"], record["sha256"])
    return {"context_sha256": price_manifest["context_sha256"],
            "source_cache_sha256": source["sha256"],
            "source_copy_files": len(frozen_source["files"]),
            "baseline_source_manifest_sha256": sha256(HERE / "baseline_source_manifest.json")}


def freeze_context(manifest: dict) -> dict:
    """Freeze tiny shared sources and precomputed news before lengthy work."""
    CONTEXT_DIR.mkdir(exist_ok=True)
    paths = {"registry": "data/universe/registry.parquet", "weather": "data/weather/nyc.parquet"}
    context_paths = {}
    for name, relative in paths.items():
        target = CONTEXT_DIR / f"{name}.parquet"
        value = verified_read_bytes(ROOT / relative, manifest["files"][relative]["sha256"])
        if target.exists() and target.read_bytes() != value:
            raise RuntimeError(f"Frozen context differs: {target}")
        if not target.exists():
            target.write_bytes(value)
        context_paths[name] = target
    news_path = CONTEXT_DIR / "news_features.parquet"
    if not news_path.exists():
        news = []
        for ticker in manifest["registry_tickers"]:
            relative = f"data/features/{ticker}.parquet"
            if relative not in manifest["files"]:
                continue
            source = ROOT / relative
            verified_read_bytes(source, manifest["files"][relative]["sha256"])
            frame = pd.read_parquet(source, columns=NEWS_FEATURE_COLUMNS)
            frame = frame.rename_axis("date").reset_index()
            frame.insert(0, "ticker", ticker)
            news.append(frame)
        pd.concat(news, ignore_index=True).to_parquet(news_path, compression="zstd", index=False)
    context_paths["news"] = news_path
    metadata = {
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "files": {name: {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)} for name, path in context_paths.items()},
        "news_policy": "Use frozen completed-session feature values on their recorded dates; zero before the news archive begins, matching existing empty-store semantics. No production NewsStore constructor or database writes.",
    }
    metadata_path = CONTEXT_DIR / "manifest.json"
    if metadata_path.exists():
        previous = json.loads(metadata_path.read_text())
        if previous["files"] != metadata["files"]:
            raise RuntimeError("Frozen context hashes changed")
    else:
        write_json(metadata_path, metadata)
    return context_paths


def build_histories(manifest: dict) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, pd.DatetimeIndex]:
    source = ROOT / manifest["five_year_cache"]["path"]
    verified_read_bytes(source, manifest["five_year_cache"]["sha256"])
    arrays = pd.read_pickle(source)
    dates = pd.DatetimeIndex(arrays["SPY"]["dates"])
    columns = ["Open", "High", "Low", "Close", "Volume"]
    histories = {}
    for ticker in sorted(set(manifest["registry_tickers"]) & set(arrays)):
        item = arrays[ticker]
        if list(item["dates"]) != list(arrays["SPY"]["dates"]):
            raise RuntimeError(f"Calendar mismatch: {ticker}")
        raw = pd.DataFrame(item["bars"], index=dates, columns=columns)
        raw = sanitize_price_history(raw)
        # Leading pre-listing absence is not history. Internal missing dates
        # remain present: no invented or forward-filled raw OHLCV bars.
        first = raw["Close"].first_valid_index()
        if first is not None:
            histories[ticker] = raw.loc[first:].copy()
    spy = pd.DataFrame(arrays["SPY"]["bars"], index=dates, columns=columns)
    return histories, spy, dates


def make_context(manifest: dict) -> dict:
    paths = freeze_context(manifest)
    histories, spy, dates = build_histories(manifest)
    registry = pd.read_parquet(paths["registry"])
    active = registry.loc[registry.active]
    sectors = active.sector.dropna().astype(str).to_dict()
    first_seen = active.first_seen.astype(str).to_dict()
    news = pd.read_parquet(paths["news"])
    news_by_ticker = {str(ticker): frame.drop(columns="ticker").set_index("date") for ticker, frame in news.groupby("ticker", sort=False)}
    daily_returns = {ticker: history.Close.pct_change() for ticker, history in histories.items()}
    buckets = {ticker: ticker_setup_bucket(history) for ticker, history in histories.items()}
    pooled = pooled_setup_seasonality(daily_returns, buckets)
    rank_by_window = {
        window: return_rank(pd.DataFrame({ticker: history.Close.pct_change(window) for ticker, history in histories.items()}))
        for window in RETURN_RANK_WINDOWS
    }
    daily_frame = pd.DataFrame(daily_returns)
    sector_return = {
        sector: daily_frame[[ticker for ticker in histories if sectors.get(ticker) == sector]].mean(axis=1)
        for sector in set(sectors.values())
    }
    breadth = build_breadth_features(histories, first_seen)
    weather = build_weather_features(dates, pd.read_parquet(paths["weather"]))
    return {
        "histories": histories, "spy": spy, "dates": dates, "news": news_by_ticker,
        "pooled": pooled, "rank_by_window": rank_by_window, "sector_return": sector_return,
        "sectors": sectors, "first_seen": first_seen, "breadth": breadth, "weather": weather,
        "input_manifest_sha256": sha256(HERE / "input_manifest.json"),
    }


def initialize_worker(context_path: str) -> None:
    global _WORKER_CONTEXT
    _WORKER_CONTEXT = pd.read_pickle(context_path)


def build_one(ticker: str) -> dict:
    context = _WORKER_CONTEXT
    history = context["histories"][ticker]
    path = FEATURE_CACHE / f"{ticker}.parquet"
    if not path.exists():
        peer_ranks = {window: ranks[ticker] for window, ranks in context["rank_by_window"].items()}
        sector_return = context["sector_return"].get(context["sectors"].get(ticker))
        frame = build_features(
            history, benchmark_history=context["spy"], peer_return_ranks=peer_ranks,
            sector_avg_return=sector_return, pooled_seasonality=context["pooled"][ticker],
            spy_history=context["spy"], market_breadth=context["breadth"],
        )
        frame = pd.concat([frame, context["weather"].reindex(frame.index)], axis=1)
        news = context["news"].get(ticker)
        if news is not None:
            frame.loc[:, NEWS_FEATURE_COLUMNS] = news.reindex(frame.index).fillna(0).to_numpy()
        frame.to_parquet(path, compression="zstd")
    return {"ticker": ticker, "path": str(path.relative_to(ROOT)), "sha256": sha256(path), "bytes": path.stat().st_size}


def prepare_prices(workers: int) -> None:
    started = time.monotonic()
    manifest = json.loads((HERE / "input_manifest.json").read_text())
    before = verify_production_inputs(manifest)
    CACHE_DIR.mkdir(exist_ok=True)
    FEATURE_CACHE.mkdir(exist_ok=True)
    if CONTEXT_FILE.exists():
        context = pd.read_pickle(CONTEXT_FILE)
        if context["input_manifest_sha256"] != sha256(HERE / "input_manifest.json"):
            raise RuntimeError("Cached context has a different input manifest")
    else:
        log("freezing_context")
        context = make_context(manifest)
        pd.to_pickle(context, CONTEXT_FILE)
    names = sorted(context["histories"])
    if "MRCY" not in names:
        raise RuntimeError("MRCY missing from immutable source context")
    log("price_features_started", tickers=len(names), workers=workers)
    records = []
    with ProcessPoolExecutor(max_workers=workers, initializer=initialize_worker, initargs=(str(CONTEXT_FILE),)) as executor:
        futures = {executor.submit(build_one, ticker): ticker for ticker in names}
        for index, future in enumerate(as_completed(futures), 1):
            records.append(future.result())
            if index % 100 == 0 or index == len(names):
                log("price_features_progress", completed=index, total=len(names), elapsed_seconds=round(time.monotonic() - started, 1))
    after = verify_production_inputs(manifest)
    output = {
        "phase": "price_features", "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.monotonic() - started, "workers": workers, "blas_threads": 1,
        "input_manifest_sha256": sha256(HERE / "input_manifest.json"),
        "source_cache_sha256": manifest["five_year_cache"]["sha256"],
        "context_sha256": sha256(CONTEXT_FILE), "ticker_count": len(names),
        "feature_files": sorted(records, key=lambda record: record["ticker"]),
        "production_before": before, "production_after": after,
        "scope": "Price feature calculation only; no prediction-model fit, no production writes, no held-out labels evaluated.",
    }
    write_json(HERE / "price_feature_manifest.json", output)
    log("price_features_complete", tickers=len(names), elapsed_seconds=round(output["elapsed_seconds"], 1))


def baseline_columns(manifest: dict) -> list[str]:
    """Use the frozen saved Rank model's exact raw input order."""
    audit = json.loads((ROOT / "research/morning_dip_patterns_2026_11_09/feature_audit.json").read_text())
    reference = audit["model_snapshot"]["artifacts"]["day_session_return_rank"]
    if reference["sha256"] != manifest["files"][reference["path"]]["sha256"]:
        raise RuntimeError("Saved model reference no longer matches the frozen manifest")
    columns = reference["members"][0]["feature_names"]
    if len(columns) != 132 or len(set(columns)) != 132:
        raise RuntimeError("Unexpected baseline raw-feature names")
    return columns


def assemble_baseline() -> None:
    """Build common rows; leave missing candidate history to its validity flag."""
    started = time.monotonic()
    manifest = json.loads((HERE / "input_manifest.json").read_text())
    price_manifest = json.loads((HERE / "price_feature_manifest.json").read_text())
    if price_manifest["input_manifest_sha256"] != sha256(HERE / "input_manifest.json"):
        raise RuntimeError("Price features use a different input manifest")
    before = verify_production_inputs(manifest, require_unchanged=False)
    frozen_check = verify_frozen_context(manifest, price_manifest)
    context = pd.read_pickle(CONTEXT_FILE)
    histories = context["histories"]
    columns = baseline_columns(manifest)
    features = {}
    for record in price_manifest["feature_files"]:
        path = ROOT / record["path"]
        verified_read_bytes(path, record["sha256"])
        features[record["ticker"]] = pd.read_parquet(path)
    source_paths = [
        "python/stock_picker/features/structure.py",
        "python/stock_picker/features/trajectory.py",
        "python/stock_picker/training/dataset.py",
    ]
    source_files = {
        path: BASELINE_SOURCE_ROOT / path if (BASELINE_SOURCE_ROOT / path).exists() else ROOT / path
        for path in source_paths
    }
    source_hashes = {path: sha256(source) for path, source in source_files.items()}
    structure_path = CACHE_DIR / "structure.pkl"
    structure_meta_path = CACHE_DIR / "structure_manifest.json"
    if structure_path.exists() and structure_meta_path.exists():
        structure_meta = json.loads(structure_meta_path.read_text())
        if structure_meta["source_sha256"] != source_hashes[source_paths[0]]:
            raise RuntimeError("Cached structure was built with different timing code")
        verified_read_bytes(structure_path, structure_meta["artifact_sha256"])
        structure = pd.read_pickle(structure_path)
    else:
        log("structure_started", tickers=len(histories), dates=len(context["dates"]))
        structure = build_structure_features(histories, features)
        pd.to_pickle(structure, structure_path)
        write_json(structure_meta_path, {
            "source_sha256": source_hashes[source_paths[0]],
            "artifact_sha256": sha256(structure_path),
            "timing": "Current opening peer gaps use previous completed-row cluster membership.",
        })
    log("structure_complete", elapsed_seconds=round(time.monotonic() - started, 1))
    trajectory_path = CACHE_DIR / "trajectory.pkl"
    if trajectory_path.exists():
        trajectory = pd.read_pickle(trajectory_path)
    else:
        log("trajectory_started")
        trajectory = build_trajectory_features(histories, context["first_seen"])
        pd.to_pickle(trajectory, trajectory_path)
    held = set(manifest["holdout_tickers"])
    names = sorted(set(histories) - held - {"SPY"})
    if len(names) != 1784 or "MRCY" not in names:
        raise RuntimeError("Unexpected training universe or MRCY exclusion")
    first_date = context["dates"][GLOBAL_WARMUP_SESSIONS]
    pieces, ticker_counts, absent_ticker_columns = [], [], []
    for ticker in names:
        combined = pd.concat([features[ticker], structure[ticker], trajectory[ticker]], axis=1)
        missing = set(columns) - set(combined.columns)
        # Ordinary pooled training takes the union across ticker columns. A
        # missing registry sector legitimately omits this one feature; retain
        # its unknown values rather than dropping that ticker or inventing0s.
        if missing - {"sector_relative_return"}:
            raise RuntimeError(f"Baseline columns missing for {ticker}: {sorted(missing)}")
        if missing:
            absent_ticker_columns.append({"ticker": ticker, "columns": sorted(missing)})
        frame = build_training_frame(histories[ticker], combined.reindex(columns=columns))
        frame = frame.loc[frame.index >= first_date].copy()
        price = histories[ticker].reindex(frame.index)
        valid = np.isfinite(price[["Open", "Close"]]).all(axis=1) & price.Open.gt(0) & price.Close.gt(0)
        frame = frame.loc[valid]
        frame = frame.rename_axis("date").reset_index()
        frame.insert(0, "ticker", ticker)
        pieces.append(frame)
        ticker_counts.append({"ticker": ticker, "rows": len(frame)})
    pooled = pd.concat(pieces, ignore_index=True).sort_values(["date", "ticker"]).reset_index(drop=True)
    if pooled.ticker.isin(held).any() or not np.isfinite(pooled[LABEL_COLUMN]).all():
        raise RuntimeError("Held-out ticker or invalid target in model panel")
    if pooled.duplicated(["date", "ticker"]).any():
        raise RuntimeError("Duplicate ticker-date key")
    pooled["ticker"] = pooled.ticker.astype("category")
    target = HERE / "baseline.parquet"
    pooled.to_parquet(target, compression="zstd", index=False)
    coverage = {
        column: {"nonmissing": int(pooled[column].notna().sum()), "fraction": float(pooled[column].notna().mean()),
                 "infinite": int(np.isinf(pooled[column]).sum())}
        for column in columns
    }
    after = verify_production_inputs(manifest, require_unchanged=False)
    for path, expected in source_hashes.items():
        if sha256(source_files[path]) != expected:
            raise RuntimeError(f"Source code changed during assembly: {path}")
    output = {
        "completed_at_utc": datetime.now(timezone.utc).isoformat(), "elapsed_seconds": time.monotonic() - started,
        "path": str(target.relative_to(ROOT)), "sha256": sha256(target), "bytes": target.stat().st_size,
        "rows": len(pooled), "tickers": int(pooled.ticker.nunique()), "dates": int(pooled.date.nunique()),
        "date_start": str(pooled.date.min().date()), "date_end": str(pooled.date.max().date()),
        "mrcy_rows": int(pooled.ticker.eq("MRCY").sum()), "heldout_rows": 0,
        "baseline_columns": columns, "raw_features": len(columns), "coverage": coverage,
        "absent_ticker_columns_kept_unknown": absent_ticker_columns,
        "rows_by_year": {str(year): int(count) for year, count in pooled.groupby(pooled.date.dt.year, observed=True).size().items()},
        "eligibility": "Global252-session warmup; finite positive current open/close and finite target. No $5, ADV, or per-ticker252-valid-history gate. Candidate missing values remain unknown when joined later.",
        "label": "Same-day close/open minus one from the same immutable split-adjusted source as all baseline price features.",
        "timing_adjustment": "Historical cluster_overnight_gap uses prior membership and current opening gaps, matching live timing.",
        "context_policy": "Frozen observed news columns and previous-calendar-day weather; first_seen-aware breadth/trajectory remain missing before their coverage. Current static sectors and universe remain retrospective.",
        "source_hashes": source_hashes, "input_manifest_sha256": sha256(HERE / "input_manifest.json"),
        "source_locations": {path: str(source.relative_to(ROOT)) for path, source in source_files.items()},
        "frozen_input_verification": frozen_check,
        "production_before": before, "production_after": after,
        "limitations": "No complete five-year corporate-action certification, no point-in-time dead-stock universe, no execution validation. Existing causal pct_change padding semantics are inherited inside baseline functions; raw bars were not filled.",
    }
    write_json(HERE / "baseline_manifest.json", output)
    write_json(HERE / "baseline_ticker_rows.json", ticker_counts)
    log("baseline_complete", rows=len(pooled), tickers=output["tickers"], dates=output["dates"], mrcy_rows=output["mrcy_rows"], bytes=output["bytes"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("prices", "assemble"), required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        raise ValueError("Use one to four workers to preserve workstation capacity")
    if args.phase == "prices":
        prepare_prices(args.workers)
    else:
        assemble_baseline()


if __name__ == "__main__":
    main()
