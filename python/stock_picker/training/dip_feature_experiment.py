"""Isolated, predeclared morning-dip ablations; never activates a saved model.

Input panels are already aligned to the entry date. They must come from the
same price basis. Rank inner selection is persisted before outer metrics are
computed. Each Fit variant uses strictly earlier SVM fits trained on that
variant's normal raw input schema, including dip inputs for the augmented model.
Numeric checkpoints bind inputs, code, rows and parameters, and permit resuming
long SVM work without accepting in-sample replacements.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import multiprocessing
import os
from pathlib import Path
import tempfile
import time
import warnings

import numpy as np
import pandas as pd

from stock_picker.features.dip_context import DIP_COLUMNS, DIP_FAMILIES, DIP_FLAG_COLUMNS
from stock_picker.features.stacked_svm import STACKED_SVM_COLUMNS
from stock_picker.training.dataset import LABEL_COLUMN
from stock_picker.training.model import (
    SVC_DEFAULT_PARAMS, SVR_DEFAULT_PARAMS, fit_stacked_svm_estimators, predict,
    train_lightgbm, train_lightgbm_rank,
)
from stock_picker.training.svm_stack import score_stacked_svm

IDENTITY = ["date", "ticker"]
SCHEMA_VERSION = 2


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def digest_json(value) -> str:
    return hashlib.sha256(json.dumps(json_safe(value), sort_keys=True, allow_nan=False).encode()).hexdigest()


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        json.dump(json_safe(value), stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        temporary = Path(stream.name)
    temporary.replace(path)


def row_digest(frame: pd.DataFrame) -> str:
    if frame.duplicated(IDENTITY).any() or frame[IDENTITY].isna().any().any():
        raise ValueError("rows require unique, non-null date/ticker identities")
    values = pd.util.hash_pandas_object(frame[IDENTITY], index=False).to_numpy(dtype="uint64")
    return hashlib.sha256(values.tobytes()).hexdigest()


class Checkpoints:
    """Atomic numeric-only arrays; exact contexts are required on every reuse."""

    def __init__(self, directory: Path, manifest: dict, *, resume: bool):
        self.directory = directory
        path = directory / "manifest.json"
        if path.exists():
            if not resume or json.loads(path.read_text()) != json_safe(manifest):
                raise ValueError("existing experiment requires --resume and an identical manifest")
        else:
            if directory.exists() and any(directory.iterdir()):
                raise ValueError("refusing a nonempty output directory without its manifest")
            atomic_json(path, manifest)
        self.identity = digest_json(manifest)

    def _path(self, key: str) -> Path:
        if not key or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for c in key):
            raise ValueError("invalid checkpoint key")
        return self.directory / "checkpoints" / key

    def load(self, key: str, context: dict):
        path = self._path(key)
        metadata_path = path.with_suffix(".json")
        if not metadata_path.exists():
            return None
        metadata = json.loads(metadata_path.read_text())
        if metadata["run"] != self.identity or metadata["context"] != json_safe(context):
            raise ValueError(f"checkpoint context mismatch: {key}")
        array_path = path.with_suffix(".npy")
        if file_digest(array_path) != metadata["sha256"]:
            raise ValueError(f"checkpoint checksum mismatch: {key}")
        values = np.load(array_path, allow_pickle=False)
        if values.dtype.kind not in "fiu" or not np.isfinite(values).all():
            raise ValueError(f"invalid checkpoint values: {key}")
        if list(values.shape) != metadata["shape"]:
            raise ValueError(f"checkpoint shape mismatch: {key}")
        return values, metadata["details"]

    def save(self, key: str, context: dict, values: np.ndarray, details: dict) -> None:
        values = np.asarray(values)
        if values.dtype.kind not in "fiu" or not np.isfinite(values).all():
            raise ValueError("only finite numeric checkpoint arrays are supported")
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
            np.save(stream, values, allow_pickle=False)
            temporary = Path(stream.name)
        array_path = path.with_suffix(".npy")
        temporary.replace(array_path)
        atomic_json(path.with_suffix(".json"), {
            "run": self.identity, "context": context, "details": details,
            "shape": list(values.shape), "sha256": file_digest(array_path),
        })


def align_panels(baseline: pd.DataFrame, dips: pd.DataFrame, raw_columns: list[str],
                 holdout: set[str]) -> pd.DataFrame:
    """Retain every baseline row, including unavailable candidate histories."""
    if set(baseline.columns) != set(IDENTITY + [LABEL_COLUMN] + raw_columns):
        raise ValueError("baseline panel must contain only identities, target and frozen raw inputs")
    if set(dips.columns) != set(IDENTITY + list(DIP_COLUMNS)):
        raise ValueError("candidate panel must contain exactly the declared dip schema")
    baseline, dips = baseline.copy(), dips.copy()
    for frame in (baseline, dips):
        frame["date"] = pd.to_datetime(frame["date"])
        row_digest(frame)
    if set(baseline.ticker) & (holdout | {"SPY"}):
        raise ValueError("holdout or benchmark rows entered the model panel")
    if not np.isfinite(baseline[LABEL_COLUMN]).all():
        raise ValueError("labels must be finite before candidate alignment")
    merged = baseline.merge(dips, on=IDENTITY, how="left", validate="one_to_one", indicator=True)
    if not merged["_merge"].eq("both").all():
        raise ValueError("candidate panel is missing baseline row identities")
    merged = merged.drop(columns="_merge").sort_values(IDENTITY).reset_index(drop=True)
    feature_names = raw_columns + list(DIP_COLUMNS)
    merged[feature_names] = merged[feature_names].replace([np.inf, -np.inf], np.nan)
    return merged[IDENTITY + [LABEL_COLUMN] + feature_names]


def temporal_window(frame: pd.DataFrame, block: dict, sessions: int):
    dates = np.sort(frame.date.unique())
    earlier = dates[dates < np.datetime64(block["from"])][-sessions:]
    if len(earlier) != sessions:
        raise ValueError(f"{block['name']} needs {sessions} prior sessions")
    train = frame.loc[frame.date.isin(earlier)].copy()
    test = frame.loc[frame.date.between(block["from"], block["through"])].copy()
    if test.empty or train.date.max() >= test.date.min():
        raise ValueError("every fit date must strictly precede every scored date")
    return train, test


def variant_columns(gains: dict[str, float] | None = None) -> dict[str, list[str]]:
    continuous = [name for name in DIP_COLUMNS if name not in DIP_FLAG_COLUMNS]
    compact = sorted((name for name in continuous if (gains or {}).get(name, 0) > 0),
                     key=lambda name: (-(gains or {})[name], name))[:8]
    return {
        "baseline": [], **{name: list(columns) for name, columns in DIP_FAMILIES.items()},
        "flags_only": list(DIP_FLAG_COLUMNS), "all": list(DIP_COLUMNS),
        "continuous_only": continuous, "compact": [*DIP_FLAG_COLUMNS, *compact],
    }


def daily_metrics(frame: pd.DataFrame, scores: np.ndarray, *, fit_gate: float = 0.005) -> list[dict]:
    if np.asarray(scores).shape != (len(frame),) or not np.isfinite(scores).all():
        raise ValueError("scores must be finite and aligned to scored rows")
    work = frame[IDENTITY + [LABEL_COLUMN]].copy()
    work["score"] = scores
    records = []
    for when, group in work.groupby("date", sort=True):
        ordered = group.sort_values(["score", "ticker"], ascending=[False, True], kind="stable")
        rank_score, rank_return = group.score.rank(), group[LABEL_COLUMN].rank()
        ic = rank_score.corr(rank_return) if rank_score.nunique() > 1 and rank_return.nunique() > 1 else np.nan
        row = {"date": str(pd.Timestamp(when).date()), "rank_ic": ic, "rows": len(group)}
        for name, selected in [("top5", ordered.head(5)), ("top20", ordered.head(20)),
                               ("gate", ordered.loc[ordered.score > fit_gate])]:
            returns = selected[LABEL_COLUMN]
            row.update({f"{name}_mean": float(returns.mean()) if len(returns) else 0.0,
                        f"{name}_n": len(returns),
                        f"{name}_hits": int(returns.gt(0).sum()),
                        f"{name}_q05": float(returns.quantile(.05)) if len(returns) else None,
                        f"{name}_worst": float(returns.min()) if len(returns) else None})
        records.append(row)
    return json_safe(records)


def summarize(records: list[dict], costs: list[int], *, fit: bool) -> dict:
    table = pd.DataFrame(records)
    result = {"days": len(table), "rows": int(table.rows.sum()), "rank_ic": table.rank_ic.mean()}
    for name in ["top5", "top20", *(["gate"] if fit else [])]:
        count = table[f"{name}_n"]
        active = count.gt(0)
        mean = table[f"{name}_mean"]
        result[name] = {
            "daily_mean": mean.mean(), "active_daily_mean": mean.loc[active].mean(),
            "trade_weighted_mean": (mean * count).sum() / count.sum() if count.sum() else None,
            "trades": int(count.sum()), "active_days": int(active.sum()), "cash_days": int((~active).sum()),
            "hit_rate": table[f"{name}_hits"].sum() / count.sum() if count.sum() else None,
            "daily_q05": mean.quantile(.05), "worst_day": mean.min(),
            "worst_selected_stock": table[f"{name}_worst"].min(),
            "cost_sensitivity": {str(bps): float((mean - active * bps / 10000).mean()) for bps in costs},
        }
    return json_safe(result)


def paired_week_interval(candidate: list[dict], baseline: list[dict], field: str,
                         replications: int, seed: int, *, cost_bps: int = 0) -> dict:
    left, right = pd.DataFrame(candidate).set_index("date"), pd.DataFrame(baseline).set_index("date")
    if not left.index.equals(right.index):
        raise ValueError("paired comparisons require identical dates")
    difference = left[field] - right[field]
    if cost_bps:
        if not field.endswith("_mean"):
            raise ValueError("transaction costs apply only to portfolio returns")
        count_field = field.removesuffix("_mean") + "_n"
        # A cash day earns zero and incurs zero trading cost. Keep all matched
        # dates, even when only one strategy trades, in the paired statistic.
        difference -= (left[count_field].gt(0).astype(int) - right[count_field].gt(0).astype(int)) * cost_bps / 10000
    difference = difference.dropna()
    if difference.empty:
        return {"mean": None, "ci95": [None, None], "weeks": 0}
    dates = pd.to_datetime(difference.index).isocalendar()
    groups = difference.groupby([dates.year.to_numpy(), dates.week.to_numpy()]).agg(["sum", "count"])
    rng = np.random.default_rng(seed)
    choices = rng.integers(0, len(groups), size=(replications, len(groups)))
    samples = groups["sum"].to_numpy()[choices].sum(axis=1) / groups["count"].to_numpy()[choices].sum(axis=1)
    return {"mean": float(difference.mean()), "ci95": np.quantile(samples, [.025, .975]).tolist(),
            "weeks": len(groups), "days": len(difference), "cost_roundtrip_bps": cost_bps,
            "date_weighting": "all matched dates, including cash days; undefined IC pairs omitted",
            "pointwise_exploratory": True}


def select_inner(results: dict, inner_names: list[str]) -> dict:
    """The caller supplies inner blocks only; reject outer blocks explicitly."""
    if set(results) != set(inner_names) or any(not name.startswith("inner_") for name in inner_names):
        raise ValueError("selection accepts precisely the declared inner blocks")
    candidates = []
    diagnostics = {}
    for name in variant_columns():
        if name == "baseline":
            continue
        deltas, counts = [], []
        for block in inner_names:
            current, base = results[block][name], results[block]["baseline"]
            current_values = [current["summary"]["rank_ic"], current["summary"]["top20"]["daily_mean"]]
            base_values = [base["summary"]["rank_ic"], base["summary"]["top20"]["daily_mean"]]
            deltas.append([float(now) - float(before) if now is not None and before is not None else np.nan
                           for now, before in zip(current_values, base_values)])
            counts.append(len(current["added_columns"]))
        eligible = all(np.isfinite(value) and value > 0 for pair in deltas for value in pair)
        diagnostics[name] = {"qualified": eligible, "paired_ic_top20_by_block": deltas}
        if eligible:
            mean_ic, mean_return = np.mean(deltas, axis=0)
            candidates.append((-mean_return, -mean_ic, np.mean(counts), name))
    selected = min(candidates)[-1] if candidates else "baseline"
    return json_safe({"selected": selected, "diagnostics": diagnostics,
                      "basis": "Rank inner only; both finite IC and top20 must improve in each block"})


def _model_checkpoint(run: Checkpoints, key: str, train: pd.DataFrame, test: pd.DataFrame,
                      columns: list[str], protocol: dict, *, fit: bool):
    if train.empty or test.empty or train.date.max() >= test.date.min():
        raise ValueError("model training must strictly precede every scored date")
    # Positive projection prevents labels/metadata/unused candidates becoming inputs.
    train = train[IDENTITY + [LABEL_COLUMN] + columns]
    context = {"train_rows": row_digest(train), "test_rows": row_digest(test), "columns": columns,
               "fit_end": str(train.date.max()), "score_start": str(test.date.min()),
               "fit": fit, "rounds": protocol["rounds"], "params": protocol["lightgbm_params"]}
    cached = run.load(key, context)
    if cached is not None:
        return cached
    print(f"{key}: fitting {len(train):,} rows, {len(columns)} inputs", flush=True)
    started = time.monotonic()
    trainer = train_lightgbm if fit else train_lightgbm_rank
    trained = trainer(train, params=protocol["lightgbm_params"], num_boost_round=protocol["rounds"],
                      included_features=set(columns))
    if trained.feature_names != columns:
        raise ValueError("trained input order/schema differs from frozen projection")
    scores = predict(trained, test[columns])
    details = {"gain": dict(zip(columns, trained.estimator.feature_importance(importance_type="gain").tolist())),
               "elapsed_seconds": time.monotonic() - started, "train_rows": len(train), "test_rows": len(test),
               "train_from": str(train.date.min()), "train_through": str(train.date.max()),
               "test_from": str(test.date.min()), "test_through": str(test.date.max())}
    run.save(key, context, scores, details)
    print(f"{key}: complete in {details['elapsed_seconds']:.1f}s", flush=True)
    return scores, details


def evaluate_block(run: Checkpoints, train: pd.DataFrame, test: pd.DataFrame, raw: list[str],
                   protocol: dict, block: dict, *, fit: bool,
                   fit_frames: dict[str, tuple[pd.DataFrame, pd.DataFrame]] | None = None) -> dict:
    output, gain = {}, {}
    variants = variant_columns()
    names = protocol["fit_variants"] if fit else protocol["initial_variants"] + protocol["refinement_variants"]
    if fit and (fit_frames is None or set(fit_frames) != set(names)):
        raise ValueError("each Fit variant requires its own chronological SVM outputs")
    for name in names:
        if name == "compact":
            variants = variant_columns(gain)
        added = variants[name]
        if fit:
            train, test = fit_frames[name]
        columns = raw + added + (list(STACKED_SVM_COLUMNS) if fit else [])
        scores, details = _model_checkpoint(run, f"{'fit' if fit else 'rank'}_{block['name']}_{name}",
                                           train, test, columns, protocol, fit=fit)
        if name == "all":
            gain = details["gain"]
        daily = daily_metrics(test, scores, fit_gate=protocol["fit_gate"])
        if not fit:
            daily = [{key: value for key, value in row.items() if not key.startswith("gate_")} for row in daily]
        output[name] = {"added_columns": added, "details": details, "daily": daily,
                        "summary": summarize(daily, protocol["cost_roundtrip_bps"], fit=fit)}
        if name != "baseline":
            uncertainty = protocol["uncertainty"]
            output[name]["paired_baseline"] = {
                field: paired_week_interval(daily, output["baseline"]["daily"], field,
                                             uncertainty["bootstrap_replications"], uncertainty["seed"])
                for field in ["rank_ic", "top5_mean", "top20_mean", *(["gate_mean"] if fit else [])]
            }
            output[name]["paired_net_baseline"] = {
                field: {str(bps): paired_week_interval(
                    daily, output["baseline"]["daily"], field,
                    uncertainty["bootstrap_replications"], uncertainty["seed"], cost_bps=bps)
                        for bps in protocol["cost_roundtrip_bps"]}
                for field in ["top5_mean", "top20_mean", *(["gate_mean"] if fit else [])]
            }
        atomic_json(run.directory / f"{'fit' if fit else 'rank'}_{block['name']}.json", output)
    return output


def _read_baseline(path: Path, through: str, from_date=None) -> pd.DataFrame:
    # Parquet pushdown excludes later labels during inner selection.
    filters = [("date", "<=", pd.Timestamp(through))]
    if from_date is not None:
        filters.append(("date", ">=", pd.Timestamp(from_date)))
    return pd.read_parquet(path, filters=filters)


def _svm_block_job(job: dict) -> dict:
    """One variant/block, seven exact fits; variant and schema bind every cache."""
    run = Checkpoints(Path(job["output"]), job["manifest"], resume=True)
    block, protocol, raw = job["block"], job["protocol"], job["raw"]
    # Read only the worker's training/scoring interval, keeping three parallel
    # full-schema SVM jobs within the recorded memory budget.
    calendar = pd.read_parquet(Path(job["baseline"]), columns=["date"])
    earlier = np.sort(pd.to_datetime(calendar.date).unique())
    earlier = earlier[earlier < np.datetime64(block["from"])][-protocol["training_sessions"]:]
    if len(earlier) != protocol["training_sessions"]:
        raise ValueError("SVM block has insufficient prior sessions")
    from_date = pd.Timestamp(earlier[0])
    baseline = _read_baseline(Path(job["baseline"]), block["through"], from_date)
    dips = pd.read_parquet(Path(job["dips"]), filters=[("date", ">=", from_date),
                                                     ("date", "<=", pd.Timestamp(block["through"]))])
    frame = align_panels(baseline, dips, job["baseline_raw"], set(job["holdout"]))
    train, test = temporal_window(frame, block, protocol["training_sessions"])
    train = train[IDENTITY + [LABEL_COLUMN] + raw]
    test = test[IDENTITY + [LABEL_COLUMN] + raw]
    context = {"variant": job["variant"], "train_rows": row_digest(train), "test_rows": row_digest(test), "columns": raw,
               "fit_end": str(train.date.max()), "score_start": str(test.date.min())}
    for column in STACKED_SVM_COLUMNS:
        key = f"svm_{job['variant']}_{block['name']}_{column}"
        if run.load(key, context) is not None:
            continue
        started = time.monotonic()
        print(f"{key}: exact fit on {len(train):,} earlier rows", flush=True)
        with warnings.catch_warnings(record=True) as observed:
            warnings.simplefilter("always")
            estimators = fit_stacked_svm_estimators(train, included_features=set(raw), outputs=(column,))
        if estimators[column].feature_names != raw:
            raise ValueError("SVM source schema changed")
        values = score_stacked_svm(test, estimators, (column,))[column].to_numpy()
        details = {"elapsed_seconds": time.monotonic() - started, "train_rows": len(train),
                   "test_rows": len(test), "fit_from": str(train.date.min()), "fit_through": str(train.date.max()),
                   "warnings": [{"category": warning.category.__name__, "message": str(warning.message)}
                                for warning in observed]}
        run.save(key, context, values, details)
        print(f"{key}: complete in {details['elapsed_seconds']:.1f}s", flush=True)
    return {"variant": job["variant"], "block": block, "context": context, "rows": len(test)}


def run_fit(run: Checkpoints, panel: pd.DataFrame, raw: list[str], protocol: dict,
            baseline_path: Path, dip_path: Path, holdout: set[str], manifest: dict) -> dict:
    blocks = [*protocol["fit_oof_blocks"], protocol["fit_outer_block"]]
    if protocol["fit_variants"] != ["baseline", "all"]:
        raise ValueError("full normal Fit confirmation is predeclared for baseline and all inputs")
    variants = variant_columns()
    jobs = [{"output": str(run.directory), "manifest": manifest, "block": block,
             "protocol": protocol, "raw": raw + variants[name], "baseline_raw": raw,
             "variant": name, "baseline": str(baseline_path), "dips": str(dip_path),
             "holdout": sorted(holdout)} for name in protocol["fit_variants"] for block in blocks]
    # Children import NumPy/sklearn after inheriting these limits. The parent
    # LightGBM threads are separately fixed in the recorded protocol.
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[name] = "1"
    with ProcessPoolExecutor(max_workers=protocol["svm_block_workers"],
                             mp_context=multiprocessing.get_context("spawn")) as pool:
        completed = list(pool.map(_svm_block_job, jobs))
    stacked = {name: [] for name in protocol["fit_variants"]}
    for item in completed:
        block = item["block"]
        scored = panel.loc[panel.date.between(block["from"], block["through"])].copy()
        if row_digest(scored) != item["context"]["test_rows"]:
            raise ValueError("SVM and head scored row order differ")
        for column in STACKED_SVM_COLUMNS:
            values, _ = run.load(f"svm_{item['variant']}_{block['name']}_{column}", item["context"])
            if values.shape != (len(scored),):
                raise ValueError("SVM output row count mismatch")
            scored[column] = values
        stacked[item["variant"]].append(scored)
    full_train, full_test = temporal_window(panel, protocol["fit_outer_block"], protocol["training_sessions"])
    fit_frames, coverage = {}, {}
    train_identity = None
    for name, tables in stacked.items():
        train = pd.concat(tables[:-1]).loc[lambda frame: frame.date.isin(full_train.date.unique())]
        test = tables[-1]
        if row_digest(test) != row_digest(full_test) or train.date.max() >= test.date.min():
            raise ValueError("Fit head chronology or scoring population mismatch")
        if train_identity is not None and row_digest(train) != train_identity:
            raise ValueError("Fit variants must use identical OOF-covered training rows")
        train_identity = row_digest(train)
        fit_frames[name] = (train, test)
        coverage[name] = {"full_training_rows": len(full_train), "oof_covered_rows": len(train),
                          "omitted_rows": len(full_train) - len(train), "first_covered_date": str(train.date.min()),
                          "full_training_sessions": full_train.date.nunique(), "covered_sessions": train.date.nunique(),
                          "svm_raw_sources": len(raw) + len(variants[name]), "test_rows": len(test)}
    atomic_json(run.directory / "fit_oof_coverage.json", coverage)
    return evaluate_block(run, full_train, full_test, raw, protocol, protocol["fit_outer_block"],
                          fit=True, fit_frames=fit_frames)


def create_manifest(root: Path, paths: dict[str, Path], raw: list[str], protocol: dict) -> dict:
    source_paths = sorted([*root.glob("python/stock_picker/features/*.py"),
                           *root.glob("python/stock_picker/training/*.py")])
    panel_manifests = {name: paths[name].with_name(paths[name].stem + "_manifest.json")
                       for name in ("baseline", "dips")}
    return {"schema": SCHEMA_VERSION, "inputs": {name: file_digest(path) for name, path in paths.items()},
            "frozen_panel_provenance": {name: file_digest(path) for name, path in panel_manifests.items()},
            "sources": {str(path.relative_to(root)): file_digest(path) for path in source_paths},
            "versions": {name: version(name) for name in ["numpy", "pandas", "lightgbm", "scikit-learn", "pyarrow"]},
            "raw_columns": raw, "candidate_columns": list(DIP_COLUMNS),
            "fit_svm_schemas": {"baseline": raw, "all": raw + list(DIP_COLUMNS)},
            "svr_params": SVR_DEFAULT_PARAMS, "svc_params": SVC_DEFAULT_PARAMS,
            "protocol": protocol, "model_store_writes": False,
            "input_authority": "Only frozen research panels and manifests are read; scheduled changes to live data files do not replace these inputs."}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--dips", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=["rank-inner", "rank-outer", "fit"], required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    root, output = args.root.resolve(), args.output.resolve()
    if root / "research" not in output.parents:
        raise ValueError("experiment outputs must be below this repository's research directory")
    paths = {name: getattr(args, name.replace("-", "_")).resolve()
             for name in ["baseline", "dips", "protocol", "snapshot", "input-manifest"]}
    protocol = json.loads(paths["protocol"].read_text())
    if protocol["families"] != {name: list(columns) for name, columns in DIP_FAMILIES.items()}:
        raise ValueError("dip definitions disagree with the frozen protocol")
    if protocol["flags"] != list(DIP_FLAG_COLUMNS) or protocol["status"] != "frozen_before_model_fits":
        raise ValueError("unfrozen or mismatched protocol")
    snapshot = json.loads(paths["snapshot"].read_text())
    if file_digest(paths["snapshot"]) != protocol["model_snapshot_sha256"]:
        raise ValueError("saved model schema snapshot differs from the frozen protocol")
    raw = snapshot["rank"]["members"][0]["feature_names"]
    fit_names = snapshot["fit"]["members"][0]["feature_names"]
    if len(raw) != 132 or set(fit_names) != set(raw) | set(STACKED_SVM_COLUMNS):
        raise ValueError("saved Rank132/Fit139 schemas do not match")
    manifest = create_manifest(root, paths, raw, protocol)
    run = Checkpoints(output, manifest, resume=args.resume)
    selection_path = output / "selection.json"
    if args.phase != "rank-inner":
        if not selection_path.exists():
            raise ValueError("freeze Rank inner selection before evaluating outer outcomes")
        if json.loads(selection_path.read_text())["run"] != run.identity:
            raise ValueError("selection belongs to another experiment context")
    through = (max(block["through"] for block in protocol["inner_blocks"])
               if args.phase == "rank-inner" else protocol["source_cutoff"])
    baseline = _read_baseline(paths["baseline"], through)
    dips = pd.read_parquet(paths["dips"], filters=[("date", "<=", pd.Timestamp(through))])
    holdout = set(json.loads(paths["input-manifest"].read_text())["holdout_tickers"])
    panel = align_panels(baseline, dips, raw, holdout)
    if "MRCY" not in set(panel.ticker):
        raise ValueError("MRCY must remain in this experiment")
    atomic_json(output / f"{args.phase}_population.json", {
        "rows": len(panel), "tickers": panel.ticker.nunique(), "sessions": panel.date.nunique(),
        "from": str(panel.date.min()), "through": str(panel.date.max()),
        "mrcy_rows": int(panel.ticker.eq("MRCY").sum()), "holdout_rows": 0,
        "candidate_missing_fraction": panel[list(DIP_COLUMNS)].isna().mean().to_dict(),
    })
    if args.phase == "fit":
        run_fit(run, panel, raw, protocol, paths["baseline"], paths["dips"], holdout, manifest)
        return
    blocks = protocol["inner_blocks" if args.phase == "rank-inner" else "rank_outer_blocks"]
    results = {}
    for block in blocks:
        train, test = temporal_window(panel, block, protocol["training_sessions"])
        results[block["name"]] = evaluate_block(run, train, test, raw, protocol, block, fit=False)
    if args.phase == "rank-inner":
        selected = select_inner(results, [block["name"] for block in blocks])
        selected.update({"run": run.identity, "frozen_at_utc": datetime.now(timezone.utc).isoformat()})
        if selection_path.exists():
            previous = json.loads(selection_path.read_text())
            if {k: v for k, v in previous.items() if k != "frozen_at_utc"} != {
                    k: json_safe(v) for k, v in selected.items() if k != "frozen_at_utc"}:
                raise ValueError("refusing to overwrite a different frozen selection")
        else:
            atomic_json(selection_path, selected)
        print(f"Frozen inner selection: {selected['selected']}", flush=True)


if __name__ == "__main__":
    main()
