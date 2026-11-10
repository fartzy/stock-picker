"""Independently audit cached predictions; never fit or change model inputs.

Run with the repository's hermetic Python environment after (or during) the
frozen experiment. Realized Low/High outcomes are joined only after selecting
stocks from cached scores. All outputs stay in a separate research directory.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

# Match the existing research executables without changing frozen Bazel targets.
# Run this script with the Python3.12 interpreter inside these runfiles.
ROOT = Path(__file__).resolve().parents[2]
RUNFILES = (ROOT / "bazel-bin/python/stock_picker/training/morning_dip_experiment.runfiles").resolve()
sys.path[:0] = list(map(str, RUNFILES.glob("rules_python++pip+*/site-packages")))
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np
import pandas as pd

KEYS = ["date", "ticker"]
TARGET = "label_day_session_return"
SVM_OUTPUTS = ["svr_oof_pred", "svc_direction_margin", "svc_gate_margin", "svc_down_gate_margin",
               "svc_strong_up_margin", "svc_top_quintile_margin", "svc_bottom_quintile_margin"]


def safe(value):
    if isinstance(value, dict):
        return {str(k): safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    return None if isinstance(value, float) and not np.isfinite(value) else value


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def identity(frame: pd.DataFrame) -> str:
    if frame[KEYS].isna().any().any() or frame.duplicated(KEYS).any():
        raise ValueError("row keys must be unique and complete")
    return hashlib.sha256(pd.util.hash_pandas_object(frame[KEYS], index=False).to_numpy("uint64").tobytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as stream:
        json.dump(safe(value), stream, indent=2, allow_nan=False)
        stream.write("\n")
        temporary = Path(stream.name)
    temporary.replace(path)


def paired_interval(left: pd.DataFrame, right: pd.DataFrame, field: str, protocol: dict,
                    *, cost_bps: int = 0) -> dict:
    require(left.index.equals(right.index), "paired dates differ")
    difference = left[field] - right[field]
    if cost_bps:
        policy = field.removesuffix("_mean")
        difference = difference - (left[f"{policy}_n"].gt(0).astype(int)
                                   - right[f"{policy}_n"].gt(0).astype(int)) * cost_bps / 10000
    difference = difference.dropna()
    if difference.empty:
        return {"mean": None, "ci95": [None, None], "days": 0, "weeks": 0}
    calendar = pd.to_datetime(difference.index).isocalendar()
    weeks = difference.groupby([calendar.year.to_numpy(), calendar.week.to_numpy()]).agg(["sum", "count"])
    rng = np.random.default_rng(protocol["uncertainty"]["seed"])
    draws = rng.integers(len(weeks), size=(protocol["uncertainty"]["bootstrap_replications"], len(weeks)))
    bootstrap = weeks["sum"].to_numpy()[draws].sum(axis=1) / weeks["count"].to_numpy()[draws].sum(axis=1)
    return safe({"mean": difference.mean(), "ci95": np.quantile(bootstrap, [.025, .975]).tolist(),
                 "days": len(difference), "weeks": len(weeks), "cost_roundtrip_bps": cost_bps,
                 "pointwise_exploratory": True, "cash_days_included": True})


def score_selections(frame: pd.DataFrame, scores: np.ndarray, gate: float, fit: bool):
    """Choose stocks without any current-session Low/High information."""
    require(scores.shape == (len(frame),) and np.isfinite(scores).all(), "invalid cached scores")
    scored = frame[KEYS + [TARGET]].assign(score=scores)
    ordered = scored.sort_values(["date", "score", "ticker"], ascending=[True, False, True], kind="stable")
    selected = {"top5": ordered.groupby("date", sort=False, observed=True).head(5),
                "top20": ordered.groupby("date", sort=False, observed=True).head(20)}
    if fit:
        selected["gate"] = ordered.loc[ordered.score > gate]
    dates = pd.Index(sorted(scored.date.unique()), name="date")
    daily = pd.DataFrame(index=dates)
    daily["rank_ic"] = scored.groupby("date", observed=True).apply(
        lambda group: group.score.rank().corr(group[TARGET].rank())
        if group.score.nunique() > 1 and group[TARGET].nunique() > 1 else np.nan,
        include_groups=False,
    ).reindex(dates)
    for name, rows in selected.items():
        groups = rows.groupby("date", observed=True)[TARGET]
        daily[f"{name}_mean"] = groups.mean().reindex(dates, fill_value=0.)
        daily[f"{name}_n"] = groups.size().reindex(dates, fill_value=0)
    return daily, selected


def fit_error_diagnostic(kind: str, scores: np.ndarray, actual: np.ndarray) -> dict | None:
    """Stock-day-weighted MAE on all scored Fit rows, never on Rank scores."""
    require(kind in {"fit", "rank"}, "unknown model kind")
    if kind == "rank":
        return None  # ListFold scores express relative ordering, not returns.
    prediction, outcome = np.asarray(scores, dtype=float), np.asarray(actual, dtype=float)
    require(prediction.ndim == 1 and prediction.shape == outcome.shape and len(outcome) > 0,
            "Fit error requires matching nonempty stock-day vectors")
    require(np.isfinite(prediction).all() and np.isfinite(outcome).all(), "nonfinite Fit error inputs")
    error = np.abs(prediction - outcome)
    return {"stock_day_mae": float(error.mean()), "absolute_error_sum": float(error.sum()),
            "scored_stock_days": len(error), "units": "return_fraction", "lower_is_better": True,
            "population": "Every scored stock-day on the common frozen outer population, including unselected stocks.",
            "weighting": "Each stock-day has equal weight; this is not an equal-date or selected-trade MAE."}


def excursion_stats(selected: pd.DataFrame, outcomes: pd.DataFrame, dates: pd.Index):
    # This is the sole outcome-sidecar join, downstream of frozen selections.
    rows = selected.merge(outcomes, on=KEYS, how="left", validate="one_to_one", indicator=True)
    require(rows._merge.eq("both").all(), "selected stock missing evaluation outcome")
    low = rows.open_to_low_return.where(rows.low_outcome_valid)
    high = rows.open_to_high_return.where(rows.high_outcome_valid)
    require(not low.gt(0).any() and not high.lt(0).any(), "impossible signed intraday excursion")
    counts = rows.groupby("date", observed=True).size().reindex(dates, fill_value=0)
    daily_low = rows.assign(low=low).groupby("date", observed=True).low.mean().reindex(dates)
    daily_low.loc[counts.eq(0)] = 0.
    return safe({"selected_stock_days": len(rows), "valid_low_stock_days": low.notna().sum(),
                 "valid_high_stock_days": high.notna().sum(),
                 "source_ohlc_consistent_stock_days": rows.source_ohlc_consistent.sum(),
                 "mean_constituent_open_to_low": low.mean(), "median_constituent_open_to_low": low.median(),
                 "p05_constituent_open_to_low": low.quantile(.05), "worst_constituent_open_to_low": low.min(),
                 "mean_constituent_open_to_high": high.mean(),
                 "daily_mean_constituent_open_to_low_including_cash": daily_low.mean(),
                 "not_portfolio_drawdown": "Constituent intraday lows occur at different times; these are not portfolio maximum drawdown or a realizable exit strategy."})


def return_concentration(selected: pd.DataFrame, evaluation_days: int) -> dict:
    """Descriptive attribution of equal-weight daily returns; no new screening."""
    if selected.empty:
        return {"unique_tickers": 0, "largest_positive_contributors": [], "largest_negative_contributors": []}
    rows = selected.copy()
    rows["contribution"] = rows[TARGET] / rows.groupby("date", observed=True)["ticker"].transform("size") / evaluation_days
    ticker = rows.groupby("ticker", observed=True).agg(
        daily_mean_return_contribution=("contribution", "sum"), selected_stock_days=(TARGET, "size"))
    ticker = ticker.reset_index().sort_values(["daily_mean_return_contribution", "ticker"], ascending=[False, True])
    extremes = rows.sort_values([TARGET, "date", "ticker"], ascending=[False, True, True])
    columns = ["date", "ticker", TARGET, "score"]
    extremes["date"] = extremes.date.dt.strftime("%Y-%m-%d")
    absolute = ticker.daily_mean_return_contribution.abs()
    return safe({"unique_tickers": len(ticker),
                 "largest_absolute_ticker_contribution_fraction": absolute.max() / absolute.sum() if absolute.sum() else None,
                 "largest_positive_contributors": ticker.head(5).to_dict("records"),
                 "largest_negative_contributors": ticker.tail(5).iloc[::-1].to_dict("records"),
                 "highest_return_selected_stock_days": extremes.head(5)[columns].to_dict("records"),
                 "lowest_return_selected_stock_days": extremes.tail(5).iloc[::-1][columns].to_dict("records"),
                 "interpretation": "Contributions sum to the gross equal-date portfolio mean; concentration is descriptive and changes no selections."})


def audit_record(record: dict, daily: pd.DataFrame, selected: dict, costs: list[int]) -> None:
    """Check stored return summaries against independently selected stock rows."""
    stored = pd.DataFrame(record["daily"]).set_index("date")
    stored.index = pd.to_datetime(stored.index)
    require(stored.index.equals(daily.index), "recorded daily dates differ")
    for column in daily:
        np.testing.assert_allclose(pd.to_numeric(stored[column], errors="coerce"), daily[column], rtol=1e-10, atol=1e-12, equal_nan=True)
    summary = record["summary"]
    np.testing.assert_allclose(summary["rank_ic"] if summary["rank_ic"] is not None else np.nan,
                               daily.rank_ic.mean(), rtol=1e-10, atol=1e-12, equal_nan=True)
    for name, rows in selected.items():
        actual = summary[name]
        mean, count = daily[f"{name}_mean"], daily[f"{name}_n"]
        require(actual["trades"] == len(rows), "trade count differs")
        require(actual["cash_days"] == int(count.eq(0).sum()), "cash-day count differs")
        np.testing.assert_allclose(actual["daily_mean"], mean.mean(), atol=1e-12)
        for cost in costs:
            expected = (mean - count.gt(0) * cost / 10000).mean()
            np.testing.assert_allclose(actual["cost_sensitivity"][str(cost)], expected, atol=1e-12)


def expected_schema(variant: str, record: dict, raw: list[str], candidates: list[str], protocol: dict, fit: bool) -> list[str]:
    # Manifest JSON sorts dictionary keys, so family-map iteration is not the
    # model input order. The manifest's explicit candidate array is authoritative.
    require(set(candidates) == {column for columns in protocol["families"].values() for column in columns}, "candidate family membership differs")
    mapping = {"baseline": [], **protocol["families"], "flags_only": protocol["flags"], "all": candidates,
               "continuous_only": [column for column in candidates if column not in protocol["flags"]]}
    added = record["added_columns"]
    if variant != "compact":
        require(added == mapping[variant], f"unexpected added columns for {variant}")
    else:
        require(added[:4] == protocol["flags"] and len(added) <= 12 and set(added) <= set(candidates), "invalid compact schema")
    return raw + added + (SVM_OUTPUTS if fit else [])


def audit_svm_checkpoints(run_dir: Path, manifest: dict, run_id: str, panel: pd.DataFrame) -> dict:
    """Audit available full-stack source outputs, including convergence warnings."""
    protocol = manifest["protocol"]
    records, missing = {}, []
    dates = np.sort(panel.date.unique())
    for variant in protocol["fit_variants"]:
        for block in protocol["fit_oof_blocks"] + [protocol["fit_outer_block"]]:
            train_dates = dates[dates < np.datetime64(block["from"])][-protocol["training_sessions"]:]
            train = panel.loc[panel.date.isin(train_dates)]
            test = panel.loc[panel.date.between(block["from"], block["through"])]
            expected = {"variant": variant, "columns": manifest["fit_svm_schemas"][variant],
                        "train_rows": identity(train), "test_rows": identity(test),
                        "fit_end": str(train.date.max()), "score_start": str(test.date.min())}
            for name in SVM_OUTPUTS:
                key = f"svm_{variant}_{block['name']}_{name}"
                path = run_dir / "checkpoints" / key
                if not path.with_suffix(".json").exists():
                    missing.append(key)
                    continue
                metadata = json.loads(path.with_suffix(".json").read_text())
                require(metadata["run"] == run_id and metadata["context"] == expected, f"SVM context mismatch: {key}")
                require(sha(path.with_suffix(".npy")) == metadata["sha256"], f"SVM checksum mismatch: {key}")
                values = np.load(path.with_suffix(".npy"), allow_pickle=False)
                require(values.shape == (len(test),) and list(values.shape) == metadata["shape"], f"SVM row shape mismatch: {key}")
                require(values.dtype.kind in "fiu" and np.isfinite(values).all(), f"invalid SVM scores: {key}")
                require(metadata["details"]["train_rows"] == len(train), f"SVM training count mismatch: {key}")
                records[key] = {"input_count": len(expected["columns"]), "train_rows": len(train), "test_rows": len(test),
                                "fit_through": expected["fit_end"], "score_from": expected["score_start"],
                                "elapsed_seconds": metadata["details"]["elapsed_seconds"],
                                "warnings": metadata["details"].get("warnings", [])}
    convergence = [key for key, record in records.items()
                   if any(warning.get("category") == "ConvergenceWarning" for warning in record["warnings"])]
    return {"audited_outputs": len(records), "missing_outputs": missing, "checkpoints": records,
            "convergence_warning_outputs": convergence,
            "svr_max_iter": manifest["svr_params"]["max_iter"],
            "svc_max_iter": manifest["svc_params"]["max_iter"],
            "convergence_qualification": "Recorded ConvergenceWarnings mean convergence was not established within the unchanged iteration limit; this audit does not refit or suppress warnings."}


def reviewed_context(here: Path, manifest: dict) -> dict:
    """Carry forward separate source/pruning reviews without reading live stores."""
    artifacts = {}
    for name in ["pruning_review.json", "ordinary_prune_final.json"]:
        payload = (here / name).read_bytes()
        artifacts[name] = {"sha256": hashlib.sha256(payload).hexdigest(), "value": json.loads(payload)}
    ordinary = artifacts["ordinary_prune_final.json"]["value"]
    return {
        "review_artifact_sha256": {name: record["sha256"] for name, record in artifacts.items()},
        "source_qualifications": artifacts["pruning_review.json"]["value"]["data_quality_findings"],
        "ordinary_configuration_snapshot": {
            "recorded_at_utc": ordinary["configuration_verified_at_utc"],
            "active_new_dip_fields": ordinary["active_new_dip_fields"],
            "ordinary_pruned_dip_fields": ordinary["ordinary_pruned_dip_fields"],
            "raw_inputs": ordinary["expected_raw_inputs"], "fit_head_inputs": ordinary["expected_fit_head_inputs"],
            "source": "ordinary_prune_final.json; a recorded configuration snapshot, not a live-store read"},
        "frozen_research_configuration": {
            "new_dip_fields": len(manifest["candidate_columns"]),
            "baseline_raw_inputs": len(manifest["fit_svm_schemas"]["baseline"]),
            "all_raw_inputs": len(manifest["fit_svm_schemas"]["all"]),
            "all_fit_head_inputs": len(manifest["fit_svm_schemas"]["all"]) + len(SVM_OUTPUTS)},
        "interpretation": "Frozen all21 research includes both ordinarily pruned ratio fields. Its153-raw/160-head Fit result does not validate the separate ordinary19-new-input,151-raw/158-head configuration. Source identity/price/volume qualifications limit confirmatory interpretation; no frozen rows were silently removed or adjusted."}


def audit(args):
    here = Path(__file__).resolve().parent
    root = here.parents[1]
    run_dir, output = args.run.resolve(), args.output.resolve()
    require(root / "research" in output.parents and run_dir not in output.parents and output != run_dir,
            "audit output must be a separate repository research directory")
    manifest = json.loads((run_dir / "manifest.json").read_text())
    run_id = hashlib.sha256(json.dumps(manifest, sort_keys=True, allow_nan=False).encode()).hexdigest()
    protocol = manifest["protocol"]
    paths = {"baseline": here / "baseline.parquet", "dips": here / "dip_panel.parquet",
             "protocol": here / "experiment_protocol.json", "snapshot": here / "saved_model_snapshot.json",
             "input-manifest": here / "input_manifest.json"}
    for name, path in paths.items():
        require(sha(path) == manifest["inputs"][name], f"frozen input changed: {name}")
    for relative, digest in manifest["sources"].items():
        require(sha(root / relative) == digest, f"frozen source changed: {relative}")
    for name in ("baseline", "dips"):
        require(sha(paths[name].with_name(paths[name].stem + "_manifest.json")) == manifest["frozen_panel_provenance"][name], "panel provenance changed")
    side_manifest_path = here / "evaluation_outcomes_manifest.json"
    side_manifest = json.loads(side_manifest_path.read_text())
    side_path = here / "evaluation_outcomes.parquet"
    require(sha(side_path) == side_manifest["sha256"], "outcome sidecar checksum mismatch")
    require(side_manifest["baseline_sha256"] == manifest["inputs"]["baseline"], "sidecar uses another baseline")
    require(side_manifest["input_manifest_sha256"] == manifest["inputs"]["input-manifest"], "sidecar uses another input manifest")
    panel = pd.read_parquet(paths["baseline"], columns=KEYS + [TARGET]).sort_values(KEYS).reset_index(drop=True)
    dips = pd.read_parquet(paths["dips"]).sort_values(KEYS).reset_index(drop=True)
    outcomes = pd.read_parquet(side_path).sort_values(KEYS).reset_index(drop=True)
    panel_identity = identity(panel)
    require(identity(dips) == identity(outcomes) == panel_identity, "evaluation/candidate rows differ from model population")
    require(panel.ticker.eq("MRCY").any(), "MRCY is missing")
    require(not set(panel.ticker) & (set(json.loads(paths["input-manifest"].read_text())["holdout_tickers"]) | {"SPY"}), "forbidden holdout/benchmark rows")
    candidates = manifest["candidate_columns"]
    panel[candidates] = dips[candidates].replace([np.inf, -np.inf], np.nan)
    del dips
    require(np.isfinite(panel[TARGET]).all(), "nonfinite outcomes")
    selection_path = run_dir / "selection.json"
    selection = json.loads(selection_path.read_text()) if selection_path.exists() else None
    if selection:
        require(selection["run"] == run_id, "selection belongs to a different run")
    report = {"generated_at_utc": datetime.now(timezone.utc).isoformat(), "run_identity": run_id,
              "audit_script_sha256": sha(Path(__file__)), "outcome_manifest_sha256": sha(side_manifest_path),
              "protocol_version": protocol["version"], "frozen_selection": selection,
              "eligible_rows": len(panel), "mrcy_eligible_rows": int(panel.ticker.eq("MRCY").sum()),
              "missing_results": [], "blocks": {}, "pooled_outer": {}, "features": {},
              "interpretation": ["Ordinary features remain subject to normal pruning; this report changes no configuration.",
                                 "Training gain measures model usage, not causal or individually proven predictive value.",
                                 "Intervals are pointwise exploratory, with no multiplicity correction; inner results guided selection.",
                                 "The Fit comparison covers one outer block; Rank refinements have Rank-only evidence.",
                                 "Realized Low/High outcomes are evaluation-only and never used to select stocks or fit models."]}
    report["reviewed_context"] = reviewed_context(here, manifest)
    daily_by_block = {}
    blocks = [("rank", block, protocol["initial_variants"] + protocol["refinement_variants"])
              for block in protocol["inner_blocks"] + protocol["rank_outer_blocks"]]
    blocks.append(("fit", protocol["fit_outer_block"], protocol["fit_variants"]))
    for kind, block, variants in blocks:
        key = f"{kind}_{block['name']}"
        result_path = run_dir / f"{key}.json"
        records = json.loads(result_path.read_text()) if result_path.exists() else {}
        if records and block["name"].startswith("outer_"):
            require(selection is not None, "outer result exists without frozen selection")
        report["blocks"][key], daily_by_block[key] = {}, {}
        dates = np.sort(panel.date.unique())
        train_dates = dates[dates < np.datetime64(block["from"])][-protocol["training_sessions"]:]
        train = panel.loc[panel.date.isin(train_dates)]
        if kind == "fit":
            covered = pd.Series(False, index=train.index)
            for oof in protocol["fit_oof_blocks"]:
                covered |= train.date.between(oof["from"], oof["through"])
            train = train.loc[covered]
        test = panel.loc[panel.date.between(block["from"], block["through"])]
        block_outcomes = outcomes.loc[outcomes.date.between(block["from"], block["through"])]
        for variant in variants:
            if variant not in records:
                report["missing_results"].append(f"{key}/{variant}")
                continue
            record = records[variant]
            checkpoint = run_dir / "checkpoints" / f"{key}_{variant}"
            metadata = json.loads(checkpoint.with_suffix(".json").read_text())
            require(metadata["run"] == run_id and sha(checkpoint.with_suffix(".npy")) == metadata["sha256"], "checkpoint provenance mismatch")
            context = metadata["context"]
            require(context["train_rows"] == identity(train) and context["test_rows"] == identity(test), "checkpoint row identities differ")
            require(context["columns"] == expected_schema(variant, record, manifest["raw_columns"], candidates, protocol, kind == "fit"), "checkpoint feature schema differs")
            require(context["rounds"] == protocol["rounds"] and context["params"] == protocol["lightgbm_params"], "checkpoint model parameters differ")
            require(bool(context["fit"]) == (kind == "fit") and train.date.max() < test.date.min(), "checkpoint model type/chronology mismatch")
            scores = np.load(checkpoint.with_suffix(".npy"), allow_pickle=False)
            require(list(scores.shape) == metadata["shape"] and scores.dtype.kind in "fiu", "invalid checkpoint array")
            daily, selected = score_selections(test, scores, protocol["fit_gate"], kind == "fit")
            audit_record(record, daily, selected, protocol["cost_roundtrip_bps"])
            daily_by_block[key][variant] = daily
            gains = metadata["details"]["gain"]
            require(gains == record["details"]["gain"], "recorded feature gain differs from checkpoint")
            total_gain = sum(gains.values())
            result = {"summary": record["summary"], "added_columns": record["added_columns"],
                      "mrcy_eligible_rows": int(test.ticker.eq("MRCY").sum()), "policies": {}, "feature_diagnostics": {}}
            error = fit_error_diagnostic(kind, scores, test[TARGET].to_numpy())
            if error is not None:
                result["fit_error"] = error
            for name, rows in selected.items():
                mrcy = rows.loc[rows.ticker.eq("MRCY")]
                result["policies"][name] = {"mrcy_selected_days": len(mrcy), "mrcy_mean_open_to_close": mrcy[TARGET].mean(),
                                            "intraday_excursion": excursion_stats(rows, block_outcomes, daily.index),
                                            "return_concentration": return_concentration(rows, len(daily))}
            for name in candidates:
                result["feature_diagnostics"][name] = {
                    "included": name in context["columns"], "training_gain": gains.get(name),
                    "fraction_of_total_training_gain": gains.get(name, 0) / total_gain if total_gain else None,
                    "train_nonmissing_fraction": train[name].notna().mean(), "test_nonmissing_fraction": test[name].notna().mean(),
                    "train_distinct_nonmissing": train[name].nunique(), "test_distinct_nonmissing": test[name].nunique()}
            report["blocks"][key][variant] = safe(result)
        if "all" in records and "compact" in records:
            gains = records["all"]["details"]["gain"]
            expected = sorted((name for name in candidates if name not in protocol["flags"] and gains.get(name, 0) > 0), key=lambda name: (-gains[name], name))[:8]
            require(records["compact"]["added_columns"] == protocol["flags"] + expected, "compact does not use only declared training gains")
        baseline_daily = daily_by_block[key].get("baseline")
        if baseline_daily is not None:
            for variant, daily in daily_by_block[key].items():
                if variant == "baseline":
                    continue
                pairs = {field: {str(cost): paired_interval(daily, baseline_daily, field, protocol, cost_bps=cost)
                                 for cost in protocol["cost_roundtrip_bps"]}
                         for field in ["top5_mean", "top20_mean"] + (["gate_mean"] if kind == "fit" else [])}
                report["blocks"][key][variant]["paired_net_baseline"] = pairs
                report["blocks"][key][variant]["paired_rank_ic"] = paired_interval(daily, baseline_daily, "rank_ic", protocol)
                if kind == "fit":
                    current_error = report["blocks"][key][variant]["fit_error"]
                    baseline_error = report["blocks"][key]["baseline"]["fit_error"]
                    require(current_error["scored_stock_days"] == baseline_error["scored_stock_days"], "Fit MAE populations differ")
                    current_error["mae_delta_vs_baseline"] = current_error["stock_day_mae"] - baseline_error["stock_day_mae"]
                for field, costs in pairs.items():
                    for cost, calculated in costs.items():
                        recorded = records[variant]["paired_net_baseline"][field][cost]
                        np.testing.assert_allclose([calculated["mean"], *calculated["ci95"]], [recorded["mean"], *recorded["ci95"]], atol=1e-12)
    # Independently reproduce the frozen research choice from audited inner data.
    inner_keys = [f"rank_{block['name']}" for block in protocol["inner_blocks"]]
    inner_variants = protocol["initial_variants"] + protocol["refinement_variants"]
    if selection and all(set(inner_variants) <= set(daily_by_block[key]) for key in inner_keys):
        qualifying = []
        for variant in inner_variants:
            if variant == "baseline":
                continue
            deltas, sizes = [], []
            for key in inner_keys:
                candidate, baseline = daily_by_block[key][variant], daily_by_block[key]["baseline"]
                deltas.append([candidate.rank_ic.mean() - baseline.rank_ic.mean(), candidate.top20_mean.mean() - baseline.top20_mean.mean()])
                sizes.append(len(report["blocks"][key][variant]["added_columns"]))
            if np.isfinite(deltas).all() and (np.asarray(deltas) > 0).all():
                mean_ic, mean_return = np.mean(deltas, axis=0)
                qualifying.append((-mean_return, -mean_ic, np.mean(sizes), variant))
        expected_selection = min(qualifying)[-1] if qualifying else "baseline"
        require(selection["selected"] == expected_selection, "frozen selection disagrees with independently audited inner metrics")
        report["frozen_selection_independently_verified"] = True
    # Pool only the predeclared outer dates, and only when a variant has every block.
    for kind, outer in [("rank", protocol["rank_outer_blocks"]), ("fit", [protocol["fit_outer_block"]])]:
        variants = protocol["fit_variants"] if kind == "fit" else protocol["initial_variants"] + protocol["refinement_variants"]
        names = [f"{kind}_{block['name']}" for block in outer]
        for variant in variants:
            if not all(variant in daily_by_block[name] and "baseline" in daily_by_block[name] for name in names):
                continue
            daily = pd.concat([daily_by_block[name][variant] for name in names]).sort_index()
            baseline_daily = pd.concat([daily_by_block[name]["baseline"] for name in names]).sort_index()
            require(daily.index.is_unique, "outer blocks overlap")
            report["pooled_outer"][f"{kind}/{variant}"] = {
                "days": len(daily), "rank_ic": safe(daily.rank_ic.mean()),
                "paired_rank_ic": paired_interval(daily, baseline_daily, "rank_ic", protocol),
                "policies": {name: {"daily_mean": safe(daily[f"{name}_mean"].mean()),
                                    "cost_daily_mean": {str(cost): safe((daily[f"{name}_mean"] - daily[f"{name}_n"].gt(0) * cost / 10000).mean()) for cost in protocol["cost_roundtrip_bps"]},
                                    "paired_net_baseline": {str(cost): paired_interval(daily, baseline_daily, f"{name}_mean", protocol, cost_bps=cost) for cost in protocol["cost_roundtrip_bps"]}}
                             for name in ["top5", "top20"] + (["gate"] if kind == "fit" else [])}}
            if kind == "fit":
                errors = [report["blocks"][name][variant]["fit_error"] for name in names]
                baseline_errors = [report["blocks"][name]["baseline"]["fit_error"] for name in names]
                count = sum(error["scored_stock_days"] for error in errors)
                require(count == sum(error["scored_stock_days"] for error in baseline_errors), "pooled Fit MAE populations differ")
                mae = sum(error["absolute_error_sum"] for error in errors) / count
                baseline_mae = sum(error["absolute_error_sum"] for error in baseline_errors) / count
                report["pooled_outer"][f"{kind}/{variant}"]["fit_error"] = {
                    "stock_day_mae": mae, "scored_stock_days": count,
                    "mae_delta_vs_baseline": mae - baseline_mae, "units": "return_fraction",
                    "weighting": "Every scored stock-day equally; no selection/gate filter; lower is better."}
    for feature in candidates:
        evidence = [{"block": name, **records["all"]["feature_diagnostics"][feature]}
                    for name, records in report["blocks"].items() if "outer_" in name and "all" in records]
        report["features"][feature] = {"outer_all_model_evidence": evidence,
            "pruning_note": "Review with normal pruning only after complete outer evidence. Zero gain means unused in these fits; it does not by itself establish that this feature should be removed. No pruning changes made."}
    report["svm_source_audit"] = audit_svm_checkpoints(run_dir, manifest, run_id, panel)
    report["complete"] = not report["missing_results"] and not report["svm_source_audit"]["missing_outputs"]
    atomic_json(output / "result_audit.json", report)
    print(json.dumps({"report": str(output / "result_audit.json"), "complete": report["complete"],
                      "audited_models": sum(len(block) for block in report["blocks"].values()),
                      "missing_models": len(report["missing_results"]), "mrcy_eligible_rows": report["mrcy_eligible_rows"]}))
    if not report["complete"] and not args.allow_partial:
        raise SystemExit("Incomplete experiment: partial audit saved; rerun after all declared results finish.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path(__file__).resolve().parent / "model_run")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "result_audit")
    parser.add_argument("--allow-partial", action="store_true")
    audit(parser.parse_args())
