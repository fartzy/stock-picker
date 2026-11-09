"""Input-only diagnostics for the frozen dip representations; never fit a model.

Uses the same hermetic package bootstrap as postprocess_results.py. Quantiles
are descriptive, not candidate trading thresholds or clipping decisions.
"""
from __future__ import annotations

from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RUNFILES = (ROOT / "bazel-bin/python/stock_picker/training/morning_dip_experiment.runfiles").resolve()
sys.path[:0] = list(map(str, RUNFILES.glob("rules_python++pip+*/site-packages")))
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np
import pandas as pd

RATIOS = ("dip_recovery_fraction", "dip_loss_2d_sigma", "dip_loss_change_sigma")
QUANTILES = (0, .0001, .001, .01, .1, .5, .9, .99, .999, .9999, 1)


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def describe(series):
    values = series.to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    result = {
        "rows": len(values), "finite_rows": len(finite),
        "nan_rows": int(np.isnan(values).sum()), "infinite_rows": int(np.isinf(values).sum()),
        "quantiles": {str(q): float(value) for q, value in zip(QUANTILES, np.quantile(finite, QUANTILES))},
        "absolute_count_above": {str(limit): int((np.abs(finite) > limit).sum()) for limit in (10, 100, 1000, 10000)},
    }
    # Exact population-moment analogue of median imputation + StandardScaler.
    # This reports the transformation's geometry without fitting any estimator.
    imputed = np.where(np.isnan(values), np.median(finite), values)
    centered = imputed - np.mean(imputed)
    std = float(np.std(imputed))
    squared = centered * centered
    total = float(squared.sum())
    greatest = np.sort(squared)[::-1]
    z = centered / std if std > 0 else centered
    result["median_imputed_population_geometry"] = {
        "mean": float(np.mean(imputed)), "std": std,
        "median_abs_z": float(np.median(np.abs(z))),
        "observed_only_median_abs_z": float(np.median(np.abs(z[np.isfinite(values)]))),
        "p99_abs_z": float(np.quantile(np.abs(z), .99)),
        "max_abs_z": float(np.max(np.abs(z))),
        "top_n_fraction_of_centered_sum_squares": {
            str(n): float(greatest[:n].sum() / total) if total else 0.0 for n in (1, 10, 100, 1000)
        },
    }
    return result


def source_components(ticker, when):
    path = ROOT / "research/averaging_down_2026_11_06/source_data/daily_five_year" / f"{ticker}.json.gz"
    bars = json.loads(gzip.decompress(path.read_bytes()))["bars"]
    history = pd.DataFrame(
        {name: [bar[key] for bar in bars] for name, key in zip(("Open", "High", "Low", "Close", "Volume"), ("o", "h", "l", "c", "v"))},
        index=pd.to_datetime([datetime.fromtimestamp(bar["t"] / 1000, timezone.utc).date() for bar in bars]),
    )
    spy_path = ROOT / "research/averaging_down_2026_11_06/source_data/daily_five_year/SPY.json.gz"
    spy_bars = json.loads(gzip.decompress(spy_path.read_bytes()))["bars"]
    calendar = pd.to_datetime([datetime.fromtimestamp(bar["t"] / 1000, timezone.utc).date() for bar in spy_bars])
    history = history.reindex(calendar)
    i = history.index.get_loc(when)
    close = history.Close
    returns = close.pct_change(fill_method=None)
    c3, c1, opening = float(close.iloc[i-3]), float(close.iloc[i-1]), float(history.Open.iloc[i])
    loss = c3 - c1
    volatility = float(returns.iloc[i-22:i-2].std(ddof=1))
    return {
        "source_file": str(path.relative_to(ROOT)), "source_sha256": digest(path),
        "open_t": opening, "close_t_minus_1": c1, "close_t_minus_3": c3,
        "closing_loss_dollars": loss,
        "closing_loss_fraction": loss / c3,
        "open_minus_prior_close_dollars": opening - c1,
        "pre_shock_volatility_20d": volatility,
        "prior_return": float(returns.iloc[i-1]), "earlier_return": float(returns.iloc[i-2]),
        "reconstructed_recovery_fraction": (opening-c1)/loss if loss > 0 else None,
        "reconstructed_loss_2d_sigma": (c1/c3-1)/(volatility*np.sqrt(2)) if volatility > 0 else None,
        "reconstructed_loss_change_sigma": (abs(returns.iloc[i-1])-abs(returns.iloc[i-2]))/volatility if volatility > 0 else None,
    }


def main():
    manifest = json.loads((HERE / "dip_panel_manifest.json").read_text())
    path = HERE / "dip_panel.parquet"
    actual_hash = digest(path)
    if actual_hash != manifest["sha256"]:
        raise ValueError("Frozen dip panel hash changed")
    frame = pd.read_parquet(path)
    protocol = json.loads((HERE / "experiment_protocol.json").read_text())
    result = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Frozen feature inputs only. No target returns or future High/Low outcomes read; no model fits, clipping, pruning or threshold changes.",
        "panel_sha256": actual_hash, "rows": len(frame),
        "all_column_finiteness": {name: {"nan": int(frame[name].isna().sum()), "infinite": int(np.isinf(frame[name]).sum())} for name in manifest["columns"]},
        "full_panel_ratios": {name: describe(frame[name]) for name in RATIOS},
        "svm_training_windows": {}, "largest_absolute_ratio_rows": {},
    }
    dates = np.sort(frame.date.unique())
    for block in [*protocol["fit_oof_blocks"], protocol["fit_outer_block"]]:
        earlier = dates[dates < np.datetime64(block["from"])][-protocol["training_sessions"]:]
        rows = frame.loc[frame.date.isin(earlier)]
        result["svm_training_windows"][block["name"]] = {
            "from": str(pd.Timestamp(earlier[0]).date()), "through": str(pd.Timestamp(earlier[-1]).date()),
            "rows": len(rows), "ratios": {name: describe(rows[name]) for name in RATIOS},
        }
    for name in RATIOS:
        rows = frame.loc[frame[name].abs().nlargest(5).index, ["date", "ticker", name]]
        examples = []
        for row in rows.itertuples(index=False, name=None):
            when, ticker, value = row
            components = source_components(ticker, when)
            reconstructed = components["reconstructed_" + name.removeprefix("dip_")]
            if reconstructed is None or not np.isclose(reconstructed, value, rtol=1e-9, atol=1e-9):
                raise ValueError(f"Raw ratio reconstruction differs for {name}/{ticker}/{when}: {reconstructed} vs {value}")
            examples.append({"date": str(when.date()), "ticker": ticker, "value": float(value), "raw_components": components})
        result["largest_absolute_ratio_rows"][name] = examples
    destination = HERE / "input_representation_review.json"
    destination.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"artifact": str(destination), "rows": len(frame), "all_column_infinite_count": sum(x["infinite"] for x in result["all_column_finiteness"].values()), "ratio_ranges": {name: [item["quantiles"]["0"], item["quantiles"]["1"]] for name, item in result["full_panel_ratios"].items()}}, indent=2))


if __name__ == "__main__":
    main()
