"""Audit preserved predictions and summarize the frozen selection's later test.

Run from the repository root with the local pandas/pyarrow interpreter.
This reads outcomes only after each model's selection has been written.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
LABEL = "label_day_session_return"


def audit_predictions() -> dict:
    panel = pd.read_parquet(ROOT / "panel.parquet", columns=["date", "ticker", LABEL])
    expected = panel.set_index(["date", "ticker"])[LABEL]
    checked_files, checked_rows = 0, 0
    for path in sorted(ROOT.glob("*/*/*.parquet")):
        report = json.loads(path.with_suffix(".json").read_text())
        assert pd.Timestamp(report["train_through"]) < pd.Timestamp(report["test_from"])
        scored = pd.read_parquet(path)
        assert not scored.duplicated(["date", "ticker"]).any(), path
        actual = scored.set_index(["date", "ticker"])[LABEL]
        wanted = expected.loc[(slice(pd.Timestamp(report["test_from"]),
                                     pd.Timestamp(report["test_through"])), slice(None))]
        pd.testing.assert_series_equal(actual.sort_index(), wanted.sort_index())
        assert np.isfinite(scored.prediction).all()
        daily = pd.read_json(path.with_suffix(".daily.json"))
        ranked = scored.sort_values(["ticker"]).sort_values("prediction", ascending=False, kind="stable")
        for count in (1, 5, 20):
            independently_selected = ranked.groupby("date", sort=True).head(count)
            mean = independently_selected.groupby("date")[LABEL].mean().sort_index()
            np.testing.assert_allclose(mean.to_numpy(), daily[f"top{count}"], atol=1e-10, rtol=0)
            np.testing.assert_allclose(mean.mean(), report[f"top{count}"], atol=1e-12, rtol=0)
        checked_files += 1
        checked_rows += len(scored)
    return {"prediction_files": checked_files, "prediction_rows": checked_rows,
            "labels_match_frozen_panel": True, "whole_date_coverage": True,
            "daily_returns_recomputed": True, "strictly_later_test_dates": True}


def paired_interval(difference: np.ndarray) -> list[float]:
    """Descriptive five-session circular block bootstrap, 2,000 replications."""
    rng = np.random.default_rng(20261111)
    means = []
    count = len(difference)
    for _ in range(2000):
        starts = rng.integers(0, count, size=(count + 4) // 5)
        indices = np.concatenate([(start + np.arange(5)) % count for start in starts])[:count]
        means.append(float(difference[indices].mean()))
    return [float(value) for value in np.quantile(means, [0.025, 0.975])]


def summarize() -> None:
    protocol = json.loads((ROOT / "protocol.json").read_text())
    summary = {"audit": audit_predictions(), "models": {},
               "limitations": "Retrospective, reused historical dates; 21 confirmation sessions; no production promotion.",
               "next_review": "After 21 new completed sessions, a source-data repair, or a model change."}
    review = []
    for model in ("rank", "fit_raw"):
        directory = ROOT / model
        selection = json.loads((directory / "selection.json").read_text())
        result = json.loads((directory / "confirmation.json").read_text())
        chosen = selection["chosen"]
        baseline, candidate = result["baseline"], result[chosen]
        before = pd.read_json(directory / "confirmation/baseline.daily.json")
        after = pd.read_json(directory / f"confirmation/{chosen}.daily.json")
        assert before.date.equals(after.date)
        metrics = {}
        for count in (1, 5, 20):
            key = f"top{count}"
            delta = (after[key] - before[key]).to_numpy()
            metrics[key] = {"baseline": baseline[key], "candidate": candidate[key],
                            "difference": candidate[key] - baseline[key],
                            "interval_95": paired_interval(delta)}
        summary["models"][model] = {
            "chosen": chosen, "added": sorted(set(candidate["columns"]) - set(baseline["columns"])),
            "confirmation_days": candidate["days"], "confirmation": metrics,
            "screen_variants": len(selection["folds"]),
            "screen": {name: {key: float(np.mean([row[key] for row in folds]))
                               for key in ("top1", "top5", "top20")}
                       for name, folds in selection["folds"].items()},
        }
        for record in protocol["prune_records"]:
            feature = record["feature"]
            individual_tests = [fold for name, folds in selection["folds"].items()
                                if name.startswith("restore_single") for fold in folds
                                if feature in fold["columns"]]
            status = "quality_blocked" if feature in protocol["quality_exclusions"] else (
                "screened" if feature in protocol["eligible_prunes"] else "unavailable")
            review.append({"model": model, "feature": feature, "reason": record["reason"],
                           "reviewed_through": protocol["screen"][-1][-1], "status": status,
                           "individual_fold_tests": len(individual_tests),
                           "included_in_selected_confirmation": feature in candidate["columns"]})
    (ROOT / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    (ROOT / "prune_review.json").write_text(json.dumps(review, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"audit": summary["audit"], "models": {
        name: {key: result[key] for key in ("chosen", "added", "confirmation_days", "confirmation")}
        for name, result in summary["models"].items()}}, indent=2))


if __name__ == "__main__":
    summarize()
