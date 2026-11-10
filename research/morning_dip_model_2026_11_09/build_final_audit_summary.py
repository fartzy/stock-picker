"""Condense a completed independent audit; descriptive attribution only."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import postprocess_results as audit
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def paired_top5_contributions(report: dict) -> dict:
    """Decompose the fixed all-minus-baseline result, without removing stocks."""
    run = HERE / "model_run"
    protocol = json.loads((run / "manifest.json").read_text())["protocol"]
    block = protocol["fit_outer_block"]
    frame = pd.read_parquet(HERE / "baseline.parquet", columns=audit.KEYS + [audit.TARGET],
                            filters=[("date", ">=", pd.Timestamp(block["from"])),
                                     ("date", "<=", pd.Timestamp(block["through"]))])
    frame = frame.sort_values(audit.KEYS).reset_index(drop=True)
    contribution = {}
    for variant in ["baseline", "all"]:
        key = run / "checkpoints" / f"fit_{block['name']}_{variant}"
        metadata = json.loads(key.with_suffix(".json").read_text())
        audit.require(metadata["run"] == report["run_identity"], "attribution run differs")
        audit.require(audit.sha(key.with_suffix(".npy")) == metadata["sha256"], "attribution score checksum mismatch")
        audit.require(audit.identity(frame) == metadata["context"]["test_rows"], "attribution row identity differs")
        scores = np.load(key.with_suffix(".npy"), allow_pickle=False)
        daily, selected = audit.score_selections(frame, scores, protocol["fit_gate"], True)
        rows = selected["top5"].copy()
        rows["contribution"] = rows[audit.TARGET] / rows.groupby("date", observed=True).ticker.transform("size") / len(daily)
        contribution[variant] = rows.groupby("ticker", observed=True).contribution.sum()
        expected = report["blocks"][f"fit_{block['name']}"][variant]["summary"]["top5"]["daily_mean"]
        np.testing.assert_allclose(contribution[variant].sum(), expected, atol=1e-12)
    table = pd.concat(contribution, axis=1).fillna(0.)
    table["delta"] = table["all"] - table.baseline
    table = table.reset_index().sort_values(["delta", "ticker"])
    expected_delta = report["blocks"][f"fit_{block['name']}"]["all"]["paired_net_baseline"]["top5_mean"]["0"]["mean"]
    np.testing.assert_allclose(table.delta.sum(), expected_delta, atol=1e-12)
    for name in ["baseline", "all", "delta"]:
        table[name + "_bps_per_day"] = table[name] * 10000
    columns = ["ticker", "baseline_bps_per_day", "all_bps_per_day", "delta_bps_per_day"]
    negative = table.loc[table.delta < 0].head(5)
    positive = table.loc[table.delta > 0].tail(5).iloc[::-1]
    return {"largest_negative_changes": negative[columns].to_dict("records"),
            "largest_positive_changes": positive[columns].to_dict("records"),
            "five_negative_changes_sum_bps_per_day": negative.delta.sum() * 10000,
            "net_all_minus_baseline_bps_per_day": expected_delta * 10000,
            "negative_five_sum_divided_by_net_delta": negative.delta.sum() / expected_delta if expected_delta else None,
            "tickers_with_negative_delta": int(table.delta.lt(0).sum()),
            "tickers_with_positive_delta": int(table.delta.gt(0).sum()),
            "union_of_selected_tickers": len(table),
            "interpretation": "Exact descriptive attribution of the fixed paired Top5 result, not a causal feature effect or a new stock-selection rule. The ratio to the net decline can exceed100% because positive changes offset negative changes. No stock or day is removed."}


def main():
    path = HERE / "result_audit/result_audit.json"
    report = json.loads(path.read_text())
    phase = json.loads((HERE / "phase_fit_status.json").read_text())
    audit.require(report["complete"] and phase["state"] == "completed" and phase["exit_code"] == 0, "Fit/audit not complete")
    svm = report["svm_source_audit"]
    audit.require(svm["audited_outputs"] == 42 and not svm["missing_outputs"], "SVM evidence incomplete")
    heads = sum(len(records) for records in report["blocks"].values())
    audit.require(heads == 47, "head evidence incomplete")
    fit = report["blocks"]["fit_outer_2026h2"]
    base, augmented = fit["baseline"], fit["all"]
    result = {
        "status": "complete_independently_verified", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "audited_model_results": heads, "audited_svm_outputs": svm["audited_outputs"],
        "run_identity": report["run_identity"], "source_report": str(path.relative_to(HERE)),
        "source_report_sha256": audit.sha(path), "summary_script_sha256": audit.sha(Path(__file__)),
        "fit_phase_exit_code": phase["exit_code"], "fit_phase_completed_at_utc": phase["completed_at_utc"],
        "mrcy_eligible_rows": report["mrcy_eligible_rows"],
        "frozen_inner_selection": report["frozen_selection"]["selected"],
        "interpretation": "The frozen all21 comparison does not establish better morning-to-close stock selection. Small Fit MAE improvement accompanies worse Top5/Top20/gate point estimates; exploratory paired return intervals span zero. It does not validate the separately pruned ordinary19 configuration.",
        "units": "Means below are percent per evaluation day; paired changes and confidence intervals are basis points per day. MAE is percentage-point error per equally weighted scored stock-day. All intervals are pointwise exploratory.",
        "fit": {"outer_from": "2026-07-01", "outer_through": "2026-11-06", "evaluation_days": base["summary"]["days"],
                "scored_stock_days_each": base["fit_error"]["scored_stock_days"],
                "oof_coverage": json.loads((HERE / "model_run/fit_oof_coverage.json").read_text()),
                "rank_ic": {"baseline": base["summary"]["rank_ic"], "all21": augmented["summary"]["rank_ic"]},
                "stock_day_mae_percentage_points": {"baseline": base["fit_error"]["stock_day_mae"] * 100,
                    "all21": augmented["fit_error"]["stock_day_mae"] * 100,
                    "all_minus_baseline": augmented["fit_error"]["mae_delta_vs_baseline"] * 100,
                    "lower_is_better": True, "includes_unselected_stocks": True}, "policies": {}},
        "rank_outer": {}, "svm_convergence": {key: svm[key] for key in ["convergence_warning_outputs", "svr_max_iter", "svc_max_iter", "convergence_qualification"]},
        "reviewed_context": report["reviewed_context"],
    }
    for policy in ["top5", "top20", "gate"]:
        b, a = base["summary"][policy], augmented["summary"][policy]
        paired = augmented["paired_net_baseline"][policy + "_mean"]
        result["fit"]["policies"][policy] = {
            "baseline_gross_daily_percent": b["daily_mean"] * 100, "all21_gross_daily_percent": a["daily_mean"] * 100,
            "cash_days": {"baseline": b["cash_days"], "all21": a["cash_days"]},
            "selected_stock_days": {"baseline": b["trades"], "all21": a["trades"]},
            "gate_rule": "score strictly greater than0.005" if policy == "gate" else "fixed top count each day",
            "cost_scenarios": {cost: {"baseline_daily_percent": b["cost_sensitivity"][cost] * 100,
                "all21_daily_percent": a["cost_sensitivity"][cost] * 100,
                "all_minus_baseline_bps_per_day": values["mean"] * 10000,
                "ci95_bps_per_day": [bound * 10000 for bound in values["ci95"]]}
                for cost, values in paired.items()},
            "mrcy_selected_days": {"baseline": base["policies"][policy]["mrcy_selected_days"], "all21": augmented["policies"][policy]["mrcy_selected_days"]},
            "constituent_excursions": {variant: record["policies"][policy]["intraday_excursion"] for variant, record in fit.items()}}
    for policy in ["top5", "top20"]:
        b = report["pooled_outer"]["rank/baseline"]["policies"][policy]
        a = report["pooled_outer"]["rank/all"]["policies"][policy]
        pair = a["paired_net_baseline"]["0"]
        result["rank_outer"][policy] = {"days": 342, "baseline_gross_daily_percent": b["daily_mean"] * 100,
            "all21_gross_daily_percent": a["daily_mean"] * 100, "all_minus_baseline_bps_per_day": pair["mean"] * 10000,
            "ci95_bps_per_day": [bound * 10000 for bound in pair["ci95"]]}
    result["fit_top5_paired_ticker_attribution"] = paired_top5_contributions(report)
    result["fit_top5_per_variant_concentration"] = {variant: record["policies"]["top5"]["return_concentration"] for variant, record in fit.items()}
    result["cautions"] = ["Six SVRs reached the unchanged iteration cap without establishing convergence;36 SVC outputs had no recorded ConvergenceWarning.",
        "Source qualifications include confirmed SPCX ticker reuse, unresolved QH price-basis discontinuity and confirmed ordinary tiny-volume errors; no retrospective row filtering or invented adjustments.",
        "Stock Low/Open aggregates are constituent excursions, not simultaneous portfolio drawdown or executable stop-loss returns.",
        "One Fit outer block versus three Rank outer blocks; current-universe survivorship and incomplete corporate-action certification remain.",
        "Frozen research all21 has153 raw/160 head inputs; ordinary19 has151 raw/158 head inputs. Ordinary configuration performance requires its own evidence."]
    destination = HERE / "final_independent_result_audit.json"
    audit.atomic_json(destination, result)
    print(destination)
    print(json.dumps(result["fit_top5_paired_ticker_attribution"], indent=2))


if __name__ == "__main__":
    main()
