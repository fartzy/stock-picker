"""After-close pipeline: refresh prices, rebuild features, retrain.

Meant to run once on weekday evenings so the next morning's live score has
yesterday's completed session in the feature store and a model fit on it.
Does not recap the universe (no 2000 market-cap fetch). Does not search a
new ensemble -- retrains the current DEFAULT / UI-selected composition.
"""

from __future__ import annotations

import functools
import sys
import traceback
from datetime import datetime
from zoneinfo import ZoneInfo

from stock_picker.features.main import main as rebuild_features
from stock_picker.ingestion.refresh_prices import main as refresh_prices
from stock_picker.ingestion.session import last_completed_session_date
from stock_picker.training.main import run_training

print = functools.partial(print, flush=True)


def run_nightly() -> int:
    started = datetime.now(ZoneInfo("America/Chicago"))
    cutoff = last_completed_session_date()
    print(f"nightly start {started.isoformat()} last_completed_session={cutoff}")
    try:
        print("=== 1/4 refresh prices ===")
        refresh_prices()
        print("=== 2/4 fill missing sectors (capped Yahoo profile pull) ===")
        from stock_picker.ingestion.fundamentals import refresh_missing_sectors

        n_sectors = refresh_missing_sectors()
        print(f"wrote sectors for {n_sectors} tickers")
        print("=== 3/4 rebuild features ===")
        rebuild_features()
        print("=== 4/4 retrain ===")
        summary = run_training()
        print(f"holdout={summary.holdout_metrics}")
        if summary.threshold_sweep:
            row = next((r for r in summary.threshold_sweep if abs(r.get("threshold", -1) - 0.005) < 1e-9), None)
            if row:
                print(f"0.5% hit={row.get('hit_rate')} n={row.get('n_trades')}")
        print(f"nightly done {datetime.now(ZoneInfo('America/Chicago')).isoformat()}")
        return 0
    except Exception:
        traceback.print_exc()
        print(f"nightly failed {datetime.now(ZoneInfo('America/Chicago')).isoformat()}", file=sys.stderr)
        return 1


def main() -> None:
    raise SystemExit(run_nightly())


if __name__ == "__main__":
    main()
