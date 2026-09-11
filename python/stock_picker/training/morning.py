"""Weekday morning scoring: today's opens through last night's model.

Runs at 8:32 CT (9:32 ET) so liquid names usually have an official open.
Writes JSON the Trading tab loads instantly -- sit down at 8:37, see the
list, click only to rescan live.
"""

from __future__ import annotations

import functools
import json
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from stock_picker.features.earnings import fetch_recent_earnings_tickers
from stock_picker.storage.paths import data_root
from stock_picker.training.buy_signal import DEFAULT_THRESHOLD, compute_buy_signals
from stock_picker.training.freshness import pipeline_freshness

print = functools.partial(print, flush=True)

DEFAULT_SIGNAL_DIR = data_root() / "buy_signals"
# 8:32 CT is 9:32 ET -- liquid names usually have an official open. One
# retry a minute later fills names that hadn't printed yet (NYSE auction lag).
MISSING_QUOTE_RETRY_SECONDS = 60
MISSING_QUOTE_RETRY_MIN = 20


def load_cached_signals(as_of: str | None = None, signal_dir: Path = DEFAULT_SIGNAL_DIR) -> dict | None:
    """Today's morning-job payload if it already finished -- a parquet-free
    JSON read so 8:37 is a click, not a 2-minute live scan."""
    path = signal_dir / (f"{as_of}.json" if as_of else "latest.json")
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _write_signals(payload: dict, as_of: str, signal_dir: Path = DEFAULT_SIGNAL_DIR) -> Path:
    signal_dir.mkdir(parents=True, exist_ok=True)
    path = signal_dir / f"{as_of}.json"
    path.write_text(json.dumps(payload, indent=2))
    (signal_dir / "latest.json").write_text(json.dumps(payload, indent=2))
    return path


def _payload_from(result, freshness) -> dict:
    return {
        "as_of": result.as_of,
        "threshold": result.threshold,
        "scored_count": result.scored_count,
        "signals": [
            {
                "ticker": signal.ticker,
                "predicted_return": signal.predicted_return,
                "open_price": signal.open_price,
                "snapshot_date": signal.snapshot_date,
            }
            for signal in result.signals
        ],
        "skipped": result.skipped,
        "top_drivers": [
            {"feature": name, "importance": value} for name, value in result.top_drivers
        ],
        "freshness": {
            "feature_snapshot_date": freshness.feature_snapshot_date,
            "model_trained_through": freshness.model_trained_through,
            "ready_for_inference": freshness.ready_for_inference,
            "detail": freshness.detail,
        },
    }


def _missing_quotes(skipped: list[dict]) -> int:
    return sum(1 for row in skipped if "no live quote" in (row.get("reason") or ""))


def run_morning(threshold: float = DEFAULT_THRESHOLD) -> int:
    started = datetime.now(ZoneInfo("America/Chicago"))
    print(f"morning start {started.isoformat()}")
    freshness = pipeline_freshness()
    print(freshness.detail)
    if not freshness.ready_for_inference:
        print("skipping score -- pipeline not current enough")
        return 1
    result = compute_buy_signals(
        threshold=threshold, earnings_fetcher=fetch_recent_earnings_tickers
    )
    missing = _missing_quotes(result.skipped)
    if missing >= MISSING_QUOTE_RETRY_MIN:
        print(f"{missing} names had no live quote -- retrying in {MISSING_QUOTE_RETRY_SECONDS}s")
        time.sleep(MISSING_QUOTE_RETRY_SECONDS)
        result = compute_buy_signals(
            threshold=threshold, earnings_fetcher=fetch_recent_earnings_tickers
        )
    payload = _payload_from(result, freshness)
    path = _write_signals(payload, result.as_of)
    print(f"scored={result.scored_count} picks={len(result.signals)} wrote {path}")
    print(f"morning done {datetime.now(ZoneInfo('America/Chicago')).isoformat()}")
    return 0


def main() -> None:
    raise SystemExit(run_morning())


if __name__ == "__main__":
    main()
