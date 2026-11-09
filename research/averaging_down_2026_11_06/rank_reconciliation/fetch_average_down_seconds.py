"""Refine only the nine ambiguous trigger minutes; never serialize credentials."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path

from stock_picker.ingestion.massive_overnight import MassiveOvernightClient, MassiveOvernightError

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "second_bars"
OUT.mkdir(exist_ok=True)
source = ROOT / "rank1_average_down_study.json"
original = ROOT / "rank1_average_down_minute_study.json"
if not original.exists():
    original.write_bytes(source.read_bytes())
study = json.loads(original.read_text())
cases = sorted({(case["ticker"], case["lower"]["trigger_at"]) for case in study["ambiguous"]})


def fetch(case):
    ticker, when = case
    start = int(datetime.fromisoformat(when).timestamp()) * 1000
    path = OUT / f"{ticker}_{start}.json.gz"
    if path.exists():
        with gzip.open(path, "rt") as handle:
            doc = json.load(handle)
        if doc["bars"]:
            return {"ticker": ticker, "when": when, "bars": len(doc["bars"]), "cached": True}
    endpoint = f"/v2/aggs/ticker/{ticker}/range/1/second/{start}/{start + 59999}"
    try:
        bars = []
        for page in MassiveOvernightClient()._pages(endpoint, {"adjusted": "false", "sort": "asc", "limit": 5000}):
            bars.extend(page.get("results") or [])
        assert all(start <= bar["t"] < start + 60000 for bar in bars)
        doc = {"ticker": ticker, "minute_start": start, "adjusted": False, "source": "Massive",
               "endpoint": endpoint, "retrieved_at": datetime.now(timezone.utc).isoformat(), "bars": bars}
        with gzip.open(path, "wt") as handle:
            json.dump(doc, handle)
        return {"ticker": ticker, "when": when, "bars": len(bars), "cached": False}
    except MassiveOvernightError as error:
        return {"ticker": ticker, "when": when, "error": str(error)}


with ThreadPoolExecutor(max_workers=4) as pool:
    results = list(pool.map(fetch, cases))
(ROOT / "average_down_second_fetch_manifest.json").write_text(json.dumps(results, indent=2) + "\n")
print(json.dumps(results, indent=2))
