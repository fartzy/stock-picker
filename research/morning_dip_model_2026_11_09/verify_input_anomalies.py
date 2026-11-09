"""Narrow read-only provider verification of suspicious feature inputs."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import sys
import urllib.request
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from zoneinfo import ZoneInfo

from review_input_tails import ROOT, HERE, pd
sys.path.insert(0, str(ROOT / "python"))
from stock_picker.ingestion.massive_overnight import API_ROOT, MassiveOvernightClient


def scrub(value):
    if isinstance(value, dict):
        return {key: scrub(item) for key, item in value.items() if key.lower() not in {"apikey", "api_key", "token"}}
    if isinstance(value, list):
        return [scrub(item) for item in value]
    if isinstance(value, str) and value.startswith("https://"):
        parsed = urlsplit(value)
        query = [(key, item) for key, item in parse_qsl(parsed.query) if key.lower() not in {"apikey", "api_key", "token"}]
        return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, urlencode(query), parsed.fragment))
    return value


def main():
    requests = [
        ("QH_actions", "/stocks/v1/splits", {"ticker": "QH", "execution_date.gte": "2026-04-01", "execution_date.lte": "2026-05-15", "limit": 1000}),
        ("QH_daily_raw", "/v2/aggs/ticker/QH/range/1/day/2026-04-20/2026-05-05", {"adjusted": "false", "sort": "asc", "limit": 1000}),
        ("QH_daily_adjusted", "/v2/aggs/ticker/QH/range/1/day/2026-04-20/2026-05-05", {"adjusted": "true", "sort": "asc", "limit": 1000}),
        ("HUBB_daily", "/v2/aggs/ticker/HUBB/range/1/day/2026-02-23/2026-02-27", {"adjusted": "false", "sort": "asc", "limit": 1000}),
        ("UHAL_B_daily", "/v2/aggs/ticker/UHAL.B/range/1/day/2026-02-23/2026-02-27", {"adjusted": "false", "sort": "asc", "limit": 1000}),
        ("SPCX_reference_old", "/v3/reference/tickers/SPCX", {"date": "2026-02-09"}),
        ("SPCX_reference_new", "/v3/reference/tickers/SPCX", {"date": "2026-06-15"}),
        ("QH_reference", "/v3/reference/tickers/QH", {"date": "2026-04-27"}),
        ("QH_inactive_lookup", "/v3/reference/tickers", {"search": "Quhuo", "active": "false", "limit": 20}),
    ]
    client = MassiveOvernightClient()
    records = []
    for name, path, parameters in requests:
        item = {"name": name, "url": API_ROOT + path, "parameters": parameters}
        try:
            payload = scrub(client._get_payload(API_ROOT + path, parameters))
            item["payload"] = payload
            print(name, payload.get("status"), "results", len(payload.get("results", [])), flush=True)
        except Exception as exc:
            # This client's errors intentionally omit authenticated request URLs.
            item["error"] = f"{type(exc).__name__}: {exc}"
            print(name, item["error"], flush=True)
        records.append(item)
    yahoo_url = "https://query1.finance.yahoo.com/v8/finance/chart/QH"
    yahoo_parameters = {
        "period1": int(datetime(2026, 4, 1, tzinfo=ZoneInfo("America/New_York")).timestamp()),
        "period2": int(datetime(2026, 5, 16, tzinfo=ZoneInfo("America/New_York")).timestamp()),
        "interval": "1d", "events": "splits",
    }
    yahoo = {"name": "QH_yahoo_chart_actions", "url": yahoo_url, "parameters": yahoo_parameters}
    try:
        request = urllib.request.Request(yahoo_url + "?" + urlencode(yahoo_parameters), headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(request, timeout=20) as response:
            yahoo["payload"] = json.load(response)
    except Exception as exc:
        yahoo["error"] = f"{type(exc).__name__}: {exc}"
    records.append(yahoo)
    local = {}
    for ticker, start, end in [("QH", "2026-04-20", "2026-05-05"), ("HUBB", "2026-02-23", "2026-02-27"), ("UHAL-B", "2026-02-23", "2026-02-27")]:
        frame = pd.read_parquet(ROOT / "data/prices" / f"{ticker}.parquet").loc[start:end]
        local[ticker] = [{"date": str(when.date()), **{str(key): float(value) for key, value in row.items()}} for when, row in frame.iterrows()]
    result = {"checked_at_utc": datetime.now(timezone.utc).isoformat(), "scope": "Nine Massive and one Yahoo narrow read-only requests plus existing local price rows. No price, feature, model, or configuration changes.", "issuer_lookup": {"url": "https://ir.quhuo.cn/news-releases", "result": "HTTP403; issuer page unavailable. Provider action metadata also inconclusive; no clean price-basis certification."}, "requests": records, "ordinary_price_rows": local}
    (HERE / "input_anomaly_verification.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
