"""NYC daily weather from Open-Meteo archive. No API key.

Calendar yesterday only -- known at the cash open. Not a 8:31 forecast.
HQ-city weather waits until UniverseStore has cities; this is one series
broadcast like VIX.
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import requests

from stock_picker.log import get_logger
from stock_picker.storage.paths import data_root

logger = get_logger(__name__)

NYC_LAT = 40.7829
NYC_LON = -73.9654
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
HDD_BASE_C = 18.0
WEATHER_START = date(2018, 1, 1)


def weather_path(data_dir=None):
    root = data_dir or data_root()
    path = root / "weather"
    path.mkdir(parents=True, exist_ok=True)
    return path / "nyc.parquet"


def fetch_nyc_daily(start: date, end: date) -> pd.DataFrame:
    params = {
        "latitude": NYC_LAT,
        "longitude": NYC_LON,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,snowfall_sum",
        "timezone": "America/New_York",
    }
    payload = requests.get(ARCHIVE_URL, params=params, timeout=60).json()
    daily = payload.get("daily") or {}
    days = daily.get("time") or []
    if not days:
        return pd.DataFrame()
    tmax = pd.to_numeric(pd.Series(daily.get("temperature_2m_max")), errors="coerce")
    tmin = pd.to_numeric(pd.Series(daily.get("temperature_2m_min")), errors="coerce")
    precip = pd.to_numeric(pd.Series(daily.get("precipitation_sum")), errors="coerce")
    snow = pd.to_numeric(pd.Series(daily.get("snowfall_sum")), errors="coerce")
    tmean = (tmax + tmin) / 2.0
    frame = pd.DataFrame(
        {
            "tmax_c": tmax.to_numpy(),
            "tmin_c": tmin.to_numpy(),
            "precip_mm": precip.to_numpy(),
            "snow_cm": snow.to_numpy(),
            "hdd": (HDD_BASE_C - tmean).clip(lower=0).to_numpy(),
            "cdd": (tmean - HDD_BASE_C).clip(lower=0).to_numpy(),
        },
        index=pd.to_datetime(days),
    )
    frame.index.name = "date"
    return frame


def refresh_nyc_weather(data_dir=None, as_of: date | None = None) -> int:
    """Download through yesterday (UTC date in Chicago is fine for as_of)."""
    end = (as_of or date.today()) - timedelta(days=1)
    if end < WEATHER_START:
        return 0
    path = weather_path(data_dir)
    existing = pd.read_parquet(path) if path.exists() else pd.DataFrame()
    start = WEATHER_START
    if not existing.empty:
        last = pd.Timestamp(existing.index.max()).date()
        start = last + timedelta(days=1)
        if start > end:
            return 0
    fresh = fetch_nyc_daily(start, end)
    if fresh.empty:
        logger.warning("Open-Meteo returned no NYC rows %s..%s", start, end)
        return 0
    if existing.empty:
        combined = fresh
    else:
        combined = pd.concat([existing, fresh])
        combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    combined.to_parquet(path)
    logger.info("nyc weather wrote %s rows through %s", len(combined), combined.index[-1].date())
    return len(fresh)


def read_nyc_weather(data_dir=None) -> pd.DataFrame:
    path = weather_path(data_dir)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def main() -> None:
    n = refresh_nyc_weather()
    logger.info("fetched %s new NYC weather days", n)


if __name__ == "__main__":
    main()
