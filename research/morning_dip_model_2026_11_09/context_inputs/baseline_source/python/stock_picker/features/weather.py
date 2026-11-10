"""Yesterday's NYC weather, known at the cash open.

Broadcast onto every ticker like VIX. Values on date T are calendar T-1
(Open-Meteo daily). Training leaves these unshifted. Live scoring overwrites
from the weather store for calendar yesterday.
"""

from __future__ import annotations

import pandas as pd

WEATHER_COLUMNS = (
    "weather_nyc_tmax_yday",
    "weather_nyc_tmin_yday",
    "weather_nyc_precip_yday",
    "weather_nyc_snow_yday",
    "weather_nyc_hdd_yday",
    "weather_nyc_cdd_yday",
)
_SOURCE = {
    "weather_nyc_tmax_yday": "tmax_c",
    "weather_nyc_tmin_yday": "tmin_c",
    "weather_nyc_precip_yday": "precip_mm",
    "weather_nyc_snow_yday": "snow_cm",
    "weather_nyc_hdd_yday": "hdd",
    "weather_nyc_cdd_yday": "cdd",
}


def yesterday_weather_row(nyc: pd.DataFrame, as_of) -> dict[str, float]:
    """Calendar day before `as_of`."""
    if nyc is None or nyc.empty:
        return {}
    when = pd.Timestamp(as_of).normalize() - pd.Timedelta(days=1)
    index = pd.DatetimeIndex(nyc.index).normalize()
    hit = nyc.loc[index == when]
    if hit.empty:
        prior = nyc.loc[index < when]
        if prior.empty:
            return {}
        hit = prior.iloc[[-1]]
    row = hit.iloc[-1]
    out = {}
    for column, source in _SOURCE.items():
        value = row.get(source)
        if value is not None and pd.notna(value):
            out[column] = float(value)
    return out


def build_weather_features(index: pd.Index, nyc: pd.DataFrame | None = None) -> pd.DataFrame:
    if nyc is None or nyc.empty or index.empty:
        return pd.DataFrame(index=index)
    rows = [yesterday_weather_row(nyc, when) for when in index]
    frame = pd.DataFrame(rows, index=index)
    return frame.reindex(columns=list(WEATHER_COLUMNS))
