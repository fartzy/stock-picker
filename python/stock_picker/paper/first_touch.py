"""Which level a session reached first, from Massive minute bars.

A daily bar whose high reaches the hold target and whose low reaches the stop
cannot say which happened first. Regular-session one-minute bars can, unless
both print inside the same minute. Minute bars for a completed session never
change, so each (ticker, session) is fetched once and cached on disk.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import date, datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

from stock_picker.storage.paths import data_root

DEFAULT_CACHE_DIR = data_root() / "prices" / "minute"
EASTERN = ZoneInfo("America/New_York")
REGULAR_OPEN = time(9, 30)
REGULAR_CLOSE = time(16, 0)

# "target" / "stop" = that level printed first; "same_minute" = both inside one
# bar; "neither" = no minute reached either; "unavailable" = no minute data.
FirstTouch = str


def regular_session_bars(results: list[dict]) -> list[dict]:
    """Minute bars inside 9:30-16:00 Eastern, in time order."""
    bars = []
    for bar in results:
        stamp = datetime.fromtimestamp(bar["t"] / 1000, EASTERN).time()
        if REGULAR_OPEN <= stamp < REGULAR_CLOSE:
            bars.append({"t": bar["t"], "h": float(bar["h"]), "l": float(bar["l"])})
    return sorted(bars, key=lambda bar: bar["t"])


def first_touch(bars: list[dict], target: float, stop: float) -> FirstTouch:
    if not bars:
        return "unavailable"
    for bar in bars:
        up, down = bar["h"] >= target, bar["l"] <= stop
        if up and down:
            return "same_minute"
        if up:
            return "target"
        if down:
            return "stop"
    return "neither"


class MinuteBarCache:
    """Regular-session minute bars per (ticker, session), fetched once."""

    def __init__(self, fetch: Callable[[str, date], list[dict]], cache_dir: Path | str | None = None) -> None:
        self._fetch = fetch
        self._dir = Path(cache_dir if cache_dir is not None else DEFAULT_CACHE_DIR)

    def _path(self, ticker: str, session: date) -> Path:
        return self._dir / ticker / f"{session.isoformat()}.json"

    def bars(self, ticker: str, session: date) -> list[dict]:
        path = self._path(ticker, session)
        if path.is_file():
            return json.loads(path.read_text())
        bars = regular_session_bars(self._fetch(ticker, session))
        if bars:
            # Empty answers are not cached: data may land later for a fresh session.
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(bars))
        return bars
