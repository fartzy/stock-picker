"""Dedicated, atomic per-ticker store for verified overnight raw bars."""

from __future__ import annotations

import os
import re
import tempfile
from datetime import date, datetime
from math import isfinite
from numbers import Real
from pathlib import Path

import pandas as pd

from stock_picker.storage.paths import data_root

DEFAULT_DATA_DIR = data_root() / "overnight_prices"
BAR_COLUMNS = ("Open", "High", "Low", "Close", "Volume")
EVIDENCE_COLUMNS = (
    "source", "basis", "action_source", "corporate_action", "action_event_ids",
    "action_verified_from", "action_verified_to", "actions_fetched_at", "bars_fetched_at",
)
ALLOWED_ACTIONS = {"verified_none", "split", "dividend"}
TICKER_PATTERN = re.compile(r"[A-Z][A-Z0-9.-]*\Z")


def _calendar(calendar):
    if calendar is None:
        import exchange_calendars as xcals

        calendar = xcals.get_calendar("XNYS")
    if getattr(calendar, "name", None) != "XNYS":
        raise ValueError("overnight bars require an XNYS session calendar")
    return calendar


def _aware_timestamp(value: object) -> bool:
    if not isinstance(value, str):
        return False
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return parsed.tzinfo is not None and parsed.utcoffset() is not None


def _validate(ticker: str, history: pd.DataFrame, provenance: pd.DataFrame, calendar) -> None:
    if not TICKER_PATTERN.fullmatch(ticker):
        raise ValueError("invalid ticker")
    if history.empty or not set(BAR_COLUMNS).issubset(history.columns):
        raise ValueError("overnight history must have OHLCV bars")
    if not set(EVIDENCE_COLUMNS).issubset(provenance.columns):
        raise ValueError("overnight history must have complete per-bar provenance")
    index = history.index
    if (
        not isinstance(index, pd.DatetimeIndex) or index.hasnans or index.has_duplicates
        or index.tz is not None or not index.equals(index.normalize())
        or not index.is_monotonic_increasing or not index.equals(provenance.index)
    ):
        raise ValueError("bars and evidence must share ordered exchange-session dates")
    exchange = _calendar(calendar)
    for stamp in index:
        try:
            is_session = exchange.is_session(stamp.date().isoformat())
        except (LookupError, RuntimeError, TypeError, ValueError, OverflowError) as exc:
            raise ValueError("XNYS session calendar is unavailable") from exc
        if not is_session:
            raise ValueError("overnight bar date is not an XNYS session")
    for values in history.loc[:, list(BAR_COLUMNS)].itertuples(index=False, name=None):
        numeric_values = []
        for value in values:
            if isinstance(value, bool) or not isinstance(value, Real):
                raise ValueError("overnight OHLCV values must be finite numbers")
            try:
                number = float(value)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError("overnight OHLCV values must be finite numbers") from exc
            if not isfinite(number):
                raise ValueError("overnight OHLCV values must be finite numbers")
            numeric_values.append(number)
        opened, high, low, close, volume = numeric_values
        if min(opened, high, low, close) <= 0 or volume < 0 or low > min(opened, close) or high < max(opened, close):
            raise ValueError("overnight OHLCV values are inconsistent")
    if not provenance["source"].eq("massive").all() or not provenance["basis"].eq("raw").all():
        raise ValueError("overnight bars require Massive raw-basis evidence")
    if not provenance["action_source"].eq("massive_actions").all():
        raise ValueError("overnight bars require Massive action-feed evidence")
    if not provenance["corporate_action"].isin(ALLOWED_ACTIONS).all():
        raise ValueError("corporate action status is unverified")
    for stamp, row in provenance.iterrows():
        from_text = row["action_verified_from"]
        to_text = row["action_verified_to"]
        try:
            verified_from = date.fromisoformat(from_text)
            verified_to = date.fromisoformat(to_text)
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid action verification window") from exc
        if from_text != verified_from.isoformat() or to_text != verified_to.isoformat():
            raise ValueError("action verification window must use canonical YYYY-MM-DD dates")
        if not verified_from <= stamp.date() <= verified_to:
            raise ValueError("action verification does not cover a stored bar")
        if not _aware_timestamp(row["actions_fetched_at"]) or not _aware_timestamp(row["bars_fetched_at"]):
            raise ValueError("fetch times must be timezone-aware ISO timestamps")
        if not isinstance(row["action_event_ids"], str):
            raise ValueError("invalid action event evidence")
        ids = row["action_event_ids"].split("|") if row["action_event_ids"] else []
        kinds = []
        for item in ids:
            kind, separator, event_id = item.partition(":")
            if not separator or not event_id or kind not in {"split", "dividend"}:
                raise ValueError("invalid action event evidence")
            kinds.append(kind)
        status = row["corporate_action"]
        if (
            (status == "verified_none" and ids)
            or (status == "split" and "split" not in kinds)
            or (status == "dividend" and ("dividend" not in kinds or "split" in kinds))
        ):
            raise ValueError("corporate action status and event IDs disagree")


class OvernightPriceStore:
    """Persists bars and same-index evidence in one replaceable Parquet file."""

    def __init__(self, data_dir: Path | str = DEFAULT_DATA_DIR, *, calendar=None) -> None:
        self._data_dir = Path(data_dir)
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._calendar = calendar

    def _path_for(self, ticker: str) -> Path:
        if not TICKER_PATTERN.fullmatch(ticker):
            raise ValueError("invalid ticker")
        return self._data_dir / f"{ticker}.parquet"

    def write(self, ticker: str, history: pd.DataFrame, provenance: pd.DataFrame, *, overwrite: bool = False) -> None:
        _validate(ticker, history, provenance, self._calendar)
        combined = pd.concat([history.loc[:, list(BAR_COLUMNS)], provenance.loc[:, list(EVIDENCE_COLUMNS)]], axis=1)
        path = self._path_for(ticker)
        if path.exists():
            if not overwrite:
                raise FileExistsError("overnight ticker history exists; explicit overwrite required")
            previous_history, _ = self.read(ticker)
            if not previous_history.index.isin(history.index).all():
                raise ValueError("replacement would discard existing overnight sessions")
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(prefix=f".{ticker}.", suffix=".parquet", dir=self._data_dir, delete=False) as temporary:
                temporary_path = Path(temporary.name)
            combined.to_parquet(temporary_path)
            os.replace(temporary_path, path)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    def read(self, ticker: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        combined = pd.read_parquet(self._path_for(ticker))
        history = combined.loc[:, list(BAR_COLUMNS)]
        provenance = combined.loc[:, list(EVIDENCE_COLUMNS)]
        _validate(ticker, history, provenance, self._calendar)
        return history, provenance
