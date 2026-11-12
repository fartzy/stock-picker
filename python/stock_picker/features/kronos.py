"""Four open-known Kronos inputs, with one formula for historical and live rows."""

from __future__ import annotations

from datetime import datetime, timezone
import math

import exchange_calendars as xcals
import pandas as pd

from stock_picker.ingestion.kronos import normalized_history, request_for_history, retained_sessions
from stock_picker.storage.kronos_store import FEATURE_VERSION, PRICE_BASIS, KronosStore

KRONOS_COLUMNS = ("kronos_close", "kronos_high", "kronos_low", "kronos_range")
CONTRACT_ATTR = "kronos_contract"
KRONOS_DESCRIPTIONS = {
    "kronos_close": "Kronos forecast close relative to today's open.",
    "kronos_high": "Kronos forecast high relative to today's open.",
    "kronos_low": "Kronos forecast low relative to today's open.",
    "kronos_range": "Kronos forecast high–low range divided by today's open.",
}


def unknown_inputs() -> dict[str, float]:
    return dict.fromkeys(KRONOS_COLUMNS, float("nan"))


def target_has_known_split(history: pd.DataFrame | None, target: str) -> bool:
    """Read action metadata only; target prices and outcomes are irrelevant."""
    if history is None or "Stock Splits" not in history:
        return False
    dates = pd.DatetimeIndex(history.index)
    if dates.tz is not None:
        dates = dates.tz_convert("America/New_York").tz_localize(None)
    actions = pd.to_numeric(history.loc[dates.normalize() == pd.Timestamp(target), "Stock Splits"], errors="coerce")
    return bool((actions.notna() & actions.ne(0)).any())


def forecast_inputs(record: dict | None, today_open: float, compatible_prior_close: float,
                    *, price_basis: str = PRICE_BASIS) -> dict[str, float]:
    """Rebase forecast/context-close ratios into the target open's price units.

    The caller must supply a prior close compatible with the observed open. A
    cache produced before a uniform price rescaling never contributes absolute
    old-unit prices. Unknown or explicitly incompatible bases stay missing.
    """
    if record is None or record.get("status") != "ok" or record.get("price_basis") != price_basis:
        return unknown_inputs()
    try:
        context_close = float(record["last_context_close"])
        values = (today_open, compatible_prior_close, context_close)
        if not all(math.isfinite(value) and value > 0 for value in values):
            return unknown_inputs()
        raw = record["forecast"]
        close, high, low, opened = (float(raw[name]) for name in ("close", "high", "low", "open"))
        if not all(math.isfinite(value) and value > 0 for value in (close, high, low, opened)):
            return unknown_inputs()
        if high < max(close, opened, low) or low > min(close, opened):
            return unknown_inputs()
        scale = compatible_prior_close / context_close / today_open
        return dict(zip(KRONOS_COLUMNS, (close * scale - 1, high * scale - 1,
                                         low * scale - 1, (high - low) * scale)))
    except (KeyError, TypeError, ValueError, OverflowError):
        return unknown_inputs()


def frame_contract(frame: pd.DataFrame, columns) -> dict | None:
    """Fail closed if a consumer needs K inputs without their exact provenance."""
    if not set(columns).intersection(KRONOS_COLUMNS):
        return None
    contract = frame.attrs.get(CONTRACT_ATTR)
    if not isinstance(contract, dict) or set(contract) != {
        "execution_fingerprint", "manifest_hash", "feature_version", "price_basis"
    } or contract["feature_version"] != FEATURE_VERSION or contract["price_basis"] != PRICE_BASIS:
        raise ValueError("Kronos inputs require their saved execution contract")
    return dict(contract)


def validate_frame_contract(frame: pd.DataFrame, columns, expected: dict | None) -> None:
    if set(columns).intersection(KRONOS_COLUMNS):
        if expected is None or frame_contract(frame, columns) != expected:
            raise ValueError("Kronos feature contract does not match the saved model")


def historical_inputs(ticker: str, history: pd.DataFrame, store: KronosStore, contract: dict) -> pd.DataFrame:
    """Exact-target cache join; neither outcomes nor future bars enter the request."""
    result = pd.DataFrame(float("nan"), index=history.index, columns=KRONOS_COLUMNS)
    normalized = normalized_history(history)
    description = store.description(contract)
    records = store.records_for_ticker(contract, ticker)
    targets = {record["target_session"] for record in records.values()}
    if description is not None:
        for original in history.index:
            target = pd.Timestamp(original)
            if target.tzinfo is not None:
                target = target.tz_convert("America/New_York").tz_localize(None)
            target = target.normalize()
            target_str = target.date().isoformat()
            if target_str not in targets:
                continue
            try:
                request = request_for_history(ticker, target_str, normalized, description["manifest"], contract["execution_fingerprint"])
                # Unknown action conventions are never repaired. A known raw split
                # boundary needs an explicitly compatible source and remains missing.
                if target_has_known_split(normalized, target_str):
                    continue
                record = records.get(request["request_key"])
                result.loc[original] = forecast_inputs(record, float(normalized.loc[target, "Open"]),
                                                      request["bars"][-1]["close"])
            except (KeyError, ValueError, TypeError, OverflowError):
                continue
    result.attrs[CONTRACT_ATTR] = dict(contract)
    return result


def ready_contract(store: KronosStore, tickers: list[str], through: str) -> tuple[dict | None, dict]:
    contract = store.collection_contract()
    if contract is None:
        return None, {"ready": False, "reason": "no collection contract"}
    coverage = store.coverage(contract, tickers, retained_sessions(through))
    return (contract if coverage["ready"] else None), coverage


def live_inputs(ticker: str, target: str, today_open: float, quote_prior_close: float,
                prior_history: pd.DataFrame | None, store: KronosStore, contract: dict,
                *, now: datetime | None = None, allow_reconstructed: bool = False,
                known_target_split: bool = False) -> dict[str, float]:
    """Actual mornings require pre-open evidence; explicit counterfactuals may reconstruct."""
    if prior_history is None or known_target_split or target_has_known_split(prior_history, target):
        return unknown_inputs()
    description = store.description(contract)
    if description is None:
        return unknown_inputs()
    try:
        history = normalized_history(prior_history)
        request = request_for_history(ticker, target, history, description["manifest"], contract["execution_fingerprint"])
        record = store.get(contract, ticker, target, request["request_key"])
        if record is None or (record.get("reconstructed", True) and not allow_reconstructed):
            return unknown_inputs()
        generated = pd.Timestamp(record["generated_at"])
        opened = xcals.get_calendar("XNYS").session_open(target)
        if (generated.tzinfo is None or (generated >= opened and not allow_reconstructed)
                or generated > pd.Timestamp(now or datetime.now(timezone.utc))):
            return unknown_inputs()
        # A quote and stored prior bar on different adjustment bases cannot be
        # combined from one scalar ratio alone. Wait for refreshed source prices.
        prior_close = request["bars"][-1]["close"]
        if not math.isclose(prior_close, quote_prior_close, rel_tol=1e-5, abs_tol=1e-8):
            return unknown_inputs()
        return forecast_inputs(record, today_open, quote_prior_close)
    except (KeyError, ValueError, TypeError, OverflowError):
        return unknown_inputs()
