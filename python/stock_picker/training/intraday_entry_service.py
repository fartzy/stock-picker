"""Fast, bounded live inference for the separate Intraday Entry model.

Only saved morning candidates are eligible. The price is a recent timestamped
trade, and features consume completed minute bars as of that trade. Fetching
prices never trains a model, writes a morning scan, or places an order.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
import math
import logging
import threading
import time
from typing import Callable

import exchange_calendars as xcals
import pandas as pd

from stock_picker.features.intraday_entry import (
    MORNING_FEATURE_COLUMNS, build_intraday_entry_features, normalize_rank_score,
)
from stock_picker.ingestion.massive_overnight import MassiveOvernightClient, TICKER_PATTERN
from stock_picker.ingestion.session import SESSION_CLOSE_HOUR, US_EASTERN, cash_session_window
from stock_picker.storage.feature_store import FeatureStore
from stock_picker.storage.model_store import ModelStore
from stock_picker.storage.scan_store import ScanStore
from stock_picker.training.intraday_entry_model import MODEL_NAME, MODEL_VERSION

MAX_TICKERS = 20
MAX_WORKERS = 6
CACHE_SECONDS = 15
MAX_QUOTE_AGE_SECONDS = 120
LOG = logging.getLogger(__name__)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _empty(ticker: str, reason: str, context: dict | None = None) -> dict:
    context = context or {}
    return {"ticker": ticker, "status": "unavailable", "reason": reason,
            "rank": context.get("rank"), "rank_score": context.get("score"),
            "entry_price": None, "observed_at": None, "probability_profit": None,
            "expected_return": None, "probability_loss_2pct": None,
            "return_from_open": None, "minutes_remaining": None, "warnings": []}


def entry_time_reason(model, observed: datetime, window) -> str | None:
    """Do not extrapolate before/after the saved checkpoint and horizon support."""
    checkpoints = model.metadata.get("dataset", {}).get("checkpoints_et", [])
    if not checkpoints:
        return "The saved model has no verified intraday time window. Train the helper again."
    first, last = min(checkpoints), max(checkpoints)
    first_clock = datetime.strptime(first, "%H:%M").time()
    last_clock = datetime.strptime(last, "%H:%M").time()
    day = observed.astimezone(US_EASTERN).date()
    first_at = datetime.combine(day, first_clock, US_EASTERN)
    last_at = datetime.combine(day, last_clock, US_EASTERN)
    minimum_remaining = SESSION_CLOSE_HOUR*60 - (last_clock.hour*60 + last_clock.minute)
    last_at = min(last_at, window.close_at - timedelta(minutes=minimum_remaining))
    if observed < first_at or observed > last_at:
        return (f"Entry estimates support {first}–{last} ET and at least "
                f"{minimum_remaining} minutes before this session's close.")
    return None


class IntradayEntryService:
    def __init__(self, *, model_store=None, scan_store=None, feature_store=None,
                 client_factory=MassiveOvernightClient, clock: Callable = utc_now):
        self._models = model_store if model_store is not None else ModelStore()
        self._scans = scan_store if scan_store is not None else ScanStore()
        self._features = feature_store if feature_store is not None else FeatureStore()
        self._client_factory, self._clock = client_factory, clock
        self._lock = threading.Lock()
        self._loaded_key = None
        self._model = None
        self._market_cache: dict = {}
        self._open_cache: dict = {}
        self._prior_cache: dict = {}
        self._market_locks: dict = {}

    def _load(self):
        path: Path = self._models.data_dir / f"{MODEL_NAME}.pkl"
        if not path.is_file():
            return None
        stat = path.stat()
        key = (stat.st_mtime_ns, stat.st_size)
        with self._lock:
            if self._loaded_key != key:
                self._model = self._models.read(MODEL_NAME)
                self._loaded_key = key
            return self._model

    def _morning(self, session: str) -> dict[str, dict]:
        scan = self._scans.read(session, kind="rank") or {}
        signals = scan.get("signals", [])[:MAX_TICKERS]
        scores = [s.get("predicted_return") for s in signals]
        result = {}
        for rank, signal in enumerate(signals, 1):
            score = signal.get("predicted_return")
            score = float(score) if isinstance(score, (int, float)) and math.isfinite(score) else None
            result[signal["ticker"]] = {"rank": rank, "score": score,
                "score_z": normalize_rank_score(score, scores), "model_run_id": scan.get("model_run_id")}
        return result

    def summary(self) -> dict:
        day = self._clock().astimezone(US_EASTERN).date().isoformat()
        defaults = list(self._morning(day))
        model = self._load()
        if model is None:
            return {"ready": False, "model_name": MODEL_NAME, "version": MODEL_VERSION,
                    "trained_at": None, "trained_through": None, "evidence_status": "experimental",
                    "feature_count": 0, "features": [], "dataset": {}, "validation": None,
                    "notes": ["Train this separate helper using the saved historical minute data."],
                    "default_tickers": defaults}
        return {**model.metadata, "default_tickers": defaults}

    def _prior_features(self, ticker: str, session: str) -> dict:
        key = (ticker, session)
        with self._lock:
            if key in self._prior_cache:
                return self._prior_cache[key]
        values = {}
        prior = xcals.get_calendar("XNYS").previous_session(session).date()
        try:
            frame = self._features.read_columns(ticker, list(MORNING_FEATURE_COLUMNS))
            # Never silently substitute an older snapshot or today's developing row.
            rows = frame[pd.to_datetime(frame.index).date == prior]
            if not rows.empty:
                values = {name: float(rows.iloc[-1][name]) for name in MORNING_FEATURE_COLUMNS
                          if name in rows and pd.notna(rows.iloc[-1][name])}
        except Exception as error:
            LOG.warning("Prior entry context unavailable for %s (%s)", ticker, type(error).__name__)
            pass
        with self._lock:
            self._prior_cache[key] = values
        return values

    def _market(self, ticker: str, day):
        key = (ticker, day.isoformat())
        with self._lock:
            request_lock = self._market_locks.setdefault(key, threading.Lock())
        # Simultaneous refreshes share the fresh cache instead of racing provider
        # requests and replacing a newer quote with an older response.
        with request_lock:
            return self._fetch_market(ticker, day)

    def _fetch_market(self, ticker: str, day):
        key = (ticker, day.isoformat())
        with self._lock:
            cached = self._market_cache.get(key)
            if cached and time.monotonic() - cached[0] < CACHE_SECONDS:
                return cached[1]
            opened = self._open_cache.get(key)
        client = self._client_factory()
        # The universe uses Yahoo class-share spelling; Massive/Polygon uses
        # dots, as in ingestion/polygon_client.py's snapshot symbol mapping.
        provider_ticker = ticker.replace("-", ".")
        if opened is None:
            opened = client.fetch_current_open(provider_ticker, day)
            price, observed = opened.last_trade, opened.last_trade_at
            with self._lock:
                self._open_cache[key] = opened
        else:
            trade = client.fetch_current_trade(provider_ticker, day)
            price, observed = trade.price, trade.observed_at
        bars = client.fetch_minute_bars(provider_ticker, day)
        result = (opened, price, observed, bars)
        with self._lock:
            # A local app needs today's cache only; avoid accumulating an archive.
            self._market_cache = {k:v for k,v in self._market_cache.items() if k[1] == key[1]}
            self._open_cache = {k:v for k,v in self._open_cache.items() if k[1] == key[1]}
            self._prior_cache = {k:v for k,v in self._prior_cache.items() if k[1] == key[1]}
            self._market_locks = {k:v for k,v in self._market_locks.items() if k[1] == key[1]}
            self._market_cache[key] = (time.monotonic(), result)
        return result

    def _one(self, ticker: str, context: dict | None, day, model) -> dict:
        if context is None:
            return _empty(ticker, "No saved top-20 morning rank for this session.")
        try:
            opened, price, observed, bars = self._market(ticker, day)
            now = self._clock()
            window = cash_session_window(now)
            if window is None or not window.contains(now) or now.astimezone(US_EASTERN).date() != day:
                return _empty(ticker, "The regular trading session is closed.", context)
            if observed is None or price is None or not math.isfinite(price) or price <= 0:
                return _empty(ticker, "No timestamped current trade is available.", context)
            if not window.contains(observed):
                return _empty(ticker, "The last trade is outside today's regular session.", context)
            unsupported_time = entry_time_reason(model, observed, window)
            if unsupported_time:
                return _empty(ticker, unsupported_time, context)
            age = (now - observed).total_seconds()
            if age < 0 or age > MAX_QUOTE_AGE_SECONDS:
                return _empty(ticker, "The latest trade is stale or has an invalid timestamp; refresh again.", context)
            first_ms = round(window.open_at.timestamp()*1000)
            opening_bar = next((b for b in bars if b.get("t") == first_ms and
                                b["t"]+60_000 <= observed.timestamp()*1000), None)
            if opening_bar is None:
                return _empty(ticker, "The completed opening minute is not available yet.", context)
            if not opened.open or abs(opening_bar["o"]/opened.open - 1) > .01:
                return _empty(ticker, "Minute data and the verified session open disagree.", context)
            prior = self._prior_features(ticker, day.isoformat())
            previous_close = opened.previous_close if opened.corporate_action != "split" else None
            features = build_intraday_entry_features(bars, observed_at=observed,
                session_open_at=window.open_at, session_close_at=window.close_at,
                entry_price=price, session_open=opened.open, previous_close=previous_close,
                rank=context["rank"], rank_score=context["score"], rank_score_z=context["score_z"],
                morning_features=prior)
            outputs = model.predict(pd.DataFrame([features]))
            warnings = ["Experimental; historical and current morning models differ."]
            if context["rank"] != 1:
                warnings.append("Lower ranks have sparse historical validation coverage.")
            if len(prior) < len(MORNING_FEATURE_COLUMNS):
                warnings.append("Some prior-session context is unavailable.")
            if features["observed_bar_fraction"] < .95:
                warnings.append("Some completed minute bars are missing.")
            return {"ticker": ticker, "status": "ok", "reason": None,
                    "rank": context["rank"], "rank_score": context["score"],
                    "morning_model_run_id": context["model_run_id"],
                    "entry_price": float(price), "observed_at": observed.isoformat(),
                    **{name: float(value[0]) for name,value in outputs.items()},
                    "return_from_open": features["return_from_open"],
                    "minutes_remaining": features["minutes_to_close"], "warnings": warnings}
        except Exception as error:
            # Provider errors can contain credential-bearing URLs. Never expose them.
            LOG.warning("Intraday entry unavailable for %s (%s)", ticker, type(error).__name__)
            if isinstance(error, ValueError) and error.args == ("missing_last_completed_minute",):
                return _empty(ticker, "The latest completed minute is missing; refresh when the price feed catches up.", context)
            return _empty(ticker, "Current price or minute data could not be verified. Try refreshing.", context)

    def predict(self, tickers: list[str]) -> dict:
        started = time.monotonic()
        now = self._clock()
        day = now.astimezone(US_EASTERN).date()
        context = self._morning(day.isoformat())
        if len(tickers) > MAX_TICKERS:
            raise ValueError("Choose at most 20 tickers.")
        tickers = list(dict.fromkeys(t.strip().upper() for t in tickers)) or list(context)
        if any(not TICKER_PATTERN.fullmatch(t) for t in tickers):
            raise ValueError("Use valid ticker symbols.")
        model = self._load()
        window = cash_session_window(now)
        reason = None
        if model is None:
            reason = "Train the Intraday Entry model first."
        elif day.isoformat() <= model.metadata["trained_through"]:
            reason = "The model was trained after this session; historical live reassessment is unavailable."
        elif window is None or not window.contains(now):
            reason = "Live entry estimates are available during the regular trading session."
        else:
            reason = entry_time_reason(model, now, window)
        if reason:
            predictions = [_empty(t, reason, context.get(t)) for t in tickers]
        else:
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
                predictions = list(pool.map(lambda ticker: self._one(ticker, context.get(ticker), day, model), tickers))
        return {"session": day.isoformat(), "generated_at": self._clock().isoformat(),
                "elapsed_ms": round((time.monotonic()-started)*1000),
                "model_trained_through": model.metadata["trained_through"] if model else None,
                "evidence_status": "experimental", "predictions": predictions}


_service: IntradayEntryService | None = None
_service_lock = threading.Lock()


def get_service() -> IntradayEntryService:
    global _service
    with _service_lock:
        if _service is None:
            _service = IntradayEntryService()
        return _service
