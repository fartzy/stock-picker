"""Leak-safe close-conditioned next-open rows and hypothetical-close features.

This module deliberately does not read the morning feature parquet. Existing OHLCV
parquets carry no per-bar source/basis or corporate-action verification, so callers
must supply a separately verified provenance frame before any row can be labeled.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
from math import isfinite
from statistics import pstdev
from typing import Literal, Sequence

import exchange_calendars as xcals
import pandas as pd


LABEL_COLUMN = "label_next_open_gap"
FEATURE_VERSION = "overnight_close_scenario_v1"
FEATURE_COLUMNS = (
    "prior_close",
    "prior_return_1d",
    "prior_volatility_5d",
    "today_open",
    "today_open_gap",
    "assumed_close",
    "assumed_day_return",
    "assumed_vs_prior_close",
    "weekday",
)
PRIOR_CLOSE_COUNT = 6  # Six completed closes yield five strictly prior close returns.
PROVENANCE_COLUMNS = ("source", "basis", "action_source", "corporate_action")


@dataclass(frozen=True)
class PriceContract:
    """One price basis/source plus an independently verified action feed.

    Raw open/close prices are comparable on ordinary days, but a split or
    ex-dividend date is excluded rather than modeled as a spurious overnight
    gap. `action_source` asserts that each bar's `corporate_action` status was
    checked against that feed; a bare OHLCV file cannot make this assertion.
    """

    source: str
    basis: Literal["raw"]
    action_source: str

    def __post_init__(self) -> None:
        if not self.source or self.basis != "raw" or not self.action_source:
            raise ValueError("A named raw-price source and corporate-action source are required")

    def artifact_metadata(self) -> dict[str, object]:
        """Persist unchanged with the later standalone overnight model."""
        return {
            **asdict(self),
            "calendar": "XNYS",
            "exchange_calendars_version": xcals.__version__,
            "feature_version": FEATURE_VERSION,
            "feature_columns": FEATURE_COLUMNS,
        }


@dataclass(frozen=True)
class CurrentOpenProvenance:
    """Evidence for the observed current-session open, supplied by its feed."""

    source: str | None
    basis: str | None
    action_source: str | None
    corporate_action: str | None


class NonfiniteDerivedValue(ValueError):
    """A valid positive input produced an unusable ratio, volatility, or label."""


@dataclass(frozen=True)
class ExcludedRow:
    session: date
    reason: str


@dataclass(frozen=True)
class DatasetBuild:
    frame: pd.DataFrame
    exclusions: tuple[ExcludedRow, ...]
    contract: PriceContract

    @property
    def exclusion_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for excluded in self.exclusions:
            counts[excluded.reason] = counts.get(excluded.reason, 0) + 1
        return counts


@dataclass(frozen=True)
class ScenarioBuild:
    features: pd.Series | None
    next_session: date | None
    exclusion_reason: str | None


def _calendar(calendar=None):
    # Official API: https://github.com/gerrymanoim/exchange_calendars/blob/master/README.md
    exchange = calendar if calendar is not None else xcals.get_calendar("XNYS")
    if getattr(exchange, "name", None) != "XNYS":
        raise ValueError("Overnight labels require the XNYS exchange calendar")
    return exchange


def next_expected_session(session: date, calendar=None) -> date | None:
    """Return an exchange session or unknown; never substitute weekday arithmetic."""
    try:
        exchange = _calendar(calendar)
        if not exchange.is_session(session.isoformat()):
            return None
        return pd.Timestamp(exchange.next_session(session.isoformat())).date()
    except (LookupError, RuntimeError, TypeError, ValueError, OverflowError):
        return None


def _adjacency_reason(sessions: Sequence[date], calendar) -> str | None:
    for current, following in zip(sessions, sessions[1:]):
        try:
            if not calendar.is_session(current.isoformat()):
                return "non_exchange_session"
        except (LookupError, RuntimeError, TypeError, ValueError, OverflowError):
            return "calendar_unavailable"
        expected = next_expected_session(current, calendar)
        if expected is None:
            return "calendar_unavailable"
        if expected != following:
            return "missing_exchange_session"
    try:
        if not calendar.is_session(sessions[-1].isoformat()):
            return "non_exchange_session"
    except (LookupError, RuntimeError, TypeError, ValueError, OverflowError):
        return "calendar_unavailable"
    return None


def _provenance_reason(provenance: pd.DataFrame | None, sessions: pd.DatetimeIndex, contract: PriceContract) -> str | None:
    if provenance is None or not set(PROVENANCE_COLUMNS).issubset(provenance.columns):
        return "missing_provenance"
    if provenance.index.has_duplicates or not sessions.isin(provenance.index).all():
        return "missing_provenance"
    rows = provenance.loc[sessions]
    if rows["source"].isna().any() or rows["action_source"].isna().any():
        return "missing_provenance"
    if rows["basis"].isna().any():
        return "unknown_basis"
    if not rows["source"].eq(contract.source).all():
        return "mixed_source"
    if not rows["basis"].eq(contract.basis).all():
        return "mixed_basis"
    if not rows["action_source"].eq(contract.action_source).all():
        return "unverified_corporate_actions"
    actions = set(rows["corporate_action"].dropna())
    if "split" in actions:
        return "split_event"
    if "dividend" in actions:
        return "dividend_event"
    if rows["corporate_action"].isna().any() or actions != {"verified_none"}:
        return "unverified_corporate_actions"
    return None


def _positive_price(value: object) -> float | None:
    try:
        price = float(value)
    except (TypeError, ValueError):
        return None
    return price if isfinite(price) and price > 0 else None


def _current_open_reason(
    provenance: CurrentOpenProvenance | None, contract: PriceContract
) -> str | None:
    if provenance is None or not provenance.source:
        return "missing_current_open_provenance"
    if not provenance.basis:
        return "unknown_open_basis"
    if provenance.source != contract.source:
        return "mixed_open_source"
    if provenance.basis != contract.basis:
        return "mixed_open_basis"
    if provenance.action_source != contract.action_source:
        return "unverified_corporate_actions"
    if provenance.corporate_action == "split":
        return "split_event"
    if provenance.corporate_action == "dividend":
        return "dividend_event"
    if provenance.corporate_action != "verified_none":
        return "unverified_corporate_actions"
    return None


def build_feature_row(
    prior_closes: Sequence[float], today_open: float, assumed_close: float, session: date
) -> pd.Series:
    """The sole train/scenario transform; no final same-day OHLCV is accepted.

    The caller must provide exactly six *completed* closes ending at t-1.
    Today's observed open and an explicit assumed close are the only t inputs.
    """
    if len(prior_closes) != PRIOR_CLOSE_COUNT:
        raise ValueError("Exactly six prior completed closes are required")
    closes = [_positive_price(price) for price in prior_closes]
    opened = _positive_price(today_open)
    assumed = _positive_price(assumed_close)
    if any(price is None for price in closes) or opened is None or assumed is None:
        raise ValueError("All prices must be finite and positive")
    closes = [float(price) for price in closes]
    prior_returns = [right / left - 1 for left, right in zip(closes, closes[1:])]
    if not all(isfinite(value) for value in prior_returns):
        raise NonfiniteDerivedValue("A prior return is nonfinite")
    try:
        prior_volatility = pstdev(prior_returns)
    except OverflowError as exc:
        raise NonfiniteDerivedValue("Prior volatility overflowed") from exc
    values = (
        closes[-1],
        prior_returns[-1],
        prior_volatility,
        opened,
        opened / closes[-1] - 1,
        assumed,
        assumed / opened - 1,
        assumed / closes[-1] - 1,
        float(session.weekday()),
    )
    if not all(isfinite(value) for value in values):
        raise NonfiniteDerivedValue("A scenario feature is nonfinite")
    return pd.Series(values, index=FEATURE_COLUMNS, dtype=float)


def build_overnight_training_frame(
    history: pd.DataFrame,
    provenance: pd.DataFrame | None,
    contract: PriceContract,
    *,
    calendar=None,
) -> DatasetBuild:
    """Pair t with the next observed bar only if it is the next XNYS session.

    Every excluded candidate is reported. The last bar is *always* unlabeled,
    even if a calendar says what the next session will be.
    """
    if not {"Open", "Close"}.issubset(history.columns):
        raise ValueError("History must contain raw Open and Close columns")
    frame = pd.DataFrame(columns=[*FEATURE_COLUMNS, LABEL_COLUMN], dtype=float)
    if history.empty:
        return DatasetBuild(frame, (), contract)
    bars = history.sort_index()
    if not isinstance(bars.index, pd.DatetimeIndex) or bars.index.hasnans:
        raise ValueError("History must have a valid DatetimeIndex")
    exclusions: list[ExcludedRow] = []
    if bars.index.tz is not None or not bars.index.equals(bars.index.normalize()):
        exclusions.extend(ExcludedRow(stamp.date(), "ambiguous_session_index") for stamp in bars.index)
        return DatasetBuild(frame, tuple(exclusions), contract)
    if bars.index.normalize().has_duplicates:
        exclusions.extend(ExcludedRow(stamp.date(), "duplicate_session") for stamp in bars.index)
        return DatasetBuild(frame, tuple(exclusions), contract)
    exchange = _calendar(calendar)
    rows: list[pd.Series] = []
    row_dates: list[pd.Timestamp] = []
    for i, stamp in enumerate(bars.index):
        if i == len(bars) - 1:
            exclusions.append(ExcludedRow(stamp.date(), "target_not_observed"))
            continue
        if i < PRIOR_CLOSE_COUNT:
            exclusions.append(ExcludedRow(stamp.date(), "insufficient_history"))
            continue
        window = bars.index[i - PRIOR_CLOSE_COUNT : i + 2]
        reason = _adjacency_reason([day.date() for day in window], exchange)
        if reason is None:
            reason = _provenance_reason(provenance, window, contract)
        if reason is not None:
            exclusions.append(ExcludedRow(stamp.date(), reason))
            continue
        today_close = _positive_price(bars.iloc[i]["Close"])
        next_open = _positive_price(bars.iloc[i + 1]["Open"])
        if today_close is None or next_open is None:
            exclusions.append(ExcludedRow(stamp.date(), "nonpositive_price"))
            continue
        try:
            features = build_feature_row(
                bars["Close"].iloc[i - PRIOR_CLOSE_COUNT : i].tolist(),
                bars.iloc[i]["Open"],
                today_close,
                stamp.date(),
            )
        except NonfiniteDerivedValue:
            exclusions.append(ExcludedRow(stamp.date(), "nonfinite_derived_value"))
            continue
        except ValueError:
            exclusions.append(ExcludedRow(stamp.date(), "nonpositive_price"))
            continue
        label = next_open / today_close - 1
        if not isfinite(label):
            exclusions.append(ExcludedRow(stamp.date(), "nonfinite_label"))
            continue
        features[LABEL_COLUMN] = label
        rows.append(features)
        row_dates.append(stamp)
    if rows:
        frame = pd.DataFrame(rows, index=pd.DatetimeIndex(row_dates))
        frame = frame.loc[:, [*FEATURE_COLUMNS, LABEL_COLUMN]]
    return DatasetBuild(frame, tuple(exclusions), contract)


def build_scenario_features(
    prior_history: pd.DataFrame,
    provenance: pd.DataFrame | None,
    *,
    session: date,
    today_open: float,
    current_open_provenance: CurrentOpenProvenance | None,
    assumed_close: float,
    assumed_close_basis: str | None,
    contract: PriceContract,
    calendar=None,
) -> ScenarioBuild:
    """Construct live features only from verified raw-basis open/close inputs."""
    next_session = next_expected_session(session, calendar)
    if next_session is None:
        return ScenarioBuild(None, None, "calendar_unavailable")
    reason = _current_open_reason(current_open_provenance, contract)
    if reason is not None:
        return ScenarioBuild(None, next_session, reason)
    if not assumed_close_basis:
        return ScenarioBuild(None, next_session, "unknown_assumed_close_basis")
    if assumed_close_basis != contract.basis:
        return ScenarioBuild(None, next_session, "mixed_assumed_close_basis")
    if "Close" not in prior_history.columns or len(prior_history) < PRIOR_CLOSE_COUNT:
        return ScenarioBuild(None, next_session, "insufficient_history")
    if not isinstance(prior_history.index, pd.DatetimeIndex) or prior_history.index.hasnans:
        return ScenarioBuild(None, next_session, "invalid_session_index")
    if prior_history.index.tz is not None or not prior_history.index.equals(
        prior_history.index.normalize()
    ):
        return ScenarioBuild(None, next_session, "ambiguous_session_index")
    if prior_history.index.normalize().has_duplicates:
        return ScenarioBuild(None, next_session, "duplicate_session")
    bars = prior_history.sort_index().iloc[-PRIOR_CLOSE_COUNT:]
    if bars.index[-1].date() >= session:
        return ScenarioBuild(None, next_session, "future_history")
    exchange = _calendar(calendar)
    reason = _adjacency_reason([*(stamp.date() for stamp in bars.index), session], exchange)
    if reason is None:
        reason = _provenance_reason(provenance, bars.index, contract)
    if reason is not None:
        return ScenarioBuild(None, next_session, reason)
    try:
        features = build_feature_row(bars["Close"].tolist(), today_open, assumed_close, session)
    except NonfiniteDerivedValue:
        return ScenarioBuild(None, next_session, "nonfinite_derived_value")
    except ValueError:
        return ScenarioBuild(None, next_session, "nonpositive_price")
    return ScenarioBuild(features, next_session, None)
