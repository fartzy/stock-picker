"""Reconstruct dated Rank top-20 picks from prior-fitted morning scores.

This is a backtest cohort, not an archive of the actual morning scans. Its
candidate universe comes from today's active registry and same-day training
rows, which require an observed same-day close. Historically delisted names
and open-only names may be missing. Rank selection happens before any optional
ticker restriction or label join, so a missing overnight bar cannot elevate a
lower-ranked name into the available cohort.
"""

from __future__ import annotations

from datetime import date
from hashlib import sha256
import os
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from stock_picker.training.rank_model import RANK_TOP_K


COHORT_SOURCE = "reconstructed_prior_fit_rank_oof"
COHORT_UNIVERSE_LIMIT = "current_active_tickers_with_observed_day_close_and_features"


def _exchange_dates(values: pd.Series, label: str) -> pd.Series:
    dates = pd.to_datetime(values, errors="raise")
    if dates.isna().any() or dates.dt.tz is not None or not dates.equals(dates.dt.normalize()):
        raise ValueError(f"{label} must be timezone-naive exchange dates")
    return dates


def select_ranked_cohort(
    day_scores: pd.DataFrame,
    day_training_frame: pd.DataFrame,
    *,
    top_k: int = RANK_TOP_K,
) -> pd.DataFrame:
    """Choose each scored session's top-K from its *complete* day cross-section.

    All morning outputs come from a model fitted before that session. Missing
    score rows fail the run instead of silently changing which names are top-K.
    Equal Rank scores break by ticker symbol, independent of frame order.
    """
    required_scores = {"ticker", "date", "day_rank_score", "trained_through"}
    required_day = {"ticker", "date"}
    if required_scores - set(day_scores) or required_day - set(day_training_frame):
        raise ValueError("Rank cohort needs dated scores, fitting cutoffs, and day-universe keys")
    if top_k < 1 or day_scores.empty or day_training_frame.empty:
        raise ValueError("Rank cohort needs a positive top_k and nonempty cross-sections")

    scores = day_scores.copy()
    universe = day_training_frame[["ticker", "date"]].copy()
    scores["date"] = _exchange_dates(scores["date"], "score dates")
    universe["date"] = _exchange_dates(universe["date"], "day-universe dates")
    scores["trained_through"] = _exchange_dates(scores["trained_through"], "score fitting cutoffs")
    if scores.duplicated(["date", "ticker"]).any() or universe.duplicated(["date", "ticker"]).any():
        raise ValueError("Rank cohort keys must be unique per ticker and date")
    if scores["ticker"].isna().any() or universe["ticker"].isna().any():
        raise ValueError("Rank cohort tickers cannot be missing")
    if not (scores["trained_through"] < scores["date"]).all():
        raise ValueError("Rank cohort scores must be fitted strictly earlier than their session")
    if not np.isfinite(scores["day_rank_score"].to_numpy(dtype=float)).all():
        raise ValueError("Rank cohort scores must be finite")

    scored_dates = scores["date"].drop_duplicates()
    missing_sessions = universe["date"].drop_duplicates()
    missing_sessions = missing_sessions.loc[
        (missing_sessions >= scored_dates.min()) & ~missing_sessions.isin(scored_dates)
    ]
    if not missing_sessions.empty:
        raise ValueError("Rank cohort is missing a complete scored day-universe session")
    expected = universe.loc[universe["date"].isin(scored_dates), ["date", "ticker"]]
    coverage = expected.merge(
        scores[["date", "ticker"]], on=["date", "ticker"], how="outer", indicator=True,
        validate="one_to_one",
    )
    if not coverage["_merge"].eq("both").all():
        raise ValueError("Rank cohort requires every ticker in each scored full-universe session")
    if (scores.groupby("date").size() < top_k).any():
        raise ValueError("Rank cohort has fewer names than the requested top-K")

    ordered = scores.sort_values(
        ["date", "day_rank_score", "ticker"], ascending=[True, False, True], kind="stable",
    )
    selected = ordered.loc[ordered.groupby("date", sort=False).cumcount() < top_k].copy()
    selected["rank_position"] = selected.groupby("date", sort=False).cumcount() + 1
    selected["selection_source"] = COHORT_SOURCE
    return selected.reset_index(drop=True)


def ranked_labeled_rows(labels: pd.DataFrame, cohort: pd.DataFrame) -> pd.DataFrame:
    """Keep labels for selected dates/names only; unlabeled final picks vanish."""
    keys = ["date", "ticker"]
    if any(name not in labels or name not in cohort for name in keys):
        raise ValueError("Ranked labels and cohort need date/ticker keys")
    if labels.duplicated(keys).any() or cohort.duplicated(keys).any():
        raise ValueError("Ranked label and cohort keys must be unique")
    return labels.merge(cohort[keys], on=keys, how="inner", validate="one_to_one")


def make_ranked_cohort_manifest(
    cohort: pd.DataFrame,
    verified_labels: pd.DataFrame,
    *,
    label_start: date | None = None,
    label_end: date | None = None,
    ticker_whitelist: set[str] | None = None,
) -> pd.DataFrame:
    """Serializable dated selections and whether each has a verified label."""
    required = {"date", "ticker", "rank_position", "day_rank_score", "trained_through", "selection_source"}
    if required - set(cohort) or {"date", "ticker"} - set(verified_labels):
        raise ValueError("cohort manifest needs selected ranks and verified label keys")
    if cohort.duplicated(["date", "ticker"]).any() or verified_labels.duplicated(["date", "ticker"]).any():
        raise ValueError("cohort manifest keys must be unique")
    selected = cohort[list(required)].copy()
    selected["date"] = _exchange_dates(selected["date"], "cohort dates")
    selected["trained_through"] = _exchange_dates(selected["trained_through"], "rank fitting cutoffs")
    verified = verified_labels[["date", "ticker"]].copy()
    verified["date"] = _exchange_dates(verified["date"], "verified label dates")
    labeled = selected.merge(
        verified.assign(_verified_label=True),
        on=["date", "ticker"], how="left", validate="one_to_one",
    )
    in_window = pd.Series(True, index=labeled.index)
    if label_start is not None:
        in_window &= labeled["date"] >= pd.Timestamp(label_start)
    if label_end is not None:
        in_window &= labeled["date"] <= pd.Timestamp(label_end)
    in_whitelist = pd.Series(True, index=labeled.index)
    if ticker_whitelist is not None:
        in_whitelist &= labeled["ticker"].isin(ticker_whitelist)
    labeled["label_status"] = np.where(
        ~in_window, "outside_requested_range",
        np.where(
            ~in_whitelist, "outside_ticker_whitelist",
            np.where(labeled["_verified_label"].eq(True), "verified_label", "no_verified_label"),
        ),
    )
    manifest = labeled.rename(columns={
        "day_rank_score": "rank_score", "trained_through": "rank_trained_through",
    })
    manifest = manifest.sort_values(["date", "rank_position", "ticker"], kind="stable")
    manifest["date"] = manifest["date"].dt.strftime("%Y-%m-%d")
    manifest["rank_trained_through"] = manifest["rank_trained_through"].dt.strftime("%Y-%m-%d")
    return manifest[[
        "date", "ticker", "rank_position", "rank_score", "rank_trained_through",
        "selection_source", "label_status",
    ]].reset_index(drop=True)


def save_ranked_cohort_manifest(manifest: pd.DataFrame, tracking_dir: Path) -> tuple[Path, str]:
    """Atomically store content-addressed CSV without replacing another run."""
    if manifest.empty:
        raise ValueError("cannot save an empty Rank cohort manifest")
    csv = manifest.to_csv(index=False, float_format="%.17g", lineterminator="\n")
    content = csv.encode("utf-8")
    digest = sha256(content).hexdigest()
    directory = tracking_dir / "overnight_cohorts"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"ranked_{manifest['date'].iloc[0]}_{manifest['date'].iloc[-1]}_{digest[:12]}.csv"
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError("existing Rank cohort manifest has different content")
        return path, digest
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=directory, prefix=".ranked-", suffix=".tmp", delete=False) as file:
            temporary = Path(file.name)
            file.write(content)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return path, digest
