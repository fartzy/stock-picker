"""Universe-wide PCA and ticker clusters.

PCA: each date, 5 components of last night's close-based cross-section
(returns / RSI / vol / ATR). Fit is that day's 2,000 names only -- no
future dates. Component signs are aligned to the previous day.

Clusters: MiniBatchKMeans (k=8) on (return_20d, volatility_20d, rsi_14)
as of that date. Recency is the rolling windows already in those columns.

cluster_overnight_gap is open-known: mean peer (Open_t / Close_{t-1} - 1)
in the same cluster, excluding self. PCA and cluster_id are close-based
and get shift(1) in the training frame.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

PCA_N = 5
CLUSTER_K = 8
PCA_SOURCE_COLUMNS = (
    "return_1d",
    "return_5d",
    "return_20d",
    "rsi_14",
    "volatility_20d",
    "atr_14",
)
CLUSTER_SOURCE_COLUMNS = ("return_20d", "volatility_20d", "rsi_14")
PCA_FEATURE_COLUMNS = tuple(f"pca_{i}" for i in range(1, PCA_N + 1))
CLUSTER_ID_COLUMN = "cluster_id"
CLUSTER_GAP_COLUMN = "cluster_overnight_gap"
STRUCTURE_COLUMNS = (*PCA_FEATURE_COLUMNS, CLUSTER_ID_COLUMN, CLUSTER_GAP_COLUMN)
MIN_CROSS_SECTION = 40


def _cross_section(
    features_by_ticker: dict[str, pd.DataFrame],
    when,
    columns: tuple[str, ...],
) -> pd.DataFrame:
    rows: dict[str, pd.Series] = {}
    for ticker, table in features_by_ticker.items():
        if when not in table.index:
            continue
        missing = [name for name in columns if name not in table.columns]
        if missing:
            continue
        row = table.loc[when, list(columns)]
        if row.isna().any():
            continue
        rows[ticker] = row
    if not rows:
        return pd.DataFrame(columns=list(columns))
    return pd.DataFrame(rows).T


def _align_pca_signs(components: np.ndarray, previous: np.ndarray | None) -> np.ndarray:
    if previous is None or previous.shape != components.shape:
        return components
    aligned = components.copy()
    for i, (now, prior) in enumerate(zip(components, previous)):
        if np.dot(now, prior) < 0:
            aligned[i] = -now
    return aligned


def _overnight_gap(history: pd.DataFrame, when) -> float | None:
    if when not in history.index or "Open" not in history.columns or "Close" not in history.columns:
        return None
    loc = history.index.get_loc(when)
    if isinstance(loc, slice) or isinstance(loc, np.ndarray):
        return None
    if loc == 0:
        return None
    prev_close = history.iloc[loc - 1]["Close"]
    today_open = history.loc[when]["Open"]
    if pd.isna(prev_close) or pd.isna(today_open) or float(prev_close) <= 0:
        return None
    return (float(today_open) - float(prev_close)) / float(prev_close)


def build_structure_features(
    histories: dict[str, pd.DataFrame],
    features_by_ticker: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """Add PCA / cluster columns onto each ticker's feature table (same index)."""
    extra = {
        ticker: pd.DataFrame(index=table.index, columns=list(STRUCTURE_COLUMNS), dtype="float64")
        for ticker, table in features_by_ticker.items()
    }
    dates = sorted({when for table in features_by_ticker.values() for when in table.index})
    previous_components: np.ndarray | None = None

    for when in dates:
        pca_frame = _cross_section(features_by_ticker, when, PCA_SOURCE_COLUMNS)
        if len(pca_frame) >= MIN_CROSS_SECTION:
            scaled = StandardScaler().fit_transform(pca_frame.to_numpy())
            n_comp = min(PCA_N, scaled.shape[0], scaled.shape[1])
            fitted = PCA(n_components=n_comp, random_state=0).fit(scaled)
            components = _align_pca_signs(fitted.components_, previous_components)
            previous_components = components
            scores = scaled @ components.T
            for i, ticker in enumerate(pca_frame.index):
                for j in range(n_comp):
                    extra[ticker].at[when, f"pca_{j + 1}"] = float(scores[i, j])

        cluster_frame = _cross_section(features_by_ticker, when, CLUSTER_SOURCE_COLUMNS)
        if len(cluster_frame) >= max(MIN_CROSS_SECTION, CLUSTER_K * 3):
            scaled = StandardScaler().fit_transform(cluster_frame.to_numpy())
            labels = MiniBatchKMeans(
                n_clusters=CLUSTER_K,
                random_state=0,
                n_init=3,
                batch_size=min(256, len(cluster_frame)),
            ).fit_predict(scaled)
            label_by_ticker = dict(zip(cluster_frame.index, labels.astype(float)))
            for ticker, label in label_by_ticker.items():
                extra[ticker].at[when, CLUSTER_ID_COLUMN] = label
            gaps: dict[str, float] = {}
            for ticker in label_by_ticker:
                history = histories.get(ticker)
                if history is None:
                    continue
                gap = _overnight_gap(history, when)
                if gap is not None:
                    gaps[ticker] = gap
            for ticker, label in label_by_ticker.items():
                peers = [
                    gaps[name]
                    for name, peer_label in label_by_ticker.items()
                    if peer_label == label and name != ticker and name in gaps
                ]
                if peers:
                    extra[ticker].at[when, CLUSTER_GAP_COLUMN] = float(np.mean(peers))

    return extra


def merge_structure(
    features_by_ticker: dict[str, pd.DataFrame],
    structure_by_ticker: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    merged = {}
    for ticker, table in features_by_ticker.items():
        extra = structure_by_ticker.get(ticker)
        merged[ticker] = table if extra is None or extra.empty else pd.concat([table, extra], axis=1)
    return merged


def fill_cluster_overnight_gaps(
    rows: list,
    quotes: dict[str, dict],
) -> None:
    """Overwrite cluster_overnight_gap on live rows from today's opens.

    Cluster membership is last night's cluster_id already on the row.
    """
    membership: dict[str, float] = {}
    for item in rows:
        if item.row is None or CLUSTER_ID_COLUMN not in item.row.columns:
            continue
        value = item.row.iloc[0][CLUSTER_ID_COLUMN]
        if pd.isna(value):
            continue
        membership[item.ticker] = float(value)
    gaps: dict[str, float] = {}
    for ticker, quote in quotes.items():
        open_px = quote.get("open")
        prev = quote.get("prev_close")
        if open_px is None or prev is None or float(prev) <= 0:
            continue
        gaps[ticker] = (float(open_px) - float(prev)) / float(prev)
    for item in rows:
        if item.ticker not in membership or item.row is None:
            continue
        if CLUSTER_GAP_COLUMN not in item.row.columns:
            continue
        label = membership[item.ticker]
        peers = [
            gaps[name]
            for name, peer_label in membership.items()
            if peer_label == label and name != item.ticker and name in gaps
        ]
        if peers:
            item.row.iloc[0, item.row.columns.get_loc(CLUSTER_GAP_COLUMN)] = float(np.mean(peers))
