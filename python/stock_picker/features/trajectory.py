"""Five-session return/volume path clusters learned in chronological order.

Each date is scored against centers fitted through *earlier* dates only. The
current date updates those centers after scoring, for use on later sessions.
Membership starts at registry first_seen to avoid a retrospective universe.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans

PATH_DAYS = 5
NORMALIZER_DAYS = 20
CLUSTER_K = 6
MIN_PATH_NAMES = 40
PATH_COLUMNS = (
    "path_cluster_distance",
    "path_cluster_margin",
    "path_cluster_crowding",
    "path_cluster_peer_return_1d",
)


def _path_vectors(history: pd.DataFrame) -> pd.DataFrame:
    """One scale-adjusted five-day path per completed session."""
    returns = history["Close"].pct_change(fill_method=None)
    volatility = returns.rolling(NORMALIZER_DAYS).std().where(lambda value: value.gt(0))
    normalized_returns = (returns / volatility).clip(-5, 5)
    volume = history["Volume"].where(history["Volume"].gt(0))
    volume_mean = volume.rolling(NORMALIZER_DAYS).mean()
    normalized_volume = np.log(volume / volume_mean).clip(-3, 3)
    return pd.DataFrame(
        {
            **{f"return_{lag}": normalized_returns.shift(lag) for lag in range(PATH_DAYS)},
            **{f"volume_{lag}": normalized_volume.shift(lag) for lag in range(PATH_DAYS)},
        },
        index=history.index,
    )


def build_trajectory_features(
    histories: dict[str, pd.DataFrame], first_seen_by_ticker: dict[str, str]
) -> dict[str, pd.DataFrame]:
    """Return four leak-safe path-cluster columns on each ticker's date index."""
    result = {
        ticker: pd.DataFrame(index=history.index, columns=PATH_COLUMNS, dtype="float64")
        for ticker, history in histories.items()
    }
    paths = {ticker: _path_vectors(history) for ticker, history in histories.items()}
    returns = {
        ticker: history["Close"].pct_change(fill_method=None)
        for ticker, history in histories.items()
    }
    first_seen_dates = {
        ticker: pd.Timestamp(first_seen)
        for ticker, first_seen in first_seen_by_ticker.items()
    }
    dates = sorted({when for path in paths.values() for when in path.index})
    model = MiniBatchKMeans(n_clusters=CLUSTER_K, random_state=0, n_init=3, batch_size=256)
    fitted = False

    for when in dates:
        names, vectors, day_returns = [], [], []
        for ticker, path in paths.items():
            first_seen = first_seen_dates.get(ticker)
            if first_seen is None or when < first_seen or when not in path.index:
                continue
            vector = path.loc[when].to_numpy(dtype=float)
            current_return = returns[ticker].loc[when]
            if np.isfinite(vector).all() and np.isfinite(current_return):
                names.append(ticker)
                vectors.append(vector)
                day_returns.append(float(current_return))
        if len(names) < max(MIN_PATH_NAMES, CLUSTER_K):
            continue

        matrix = np.asarray(vectors, dtype=float)
        if fitted:
            distances = model.transform(matrix)
            labels = distances.argmin(axis=1)
            nearest = np.sort(distances, axis=1)[:, :2]
            counts = np.bincount(labels, minlength=CLUSTER_K)
            sums = np.bincount(labels, weights=day_returns, minlength=CLUSTER_K)
            for index, ticker in enumerate(names):
                label = labels[index]
                result[ticker].at[when, "path_cluster_distance"] = float(nearest[index, 0])
                result[ticker].at[when, "path_cluster_margin"] = float(nearest[index, 1] - nearest[index, 0])
                result[ticker].at[when, "path_cluster_crowding"] = float(counts[label] / len(names))
                if counts[label] > 1:
                    result[ticker].at[when, "path_cluster_peer_return_1d"] = float(
                        (sums[label] - day_returns[index]) / (counts[label] - 1)
                    )
        model.partial_fit(matrix)
        fitted = True

    return result
