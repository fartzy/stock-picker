"""Walk-forward, date-based train/test splits for the pooled cross-sectional dataset.

Splitting by row index would leak dates across train/test when multiple tickers' rows
are interleaved in the pooled DataFrame. Instead this splits on the unique sorted
trading dates, then maps each fold's date sets back to row masks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def walk_forward_splits(
    dates: pd.Series, n_splits: int = 4
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return `n_splits` (train_mask, test_mask) boolean-array pairs aligned to `dates`.

    Unique sorted dates are partitioned into `n_splits + 1` contiguous blocks. Fold k
    trains on every date in blocks[0..k] and tests on block k+1, so every train date
    strictly precedes every test date in every fold.
    """
    unique_dates = np.sort(dates.unique())
    blocks = np.array_split(unique_dates, n_splits + 1)

    splits = []
    for k in range(n_splits):
        train_dates = set(np.concatenate(blocks[: k + 1]))
        test_dates = set(blocks[k + 1])
        train_mask = dates.isin(train_dates).to_numpy()
        test_mask = dates.isin(test_dates).to_numpy()
        splits.append((train_mask, test_mask))

    return splits
