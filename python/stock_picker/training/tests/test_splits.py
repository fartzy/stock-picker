import pandas as pd

from stock_picker.training.splits import select_holdout_tickers, walk_forward_splits


def test_select_holdout_tickers_is_deterministic():
    tickers = [f"T{i}" for i in range(50)]

    first = select_holdout_tickers(tickers, fraction=0.1, seed=42)
    second = select_holdout_tickers(tickers, fraction=0.1, seed=42)

    assert first == second


def test_select_holdout_tickers_returns_approximately_the_requested_fraction():
    tickers = [f"T{i}" for i in range(100)]

    holdout = select_holdout_tickers(tickers, fraction=0.1)

    assert len(holdout) == 10


def test_select_holdout_tickers_only_returns_input_tickers():
    tickers = [f"T{i}" for i in range(20)]

    holdout = select_holdout_tickers(tickers, fraction=0.25)

    assert holdout <= set(tickers)


def test_select_holdout_tickers_returns_at_least_one_for_a_small_universe():
    tickers = ["A", "B", "C"]

    holdout = select_holdout_tickers(tickers, fraction=0.1)

    assert len(holdout) >= 1


def test_folds_train_dates_precede_test_dates():
    dates = pd.Series(pd.date_range("2026-01-01", periods=40, freq="B"))
    pooled_dates = (
        pd.concat([dates, dates], ignore_index=True)
        .sample(frac=1, random_state=0)
        .reset_index(drop=True)
    )

    splits = walk_forward_splits(pooled_dates, n_splits=4)

    assert len(splits) == 4
    for train_mask, test_mask in splits:
        train_dates = pooled_dates[train_mask]
        test_dates = pooled_dates[test_mask]
        assert train_dates.max() < test_dates.min()


def test_folds_cover_progressively_more_history():
    dates = pd.Series(pd.date_range("2026-01-01", periods=20, freq="B"))

    splits = walk_forward_splits(dates, n_splits=3)

    train_counts = [train_mask.sum() for train_mask, _ in splits]
    assert train_counts == sorted(train_counts)
    assert train_counts[0] < train_counts[-1]


def test_no_train_test_overlap_within_a_fold():
    dates = pd.Series(pd.date_range("2026-01-01", periods=20, freq="B"))

    splits = walk_forward_splits(dates, n_splits=3)

    for train_mask, test_mask in splits:
        assert not (train_mask & test_mask).any()
