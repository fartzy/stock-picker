import pandas as pd

from stock_picker.training.splits import walk_forward_splits


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
