from stock_picker.parallel import run_buckets, split_buckets


def test_split_buckets_is_contiguous_slices():
    assert split_buckets(list(range(5)), 2) == [[0, 1], [2, 3], [4]]
    assert split_buckets([], 200) == []


def test_run_buckets_skips_the_pool_for_one_chunk():
    seen: list[list[int]] = []

    def work(chunk):
        seen.append(list(chunk))
        return sum(chunk)

    assert run_buckets(work, [1, 2, 3], bucket_size=10, workers=10) == [6]
    assert seen == [[1, 2, 3]]


def test_run_buckets_covers_every_item_once():
    def work(chunk):
        return list(chunk)

    results = run_buckets(work, list(range(6)), bucket_size=2, workers=3)
    assert sorted(item for chunk in results for item in chunk) == list(range(6))
