"""Morning-scale bucket parallelism.

The live universe is ~2,000 names. Yahoo's v7 quote URL and per-ticker
parquet scoring both work in chunks of 200, which is 10 buckets. Quotes
and inference share this helper so there is one thread-pool pattern, not
two copies of ThreadPoolExecutor.

Results come back in completion order. Callers that need a ranking sort
after merge.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TypeVar

BUCKET_SIZE = 200
WORKERS = 10

T = TypeVar("T")
R = TypeVar("R")


def split_buckets(items: Sequence[T], size: int = BUCKET_SIZE) -> list[list[T]]:
    if not items:
        return []
    return [list(items[i : i + size]) for i in range(0, len(items), size)]


def run_buckets(
    work: Callable[[Sequence[T]], R],
    items: Sequence[T],
    *,
    bucket_size: int = BUCKET_SIZE,
    workers: int = WORKERS,
) -> list[R]:
    """Run `work` once per bucket.

    A single bucket stays on this thread -- tests and leftover quote
    fallbacks should not pay for a pool of 10.
    """
    chunks = split_buckets(items, bucket_size)
    if not chunks:
        return []
    if len(chunks) == 1:
        return [work(chunks[0])]
    n_workers = min(workers, len(chunks))
    results: list[R] = []
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = [pool.submit(work, chunk) for chunk in chunks]
        for future in as_completed(futures):
            results.append(future.result())
    return results
