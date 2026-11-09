# Local work reconciliation: November 9, 2026

The publication branch starts at `e346ee4db`, two commits ahead of the freshly
fetched `origin/main`. Those two changes use completed-session prices in What If
and display the complete trade return. They are included with the Jev work.

The audit reviewed all 130 local branches, five stashes, and four extant secondary
worktrees. The secondary worktrees were clean. Five branch heads are ancestors of
main, and 109 other heads exactly match already-merged PR heads. The remaining 16
branches are classified with evidence in `branch-audit.json`; the only active
unmerged source changes are the two What If commits above. No open PR existed at
the start of the audit. Historical branches and stash objects remain intact.

## Preserved historical source

`stash-inventory.json` records every stash commit, changed path, source patch, and
untracked source file. The `stash-*-source.patch` files preserve the original
tracked source differences. `stash-*-untracked/` preserves original source and
planning notes that were not tracked in the stash's parent commit.

These are historical evidence, not patches to apply to current code. In
particular, stash 0 contains the now-shipped morning quote recovery; stash 1
contains earlier SVM, trade, and What If implementations; stash 2 contains the
old lambdarank implementation that predates ListFold. Applying them would roll
back subsequent fixes. Their superseded binary/data paths are enumerated in the
inventory; the current data snapshot is published instead of reverting those
paths. The older September handoff already exists in the October 28 archive.

## Published data and research

All outstanding price and feature parquet files, existing trained and archived
model files, run history, trade ledger and broker confirmations, paper book,
weather, dated picks, and both November research directories are included. No
model was retrained as part of this publication.

`data/snapshots/2026-11-09/` preserves normally ignored durable scan JSON, the
scan/news databases, morning diagnostic JSON, and consistent copies of the three
tracked SQLite stores. Its manifest records each file's capture time and hash.
The archive is intended for reproducibility and recovery; live persistence stays
in the normal `data/` locations.

The 205.5 MiB research cache `five_year_arrays.pkl` exceeds GitHub's per-file size
limit. It remains at its original local path and is excluded by one exact ignore
rule. Its complete contents are published as bounded gzip pieces beside it;
`five_year_arrays.pkl.archive.json` gives source and piece hashes. Concatenate
the pieces in filename order and decompress to recover the original bytes:

```sh
cat research/averaging_down_2026_11_06/five_year_arrays.pkl.gz.part-* | gzip -dc > /tmp/five_year_arrays.pkl
shasum -a 256 /tmp/five_year_arrays.pkl
```

The expected original SHA-256 is
`d8d84a24cae4572c22c8e96bfd4dd7c5ddaa7c43e465918b77d5c8fed6a243e2`.
The archive was decompressed and checked against the original before publication.

## Deliberate exclusions

Credentials and `.env` files, dependency/build caches, generated frontend output,
Python bytecode, lock files, and SQLite WAL/SHM sidecars are not publication data.
The SQLite backup API incorporates committed WAL state into self-contained
database copies. The oversized raw research pickle is represented losslessly by
its archived pieces; it is not discarded. Old stale database/model snapshots in
stashes are superseded by the current complete data, while their source changes
remain reviewable in this archive.
