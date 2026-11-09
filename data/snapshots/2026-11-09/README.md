# Durable local state: November 9, 2026

`durable-local-state.tar.gz` contains the paths and SHA-256 hashes listed in
`manifest.json`. It preserves normally ignored saved scans, news, and morning
diagnostics, plus consistent copies of the tracked trades, paper-book, and MLflow
databases. Each database was copied with SQLite's backup API and passed an
integrity check. The archive was then read back and every member's size and hash
verified.

The individual capture times in the manifest matter: this is a reproducible copy
of each store, not one transaction spanning all databases. A later Jev prediction
created after this capture remains in its normal tracked persistence location.

Inspect or restore into a separate directory:

```sh
tar -tzf data/snapshots/2026-11-09/durable-local-state.tar.gz
mkdir -p /tmp/stock-picker-state-2026-11-09
tar -xzf data/snapshots/2026-11-09/durable-local-state.tar.gz -C /tmp/stock-picker-state-2026-11-09
```

To use these copies in a fresh checkout, stop processes using its databases first,
then copy the desired `data/` paths from the extracted directory. Restore only to
the checkout you intend to replace. Do not copy stale WAL/SHM sidecars along with
the databases. Existing local state is not modified by the archive or inspection
commands above.
