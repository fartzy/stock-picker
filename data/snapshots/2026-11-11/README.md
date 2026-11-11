# Durable app data snapshot

Captured November 11, 2026 without stopping the API. The [manifest](manifest.json) records a capture timestamp and SHA-256 for every artifact. Individual files/stores have their own capture times; this is not a single transaction across all databases.

The `files/` tree preserves 97 normally ignored durable scan and morning-check files. The `sqlite/` tree contains seven compressed SQLite backups: saved scans, news, Jev forecasts, Jev outcomes, paper book, trade ledger and MLflow. SQLite's backup API incorporates committed WAL contents. Each backup passed `PRAGMA integrity_check` and decompression/hash verification.

The largest compressed backup is approximately 65.7 MiB and remains below GitHub's 100 MiB per-file limit. Uncompressed live news data remains in its normal ignored location.

To inspect a backup without replacing a live store:

```sh
gzip -dc data/snapshots/2026-11-11/sqlite/data/trades/stockpicker.db.gz > /tmp/stock-picker-trades-20261111.db
sqlite3 /tmp/stock-picker-trades-20261111.db 'PRAGMA integrity_check;'
shasum -a 256 /tmp/stock-picker-trades-20261111.db
```

Compare the uncompressed SHA-256 with that archive's `uncompressed_sha256` in the manifest. Restoring live data is a separate action; this capture does not alter live database contents. Credentials, dependency/build caches, bytecode and WAL/SHM sidecars are excluded. The logical WAL contents are preserved in the backups.

The capture/validation script and audit records are in [the check-in archive](../../../docs/archive/local-checkin-2026-11-11/README.md).
