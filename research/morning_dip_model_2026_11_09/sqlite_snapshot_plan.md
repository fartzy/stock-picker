# Final live-database snapshot plan

Status: executed and verified at 2026-11-10T02:29:42.070718+00:00. See the immutable
[execution manifest](final_sqlite_snapshot_manifest.json) and
[publication verification](ordinary_training_completion_verification.json). The three
transaction-consistent snapshots passed integrity and foreign-key checks. The API/UI
remained running, and live database files were not replaced. The procedure below is
retained for reproducibility; no resnapshotting is needed merely to clear Git status.
Do not overwrite, rename, checkpoint, vacuum, reset, or restore a live database.

## Scope and runtime

Capture all committed state, including pre-existing user changes, from exactly:

- `data/jev/forecasts.db`
- `data/mlruns/mlflow.db`
- `data/paper_book/paper_book.db`

Use the application's hermetic runtime:

```sh
bazel-bin/python/stock_picker/api/main.runfiles/rules_python++python+python_3_12_aarch64-apple-darwin/bin/python3
```

Read-only preflight found JEV and MLflow in DELETE journal mode. Paper Book has a WAL
header and reads successfully through application Python 3.12 / SQLite 3.50.4.
System Python 3.9 / SQLite 3.54.0 failed Paper Book schema reads with `unable to open
database file`; do not use that runtime or an `immutable=1` bypass. Directory/file
ownership and permissions were normal. At that preflight, no final backup or staging had been performed; the execution manifest now records the completed backups. Git staging is coordinated separately.

## Consistent backup

The prepared helper is
`snapshot_live_databases.py`; it requires an explicit execution flag and the
completed publication audit. Root will run it only after its final go-ahead:

```sh
bazel-bin/python/stock_picker/api/main.runfiles/rules_python++python+python_3_12_aarch64-apple-darwin/bin/python3 research/morning_dip_model_2026_11_09/snapshot_live_databases.py --approved-after-audit
```

For each file it opens the live source with `mode=ro`, sets connection-local
`query_only`, and establishes a read transaction. Python's supported
`Connection.backup` copies that SQLite view to a new ignored snapshot file. This
includes committed pages visible through WAL; it does not copy the main `.db`
file while ignoring WAL, nor copy `.db`, `-wal`, and `-shm` at unrelated times.
The read transaction is released immediately after the small backup completes.
No live journal-mode changes or checkpoint requests are issued.

The destination alone is converted to DELETE journal mode, closed, reopened
read-only, and checked with `PRAGMA integrity_check` and `foreign_key_check`.
Require `ok`, zero foreign-key violations, and no nonempty destination WAL or
journal. Record schema hashes, per-table row counts, byte sizes, SHA-256 hashes,
source/destination paths, snapshot times, runtime versions, and the publication
audit hash. Read-only snapshot files and a final manifest preserve that evidence.

Each database is internally consistent at its recorded snapshot time. The three
backups are sequential, not one cross-database transaction. Changes committed by
other app requests after a database's snapshot remain live and belong to a later
snapshot; they must not be discarded to make Git look clean.

## Stage the backup bytes without replacing live databases

Coordinate the shared Git index with root/design first. Use the manifest's exact
three files. For each path, preserve its existing tracked mode, write the verified
snapshot bytes as a Git blob, and point that path's index entry to the blob:

```python
mode = subprocess.check_output(
    ["git", "ls-files", "--stage", "--", live_path], text=True
).split()[0]
blob = subprocess.check_output(
    ["git", "hash-object", "-w", "--", snapshot_path], text=True
).strip()
subprocess.run(
    ["git", "update-index", "--cacheinfo", f"{mode},{blob},{live_path}"], check=True
)
staged_bytes = subprocess.check_output(["git", "show", f":{live_path}"])
assert hashlib.sha256(staged_bytes).hexdigest() == expected_snapshot_sha256
```

This changes the Git index only; live files and active SQLite connections are
untouched. Do not `git add` the live databases afterward, because it would replace
the consistent snapshot blobs with arbitrary live-file bytes. Never stage WAL,
SHM, or journal sidecars. Stage the manifest and other approved static artifacts
separately, inspect the exact staged allowlist, and run `git diff --cached --check`.
After commit, verify `git show HEAD:<path>` hashes against the manifest before push.

A live database may remain shown as modified after this commit because its current
bytes differ from the consistent backup or new requests have committed. Report that
honestly; do not use checkout/reset/stash/assume-unchanged to hide or erase it.

## Documentation verified

- [Python Connection.backup](https://docs.python.org/3/library/sqlite3.html#sqlite3.Connection.backup)
- [SQLite online backup API](https://www.sqlite.org/backup.html)
- [SQLite WAL and read-only access](https://www.sqlite.org/wal.html)
- [Git update-index --cacheinfo](https://git-scm.com/docs/git-update-index)
- [Git hash-object](https://git-scm.com/docs/git-hash-object)

Official-document fetch hashes are recorded in `operational_api_references.json`.
A disposable `/tmp` WAL database verified that the application runtime backup
includes committed WAL-only rows and produces a standalone destination passing
integrity and foreign-key checks. This test used no live project database.
