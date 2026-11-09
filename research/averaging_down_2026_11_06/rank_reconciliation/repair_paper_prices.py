"""Preview/apply a price-only repair; retain picks, reviews, scans and replays.

PYTHONPATH=python .venv/bin/python <this-file> [--apply]
The live database is backed up before an optimistic, transactional update.
"""
from collections import Counter
from datetime import datetime, timezone
import argparse
import hashlib
import json
from pathlib import Path
import sqlite3

from stock_picker.paper.book import _bar, _session_from_bar, paper_book_view
from stock_picker.storage.paper_book_store import PaperPick
from stock_picker.storage.price_store import PriceStore

ROOT = Path(__file__).resolve().parent
DATABASE = Path("data/paper_book/paper_book.db").resolve()
CUTOFF = "2026-11-06"


def rows(connection):
    connection.row_factory = sqlite3.Row
    return [dict(row) for row in connection.execute(
        "SELECT * FROM paper_picks ORDER BY as_of, scan_id, kind, rank"
    )]


def scan_digest():
    path = Path("data/buy_signals/scans.db").resolve()
    digest = hashlib.sha256()
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as connection:
        for row in connection.execute("SELECT as_of,kind,payload FROM scans ORDER BY as_of,kind"):
            digest.update(json.dumps(row).encode())
    return digest.hexdigest()


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--apply", action="store_true")
args = parser.parse_args()
with sqlite3.connect(f"file:{DATABASE}?mode=ro", uri=True) as source:
    before = rows(source)
scans_before = scan_digest()
store = PriceStore()
histories = {}
changes, after, missing = [], [], []
for row in before:
    updated = dict(row)
    if not row["scan_id"] and row["as_of"] <= CUTOFF:
        ticker = row["ticker"]
        if ticker not in histories:
            try:
                histories[ticker] = store.read(ticker)
            except FileNotFoundError:
                histories[ticker] = None
        history = histories[ticker]
        opened, closed, returned = _session_from_bar(_bar(history, row["as_of"]) if history is not None else None)
        updated.update(open_price=opened, close_price=closed, session_return=returned)
        if returned is None:
            missing.append({key: row[key] for key in ("as_of", "kind", "rank", "ticker")})
        if updated != row:
            changes.append({"before": row, "after": updated})
    after.append(updated)

assert scan_digest() == scans_before
summary = {
    "applied": args.apply, "cutoff": CUTOFF, "total_rows": len(before),
    "changed_rows": len(changes), "changed_by_kind": dict(Counter(c["before"]["kind"] for c in changes)),
    "replay_rows_unchanged": sum(bool(row["scan_id"]) for row in before),
    "missing_session_prices": missing, "scan_digest": scans_before,
}
if args.apply:
    backup = ROOT / ("paper_book_before_repair_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + ".db")
    with sqlite3.connect(f"file:{DATABASE}?mode=ro", uri=True) as source, sqlite3.connect(backup) as target:
        source.backup(target)
    with sqlite3.connect(DATABASE) as connection:
        connection.execute("BEGIN IMMEDIATE")
        assert rows(connection) == before, "Paper book changed during preview; abort and retry"
        for change in changes:
            row = change["after"]
            connection.execute(
                "UPDATE paper_picks SET open_price=?,close_price=?,session_return=? "
                "WHERE as_of=? AND scan_id=? AND kind=? AND rank=?",
                tuple(row[key] for key in ("open_price", "close_price", "session_return", "as_of", "scan_id", "kind", "rank")),
            )
        assert rows(connection) == after, "Repair changed more than the planned price fields"
        assert scan_digest() == scans_before, "Original scans changed during repair"
    summary["backup"] = str(backup)

picks = [PaperPick(**{**row, "news_check": json.loads(row["news_check"]) if row["news_check"] else None}) for row in after]
(ROOT / "repaired_paper_book_snapshot.json").write_text(json.dumps(paper_book_view(picks, kind="both"), indent=2))
(ROOT / "paper_price_repair_changes.json").write_text(json.dumps(changes, indent=2))
(ROOT / "paper_price_repair_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
