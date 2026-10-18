"""News judgments joined to what the pick actually did that day.

Every paper-book pick already carries the morning's news check and its
open->close return. This module turns that into two things:

- `news_outcome_rows`: one labelled row per judged pick (headlines, gap,
  judge, flag, Jev's news score, return, and whether it dropped hard). This is
  the training set for a learned "will it drop" news model once enough days
  accumulate; see ADR 0026.
- `summarize_news_outcomes`: flagged vs clear, scored on the question that
  matters -- did the flag point at the worse days?

Pure functions over `PaperPick`s; persistence stays in the paper book.
"""

from __future__ import annotations

import statistics
from collections.abc import Iterable

from stock_picker.storage.paper_book_store import PaperPick

# A pick that fell this much open->close is a "drop" -- what a news skip exists to avoid.
DROP_THRESHOLD = -0.05


def _gap(pick: PaperPick) -> float | None:
    if pick.prev_close and pick.open_price:
        return pick.open_price / pick.prev_close - 1
    return None


def news_outcome_rows(picks: Iterable[PaperPick], jev_by_pick: dict | None = None) -> list[dict]:
    """Labelled rows for live picks whose news reached a judge and whose day closed."""
    jev_by_pick = jev_by_pick or {}
    rows = []
    for pick in picks:
        check = pick.news_check or {}
        if pick.scan_id or pick.session_return is None or not check.get("judge"):
            continue
        jev = jev_by_pick.get((pick.as_of, pick.kind, pick.ticker)) or {}
        rows.append({
            "as_of": pick.as_of,
            "kind": pick.kind,
            "rank": pick.rank,
            "ticker": pick.ticker,
            "judge": check.get("judge"),
            "flagged": bool(check.get("flag")),
            "flag": check.get("flag"),
            "jev_news_skip": jev.get("news_skip"),
            "gap": _gap(pick),
            "headlines": [a.get("headline") for a in check.get("articles") or [] if a.get("headline")],
            "session_return": pick.session_return,
            "dropped": pick.session_return <= DROP_THRESHOLD,
        })
    return rows


def _group(rows: list[dict]) -> dict:
    returns = [row["session_return"] for row in rows]
    return {
        "n": len(rows),
        "avg": statistics.fmean(returns) if returns else None,
        "win_rate": sum(r > 0 for r in returns) / len(returns) if returns else None,
        "drop_rate": sum(row["dropped"] for row in rows) / len(rows) if rows else None,
    }


def flag_points_right(flagged: list[float], clear: list[float]) -> float | None:
    """Share of (flagged, clear) pairs where the flagged pick did worse. 0.5 = no skill."""
    if not flagged or not clear:
        return None
    worse = sum((f < c) + 0.5 * (f == c) for f in flagged for c in clear)
    return worse / (len(flagged) * len(clear))


def summarize_news_outcomes(rows: list[dict]) -> dict:
    flagged = [row for row in rows if row["flagged"]]
    clear = [row for row in rows if not row["flagged"]]
    return {
        "days": len({row["as_of"] for row in rows}),
        "flagged": _group(flagged),
        "clear": _group(clear),
        "flag_right_rate": flag_points_right(
            [row["session_return"] for row in flagged], [row["session_return"] for row in clear]
        ),
        "drop_threshold": DROP_THRESHOLD,
    }
