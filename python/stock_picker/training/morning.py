"""Weekday morning scoring: today's opens through last night's model.

Runs at 8:32 CT. Rank and Fit score in parallel on one quote pull.
Rank is published as soon as Rank finishes (target ~8:35 CT).
"""

from __future__ import annotations

import fcntl
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime
from pathlib import Path

from stock_picker.ingestion.session import CHICAGO_TIMEZONE
from stock_picker.log import get_logger

from stock_picker.features.earnings import fetch_recent_earnings_tickers
from stock_picker.features.quotes import fetch_ticker_quotes
from stock_picker.storage.universe_store import UniverseStore
from stock_picker.training.news_day_judge import fetch_recent_news_flags
from stock_picker.storage.paths import data_root
from stock_picker.storage.scan_store import ScanStore
from stock_picker.training.buy_signal import DEFAULT_THRESHOLD, compute_buy_signals, compute_rank_signals
from stock_picker.training.freshness import pipeline_freshness
from stock_picker.training.notify import (
    format_not_ready_email,
    format_picks_email,
    format_rank_picks,
    publish_picks,
    send_email,
    write_picks_files,
)
from stock_picker.storage.training_config_store import TrainingConfigStore
from stock_picker.training.rank_model import RANK_NEWS_TOP_K, RANK_TOP_K

logger = get_logger(__name__)

DEFAULT_SIGNAL_DIR = data_root() / "buy_signals"
_LOCK_PATH = data_root() / "buy_signals" / "morning.lock"


def _try_lock_morning():
    """Non-blocking lock so 8:32 cannot start a second scan.

    Returns an open file that must stay open until scoring finishes, or
    None if another process already holds the lock.
    """
    _LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    handle = open(_LOCK_PATH, "a")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        return None
    return handle


def load_cached_signals(
    as_of: str | None = None,
    signal_dir: Path = DEFAULT_SIGNAL_DIR,
    kind: str = "fit",
) -> dict | None:
    day = as_of or date.today().isoformat()
    return ScanStore(data_dir=signal_dir).read(day, kind)


def _write_signals(payload: dict, as_of: str, signal_dir: Path = DEFAULT_SIGNAL_DIR) -> Path:
    kind = payload.get("kind") or ("rank" if as_of.endswith("-rank") else "fit")
    day = payload.get("as_of") or as_of.removesuffix("-rank")
    ScanStore(data_dir=signal_dir).write(day, kind, payload)
    name = f"{day}-rank.json" if kind == "rank" else f"{day}.json"
    return signal_dir / name


def _payload_from(result, freshness) -> dict:
    return {
        "as_of": result.as_of,
        "threshold": result.threshold,
        "signals": [
            {
                "ticker": signal.ticker,
                "predicted_return": signal.predicted_return,
                "open_price": signal.open_price,
                "snapshot_date": signal.snapshot_date,
                "news_flag": signal.news_flag,
            }
            for signal in result.signals
        ],
        "scored_count": result.scored_count,
        "skipped": result.skipped,
        "top_drivers": [
            {"feature": name, "importance": value} for name, value in result.top_drivers
        ],
        "freshness": {
            "feature_snapshot_date": freshness.feature_snapshot_date,
            "model_trained_through": freshness.model_trained_through,
            "ready_for_inference": freshness.ready_for_inference,
            "detail": freshness.detail,
        },
    }


def score_from_quotes(
    quotes: dict[str, dict],
    threshold: float = DEFAULT_THRESHOLD,
    persist: bool = True,
    news_fetcher=fetch_recent_news_flags,
    earnings_fetcher=fetch_recent_earnings_tickers,
):
    """Rank + Fit in parallel on a quote map, then news on Fit + Rank top 10.

    Same path the 8:32 job and Test run use. persist=False skips cache/git/email
    so a test run does not overwrite Monday's files.
    """
    freshness = pipeline_freshness()

    def quote_fetcher(tickers, as_of=None):
        return quotes

    def _rank():
        return compute_rank_signals(
            top_k=RANK_TOP_K,
            quote_fetcher=quote_fetcher,
            earnings_fetcher=earnings_fetcher,
            news_fetcher=None,
        )

    def _fit():
        return compute_buy_signals(
            threshold=threshold,
            quote_fetcher=quote_fetcher,
            earnings_fetcher=earnings_fetcher,
            news_fetcher=None,
        )

    rank_result = None
    rank_n = 0
    rank_text = ""
    with ThreadPoolExecutor(max_workers=2) as pool:
        rank_future = pool.submit(_rank)
        fit_future = pool.submit(_fit)
        try:
            rank_result = rank_future.result()
            if persist and rank_result.signals:
                rank_payload = _payload_from(rank_result, freshness)
                rank_payload["kind"] = "rank"
                _write_signals(rank_payload, rank_result.as_of)
                rank_text = format_rank_picks(rank_payload, k=RANK_TOP_K)
                write_picks_files(rank_result.as_of, rank_text)
                publish_picks(rank_result.as_of, rank_text)
                rank_n = len(rank_result.signals)
                logger.info("rank_top=%s published first", rank_n)
        except Exception:
            logger.exception("ranking skipped")
        result = fit_future.result()

    if not persist and rank_result is not None:
        rank_n = len(rank_result.signals)

    def _apply_flags(flags):
        for signal in result.signals:
            if signal.ticker in flags:
                signal.news_flag = flags[signal.ticker]
        if rank_result is not None:
            for signal in rank_result.signals:
                if signal.ticker in flags:
                    signal.news_flag = flags[signal.ticker]

    def _news(tickers, label):
        names = list(dict.fromkeys(tickers))
        if not names or news_fetcher is None:
            return {}
        logger.info("news check %s names (%s)", len(names), label)
        return news_fetcher(names, date.fromisoformat(result.as_of)) or {}

    first = [s.ticker for s in result.signals]
    if rank_result is not None:
        first.extend(s.ticker for s in rank_result.signals[:RANK_NEWS_TOP_K])
    _apply_flags(_news(first, f"fit + rank 1-{RANK_NEWS_TOP_K}"))
    if persist:
        if rank_result is not None and rank_result.signals:
            rank_payload = _payload_from(rank_result, freshness)
            rank_payload["kind"] = "rank"
            _write_signals(rank_payload, rank_result.as_of)
            rank_text = format_rank_picks(rank_payload, k=RANK_TOP_K)
        fit_payload = _payload_from(result, freshness)
        _write_signals(fit_payload, result.as_of)
        _subject, fit_body = format_picks_email(fit_payload)
        body = (rank_text + "\n" + fit_body) if rank_text else fit_body
        write_picks_files(result.as_of, body)
        publish_picks(result.as_of, body)
        logger.info("first publish GitHub + cache (Fit + Rank, news on rank 1-10)")

    rest = []
    if rank_result is not None:
        rest = [s.ticker for s in rank_result.signals[RANK_NEWS_TOP_K:RANK_TOP_K]]
    rest_flags = _news(rest, f"rank {RANK_NEWS_TOP_K + 1}-{RANK_TOP_K} after first publish")
    _apply_flags(rest_flags)
    if persist and rest_flags and rank_result is not None and rank_result.signals:
        rank_payload = _payload_from(rank_result, freshness)
        rank_payload["kind"] = "rank"
        _write_signals(rank_payload, rank_result.as_of)
        rank_text = format_rank_picks(rank_payload, k=RANK_TOP_K)
        fit_payload = _payload_from(result, freshness)
        _subject, fit_body = format_picks_email(fit_payload)
        publish_picks(result.as_of, rank_text + "\n" + fit_body)
        logger.info("republish GitHub after rank 11-20 news")
    return freshness, rank_result, result, rank_n, rank_text


def run_morning(threshold: float = DEFAULT_THRESHOLD, ignore_disabled: bool = False) -> int:
    started = datetime.now(CHICAGO_TIMEZONE)
    logger.info("morning start %s", started.isoformat())
    freshness = pipeline_freshness()
    logger.info("%s", freshness.detail)
    if not freshness.ready_for_inference:
        logger.warning("skipping score -- pipeline not current enough")
        subject, body = format_not_ready_email(freshness.as_of, freshness.detail)
        logger.info(
            "published=%s emailed=%s",
            publish_picks(freshness.as_of, body),
            send_email(subject, body),
        )
        return 1
    if not ignore_disabled and not TrainingConfigStore().read().morning_job_enabled:
        logger.warning("morning job disabled -- skipping scheduled run")
        return 0

    lock = _try_lock_morning()
    if lock is None:
        logger.warning("morning already running -- skipping")
        return 0

    try:
        quotes = fetch_ticker_quotes(UniverseStore().active_tickers())
        logger.info("quotes=%s", len(quotes))
        freshness, rank_result, result, rank_n, rank_text = score_from_quotes(
            quotes, threshold=threshold, persist=True
        )
        payload = _payload_from(result, freshness)
        path = DEFAULT_SIGNAL_DIR / f"{result.as_of}.json"
        subject, body = format_picks_email(payload)
        if rank_text:
            body = rank_text + "\n" + body
        published = True
        mailed = send_email(subject, body)
        logger.info(
            "scored=%s picks=%s rank_top=%s wrote %s published=%s emailed=%s",
            result.scored_count,
            len(result.signals),
            rank_n,
            path,
            published,
            mailed,
        )
        logger.info("morning done %s", datetime.now(CHICAGO_TIMEZONE).isoformat())
        return 0
    finally:
        lock.close()



def main() -> None:
    raise SystemExit(run_morning())


if __name__ == "__main__":
    main()
