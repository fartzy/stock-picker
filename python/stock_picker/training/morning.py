"""Weekday morning scoring: today's opens through last night's model.

Runs at 8:32 CT. Rank and Fit score in parallel on one quote pull.
Rank is published as soon as Rank finishes (target ~8:35 CT).
"""

from __future__ import annotations

import functools
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime
from pathlib import Path
from stock_picker.ingestion.session import CHICAGO_TIMEZONE

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

print = functools.partial(print, flush=True)

DEFAULT_SIGNAL_DIR = data_root() / "buy_signals"


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
                print(f"rank_top={rank_n} published first")
        except Exception as exc:
            print(f"ranking skipped: {exc}")
        result = fit_future.result()

    news_tickers = [s.ticker for s in result.signals]
    if rank_result is not None:
        news_tickers.extend(s.ticker for s in rank_result.signals[:RANK_NEWS_TOP_K])
        if not persist:
            rank_n = len(rank_result.signals)
    unique_news = list(dict.fromkeys(news_tickers))
    if unique_news and news_fetcher is not None:
        print(f"news check {len(unique_news)} names (fit + rank top {RANK_NEWS_TOP_K})")
        flags = news_fetcher(unique_news, date.fromisoformat(result.as_of)) or {}
        for signal in result.signals:
            signal.news_flag = flags.get(signal.ticker)
        if rank_result is not None:
            for signal in rank_result.signals:
                signal.news_flag = flags.get(signal.ticker)
            if persist and rank_result.signals:
                rank_payload = _payload_from(rank_result, freshness)
                rank_payload["kind"] = "rank"
                _write_signals(rank_payload, rank_result.as_of)
                rank_text = format_rank_picks(rank_payload, k=RANK_TOP_K)
    return freshness, rank_result, result, rank_n, rank_text


def run_morning(threshold: float = DEFAULT_THRESHOLD, ignore_disabled: bool = False) -> int:
    started = datetime.now(CHICAGO_TIMEZONE)
    print(f"morning start {started.isoformat()}")
    freshness = pipeline_freshness()
    print(freshness.detail)
    if not freshness.ready_for_inference:
        print("skipping score -- pipeline not current enough")
        subject, body = format_not_ready_email(freshness.as_of, freshness.detail)
        print(f"published={publish_picks(freshness.as_of, body)} emailed={send_email(subject, body)}")
        return 1
    if not ignore_disabled and not TrainingConfigStore().read().morning_job_enabled:
        print("morning job disabled -- skipping scheduled run")
        return 0

    quotes = fetch_ticker_quotes(UniverseStore().active_tickers())
    print(f"quotes={len(quotes)}")
    freshness, rank_result, result, rank_n, rank_text = score_from_quotes(
        quotes, threshold=threshold, persist=True
    )
    payload = _payload_from(result, freshness)
    path = _write_signals(payload, result.as_of)
    subject, body = format_picks_email(payload)
    if rank_text:
        body = rank_text + "\n" + body
    published = publish_picks(result.as_of, body)
    mailed = send_email(subject, body)
    print(
        f"scored={result.scored_count} picks={len(result.signals)} "
        f"rank_top={rank_n} wrote {path} published={published} emailed={mailed}"
    )
    print(f"morning done {datetime.now(CHICAGO_TIMEZONE).isoformat()}")
    return 0


def main() -> None:
    raise SystemExit(run_morning())


if __name__ == "__main__":
    main()
