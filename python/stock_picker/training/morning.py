"""Weekday morning scoring: today's opens through last night's model.

Runs at 8:30:10 CT. Rank and Fit predict on one shared open-known matrix.
Rank is published as soon as Rank finishes (target ~8:35 CT).
"""

from __future__ import annotations

import fcntl
from concurrent.futures import ThreadPoolExecutor
import time
from datetime import date, datetime
from pathlib import Path

from stock_picker.ingestion.session import CHICAGO_TIMEZONE, cash_session_date, session_has_closed
from stock_picker.log import get_logger

from stock_picker.features.earnings import fetch_recent_earnings_tickers
from stock_picker.features.quotes import fetch_ticker_quotes
from stock_picker.storage.feature_store import FeatureStore
from stock_picker.storage.price_store import PriceStore
from stock_picker.storage.universe_store import UniverseStore
from stock_picker.training.news_day_judge import fetch_recent_news_flags
from stock_picker.storage.paths import data_root
from stock_picker.storage.scan_store import ScanStore
from stock_picker.training.buy_signal import (
    DEFAULT_THRESHOLD,
    OPEN_KNOWN_COLUMNS_SET,
    SCORE_BUCKET,
    SCORE_WORKERS,
    compute_buy_signals,
    compute_rank_signals,
)
from stock_picker.training.ensemble import ensemble_feature_names
from stock_picker.training.live_rows import prepare_live_rows
from stock_picker.training.freshness import pipeline_freshness
from stock_picker.training.notify import (
    format_not_ready_email,
    format_picks_email,
    format_rank_picks,
    publish_picks,
    send_email,
    write_picks_files,
)
from stock_picker.storage.model_store import ModelStore
from stock_picker.storage.training_config_store import TrainingConfigStore
from stock_picker.training.news_ingest import ingest_universe_news
from stock_picker.training.rank_model import RANK_MODEL_NAME, RANK_NEWS_TOP_K, RANK_TOP_K

logger = get_logger(__name__)

DEFAULT_SIGNAL_DIR = data_root() / "buy_signals"
_LOCK_PATH = data_root() / "buy_signals" / "morning.lock"

# News is applied in priority batches so the highest-conviction names surface
# first (viewable/published) while the rest enrich in the background. The first
# batch is deliberately tiny -- just the top few of each list -- so the first
# paint costs one parallel news wave (~5s), not one covering every pick.
RANK_FIRST_NEWS = 4
FIT_FIRST_NEWS = 2


def _try_lock_morning():
    """Non-blocking lock so 8:30 cannot start a second scan.

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


def morning_lock_held() -> bool:
    """True while launchd or another process is mid-score."""
    handle = _try_lock_morning()
    if handle is None:
        return True
    handle.close()
    return False


def load_cached_signals(
    as_of: str | None = None,
    signal_dir: Path = DEFAULT_SIGNAL_DIR,
    kind: str = "fit",
    now: datetime | None = None,
) -> dict | None:
    """Today's scan only, while the cash session is open.

    Rank writes a few minutes before Fit. Falling back to `latest` in that
    window labels Wednesday's names with Thursday's date. After the bell,
    latest is fine for looking back.
    """
    store = ScanStore(data_dir=signal_dir)
    if as_of:
        return store.read(as_of, kind)
    today = cash_session_date(now).isoformat()
    payload = store.read(today, kind)
    if payload:
        return payload
    if not session_has_closed(now):
        return None
    return store.latest(kind)


def _write_signals(payload: dict, as_of: str, signal_dir: Path = DEFAULT_SIGNAL_DIR) -> Path:
    kind = payload.get("kind") or ("rank" if as_of.endswith("-rank") else "fit")
    day = payload.get("as_of") or as_of.removesuffix("-rank")
    ScanStore(data_dir=signal_dir).write(day, kind, payload)
    name = f"{day}-rank.json" if kind == "rank" else f"{day}.json"
    return signal_dir / name


def _payload_from(result, freshness, model_run_id: str | None = None) -> dict:
    from stock_picker.storage.training_config_store import TrainingConfigStore

    run_id = model_run_id
    if run_id is None:
        run_id = TrainingConfigStore().read().selected_run_id
    return {
        "as_of": result.as_of,
        "model_run_id": run_id,
        "threshold": result.threshold,
        "signals": [
            {
                "ticker": signal.ticker,
                "predicted_return": signal.predicted_return,
                "open_price": signal.open_price,
                "snapshot_date": signal.snapshot_date,
                "news_flag": signal.news_flag,
                "news_checked": signal.news_checked,
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
    as_of: date | None = None,
    model_run_id: str | None = None,
    on_progress=None,
):
    """Build open-known rows once, then Rank + Fit only predict.

    Same path the 8:30 job and Test run use. persist=False skips cache/git/email
    so a test run does not overwrite Monday's files.

    `on_progress(rank_result, fit_result)`, if given, fires after every news
    batch -- the Test run job uses it to surface partial picks (top few
    news-checked) for viewing while later batches keep judging. The real
    morning path passes nothing and instead surfaces the same staging via its
    existing GitHub/cache publish points below.
    """
    t0 = time.perf_counter()
    freshness = pipeline_freshness()
    as_of = as_of or date.today()
    from stock_picker.storage.ticker_blacklist_store import blacklisted_tickers

    blocked = blacklisted_tickers()
    tickers = [ticker for ticker in UniverseStore().active_tickers() if ticker not in blocked]
    logger.info("score start names=%s blocked=%s persist=%s", len(tickers), sorted(blocked), persist)
    try:
        earnings = earnings_fetcher(tickers, as_of) or set()
    except TypeError:
        earnings = earnings_fetcher(tickers) or set()
    logger.info("earnings %.1fs n=%s", time.perf_counter() - t0, len(earnings))
    t_rows = time.perf_counter()
    spy_quote = quotes.get("SPY")
    spy_open = spy_quote.get("open") if spy_quote else None
    spy_prev_close = spy_quote.get("prev_close") if spy_quote else None

    from stock_picker.training.main import MODEL_NAME

    # Mirrors compute_buy_signals' own model_name resolution exactly (a
    # given model_run_id wins; otherwise fall back to the globally selected
    # live run, same as passing model_name=None would) -- fit_model_name
    # must name the SAME model _fit() below will actually load, or the
    # excluded-features derivation would be for the wrong model.
    fit_model_name = (
        f"{MODEL_NAME}_{model_run_id}"
        if model_run_id
        else (
            f"{MODEL_NAME}_{selected_run_id}"
            if (selected_run_id := TrainingConfigStore().read().selected_run_id)
            else MODEL_NAME
        )
    )
    model_store = ModelStore()
    # Union, not either alone: this row matrix is shared, so a column either
    # model's own feature_names actually needs must be computed -- excluding
    # it here would reindex that model to NaN for it (see
    # ensemble.ensemble_feature_names). Derived from these two SPECIFIC
    # loaded models, never the current global pruned-features state, for the
    # same reason. Missing model -> contributes nothing, handled the same as
    # "not persisted yet" further down in compute_buy_signals/compute_rank_signals.
    needed_features: set[str] = set()
    for name in (fit_model_name, RANK_MODEL_NAME):
        if model_store.exists(name):
            needed_features |= ensemble_feature_names(model_store.read(name))
    excluded_features = OPEN_KNOWN_COLUMNS_SET - needed_features if needed_features else None

    live_rows = prepare_live_rows(
        tickers=tickers,
        quotes=quotes,
        earnings=earnings,
        feature_store=FeatureStore(),
        price_store=PriceStore(),
        as_of=as_of,
        spy_open=spy_open,
        spy_prev_close=spy_prev_close,
        bucket_size=SCORE_BUCKET,
        workers=SCORE_WORKERS,
        excluded_features=excluded_features,
    )
    scored = sum(1 for row in live_rows if row.row is not None)
    logger.info("live_rows %.1fs scored=%s skipped=%s", time.perf_counter() - t_rows, scored, len(live_rows) - scored)

    def quote_fetcher(names, as_of=None):
        return quotes

    def _rank():
        return compute_rank_signals(
            top_k=RANK_TOP_K,
            as_of=as_of,
            quote_fetcher=quote_fetcher,
            earnings_fetcher=None,
            news_fetcher=None,
            live_rows=live_rows,
        )

    def _fit():
        return compute_buy_signals(
            threshold=threshold,
            as_of=as_of,
            quote_fetcher=quote_fetcher,
            earnings_fetcher=None,
            news_fetcher=None,
            live_rows=live_rows,
            model_name=fit_model_name,
        )

    rank_result = None
    rank_n = 0
    rank_text = ""
    t_pred = time.perf_counter()
    with ThreadPoolExecutor(max_workers=2) as pool:
        rank_future = pool.submit(_rank)
        fit_future = pool.submit(_fit)
        try:
            rank_result = rank_future.result()
            if persist and rank_result.signals:
                rank_payload = _payload_from(rank_result, freshness, model_run_id)
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
    logger.info(
        "predict %.1fs rank=%s fit=%s",
        time.perf_counter() - t_pred,
        0 if rank_result is None else len(rank_result.signals),
        len(result.signals),
    )

    if not persist and rank_result is not None:
        rank_n = len(rank_result.signals)

    def _apply_news(checked: set[str], flags: dict[str, str]) -> None:
        # Mark every judged ticker news_checked (so a blank News column reads
        # "clear", not "still checking") and attach a flag where one was found.
        # Applies to both lists so a name in fit and rank updates in both.
        signals = list(result.signals)
        if rank_result is not None:
            signals.extend(rank_result.signals)
        for signal in signals:
            if signal.ticker in checked:
                signal.news_checked = True
                if signal.ticker in flags:
                    signal.news_flag = flags[signal.ticker]

    def _news(tickers, label):
        names = list(dict.fromkeys(tickers))
        if not names or news_fetcher is None:
            return {}
        logger.info("news check %s names (%s)", len(names), label)
        t_news = time.perf_counter()
        flags = news_fetcher(names, date.fromisoformat(result.as_of)) or {}
        logger.info("news %s %.1fs flagged=%s", label, time.perf_counter() - t_news, len(flags))
        return flags

    def _publish(first: bool) -> None:
        """Write + publish the current picks (news flags applied so far). The
        first call also writes the local cache files; later calls only
        republish the GitHub/email body after news changed something."""
        nonlocal rank_text
        if rank_result is not None and rank_result.signals:
            rank_payload = _payload_from(rank_result, freshness, model_run_id)
            rank_payload["kind"] = "rank"
            _write_signals(rank_payload, rank_result.as_of)
            rank_text = format_rank_picks(rank_payload, k=RANK_TOP_K)
        fit_payload = _payload_from(result, freshness, model_run_id)
        _write_signals(fit_payload, result.as_of)
        _subject, fit_body = format_picks_email(fit_payload)
        body = (rank_text + "\n" + fit_body) if rank_text else fit_body
        if first:
            write_picks_files(result.as_of, body)
        publish_picks(result.as_of, body)

    rank_signals = rank_result.signals if rank_result is not None else []
    fit_signals = result.signals

    def _names(*groups) -> list[str]:
        names: list[str] = []
        for group in groups:
            names.extend(s.ticker for s in group)
        return names

    # Priority batches: the smallest, highest-conviction set first so it can be
    # viewed/published after a single parallel news wave, then the remainder.
    stages = [
        (
            f"first paint (rank 1-{RANK_FIRST_NEWS} + fit 1-{FIT_FIRST_NEWS})",
            _names(rank_signals[:RANK_FIRST_NEWS], fit_signals[:FIT_FIRST_NEWS]),
        ),
        (
            f"rest of fit + rank {RANK_FIRST_NEWS + 1}-{RANK_NEWS_TOP_K}",
            _names(fit_signals[FIT_FIRST_NEWS:], rank_signals[RANK_FIRST_NEWS:RANK_NEWS_TOP_K]),
        ),
        (
            f"rank {RANK_NEWS_TOP_K + 1}-{RANK_TOP_K}",
            _names(rank_signals[RANK_NEWS_TOP_K:RANK_TOP_K]),
        ),
    ]

    seen: set[str] = set()
    for index, (label, names) in enumerate(stages):
        fresh = [ticker for ticker in dict.fromkeys(names) if ticker not in seen]
        seen.update(fresh)
        flags = _news(fresh, label) if fresh else {}
        _apply_news(set(fresh), flags)
        if persist:
            # Stage 0 is the first publish (unconditional, gets the picks out);
            # later stages only republish if their news actually changed a flag.
            if index == 0:
                _publish(first=True)
                logger.info("first publish GitHub + cache (%s)", label)
            elif flags:
                _publish(first=False)
                logger.info("republish GitHub after %s news", label)
        if on_progress is not None:
            on_progress(rank_result, result)
    logger.info("score done %.1fs", time.perf_counter() - t0)
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
    if not ignore_disabled and load_cached_signals(kind="rank"):
        logger.info("today's rank already on disk -- skipping scheduled run")
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
    finally:
        lock.close()

    try:
        ingest_universe_news()
    except Exception:
        logger.exception("universe news ingest failed")
    return 0


def main() -> None:
    raise SystemExit(run_morning())


if __name__ == "__main__":
    main()
