"""After-close pipeline: refresh prices, rebuild features, retrain.

Meant to run once on weekday evenings so the next morning's live score has
yesterday's completed session in the feature store and a model fit on it.
Does not recap the universe (no 2000 market-cap fetch). Does not search a
new ensemble -- retrains the current DEFAULT / UI-selected composition.
"""

from __future__ import annotations

from datetime import datetime

from stock_picker.ingestion.session import CHICAGO_TIMEZONE, last_completed_session_date
from stock_picker.log import get_logger

from stock_picker.features.main import main as rebuild_features
from stock_picker.ingestion.refresh_prices import main as refresh_prices

from stock_picker.training.main import main as persist_training

logger = get_logger(__name__)


def run_nightly() -> int:
    started = datetime.now(CHICAGO_TIMEZONE)
    cutoff = last_completed_session_date()
    logger.info("nightly start %s last_completed_session=%s", started.isoformat(), cutoff)
    try:
        logger.info("=== 1/6 refresh prices (Yahoo, then Finnhub leftover bars) ===")
        refresh_prices()
        logger.info("=== 2/6 resume universe news ingest ===")
        from stock_picker.training.news_ingest import ingest_universe_news

        ingest_universe_news()
        logger.info("=== 3/6 fill missing sectors (capped Yahoo profile pull) ===")
        from stock_picker.ingestion.fundamentals import refresh_missing_sectors

        n_sectors = refresh_missing_sectors()
        logger.info("wrote sectors for %s tickers", n_sectors)
        logger.info("=== 4/6 NYC weather through yesterday ===")
        from stock_picker.ingestion.weather import refresh_nyc_weather

        refresh_nyc_weather()
        logger.info("=== 5/6 rebuild features ===")
        rebuild_features()
        logger.info("=== 6/6 retrain (return ensemble + rank if selected) ===")
        # main() persists the pickle *and* appends TrainingRunStore --
        # run_training() alone overwrites latest without a run record, so
        # pipeline_freshness still thinks the model is days behind.
        persist_training()
        logger.info("nightly done %s", datetime.now(CHICAGO_TIMEZONE).isoformat())
        return 0
    except Exception:
        logger.exception("nightly failed %s", datetime.now(CHICAGO_TIMEZONE).isoformat())
        return 1


def main() -> None:
    raise SystemExit(run_nightly())


if __name__ == "__main__":
    main()
