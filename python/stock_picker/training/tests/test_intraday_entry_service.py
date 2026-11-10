from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from stock_picker.storage.model_store import ModelStore
from stock_picker.ingestion.session import cash_session_window
from stock_picker.training.intraday_entry_service import IntradayEntryService, entry_time_reason


NOW = datetime(2026, 11, 10, 15, 0, tzinfo=timezone.utc)
OPEN = NOW.replace(hour=14, minute=30)


class FakeModel:
    metadata = {"trained_through": "2026-10-01", "dataset": {"checkpoints_et": ["09:35", "15:00"]}}

    def __init__(self):
        self.inputs = []

    def predict(self, frame):
        self.inputs.append(frame)
        return {"probability_profit": np.array([.6]), "expected_return": np.array([.01]),
                "probability_loss_2pct": np.array([.1])}


class Client:
    def __init__(self, observed=None):
        self.calls = 0
        self.tickers = []
        self.observed = observed or NOW - timedelta(seconds=30)

    def fetch_current_open(self, ticker, day):
        self.calls += 1
        self.tickers.append(ticker)
        return SimpleNamespace(open=100., previous_close=101., last_trade=100.,
                               last_trade_at=self.observed, corporate_action="verified_none")

    def fetch_minute_bars(self, ticker, day):
        self.tickers.append(ticker)
        return [{"t": int((OPEN+timedelta(minutes=i)).timestamp()*1000),
                 "o": 100., "h": 100. if i < 29 else 500., "l": 100.,
                 "c": 100. if i < 29 else 500., "v": 100., "vw": 100.}
                for i in range(40)]


def service(tmp_path, *, clock=lambda: NOW, client=None):
    client = client or Client()
    scan = {"signals": [{"ticker": "AAA", "predicted_return": .5},
                        {"ticker": "BBB", "predicted_return": .2}], "model_run_id": "morning-test"}
    result = IntradayEntryService(model_store=ModelStore(tmp_path),
        scan_store=SimpleNamespace(read=lambda *a, **kw: scan),
        feature_store=SimpleNamespace(read_columns=lambda *args: pd.DataFrame()),
        client_factory=lambda: client, clock=clock)
    model = FakeModel()
    result._load = lambda: model
    return result, client, model


def test_asof_price_and_completed_bars_cache_and_rank_context(tmp_path):
    entry, client, model = service(tmp_path)
    result = entry.predict(["aaa"])
    row = result["predictions"][0]
    assert row["status"] == "ok", row
    assert row["rank"] == 1
    assert row["entry_price"] == 100
    assert row["observed_at"] == (NOW-timedelta(seconds=30)).isoformat()
    # 14:59's eventual 500 high/close is unavailable at 14:59:30.
    assert model.inputs[0].iloc[0]["drawdown_from_high_so_far"] == 0
    assert model.inputs[0].iloc[0]["morning_score_z"] == pytest.approx(1)
    entry.predict(["AAA"])
    assert client.calls == 1


def test_stale_and_future_prices_are_unavailable(tmp_path):
    for observed in (NOW-timedelta(minutes=3), NOW+timedelta(seconds=1)):
        entry, _, model = service(tmp_path, client=Client(observed))
        result = entry.predict(["AAA"])
        assert result["predictions"][0]["status"] == "unavailable"
        assert not model.inputs


def test_closed_session_missing_rank_and_limits_never_make_up_odds(tmp_path):
    entry, client, model = service(tmp_path, clock=lambda: NOW.replace(hour=22))
    assert entry.predict(["AAA"])["predictions"][0]["status"] == "unavailable"
    assert client.calls == 0
    entry, client, model = service(tmp_path)
    assert entry.predict(["UNKNOWN"])["predictions"][0]["status"] == "unavailable"
    assert client.calls == 0
    with pytest.raises(ValueError, match="20 tickers"):
        entry.predict(["AAA"]*21)
    with pytest.raises(ValueError, match="valid ticker"):
        entry.predict(["../secrets"])


def test_prior_features_only_exact_previous_session(tmp_path):
    entry, _, _ = service(tmp_path)
    frame = pd.DataFrame({"return_1d": [.1, .2, 999.]},
        index=pd.to_datetime(["2026-11-06", "2026-11-09", "2026-11-10"]))
    entry._features = SimpleNamespace(read_columns=lambda *args: frame)
    assert entry._prior_features("AAA", "2026-11-10") == {"return_1d": .2}


def test_provider_exception_does_not_expose_url_or_secret(tmp_path):
    entry, client, _ = service(tmp_path)
    def failed(*args):
        raise ValueError("https://provider?apiKey=SECRET")
    client.fetch_current_open = failed
    result = entry.predict(["AAA"])
    assert result["predictions"][0]["status"] == "unavailable"
    assert "SECRET" not in str(result)


def test_class_shares_use_provider_alias_but_keep_original_ui_symbol(tmp_path):
    entry, client, _ = service(tmp_path)
    entry._scans = SimpleNamespace(read=lambda *a, **kw: {
        "signals": [{"ticker": "BRK-B", "predicted_return": .5}], "model_run_id": "test"})
    row = entry.predict(["BRK-B"])["predictions"][0]
    assert row["status"] == "ok", row
    assert row["ticker"] == "BRK-B"
    assert client.tickers == ["BRK.B", "BRK.B"]


def test_missing_latest_minute_has_actionable_safe_reason(tmp_path):
    entry, client, _ = service(tmp_path)
    bars = client.fetch_minute_bars("AAA", NOW.date())
    missing = int((OPEN+timedelta(minutes=28)).timestamp()*1000)
    client.fetch_minute_bars = lambda *args: [b for b in bars if b["t"] != missing]
    row = entry.predict(["AAA"])["predictions"][0]
    assert row["status"] == "unavailable"
    assert "latest completed minute is missing" in row["reason"]


@pytest.mark.parametrize("stamp, supported", [
    ("2026-11-10T14:34:59+00:00", False),  # Before 09:35 ET.
    ("2026-11-10T14:35:00+00:00", True),
    ("2026-11-10T20:00:00+00:00", True),   # 15:00 ET, one hour left.
    ("2026-11-10T20:00:01+00:00", False),
    ("2026-11-27T17:00:00+00:00", True),   # Early close: 12:00 ET, one hour left.
    ("2026-11-27T17:00:01+00:00", False),
])
def test_supported_time_window_and_early_close_horizon(stamp, supported):
    observed = datetime.fromisoformat(stamp)
    reason = entry_time_reason(FakeModel(), observed, cash_session_window(observed))
    assert (reason is None) == supported
    if reason:
        assert "09:35–15:00 ET" in reason
        assert "60 minutes" in reason


def test_supported_request_clock_cannot_admit_an_earlier_unsupported_trade(tmp_path):
    now = OPEN + timedelta(minutes=5)
    entry, _, model = service(tmp_path, clock=lambda: now,
                              client=Client(now-timedelta(seconds=1)))
    row = entry.predict(["AAA"])["predictions"][0]
    assert row["status"] == "unavailable"
    assert "09:35–15:00 ET" in row["reason"]
    assert not model.inputs
