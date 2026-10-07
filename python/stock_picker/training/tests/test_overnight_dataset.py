"""Contract tests for the close-conditioned next-open dataset."""

from datetime import date

import exchange_calendars as xcals
import pandas as pd
import pytest

from stock_picker.training.overnight_dataset import (
    CurrentOpenProvenance,
    FEATURE_COLUMNS,
    LABEL_COLUMN,
    PriceContract,
    build_feature_row,
    build_overnight_training_frame,
    build_scenario_features,
    next_expected_session,
)


CONTRACT = PriceContract(source="massive", basis="raw", action_source="massive_actions")
DATES = pd.to_datetime(
    [
        "2026-09-24", "2026-09-25", "2026-09-28", "2026-09-29",
        "2026-09-30", "2026-10-01", "2026-10-02", "2026-10-05",
        "2026-10-06",
    ]
)


@pytest.fixture
def calendar():
    return xcals.get_calendar("XNYS")


def make_history(dates=DATES):
    history = pd.DataFrame({"Open": 10.0, "Close": 10.0}, index=dates)
    provenance = pd.DataFrame(
        {
            "source": CONTRACT.source,
            "basis": CONTRACT.basis,
            "action_source": CONTRACT.action_source,
            "corporate_action": "verified_none",
        },
        index=dates,
    )
    return history, provenance


def reasons(result, session):
    return {ex.reason for ex in result.exclusions if ex.session == date.fromisoformat(session)}


def test_example_features_and_label_do_not_assume_a_next_open(calendar):
    history, provenance = make_history()
    history.loc["2026-10-02", ["Open", "Close"]] = [9.0, 9.5]
    history.loc["2026-10-05", "Open"] = 9.69

    result = build_overnight_training_frame(history, provenance, CONTRACT, calendar=calendar)
    row = result.frame.loc["2026-10-02"]

    assert tuple(result.frame.columns) == (*FEATURE_COLUMNS, LABEL_COLUMN)
    assert row["prior_close"] == 10.0
    assert row["prior_return_5d"] == 0.0
    assert row["today_open"] == 9.0
    assert row["today_open_gap"] == pytest.approx(-0.10)
    assert row["assumed_close"] == 9.5
    assert row["assumed_day_return"] == pytest.approx(9.5 / 9 - 1)
    assert row[LABEL_COLUMN] == pytest.approx(0.02)
    assert row["weekday"] == 4


def test_prior_five_session_return_uses_only_completed_closes():
    row = build_feature_row([8.0, 9.0, 9.0, 9.5, 10.0, 10.0], 9.0, 9.5, date(2026, 10, 2))
    assert row["prior_return_5d"] == pytest.approx(0.25)
    assert row["assumed_day_return"] == pytest.approx(9.5 / 9.0 - 1)


def test_friday_to_monday_and_holiday_pairing(calendar):
    assert next_expected_session(date(2026, 10, 2), calendar) == date(2026, 10, 5)
    assert next_expected_session(date(2026, 9, 4), calendar) == date(2026, 9, 8)

    dates = calendar.sessions_in_range("2026-08-26", "2026-09-09")
    history, provenance = make_history(dates)
    result = build_overnight_training_frame(history, provenance, CONTRACT, calendar=calendar)
    assert date(2026, 9, 4) in result.frame.index.date
    assert result.frame.loc["2026-09-04", LABEL_COLUMN] == 0.0


def test_missing_exchange_session_is_excluded(calendar):
    history, provenance = make_history()
    history = history.drop(pd.Timestamp("2026-10-05"))
    provenance = provenance.drop(pd.Timestamp("2026-10-05"))
    result = build_overnight_training_frame(history, provenance, CONTRACT, calendar=calendar)
    assert "missing_exchange_session" in reasons(result, "2026-10-02")
    assert pd.Timestamp("2026-10-02") not in result.frame.index


def test_duplicate_session_rejects_ambiguous_history(calendar):
    history, provenance = make_history()
    history = pd.concat([history, history.iloc[[6]]]).sort_index()
    result = build_overnight_training_frame(history, provenance, CONTRACT, calendar=calendar)
    assert result.frame.empty
    assert "duplicate_session" in reasons(result, "2026-10-02")


def test_nonpositive_price_is_excluded(calendar):
    history, provenance = make_history()
    history.loc["2026-10-05", "Open"] = 0.0
    result = build_overnight_training_frame(history, provenance, CONTRACT, calendar=calendar)
    assert "nonpositive_price" in reasons(result, "2026-10-02")


def test_large_verified_gap_is_not_clipped(calendar):
    history, provenance = make_history()
    history.loc["2026-10-05", "Open"] = 24.0
    result = build_overnight_training_frame(history, provenance, CONTRACT, calendar=calendar)
    assert result.frame.loc["2026-10-02", LABEL_COLUMN] == pytest.approx(1.4)


def test_mixed_or_unknown_price_basis_is_excluded(calendar):
    history, provenance = make_history()
    provenance.loc["2026-10-05", "basis"] = "adjusted"
    mixed = build_overnight_training_frame(history, provenance, CONTRACT, calendar=calendar)
    assert "mixed_basis" in reasons(mixed, "2026-10-02")

    provenance.loc["2026-10-05", "basis"] = None
    unknown = build_overnight_training_frame(history, provenance, CONTRACT, calendar=calendar)
    assert "unknown_basis" in reasons(unknown, "2026-10-02")


def test_mixed_source_and_unproven_legacy_parquet_are_excluded(calendar):
    history, provenance = make_history()
    provenance.loc["2026-10-05", "source"] = "finnhub"
    mixed = build_overnight_training_frame(history, provenance, CONTRACT, calendar=calendar)
    assert "mixed_source" in reasons(mixed, "2026-10-02")

    legacy = build_overnight_training_frame(history, None, CONTRACT, calendar=calendar)
    assert legacy.frame.empty
    assert "missing_provenance" in reasons(legacy, "2026-10-02")


@pytest.mark.parametrize(
    ("event", "expected"),
    [("split", "split_event"), ("dividend", "dividend_event"),
     ("unverified", "unverified_corporate_actions")],
)
def test_corporate_action_discontinuity_is_excluded(calendar, event, expected):
    history, provenance = make_history()
    history.loc["2026-10-05", "Open"] = 5.0 if event == "split" else 9.75
    provenance.loc["2026-10-05", "corporate_action"] = event
    result = build_overnight_training_frame(history, provenance, CONTRACT, calendar=calendar)
    assert expected in reasons(result, "2026-10-02")


def test_unlabeled_final_row_is_reported_and_dropped(calendar):
    history, provenance = make_history()
    result = build_overnight_training_frame(history, provenance, CONTRACT, calendar=calendar)
    assert "target_not_observed" in reasons(result, "2026-10-06")
    assert pd.Timestamp("2026-10-06") not in result.frame.index
    assert result.exclusion_counts["target_not_observed"] == 1


def test_contract_metadata_pins_basis_source_and_feature_order():
    metadata = CONTRACT.artifact_metadata()
    assert metadata["basis"] == "raw"
    assert metadata["source"] == "massive"
    assert metadata["feature_columns"] == FEATURE_COLUMNS
    assert metadata["calendar"] == "XNYS"
    assert metadata["exchange_calendars_version"] == xcals.__version__


def test_training_and_scenario_features_are_identical(calendar):
    history, provenance = make_history()
    history.loc["2026-10-02", ["Open", "Close"]] = [9.0, 9.5]
    trained = build_overnight_training_frame(history, provenance, CONTRACT, calendar=calendar)
    scenario = build_scenario_features(
        history.loc[:"2026-10-01"],
        provenance.loc[:"2026-10-01"],
        session=date(2026, 10, 2),
        today_open=9.0,
        current_open_provenance=CurrentOpenProvenance(**provenance.loc["2026-10-02"].to_dict()),
        assumed_close=9.5,
        assumed_close_basis="raw",
        contract=CONTRACT,
        calendar=calendar,
    )
    assert scenario.exclusion_reason is None
    assert scenario.next_session == date(2026, 10, 5)
    pd.testing.assert_series_equal(
        trained.frame.loc["2026-10-02", list(FEATURE_COLUMNS)],
        scenario.features,
        check_names=False,
    )


@pytest.mark.parametrize(
    ("open_evidence", "assumed_basis", "expected"),
    [
        (None, "raw", "missing_current_open_provenance"),
        (CurrentOpenProvenance("massive", None, "massive_actions", "verified_none"),
         "raw", "unknown_open_basis"),
        (CurrentOpenProvenance("massive", "adjusted", "massive_actions", "verified_none"),
         "raw", "mixed_open_basis"),
        (CurrentOpenProvenance("yahoo", "raw", "massive_actions", "verified_none"),
         "raw", "mixed_open_source"),
        (CurrentOpenProvenance("massive", "raw", "massive_actions", "verified_none"),
         None, "unknown_assumed_close_basis"),
        (CurrentOpenProvenance("massive", "raw", "massive_actions", "verified_none"),
         "adjusted", "mixed_assumed_close_basis"),
    ],
)
def test_scenario_rejects_unverified_open_or_assumed_basis(
    calendar, open_evidence, assumed_basis, expected
):
    history, provenance = make_history()
    scenario = build_scenario_features(
        history.loc[:"2026-10-01"],
        provenance.loc[:"2026-10-01"],
        session=date(2026, 10, 2),
        today_open=9.0,
        current_open_provenance=open_evidence,
        assumed_close=9.5,
        assumed_close_basis=assumed_basis,
        contract=CONTRACT,
        calendar=calendar,
    )
    assert scenario.features is None
    assert scenario.exclusion_reason == expected


@pytest.mark.parametrize(("status", "expected"), [("split", "split_event"), ("dividend", "dividend_event")])
def test_scenario_rejects_current_session_corporate_action(calendar, status, expected):
    history, provenance = make_history()
    current = CurrentOpenProvenance("massive", "raw", "massive_actions", status)
    scenario = build_scenario_features(
        history.loc[:"2026-10-01"], provenance.loc[:"2026-10-01"],
        session=date(2026, 10, 2), today_open=9.0,
        current_open_provenance=current, assumed_close=9.5,
        assumed_close_basis="raw", contract=CONTRACT, calendar=calendar,
    )
    assert scenario.exclusion_reason == expected


def test_extreme_positive_prices_cannot_generate_nonfinite_features_or_label(calendar):
    history, provenance = make_history()
    history.loc["2026-09-30", "Close"] = 1e-308
    history.loc["2026-10-01", "Close"] = 1e308
    result = build_overnight_training_frame(history, provenance, CONTRACT, calendar=calendar)
    assert "nonfinite_derived_value" in reasons(result, "2026-10-02")

    history, provenance = make_history()
    history.loc["2026-10-02", "Close"] = 1e-308
    history.loc["2026-10-05", "Open"] = 1e308
    result = build_overnight_training_frame(history, provenance, CONTRACT, calendar=calendar)
    assert "nonfinite_label" in reasons(result, "2026-10-02")

    history, provenance = make_history()
    history.loc["2026-10-01", "Close"] = 1e-308
    scenario = build_scenario_features(
        history.loc[:"2026-10-01"], provenance.loc[:"2026-10-01"],
        session=date(2026, 10, 2), today_open=9.0,
        current_open_provenance=CurrentOpenProvenance(
            "massive", "raw", "massive_actions", "verified_none"
        ),
        assumed_close=1e308, assumed_close_basis="raw", contract=CONTRACT,
        calendar=calendar,
    )
    assert scenario.exclusion_reason == "nonfinite_derived_value"


@pytest.mark.parametrize("shift", ["utc", "noon"])
def test_ambiguous_session_index_is_rejected(calendar, shift):
    history, provenance = make_history()
    if shift == "utc":
        history.index = history.index.tz_localize("UTC")
    else:
        history.index = history.index + pd.Timedelta(hours=12)
    result = build_overnight_training_frame(history, provenance, CONTRACT, calendar=calendar)
    assert result.frame.empty
    assert any(ex.reason == "ambiguous_session_index" for ex in result.exclusions)

    scenario = build_scenario_features(
        history.loc[:history.index[5]], provenance,
        session=date(2026, 10, 2), today_open=9.0,
        current_open_provenance=CurrentOpenProvenance(
            "massive", "raw", "massive_actions", "verified_none"
        ),
        assumed_close=9.5, assumed_close_basis="raw", contract=CONTRACT,
        calendar=calendar,
    )
    assert scenario.exclusion_reason == "ambiguous_session_index"


def test_unknown_calendar_result_does_not_guess_weekdays():
    class UnavailableCalendar:
        name = "XNYS"

        def is_session(self, _session):
            raise RuntimeError("calendar unavailable")

    assert next_expected_session(date(2026, 10, 2), UnavailableCalendar()) is None
