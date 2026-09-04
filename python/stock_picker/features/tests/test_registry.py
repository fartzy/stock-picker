from datetime import date

from stock_picker.features.catalog import list_feature_columns
from stock_picker.features.registry import DEFAULT_TTL_DAYS, build_registry, check_freshness
from stock_picker.features.tests.fixtures import synthetic_history


def test_build_registry_has_one_feature_view_per_category():
    history = synthetic_history(n=140)
    catalog = list_feature_columns(history)

    feature_views, _ = build_registry(history)

    assert {v.name for v in feature_views} == set(catalog)
    for view in feature_views:
        assert view.features == catalog[view.name]
        assert view.entities == ["ticker"]


def test_build_registry_feature_service_references_every_view():
    history = synthetic_history(n=140)

    feature_views, feature_services = build_registry(history)

    assert len(feature_services) == 1
    service = feature_services[0]
    assert set(service.feature_views) == {v.name for v in feature_views}


def test_cross_sectional_view_is_tagged_cross_ticker():
    history = synthetic_history(n=140)

    feature_views, _ = build_registry(history)

    cross_sectional = next(v for v in feature_views if v.name == "cross_sectional")
    assert cross_sectional.tags.get("cross_ticker") == "true"


def test_check_freshness_ok_within_ttl():
    history = synthetic_history(n=140)
    feature_views, _ = build_registry(history)
    momentum_view = next(v for v in feature_views if v.name == "momentum")

    result = check_freshness(
        momentum_view, snapshot_date=date(2026, 1, 1), as_of_date=date(2026, 1, 2)
    )

    assert result.ok
    assert result.age_days == 1
    assert result.ttl_days == DEFAULT_TTL_DAYS


def test_check_freshness_stale_beyond_ttl():
    history = synthetic_history(n=140)
    feature_views, _ = build_registry(history)
    momentum_view = next(v for v in feature_views if v.name == "momentum")

    result = check_freshness(
        momentum_view, snapshot_date=date(2026, 1, 1), as_of_date=date(2026, 1, 5)
    )

    assert not result.ok
    assert result.age_days == 4
