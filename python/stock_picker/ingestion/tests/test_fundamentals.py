from stock_picker.ingestion.fundamentals import sector_from_info


def test_sector_from_info_prefers_sector():
    assert sector_from_info({"sector": "Technology", "industry": "Consumer Electronics"}) == "Technology"


def test_sector_from_info_skips_empty():
    assert sector_from_info({}) is None
    assert sector_from_info({"sector": "  "}) is None
