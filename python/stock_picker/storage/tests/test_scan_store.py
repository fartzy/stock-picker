from stock_picker.storage.scan_store import JsonScanLog, ScanStore


def test_write_then_read_round_trips_a_fit_scan(tmp_path):
    store = ScanStore(data_dir=tmp_path)
    payload = {"as_of": "2026-09-18", "kind": "fit", "signals": [{"ticker": "XENE"}]}

    store.write("2026-09-18", "fit", payload)

    loaded = store.read("2026-09-18", "fit")
    assert loaded["signals"][0]["ticker"] == "XENE"
    assert (tmp_path / "2026-09-18.json").is_file()
    assert (tmp_path / "latest.json").is_file()


def test_rank_and_fit_are_separate_rows(tmp_path):
    store = ScanStore(data_dir=tmp_path)
    store.write("2026-09-18", "fit", {"as_of": "2026-09-18", "kind": "fit", "signals": [{"ticker": "A"}]})
    store.write("2026-09-18", "rank", {"as_of": "2026-09-18", "kind": "rank", "signals": [{"ticker": "B"}]})

    assert store.read("2026-09-18", "fit")["signals"][0]["ticker"] == "A"
    assert store.read("2026-09-18", "rank")["signals"][0]["ticker"] == "B"
    assert (tmp_path / "2026-09-18-rank.json").is_file()


def test_write_replaces_same_day_kind(tmp_path):
    store = ScanStore(data_dir=tmp_path)
    store.write("2026-09-18", "fit", {"as_of": "2026-09-18", "signals": [{"ticker": "OLD"}]})
    store.write("2026-09-18", "fit", {"as_of": "2026-09-18", "signals": [{"ticker": "NEW"}]})

    assert store.read("2026-09-18", "fit")["signals"][0]["ticker"] == "NEW"


def test_read_is_none_when_missing(tmp_path):
    assert ScanStore(data_dir=tmp_path).read("2026-09-18", "fit") is None


def test_sqlite_imports_existing_json_once(tmp_path):
    json_log = JsonScanLog(tmp_path)
    json_log.write("2026-09-18", "fit", {"as_of": "2026-09-18", "kind": "fit", "signals": [{"ticker": "FLY"}]})

    store = ScanStore(data_dir=tmp_path)

    assert store.read("2026-09-18", "fit")["signals"][0]["ticker"] == "FLY"


def test_days_lists_fit_and_rank(tmp_path):
    store = ScanStore(data_dir=tmp_path)
    store.write("2026-09-17", "fit", {"as_of": "2026-09-17", "signals": []})
    store.write("2026-09-18", "rank", {"as_of": "2026-09-18", "kind": "rank", "signals": []})

    assert store.days() == ["2026-09-17", "2026-09-18"]


def test_json_backend_still_round_trips_when_injected(tmp_path):
    store = ScanStore(data_dir=tmp_path, backend=JsonScanLog(tmp_path))
    store.write("2026-09-18", "fit", {"as_of": "2026-09-18", "signals": []})

    assert (tmp_path / "scans.db").exists() is False
    assert store.read("2026-09-18", "fit")["as_of"] == "2026-09-18"
