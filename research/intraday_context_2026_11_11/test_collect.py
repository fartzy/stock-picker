from datetime import datetime, timezone
import json

import pandas as pd
import pytest

from research.intraday_context_2026_11_11 import collect


def test_bounded_plan_covers_lookbacks_without_overlaps():
    frame = pd.DataFrame({"ticker": ["A", "A"], "session": ["2026-10-01", "2026-10-20"]})
    plan = collect.plan_ranges(frame)
    for ticker in ("A", "SPY"):
        ranges = [(start, end) for name, start, end in plan if name == ticker]
        assert ranges[0][0] == "2026-08-17"
        assert ranges[-1][1] == "2026-10-20"
        for start, end in ranges:
            assert (collect.date.fromisoformat(end)-collect.date.fromisoformat(start)).days <= 27
        for previous, current in zip(ranges, ranges[1:]):
            assert collect.date.fromisoformat(current[0])-collect.date.fromisoformat(previous[1]) == collect.timedelta(days=1)


def test_collection_is_resumable_checksummed_and_ignores_no_failures(tmp_path, monkeypatch):
    monkeypatch.setattr(collect, "ROOT", tmp_path)
    calls = []
    stamp = int(datetime(2026, 10, 1, 14, tzinfo=timezone.utc).timestamp()*1000)
    class Client:
        def _pages(self, path, params):
            calls.append(path)
            yield {"status": "OK", "ticker": "A", "adjusted": False,
                   "results": [dict(t=stamp, o=10, h=11, l=9, c=10, vw=10, v=100)]}
    monkeypatch.setattr(collect, "MassiveOvernightClient", Client)
    spec = ("A", "2026-10-01", "2026-10-01")
    first = collect.collect_range(spec, tmp_path / "sources")
    second = collect.collect_range(spec, tmp_path / "sources")
    assert not first["cached"] and second["cached"] and len(calls) == 1
    pointer = tmp_path / first["pointer"]
    document = collect.read_archive(pointer)
    assert document["bars"][0]["t"] == stamp and document["complete"]
    archive = pointer.parent / json.loads(pointer.read_text())["archive"]
    archive.write_bytes(b"damaged")
    with pytest.raises(ValueError, match="checksum"):
        collect.collect_range(spec, tmp_path / "sources")
    assert len(calls) == 1


def test_bad_payload_retries_without_certifying_complete(tmp_path, monkeypatch):
    monkeypatch.setattr(collect, "ROOT", tmp_path)
    monkeypatch.setattr(collect.time, "sleep", lambda _: None)
    calls = []
    class Client:
        def _pages(self, path, params):
            calls.append(path)
            yield {"status": "ERROR", "ticker": "A", "adjusted": False}
    monkeypatch.setattr(collect, "MassiveOvernightClient", Client)
    result = collect.collect_range(("A", "2026-10-01", "2026-10-01"), tmp_path / "sources")
    assert result["error_type"] == "ValueError" and len(calls) == 3
    assert not (tmp_path / "sources").exists()
