"""Small log of brokerage fees, separate from fills.

Fills stay ticker/side/shares/price/time. Fees are rare and not a P&L
column -- this JSON list is for an expandable "show fees" view.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from stock_picker.storage.paths import data_root

DEFAULT_DATA_DIR = data_root() / "fees"


@dataclass(frozen=True)
class Fee:
    ticker: str
    day: str  # YYYY-MM-DD
    side: str  # "buy" | "sell"
    amount: float
    note: str = ""


class FeeStore:
    def __init__(self, data_dir: Path | str = DEFAULT_DATA_DIR) -> None:
        self._path = Path(data_dir) / "fees.json"
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def read(self) -> list[Fee]:
        if not self._path.exists():
            return []
        raw = json.loads(self._path.read_text())
        return [Fee(**row) for row in raw]

    def append(self, fee: Fee) -> None:
        rows = self.read()
        if fee in rows:
            return
        rows.append(fee)
        self._path.write_text(json.dumps([asdict(row) for row in rows], indent=2) + "\n")
