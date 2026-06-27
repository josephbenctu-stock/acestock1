from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd


def load(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, encoding="utf-8-sig")


def append(path: str | Path, row: dict) -> pd.DataFrame:
    path = Path(path)
    old = load(path)
    if "trade_id" not in row:
        row = {**row, "trade_id": f"P{datetime.now():%Y%m%d%H%M%S}"}
    out = pd.concat([old, pd.DataFrame([row])], ignore_index=True) if not old.empty else pd.DataFrame([row])
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False, encoding="utf-8-sig")
    return out
