from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd


def ensure_db(path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as con:
        con.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)")


def write_frame(path: str | Path, table: str, df: pd.DataFrame) -> None:
    ensure_db(path)
    with sqlite3.connect(path) as con:
        df.to_sql(table, con, if_exists="append", index=False)
