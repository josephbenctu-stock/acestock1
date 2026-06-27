from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


def normalize_code(raw: str, default_suffix: str = ".TW") -> str:
    s = str(raw).strip().upper().replace(" ", "")
    if s.endswith(".TW") or s.endswith(".TWO"):
        return s
    m = re.search(r"(\d{4,6})", s)
    return m.group(1) + default_suffix if m else s


def clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        if "Open" in df.columns.get_level_values(0):
            df = df.droplevel(1, axis=1)
        else:
            df.columns = df.columns.get_level_values(-1)
    cols = {c: str(c).title().replace(" ", "") for c in df.columns}
    df = df.rename(columns=cols)
    needed = ["Open", "High", "Low", "Close", "Volume"]
    if not all(c in df.columns for c in needed):
        return pd.DataFrame()
    out = df[needed].dropna().copy()
    if isinstance(out.index, pd.DatetimeIndex):
        out.index = out.index.tz_localize(None) if out.index.tz is not None else out.index
    return out


def read_csv_ohlcv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Date"])
    return clean_ohlcv(df.set_index("Date"))
