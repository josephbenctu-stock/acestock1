from __future__ import annotations

import numpy as np
import pandas as pd


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add core indicators used by the triangle strategy.

    Required input columns: Open, High, Low, Close, Volume.
    """
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    out["MA5"] = out["Close"].rolling(5).mean()
    out["MA10"] = out["Close"].rolling(10).mean()
    out["MA20"] = out["Close"].rolling(20).mean()
    out["MA60"] = out["Close"].rolling(60).mean()
    out["VOL20"] = out["Volume"].rolling(20).mean()
    mid = out["Close"].rolling(20).mean()
    std = out["Close"].rolling(20).std()
    out["BB_UPPER"] = mid + 2 * std
    out["BB_LOWER"] = mid - 2 * std
    out["BB_WIDTH"] = (out["BB_UPPER"] - out["BB_LOWER"]) / mid
    low_n = out["Low"].rolling(9).min()
    high_n = out["High"].rolling(9).max()
    rsv = (out["Close"] - low_n) / (high_n - low_n) * 100
    out["K"] = rsv.ewm(com=2, adjust=False).mean()
    out["D"] = out["K"].ewm(com=2, adjust=False).mean()
    out.replace([np.inf, -np.inf], np.nan, inplace=True)
    return out
