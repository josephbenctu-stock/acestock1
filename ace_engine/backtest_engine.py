from __future__ import annotations

import pandas as pd


def summarize_trades(trades: pd.DataFrame) -> pd.DataFrame:
    if trades is None or trades.empty:
        return pd.DataFrame()
    pnl = trades.get("報酬%", pd.Series(dtype=float)).astype(float)
    return pd.DataFrame([{
        "交易筆數": len(trades),
        "勝率%": round((pnl > 0).mean() * 100, 2) if len(pnl) else 0,
        "平均報酬%": round(pnl.mean(), 2) if len(pnl) else 0,
        "最大獲利%": round(pnl.max(), 2) if len(pnl) else 0,
        "最大虧損%": round(pnl.min(), 2) if len(pnl) else 0,
    }])
