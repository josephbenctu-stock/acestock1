from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd


@dataclass
class TriangleQuality:
    score: int
    label: str
    details: List[str]
    warnings: List[str]


def _body_high(df: pd.DataFrame) -> pd.Series:
    return pd.concat([df["Open"], df["Close"]], axis=1).max(axis=1)


def _body_low(df: pd.DataFrame) -> pd.Series:
    return pd.concat([df["Open"], df["Close"]], axis=1).min(axis=1)


def simple_triangle_lines(df: pd.DataFrame, lookback: int = 45) -> dict:
    """Lightweight deterministic triangle approximation for tests/CLI.

    The full Streamlit app has a richer pivot-based engine; this function is
    intentionally small and stable for unit tests and scheduled scripts.
    """
    if df is None or len(df) < lookback:
        return {"valid": False, "reason": "資料不足"}
    w = df.tail(lookback).copy()
    x = np.arange(len(w))
    hi = _body_high(w).to_numpy(dtype=float)
    lo = _body_low(w).to_numpy(dtype=float)
    up = np.polyfit(x, hi, 1)
    dn = np.polyfit(x, lo, 1)
    upper = up[0] * x + up[1]
    lower = dn[0] * x + dn[1]
    price = float(w["Close"].iloc[-1])
    width_now = (upper[-1] - lower[-1]) / max(price, 0.01)
    width_start = (upper[0] - lower[0]) / max(float(w["Close"].iloc[0]), 0.01)
    inside = ((hi <= upper * 1.02) & (lo >= lower * 0.98)).mean()
    valid = bool(upper[-1] > lower[-1] and width_now < 0.25 and inside >= 0.70 and width_now < width_start)
    if up[0] < 0 and dn[0] > 0:
        pattern = "對稱三角"
    elif abs(up[0]) <= abs(dn[0]) * 0.35 and dn[0] >= 0:
        pattern = "上升三角"
    elif abs(dn[0]) <= abs(up[0]) * 0.35 and up[0] <= 0:
        pattern = "下降三角"
    else:
        pattern = "疑似收斂"
    return {
        "valid": valid,
        "pattern": pattern,
        "upper": float(upper[-1]),
        "lower": float(lower[-1]),
        "width_now": float(width_now),
        "width_start": float(width_start),
        "inside_ratio": float(inside),
        "upper_slope": float(up[0]),
        "lower_slope": float(dn[0]),
    }


def quality_from_lines(lines: dict) -> TriangleQuality:
    if not lines or not lines.get("valid"):
        return TriangleQuality(0, "未形成", [], [lines.get("reason", "三角條件不足") if lines else "三角條件不足"])
    score = 0
    details: List[str] = []
    warnings: List[str] = []
    inside = float(lines.get("inside_ratio", 0))
    width_now = float(lines.get("width_now", 9))
    width_start = float(lines.get("width_start", 9))
    if inside >= 0.88:
        score += 35; details.append("線內比例高")
    elif inside >= 0.75:
        score += 22; details.append("線內比例尚可")
    else:
        warnings.append("線內比例偏低")
    if width_now <= 0.16:
        score += 30; details.append("尾端寬度收斂")
    elif width_now <= 0.25:
        score += 18; details.append("尾端寬度尚可")
    if width_start > 0 and width_now / width_start <= 0.65:
        score += 25; details.append("收斂幅度明顯")
    else:
        warnings.append("收斂幅度不夠明確")
    if lines.get("pattern") in ["上升三角", "對稱三角"]:
        score += 10
    score = int(max(0, min(100, score)))
    label = "優秀" if score >= 82 else "良好" if score >= 68 else "普通" if score >= 52 else "偏弱"
    return TriangleQuality(score, label, details, warnings)
