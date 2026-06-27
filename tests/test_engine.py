from __future__ import annotations

from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from ace_engine.indicators import add_indicators
from ace_engine.triangle_engine import simple_triangle_lines, quality_from_lines
from ace_engine.risk_engine import risk_reward, position_size_by_risk
from ace_engine.signal_engine import infer_stage
from ace_engine.paper_trading import append, load


def synthetic_triangle(n: int = 120) -> pd.DataFrame:
    idx = pd.date_range("2025-01-01", periods=n, freq="B")
    x = np.arange(n)
    center = 50 + 0.03 * x
    amp = np.linspace(10, 2, n)
    close = center + np.sin(x / 3) * amp * 0.22
    open_ = close + np.cos(x / 4) * 0.25
    high = np.maximum(open_, close) + amp * 0.50
    low = np.minimum(open_, close) - amp * 0.50
    volume = np.linspace(2000000, 900000, n)
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume}, index=idx)


def test_indicators_and_triangle_quality():
    df = add_indicators(synthetic_triangle())
    assert {"MA20", "MA60", "BB_UPPER", "K", "D"}.issubset(df.columns)
    lines = simple_triangle_lines(df.dropna(), lookback=45)
    assert lines["upper"] > lines["lower"]
    q = quality_from_lines(lines)
    assert 0 <= q.score <= 100


def test_risk_reward_and_position_size():
    rr = risk_reward(price=50, upper=55, lower=48)
    assert rr.rr > 1
    assert rr.risk_pct > 0
    assert position_size_by_risk(300000, 50, 48, 1.0) == 1500


def test_signal_stage():
    assert infer_stage({"breakout": True}, True, "收斂尾端") == "突破加碼候選"
    assert infer_stage({"support_touch": True}, True, "收斂尾端") == "基本單/試單候選"
    assert infer_stage({"stop_unbroken": True}, True, "收斂尾端") == "停損/風險檢查"


def test_paper_trading_append_load():
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "paper.csv"
        append(path, {"ticker": "2330.TW", "entry_price": 100, "shares": 1000})
        df = load(path)
        assert len(df) == 1
        assert df.iloc[0]["ticker"] == "2330.TW"
