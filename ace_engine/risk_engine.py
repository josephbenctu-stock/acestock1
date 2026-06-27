from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class RiskReward:
    rr: float
    risk_pct: float
    reward_pct: float
    label: str
    decision: str
    target: float


def risk_reward(price: float, upper: float, lower: float, breakout: bool = False) -> RiskReward:
    if not all(math.isfinite(x) for x in [price, upper, lower]) or price <= 0 or upper <= lower:
        return RiskReward(float("nan"), float("nan"), float("nan"), "無法計算", "資料不足", float("nan"))
    risk_pct = max((price - lower) / price, 0.001)
    width = upper - lower
    target = upper + 0.55 * width if breakout or price >= upper else upper
    reward_pct = max((target - price) / price, 0.0)
    rr = reward_pct / risk_pct
    if rr >= 3:
        label, decision = "優秀", "優先觀察"
    elif rr >= 2:
        label, decision = "良好", "可列入候選"
    elif rr >= 1.5:
        label, decision = "普通", "需搭配型態/籌碼"
    else:
        label, decision = "不足", "不宜追價"
    return RiskReward(rr, risk_pct * 100, reward_pct * 100, label, decision, target)


def position_size_by_risk(capital: float, entry: float, stop: float, risk_pct: float = 1.0) -> int:
    """Shares allowed if the max loss is capped to risk_pct of capital."""
    if capital <= 0 or entry <= 0 or stop <= 0 or entry <= stop:
        return 0
    max_loss = capital * risk_pct / 100
    return int(max_loss // (entry - stop))
