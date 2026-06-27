from __future__ import annotations

from datetime import datetime
from typing import Iterable

import pandas as pd


def markdown_report(results: Iterable[dict], title: str = "艾斯三角收斂 V7 策略報告") -> str:
    rows = list(results or [])
    lines = [f"# {title}", "", f"產生時間：{datetime.now():%Y-%m-%d %H:%M}", ""]
    if not rows:
        lines.append("目前沒有候選股。")
        return "\n".join(lines)
    df = pd.DataFrame(rows)
    lines += [f"候選股數：{len(df)}", ""]
    cols = [c for c in ["代號", "名稱", "評分", "等級", "動作", "型態", "型態品質", "風報比"] if c in df.columns]
    lines += ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.head(20).iterrows():
        lines.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    return "\n".join(lines)
