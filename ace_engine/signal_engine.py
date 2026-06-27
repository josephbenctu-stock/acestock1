from __future__ import annotations


def infer_stage(flags: dict, triangle_valid: bool, maturity: str) -> str:
    if flags.get("stop_unbroken") or flags.get("stop_after_break"):
        return "停損/風險檢查"
    if flags.get("bb_leave_upper") or flags.get("long_upper_black"):
        return "獲利出場觀察"
    if flags.get("pullback_upper"):
        return "回測上線加碼候選"
    if flags.get("breakout"):
        return "突破加碼候選"
    if flags.get("small_k_vol_contract") or flags.get("support_touch"):
        return "基本單/試單候選"
    if flags.get("touch_60ma") and flags.get("kd_cross"):
        return "早期試單候選"
    if triangle_valid and maturity == "收斂尾端":
        return "觀察：等量縮/突破"
    if triangle_valid:
        return "觀察清單"
    return "未達條件"
