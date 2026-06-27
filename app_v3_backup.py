# -*- coding: utf-8 -*-
"""
ACE Triangle Strategy Stock Screener V3
A mobile-friendly Streamlit app that screens Taiwan stocks for triangle-contraction patterns.

This program is an educational implementation of the rules described in the uploaded lecture.
It is NOT financial advice. Always verify charts manually before trading.
"""
from __future__ import annotations

import io
import re
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

try:
    import yfinance as yf
except Exception:  # pragma: no cover
    yf = None


# -----------------------------
# Page / Mobile UI
# -----------------------------
st.set_page_config(
    page_title="艾斯三角收斂選股器 V3",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 1rem; padding-bottom: 2rem;}
    div[data-testid="stMetricValue"] {font-size: 1.35rem;}
    .small-note {font-size: 0.85rem; opacity: 0.78;}
    .signal-card {
        padding: 0.85rem 1rem; border: 1px solid rgba(128,128,128,.25);
        border-radius: 14px; margin: .35rem 0 .75rem 0;
        background: rgba(128,128,128,.04);
    }
    .pill {display:inline-block; padding:.18rem .45rem; border-radius:999px; border:1px solid rgba(128,128,128,.35); margin:.1rem .15rem .1rem 0; font-size:.82rem;}
    .gradeA {font-weight: 700;}
    @media (max-width: 640px) {
        .block-container {padding-left: .65rem; padding-right: .65rem;}
        [data-testid="stSidebar"] {width: 88vw !important;}
        .stDataFrame {font-size: 0.75rem;}
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------
# Data structures
# -----------------------------
@dataclass
class TriangleResult:
    is_valid: bool
    pattern: str
    score: int
    upper_line: Optional[pd.Series]
    lower_line: Optional[pd.Series]
    upper_today: Optional[float]
    lower_today: Optional[float]
    width_ratio: Optional[float]
    inside_ratio: Optional[float]
    upper_change_pct: Optional[float]
    lower_change_pct: Optional[float]
    touch_upper: int
    touch_lower: int
    maturity: str
    tags: List[str]
    reasons: List[str]
    warnings: List[str]


# -----------------------------
# Helpers
# -----------------------------
def normalize_code(raw: str, default_suffix: str = ".TW") -> str:
    s = str(raw).strip().upper().replace(" ", "")
    if not s:
        return ""
    if s.endswith(".TW") or s.endswith(".TWO"):
        return s
    m = re.search(r"(\d{4,6})", s)
    if not m:
        return ""
    return m.group(1) + default_suffix


def pure_code(ticker: str) -> str:
    return str(ticker).upper().replace(".TW", "").replace(".TWO", "")


def safe_float(x, default=np.nan) -> float:
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def clean_yf_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        if "Open" in df.columns.get_level_values(0):
            df = df.droplevel(1, axis=1)
        elif "Open" in df.columns.get_level_values(-1):
            df.columns = df.columns.get_level_values(-1)
    df = df.copy()
    rename = {c: str(c).title().replace(" ", "") for c in df.columns}
    df = df.rename(columns=rename)
    needed = ["Open", "High", "Low", "Close", "Volume"]
    if not all(c in df.columns for c in needed):
        return pd.DataFrame()
    df = df[needed].dropna()
    df = df[df["Volume"] >= 0]
    if isinstance(df.index, pd.DatetimeIndex):
        df.index = df.index.tz_localize(None) if df.index.tz is not None else df.index
    return df


@st.cache_data(ttl=60 * 60 * 12, show_spinner=False)
def fetch_ohlcv(ticker: str, period: str = "18mo") -> pd.DataFrame:
    if yf is None:
        return pd.DataFrame()
    try:
        df = yf.download(
            ticker,
            period=period,
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
        )
        return clean_yf_df(df)
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=60 * 60 * 24, show_spinner=False)
def load_tw_universe() -> pd.DataFrame:
    """Fetch listed and OTC common-stock tickers from the TWSE ISIN page, including industry when available."""
    rows: List[Tuple[str, str, str, str, str]] = []
    modes = [
        ("上市", ".TW", "https://isin.twse.com.tw/isin/C_public.jsp?strMode=2"),
        ("上櫃", ".TWO", "https://isin.twse.com.tw/isin/C_public.jsp?strMode=4"),
    ]
    for market, suffix, url in modes:
        try:
            tables = pd.read_html(url, encoding="big5")
            if not tables:
                continue
            df = tables[0].copy()
            if df.shape[0] > 2:
                maybe_header = df.iloc[0].astype(str).tolist()
                if any("有價證券" in x or "代號" in x for x in maybe_header):
                    df.columns = maybe_header
                    df = df.iloc[1:].reset_index(drop=True)
            first_col = df.columns[0]
            industry_col = next((c for c in df.columns if "產業" in str(c)), None)
            extracted = df[first_col].astype(str).str.extract(r"^(\d{4})\s+(.+)$")
            for idx, row in df.iterrows():
                text = str(row[first_col])
                m = re.match(r"^(\d{4})\s+(.+)$", text)
                if not m:
                    continue
                code, name = m.group(1).strip(), m.group(2).strip()
                if not re.fullmatch(r"[1-9]\d{3}", code):
                    continue
                industry = "未分類"
                if industry_col is not None:
                    raw_ind = str(row.get(industry_col, "")).strip()
                    if raw_ind and raw_ind.lower() not in ["nan", "none"]:
                        industry = raw_ind
                rows.append((code, name, market, industry, code + suffix))
        except Exception:
            continue
    if not rows:
        rows = [
            ("2330", "台積電", "上市", "半導體業", "2330.TW"),
            ("2317", "鴻海", "上市", "其他電子業", "2317.TW"),
            ("2454", "聯發科", "上市", "半導體業", "2454.TW"),
            ("2308", "台達電", "上市", "電子零組件業", "2308.TW"),
            ("6488", "環球晶", "上櫃", "半導體業", "6488.TWO"),
        ]
    out = pd.DataFrame(rows, columns=["code", "name", "market", "industry", "ticker"])
    return out.drop_duplicates("ticker").sort_values(["market", "industry", "code"]).reset_index(drop=True)


def parse_chip_csv(uploaded_file) -> pd.DataFrame:
    if uploaded_file is None:
        return pd.DataFrame()
    try:
        raw = uploaded_file.read()
        df = None
        for enc in ["utf-8-sig", "big5", "cp950", "utf-8"]:
            try:
                df = pd.read_csv(io.BytesIO(raw), encoding=enc)
                break
            except Exception:
                df = None
        if df is None or df.empty:
            return pd.DataFrame()
        df.columns = [str(c).strip() for c in df.columns]
        code_col = next((c for c in df.columns if c.lower() in ["code", "stock", "ticker", "股票代號", "證券代號", "代號"]), None)
        date_col = next((c for c in df.columns if c.lower() in ["date", "日期", "交易日"]), None)
        diff_col = next((c for c in df.columns if c.lower() in ["diff", "buyers_sellers_diff", "買賣家數差", "買賣家數差值"]), None)
        if code_col is None or diff_col is None:
            return pd.DataFrame()
        out = pd.DataFrame()
        out["code"] = df[code_col].astype(str).str.extract(r"(\d{4,6})")[0]
        out["diff"] = pd.to_numeric(df[diff_col], errors="coerce")
        if date_col is not None:
            out["date"] = pd.to_datetime(df[date_col], errors="coerce")
        else:
            out["date"] = pd.NaT
        return out.dropna(subset=["code", "diff"])
    except Exception:
        return pd.DataFrame()


def grade_from_score(score: int) -> str:
    if score >= 80:
        return "A 優先看"
    if score >= 65:
        return "B 觀察/試單"
    if score >= 50:
        return "C 型態未成熟"
    return "D 略過"


def pct(x: float) -> str:
    if not np.isfinite(x):
        return "-"
    return f"{x * 100:.2f}%"


# -----------------------------
# Indicators
# -----------------------------
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for n in [5, 10, 20, 60, 120]:
        df[f"MA{n}"] = df["Close"].rolling(n).mean()
    df["VOL5"] = df["Volume"].rolling(5).mean()
    df["VOL20"] = df["Volume"].rolling(20).mean()
    df["VOL60"] = df["Volume"].rolling(60).mean()

    mid = df["Close"].rolling(20).mean()
    std = df["Close"].rolling(20).std()
    df["BB_MID"] = mid
    df["BB_UPPER"] = mid + 2 * std
    df["BB_LOWER"] = mid - 2 * std
    df["BB_WIDTH"] = (df["BB_UPPER"] - df["BB_LOWER"]) / mid

    low9 = df["Low"].rolling(9).min()
    high9 = df["High"].rolling(9).max()
    rsv = (df["Close"] - low9) / (high9 - low9).replace(0, np.nan) * 100
    df["K"] = rsv.ewm(com=2, adjust=False).mean()
    df["D"] = df["K"].ewm(com=2, adjust=False).mean()
    df["BODY_HIGH"] = pd.concat([df["Open"], df["Close"]], axis=1).max(axis=1)
    df["BODY_LOW"] = pd.concat([df["Open"], df["Close"]], axis=1).min(axis=1)
    return df



# -----------------------------
# V3: Market filter, sector ranking, watch/holding helpers
# -----------------------------
def market_regime_from_df(df_raw: pd.DataFrame, label: str) -> Dict:
    """Classify broad-market condition using simple moving-average context."""
    if df_raw is None or df_raw.empty or len(df_raw) < 140:
        return {"指數": label, "日期": "-", "收盤": np.nan, "狀態": "資料不足", "偏移分數": 0, "說明": "無足夠資料"}
    df = add_indicators(df_raw).dropna().copy()
    if len(df) < 80:
        return {"指數": label, "日期": "-", "收盤": np.nan, "狀態": "資料不足", "偏移分數": 0, "說明": "無足夠資料"}
    t = df.iloc[-1]
    ma20, ma60, ma120 = safe_float(t["MA20"]), safe_float(t["MA60"]), safe_float(t["MA120"])
    close = safe_float(t["Close"])
    ma60_prev = safe_float(df["MA60"].iloc[-20])
    date = df.index[-1].strftime("%Y-%m-%d") if hasattr(df.index[-1], "strftime") else str(df.index[-1])
    if close > ma20 > ma60 and ma60 >= ma60_prev:
        state, bias, desc = "多頭", 6, "收盤站上月線且月線高於季線，季線走升"
    elif close < ma60 and ma20 < ma60:
        state, bias, desc = "空頭", -12, "收盤跌破季線且月線低於季線"
    elif close > ma60 or ma20 > ma60:
        state, bias, desc = "盤整偏多", 2, "仍在季線附近或月線偏強"
    else:
        state, bias, desc = "盤整偏弱", -5, "大盤趨勢未明顯偏多"
    return {"指數": label, "日期": date, "收盤": round(close, 2), "狀態": state, "偏移分數": bias, "說明": desc}


@st.cache_data(ttl=60 * 30, show_spinner=False)
def get_market_context() -> Dict:
    twii = market_regime_from_df(fetch_ohlcv("^TWII", period="2y"), "加權指數")
    twoii = market_regime_from_df(fetch_ohlcv("^TWOII", period="2y"), "櫃買指數")
    rows = [twii, twoii]
    biases = [r["偏移分數"] for r in rows if isinstance(r.get("偏移分數"), (int, float))]
    if not biases:
        combined_bias, combined_label = 0, "大盤資料不足"
    else:
        avg = float(np.mean(biases))
        if min(biases) <= -10:
            combined_bias, combined_label = -10, "大盤偏空，降低進場分數"
        elif avg >= 5:
            combined_bias, combined_label = 6, "大盤多頭，提高三角策略分數"
        elif avg >= 1:
            combined_bias, combined_label = 2, "大盤盤整偏多，小幅加分"
        elif avg <= -4:
            combined_bias, combined_label = -5, "大盤盤整偏弱，降低分數"
        else:
            combined_bias, combined_label = 0, "大盤中性，不調整分數"
    return {"rows": rows, "combined_bias": int(combined_bias), "combined_label": combined_label}


def detect_enhanced_fake_break(df: pd.DataFrame, tri: TriangleResult, tol: float) -> Tuple[int, List[str], List[str], List[str]]:
    """Detect the lecture-style fake lower-line break: long lower shadow, quiet volume, 1-2 day stand-back."""
    if tri.lower_line is None or df is None or df.empty:
        return 0, [], [], []
    idx = tri.lower_line.index.intersection(df.index)
    if len(idx) < 8:
        return 0, [], [], []
    w = df.loc[idx].copy().tail(10)
    lower = tri.lower_line.loc[w.index]
    tags, reasons, warnings = [], [], []
    score = 0
    for pos in range(max(0, len(w) - 6), len(w)):
        day = w.iloc[pos]
        line = safe_float(lower.iloc[pos])
        if line <= 0:
            continue
        broke = bool(day["Low"] < line * (1 - tol))
        if not broke:
            continue
        end = min(len(w) - 1, pos + 2)
        stand_back = bool(w["Close"].iloc[pos:end + 1].max() >= line * (1 - tol / 2))
        vol_quiet = True
        if np.isfinite(safe_float(day.get("VOL20", np.nan))) and safe_float(day.get("VOL20", np.nan)) > 0:
            vol_quiet = bool(day["Volume"] <= day["VOL20"] * 1.15)
        lower_shadow = min(day["Open"], day["Close"]) - day["Low"]
        candle_range = max(day["High"] - day["Low"], safe_float(day["Close"]) * 0.001)
        long_lower_shadow = bool(lower_shadow / candle_range >= 0.35)
        rebound_red = bool(w["Close"].iloc[end] >= w["Open"].iloc[end]) if end > pos else False
        if stand_back and vol_quiet and long_lower_shadow:
            score = 8 if rebound_red else 6
            tags.append("主力洗盤型假跌破")
            reasons.append("近幾日出現跌破下線、長下影、未明顯出量，且1–2日內站回")
            break
    return score, tags, reasons, warnings


def latest_long_red_half(df: pd.DataFrame, tri: TriangleResult, tol: float) -> Tuple[Optional[float], Optional[pd.Timestamp], str]:
    """Find the latest bullish long candle after/near an upper-line breakout and return its body midpoint."""
    if tri.upper_line is None or df is None or df.empty:
        return None, None, ""
    idx = tri.upper_line.index.intersection(df.index)
    if len(idx) < 5:
        return None, None, ""
    w = df.loc[idx].copy().tail(18)
    upper = tri.upper_line.loc[w.index]
    for i in range(len(w) - 1, -1, -1):
        row = w.iloc[i]
        price = safe_float(row["Close"])
        if price <= 0:
            continue
        is_bull = bool(row["Close"] > row["Open"])
        body_pct = safe_float((row["Close"] - row["Open"]) / price)
        vol_ok = True
        if np.isfinite(safe_float(row.get("VOL20", np.nan))) and safe_float(row.get("VOL20", np.nan)) > 0:
            vol_ok = bool(row["Volume"] >= row["VOL20"] * 1.15)
        near_break = bool(row["Close"] >= safe_float(upper.iloc[i]) * (1 + tol * 0.5))
        if is_bull and body_pct >= 0.035 and near_break and vol_ok:
            half = safe_float(row["Open"] + (row["Close"] - row["Open"]) / 2)
            dt = w.index[i]
            return half, dt, "突破長紅K一半"
    return None, None, ""


def build_daily_plan(results: List[Dict]) -> pd.DataFrame:
    rows = []
    for r in results or []:
        action = str(r.get("動作", ""))
        tags = str(r.get("標籤", ""))
        warnings = str(r.get("警示", ""))
        if "停損" in action or "跌破" in warnings:
            bucket = "⑤ 停損/風險優先檢查"
        elif "獲利" in action or "離開布林" in tags or "長上影" in tags:
            bucket = "⑥ 獲利出場觀察"
        elif "回測上線" in action:
            bucket = "④ 已突破等回測/回測加碼"
        elif "突破" in action:
            bucket = "③ 接近或已突破上線"
        elif "下線" in tags or "試單" in action or "基本單" in action:
            bucket = "② 接近下線/試單候選"
        else:
            bucket = "① 明日優先觀察"
        rows.append({
            "清單": bucket,
            "代號": r.get("代號"), "名稱": r.get("名稱"), "產業": r.get("產業", ""),
            "收盤": r.get("收盤"), "評分": r.get("評分"), "等級": r.get("等級"),
            "動作": r.get("動作"), "距上線%": r.get("距上線%"), "距下線%": r.get("距下線%"),
            "標籤": r.get("標籤"), "警示": r.get("警示"),
        })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["清單", "評分"], ascending=[True, False])


def sector_strength_table(results: List[Dict]) -> pd.DataFrame:
    if not results:
        return pd.DataFrame()
    df = pd.DataFrame([{k: v for k, v in r.items() if not str(k).startswith("_")} for r in results])
    if "產業" not in df.columns:
        df["產業"] = "未分類"
    rows = []
    for industry, sub in df.groupby("產業", dropna=False):
        top = sub.sort_values("評分", ascending=False).head(5)
        rows.append({
            "產業": industry or "未分類",
            "候選檔數": len(sub),
            "A級檔數": int(sub["等級"].astype(str).str.startswith("A").sum()),
            "平均分數": round(float(sub["評分"].mean()), 1),
            "最高分": int(sub["評分"].max()),
            "代表股票": "、".join([f"{a}{b}" for a, b in zip(top["代號"].astype(str), top["名稱"].astype(str))]),
        })
    out = pd.DataFrame(rows)
    return out.sort_values(["A級檔數", "候選檔數", "平均分數"], ascending=False)


def parse_holdings_csv_or_text(uploaded_file, text: str, default_suffix: str) -> pd.DataFrame:
    frames = []
    if uploaded_file is not None:
        raw = uploaded_file.read()
        for enc in ["utf-8-sig", "big5", "cp950", "utf-8"]:
            try:
                df = pd.read_csv(io.BytesIO(raw), encoding=enc)
                frames.append(df)
                break
            except Exception:
                pass
    if text.strip():
        rows = []
        for line in text.strip().splitlines():
            parts = [x.strip() for x in re.split(r"[,，\s]+", line) if x.strip()]
            if len(parts) >= 1:
                rows.append({
                    "code": parts[0],
                    "buy_price": parts[1] if len(parts) > 1 else np.nan,
                    "shares": parts[2] if len(parts) > 2 else np.nan,
                    "entry_reason": " ".join(parts[3:]) if len(parts) > 3 else "",
                })
        if rows:
            frames.append(pd.DataFrame(rows))
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df.columns = [str(c).strip() for c in df.columns]
    colmap = {}
    for c in df.columns:
        lc = c.lower()
        if lc in ["code", "ticker", "股票代號", "代號", "證券代號"]:
            colmap[c] = "code"
        elif lc in ["buy_price", "cost", "成本", "買進價", "進場價"]:
            colmap[c] = "buy_price"
        elif lc in ["shares", "qty", "股數", "庫存股數", "張數"]:
            colmap[c] = "shares"
        elif lc in ["entry_reason", "reason", "進場理由", "備註"]:
            colmap[c] = "entry_reason"
    df = df.rename(columns=colmap)
    if "code" not in df.columns:
        return pd.DataFrame()
    out = pd.DataFrame()
    out["ticker"] = df["code"].astype(str).apply(lambda x: normalize_code(x, default_suffix))
    out["code"] = out["ticker"].apply(pure_code)
    out["buy_price"] = pd.to_numeric(df.get("buy_price", np.nan), errors="coerce")
    out["shares"] = pd.to_numeric(df.get("shares", np.nan), errors="coerce")
    out["entry_reason"] = df.get("entry_reason", "").astype(str) if "entry_reason" in df.columns else ""
    out["name"] = out["code"]
    return out.dropna(subset=["ticker"]).drop_duplicates("ticker")

def local_pivots(values: pd.Series, mode: str = "high", span: int = 2) -> List[int]:
    arr = values.to_numpy(dtype=float)
    idx: List[int] = []
    for i in range(span, len(arr) - span):
        window = arr[i - span:i + span + 1]
        if np.any(np.isnan(window)):
            continue
        if mode == "high" and arr[i] >= np.max(window):
            idx.append(i)
        if mode == "low" and arr[i] <= np.min(window):
            idx.append(i)
    return idx


def fit_line_from_points(n: int, indices: List[int], values: pd.Series, kind: str) -> Optional[np.ndarray]:
    y = values.to_numpy(dtype=float)
    x = np.arange(n)
    pts = [i for i in indices if 0 <= i < n and not np.isnan(y[i])]

    if len(pts) < 2:
        cut = max(8, int(n * 0.45))
        if kind == "upper":
            i1 = int(np.nanargmax(y[:cut]))
            i2 = cut + int(np.nanargmax(y[cut:])) if cut < n - 3 else n - 1
        else:
            i1 = int(np.nanargmin(y[:cut]))
            i2 = cut + int(np.nanargmin(y[cut:])) if cut < n - 3 else n - 1
        pts = [i1, i2]

    pts = sorted(set(pts))
    if len(pts) > 6:
        if kind == "upper":
            pts = sorted(pts, key=lambda i: y[i], reverse=True)[:6]
        else:
            pts = sorted(pts, key=lambda i: y[i])[:6]
        pts = sorted(pts)

    if len(pts) < 2 or max(pts) - min(pts) < max(8, n * 0.22):
        return None

    m, b = np.polyfit(np.array(pts), y[pts], 1)
    line = m * x + b
    if kind == "upper":
        violation = np.nanpercentile(y - line, 88)
        if violation > 0:
            line = line + violation
    else:
        violation = np.nanpercentile(line - y, 88)
        if violation > 0:
            line = line - violation
    return line


def _empty_triangle(reason: str, pattern: str = "未形成") -> TriangleResult:
    return TriangleResult(False, pattern, 0, None, None, None, None, None, None, None, None, 0, 0, "未成熟", [], [reason], [])


def detect_triangle(df: pd.DataFrame, lookback: int, tol: float, tail_width_limit: float) -> TriangleResult:
    if len(df) < max(lookback, 70):
        return _empty_triangle("資料不足", "資料不足")

    w = df.tail(lookback).copy()
    n = len(w)
    body_high = w.get("BODY_HIGH", pd.concat([w["Open"], w["Close"]], axis=1).max(axis=1))
    body_low = w.get("BODY_LOW", pd.concat([w["Open"], w["Close"]], axis=1).min(axis=1))

    hi_idx = local_pivots(body_high.reset_index(drop=True), "high", span=2)
    lo_idx = local_pivots(body_low.reset_index(drop=True), "low", span=2)
    upper_np = fit_line_from_points(n, hi_idx, body_high.reset_index(drop=True), "upper")
    lower_np = fit_line_from_points(n, lo_idx, body_low.reset_index(drop=True), "lower")

    if upper_np is None or lower_np is None:
        return _empty_triangle("無法找到足夠的高低點連線")

    upper = pd.Series(upper_np, index=w.index)
    lower = pd.Series(lower_np, index=w.index)
    close = w["Close"]
    price = safe_float(close.iloc[-1])

    if price <= 0 or upper.iloc[-1] <= lower.iloc[-1]:
        return TriangleResult(False, "未形成", 0, upper, lower, safe_float(upper.iloc[-1]), safe_float(lower.iloc[-1]), None, None, None, None, 0, 0, "未成熟", [], ["上下線無有效收斂"], [])

    width_old = safe_float((upper.iloc[0] - lower.iloc[0]) / close.iloc[0])
    width_now = safe_float((upper.iloc[-1] - lower.iloc[-1]) / price)
    width_ratio = width_now / width_old if width_old and width_old > 0 else np.nan
    upper_change = safe_float((upper.iloc[-1] - upper.iloc[0]) / upper.iloc[0])
    lower_change = safe_float((lower.iloc[-1] - lower.iloc[0]) / lower.iloc[0])

    inside = (body_high <= upper * (1 + tol)) & (body_low >= lower * (1 - tol))
    inside_ratio = float(inside.mean())

    near_upper = (np.abs(body_high.to_numpy() / upper.to_numpy() - 1) <= max(tol * 1.6, 0.018))
    near_lower = (np.abs(body_low.to_numpy() / lower.to_numpy() - 1) <= max(tol * 1.6, 0.018))
    touch_upper = int(pd.Series(near_upper).rolling(3).max().sum() // 2)
    touch_lower = int(pd.Series(near_lower).rolling(3).max().sum() // 2)

    lower_break = body_low < lower * (1 - tol)
    lower_break_count = int(lower_break.sum())
    fake_break_ok = False
    if 0 < lower_break_count <= 2:
        breaks = np.where(lower_break.to_numpy())[0]
        ok_list = []
        for b in breaks:
            after = min(n - 1, b + 2)
            stand_back = bool(w["Close"].iloc[b:after + 1].max() >= lower.iloc[b] * (1 - tol / 2))
            vol_quiet = True
            if "VOL20" in w.columns and np.isfinite(safe_float(w["VOL20"].iloc[b])) and safe_float(w["VOL20"].iloc[b]) > 0:
                vol_quiet = bool(w["Volume"].iloc[b] <= w["VOL20"].iloc[b] * 1.15)
            lower_shadow = min(w["Open"].iloc[b], w["Close"].iloc[b]) - w["Low"].iloc[b]
            candle_range = max(w["High"].iloc[b] - w["Low"].iloc[b], price * 0.001)
            long_lower_shadow = bool(lower_shadow / candle_range >= 0.35)
            ok_list.append(stand_back and (vol_quiet or long_lower_shadow))
        fake_break_ok = all(ok_list)

    if upper_change <= -0.03 and lower_change >= 0.03:
        pattern = "對稱三角"
    elif abs(upper_change) <= 0.06 and lower_change >= 0.03:
        pattern = "上升三角"
    elif upper_change <= -0.03 and abs(lower_change) <= 0.06:
        pattern = "下降三角"
    else:
        pattern = "疑似收斂"

    converging = width_ratio < 0.88 and width_now < 0.24
    tail = width_now <= tail_width_limit
    price_near_zone = (lower.iloc[-1] * (1 - 2 * tol)) <= price <= (upper.iloc[-1] * (1 + 2 * tol))
    touch_ok = (touch_upper >= 2 and touch_lower >= 2) or fake_break_ok
    valid = bool(pattern in ["對稱三角", "上升三角", "下降三角", "疑似收斂"] and converging and price_near_zone and (inside_ratio >= 0.68 or fake_break_ok) and touch_ok)

    if width_now <= tail_width_limit * 0.75 and inside_ratio >= 0.72:
        maturity = "收斂尾端"
    elif width_ratio <= 0.72:
        maturity = "收斂中段"
    else:
        maturity = "剛成形"

    tags: List[str] = []
    reasons: List[str] = []
    warnings: List[str] = []
    score = 0

    if valid:
        score += 30
        tags.append(pattern)
        reasons.append(f"近 {lookback} 日上下線收斂，寬度比約 {width_ratio:.2f}，觸線約上{touch_upper}/下{touch_lower}次")
    if pattern in ["對稱三角", "上升三角"]:
        score += 10
        reasons.append("型態偏講義較適合盤整中布局的對稱／上升三角")
    elif pattern == "下降三角":
        warnings.append("下降三角偏弱，較適合等待有效突破上線後再追蹤")
    if tail:
        score += 10
        tags.append("收斂尾端")
    if inside_ratio >= 0.78:
        score += 7
        tags.append("線內整理")
    if fake_break_ok:
        score += 5
        tags.append("假跌破站回")
    if touch_upper >= 2 and touch_lower >= 2:
        score += 5
        tags.append("觸線次數足夠")
    if width_now < 0.13:
        score += 4
        tags.append("波動已縮小")
    if not valid:
        reasons.append(f"寬度比={width_ratio:.2f}、線內比例={inside_ratio:.2f}、上/下觸線={touch_upper}/{touch_lower}，尚未達主要條件")

    return TriangleResult(
        bool(valid), pattern, int(min(score, 67)), upper, lower,
        safe_float(upper.iloc[-1]), safe_float(lower.iloc[-1]), safe_float(width_ratio),
        safe_float(inside_ratio), safe_float(upper_change), safe_float(lower_change),
        touch_upper, touch_lower, maturity,
        tags, reasons, warnings,
    )


def chip_score_for_code(chip_df: pd.DataFrame, code: str, xq_reverse: bool, days: int = 20) -> Tuple[int, List[str], List[str]]:
    if chip_df is None or chip_df.empty:
        return 0, [], ["未匯入買賣家數差，因此籌碼集中條件未納入評分"]
    sub = chip_df[chip_df["code"].astype(str) == str(code)].copy()
    if sub.empty:
        return 0, [], ["此股沒有買賣家數差資料"]
    if "date" in sub.columns and sub["date"].notna().any():
        sub = sub.sort_values("date")
    sub = sub.tail(days)
    diff = sub["diff"].astype(float)
    if xq_reverse:
        concentrated_days = (diff >= 50).sum()
        strong_days = (diff >= 80).sum()
    else:
        concentrated_days = (diff <= -50).sum()
        strong_days = (diff <= -80).sum()
    ratio = concentrated_days / max(1, len(diff))
    tags, warnings = [], []
    score = 0
    if ratio >= 0.60:
        score += 9
        tags.append(f"籌碼集中 {concentrated_days}/{len(diff)}日")
    if strong_days >= max(5, int(len(diff) * 0.35)):
        score += 6
        tags.append("買賣家數差強集中")
    if score == 0:
        warnings.append("買賣家數差未達連續集中條件")
    return score, tags, warnings


def detect_double_triangle_model(df: pd.DataFrame, tri: TriangleResult, lookback: int) -> Tuple[int, List[str], List[str], List[str]]:
    """A practical approximation of the lecture's high-level triangle-after-triangle model."""
    if df is None or len(df) < 180 or tri.upper_line is None or not tri.is_valid:
        return 0, [], [], []
    tags, reasons, warnings = [], [], []
    score = 0

    end = len(df)
    start_triangle = max(0, end - lookback)
    before = df.iloc[:start_triangle].copy()
    recent_all = df.tail(140).copy()
    if len(before) < 90:
        return 0, [], [], []

    pre_base = before.tail(min(180, len(before)))
    base_range = (pre_base["High"].max() - pre_base["Low"].min()) / max(pre_base["Close"].median(), 0.01)
    has_dormancy = bool(base_range <= 0.75 and pre_base["VOL60"].dropna().median() > 0)

    vol_break = False
    break_indices = []
    close = df["Close"]
    vol = df["Volume"]
    rolling_high = close.rolling(90).max().shift(1)
    vol60 = df["VOL60"]
    scan_from = max(90, len(df) - 160)
    for i in range(scan_from, start_triangle):
        if i < 1 or not np.isfinite(safe_float(rolling_high.iloc[i])) or not np.isfinite(safe_float(vol60.iloc[i])):
            continue
        if close.iloc[i] >= rolling_high.iloc[i] * 1.03 and vol.iloc[i] >= vol60.iloc[i] * 1.8:
            vol_break = True
            break_indices.append(i)

    near_high = bool(df["Close"].iloc[-1] >= recent_all["High"].max() * 0.78)
    prior_break_not_lost = True
    if break_indices:
        b = break_indices[-1]
        prior_break_not_lost = bool(df["Close"].iloc[-1] >= df["Close"].iloc[b] * 0.88)

    if has_dormancy and vol_break and near_high and prior_break_not_lost:
        score += 12
        tags.append("高檔三角再三角")
        reasons.append("曾有沉澱後帶量突破，且目前在高檔再度形成收斂")
    elif vol_break and tri.pattern in ["上升三角", "對稱三角"] and near_high:
        score += 6
        tags.append("三角再三角雛形")
        reasons.append("近期曾帶量創高，目前仍在相對高檔整理")
    return score, tags, reasons, warnings


def infer_action_stage(
    tri: TriangleResult,
    small_k_vol_contract: bool,
    support_touch: bool,
    breakout: bool,
    pullback_upper: bool,
    stop_unbroken: bool,
    stop_after_break: bool,
    bb_leave_upper: bool,
    long_upper_black: bool,
    touch_60ma: bool,
    kd_cross: bool,
) -> str:
    if stop_unbroken or stop_after_break:
        return "停損/風險檢查"
    if bb_leave_upper or long_upper_black:
        return "獲利出場觀察"
    if pullback_upper:
        return "回測上線加碼候選"
    if breakout:
        return "突破加碼候選"
    if small_k_vol_contract or support_touch:
        return "基本單/試單候選"
    if touch_60ma and kd_cross:
        return "早期試單候選"
    if tri.is_valid and tri.maturity == "收斂尾端":
        return "觀察：等量縮/突破"
    if tri.is_valid:
        return "觀察清單"
    return "未達條件"


def evaluate_stock(
    ticker: str,
    name: str,
    df_raw: pd.DataFrame,
    lookback: int,
    tol: float,
    tail_width_limit: float,
    chip_df: pd.DataFrame,
    xq_reverse: bool,
    use_chip: bool,
    avg_lot_limit: int,
    include_early: bool = True,
    industry: str = "自選",
    market_bias: int = 0,
    market_label: str = "未套用",
) -> Optional[Dict]:
    if df_raw is None or df_raw.empty or len(df_raw) < max(lookback, 90):
        return None
    df = add_indicators(df_raw).dropna().copy()
    if len(df) < lookback:
        return None
    tri = detect_triangle(df, lookback, tol, tail_width_limit)
    today = df.iloc[-1]
    prev = df.iloc[-2]
    code = pure_code(ticker)
    price = safe_float(today["Close"])
    if price <= 0:
        return None

    score = tri.score
    tags = list(tri.tags)
    reasons = list(tri.reasons)
    warnings = list(tri.warnings)
    if market_bias > 0:
        score += market_bias
        tags.append(f"大盤濾網+{market_bias}")
        reasons.append(market_label)
    elif market_bias < 0:
        score += market_bias
        warnings.append(market_label)
        tags.append(f"大盤濾網{market_bias}")

    candle_body = abs(today["Close"] - today["Open"])
    candle_range = max(today["High"] - today["Low"], price * 0.001)
    body_pct = candle_body / price
    vol_ratio = safe_float(today["Volume"] / today["VOL20"], 99)
    avg_lots20 = safe_float(today["VOL20"] / 1000, np.nan)

    upper_today = tri.upper_today if tri.upper_today else np.nan
    lower_today = tri.lower_today if tri.lower_today else np.nan
    prev_upper = safe_float(tri.upper_line.iloc[-2]) if tri.upper_line is not None else np.nan

    small_k_vol_contract = bool(tri.is_valid and body_pct <= 0.025 and vol_ratio <= 0.72 and price >= lower_today * (1 - tol))
    support_touch = bool(tri.is_valid and today["Low"] <= lower_today * (1 + 2 * tol) and today["Close"] >= lower_today * (1 - tol))
    breakout = bool(tri.upper_line is not None and today["Close"] > upper_today * (1 + tol) and prev["Close"] <= prev_upper * (1 + tol))
    not_chasing_too_far = bool(np.isfinite(upper_today) and today["Close"] <= upper_today * 1.06)

    recent_break = False
    pullback_upper = False
    if tri.upper_line is not None:
        recent = df.tail(min(10, len(df))).copy()
        line_recent = tri.upper_line.tail(min(10, len(tri.upper_line)))
        if len(recent) == len(line_recent):
            recent_break = bool((recent["Close"].to_numpy() > line_recent.to_numpy() * (1 + tol)).any())
            pullback_upper = bool(recent_break and today["Low"] <= upper_today * (1 + 2 * tol) and today["Close"] >= upper_today * (1 - tol))

    if small_k_vol_contract:
        score += 12
        tags.append("尾端小K量縮")
        reasons.append("符合收斂尾端小K線量縮觀察")
    if support_touch:
        score += 10
        tags.append("下線不破")
        reasons.append("今日接近／回測下線且收盤未破")
    if breakout and not_chasing_too_far:
        score += 14
        tags.append("初站上上線")
        reasons.append("今日收盤初步突破上線且未過度追高")
    elif breakout and not not_chasing_too_far:
        score += 5
        tags.append("突破但偏遠")
        warnings.append("已突破但離上線偏遠，講義提醒跳空大漲不宜急追")
    if pullback_upper:
        score += 12
        tags.append("回測上線不破")
        reasons.append("近期突破後，今日回測上線仍有支撐")

    fb_score, fb_tags, fb_reasons, fb_warnings = detect_enhanced_fake_break(df, tri, tol)
    if fb_score:
        score += fb_score
        tags += fb_tags
        reasons += fb_reasons
        warnings += fb_warnings

    red_half, red_half_date, red_half_label = latest_long_red_half(df, tri, tol)
    if red_half is not None and np.isfinite(red_half):
        tags.append("長紅K一半價")
        reasons.append(f"最近突破長紅K一半價約 {red_half:.2f}，可作為保守獲利/風險檢查線")

    ma_trend = bool(today["Close"] > today["MA60"] and today["MA20"] > today["MA60"])
    if ma_trend:
        score += 7
        tags.append("中期趨勢偏多")
    else:
        warnings.append("中期均線趨勢未明顯偏多")

    ma_compress = bool(
        max(today["MA5"], today["MA10"], today["MA20"]) / max(0.01, min(today["MA5"], today["MA10"], today["MA20"])) <= 1.035
    )
    if ma_compress:
        score += 5
        tags.append("5/10/20均線糾結")

    touch_60ma = bool(today["Low"] <= today["MA60"] * 1.02 and today["Close"] >= today["MA60"] * 0.995)
    if touch_60ma:
        score += 5
        tags.append("回測60MA不破")

    kd_cross = bool(prev["K"] <= prev["D"] and today["K"] > today["D"] and today["K"] < 55)
    if kd_cross:
        score += 5
        tags.append("KD低檔黃金交叉")

    bb_contract = bool(today["BB_WIDTH"] <= df["BB_WIDTH"].tail(120).quantile(0.30))
    if bb_contract:
        score += 4
        tags.append("布林收縮")

    if avg_lots20 <= avg_lot_limit:
        score += 4
        tags.append(f"均量≤{avg_lot_limit:,}張")
    else:
        warnings.append(f"20日均量約 {avg_lots20:,.0f} 張，已高於小資本股篩選上限")

    model_score, mtags, mreasons, mwarn = detect_double_triangle_model(df, tri, lookback)
    score += model_score
    tags += mtags
    reasons += mreasons
    warnings += mwarn

    if use_chip:
        cs, ctags, cwarn = chip_score_for_code(chip_df, code, xq_reverse=xq_reverse)
        score += cs
        tags += ctags
        warnings += cwarn

    stop_unbroken = bool(tri.lower_line is not None and today["Close"] < lower_today * (1 - tol))
    stop_after_break = bool(recent_break and today["Close"] < upper_today * (1 - tol))
    bb_leave_upper = bool(today["High"] < today["BB_UPPER"] and prev["High"] >= prev["BB_UPPER"])
    upper_shadow = today["High"] - max(today["Open"], today["Close"])
    long_upper_black = bool(today["Close"] < today["Open"] and upper_shadow / candle_range >= 0.45 and vol_ratio >= 1.6)
    red_half_broken = bool(red_half is not None and np.isfinite(red_half) and today["Close"] < red_half)
    if red_half_broken:
        warnings.append(f"獲利退場觀察：收盤跌破突破長紅K一半價 {red_half:.2f}")
        tags.append("跌破長紅K一半")
    if stop_unbroken:
        warnings.append("停損警示：未突破前收盤跌破下線")
        tags.append("⚠跌破下線")
    if stop_after_break:
        warnings.append("停損警示：突破後又跌破上線")
        tags.append("⚠跌破上線")
    if bb_leave_upper:
        warnings.append("獲利退場觀察：K線疑似離開布林上軌")
        tags.append("離開布林上軌")
    if long_upper_black:
        warnings.append("獲利退場觀察：翻黑長上影且放量")
        tags.append("翻黑長上影")

    score = int(max(0, min(100, score)))
    action = infer_action_stage(
        tri, small_k_vol_contract, support_touch, breakout, pullback_upper,
        stop_unbroken, stop_after_break, bb_leave_upper, (long_upper_black or red_half_broken), touch_60ma, kd_cross
    )

    if not tri.is_valid and score < 55 and not (include_early and touch_60ma and kd_cross):
        return None

    return {
        "代號": code,
        "名稱": name or code,
        "產業": industry or "未分類",
        "Ticker": ticker,
        "大盤": market_label,
        "日期": df.index[-1].strftime("%Y-%m-%d") if hasattr(df.index[-1], "strftime") else str(df.index[-1]),
        "收盤": round(price, 2),
        "評分": score,
        "等級": grade_from_score(score),
        "動作": action,
        "型態": tri.pattern,
        "成熟度": tri.maturity,
        "上線": round(upper_today, 2) if np.isfinite(upper_today) else np.nan,
        "下線": round(lower_today, 2) if np.isfinite(lower_today) else np.nan,
        "距上線%": round((price / upper_today - 1) * 100, 2) if np.isfinite(upper_today) else np.nan,
        "距下線%": round((price / lower_today - 1) * 100, 2) if np.isfinite(lower_today) else np.nan,
        "觸線": f"上{tri.touch_upper}/下{tri.touch_lower}",
        "20日均量(張)": round(avg_lots20, 0) if np.isfinite(avg_lots20) else np.nan,
        "量比": round(vol_ratio, 2),
        "標籤": "、".join(dict.fromkeys(tags)),
        "理由": "；".join(dict.fromkeys(reasons)),
        "警示": "；".join(dict.fromkeys(warnings)),
        "_df": df,
        "_tri": tri,
        "_key_levels": {
            "upper": upper_today,
            "lower": lower_today,
            "red_half": red_half,
            "red_half_date": str(red_half_date) if red_half_date is not None else "",
            "ma60": safe_float(today.get("MA60", np.nan)),
            "bb_upper": safe_float(today.get("BB_UPPER", np.nan)),
        },
        "_flags": {
            "small_k_vol_contract": small_k_vol_contract,
            "support_touch": support_touch,
            "breakout": breakout,
            "pullback_upper": pullback_upper,
            "stop_unbroken": stop_unbroken,
            "stop_after_break": stop_after_break,
            "bb_leave_upper": bb_leave_upper,
            "long_upper_black": long_upper_black,
            "red_half_broken": red_half_broken,
            "touch_60ma": touch_60ma,
            "kd_cross": kd_cross,
        },
    }


def plot_stock(result: Dict, show_volume: bool = True):
    df = result["_df"].tail(150)
    tri: TriangleResult = result["_tri"]
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        name="K線"
    ))
    for ma in ["MA5", "MA10", "MA20", "MA60"]:
        fig.add_trace(go.Scatter(x=df.index, y=df[ma], mode="lines", name=ma, line=dict(width=1)))
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_UPPER"], mode="lines", name="BB上軌", line=dict(width=1, dash="dot")))
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_LOWER"], mode="lines", name="BB下軌", line=dict(width=1, dash="dot")))
    if tri.upper_line is not None and tri.lower_line is not None:
        u = tri.upper_line.tail(min(len(tri.upper_line), 150))
        l = tri.lower_line.tail(min(len(tri.lower_line), 150))
        fig.add_trace(go.Scatter(x=u.index, y=u, mode="lines", name="三角上線", line=dict(width=3)))
        fig.add_trace(go.Scatter(x=l.index, y=l, mode="lines", name="三角下線", line=dict(width=3)))
        if np.isfinite(safe_float(tri.upper_today)):
            half_idx = df.index[-1]
            fig.add_annotation(x=half_idx, y=tri.upper_today, text="上線", showarrow=True, yshift=10)
        if np.isfinite(safe_float(tri.lower_today)):
            fig.add_annotation(x=df.index[-1], y=tri.lower_today, text="下線", showarrow=True, yshift=-10)
    levels = result.get("_key_levels", {})
    red_half = levels.get("red_half")
    if red_half is not None and np.isfinite(safe_float(red_half)):
        fig.add_hline(y=red_half, line_dash="dash", annotation_text="長紅K一半", annotation_position="bottom right")
    ma60 = levels.get("ma60")
    if ma60 is not None and np.isfinite(safe_float(ma60)):
        fig.add_hline(y=ma60, line_dash="dot", annotation_text="60MA", annotation_position="bottom left")
    fig.update_layout(
        height=560,
        margin=dict(l=10, r=10, t=35, b=10),
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        title=f"{result['代號']} {result['名稱']}｜{result['動作']}｜{result['等級']}｜評分 {result['評分']}",
    )
    return fig


# -----------------------------
# Backtest
# -----------------------------
def backtest_single(
    ticker: str,
    name: str,
    df_raw: pd.DataFrame,
    lookback: int,
    tol: float,
    tail_width_limit: float,
    min_score_bt: int,
    signal_modes: List[str],
    hold_days: int = 20,
    cooldown_days: int = 10,
) -> pd.DataFrame:
    if df_raw is None or df_raw.empty or len(df_raw) < max(160, lookback + hold_days + 90):
        return pd.DataFrame()
    out = []
    last_signal_i = -999
    max_i = len(df_raw) - hold_days - 1
    start_i = max(140, lookback + 80)
    for i in range(start_i, max_i):
        if i - last_signal_i < cooldown_days:
            continue
        hist = df_raw.iloc[: i + 1].copy()
        res = evaluate_stock(
            ticker=ticker,
            name=name,
            df_raw=hist,
            lookback=lookback,
            tol=tol,
            tail_width_limit=tail_width_limit,
            chip_df=pd.DataFrame(),
            xq_reverse=False,
            use_chip=False,
            avg_lot_limit=100000,
            include_early=True,
        )
        if res is None or res["評分"] < min_score_bt:
            continue
        action = res["動作"]
        if signal_modes and not any(m in action for m in signal_modes):
            continue
        entry = safe_float(df_raw["Close"].iloc[i])
        if entry <= 0:
            continue
        future = df_raw.iloc[i + 1 : i + hold_days + 1]
        ret5 = safe_float(future["Close"].iloc[min(4, len(future) - 1)] / entry - 1)
        ret10 = safe_float(future["Close"].iloc[min(9, len(future) - 1)] / entry - 1)
        ret20 = safe_float(future["Close"].iloc[-1] / entry - 1)
        max_gain = safe_float(future["High"].max() / entry - 1)
        max_dd = safe_float(future["Low"].min() / entry - 1)
        out.append({
            "代號": pure_code(ticker),
            "名稱": name,
            "日期": df_raw.index[i].strftime("%Y-%m-%d") if hasattr(df_raw.index[i], "strftime") else str(df_raw.index[i]),
            "訊號": action,
            "分數": res["評分"],
            "型態": res["型態"],
            "5日%": round(ret5 * 100, 2),
            "10日%": round(ret10 * 100, 2),
            "20日%": round(ret20 * 100, 2),
            "20日最大漲幅%": round(max_gain * 100, 2),
            "20日最大回撤%": round(max_dd * 100, 2),
        })
        last_signal_i = i
    return pd.DataFrame(out)


def summarize_backtest(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    rows = []
    for group_name, sub in [("全部", trades)] + list(trades.groupby("訊號")):
        rows.append({
            "類別": group_name,
            "次數": len(sub),
            "5日勝率": f"{(sub['5日%'] > 0).mean() * 100:.1f}%",
            "10日勝率": f"{(sub['10日%'] > 0).mean() * 100:.1f}%",
            "20日勝率": f"{(sub['20日%'] > 0).mean() * 100:.1f}%",
            "5日平均%": round(sub["5日%"].mean(), 2),
            "10日平均%": round(sub["10日%"].mean(), 2),
            "20日平均%": round(sub["20日%"].mean(), 2),
            "平均最大回撤%": round(sub["20日最大回撤%"].mean(), 2),
        })
    return pd.DataFrame(rows)


# -----------------------------
# Scanner utility
# -----------------------------
def scan_rows(rows: pd.DataFrame, period: str, lookback: int, tol: float, tail_width_limit: float, chip_df: pd.DataFrame, xq_reverse: bool, use_chip: bool, avg_lot_limit: int, min_score: int, max_scan: int, market_bias: int = 0, market_label: str = "未套用") -> List[Dict]:
    progress = st.progress(0)
    status = st.empty()
    results: List[Dict] = []
    rows = rows.drop_duplicates("ticker").head(max_scan).reset_index(drop=True)
    for i, row in rows.iterrows():
        ticker = row["ticker"]
        name = row.get("name", pure_code(ticker))
        industry = row.get("industry", "未分類")
        status.info(f"掃描中 {i + 1}/{len(rows)}：{ticker} {name}")
        df = fetch_ohlcv(ticker, period=period)
        res = evaluate_stock(
            ticker=ticker,
            name=name,
            df_raw=df,
            lookback=lookback,
            tol=tol,
            tail_width_limit=tail_width_limit,
            chip_df=chip_df,
            xq_reverse=xq_reverse,
            use_chip=use_chip,
            avg_lot_limit=avg_lot_limit,
            industry=industry,
            market_bias=market_bias,
            market_label=market_label,
        )
        if res is not None and res["評分"] >= min_score:
            results.append(res)
        progress.progress((i + 1) / max(1, len(rows)))
        time.sleep(0.02)
    status.success(f"掃描完成：找到 {len(results)} 檔候選。")
    return sorted(results, key=lambda x: x["評分"], reverse=True)


def make_tickers_from_text(codes_text: str, suffix: str) -> pd.DataFrame:
    parsed = [normalize_code(x, suffix) for x in re.split(r"[,，\n\s]+", codes_text) if x.strip()]
    tickers_df = pd.DataFrame({"ticker": [x for x in parsed if x], "code": [pure_code(x) for x in parsed if x]})
    tickers_df["name"] = tickers_df["code"]
    tickers_df["industry"] = "自選"
    return tickers_df.drop_duplicates("ticker")


# -----------------------------
# App layout
# -----------------------------
st.title("📈 艾斯三角收斂選股器 V3")
st.caption("手機友善版｜V3 新增：大盤濾網、族群強度、每日清單、持股追蹤、假跌破強化、關鍵價位自動畫線。")

with st.expander("這版加入了哪些強化？", expanded=False):
    st.markdown(
        """
        - **更精準三角判斷**：加入觸線次數、假跌破站回、收斂成熟度。
        - **操作狀態**：自動標示觀察、早期試單、基本單、突破加碼、回測加碼、停損或獲利出場觀察。
        - **評分與等級**：A/B/C/D 分級，方便全市場掃描後優先看圖。
        - **三角再三角模型**：抓沉澱後帶量突破、再度高檔收斂的飆股雛形。
        - **資金控管**：依本金與每單位比例，自動估算基本單與加碼金額。
        - **簡易回測**：檢查訊號後 5/10/20 日報酬、勝率、最大回撤。
        - **提醒清單**：自選股每日打開 App 後，自動提示接近上線、接近下線、破線等狀態。
        - **V3 大盤/族群**：加入加權與櫃買大盤濾網，並統計哪個產業最多候選股。
        - **V3 每日清單**：把掃描結果自動分成明日優先觀察、下線試單、突破、回測、停損與獲利出場。
        - **V3 持股追蹤**：輸入庫存成本後，自動檢查目前續抱、加碼、停損或獲利警示。
        - **V3 假跌破強化**：偵測跌破下線、長下影、未出量、1–2日內站回的洗盤型訊號。
        - **V3 關鍵價位線**：圖表自動畫上/下線、60MA、突破長紅K一半價。
        """
    )

st.sidebar.header("共用參數")
period = st.sidebar.selectbox("抓取資料期間", ["1y", "18mo", "2y", "5y"], index=1)
lookback = st.sidebar.slider("三角觀察天數", 25, 90, 45, 5)
tol = st.sidebar.slider("畫線容忍度", 0.005, 0.04, 0.018, 0.001, format="%.3f")
tail_width_limit = st.sidebar.slider("收斂尾端寬度上限", 0.06, 0.25, 0.15, 0.01)
avg_lot_limit = st.sidebar.number_input("小資本股20日均量上限（張）", min_value=500, max_value=100000, value=10000, step=500)
min_score = st.sidebar.slider("最低顯示評分", 0, 100, 58, 1)
max_scan = st.sidebar.slider("本次最多掃描檔數", 10, 800, 100, 10)

st.sidebar.header("籌碼資料（選填）")
chip_file = st.sidebar.file_uploader("上傳買賣家數差 CSV", type=["csv"])
xq_reverse = st.sidebar.checkbox("我的買賣家數差來自 XQ 電腦版（正數代表集中）", value=False)
use_chip = st.sidebar.checkbox("將買賣家數差納入評分", value=chip_file is not None)
chip_df = parse_chip_csv(chip_file)
if chip_file is not None and chip_df.empty:
    st.sidebar.warning("CSV欄位需含 code/股票代號 與 diff/買賣家數差，可另含 date/日期。")

st.sidebar.header("大盤濾網")
use_market_filter = st.sidebar.checkbox("套用大盤環境加減分", value=True)
market_bias = 0
market_label = "未套用大盤濾網"
market_context = None
if use_market_filter:
    market_context = get_market_context()
    market_bias = market_context.get("combined_bias", 0)
    market_label = market_context.get("combined_label", "大盤中性")
    st.sidebar.caption(f"{market_label}｜分數調整 {market_bias:+d}")
else:
    st.sidebar.caption("未套用大盤加減分。")

scanner_tab, market_tab, daily_tab, watch_tab, holdings_tab, capital_tab, backtest_tab, help_tab = st.tabs(["🚀 選股掃描", "🌡️ 大盤/族群", "📋 每日清單", "🔔 自選提醒", "📊 持股追蹤", "💰 資金控管", "🧪 回測", "📘 說明"])

with scanner_tab:
    st.subheader("選股掃描")
    mode = st.radio("股票來源", ["手動輸入", "全市場自動掃描"], index=0, horizontal=True)

    if mode == "手動輸入":
        default_suffix = st.selectbox("未輸入市場時預設", [".TW 上市", ".TWO 上櫃"], index=0)
        suffix = default_suffix.split()[0]
        codes_text = st.text_area(
            "輸入股票代號，用逗號或換行分隔",
            value="2330, 2317, 2454, 2308, 6488.TWO",
            height=100,
        )
        tickers_df = make_tickers_from_text(codes_text, suffix)
    else:
        universe = load_tw_universe()
        markets = st.multiselect("市場", ["上市", "上櫃"], default=["上市", "上櫃"])
        keyword = st.text_input("名稱／代號篩選（可留空）", value="")
        tickers_df = universe[universe["market"].isin(markets)].copy()
        if keyword.strip():
            kw = keyword.strip()
            tickers_df = tickers_df[tickers_df["code"].str.contains(kw) | tickers_df["name"].str.contains(kw, na=False)]
        st.write(f"可掃描清單：{len(tickers_df):,} 檔；本次上限 {max_scan:,} 檔。")

    start_scan = st.button("🚀 開始掃描", type="primary", use_container_width=True)
    if start_scan:
        if tickers_df.empty:
            st.error("沒有可掃描的股票。")
            st.stop()
        if yf is None:
            st.error("尚未安裝 yfinance。請先執行：pip install -r requirements.txt")
            st.stop()
        results = scan_rows(tickers_df, period, lookback, tol, tail_width_limit, chip_df, xq_reverse, use_chip, avg_lot_limit, min_score, max_scan, market_bias=market_bias, market_label=market_label)
        st.session_state["last_results"] = results

    results = st.session_state.get("last_results", [])
    if results:
        table_cols = ["代號", "名稱", "產業", "日期", "收盤", "評分", "等級", "動作", "型態", "成熟度", "上線", "下線", "距上線%", "距下線%", "觸線", "20日均量(張)", "量比", "標籤", "警示"]
        display_df = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in results])
        st.dataframe(display_df[table_cols], use_container_width=True, hide_index=True)
        csv = display_df[table_cols + ["理由"]].to_csv(index=False).encode("utf-8-sig")
        st.download_button("下載結果 CSV", csv, file_name=f"ace_triangle_scan_v2_{datetime.now():%Y%m%d_%H%M}.csv", mime="text/csv")

        st.subheader("個股圖表與判讀")
        options = [f"{r['代號']} {r['名稱']}｜{r['動作']}｜{r['等級']}｜評分 {r['評分']}" for r in results]
        selected = st.selectbox("選擇個股", options)
        idx = options.index(selected)
        r = results[idx]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("評分", r["評分"])
        c2.metric("等級", r["等級"])
        c3.metric("距上線", f"{r['距上線%']}%")
        c4.metric("距下線", f"{r['距下線%']}%")
        tags_html = "".join([f"<span class='pill'>{t}</span>" for t in str(r["標籤"]).split("、") if t])
        st.markdown(f"<div class='signal-card'><b>標籤：</b><br>{tags_html}<br><b>理由：</b>{r['理由']}<br><b>警示：</b>{r['警示']}</div>", unsafe_allow_html=True)
        st.plotly_chart(plot_stock(r), use_container_width=True)


with market_tab:
    st.subheader("大盤環境與族群強度")
    st.write("大盤濾網是用加權指數與櫃買指數的 20/60/120 日均線關係做粗略判斷；偏空時會降低個股評分，避免在熊市硬做三角。")
    ctx = market_context if market_context is not None else get_market_context()
    mdf = pd.DataFrame(ctx.get("rows", []))
    if not mdf.empty:
        st.dataframe(mdf, use_container_width=True, hide_index=True)
        st.metric("目前大盤濾網", ctx.get("combined_label", "-"), f"{ctx.get('combined_bias', 0):+d} 分")
    else:
        st.warning("目前抓不到大盤資料；可能是 Yahoo Finance 暫時無法回傳台股指數。")

    st.subheader("族群強度排行")
    results = st.session_state.get("last_results", [])
    if not results:
        st.info("請先到『選股掃描』完成一次掃描，這裡會統計哪些產業最多候選股與 A 級股。")
    else:
        sec = sector_strength_table(results)
        if sec.empty:
            st.info("目前沒有可統計的族群資料。")
        else:
            st.dataframe(sec, use_container_width=True, hide_index=True)
            st.download_button("下載族群排行 CSV", sec.to_csv(index=False).encode("utf-8-sig"), file_name=f"ace_triangle_sector_strength_{datetime.now():%Y%m%d_%H%M}.csv", mime="text/csv")

with daily_tab:
    st.subheader("每日盤後觀察清單")
    st.write("這裡會把最近一次掃描結果自動分流，讓你隔天不用重新看一大張表。")
    results = st.session_state.get("last_results", [])
    plan_df = build_daily_plan(results)
    if plan_df.empty:
        st.info("請先到『選股掃描』完成一次掃描。")
    else:
        buckets = plan_df["清單"].drop_duplicates().tolist()
        for b in buckets:
            sub = plan_df[plan_df["清單"] == b].copy()
            with st.expander(f"{b}｜{len(sub)} 檔", expanded=b.startswith("①") or b.startswith("②")):
                st.dataframe(sub.drop(columns=["清單"]), use_container_width=True, hide_index=True)
        st.download_button("下載每日觀察清單 CSV", plan_df.to_csv(index=False).encode("utf-8-sig"), file_name=f"ace_triangle_daily_plan_{datetime.now():%Y%m%d_%H%M}.csv", mime="text/csv")

with watch_tab:
    st.subheader("自選股提醒")
    st.write("這不是手機推播；它會在你每天打開 App 時，幫自選股重新判斷是否接近上線、下線、突破或破線。")
    watch_suffix = st.selectbox("自選股未輸入市場時預設", [".TW 上市", ".TWO 上櫃"], index=0, key="watch_suffix")
    watch_codes = st.text_area("自選股代號", value="", placeholder="例如：1519, 3017, 6488.TWO", height=90, key="watch_codes")
    alert_buffer = st.slider("提醒緩衝距離（%）", 0.5, 8.0, 2.0, 0.5) / 100
    if st.button("🔔 檢查提醒", use_container_width=True):
        wdf = make_tickers_from_text(watch_codes, watch_suffix.split()[0])
        if wdf.empty:
            st.warning("請先輸入自選股。")
        else:
            alerts = []
            for _, row in wdf.iterrows():
                df = fetch_ohlcv(row["ticker"], period=period)
                res = evaluate_stock(row["ticker"], row["name"], df, lookback, tol, tail_width_limit, chip_df, xq_reverse, use_chip, avg_lot_limit, include_early=True)
                if res is None:
                    continue
                price = res["收盤"]
                upper = res["上線"]
                lower = res["下線"]
                notes = []
                if np.isfinite(upper):
                    if price >= upper * (1 + tol):
                        notes.append("已站上上線")
                    elif abs(price / upper - 1) <= alert_buffer:
                        notes.append("接近上線，準備觀察突破")
                if np.isfinite(lower):
                    if price < lower * (1 - tol):
                        notes.append("跌破下線，檢查停損")
                    elif abs(price / lower - 1) <= alert_buffer:
                        notes.append("接近下線，觀察是否下線不破")
                if "離開布林上軌" in res["標籤"] or "翻黑長上影" in res["標籤"]:
                    notes.append("獲利出場訊號觀察")
                if notes:
                    alerts.append({"代號": res["代號"], "名稱": res["名稱"], "日期": res["日期"], "收盤": price, "動作": res["動作"], "評分": res["評分"], "提醒": "；".join(notes), "警示": res["警示"]})
            if alerts:
                st.dataframe(pd.DataFrame(alerts), use_container_width=True, hide_index=True)
            else:
                st.info("目前沒有接近上下線、突破或停損的提醒。")


with holdings_tab:
    st.subheader("持股追蹤模式")
    st.write("輸入目前庫存後，App 會用同一套三角規則檢查是否續抱、加碼、停損或獲利出場觀察。")
    h_suffix = st.selectbox("持股未輸入市場時預設", [".TW 上市", ".TWO 上櫃"], index=0, key="holding_suffix")
    sample = pd.DataFrame([
        {"code": "2330", "buy_price": 600, "shares": 1000, "entry_reason": "三角突破"},
        {"code": "6488.TWO", "buy_price": 50, "shares": 2000, "entry_reason": "下線不破試單"},
    ])
    st.download_button("下載持股CSV範本", sample.to_csv(index=False).encode("utf-8-sig"), file_name="ace_holdings_template.csv", mime="text/csv")
    h_file = st.file_uploader("上傳持股 CSV（欄位：code,buy_price,shares,entry_reason）", type=["csv"], key="holding_file")
    h_text = st.text_area("或直接輸入：代號,買進價,股數,進場理由", value="", placeholder="例如：\n2330,600,1000,三角突破\n6488.TWO,50,2000,下線不破試單", height=100)
    if st.button("📊 檢查持股", use_container_width=True):
        hdf = parse_holdings_csv_or_text(h_file, h_text, h_suffix.split()[0])
        if hdf.empty:
            st.warning("請上傳持股 CSV 或輸入至少一檔持股。")
        else:
            rows = []
            full_results = []
            for _, row in hdf.iterrows():
                df = fetch_ohlcv(row["ticker"], period=period)
                res = evaluate_stock(
                    row["ticker"], row.get("name", row["code"]), df, lookback, tol, tail_width_limit,
                    chip_df, xq_reverse, use_chip, avg_lot_limit, include_early=True,
                    industry="持股", market_bias=market_bias, market_label=market_label,
                )
                if res is None and df is not None and not df.empty:
                    df2 = add_indicators(df).dropna().copy()
                    if not df2.empty:
                        price = safe_float(df2["Close"].iloc[-1])
                        res = {"代號": row["code"], "名稱": row["code"], "收盤": price, "評分": 0, "等級": "持股追蹤", "動作": "未達三角條件", "警示": "目前不符合三角候選；仍可用成本與停損自行管理", "標籤": "", "_df": df2, "_tri": _empty_triangle("未達條件"), "_key_levels": {}}
                if res is None:
                    rows.append({"代號": row["code"], "狀態": "抓不到資料"})
                    continue
                buy_price = safe_float(row.get("buy_price", np.nan))
                shares = safe_float(row.get("shares", np.nan))
                price = safe_float(res.get("收盤", np.nan))
                pl_pct = (price / buy_price - 1) * 100 if np.isfinite(buy_price) and buy_price > 0 and np.isfinite(price) else np.nan
                pl_amt = (price - buy_price) * shares if np.isfinite(pl_pct) and np.isfinite(shares) else np.nan
                rows.append({
                    "代號": res.get("代號"), "名稱": res.get("名稱"), "收盤": price,
                    "成本": round(buy_price, 2) if np.isfinite(buy_price) else np.nan,
                    "股數": int(shares) if np.isfinite(shares) else np.nan,
                    "損益%": round(pl_pct, 2) if np.isfinite(pl_pct) else np.nan,
                    "損益金額": round(pl_amt, 0) if np.isfinite(pl_amt) else np.nan,
                    "目前動作": res.get("動作"), "評分": res.get("評分"),
                    "進場理由": row.get("entry_reason", ""), "標籤": res.get("標籤", ""), "警示": res.get("警示", ""),
                })
                full_results.append(res)
            if rows:
                hold_result = pd.DataFrame(rows)
                st.dataframe(hold_result, use_container_width=True, hide_index=True)
                st.download_button("下載持股追蹤 CSV", hold_result.to_csv(index=False).encode("utf-8-sig"), file_name=f"ace_triangle_holdings_{datetime.now():%Y%m%d_%H%M}.csv", mime="text/csv")
            if full_results:
                options = [f"{r.get('代號')} {r.get('名稱')}｜{r.get('目前動作', r.get('動作', ''))}" for r in full_results]
                chosen = st.selectbox("查看持股圖表", options, key="holding_chart_select")
                r = full_results[options.index(chosen)]
                if "_df" in r and "_tri" in r:
                    st.plotly_chart(plot_stock(r), use_container_width=True)

with capital_tab:
    st.subheader("資金控管計算器")
    capital = st.number_input("波段資金總額", min_value=1000, max_value=100000000, value=300000, step=10000)
    unit_pct = st.slider("每一基本單位占資金比例", 1, 25, 10, 1) / 100
    max_pct = st.slider("單一股票最高投入比例", 5, 80, 40, 5) / 100
    price_for_size = st.number_input("欲估算股票價格", min_value=1.0, max_value=10000.0, value=50.0, step=0.5)
    unit_cash = capital * unit_pct
    max_cash = capital * max_pct
    shares_unit = int(unit_cash // price_for_size)
    lots_unit = shares_unit / 1000
    st.metric("每一基本單位金額", f"{unit_cash:,.0f} 元")
    st.metric("單股最高投入", f"{max_cash:,.0f} 元")
    st.metric("每一基本單位約可買", f"{shares_unit:,} 股（{lots_unit:.2f} 張）")
    plan = pd.DataFrame([
        {"階段": "收斂區間符合籌碼集中／尾端量縮", "動作": "基本單", "投入金額": round(unit_cash), "累計投入": round(unit_cash)},
        {"階段": "碰下線不破", "動作": "再加 1 單位", "投入金額": round(unit_cash), "累計投入": round(unit_cash * 2)},
        {"階段": "收盤站上上線", "動作": "突破加碼 1 單位", "投入金額": round(unit_cash), "累計投入": round(unit_cash * 3)},
        {"階段": "突破後回測上線不破", "動作": "回測加碼 1 單位", "投入金額": round(unit_cash), "累計投入": round(min(unit_cash * 4, max_cash))},
        {"階段": "跌破下線／突破後跌破上線", "動作": "照紀律停損或降部位", "投入金額": 0, "累計投入": "依持股出場"},
    ])
    st.dataframe(plan, use_container_width=True, hide_index=True)
    st.info("建議把這個表當成『計畫』，不要因為單檔分數高就超過最大投入比例。")

with backtest_tab:
    st.subheader("簡易回測")
    st.write("回測會用過去每天的收盤資料重新判斷訊號，再看之後 5/10/20 日報酬。這是策略健檢，不代表未來績效。")
    bt_suffix = st.selectbox("回測未輸入市場時預設", [".TW 上市", ".TWO 上櫃"], index=0, key="bt_suffix")
    bt_codes = st.text_area("回測股票代號", value="2330, 2317, 2454", height=90, key="bt_codes")
    bt_period = st.selectbox("回測資料期間", ["2y", "5y"], index=1)
    bt_min_score = st.slider("回測最低訊號分數", 50, 95, 68, 1)
    bt_modes = st.multiselect("納入哪些訊號", ["基本單", "早期試單", "突破加碼", "回測上線", "觀察"], default=["基本單", "早期試單", "突破加碼", "回測上線"])
    if st.button("🧪 開始回測", use_container_width=True):
        btdf = make_tickers_from_text(bt_codes, bt_suffix.split()[0]).head(30)
        if btdf.empty:
            st.warning("請輸入要回測的股票。")
        else:
            all_trades = []
            progress = st.progress(0)
            for i, row in btdf.iterrows():
                st.write(f"回測 {row['ticker']} ...")
                df = fetch_ohlcv(row["ticker"], period=bt_period)
                trades = backtest_single(row["ticker"], row["name"], df, lookback, tol, tail_width_limit, bt_min_score, bt_modes)
                if not trades.empty:
                    all_trades.append(trades)
                progress.progress((i + 1) / max(1, len(btdf)))
            if not all_trades:
                st.warning("沒有找到符合條件的歷史訊號。可降低回測最低分數或改變股票。")
            else:
                trades_df = pd.concat(all_trades, ignore_index=True)
                st.subheader("回測摘要")
                st.dataframe(summarize_backtest(trades_df), use_container_width=True, hide_index=True)
                st.subheader("歷史訊號明細")
                st.dataframe(trades_df.sort_values("日期", ascending=False), use_container_width=True, hide_index=True)
                st.download_button("下載回測明細 CSV", trades_df.to_csv(index=False).encode("utf-8-sig"), file_name=f"ace_triangle_backtest_{datetime.now():%Y%m%d_%H%M}.csv", mime="text/csv")

with help_tab:
    st.subheader("使用說明與限制")
    st.markdown(
        """
        **建議工作流**
        1. 先用「全市場自動掃描」找 A/B 級候選。
        2. 點進圖表看三角上線、下線是否合理。
        3. 把有興趣的股票放進「自選提醒」。
        4. 用「資金控管」先決定基本單與最大投入。
        5. 用「回測」檢查這檔股票過去是否適合這套邏輯。

        **資料限制**
        - Yahoo Finance 的台股資料可能延遲、缺漏，盤中即時性不足。
        - 買賣家數差不是公開免費欄位，需要你從 XQ、籌碼K線或三竹等工具自行匯出 CSV。
        - 三角畫線本來就有主觀性，程式只能做候選股初篩；正式操作仍要人工看圖確認。
        """
    )

st.markdown(
    """
    <div class='small-note'>
    免責聲明：本程式僅供教學與研究，把講義中的技術條件轉成可量化篩選器，不構成任何投資建議。
    Yahoo Finance 日K資料可能延遲或缺漏；買賣家數差若未匯入，程式無法驗證籌碼集中條件。交易前請用券商軟體/XQ/籌碼K線等工具再次人工確認。
    </div>
    """,
    unsafe_allow_html=True,
)
