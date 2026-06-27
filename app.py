# -*- coding: utf-8 -*-
"""
ACE Triangle Strategy Stock Screener V8
A mobile-friendly Streamlit app that screens Taiwan stocks for triangle-contraction patterns.

This program is an educational implementation of the rules described in the uploaded lecture.
It is NOT financial advice. Always verify charts manually before trading.
"""
from __future__ import annotations

import io
import os
import json
import re
import time
import sqlite3
import traceback
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import requests

try:
    import yfinance as yf
except Exception:  # pragma: no cover
    yf = None


# -----------------------------
# Local storage / V5 persistence
# -----------------------------
APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
HISTORY_FILE = DATA_DIR / "scan_history.csv"
ERROR_LOG_FILE = DATA_DIR / "error_log.csv"
SETTINGS_FILE = DATA_DIR / "settings.json"
DB_FILE = DATA_DIR / "ace_triangle.db"
OHLCV_CACHE_DIR = DATA_DIR / "ohlcv_cache"
OHLCV_CACHE_DIR.mkdir(exist_ok=True)
PAPER_TRADES_FILE = DATA_DIR / "paper_trades.csv"
REPORTS_DIR = DATA_DIR / "reports"
REPORTS_DIR.mkdir(exist_ok=True)



# -----------------------------
# Page / Mobile UI
# -----------------------------
st.set_page_config(
    page_title="艾斯三角收斂選股器 V8",
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


# -----------------------------
# V6: data reliability, SQLite, logging
# -----------------------------
def log_error(context: str, ticker: str = "", detail: str = "", exc: Optional[BaseException] = None) -> None:
    """Append a lightweight error record. Never interrupts the app."""
    try:
        row = {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "context": context,
            "ticker": ticker,
            "detail": detail or (str(exc)[:300] if exc else ""),
            "trace": traceback.format_exc(limit=2)[:800] if exc else "",
        }
        out = pd.DataFrame([row])
        if ERROR_LOG_FILE.exists():
            old = pd.read_csv(ERROR_LOG_FILE, encoding="utf-8-sig")
            out = pd.concat([old, out], ignore_index=True).tail(2000)
        out.to_csv(ERROR_LOG_FILE, index=False, encoding="utf-8-sig")
    except Exception:
        pass


def _cache_file_for_ticker(ticker: str, period: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", f"{ticker}_{period}")
    return OHLCV_CACHE_DIR / f"{safe}.csv"


def save_ohlcv_cache(ticker: str, period: str, df: pd.DataFrame) -> None:
    try:
        if df is None or df.empty:
            return
        out = df.copy()
        out.index.name = "Date"
        out.to_csv(_cache_file_for_ticker(ticker, period), encoding="utf-8-sig")
    except Exception as exc:
        log_error("save_ohlcv_cache", ticker, str(_cache_file_for_ticker(ticker, period)), exc)


def load_ohlcv_cache(ticker: str, period: str) -> pd.DataFrame:
    try:
        path = _cache_file_for_ticker(ticker, period)
        if not path.exists():
            return pd.DataFrame()
        df = pd.read_csv(path, encoding="utf-8-sig", parse_dates=["Date"])
        df = df.set_index("Date")
        return clean_yf_df(df)
    except Exception as exc:
        log_error("load_ohlcv_cache", ticker, period, exc)
        return pd.DataFrame()


def _to_float_tw(value) -> float:
    """Parse TW market numeric strings like '1,234', '--', or '除權息' safely."""
    try:
        s = str(value).strip().replace(",", "").replace("--", "").replace("－", "")
        s = re.sub(r"[^0-9.\-]", "", s)
        return float(s) if s not in ["", ".", "-"] else np.nan
    except Exception:
        return np.nan


def _parse_tw_market_date(value) -> pd.Timestamp:
    """Parse ROC dates such as 113/09/02 or AD dates such as 2024/09/02."""
    s = str(value).strip()
    parts = re.split(r"[/\-.]", s)
    if len(parts) >= 3 and all(x.strip().isdigit() for x in parts[:3]):
        y, m, d = [int(x) for x in parts[:3]]
        if y < 1911:
            y += 1911
        return pd.Timestamp(year=y, month=m, day=d)
    return pd.to_datetime(value, errors="coerce")


def _month_starts(months_back: int) -> List[pd.Timestamp]:
    today = pd.Timestamp.today().normalize().replace(day=1)
    months = []
    for i in range(max(1, int(months_back))):
        months.append(today - pd.DateOffset(months=i))
    return months


def _official_ohlcv_from_payload(payload: dict, market: str) -> pd.DataFrame:
    """Convert a TWSE/TPEX JSON payload into OHLCV. Designed to be tolerant to minor schema changes."""
    if not isinstance(payload, dict):
        return pd.DataFrame()
    data = payload.get("data") or payload.get("aaData") or payload.get("tables") or []
    fields = payload.get("fields") or payload.get("columns") or payload.get("headers") or []
    if isinstance(data, dict):
        # Some newer APIs wrap rows under a key.
        for v in data.values():
            if isinstance(v, list):
                data = v
                break
    if not isinstance(data, list) or not data:
        return pd.DataFrame()
    if not fields or not isinstance(fields, list):
        # Reasonable default for TWSE STOCK_DAY-like rows.
        fields = ["日期", "成交股數", "成交金額", "開盤價", "最高價", "最低價", "收盤價", "漲跌價差", "成交筆數"]
    rows = []
    for raw in data:
        if not isinstance(raw, (list, tuple)) or len(raw) < 6:
            continue
        mapping = {str(fields[i]): raw[i] for i in range(min(len(fields), len(raw)))}
        def find_value(keywords: List[str], fallback_idx: Optional[int] = None):
            for k, v in mapping.items():
                if all(kw in str(k) for kw in keywords):
                    return v
            if fallback_idx is not None and fallback_idx < len(raw):
                return raw[fallback_idx]
            return np.nan
        dt = _parse_tw_market_date(raw[0])
        if pd.isna(dt):
            continue
        open_v = find_value(["開盤"], 3)
        high_v = find_value(["最高"], 4)
        low_v = find_value(["最低"], 5)
        close_v = find_value(["收盤"], 6)
        vol_v = find_value(["成交", "股"], 1)
        # TPEX old endpoint sometimes reports trading volume in shares as well; keep as-is.
        o, h, l, c, v = [_to_float_tw(x) for x in [open_v, high_v, low_v, close_v, vol_v]]
        if not all(np.isfinite(x) for x in [o, h, l, c]):
            continue
        rows.append({"Date": dt, "Open": o, "High": h, "Low": l, "Close": c, "Volume": v if np.isfinite(v) else 0})
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows).drop_duplicates("Date").sort_values("Date").set_index("Date")
    return clean_yf_df(out)


def fetch_twse_official_month(code: str, month: pd.Timestamp) -> pd.DataFrame:
    """Fetch one month of listed-stock OHLCV from TWSE STOCK_DAY endpoint."""
    date_s = month.strftime("%Y%m01")
    urls = [
        f"https://www.twse.com.tw/exchangeReport/STOCK_DAY?response=json&date={date_s}&stockNo={code}",
        f"https://www.twse.com.tw/rwd/zh/afterTrading/STOCK_DAY?response=json&date={date_s}&stockNo={code}",
    ]
    headers = {"User-Agent": "Mozilla/5.0"}
    for url in urls:
        try:
            resp = requests.get(url, headers=headers, timeout=12)
            if resp.status_code != 200:
                continue
            payload = resp.json()
            df = _official_ohlcv_from_payload(payload, "TWSE")
            if not df.empty:
                return df
        except Exception as exc:
            log_error("fetch_twse_official_month", code, url, exc)
    return pd.DataFrame()


def fetch_tpex_official_month(code: str, month: pd.Timestamp) -> pd.DataFrame:
    """Fetch one month of OTC OHLCV from TPEx. Tries both current and legacy public endpoints."""
    headers = {"User-Agent": "Mozilla/5.0"}
    roc = f"{month.year - 1911}/{month.month:02d}"
    ad_slash = month.strftime("%Y/%m/01")
    urls = [
        f"https://www.tpex.org.tw/www/zh-tw/afterTrading/tradingStock?code={code}&date={ad_slash}&response=json",
        f"https://www.tpex.org.tw/web/stock/aftertrading/daily_trading_info/st43_result.php?l=zh-tw&d={roc}&stkno={code}",
    ]
    for url in urls:
        try:
            resp = requests.get(url, headers=headers, timeout=12)
            if resp.status_code != 200:
                continue
            payload = resp.json()
            df = _official_ohlcv_from_payload(payload, "TPEX")
            if not df.empty:
                return df
        except Exception as exc:
            log_error("fetch_tpex_official_month", code, url, exc)
    return pd.DataFrame()


@st.cache_data(ttl=60 * 60 * 12, show_spinner=False)
def fetch_official_ohlcv(ticker: str, months_back: int = 6) -> pd.DataFrame:
    """Official-source fallback for Taiwan stocks.

    It is intentionally used as a fallback/supplement rather than the main source for full-market scans,
    because official monthly endpoints are slower than Yahoo. It is most useful when Yahoo data is stale,
    empty, or missing recent bars.
    """
    ticker = normalize_code(ticker) if "." not in str(ticker) else str(ticker).upper().strip()
    code = pure_code(ticker)
    if not re.fullmatch(r"\d{4,6}", code):
        return pd.DataFrame()
    listed = ticker.endswith(".TW")
    frames = []
    for month in _month_starts(months_back):
        df = fetch_twse_official_month(code, month) if listed else fetch_tpex_official_month(code, month)
        if not df.empty:
            frames.append(df)
            time.sleep(0.03)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames).drop_duplicates().sort_index()
    out = out[~out.index.duplicated(keep="last")]
    return clean_yf_df(out)


def merge_ohlcv_keep_latest(primary: pd.DataFrame, supplement: pd.DataFrame) -> pd.DataFrame:
    if primary is None or primary.empty:
        return clean_yf_df(supplement)
    if supplement is None or supplement.empty:
        return clean_yf_df(primary)
    p = primary.copy(); s = supplement.copy()
    p["_source_rank"] = 1
    s["_source_rank"] = 2
    merged = pd.concat([p, s]).sort_index()
    merged = merged.sort_values("_source_rank").drop(columns=["_source_rank"]).groupby(level=0).last()
    return clean_yf_df(merged.sort_index())


def ohlcv_quality_report(df: pd.DataFrame, ticker: str = "") -> Dict[str, object]:
    """Return a compact data-quality report that can be surfaced in scan results."""
    if df is None or df.empty:
        return {"狀態": "無資料", "天數": 0, "最新日期": "-", "警示": "抓不到日K資料"}
    warnings: List[str] = []
    d = df.copy()
    if isinstance(d.index, pd.DatetimeIndex):
        d = d.sort_index()
        latest = d.index[-1]
        latest_s = latest.strftime("%Y-%m-%d")
        age_days = (pd.Timestamp.now().normalize() - pd.Timestamp(latest).normalize()).days
        if age_days >= 5:
            warnings.append(f"最新日K距今{age_days}天，請確認是否休市或資料延遲")
    else:
        latest_s = str(d.index[-1])
    need = ["Open", "High", "Low", "Close", "Volume"]
    if not all(c in d.columns for c in need):
        warnings.append("欄位不完整")
    if len(d) < 90:
        warnings.append("歷史資料不足90日")
    if len(d) >= 2 and (d[need].tail(90).isna().any().any()):
        warnings.append("近90日有缺值")
    if "Volume" in d.columns and len(d) >= 20 and (pd.to_numeric(d["Volume"].tail(20), errors="coerce") <= 0).mean() > 0.3:
        warnings.append("近20日零量比例偏高")
    if all(c in d.columns for c in ["High", "Low", "Close"]):
        bad = ((d["High"] < d[["Open", "Close", "Low"]].max(axis=1)) | (d["Low"] > d[["Open", "Close", "High"]].min(axis=1))).tail(120).sum()
        if bad:
            warnings.append(f"近120日有{int(bad)}筆OHLC邏輯異常")
    status = "正常" if not warnings else "需檢查"
    return {"狀態": status, "天數": int(len(d)), "最新日期": latest_s, "警示": "；".join(warnings)}


def init_db() -> None:
    try:
        with sqlite3.connect(DB_FILE) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS scan_history (
                    scan_id TEXT,
                    scan_time TEXT,
                    code TEXT,
                    ticker TEXT,
                    name TEXT,
                    industry TEXT,
                    data_date TEXT,
                    close REAL,
                    score INTEGER,
                    grade TEXT,
                    action TEXT,
                    pattern TEXT,
                    maturity TEXT,
                    upper_price REAL,
                    lower_price REAL,
                    tags TEXT,
                    warnings TEXT,
                    note TEXT,
                    raw_json TEXT,
                    PRIMARY KEY (scan_id, code)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS app_events (
                    event_time TEXT,
                    event_type TEXT,
                    ticker TEXT,
                    detail TEXT
                )
                """
            )
    except Exception as exc:
        log_error("init_db", detail=str(DB_FILE), exc=exc)


def append_scan_results_to_db(df: pd.DataFrame) -> None:
    if df is None or df.empty:
        return
    try:
        init_db()
        rows = []
        for _, r in df.iterrows():
            raw = r.to_dict()
            rows.append({
                "scan_id": str(r.get("scan_id", "")),
                "scan_time": str(r.get("scan_time", "")),
                "code": str(r.get("代號", "")),
                "ticker": str(r.get("Ticker", "")),
                "name": str(r.get("名稱", "")),
                "industry": str(r.get("產業", "")),
                "data_date": str(r.get("日期", "")),
                "close": safe_float(r.get("收盤", np.nan)),
                "score": int(safe_float(r.get("評分", 0), 0)),
                "grade": str(r.get("等級", "")),
                "action": str(r.get("動作", "")),
                "pattern": str(r.get("型態", "")),
                "maturity": str(r.get("成熟度", "")),
                "upper_price": safe_float(r.get("上線", np.nan)),
                "lower_price": safe_float(r.get("下線", np.nan)),
                "tags": str(r.get("標籤", "")),
                "warnings": str(r.get("警示", "")),
                "note": str(r.get("note", "")),
                "raw_json": json.dumps(raw, ensure_ascii=False, default=str),
            })
        with sqlite3.connect(DB_FILE) as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO scan_history
                (scan_id, scan_time, code, ticker, name, industry, data_date, close, score, grade, action, pattern, maturity, upper_price, lower_price, tags, warnings, note, raw_json)
                VALUES (:scan_id, :scan_time, :code, :ticker, :name, :industry, :data_date, :close, :score, :grade, :action, :pattern, :maturity, :upper_price, :lower_price, :tags, :warnings, :note, :raw_json)
                """,
                rows,
            )
    except Exception as exc:
        log_error("append_scan_results_to_db", detail=f"rows={len(df)}", exc=exc)


def load_scan_history_from_db() -> pd.DataFrame:
    try:
        init_db()
        with sqlite3.connect(DB_FILE) as conn:
            return pd.read_sql_query("SELECT * FROM scan_history ORDER BY scan_time DESC", conn)
    except Exception as exc:
        log_error("load_scan_history_from_db", detail=str(DB_FILE), exc=exc)
        return pd.DataFrame()


def load_app_settings() -> Dict[str, object]:
    try:
        if SETTINGS_FILE.exists():
            return json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
    except Exception as exc:
        log_error("load_app_settings", detail=str(SETTINGS_FILE), exc=exc)
    return {}


def save_app_settings(settings: Dict[str, object]) -> None:
    try:
        SETTINGS_FILE.write_text(json.dumps(settings, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as exc:
        log_error("save_app_settings", detail=str(SETTINGS_FILE), exc=exc)


def cache_summary() -> pd.DataFrame:
    rows = []
    for p in sorted(OHLCV_CACHE_DIR.glob("*.csv")):
        try:
            stat = p.stat()
            rows.append({"檔案": p.name, "大小KB": round(stat.st_size/1024, 1), "更新時間": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")})
        except Exception:
            pass
    return pd.DataFrame(rows).sort_values("更新時間", ascending=False) if rows else pd.DataFrame()


@st.cache_data(ttl=60 * 60 * 6, show_spinner=False)
def fetch_ohlcv(ticker: str, period: str = "18mo") -> pd.DataFrame:
    """Fetch daily OHLCV with Yahoo + local cache + official TWSE/TPEx fallback.

    V8 change: Yahoo remains the fast primary source. If Yahoo is empty/stale, the app
    supplements it with Taiwan official monthly data when possible, then falls back to
    local cache. This keeps tomorrow's manual tests from being blocked by one data source.
    """
    ticker = normalize_code(ticker) if "." not in str(ticker).upper() and str(ticker).strip() else str(ticker).upper().strip()
    cached = load_ohlcv_cache(ticker, period)
    cleaned = pd.DataFrame()

    if yf is not None:
        try:
            df = yf.download(
                ticker,
                period=period,
                interval="1d",
                auto_adjust=False,
                progress=False,
                threads=False,
            )
            cleaned = clean_yf_df(df)
        except Exception as exc:
            log_error("fetch_ohlcv_yahoo", ticker, period, exc)
    else:
        log_error("fetch_ohlcv", ticker, "yfinance is not installed")

    base = cleaned if not cleaned.empty else cached
    # Official fallback/supplement for Taiwanese stocks. We only request recent months when
    # Yahoo/cache exists, and a longer range only when both are empty. This avoids making
    # all-market scans unnecessarily slow.
    try:
        needs_recent_patch = False
        if base is None or base.empty:
            needs_recent_patch = True
            months = 18
        elif ticker.endswith((".TW", ".TWO")) and isinstance(base.index, pd.DatetimeIndex):
            age_days = (pd.Timestamp.now().normalize() - pd.Timestamp(base.index[-1]).normalize()).days
            needs_recent_patch = age_days >= 2
            months = 4
        else:
            months = 0
        if needs_recent_patch and ticker.endswith((".TW", ".TWO")):
            official = fetch_official_ohlcv(ticker, months_back=months)
            if not official.empty:
                base = merge_ohlcv_keep_latest(base, official)
                log_error("fetch_ohlcv_official_patch", ticker, f"官方資料補強 {len(official)} 筆")
    except Exception as exc:
        log_error("fetch_ohlcv_official_patch", ticker, period, exc)

    final = clean_yf_df(base)
    if not final.empty:
        save_ohlcv_cache(ticker, period, final)
        return final
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




def triangle_quality_profile(df: pd.DataFrame, tri: TriangleResult, tol: float = 0.018) -> Dict[str, object]:
    """V7: score the triangle shape itself, separated from trade/action score."""
    if tri is None or not tri.is_valid or tri.upper_line is None or tri.lower_line is None or df is None or df.empty:
        return {"score": 0, "label": "未形成", "details": ["三角條件不足"], "warnings": []}
    details: List[str] = []
    warnings: List[str] = []
    score = 0

    if tri.touch_upper >= 2:
        score += 15; details.append(f"上線觸碰 {tri.touch_upper} 次")
    else:
        warnings.append("上線觸碰次數偏少")
    if tri.touch_lower >= 2:
        score += 15; details.append(f"下線觸碰 {tri.touch_lower} 次")
    else:
        warnings.append("下線觸碰次數偏少")

    inside_ratio = safe_float(tri.inside_ratio, 0)
    if inside_ratio >= 0.88:
        score += 18; details.append(f"線內比例 {inside_ratio:.0%}")
    elif inside_ratio >= 0.78:
        score += 10; details.append(f"線內比例尚可 {inside_ratio:.0%}")
    else:
        warnings.append(f"線內比例偏低 {inside_ratio:.0%}")

    width_ratio = safe_float(tri.width_ratio, np.nan)
    if np.isfinite(width_ratio) and width_ratio <= 0.55:
        score += 14; details.append("收斂幅度明確")
    elif np.isfinite(width_ratio) and width_ratio <= 0.75:
        score += 8; details.append("收斂幅度尚可")
    else:
        warnings.append("收斂幅度不足")

    if tri.maturity == "收斂尾端":
        score += 14; details.append("已接近收斂尾端")
    elif tri.maturity == "收斂中段":
        score += 8; details.append("處於收斂中段")
    else:
        warnings.append("收斂成熟度偏早")

    upper_change = safe_float(tri.upper_change_pct, 0)
    lower_change = safe_float(tri.lower_change_pct, 0)
    if tri.pattern == "上升三角" and lower_change > -0.05:
        score += 8; details.append("下線墊高/持平，符合上升三角精神")
    elif tri.pattern == "對稱三角" and upper_change < 0 and lower_change > 0:
        score += 8; details.append("上下線對稱收斂")
    elif tri.pattern == "下降三角":
        score += 4; warnings.append("下降三角較適合等突破確認")
    else:
        score += 4

    try:
        recent = df.tail(min(35, len(df))).copy()
        if "VOL20" in recent.columns:
            vol_slope = np.polyfit(np.arange(len(recent)), recent["Volume"].fillna(method="ffill").to_numpy(dtype=float), 1)[0]
            if vol_slope < 0 or safe_float(recent["Volume"].iloc[-1] / max(recent["VOL20"].iloc[-1], 1), 1) <= 0.85:
                score += 8; details.append("收斂期量能有萎縮")
            else:
                warnings.append("量能未明顯萎縮")
    except Exception:
        pass

    score = int(max(0, min(100, score)))
    if score >= 82:
        label = "優秀"
    elif score >= 68:
        label = "良好"
    elif score >= 52:
        label = "普通"
    else:
        label = "偏弱"
    return {"score": score, "label": label, "details": details, "warnings": warnings}


def risk_reward_profile(price: float, upper: float, lower: float, breakout: bool = False) -> Dict[str, object]:
    """V7: approximate reward/risk using current price, lower stop, and upper/breakout objective."""
    if not all(np.isfinite(x) for x in [price, upper, lower]) or price <= 0 or upper <= lower:
        return {"rr": np.nan, "label": "無法計算", "risk_pct": np.nan, "reward_pct": np.nan, "decision": "資料不足"}
    risk_pct = max((price - lower) / price, 0.001)
    width = upper - lower
    target = upper + 0.55 * width if breakout or price >= upper else upper
    reward_pct = max((target - price) / price, 0.0)
    rr = reward_pct / risk_pct if risk_pct > 0 else np.nan
    if rr >= 3:
        label = "優秀"
        decision = "優先觀察"
    elif rr >= 2:
        label = "良好"
        decision = "可列入候選"
    elif rr >= 1.5:
        label = "普通"
        decision = "需搭配型態/籌碼"
    else:
        label = "不足"
        decision = "不宜追價"
    return {"rr": rr, "label": label, "risk_pct": risk_pct * 100, "reward_pct": reward_pct * 100, "decision": decision, "target": target}


def side_style_profile(flags: Dict[str, bool], tri: TriangleResult, price: float, upper: float, lower: float, rr: float) -> Dict[str, object]:
    """V7: separate left-side and right-side suitability scores."""
    left = 0
    right = 0
    if flags.get("support_touch"):
        left += 26
    if flags.get("small_k_vol_contract"):
        left += 22
    if flags.get("touch_60ma"):
        left += 18
    if flags.get("kd_cross"):
        left += 14
    if np.isfinite(price) and np.isfinite(lower) and lower > 0 and abs(price / lower - 1) <= 0.04:
        left += 12
    if np.isfinite(rr) and rr >= 2:
        left += 8

    if flags.get("breakout"):
        right += 35
    if flags.get("pullback_upper"):
        right += 30
    if tri.pattern in ["上升三角", "對稱三角"]:
        right += 10
    if np.isfinite(price) and np.isfinite(upper) and price >= upper * 0.985:
        right += 12
    if tri.maturity == "收斂尾端":
        right += 8
    if np.isfinite(rr) and rr >= 1.7:
        right += 5

    left = int(max(0, min(100, left)))
    right = int(max(0, min(100, right)))
    if left > right + 12:
        preferred = "左側試單"
    elif right > left + 12:
        preferred = "右側突破"
    else:
        preferred = "平衡觀察"
    return {"left": left, "right": right, "preferred": preferred}


def style_mode_adjustment(style_mode: str, flags: Dict[str, bool], rr: float, left_score: int, right_score: int, tri: TriangleResult) -> Tuple[int, List[str], List[str], List[str]]:
    """V7: adapt ranking to user's operating style without changing the raw signal logic."""
    mode = style_mode or "平衡"
    delta = 0
    tags: List[str] = []
    reasons: List[str] = []
    warnings: List[str] = []
    if mode == "左側模式":
        if left_score >= 55:
            delta += 8; tags.append("左側模式加分"); reasons.append("接近下線/量縮/60MA/KD 等左側條件較完整")
        if flags.get("breakout") and right_score > left_score + 20:
            delta -= 5; warnings.append("左側模式：已偏突破段，注意追價")
    elif mode == "右側模式":
        if right_score >= 55:
            delta += 8; tags.append("右側模式加分"); reasons.append("突破/回測上線條件較完整")
        if left_score > right_score + 25 and not flags.get("breakout"):
            delta -= 4; warnings.append("右側模式：尚未突破，先列觀察")
    elif mode == "保守模式":
        if np.isfinite(rr) and rr >= 2.0 and tri.maturity == "收斂尾端":
            delta += 6; tags.append("保守模式：風報比佳")
        if not np.isfinite(rr) or rr < 1.5:
            delta -= 8; warnings.append("保守模式：風報比不足")
        if tri.pattern == "下降三角":
            delta -= 5; warnings.append("保守模式：下降三角需等突破確認")
    elif mode == "短波段模式":
        if flags.get("breakout") or flags.get("pullback_upper"):
            delta += 5; tags.append("短波段模式：表態較明確")
        if tri.maturity == "剛成形":
            delta -= 4
    elif mode == "中波段模式":
        if tri.pattern in ["上升三角", "對稱三角"] and tri.maturity in ["收斂中段", "收斂尾端"]:
            delta += 4; tags.append("中波段模式：型態可追蹤")
        if flags.get("long_upper_black") or flags.get("bb_leave_upper"):
            delta -= 7; warnings.append("中波段模式：出現獲利/反轉警示")
    return delta, tags, reasons, warnings


def no_trade_filters(df: pd.DataFrame, tri: TriangleResult, price: float, upper_today: float, lower_today: float, market_bias: int = 0, breakout: bool = False) -> Tuple[int, List[str], List[str]]:
    """V6 negative filters: explicit 'do not trade' conditions.

    These do not automatically hide a stock, because the user may still want to watch it,
    but they reduce the score and surface a clear warning.
    """
    score_delta = 0
    tags: List[str] = []
    warnings: List[str] = []
    if not np.isfinite(price) or price <= 0:
        return score_delta, tags, warnings

    if market_bias <= -10 and not breakout:
        score_delta -= 8
        tags.append("不要交易：大盤偏空")
        warnings.append("不要交易濾網：大盤偏空且個股尚未有效突破，建議降低倉位或只觀察")

    if np.isfinite(upper_today) and np.isfinite(lower_today) and upper_today > lower_today:
        upside = (upper_today - price) / price
        downside = max((price - lower_today) / price, 0.001)
        rr = upside / downside if downside > 0 else np.nan
        # If price is in the middle/upper part of the triangle, the stop distance may be too far.
        if not breakout and np.isfinite(rr) and rr < 1.15 and price > lower_today * 1.035:
            score_delta -= 8
            tags.append("不要交易：風報比不足")
            warnings.append(f"不要交易濾網：離下線停損距離相對太遠，至上線/至下線風報比約 {rr:.2f}")
        width = (upper_today - lower_today) / price
        if width > 0.32:
            score_delta -= 7
            tags.append("不要交易：開口過大")
            warnings.append("不要交易濾網：三角開口過大，波段停損距離不漂亮")

    if len(df) >= 80:
        t = df.iloc[-1]
        if np.isfinite(safe_float(t.get("MA60", np.nan))) and price < safe_float(t.get("MA60")) * 0.93 and not breakout:
            score_delta -= 6
            tags.append("不要交易：季線下方偏弱")
            warnings.append("不要交易濾網：股價明顯低於60MA，除非是明確左側試單，否則不宜當成主策略候選")
        if len(df) >= 21:
            rise20 = safe_float(price / df["Close"].iloc[-21] - 1, 0.0)
            vol_ratio = safe_float(t.get("Volume", np.nan) / max(safe_float(t.get("VOL20", np.nan), 0.0), 1), np.nan)
            if rise20 > 0.55 and vol_ratio > 1.8 and not breakout:
                score_delta -= 7
                tags.append("不要交易：短線過熱")
                warnings.append("不要交易濾網：近20日漲幅與量能都偏熱，容易追在震盪區")

    return score_delta, tags, warnings

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
    style_mode: str = "平衡模式",
    debug_show_all: bool = False,
) -> Optional[Dict]:
    if df_raw is None or df_raw.empty or len(df_raw) < max(lookback, 90):
        return None
    quality = ohlcv_quality_report(df_raw, ticker)
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

    # V5 anti-noise filters: avoid signals that are hard to execute or too extended.
    recent_30 = df.tail(min(30, len(df))).copy()
    recent_prev_close = recent_30["Close"].shift(1)
    gap_ratio_30 = float(((recent_30["Open"] / recent_prev_close - 1).abs() > 0.07).sum() / max(1, len(recent_30) - 1)) if len(recent_30) > 2 else 0.0
    ma20_dist = safe_float(today["Close"] / today["MA20"] - 1, 0.0) if np.isfinite(safe_float(today.get("MA20", np.nan))) else 0.0
    rise_20 = safe_float(today["Close"] / df["Close"].iloc[-21] - 1, 0.0) if len(df) >= 21 else 0.0
    if np.isfinite(avg_lots20) and avg_lots20 < 300:
        score -= 14
        warnings.append("反過濾：20日均量低於300張，流動性與滑價風險較高")
        tags.append("⚠流動性低")
    if gap_ratio_30 >= 0.18:
        score -= 6
        warnings.append("反過濾：近30日跳空比例偏高，停損價較難控制")
        tags.append("⚠跳空風險")
    if ma20_dist > 0.22:
        score -= 7
        warnings.append("反過濾：股價距20MA過遠，追價風險升高")
        tags.append("⚠短線過熱")
    if rise_20 > 0.45 and not breakout:
        score -= 5
        warnings.append("反過濾：近20日漲幅過大且非初突破，需防震盪洗盤")
    if tri.width_ratio is not None and np.isfinite(safe_float(tri.width_ratio)) and np.isfinite(safe_float(tri.upper_today)) and np.isfinite(safe_float(tri.lower_today)):
        raw_width = safe_float((tri.upper_today - tri.lower_today) / max(price, 0.01), 0.0)
        if raw_width > 0.26:
            score -= 6
            warnings.append("反過濾：三角開口仍大，風險報酬比可能不足")
            tags.append("⚠開口偏大")

    # V6 explicit no-trade filters: warn when the setup exists but the risk/reward or context is poor.
    nt_delta, nt_tags, nt_warnings = no_trade_filters(df, tri, price, upper_today, lower_today, market_bias=market_bias, breakout=breakout)
    score += nt_delta
    tags += nt_tags
    warnings += nt_warnings

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

    v7_flags = {
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
    }
    quality_profile = triangle_quality_profile(df, tri, tol)
    rr_profile = risk_reward_profile(price, upper_today, lower_today, breakout=breakout)
    side_profile = side_style_profile(v7_flags, tri, price, upper_today, lower_today, safe_float(rr_profile.get("rr"), np.nan))
    style_delta, style_tags, style_reasons, style_warnings = style_mode_adjustment(style_mode, v7_flags, safe_float(rr_profile.get("rr"), np.nan), side_profile["left"], side_profile["right"], tri)
    score += style_delta
    tags += style_tags
    reasons += style_reasons
    warnings += style_warnings
    if quality_profile["score"] >= 82:
        score += 4
        tags.append("型態品質優秀")
    elif quality_profile["score"] < 52 and tri.is_valid:
        score -= 5
        warnings.append("V7 型態品質分偏低，建議人工看圖確認是否只是雜訊")
    if np.isfinite(safe_float(rr_profile.get("rr"), np.nan)):
        rr_value = safe_float(rr_profile.get("rr"), np.nan)
        if rr_value >= 2.0:
            score += 4
            tags.append("風報比佳")
        elif rr_value < 1.2 and not breakout:
            score -= 6
            warnings.append("V7 風險報酬比偏低，不宜追價")

    score = int(max(0, min(100, score)))

    if not tri.is_valid and score < 55 and not (include_early and touch_60ma and kd_cross):
        if not debug_show_all:
            return None
        tags.append("除錯：未達三角核心條件")
        warnings.append("除錯模式：此股資料可讀取，但目前未形成有效三角或早期試單訊號，所以正式模式會被排除")
        action = "除錯：未達條件"

    return {
        "代號": code,
        "名稱": name or code,
        "產業": industry or "未分類",
        "Ticker": ticker,
        "大盤": market_label,
        "日期": df.index[-1].strftime("%Y-%m-%d") if hasattr(df.index[-1], "strftime") else str(df.index[-1]),
        "資料狀態": quality.get("狀態", "-"),
        "資料天數": quality.get("天數", 0),
        "最新資料日": quality.get("最新日期", "-"),
        "資料警示": quality.get("警示", ""),
        "收盤": round(price, 2),
        "評分": score,
        "等級": grade_from_score(score),
        "動作": action,
        "操作模式": style_mode,
        "型態": tri.pattern,
        "型態品質": quality_profile.get("label", "-"),
        "型態品質分": quality_profile.get("score", 0),
        "風報比": round(safe_float(rr_profile.get("rr"), np.nan), 2) if np.isfinite(safe_float(rr_profile.get("rr"), np.nan)) else np.nan,
        "風險%": round(safe_float(rr_profile.get("risk_pct"), np.nan), 2) if np.isfinite(safe_float(rr_profile.get("risk_pct"), np.nan)) else np.nan,
        "潛在報酬%": round(safe_float(rr_profile.get("reward_pct"), np.nan), 2) if np.isfinite(safe_float(rr_profile.get("reward_pct"), np.nan)) else np.nan,
        "風報判斷": rr_profile.get("label", "-"),
        "偏好型態": side_profile.get("preferred", "-"),
        "左側分": side_profile.get("left", 0),
        "右側分": side_profile.get("right", 0),
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
        "_flags": v7_flags,
        "_v7": {
            "quality_profile": quality_profile,
            "risk_reward": rr_profile,
            "side_profile": side_profile,
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




def strategy_backtest_single(
    ticker: str,
    name: str,
    df_raw: pd.DataFrame,
    lookback: int,
    tol: float,
    tail_width_limit: float,
    min_score_bt: int,
    capital: float = 1_000_000,
    unit_pct: float = 10.0,
    max_units: int = 4,
    fee_rate: float = 0.001425,
    tax_rate: float = 0.003,
    slippage: float = 0.001,
) -> pd.DataFrame:
    """V6 full-process backtest that approximates the lecture-style scale-in/scale-out workflow.

    It is intentionally conservative: entries/exits use closing price plus costs and slippage.
    The goal is not perfect execution simulation, but to verify whether the complete rule set
    behaves reasonably before the user considers it in real life.
    """
    if df_raw is None or df_raw.empty or len(df_raw) < max(180, lookback + 90):
        return pd.DataFrame()
    unit_cash = capital * unit_pct / 100
    max_units = int(max(1, max_units))
    trades: List[Dict[str, object]] = []
    position_shares = 0
    cash_in_position = 0.0
    units = 0
    avg_cost = np.nan
    entry_date = ""
    entry_reasons: List[str] = []
    last_add_i = -999

    start_i = max(140, lookback + 80)
    max_i = len(df_raw) - 1
    for i in range(start_i, max_i + 1):
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
            market_bias=0,
            market_label="策略回測未套大盤",
        )
        if res is None:
            continue
        date_s = df_raw.index[i].strftime("%Y-%m-%d") if hasattr(df_raw.index[i], "strftime") else str(df_raw.index[i])
        close = safe_float(df_raw["Close"].iloc[i])
        if close <= 0:
            continue
        action = str(res.get("動作", ""))
        tags = str(res.get("標籤", ""))
        should_enter = res.get("評分", 0) >= min_score_bt and any(k in action for k in ["基本單", "早期試單", "突破加碼", "回測上線"])
        should_add = position_shares > 0 and res.get("評分", 0) >= min_score_bt and units < max_units and ("突破加碼" in action or "回測上線" in action or "下線不破" in tags)
        should_exit = position_shares > 0 and ("停損" in action or "獲利" in action or "跌破長紅K一半" in tags or "跌破下線" in tags or "跌破上線" in tags)

        if position_shares == 0 and should_enter:
            buy_price = close * (1 + slippage)
            shares = int(unit_cash // buy_price)
            if shares <= 0:
                continue
            fee = buy_price * shares * fee_rate
            position_shares = shares
            cash_in_position = buy_price * shares + fee
            avg_cost = cash_in_position / shares
            units = 1
            entry_date = date_s
            entry_reasons = [action]
            last_add_i = i
            continue

        if should_add and i - last_add_i >= 2:
            buy_price = close * (1 + slippage)
            shares = int(unit_cash // buy_price)
            if shares > 0:
                fee = buy_price * shares * fee_rate
                cash_in_position += buy_price * shares + fee
                position_shares += shares
                avg_cost = cash_in_position / position_shares
                units += 1
                entry_reasons.append(action)
                last_add_i = i

        if should_exit:
            sell_price = close * (1 - slippage)
            gross = sell_price * position_shares
            fee = gross * fee_rate
            tax = gross * tax_rate
            net = gross - fee - tax
            pnl = net - cash_in_position
            ret = pnl / cash_in_position if cash_in_position > 0 else np.nan
            trades.append({
                "代號": pure_code(ticker), "名稱": name, "進場日": entry_date, "出場日": date_s,
                "進場理由": " / ".join(entry_reasons), "出場理由": action,
                "最大單位": units, "股數": position_shares, "平均成本": round(avg_cost, 2),
                "出場價": round(sell_price, 2), "損益": round(pnl, 0), "報酬%": round(ret * 100, 2) if np.isfinite(ret) else np.nan,
                "持有天數": len(pd.date_range(entry_date, date_s, freq="D")) if entry_date else np.nan,
                "標籤": tags,
            })
            position_shares = 0
            cash_in_position = 0.0
            units = 0
            avg_cost = np.nan
            entry_date = ""
            entry_reasons = []
            last_add_i = i

    # Mark-to-market any remaining position at the last close.
    if position_shares > 0:
        date_s = df_raw.index[-1].strftime("%Y-%m-%d") if hasattr(df_raw.index[-1], "strftime") else str(df_raw.index[-1])
        close = safe_float(df_raw["Close"].iloc[-1])
        sell_price = close * (1 - slippage)
        gross = sell_price * position_shares
        net = gross - gross * fee_rate - gross * tax_rate
        pnl = net - cash_in_position
        ret = pnl / cash_in_position if cash_in_position > 0 else np.nan
        trades.append({
            "代號": pure_code(ticker), "名稱": name, "進場日": entry_date, "出場日": date_s,
            "進場理由": " / ".join(entry_reasons), "出場理由": "期末持倉結算",
            "最大單位": units, "股數": position_shares, "平均成本": round(avg_cost, 2),
            "出場價": round(sell_price, 2), "損益": round(pnl, 0), "報酬%": round(ret * 100, 2) if np.isfinite(ret) else np.nan,
            "持有天數": len(pd.date_range(entry_date, date_s, freq="D")) if entry_date else np.nan,
            "標籤": "期末結算",
        })
    return pd.DataFrame(trades)


def summarize_strategy_trades(trades: pd.DataFrame) -> pd.DataFrame:
    if trades is None or trades.empty:
        return pd.DataFrame()
    return pd.DataFrame([{
        "交易次數": len(trades),
        "勝率": f"{(trades['損益'] > 0).mean() * 100:.1f}%",
        "平均報酬%": round(trades["報酬%"].mean(), 2),
        "總損益": round(trades["損益"].sum(), 0),
        "最大獲利": round(trades["損益"].max(), 0),
        "最大虧損": round(trades["損益"].min(), 0),
        "平均持有天數": round(trades["持有天數"].mean(), 1) if "持有天數" in trades else np.nan,
        "最大連虧": max_losing_streak(trades["損益"].tolist()),
    }])


def max_losing_streak(pnls: List[float]) -> int:
    best = cur = 0
    for p in pnls:
        if safe_float(p, 0) < 0:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best

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
def scan_rows(rows: pd.DataFrame, period: str, lookback: int, tol: float, tail_width_limit: float, chip_df: pd.DataFrame, xq_reverse: bool, use_chip: bool, avg_lot_limit: int, min_score: int, max_scan: int, market_bias: int = 0, market_label: str = "未套用", style_mode: str = "平衡模式", debug_show_all: bool = False) -> List[Dict]:
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
            style_mode=style_mode,
            debug_show_all=debug_show_all,
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
# V5 history, signal tracking, plan, export helpers
# -----------------------------
def result_to_storage_row(result: Dict, scan_id: str, note: str = "") -> Dict:
    row = {k: v for k, v in result.items() if not k.startswith("_")}
    row["scan_id"] = scan_id
    row["scan_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    row["note"] = note
    return row


def save_scan_results(results: List[Dict], note: str = "") -> str:
    if not results:
        return ""
    scan_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    rows = [result_to_storage_row(r, scan_id, note) for r in results]
    out = pd.DataFrame(rows)
    if HISTORY_FILE.exists():
        old = pd.read_csv(HISTORY_FILE, encoding="utf-8-sig")
        out = pd.concat([old, out], ignore_index=True)
    out = out.drop_duplicates(subset=["scan_id", "代號"], keep="last")
    out.to_csv(HISTORY_FILE, index=False, encoding="utf-8-sig")
    append_scan_results_to_db(out)
    return scan_id


def load_scan_history(uploaded_file=None) -> pd.DataFrame:
    frames = []
    db_hist = load_scan_history_from_db()
    if not db_hist.empty:
        # Convert SQLite canonical columns back to display-style columns used by the app.
        db_hist = db_hist.rename(columns={
            "code": "代號", "ticker": "Ticker", "name": "名稱", "industry": "產業",
            "data_date": "日期", "close": "收盤", "score": "評分", "grade": "等級",
            "action": "動作", "pattern": "型態", "maturity": "成熟度",
            "upper_price": "上線", "lower_price": "下線", "tags": "標籤", "warnings": "警示",
        })
        frames.append(db_hist)
    if HISTORY_FILE.exists():
        try:
            frames.append(pd.read_csv(HISTORY_FILE, encoding="utf-8-sig"))
        except Exception:
            pass
    if uploaded_file is not None:
        raw = uploaded_file.read()
        for enc in ["utf-8-sig", "big5", "cp950", "utf-8"]:
            try:
                frames.append(pd.read_csv(io.BytesIO(raw), encoding=enc))
                break
            except Exception:
                continue
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    if "scan_time" in df.columns:
        df["scan_time"] = pd.to_datetime(df["scan_time"], errors="coerce")
        df = df.sort_values("scan_time", ascending=False)
    subset = [c for c in ["scan_id", "代號"] if c in df.columns]
    return df.drop_duplicates(subset=subset, keep="first") if subset else df


def track_signal_outcomes(history: pd.DataFrame, default_suffix: str = ".TW", period: str = "2y") -> pd.DataFrame:
    if history is None or history.empty:
        return pd.DataFrame()
    rows = []
    work = history.copy().head(500)
    for _, row in work.iterrows():
        code = str(row.get("代號", "")).strip()
        if not code or code.lower() == "nan":
            continue
        ticker = str(row.get("Ticker", "")).strip()
        if ticker.lower() in ["", "nan", "none"]:
            ticker = normalize_code(code, default_suffix)
        df = fetch_ohlcv(ticker, period=period)
        if df is None or df.empty:
            continue
        signal_date = pd.to_datetime(row.get("日期", row.get("scan_time", None)), errors="coerce")
        if pd.isna(signal_date):
            continue
        idxs = np.where(pd.to_datetime(df.index) >= signal_date)[0]
        if len(idxs) == 0:
            continue
        i = int(idxs[0])
        entry = safe_float(df["Close"].iloc[i])
        latest = safe_float(df["Close"].iloc[-1])
        out = {k: row.get(k, "") for k in ["scan_id", "scan_time", "代號", "名稱", "產業", "日期", "收盤", "評分", "等級", "動作", "型態", "操作模式", "偏好型態", "風報判斷", "標籤"]}
        out["追蹤天數"] = int(len(df) - i - 1)
        out["訊號收盤"] = round(entry, 2) if np.isfinite(entry) else np.nan
        out["最新價"] = round(latest, 2) if np.isfinite(latest) else np.nan
        out["至今%"] = round((latest / entry - 1) * 100, 2) if entry > 0 and np.isfinite(latest) else np.nan
        for h in [1, 3, 5, 10, 20, 60]:
            if i + h < len(df) and entry > 0:
                out[f"{h}日%"] = round((safe_float(df["Close"].iloc[i + h]) / entry - 1) * 100, 2)
            else:
                out[f"{h}日%"] = np.nan
        rows.append(out)
    return pd.DataFrame(rows)


def summarize_signal_quality(outcomes: pd.DataFrame) -> pd.DataFrame:
    if outcomes is None or outcomes.empty:
        return pd.DataFrame()
    rows = []
    for col in ["動作", "型態", "等級"]:
        if col not in outcomes.columns:
            continue
        for key, sub in outcomes.groupby(col, dropna=False):
            rows.append({
                "分組": col,
                "類別": key,
                "樣本數": len(sub),
                "5日勝率": f"{(sub['5日%'] > 0).mean() * 100:.1f}%" if "5日%" in sub else "-",
                "10日勝率": f"{(sub['10日%'] > 0).mean() * 100:.1f}%" if "10日%" in sub else "-",
                "20日勝率": f"{(sub['20日%'] > 0).mean() * 100:.1f}%" if "20日%" in sub else "-",
                "5日平均%": round(sub["5日%"].mean(), 2) if "5日%" in sub else np.nan,
                "10日平均%": round(sub["10日%"].mean(), 2) if "10日%" in sub else np.nan,
                "20日平均%": round(sub["20日%"].mean(), 2) if "20日%" in sub else np.nan,
                "至今平均%": round(sub["至今%"].mean(), 2) if "至今%" in sub else np.nan,
            })
    return pd.DataFrame(rows).sort_values(["分組", "樣本數"], ascending=[True, False]) if rows else pd.DataFrame()


def optimize_parameters_from_outcomes(outcomes: pd.DataFrame, min_samples: int = 8) -> Tuple[pd.DataFrame, str]:
    """V8: derive practical parameter suggestions from tracked signal outcomes.

    This is intentionally conservative: it does not overfit by choosing dozens of parameters.
    It only suggests display threshold, preferred style/pattern/action buckets, and warnings when samples are too small.
    """
    if outcomes is None or outcomes.empty:
        return pd.DataFrame(), "目前沒有訊號追蹤結果，先累積幾次掃描與紙上交易後再優化。"
    df = outcomes.copy()
    for col in ["評分", "5日%", "10日%", "20日%", "60日%"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    horizon = "20日%" if "20日%" in df.columns and df["20日%"].notna().sum() >= min_samples else ("10日%" if "10日%" in df.columns else "5日%")
    rows = []
    if "評分" in df.columns:
        for th in [50, 55, 60, 65, 70, 75, 80, 85]:
            sub = df[df["評分"] >= th].dropna(subset=[horizon]) if horizon in df.columns else pd.DataFrame()
            if len(sub) == 0:
                continue
            rows.append({
                "類型": "最低評分門檻",
                "設定": f"評分 >= {th}",
                "樣本數": len(sub),
                "勝率": round((sub[horizon] > 0).mean() * 100, 1),
                "平均報酬%": round(sub[horizon].mean(), 2),
                "中位數報酬%": round(sub[horizon].median(), 2),
                "建議": "樣本足夠可考慮" if len(sub) >= min_samples else "樣本偏少，僅參考",
            })
    for group_col in ["動作", "型態", "等級", "操作模式", "偏好型態"]:
        if group_col not in df.columns or horizon not in df.columns:
            continue
        for key, sub in df.dropna(subset=[horizon]).groupby(group_col, dropna=False):
            if len(sub) < max(3, min_samples // 2):
                continue
            rows.append({
                "類型": group_col,
                "設定": str(key),
                "樣本數": len(sub),
                "勝率": round((sub[horizon] > 0).mean() * 100, 1),
                "平均報酬%": round(sub[horizon].mean(), 2),
                "中位數報酬%": round(sub[horizon].median(), 2),
                "建議": "優先保留" if (sub[horizon] > 0).mean() >= 0.55 and sub[horizon].mean() > 0 else "可降權/觀察",
            })
    out = pd.DataFrame(rows)
    if out.empty:
        return out, f"目前可用樣本不足；建議至少累積 {min_samples} 筆以上有 {horizon} 的訊號結果。"
    out = out.sort_values(["平均報酬%", "勝率", "樣本數"], ascending=[False, False, False]).reset_index(drop=True)
    best_threshold = out[out["類型"] == "最低評分門檻"].head(1)
    msg = f"以 {horizon} 作為優化觀察期。"
    if not best_threshold.empty:
        msg += f" 目前較佳門檻：{best_threshold.iloc[0]['設定']}（樣本 {int(best_threshold.iloc[0]['樣本數'])}，勝率 {best_threshold.iloc[0]['勝率']}%，平均 {best_threshold.iloc[0]['平均報酬%']}%）。"
    if len(df.dropna(subset=[horizon])) < min_samples:
        msg += " 但樣本仍偏少，請先當作初步方向，不要直接大幅調參。"
    return out, msg


def build_parameter_profile(opt_table: pd.DataFrame, current_min_score: int, current_style: str) -> Dict[str, object]:
    """Convert optimization table into a small JSON profile that can be saved and reviewed later."""
    profile = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "current_min_score": current_min_score,
        "current_style_mode": current_style,
        "suggested_min_score": current_min_score,
        "preferred_buckets": [],
        "note": "低樣本時僅供觀察，不建議直接大幅調整。",
    }
    if opt_table is not None and not opt_table.empty:
        th = opt_table[opt_table["類型"] == "最低評分門檻"].head(1)
        if not th.empty:
            m = re.search(r"(\d+)", str(th.iloc[0]["設定"]))
            if m:
                profile["suggested_min_score"] = int(m.group(1))
        keep = opt_table[(opt_table["建議"] == "優先保留") & (opt_table["類型"] != "最低評分門檻")].head(8)
        profile["preferred_buckets"] = keep[["類型", "設定", "樣本數", "勝率", "平均報酬%"]].to_dict("records") if not keep.empty else []
    return profile


def generate_trade_plan(result: Dict, capital: float, unit_pct: float, max_pct: float, risk_pct: float = 1.0) -> pd.DataFrame:
    price = safe_float(result.get("收盤", np.nan))
    upper = safe_float(result.get("上線", np.nan))
    lower = safe_float(result.get("下線", np.nan))
    key_levels = result.get("_key_levels", {}) if isinstance(result.get("_key_levels", {}), dict) else {}
    red_half = safe_float(key_levels.get("red_half", np.nan))
    unit_cash = capital * unit_pct / 100
    max_cash = capital * max_pct / 100
    max_risk_cash = capital * risk_pct / 100
    stop_ref = lower if np.isfinite(lower) else (price * 0.93 if np.isfinite(price) else np.nan)
    if np.isfinite(upper) and np.isfinite(price) and price > upper:
        stop_ref = upper
    per_share_risk = max(price - stop_ref, 0.01) if np.isfinite(price) and np.isfinite(stop_ref) else np.nan
    risk_shares = int(max_risk_cash // per_share_risk) if np.isfinite(per_share_risk) and per_share_risk > 0 else np.nan
    unit_shares = int(unit_cash // price) if np.isfinite(price) and price > 0 else np.nan
    rows = [
        {"階段": "觀察", "觸發條件": "型態清楚但未到進場點", "價格/規則": f"上線約 {upper:.2f}；下線約 {lower:.2f}" if np.isfinite(upper) and np.isfinite(lower) else "等圖形更明確", "建議單位": "0", "風險處理": "不追高，不預設突破"},
        {"階段": "基本單/試單", "觸發條件": "收斂尾端小K量縮，或下線不破", "價格/規則": f"靠近下線 {lower:.2f} 不破" if np.isfinite(lower) else "依下線判斷", "建議單位": "1 單位", "風險處理": "收盤跌破下線/隔天開低轉弱就退"},
        {"階段": "突破加碼", "觸發條件": "收盤初站上上線且未離上線太遠", "價格/規則": f"站上 {upper:.2f}" if np.isfinite(upper) else "站上三角上線", "建議單位": "再加 1 單位", "風險處理": "突破後若跌回上線下，先出場或降部位"},
        {"階段": "回測加碼", "觸發條件": "突破後回測上線不破", "價格/規則": f"回測 {upper:.2f} 有撐" if np.isfinite(upper) else "回測上線不破", "建議單位": "再加 1 單位", "風險處理": "跌破上線則全部檢查"},
        {"階段": "獲利/風險", "觸發條件": "離開布林上軌、翻黑長上影、跌破長紅K一半", "價格/規則": f"長紅K一半約 {red_half:.2f}" if np.isfinite(red_half) else "依圖表訊號", "建議單位": "分批減碼", "風險處理": "保留最後單位看布林或破線"},
    ]
    out = pd.DataFrame(rows)
    out.attrs["unit_cash"] = unit_cash
    out.attrs["max_cash"] = max_cash
    out.attrs["unit_shares"] = unit_shares
    out.attrs["risk_shares"] = risk_shares
    out.attrs["stop_ref"] = stop_ref
    out.attrs["max_risk_cash"] = max_risk_cash
    return out


def make_daily_message(results: List[Dict], max_items: int = 12) -> str:
    if not results:
        return "艾斯三角收斂 V8：目前沒有掃描結果。"
    plan = build_daily_plan(results)
    if plan.empty:
        top = results[:max_items]
        lines = ["艾斯三角收斂 V8｜今日候選"]
        for r in top:
            lines.append(f"{r.get('代號')} {r.get('名稱')}｜{r.get('等級')}｜{r.get('動作')}｜評分{r.get('評分')}｜收{r.get('收盤')}")
        return "\n".join(lines)
    lines = [f"艾斯三角收斂 V8｜盤後清單 {datetime.now():%Y-%m-%d %H:%M}"]
    count = 0
    for bucket, sub in plan.groupby("清單"):
        lines.append(f"\n【{bucket}】")
        for _, row in sub.head(max(1, max_items // 4)).iterrows():
            lines.append(f"{row.get('代號')} {row.get('名稱')}｜{row.get('動作')}｜評分{row.get('評分')}｜收{row.get('收盤')}｜{row.get('標籤')}")
            count += 1
            if count >= max_items:
                break
        if count >= max_items:
            break
    return "\n".join(lines)


def send_text_to_telegram(text: str, token: str, chat_id: str) -> Tuple[bool, str]:
    """Send a text message to Telegram. Returns (ok, message) for Streamlit display."""
    if not token or not chat_id:
        return False, "缺少 TELEGRAM_BOT_TOKEN 或 TELEGRAM_CHAT_ID"
    try:
        resp = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            data={"chat_id": chat_id, "text": text[:3900]},
            timeout=20,
        )
        if resp.status_code == 200:
            return True, "Telegram 已送出"
        return False, f"Telegram 回應 {resp.status_code}: {resp.text[:300]}"
    except Exception as exc:
        log_error("send_text_to_telegram", detail=str(exc), exc=exc)
        return False, f"Telegram 發送失敗：{exc}"


def send_text_to_webhook(text: str, webhook_url: str) -> Tuple[bool, str]:
    """Send report text to a generic webhook. Useful for Make/Zapier/Discord/LINE bridges."""
    if not webhook_url:
        return False, "未填 webhook URL"
    try:
        resp = requests.post(webhook_url, json={"text": text, "source": "ace_triangle_app", "time": datetime.now().isoformat()}, timeout=20)
        if 200 <= resp.status_code < 300:
            return True, "Webhook 已送出"
        return False, f"Webhook 回應 {resp.status_code}: {resp.text[:300]}"
    except Exception as exc:
        log_error("send_text_to_webhook", detail=str(exc), exc=exc)
        return False, f"Webhook 發送失敗：{exc}"


def save_daily_report_files(report_md: str, message: str) -> Dict[str, str]:
    REPORTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    md_path = REPORTS_DIR / f"ace_daily_report_{stamp}.md"
    txt_path = REPORTS_DIR / f"ace_daily_message_{stamp}.txt"
    md_path.write_text(report_md, encoding="utf-8")
    txt_path.write_text(message, encoding="utf-8")
    return {"markdown": str(md_path), "text": str(txt_path)}


def scheduled_scan_script_text() -> str:
    try:
        return (APP_DIR / "scripts" / "scheduled_scan.py").read_text(encoding="utf-8")
    except Exception:
        return "# scheduled_scan.py not found"


def github_actions_workflow_text() -> str:
    try:
        return (APP_DIR / ".github" / "workflows" / "ace_daily_scan.yml").read_text(encoding="utf-8")
    except Exception:
        return "# ace_daily_scan.yml not found"



# -----------------------------
# V7 paper trading and report helpers
# -----------------------------
def load_paper_trades() -> pd.DataFrame:
    if not PAPER_TRADES_FILE.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(PAPER_TRADES_FILE, encoding="utf-8-sig")
    except Exception as exc:
        log_error("load_paper_trades", detail=str(exc), exc=exc)
        return pd.DataFrame()


def save_paper_trades(df: pd.DataFrame) -> None:
    try:
        PAPER_TRADES_FILE.parent.mkdir(exist_ok=True)
        df.to_csv(PAPER_TRADES_FILE, index=False, encoding="utf-8-sig")
    except Exception as exc:
        log_error("save_paper_trades", detail=str(exc), exc=exc)


def append_paper_trade(row: Dict[str, object]) -> None:
    old = load_paper_trades()
    out = pd.concat([old, pd.DataFrame([row])], ignore_index=True) if not old.empty else pd.DataFrame([row])
    save_paper_trades(out)


def mark_to_market_paper_trades(df: pd.DataFrame, period: str, lookback: int, tol: float, tail_width_limit: float, avg_lot_limit: int, market_bias: int, market_label: str, style_mode: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    rows: List[Dict[str, object]] = []
    for _, tr in df.iterrows():
        row = tr.to_dict()
        ticker = str(row.get("ticker", row.get("Ticker", ""))).strip()
        if not ticker:
            rows.append(row); continue
        try:
            raw = fetch_ohlcv(ticker, period=period)
            res = evaluate_stock(ticker, str(row.get("name", row.get("code", ticker))), raw, lookback, tol, tail_width_limit, pd.DataFrame(), False, False, avg_lot_limit, include_early=True, industry="紙上交易", market_bias=market_bias, market_label=market_label, style_mode=style_mode)
            if res is not None:
                price = safe_float(res.get("收盤"), np.nan)
                entry = safe_float(row.get("entry_price"), np.nan)
                shares = safe_float(row.get("shares"), np.nan)
                row["current_price"] = round(price, 2) if np.isfinite(price) else np.nan
                row["current_action"] = res.get("動作", "-")
                row["current_score"] = res.get("評分", np.nan)
                row["current_rr"] = res.get("風報比", np.nan)
                row["current_warning"] = res.get("警示", "")
                if np.isfinite(price) and np.isfinite(entry) and entry > 0:
                    row["pnl_pct"] = round((price / entry - 1) * 100, 2)
                    row["pnl_amt"] = round((price - entry) * shares, 0) if np.isfinite(shares) else np.nan
        except Exception as exc:
            row["current_warning"] = f"更新失敗：{exc}"
            log_error("paper_mark_to_market", ticker, str(exc), exc)
        rows.append(row)
    return pd.DataFrame(rows)


def close_paper_trade(trade_id: str, exit_price: float, exit_note: str = "") -> bool:
    df = load_paper_trades()
    if df.empty or "trade_id" not in df.columns:
        return False
    mask = df["trade_id"].astype(str) == str(trade_id)
    if not mask.any():
        return False
    idx = df.index[mask][0]
    entry = safe_float(df.loc[idx, "entry_price"], np.nan)
    shares = safe_float(df.loc[idx, "shares"], np.nan)
    df.loc[idx, "status"] = "closed"
    df.loc[idx, "exit_date"] = datetime.now().strftime("%Y-%m-%d")
    df.loc[idx, "exit_price"] = exit_price
    df.loc[idx, "exit_note"] = exit_note
    if np.isfinite(entry) and entry > 0 and np.isfinite(exit_price):
        df.loc[idx, "realized_pct"] = round((exit_price / entry - 1) * 100, 2)
        if np.isfinite(shares):
            df.loc[idx, "realized_amt"] = round((exit_price - entry) * shares, 0)
    save_paper_trades(df)
    return True


def generate_strategy_report(results: List[Dict], market_ctx: Optional[Dict] = None) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [f"# 艾斯三角收斂 V8 盤後策略報告", "", f"產生時間：{now}", ""]
    if market_ctx:
        lines += [f"## 大盤狀態", f"- {market_ctx.get('combined_label', '未套用')}（分數調整 {market_ctx.get('combined_bias', 0):+d}）", ""]
    if not results:
        lines += ["今日尚無掃描結果。"]
        return "\n".join(lines)
    df = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in results])
    lines += ["## 今日摘要", f"- 候選股：{len(df)} 檔", f"- A級：{int((df['等級'] == 'A').sum())} 檔", f"- 平均評分：{safe_float(df['評分'].mean(), 0):.1f}", ""]
    if "產業" in df.columns:
        sec = sector_strength_table(results).head(8)
        if not sec.empty:
            lines += ["## 族群強度前段", "| 產業 | 候選數 | A級數 | 平均分 |", "|---|---:|---:|---:|"]
            for _, r in sec.iterrows():
                lines.append(f"| {r['產業']} | {r['候選數']} | {r['A級數']} | {r['平均評分']} |")
            lines.append("")
    top = df.sort_values(["等級", "評分"], ascending=[True, False]).head(15)
    cols = [c for c in ["代號", "名稱", "收盤", "評分", "等級", "動作", "型態", "型態品質", "風報比", "偏好型態", "上線", "下線"] if c in top.columns]
    lines += ["## 優先候選", "| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in top.iterrows():
        lines.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    lines.append("")
    daily = build_daily_plan(results)
    if not daily.empty:
        lines += ["## 明日行動清單"]
        for bucket in daily["清單"].drop_duplicates().tolist():
            sub = daily[daily["清單"] == bucket].head(8)
            names = "、".join([f"{x['代號']}({x['動作']})" for _, x in sub.iterrows()])
            lines.append(f"- {bucket}：{names}")
        lines.append("")
    lines += ["## 使用提醒", "- 這份報告只做候選與計畫整理，不是買賣建議。", "- 進出場仍需用券商軟體/XQ/TradingView確認K線、量能、籌碼與流動性。"]
    return "\n".join(lines)

# -----------------------------
# App layout
# -----------------------------
st.title("📈 艾斯三角收斂選股器 V8")
st.caption("手機友善版｜V8 新增：官方資料補強、每日報告推播、參數優化建議；保留V7模組化、紙上交易、風險報酬比與測試工具。")

with st.expander("V8 累積強化內容", expanded=False):
    st.markdown(
        """
        - **V8 多層資料保護**：Yahoo 失敗或資料過舊時，嘗試 TWSE/TPEx 官方資料補強，並讀取本機 OHLCV 快取。
        - **V6/V8 SQLite 資料庫**：掃描歷史同步寫入 `data/ace_triangle.db`，方便長期追蹤與備份。
        - **V8 資料健檢頁**：檢查快取、單檔日K品質、SQLite歷史與錯誤紀錄。
        - **V6 不要交易濾網**：大盤偏空、風報比不足、開口過大、季線下方偏弱、短線過熱都會降分與警示。
        - **V7/V8 完整分批策略回測**：模擬基本單、突破加碼、回測加碼、停損/獲利出場，並納入交易成本。
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
        - **V5 掃描歷史資料庫**：每次掃描可自動存成歷史紀錄，之後追蹤訊號後 1/3/5/10/20/60 日績效。
        - **V5 訊號品質追蹤**：依動作、型態、等級統計勝率與平均報酬，逐步找出最適合你的條件。
        - **V5 交易計畫產生器**：點選個股後自動產生進場、加碼、停損、獲利與分批單位計畫。
        - **V5 反過濾條件**：排除流動性不足、跳空風險高、過熱追價、三角開口太大的訊號。
        - **V5 排程範本**：內建 GitHub Actions / VPS 每日掃描與 Telegram 推播範本。
        """
    )

st.sidebar.header("共用參數")
period = st.sidebar.selectbox("抓取資料期間", ["1y", "18mo", "2y", "5y"], index=1)
lookback = st.sidebar.slider("三角觀察天數", 25, 90, 45, 5)
tol = st.sidebar.slider("畫線容忍度", 0.005, 0.04, 0.018, 0.001, format="%.3f")
tail_width_limit = st.sidebar.slider("收斂尾端寬度上限", 0.06, 0.25, 0.15, 0.01)
avg_lot_limit = st.sidebar.number_input("小資本股20日均量上限（張）", min_value=500, max_value=100000, value=10000, step=500)
min_score = st.sidebar.slider("最低顯示評分", 0, 100, 58, 1)
debug_show_all = st.sidebar.checkbox(
    "除錯模式：顯示未達三角條件股票",
    value=False,
    help="測試資料源或新部署時可開啟。開啟後會列出資料抓得到但未形成三角候選的股票，方便判斷是資料問題還是條件問題。正式選股請關閉。",
)
max_scan = st.sidebar.slider("本次最多掃描檔數", 10, 800, 100, 10)

st.sidebar.header("籌碼資料（選填）")
chip_file = st.sidebar.file_uploader("上傳買賣家數差 CSV", type=["csv"])
xq_reverse = st.sidebar.checkbox("我的買賣家數差來自 XQ 電腦版（正數代表集中）", value=False)
use_chip = st.sidebar.checkbox("將買賣家數差納入評分", value=chip_file is not None)
chip_df = parse_chip_csv(chip_file)
if chip_file is not None and chip_df.empty:
    st.sidebar.warning("CSV欄位需含 code/股票代號 與 diff/買賣家數差，可另含 date/日期。")

st.sidebar.header("V8 操作風格")
style_mode = st.sidebar.selectbox(
    "排序偏好",
    ["平衡模式", "左側模式", "右側模式", "保守模式", "短波段模式", "中波段模式"],
    index=0,
    help="只調整排序與警示，不會把講義核心進出場規則改掉。",
)
st.sidebar.caption("左側偏下線不破/量縮/60MA/KD；右側偏突破上線/回測上線。")

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

scanner_tab, market_tab, daily_tab, plan_tab, paper_tab, report_tab, watch_tab, holdings_tab, capital_tab, history_tab, backtest_tab, schedule_tab, health_tab, help_tab = st.tabs(["🚀 選股掃描", "🌡️ 大盤/族群", "📋 每日清單", "📝 交易計畫", "🧾 紙上交易", "📑 策略報告", "🔔 自選提醒", "📊 持股追蹤", "💰 資金控管", "📚 歷史/訊號追蹤", "🧪 回測", "⏰ 排程/穩定性", "🛡️ 資料健檢", "📘 說明"])

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
        results = scan_rows(tickers_df, period, lookback, tol, tail_width_limit, chip_df, xq_reverse, use_chip, avg_lot_limit, min_score, max_scan, market_bias=market_bias, market_label=market_label, style_mode=style_mode, debug_show_all=debug_show_all)
        st.session_state["last_results"] = results
        if results:
            scan_id = save_scan_results(results, note=f"{mode}｜{market_label}｜{style_mode}")
            st.success(f"已儲存本次掃描歷史：{scan_id}")

    results = st.session_state.get("last_results", [])
    if results:
        table_cols = ["代號", "名稱", "產業", "日期", "資料狀態", "最新資料日", "收盤", "評分", "等級", "動作", "操作模式", "型態", "型態品質", "型態品質分", "風報比", "風報判斷", "偏好型態", "左側分", "右側分", "成熟度", "上線", "下線", "距上線%", "距下線%", "觸線", "20日均量(張)", "量比", "標籤", "警示"]
        display_df = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in results])
        st.dataframe(display_df[table_cols], use_container_width=True, hide_index=True)
        csv = display_df[table_cols + ["理由"]].to_csv(index=False).encode("utf-8-sig")
        st.download_button("下載結果 CSV", csv, file_name=f"ace_triangle_scan_v7_{datetime.now():%Y%m%d_%H%M}.csv", mime="text/csv")

        st.subheader("個股圖表與判讀")
        options = [f"{r['代號']} {r['名稱']}｜{r['動作']}｜{r['等級']}｜評分 {r['評分']}" for r in results]
        selected = st.selectbox("選擇個股", options)
        idx = options.index(selected)
        r = results[idx]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("評分", r["評分"])
        c2.metric("等級", r["等級"])
        c3.metric("風報比", r.get("風報比", "-"))
        c4.metric("型態品質", f"{r.get('型態品質', '-')}/{r.get('型態品質分', '-')}")
        q1, q2, q3, q4 = st.columns(4)
        q1.metric("距上線", f"{r['距上線%']}%")
        q2.metric("距下線", f"{r['距下線%']}%")
        q3.metric("左側分", r.get("左側分", "-"))
        q4.metric("右側分", r.get("右側分", "-"))
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


with plan_tab:
    st.subheader("交易計畫產生器")
    results = st.session_state.get("last_results", [])
    if not results:
        st.info("請先在『選股掃描』完成一次掃描，再到這裡替候選股產生操作計畫。")
    else:
        options = [f"{r['代號']} {r['名稱']}｜{r['動作']}｜{r['等級']}｜評分 {r['評分']}" for r in results]
        chosen = st.selectbox("選擇要規劃的股票", options, key="plan_stock_select")
        r = results[options.index(chosen)]
        pc1, pc2, pc3, pc4 = st.columns(4)
        plan_capital = pc1.number_input("波段資金", min_value=1000, max_value=100000000, value=300000, step=10000, key="plan_capital")
        plan_unit_pct = pc2.number_input("每單位%", min_value=1.0, max_value=30.0, value=10.0, step=1.0, key="plan_unit_pct")
        plan_max_pct = pc3.number_input("單股上限%", min_value=5.0, max_value=80.0, value=40.0, step=5.0, key="plan_max_pct")
        plan_risk_pct = pc4.number_input("單筆風險%", min_value=0.2, max_value=10.0, value=1.0, step=0.2, key="plan_risk_pct")
        plan_df = generate_trade_plan(r, plan_capital, plan_unit_pct, plan_max_pct, plan_risk_pct)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("基本單金額", f"{plan_df.attrs['unit_cash']:,.0f} 元")
        m2.metric("單股最高投入", f"{plan_df.attrs['max_cash']:,.0f} 元")
        m3.metric("基本單股數", f"{plan_df.attrs['unit_shares']:,}" if pd.notna(plan_df.attrs['unit_shares']) else "-")
        m4.metric("風險股數上限", f"{plan_df.attrs['risk_shares']:,}" if pd.notna(plan_df.attrs['risk_shares']) else "-")
        st.dataframe(plan_df, use_container_width=True, hide_index=True)
        st.markdown(f"<div class='signal-card'><b>目前判讀：</b>{r.get('動作')}｜{r.get('等級')}｜評分 {r.get('評分')}<br><b>標籤：</b>{r.get('標籤')}<br><b>警示：</b>{r.get('警示')}</div>", unsafe_allow_html=True)
        st.plotly_chart(plot_stock(r), use_container_width=True)
        st.download_button("下載交易計畫 CSV", plan_df.to_csv(index=False).encode("utf-8-sig"), file_name=f"ace_trade_plan_{r['代號']}_{datetime.now():%Y%m%d_%H%M}.csv", mime="text/csv")


with paper_tab:
    st.subheader("V7 紙上交易模式")
    st.write("先把候選股以模擬單追蹤 1–3 個月，驗證訊號品質與自己的執行紀律，再決定是否投入真實資金。")
    results = st.session_state.get("last_results", [])
    if results:
        options = [f"{r['代號']} {r['名稱']}｜{r['動作']}｜{r['等級']}｜評分 {r['評分']}" for r in results]
        chosen = st.selectbox("從最近掃描結果建立紙上交易", options, key="paper_stock_select")
        r = results[options.index(chosen)]
        pc1, pc2, pc3, pc4 = st.columns(4)
        p_entry = pc1.number_input("模擬進場價", min_value=0.01, value=float(r.get("收盤", 1.0)), step=0.1, key="paper_entry")
        p_shares = pc2.number_input("模擬股數", min_value=1, max_value=10000000, value=1000, step=100, key="paper_shares")
        default_stop = safe_float(r.get("下線", np.nan), np.nan)
        p_stop = pc3.number_input("停損參考價", min_value=0.0, value=float(default_stop) if np.isfinite(default_stop) else 0.0, step=0.1, key="paper_stop")
        p_units = pc4.number_input("模擬單位數", min_value=1, max_value=8, value=1, step=1, key="paper_units")
        p_note = st.text_input("進場理由/觀察重點", value=f"{r.get('動作')}｜{r.get('標籤','')[:60]}", key="paper_note")
        if st.button("➕ 建立紙上交易", use_container_width=True):
            tid = f"P{datetime.now():%Y%m%d%H%M%S}_{r['代號']}"
            append_paper_trade({
                "trade_id": tid,
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "status": "open",
                "ticker": r.get("Ticker"),
                "code": r.get("代號"),
                "name": r.get("名稱"),
                "entry_date": r.get("日期"),
                "entry_price": p_entry,
                "shares": int(p_shares),
                "units": int(p_units),
                "stop_price": p_stop,
                "entry_signal": r.get("動作"),
                "entry_score": r.get("評分"),
                "entry_grade": r.get("等級"),
                "entry_rr": r.get("風報比"),
                "style_mode": style_mode,
                "note": p_note,
            })
            st.success(f"已建立紙上交易：{tid}")
    else:
        st.info("請先到『選股掃描』完成一次掃描，再從候選股建立紙上交易。")

    trades = load_paper_trades()
    if trades.empty:
        st.info("目前尚無紙上交易紀錄。")
    else:
        st.subheader("紙上交易追蹤")
        open_only = st.checkbox("只看未平倉", value=True, key="paper_open_only")
        show = trades.copy()
        if open_only and "status" in show.columns:
            show = show[show["status"].astype(str) == "open"].copy()
        if st.button("🔄 更新紙上交易現值", use_container_width=True):
            show = mark_to_market_paper_trades(show, period, lookback, tol, tail_width_limit, avg_lot_limit, market_bias, market_label, style_mode)
            st.session_state["paper_mtm"] = show
        mtm = st.session_state.get("paper_mtm", show)
        st.dataframe(mtm, use_container_width=True, hide_index=True)
        st.download_button("下載紙上交易 CSV", trades.to_csv(index=False).encode("utf-8-sig"), file_name=f"ace_paper_trades_{datetime.now():%Y%m%d_%H%M}.csv", mime="text/csv")
        open_df = trades[trades.get("status", pd.Series(dtype=str)).astype(str) == "open"].copy() if "status" in trades else pd.DataFrame()
        if not open_df.empty:
            st.subheader("手動結案")
            close_opt = st.selectbox("選擇要結案的紙上交易", open_df["trade_id"].astype(str).tolist(), key="paper_close_select")
            close_price = st.number_input("結案價", min_value=0.0, value=0.0, step=0.1, key="paper_close_price")
            close_note = st.text_input("結案備註", value="依規則出場/手動結案", key="paper_close_note")
            if st.button("✅ 結案紙上交易", use_container_width=True):
                if close_price <= 0:
                    st.warning("請輸入結案價。")
                elif close_paper_trade(close_opt, close_price, close_note):
                    st.success("已結案。")
                else:
                    st.error("找不到該筆紙上交易。")


with report_tab:
    st.subheader("V8 策略報告與一鍵推播")
    st.write("把最近一次掃描結果整理成盤後報告，方便複盤、存檔或直接推送到 Telegram / Webhook。")
    results = st.session_state.get("last_results", [])
    ctx = market_context if market_context is not None else (get_market_context() if use_market_filter else None)
    report_md = generate_strategy_report(results, ctx)
    daily_msg = make_daily_message(results, max_items=16)
    st.text_area("Markdown 報告", value=report_md, height=420)
    if st.button("💾 儲存報告與推播文字到 data/reports", use_container_width=True):
        paths = save_daily_report_files(report_md, daily_msg)
        st.success(f"已儲存：{Path(paths['markdown']).name} / {Path(paths['text']).name}")
    st.download_button("下載策略報告 Markdown", report_md.encode("utf-8-sig"), file_name=f"ace_strategy_report_v8_{datetime.now():%Y%m%d_%H%M}.md", mime="text/markdown")
    st.download_button("下載推播文字 TXT", daily_msg.encode("utf-8-sig"), file_name=f"ace_daily_message_v8_{datetime.now():%Y%m%d_%H%M}.txt", mime="text/plain")

    with st.expander("V8 一鍵推播設定", expanded=False):
        st.caption("Token 可留空，程式會優先讀取環境變數 TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID / ACE_WEBHOOK_URL。")
        push_text = st.text_area("即將推送的文字", value=daily_msg, height=220, key="report_push_text")
        tg1, tg2 = st.columns(2)
        tg_token_in = tg1.text_input("Telegram Bot Token", value="", type="password", key="report_tg_token")
        tg_chat_in = tg2.text_input("Telegram Chat ID", value="", type="password", key="report_tg_chat")
        webhook_in = st.text_input("Webhook URL（選填，可接 Make/Zapier/Discord/LINE橋接）", value="", type="password", key="report_webhook")
        b1, b2 = st.columns(2)
        if b1.button("📨 推送到 Telegram", use_container_width=True, key="report_send_tg"):
            ok, msg = send_text_to_telegram(push_text, tg_token_in or os.getenv("TELEGRAM_BOT_TOKEN", ""), tg_chat_in or os.getenv("TELEGRAM_CHAT_ID", ""))
            st.success(msg) if ok else st.error(msg)
        if b2.button("🔗 推送到 Webhook", use_container_width=True, key="report_send_webhook"):
            ok, msg = send_text_to_webhook(push_text, webhook_in or os.getenv("ACE_WEBHOOK_URL", ""))
            st.success(msg) if ok else st.error(msg)


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
                res = evaluate_stock(row["ticker"], row["name"], df, lookback, tol, tail_width_limit, chip_df, xq_reverse, use_chip, avg_lot_limit, include_early=True, style_mode=style_mode)
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
                    style_mode=style_mode,
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


with backtest_tab:
    st.divider()
    st.subheader("V7 完整分批策略回測")
    st.write("這個回測會近似講義的基本單、突破加碼、回測加碼與破線/獲利出場流程，並納入手續費、證交稅與滑價。")
    with st.expander("設定完整策略回測", expanded=False):
        sbt_suffix = st.selectbox("完整回測未輸入市場時預設", [".TW 上市", ".TWO 上櫃"], index=0, key="sbt_suffix")
        sbt_codes = st.text_area("完整策略回測股票代號", value="2330, 2317, 2454", height=90, key="sbt_codes")
        sbt_period = st.selectbox("完整策略回測資料期間", ["2y", "5y"], index=1, key="sbt_period")
        sbt_c1, sbt_c2, sbt_c3, sbt_c4 = st.columns(4)
        sbt_capital = sbt_c1.number_input("回測本金", min_value=10000, max_value=100000000, value=1000000, step=50000, key="sbt_capital")
        sbt_unit_pct = sbt_c2.number_input("每單位%", min_value=1.0, max_value=30.0, value=10.0, step=1.0, key="sbt_unit_pct")
        sbt_max_units = sbt_c3.number_input("最多單位", min_value=1, max_value=8, value=4, step=1, key="sbt_max_units")
        sbt_min_score = sbt_c4.slider("完整回測最低分數", 50, 95, 68, 1, key="sbt_min_score")
        fee_c1, fee_c2, fee_c3 = st.columns(3)
        sbt_fee = fee_c1.number_input("手續費率", min_value=0.0, max_value=0.01, value=0.001425, step=0.0001, format="%.6f", key="sbt_fee")
        sbt_tax = fee_c2.number_input("證交稅率", min_value=0.0, max_value=0.01, value=0.003, step=0.0001, format="%.6f", key="sbt_tax")
        sbt_slip = fee_c3.number_input("滑價", min_value=0.0, max_value=0.02, value=0.001, step=0.0005, format="%.4f", key="sbt_slip")
        if st.button("🧪 執行完整分批策略回測", use_container_width=True):
            btdf = make_tickers_from_text(sbt_codes, sbt_suffix.split()[0]).head(25)
            all_trades = []
            progress = st.progress(0)
            for i, row in btdf.iterrows():
                st.write(f"完整策略回測 {row['ticker']} ...")
                df = fetch_ohlcv(row["ticker"], period=sbt_period)
                trades = strategy_backtest_single(
                    row["ticker"], row["name"], df, lookback, tol, tail_width_limit,
                    sbt_min_score, capital=sbt_capital, unit_pct=sbt_unit_pct,
                    max_units=sbt_max_units, fee_rate=sbt_fee, tax_rate=sbt_tax, slippage=sbt_slip
                )
                if not trades.empty:
                    all_trades.append(trades)
                progress.progress((i + 1) / max(1, len(btdf)))
            if not all_trades:
                st.warning("沒有產生完整策略交易紀錄。可降低分數、放寬參數或換股票。")
            else:
                strat_trades = pd.concat(all_trades, ignore_index=True)
                st.session_state["last_strategy_backtest"] = strat_trades
    strat_trades = st.session_state.get("last_strategy_backtest", pd.DataFrame())
    if strat_trades is not None and not strat_trades.empty:
        st.subheader("完整策略回測摘要")
        st.dataframe(summarize_strategy_trades(strat_trades), use_container_width=True, hide_index=True)
        st.subheader("完整策略交易明細")
        st.dataframe(strat_trades.sort_values("出場日", ascending=False), use_container_width=True, hide_index=True)
        st.download_button("下載完整策略回測 CSV", strat_trades.to_csv(index=False).encode("utf-8-sig"), file_name=f"ace_triangle_strategy_backtest_{datetime.now():%Y%m%d_%H%M}.csv", mime="text/csv")


with history_tab:
    st.subheader("掃描歷史與訊號品質追蹤")
    st.write("V8 會把每次掃描的候選股存到 CSV 與 SQLite，並保留候選股，之後可以追蹤訊號後續 1/3/5/10/20/60 日表現。這比只看單次篩選更能驗證策略品質。")
    upload_hist = st.file_uploader("匯入先前下載的掃描歷史 CSV（選填）", type=["csv"], key="history_upload")
    hist = load_scan_history(upload_hist)
    if hist.empty:
        st.info("目前沒有掃描歷史。先到『選股掃描』跑一次，系統會自動儲存。")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("歷史筆數", f"{len(hist):,}")
        c2.metric("掃描批次", f"{hist.get('scan_id', pd.Series()).nunique() if 'scan_id' in hist else 0:,}")
        c3.metric("最近掃描", str(hist['scan_time'].max())[:19] if 'scan_time' in hist and hist['scan_time'].notna().any() else "-")
        show_n = st.slider("顯示最近幾筆歷史", 20, 500, 100, 20)
        st.dataframe(hist.head(show_n), use_container_width=True, hide_index=True)
        st.download_button("下載完整掃描歷史 CSV", hist.to_csv(index=False).encode("utf-8-sig"), file_name=f"ace_scan_history_{datetime.now():%Y%m%d_%H%M}.csv", mime="text/csv")
        if st.button("📈 追蹤歷史訊號後續表現", use_container_width=True):
            outcomes = track_signal_outcomes(hist, default_suffix=".TW", period="5y")
            if outcomes.empty:
                st.warning("目前無法計算後續表現，可能是資料源抓不到或歷史日期不足。")
            else:
                st.session_state["last_outcomes"] = outcomes
        outcomes = st.session_state.get("last_outcomes", pd.DataFrame())
        if outcomes is not None and not outcomes.empty:
            st.subheader("訊號後續表現")
            st.dataframe(outcomes, use_container_width=True, hide_index=True)
            qual = summarize_signal_quality(outcomes)
            if not qual.empty:
                st.subheader("訊號品質統計")
                st.dataframe(qual, use_container_width=True, hide_index=True)
            st.download_button("下載訊號追蹤 CSV", outcomes.to_csv(index=False).encode("utf-8-sig"), file_name=f"ace_signal_outcomes_{datetime.now():%Y%m%d_%H%M}.csv", mime="text/csv")

            st.subheader("V8 參數優化建議")
            st.write("根據已追蹤的歷史訊號，估算較適合的最低分數門檻、較有效的動作/型態/模式。低樣本時只當參考，不建議立刻大幅調參。")
            opt_min_samples = st.slider("優化最低樣本數", 3, 50, 8, 1, key="opt_min_samples")
            if st.button("🧭 產生參數優化建議", use_container_width=True, key="make_param_opt"):
                opt_table, opt_msg = optimize_parameters_from_outcomes(outcomes, min_samples=opt_min_samples)
                st.session_state["last_param_opt"] = opt_table
                st.session_state["last_param_profile"] = build_parameter_profile(opt_table, min_score, style_mode)
                st.session_state["last_param_opt_msg"] = opt_msg
            opt_table = st.session_state.get("last_param_opt", pd.DataFrame())
            opt_profile = st.session_state.get("last_param_profile", {})
            opt_msg = st.session_state.get("last_param_opt_msg", "")
            if opt_msg:
                st.info(opt_msg)
            if opt_table is not None and not opt_table.empty:
                st.dataframe(opt_table, use_container_width=True, hide_index=True)
                profile_json = json.dumps(opt_profile, ensure_ascii=False, indent=2)
                st.json(opt_profile)
                if st.button("💾 儲存參數建議到 data/settings.json", use_container_width=True, key="save_param_profile"):
                    settings = load_app_settings()
                    settings["v8_parameter_profile"] = opt_profile
                    save_app_settings(settings)
                    st.success("已儲存。下次可在資料健檢或 settings.json 查看。")
                st.download_button("下載參數優化建議 JSON", profile_json.encode("utf-8-sig"), file_name=f"ace_parameter_profile_{datetime.now():%Y%m%d_%H%M}.json", mime="application/json")

with schedule_tab:
    st.subheader("V8 每日排程、報告與推播工具")
    st.write("Streamlit 本身不會在你關掉網頁後自動掃描；V8 提供 GitHub Actions / VPS 排程範本，可在收盤後自動掃描、產生報告並推送 Telegram/Webhook。")
    results = st.session_state.get("last_results", [])
    msg = make_daily_message(results, max_items=16)
    st.text_area("可複製到 LINE / Telegram 的盤後文字", value=msg, height=220, key="schedule_msg")
    sc1, sc2 = st.columns(2)
    if sc1.button("💾 儲存今日推播文字", use_container_width=True, key="schedule_save_msg"):
        paths = save_daily_report_files(generate_strategy_report(results, market_context if use_market_filter else None), msg)
        st.success(f"已儲存：{Path(paths['text']).name}")
    with st.expander("手動推播測試", expanded=False):
        tg1, tg2 = st.columns(2)
        tg_token_in = tg1.text_input("Telegram Bot Token", value="", type="password", key="schedule_tg_token")
        tg_chat_in = tg2.text_input("Telegram Chat ID", value="", type="password", key="schedule_tg_chat")
        webhook_in = st.text_input("Webhook URL", value="", type="password", key="schedule_webhook")
        p1, p2 = st.columns(2)
        if p1.button("📨 測試 Telegram 推播", use_container_width=True, key="schedule_send_tg"):
            ok, sent_msg = send_text_to_telegram(msg, tg_token_in or os.getenv("TELEGRAM_BOT_TOKEN", ""), tg_chat_in or os.getenv("TELEGRAM_CHAT_ID", ""))
            st.success(sent_msg) if ok else st.error(sent_msg)
        if p2.button("🔗 測試 Webhook 推播", use_container_width=True, key="schedule_send_webhook"):
            ok, sent_msg = send_text_to_webhook(msg, webhook_in or os.getenv("ACE_WEBHOOK_URL", ""))
            st.success(sent_msg) if ok else st.error(sent_msg)
    st.download_button("下載 scheduled_scan.py 範本", scheduled_scan_script_text().encode("utf-8"), file_name="scheduled_scan.py", mime="text/x-python")
    st.download_button("下載 GitHub Actions workflow 範本", github_actions_workflow_text().encode("utf-8"), file_name="ace_daily_scan.yml", mime="text/yaml")
    st.markdown("""
    **部署方式簡述**
    1. 把 `scheduled_scan.py` 放到 GitHub 專案的 `scripts/` 資料夾。  
    2. 把 `ace_daily_scan.yml` 放到 `.github/workflows/` 資料夾。  
    3. 在 GitHub Actions 設定 `WATCHLIST`、`MIN_SCORE`、`PERIOD`、`OFFICIAL_FALLBACK_MONTHS`。  
    4. 若要推播，設定 `TELEGRAM_BOT_TOKEN`、`TELEGRAM_CHAT_ID`，或 `ACE_WEBHOOK_URL`。  
    5. 預設排程是台灣時間約 14:40，適合盤後掃描。
    """)
    st.subheader("錯誤紀錄")
    if ERROR_LOG_FILE.exists():
        try:
            err = pd.read_csv(ERROR_LOG_FILE, encoding="utf-8-sig").tail(200)
            st.dataframe(err, use_container_width=True, hide_index=True)
        except Exception:
            st.info("錯誤紀錄檔存在，但暫時無法讀取。")
    else:
        st.info("目前沒有錯誤紀錄。")


with health_tab:
    st.subheader("V8 資料健檢與系統狀態")
    st.write("這裡用來確認資料是不是新鮮、快取是否正常、SQLite 歷史庫是否可讀，以及掃描時是否有錯誤；V8 也可搭配 scripts/validate_project.py 與 tests 做工程檢查。")
    hc1, hc2, hc3, hc4 = st.columns(4)
    hc1.metric("SQLite資料庫", "存在" if DB_FILE.exists() else "尚未建立")
    hc2.metric("OHLCV快取檔", f"{len(list(OHLCV_CACHE_DIR.glob('*.csv'))):,}")
    hc3.metric("CSV歷史", "存在" if HISTORY_FILE.exists() else "尚無")
    hc4.metric("錯誤紀錄", "存在" if ERROR_LOG_FILE.exists() else "無")

    st.subheader("快取資料摘要")
    cs = cache_summary()
    if cs.empty:
        st.info("目前尚無日K快取；掃描一次後會自動建立。")
    else:
        st.dataframe(cs.head(200), use_container_width=True, hide_index=True)

    st.subheader("單檔資料健檢")
    h1, h2 = st.columns([2, 1])
    q_ticker_raw = h1.text_input("輸入要檢查的股票", value="2330.TW", key="quality_ticker")
    q_period = h2.selectbox("健檢期間", ["1y", "18mo", "2y", "5y"], index=1, key="quality_period")
    if st.button("檢查資料品質", use_container_width=True):
        q_ticker = normalize_code(q_ticker_raw, ".TW") if "." not in q_ticker_raw.upper() else q_ticker_raw.upper().strip()
        q_df = fetch_ohlcv(q_ticker, q_period)
        qr = ohlcv_quality_report(q_df, q_ticker)
        st.json(qr)
        if not q_df.empty:
            st.dataframe(q_df.tail(10).reset_index(), use_container_width=True, hide_index=True)

    st.subheader("SQLite掃描歷史摘要")
    dbh = load_scan_history_from_db()
    if dbh.empty:
        st.info("SQLite 尚無掃描歷史。到『選股掃描』跑一次後會寫入。")
    else:
        st.dataframe(dbh.head(100), use_container_width=True, hide_index=True)
        st.download_button("下載SQLite歷史CSV", dbh.to_csv(index=False).encode("utf-8-sig"), file_name=f"ace_sqlite_scan_history_{datetime.now():%Y%m%d_%H%M}.csv", mime="text/csv")

    st.subheader("錯誤紀錄")
    if ERROR_LOG_FILE.exists():
        try:
            err = pd.read_csv(ERROR_LOG_FILE, encoding="utf-8-sig").tail(300)
            st.dataframe(err, use_container_width=True, hide_index=True)
            st.download_button("下載錯誤紀錄", err.to_csv(index=False).encode("utf-8-sig"), file_name=f"ace_error_log_{datetime.now():%Y%m%d_%H%M}.csv", mime="text/csv")
        except Exception as exc:
            st.warning(f"錯誤紀錄暫時無法讀取：{exc}")
    else:
        st.info("目前沒有錯誤紀錄。")


with help_tab:
    st.subheader("使用說明與限制")
    st.markdown(
        """
        **建議工作流**
        1. 先用「全市場自動掃描」找 A/B 級候選。
        2. 點進圖表看三角上線、下線是否合理。
        3. 把有興趣的股票放進「自選提醒」。
        4. 用「資金控管」先決定基本單與最大投入。
        5. 用「交易計畫」先寫好進場、加碼、停損與出場條件。
        6. 用「歷史/訊號追蹤」追蹤這些訊號後續是否真的有效。
        7. 用「回測」檢查這檔股票過去是否適合這套邏輯。

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
