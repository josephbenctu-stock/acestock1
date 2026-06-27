# -*- coding: utf-8 -*-
"""ACE Triangle Daily Scanner V8.

Lightweight scheduled scanner for GitHub Actions or a VPS.
It keeps local CSV cache, writes scan history, records errors, and can push Telegram.
For the full visual/manual workflow, use app.py in Streamlit.
"""
from __future__ import annotations

import os
import re
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import yfinance as yf
import requests

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
CACHE_DIR = DATA_DIR / "ohlcv_cache"
DATA_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)
HISTORY_FILE = DATA_DIR / "scheduled_scan_history.csv"
ERROR_FILE = DATA_DIR / "scheduled_error_log.csv"
REPORT_DIR = DATA_DIR / "reports"
REPORT_DIR.mkdir(exist_ok=True)

WATCHLIST = [x.strip() for x in os.getenv("WATCHLIST", "2330.TW,2317.TW,2454.TW").split(",") if x.strip()]
MIN_SCORE = int(os.getenv("MIN_SCORE", "60"))
PERIOD = os.getenv("PERIOD", "18mo")
OFFICIAL_FALLBACK_MONTHS = int(os.getenv("OFFICIAL_FALLBACK_MONTHS", "4"))


def normalize_code(raw: str, default_suffix: str = ".TW") -> str:
    s = str(raw).strip().upper().replace(" ", "")
    if s.endswith(".TW") or s.endswith(".TWO"):
        return s
    m = re.search(r"(\d{4,6})", s)
    return m.group(1) + default_suffix if m else s


def log_error(context: str, ticker: str = "", detail: str = "") -> None:
    row = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "context": context,
        "ticker": ticker,
        "detail": detail[:500],
        "trace": traceback.format_exc(limit=2)[:800],
    }
    out = pd.DataFrame([row])
    if ERROR_FILE.exists():
        try:
            out = pd.concat([pd.read_csv(ERROR_FILE, encoding="utf-8-sig"), out], ignore_index=True).tail(2000)
        except Exception:
            pass
    out.to_csv(ERROR_FILE, index=False, encoding="utf-8-sig")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        if "Open" in df.columns.get_level_values(0):
            df = df.droplevel(1, axis=1)
        else:
            df.columns = df.columns.get_level_values(-1)
    cols = {c: str(c).title().replace(" ", "") for c in df.columns}
    df = df.rename(columns=cols)
    need = ["Open", "High", "Low", "Close", "Volume"]
    if not all(c in df.columns for c in need):
        return pd.DataFrame()
    df = df[need].dropna()
    if isinstance(df.index, pd.DatetimeIndex):
        df.index = df.index.tz_localize(None) if df.index.tz is not None else df.index
    return df


def _to_float_tw(value) -> float:
    try:
        s = str(value).strip().replace(",", "")
        s = re.sub(r"[^0-9.\-]", "", s)
        return float(s) if s not in ["", ".", "-"] else np.nan
    except Exception:
        return np.nan


def _parse_tw_date(value) -> pd.Timestamp:
    parts = re.split(r"[/\-.]", str(value).strip())
    if len(parts) >= 3 and all(x.isdigit() for x in parts[:3]):
        y, m, d = [int(x) for x in parts[:3]]
        if y < 1911:
            y += 1911
        return pd.Timestamp(year=y, month=m, day=d)
    return pd.to_datetime(value, errors="coerce")


def _month_starts(months_back: int) -> List[pd.Timestamp]:
    today = pd.Timestamp.today().normalize().replace(day=1)
    return [today - pd.DateOffset(months=i) for i in range(max(1, int(months_back)))]


def _payload_to_ohlcv(payload: dict) -> pd.DataFrame:
    data = payload.get("data") or payload.get("aaData") or [] if isinstance(payload, dict) else []
    fields = payload.get("fields") or ["日期", "成交股數", "成交金額", "開盤價", "最高價", "最低價", "收盤價"] if isinstance(payload, dict) else []
    rows = []
    if not isinstance(data, list):
        return pd.DataFrame()
    for raw in data:
        if not isinstance(raw, (list, tuple)) or len(raw) < 6:
            continue
        mapping = {str(fields[i]): raw[i] for i in range(min(len(fields), len(raw)))}
        def find(kw, idx):
            for k, v in mapping.items():
                if kw in str(k):
                    return v
            return raw[idx] if idx < len(raw) else np.nan
        dt = _parse_tw_date(raw[0])
        if pd.isna(dt):
            continue
        vals = [_to_float_tw(find("開盤", 3)), _to_float_tw(find("最高", 4)), _to_float_tw(find("最低", 5)), _to_float_tw(find("收盤", 6)), _to_float_tw(find("成交股", 1))]
        if not all(np.isfinite(x) for x in vals[:4]):
            continue
        rows.append({"Date": dt, "Open": vals[0], "High": vals[1], "Low": vals[2], "Close": vals[3], "Volume": vals[4] if np.isfinite(vals[4]) else 0})
    return clean(pd.DataFrame(rows).set_index("Date")) if rows else pd.DataFrame()


def fetch_official(ticker: str, months_back: int = 4) -> pd.DataFrame:
    ticker = normalize_code(ticker)
    code = re.sub(r"\D", "", ticker)
    if not code:
        return pd.DataFrame()
    headers = {"User-Agent": "Mozilla/5.0"}
    frames = []
    for month in _month_starts(months_back):
        if ticker.endswith(".TW"):
            date_s = month.strftime("%Y%m01")
            urls = [
                f"https://www.twse.com.tw/exchangeReport/STOCK_DAY?response=json&date={date_s}&stockNo={code}",
                f"https://www.twse.com.tw/rwd/zh/afterTrading/STOCK_DAY?response=json&date={date_s}&stockNo={code}",
            ]
        else:
            roc = f"{month.year - 1911}/{month.month:02d}"
            ad = month.strftime("%Y/%m/01")
            urls = [
                f"https://www.tpex.org.tw/www/zh-tw/afterTrading/tradingStock?code={code}&date={ad}&response=json",
                f"https://www.tpex.org.tw/web/stock/aftertrading/daily_trading_info/st43_result.php?l=zh-tw&d={roc}&stkno={code}",
            ]
        for url in urls:
            try:
                r = requests.get(url, headers=headers, timeout=12)
                if r.status_code == 200:
                    df = _payload_to_ohlcv(r.json())
                    if not df.empty:
                        frames.append(df)
                        break
            except Exception as exc:
                log_error("official_fetch", ticker, str(exc))
        time.sleep(0.03)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames).sort_index()
    return clean(out[~out.index.duplicated(keep="last")])


def merge_ohlcv(primary: pd.DataFrame, supplement: pd.DataFrame) -> pd.DataFrame:
    if primary is None or primary.empty:
        return clean(supplement)
    if supplement is None or supplement.empty:
        return clean(primary)
    out = pd.concat([primary, supplement]).sort_index()
    out = out[~out.index.duplicated(keep="last")]
    return clean(out)


def cache_path(ticker: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", f"{ticker}_{PERIOD}")
    return CACHE_DIR / f"{safe}.csv"


def fetch(ticker: str) -> pd.DataFrame:
    ticker = normalize_code(ticker)
    df = pd.DataFrame()
    try:
        df = clean(yf.download(ticker, period=PERIOD, interval="1d", progress=False, auto_adjust=False, threads=False))
    except Exception as exc:
        log_error("yfinance_download", ticker, str(exc))
    try:
        needs_patch = df.empty
        if not df.empty and isinstance(df.index, pd.DatetimeIndex):
            age = (pd.Timestamp.now().normalize() - pd.Timestamp(df.index[-1]).normalize()).days
            needs_patch = age >= 2
        if needs_patch and ticker.endswith((".TW", ".TWO")):
            official = fetch_official(ticker, months_back=OFFICIAL_FALLBACK_MONTHS if not df.empty else max(OFFICIAL_FALLBACK_MONTHS, 18))
            if not official.empty:
                df = merge_ohlcv(df, official)
                log_error("official_patch", ticker, f"official rows={len(official)}")
    except Exception as exc:
        log_error("official_patch_failed", ticker, str(exc))
    if not df.empty:
        out = df.copy(); out.index.name = "Date"
        out.to_csv(cache_path(ticker), encoding="utf-8-sig")
        return df
    try:
        p = cache_path(ticker)
        if p.exists():
            return clean(pd.read_csv(p, encoding="utf-8-sig", parse_dates=["Date"]).set_index("Date"))
    except Exception as exc:
        log_error("cache_load", ticker, str(exc))
    return pd.DataFrame()


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA60"] = df["Close"].rolling(60).mean()
    df["VOL20"] = df["Volume"].rolling(20).mean()
    mid = df["Close"].rolling(20).mean(); std = df["Close"].rolling(20).std()
    df["BB_UPPER"] = mid + 2 * std
    df["BB_WIDTH"] = (4 * std) / mid
    return df


def quality(df: pd.DataFrame) -> tuple[str, str]:
    if df.empty:
        return "無資料", "抓不到資料"
    warnings = []
    latest = df.index[-1].strftime("%Y-%m-%d") if hasattr(df.index[-1], "strftime") else str(df.index[-1])
    if len(df) < 90:
        warnings.append("資料不足90日")
    if df.tail(90).isna().any().any():
        warnings.append("近90日缺值")
    return ("正常" if not warnings else "需檢查"), latest + ("｜" + "；".join(warnings) if warnings else "")


def simple_score(df: pd.DataFrame) -> dict | None:
    df = add_indicators(df).dropna()
    if len(df) < 80:
        return None
    w = df.tail(45).copy()
    price = float(w["Close"].iloc[-1])
    if price <= 0:
        return None
    hi = pd.concat([w["Open"], w["Close"]], axis=1).max(axis=1)
    lo = pd.concat([w["Open"], w["Close"]], axis=1).min(axis=1)
    x = np.arange(len(w))
    up = np.polyfit(x, hi, 1); dn = np.polyfit(x, lo, 1)
    upper = float(up[0] * x[-1] + up[1]); lower = float(dn[0] * x[-1] + dn[1])
    width = (upper - lower) / price if price else 9
    vol_ratio = float(w["Volume"].iloc[-1] / max(w["VOL20"].iloc[-1], 1))
    score = 0; tags: list[str] = []; warnings: list[str] = []
    if width < 0.18 and upper > lower:
        score += 35; tags.append("收斂")
    if price >= lower * 0.99 and price <= upper * 1.03:
        score += 15; tags.append("線內/近突破")
    if abs(w["Close"].iloc[-1] - w["Open"].iloc[-1]) / price < 0.025 and vol_ratio < 0.8:
        score += 15; tags.append("小K量縮")
    if price > upper * 1.01:
        score += 20; tags.append("突破上線")
    if w["Close"].iloc[-1] > w["MA60"].iloc[-1]:
        score += 10; tags.append("站上60MA")
    if width > 0.32:
        score -= 10; warnings.append("不要交易：三角開口過大")
    if price < w["MA60"].iloc[-1] * 0.93:
        score -= 8; warnings.append("不要交易：季線下方偏弱")
    return {"收盤": round(price, 2), "評分": int(max(0, min(100, score))), "上線": round(upper, 2), "下線": round(lower, 2), "標籤": "、".join(tags), "警示": "；".join(warnings)}


def build_report(rows: pd.DataFrame, scan_id: str) -> tuple[str, str]:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    if rows is None or rows.empty:
        text = f"艾斯三角收斂每日掃描 V8｜{stamp}\n今日無符合候選"
        md = f"# 艾斯三角收斂每日掃描 V8\n\n- 時間：{stamp}\n- scan_id：{scan_id}\n- 結果：今日無符合候選\n"
        return text, md
    top = rows.sort_values("評分", ascending=False).head(20)
    lines = [f"艾斯三角收斂每日掃描 V8｜{stamp}", f"候選：{len(rows)} 檔｜門檻：{MIN_SCORE}"]
    for _, r in top.iterrows():
        lines.append(f"{r.get('Ticker')}｜評分{r.get('評分')}｜收{r.get('收盤')}｜{r.get('標籤')}｜{r.get('警示')}")
    md_lines = ["# 艾斯三角收斂每日掃描 V8", "", f"- 時間：{stamp}", f"- scan_id：{scan_id}", f"- 候選：{len(rows)} 檔", "", "## 候選清單", ""]
    md_lines.append(top.to_markdown(index=False))
    return "\n".join(lines), "\n".join(md_lines)


def send_telegram(text: str) -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN"); chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        return
    requests.post(f"https://api.telegram.org/bot{token}/sendMessage", data={"chat_id": chat_id, "text": text[:3900]}, timeout=20)


def send_webhook(text: str) -> None:
    url = os.getenv("ACE_WEBHOOK_URL")
    if not url:
        return
    requests.post(url, json={"text": text, "source": "ace_triangle_scheduled_scan", "time": datetime.now().isoformat()}, timeout=20)


def main() -> None:
    rows = []
    scan_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    for ticker in WATCHLIST:
        ticker = normalize_code(ticker)
        try:
            df = fetch(ticker)
            q_status, q_note = quality(df)
            res = simple_score(df)
            if res and res["評分"] >= MIN_SCORE:
                res.update({"scan_id": scan_id, "Ticker": ticker, "日期": datetime.now().strftime("%Y-%m-%d"), "資料狀態": q_status, "資料說明": q_note})
                rows.append(res)
        except Exception as exc:
            log_error("scan", ticker, str(exc))
        time.sleep(0.12)

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("評分", ascending=False)
        if HISTORY_FILE.exists():
            try:
                out = pd.concat([pd.read_csv(HISTORY_FILE, encoding="utf-8-sig"), out], ignore_index=True)
            except Exception:
                pass
        out.to_csv(HISTORY_FILE, index=False, encoding="utf-8-sig")
    latest = out.tail(len(rows)) if rows else pd.DataFrame()
    latest.to_csv("ace_daily_scan.csv", index=False, encoding="utf-8-sig")
    msg, report_md = build_report(latest, scan_id)
    (REPORT_DIR / f"ace_daily_report_{scan_id}.md").write_text(report_md, encoding="utf-8")
    (REPORT_DIR / f"ace_daily_message_{scan_id}.txt").write_text(msg, encoding="utf-8")
    try:
        send_telegram(msg)
        send_webhook(msg)
    except Exception as exc:
        log_error("push", detail=str(exc))
    print(msg)


if __name__ == "__main__":
    main()
