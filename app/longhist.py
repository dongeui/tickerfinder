"""장기(약 10년) 일봉 캐시. 백테스트(확률 계산) 전용.

스캔용 캐시(data/prices, 약 600일, 국내는 네이버 미수정주가)와 분리해서 data/prices10y 에 저장한다.
국내도 yfinance(005930.KS / 036930.KQ)로 받아 분할·감자가 수정주가로 반영된 시계열을 쓴다.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

from .scanner import KST, drop_partial_bar
from .universe import DATA_DIR

log = logging.getLogger(__name__)
LONG_DIR = DATA_DIR / "prices10y"
LONG_DIR.mkdir(exist_ok=True)
YEARS = int(os.environ.get("LONG_YEARS", "20"))
INDEX_SYMS = {"IXIC": "^IXIC", "KS11": "^KS11"}

STATE = {"running": False, "stage": "", "done": 0, "total": 0, "started_at": None, "finished_at": None, "error": None}
_lock = threading.Lock()


def _set(**kw):
    with _lock:
        STATE.update(kw)


def yf_symbol(u: dict) -> str:
    if u["region"] == "KR":
        return u["ticker"] + (".KQ" if u.get("market") == "KOSDAQ" else ".KS")
    return u["ticker"]


def _path(ticker: str) -> Path:
    return LONG_DIR / f"{ticker.replace('^', '_')}.parquet"


def load_long(ticker: str) -> pd.DataFrame | None:
    p = _path(ticker)
    if not p.exists():
        return None
    try:
        df = pd.read_parquet(p)
        return df if len(df) else None
    except Exception:  # noqa: BLE001
        return None


def signature() -> tuple:
    """캐시 변경 감지용 (파일 수, 최신 mtime)."""
    files = list(LONG_DIR.glob("*.parquet"))
    return (len(files), max((f.stat().st_mtime_ns for f in files), default=0))


def _fresh(ticker: str, max_age_days: int = 3) -> bool:
    df = load_long(ticker)
    if df is None or len(df) < 250:
        return False
    return (datetime.now(KST).date() - df.index[-1].date()).days <= max_age_days


def _save(ticker: str, region: str, new: pd.DataFrame | None) -> None:
    if new is None or not len(new):
        return
    new = new[[c for c in ("Open", "High", "Low", "Close", "Volume") if c in new.columns]].dropna(subset=["Close"]).copy()
    new.index = pd.to_datetime(new.index).tz_localize(None).normalize()
    new = new[~new.index.duplicated(keep="last")]
    old = load_long(ticker)
    if old is not None:
        # 수정주가가 바뀌었으면(분할 등) 새 데이터로 교체
        common = old.index.intersection(new.index)
        if len(common) and ((old.loc[common, "Close"] / new.loc[common, "Close"]) - 1).abs().max() > 0.01:
            df = new
        else:
            df = pd.concat([old, new])
            df = df[~df.index.duplicated(keep="last")]
    else:
        df = new
    df = drop_partial_bar(df.sort_index(), region)
    try:
        df.to_parquet(_path(ticker))
    except Exception as e:  # noqa: BLE001
        log.warning("long cache write failed %s: %s", ticker, e)


def _download(symbols: list[str], start: str) -> dict[str, pd.DataFrame]:
    got: dict[str, pd.DataFrame] = {}
    if not symbols:
        return got
    try:
        d = yf.download(symbols, start=start, group_by="ticker", progress=False, auto_adjust=True, threads=True)
    except Exception as e:  # noqa: BLE001
        log.warning("long download failed: %s", e)
        return got
    if d is None or not len(d):
        return got
    for s in symbols:
        try:
            df = d[s] if isinstance(d.columns, pd.MultiIndex) else d
            df = df.dropna(subset=["Close"])
            if len(df):
                got[s] = df
        except KeyError:
            pass
    return got


def fetch_all(uni: list[dict], years: int = YEARS, chunk: int = 150, force: bool = False) -> None:
    """유니버스 전체 + 지수의 장기 일봉을 받아 캐시. 신선한 캐시는 최근 40일만 증분."""
    with _lock:
        if STATE["running"]:
            raise RuntimeError("already running")
        STATE.update(running=True, stage="장기 데이터 수집", done=0, total=len(uni) + len(INDEX_SYMS), error=None,
                     started_at=datetime.now(KST).isoformat(timespec="seconds"), finished_at=None)
    try:
        full_start = (datetime.now(KST) - timedelta(days=365 * years)).strftime("%Y-%m-%d")
        incr_start = (datetime.now(KST) - timedelta(days=40)).strftime("%Y-%m-%d")
        items = [(u["ticker"], u["region"], yf_symbol(u)) for u in uni]
        items += [(k, "KR" if k == "KS11" else "US", v) for k, v in INDEX_SYMS.items()]
        full = [it for it in items if force or not _fresh(it[0])]
        incr = [it for it in items if it not in full]
        for group, start in ((full, full_start), (incr, incr_start)):
            for i in range(0, len(group), chunk):
                part = group[i:i + chunk]
                got = _download([s for _, _, s in part], start)
                for ticker, region, sym in part:
                    _save(ticker, region, got.get(sym))
                with _lock:
                    STATE["done"] += len(part)
                time.sleep(0.5)
        _set(stage="수집 완료")
    except Exception as e:  # noqa: BLE001
        log.exception("long fetch failed")
        _set(error=str(e), stage="실패")
        raise
    finally:
        _set(running=False, finished_at=datetime.now(KST).isoformat(timespec="seconds"))
