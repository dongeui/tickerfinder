"""종목 개요·재무 요약 (yfinance Ticker.info), data/profiles/<ticker>.json 에 7일 캐시."""
from __future__ import annotations

import json
import logging
import re
import time
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path

from .longhist import yf_symbol
from .scanner import KST
from .universe import DATA_DIR

log = logging.getLogger("tickerfinder")
PROF_DIR = DATA_DIR / "profiles"
TTL_SEC = 7 * 86400
FIELDS = ("longName", "sector", "industry", "country", "website", "fullTimeEmployees", "longBusinessSummary", "financialCurrency", "quoteType",
          "marketCap", "enterpriseValue", "trailingPE", "forwardPE", "priceToBook", "enterpriseToEbitda",
          "totalRevenue", "revenueGrowth", "grossMargins", "operatingMargins", "profitMargins", "netIncomeToCommon", "earningsGrowth", "ebitda",
          "freeCashflow", "operatingCashflow", "totalCash", "totalDebt", "debtToEquity", "currentRatio", "returnOnEquity", "returnOnAssets",
          "dividendYield", "payoutRatio", "beta", "recommendationKey", "recommendationMean", "targetMeanPrice", "numberOfAnalystOpinions",
          "fiftyTwoWeekHigh", "fiftyTwoWeekLow", "sharesOutstanding", "heldPercentInstitutions", "shortPercentOfFloat", "mostRecentQuarter", "lastFiscalYearEnd")


def _uni_entry(ticker: str) -> dict | None:
    try:
        for u in json.loads((DATA_DIR / "universe.json").read_text()):
            if u.get("ticker") == ticker:
                return u
    except (OSError, json.JSONDecodeError):
        pass
    from . import etf  # 국내상장 ETF (universe.json 에 없음)
    return etf.uni_entry(ticker)


def translate_ko(text: str, timeout: float = 12.0) -> str | None:
    """영문 사업 요약 → 한글 (구글 번역 비공식 gtx 엔드포인트, 키 불필요). 문장 단위로 1,500자씩 나눠 보낸다. 실패하면 None."""
    text = (text or "").strip()
    if not text:
        return None
    sents = re.split(r"(?<=[.!?])\s+", text)
    chunks, cur = [], ""
    for sn in sents:
        if len(cur) + len(sn) + 1 > 1500 and cur:
            chunks.append(cur)
            cur = sn
        else:
            cur = (cur + " " + sn).strip()
    if cur:
        chunks.append(cur)
    out = []
    for ch in chunks:
        q = urllib.parse.urlencode({"client": "gtx", "sl": "en", "tl": "ko", "dt": "t", "q": ch})
        req = urllib.request.Request("https://translate.googleapis.com/translate_a/single?" + q, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            j = json.loads(r.read())
        out.append("".join(seg[0] for seg in j[0] if seg and seg[0]))
    ko = " ".join(out).strip()
    return ko or None


def _save(ticker: str, data: dict) -> None:
    try:
        PROF_DIR.mkdir(parents=True, exist_ok=True)
        tmp = _path(ticker).with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False))
        tmp.replace(_path(ticker))
    except OSError as e:
        log.warning("profile cache write failed %s: %s", ticker, e)


def _with_ko(ticker: str, data: dict | None) -> dict | None:
    """사업 요약 한글 번역이 없으면 번역해서 캐시에 덧붙인다 (yfinance 재조회 없이). 번역 실패는 조용히 넘긴다."""
    if not data or not data.get("longBusinessSummary") or data.get("summary_ko") or data.get("summary_ko_failed_at"):
        return data
    try:
        ko = translate_ko(data["longBusinessSummary"])
    except Exception as e:  # noqa: BLE001
        log.warning("summary translate failed %s: %s", ticker, e)
        ko = None
    cached = _load(ticker) or data
    if ko:
        cached["summary_ko"] = ko
        cached.pop("summary_ko_failed_at", None)
    else:
        cached["summary_ko_failed_at"] = datetime.now(KST).isoformat(timespec="seconds")  # 다음 요청 때 다시 시도하지 않도록 (재조회 시 초기화)
    _save(ticker, cached)
    return {**data, **{k: cached[k] for k in ("summary_ko", "summary_ko_failed_at") if k in cached}}


def _path(ticker: str) -> Path:
    return PROF_DIR / f"{ticker.replace('^', '_')}.json"


def _load(ticker: str) -> dict | None:
    p = _path(ticker)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _fetch(ticker: str, symbol: str) -> dict:
    import yfinance as yf  # 지연 import (서버 기동 시간)

    info = yf.Ticker(symbol).info or {}
    out = {k: info.get(k) for k in FIELDS}
    for k in ("mostRecentQuarter", "lastFiscalYearEnd"):
        v = out.get(k)
        if isinstance(v, (int, float)) and v > 0:
            out[k] = datetime.fromtimestamp(v, KST).strftime("%Y-%m-%d")
    out.update(ticker=ticker, symbol=symbol, fetched_at=datetime.now(KST).isoformat(timespec="seconds"), ok=bool(info.get("longName") or info.get("shortName")))
    return out


def get(ticker: str, force: bool = False) -> dict | None:
    """캐시(7일)된 개요/재무. 없거나 오래됐으면 yfinance 로 받는다. 실패 시 오래된 캐시라도 반환(stale=True)."""
    cached = _load(ticker)
    if cached and not force:
        try:
            age = time.time() - datetime.fromisoformat(cached["fetched_at"]).timestamp()
        except (KeyError, ValueError):
            age = TTL_SEC + 1
        if age < TTL_SEC and cached.get("ok"):
            return _with_ko(ticker, cached)
    u = _uni_entry(ticker)
    symbol = yf_symbol(u) if u else ticker
    try:
        fresh = _fetch(ticker, symbol)
    except Exception as e:  # noqa: BLE001 - 네트워크/야후 오류는 캐시로 대체
        log.warning("profile fetch failed %s (%s): %s", ticker, symbol, e)
        if cached:
            return {**cached, "stale": True}
        return None
    if not fresh["ok"]:
        return {**cached, "stale": True} if cached else None
    if cached and cached.get("summary_ko") and cached.get("longBusinessSummary") == fresh.get("longBusinessSummary"):
        fresh["summary_ko"] = cached["summary_ko"]  # 원문이 같으면 번역 재사용
    _save(ticker, fresh)
    return _with_ko(ticker, fresh)
