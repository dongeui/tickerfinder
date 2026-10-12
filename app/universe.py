"""스캔 대상 종목 유니버스.

- 국내: KOSPI 전 종목(보통주) + KOSDAQ 시가총액 상위 N개 (기본 100)
- 해외: S&P500 + Nasdaq-100
"""
from __future__ import annotations

import io
import json
import logging
import re
from pathlib import Path

import FinanceDataReader as fdr
import pandas as pd
import requests

log = logging.getLogger(__name__)
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}

EXCLUDE_NAME = re.compile(r"스팩|SPAC|\d+호$|리츠|ETN|인프라|우$|우B$|우C$|\(전환\)")


def _kr_common_stock(df: pd.DataFrame) -> pd.DataFrame:
    """보통주만 (코드 끝자리 0), 스팩/우선주/리츠 등 제외."""
    df = df[df["Code"].str.fullmatch(r"\d{6}")]
    df = df[df["Code"].str.endswith("0")]
    df = df[~df["Name"].str.contains(EXCLUDE_NAME, regex=True, na=False)]
    return df


def kr_universe(kosdaq_top: int = 100, kospi_min_marcap: float = 0) -> list[dict]:
    out = []
    kospi = _kr_common_stock(fdr.StockListing("KOSPI"))
    if kospi_min_marcap:
        kospi = kospi[kospi["Marcap"] >= kospi_min_marcap]
    kosdaq = _kr_common_stock(fdr.StockListing("KOSDAQ")).sort_values("Marcap", ascending=False).head(kosdaq_top)
    for df, mkt in ((kospi, "KOSPI"), (kosdaq, "KOSDAQ")):
        for r in df.itertuples():
            out.append({
                "ticker": r.Code,
                "name": r.Name,
                "market": mkt,
                "region": "KR",
                "marcap": float(r.Marcap) if pd.notna(r.Marcap) else None,
            })
    return out


def _wiki_table(url: str, min_rows: int = 90) -> pd.DataFrame | None:
    try:
        html = requests.get(url, headers=UA, timeout=20).text
        for t in pd.read_html(io.StringIO(html)):
            cols = {str(c).lower(): c for c in t.columns}
            key = next((cols[k] for k in cols if "ticker" in k or "symbol" in k), None)
            if key is not None and len(t) >= min_rows:
                name_col = next((cols[k] for k in cols if "company" in k or "security" in k), None)
                sector_col = next((cols[k] for k in cols if "sector" in k), None)
                return pd.DataFrame({
                    "ticker": t[key].astype(str),
                    "name": t[name_col].astype(str) if name_col is not None else t[key].astype(str),
                    "sector": t[sector_col].astype(str) if sector_col is not None else "",
                })
    except Exception as e:  # noqa: BLE001
        log.warning("wiki fetch failed %s: %s", url, e)
    return None


def _sp500() -> pd.DataFrame:
    t = _wiki_table("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", 400)
    if t is None:
        f = fdr.StockListing("S&P500")
        t = pd.DataFrame({"ticker": f["Symbol"], "name": f["Name"], "sector": f.get("Sector", "")})
    return t


def _nasdaq100() -> pd.DataFrame | None:
    # 1) 나스닥 공식 API
    try:
        j = requests.get("https://api.nasdaq.com/api/quote/list-type/nasdaq100",
                         headers={**UA, "Accept": "application/json"}, timeout=20).json()
        rows = j["data"]["data"]["rows"]
        if len(rows) >= 90:
            return pd.DataFrame({"ticker": [r["symbol"] for r in rows],
                                 "name": [re.sub(r"\s+(Common Stock|Class [A-C] .*|Ordinary Shares.*)$", "", r["companyName"]) for r in rows],
                                 "sector": [r.get("sector", "") for r in rows]})
    except Exception as e:  # noqa: BLE001
        log.warning("nasdaq api failed: %s", e)
    # 2) stockanalysis.com
    try:
        html = requests.get("https://stockanalysis.com/list/nasdaq-100-stocks/", headers=UA, timeout=20).text
        t = next(t for t in pd.read_html(io.StringIO(html)) if len(t) >= 90 and "Symbol" in t.columns)
        return pd.DataFrame({"ticker": t["Symbol"].astype(str), "name": t["Company Name"].astype(str), "sector": ""})
    except Exception as e:  # noqa: BLE001
        log.warning("stockanalysis fetch failed: %s", e)
    # 3) 위키
    return _wiki_table("https://en.wikipedia.org/wiki/Nasdaq-100", 90)


def _parse_cap(v: str) -> float | None:
    try:
        v = str(v).strip().replace(",", "")
        mult = {"T": 1e12, "B": 1e9, "M": 1e6}.get(v[-1], None)
        return float(v[:-1]) * mult if mult else float(v)
    except Exception:  # noqa: BLE001
        return None


def _us_marcap() -> dict[str, float]:
    """stockanalysis.com 리스트에서 S&P500 / Nasdaq-100 시가총액(USD)."""
    out: dict[str, float] = {}
    for url in ("https://stockanalysis.com/list/sp-500-stocks/", "https://stockanalysis.com/list/nasdaq-100-stocks/"):
        try:
            html = requests.get(url, headers=UA, timeout=20).text
            t = next(t for t in pd.read_html(io.StringIO(html)) if len(t) >= 80 and "Symbol" in t.columns and "Market Cap" in t.columns)
            for sym, cap in zip(t["Symbol"].astype(str), t["Market Cap"]):
                c = _parse_cap(cap)
                if c:
                    out[sym.strip().replace(".", "-")] = c
        except Exception as e:  # noqa: BLE001
            log.warning("US marcap fetch failed %s: %s", url, e)
    return out


def us_universe() -> list[dict]:
    sp = _sp500()
    nd = _nasdaq100()
    caps = _us_marcap()
    seen: dict[str, dict] = {}
    for df, tag in ((sp, "S&P500"), (nd, "NASDAQ100")):
        if df is None:
            continue
        for r in df.itertuples():
            sym = str(r.ticker).strip().replace(".", "-")  # BRK.B → BRK-B (야후 표기)
            if not sym or sym == "nan":
                continue
            if sym in seen:
                seen[sym]["market"] = "S&P500+NDX"
            else:
                seen[sym] = {"ticker": sym, "name": r.name, "market": tag, "region": "US", "sector": r.sector,
                             "marcap": caps.get(sym)}  # USD
    return list(seen.values())


def build_universe(kosdaq_top: int = 100, use_cache_on_fail: bool = True) -> list[dict]:
    cache = DATA_DIR / "universe.json"
    try:
        uni = kr_universe(kosdaq_top) + us_universe()
        cache.write_text(json.dumps(uni, ensure_ascii=False))
        return uni
    except Exception:
        if use_cache_on_fail and cache.exists():
            log.exception("universe build failed, using cache")
            return json.loads(cache.read_text())
        raise
