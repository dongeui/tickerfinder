"""국내상장 · 미국시장 추종 ETF 전용 스캔.

- 종목 목록: FinanceDataReader `StockListing('ETF/KR')` (네이버) 중 해외주식형(Category 4)이면서 이름이 미국 지수·테마를 가리키는 것.
  채권·혼합·통화·원자재형과 중국/일본/글로벌 등 비(非)미국은 제외. 환노출·환헤지(H) 모두 포함.
- 가격: scanner.fetch_kr_one (네이버 일봉, data/prices 캐시) → indicators.compute_metrics 로 매수/매도 스캔과 같은 지표.
- 결과: data/etf.json. 국내 스캔(정규/장중)이 끝날 때마다 scanner.AFTER_SCAN 훅으로 같이 갱신된다.
"""
from __future__ import annotations

import json
import logging
import re
import threading
from datetime import datetime, timedelta

import FinanceDataReader as fdr

from . import scanner
from .indicators import compute_metrics
from .universe import DATA_DIR

log = logging.getLogger(__name__)
RESULTS = DATA_DIR / "etf.json"
UNIVERSE = DATA_DIR / "etf_universe.json"
UNIVERSE_TTL_H = 20

US_PAT = re.compile(r"미국|나스닥|S&P|다우존스|필라델피아|러셀")
NON_US = re.compile(r"차이나|중국|일본|대만|유럽|한중|아시아|코리아|글로벌|K-|신흥|인도|베트남|홍콩|선진국|월드|한국")
SECTOR_WORDS = re.compile(r"테크놀로지|헬스케어|금융|에너지|산업재|소비재|유틸리티|커뮤니케이션|필수|성장주|가치주|모멘텀|TOP10|TOP\d|변동성|ESG|GOLD|버퍼|바이오|반도체|테크|배당|커버드콜|동일가중|저변동|퀄리티|서학개미|베스트셀러|빌리어네어|AI|로봇|우주|양자|원자력|SMR|전력|소프트웨어|방산|클린|부동산|리츠|인프라|데이터센터|클라우드|광통신|CPU|메모리|칩메이커|팹리스|에이전트|휴머노이드|드론|로보택시|천연가스|원유|은행|소비트렌드|중소형|제조업|스마트모빌리티|넥스트|WideMoat|IT인터넷|ETF산업|메디컬|게임|치매|우선증권|S&P500 TOP10|성장기업|방어배당|고배당|배당귀족|배당킹")

STATE = {"running": False, "stage": "", "done": 0, "total": 0, "started_at": None, "finished_at": None, "error": None}
_lock = threading.Lock()


def _set(**kw):
    with _lock:
        STATE.update(kw)


def classify(name: str) -> dict:
    """유형(kind)·환헤지(fx) 판정. fx: 'H'(환헤지) | 'open'(환노출) | 'JPY'(엔화노출)."""
    if re.search(r"레버리지|인버스|2X", name):
        kind = "레버리지·인버스"
    elif "커버드콜" in name:
        kind = "커버드콜"
    elif "배당" in name:
        kind = "배당"
    elif re.search(r"S&P500|나스닥100|다우존스30|러셀2000", name) and not SECTOR_WORDS.search(name):
        kind = "대표지수"
    else:
        kind = "섹터·테마"
    if "엔화노출" in name:
        fx = "JPY"
    elif re.search(r"\(H\)|합성 H\)|\(합성H\)", name):
        fx = "H"
    else:
        fx = "open"
    brand = name.split(" ", 1)[0] if " " in name else ""
    return {"kind": kind, "fx": fx, "brand": brand}


def _is_target(name: str, category) -> bool:
    try:
        cat = int(category)
    except (TypeError, ValueError):
        cat = None
    if cat != 4:  # 해외 주식형만 (채권 6 · 기타/혼합/통화 7 · 원자재 5 제외)
        return False
    if NON_US.search(name) or not US_PAT.search(name):
        return False
    return True


def _load_universe_cache() -> dict | None:
    try:
        return json.loads(UNIVERSE.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def universe(force: bool = False) -> list[dict]:
    """네이버 ETF 목록 → 대상만. 하루 캐시, 실패 시 이전 캐시."""
    cached = _load_universe_cache()
    if cached and not force:
        try:
            age = datetime.now(scanner.KST) - datetime.fromisoformat(cached["fetched_at"])
            if age < timedelta(hours=UNIVERSE_TTL_H):
                return cached["items"]
        except (KeyError, ValueError):
            pass
    try:
        df = fdr.StockListing("ETF/KR")
        items = []
        for r in df.itertuples(index=False):
            name = str(getattr(r, "Name", "")).strip()
            code = str(getattr(r, "Symbol", "")).strip()
            if not code or not _is_target(name, getattr(r, "Category", None)):
                continue

            def num(v):
                try:
                    f = float(v)
                    return f if f == f else None
                except (TypeError, ValueError):
                    return None

            nav, price = num(getattr(r, "NAV", None)), num(getattr(r, "Price", None))
            marcap = num(getattr(r, "MarCap", None))
            items.append({
                "ticker": code, "name": name, "region": "KR", "market": "KOSPI",
                **classify(name),
                "marcap": marcap * 1e8 if marcap else None,  # 억원 → 원
                "nav": nav, "list_price": price,
                "premium": round((price / nav - 1) * 100, 2) if nav and price else None,
            })
        if not items:
            raise RuntimeError("ETF 목록이 비어 있음")
        scanner.atomic_write_text(UNIVERSE, json.dumps(
            {"fetched_at": datetime.now(scanner.KST).isoformat(timespec="seconds"), "items": items}, ensure_ascii=False))
        return items
    except Exception as e:  # noqa: BLE001
        log.warning("ETF listing failed: %s", e)
        if cached:
            return cached["items"]
        raise


def uni_entry(ticker: str) -> dict | None:
    cached = _load_universe_cache()
    if not cached:
        return None
    for u in cached.get("items", []):
        if u.get("ticker") == ticker:
            return u
    return None


def load_results() -> dict | None:
    try:
        return json.loads(RESULTS.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def run(intraday: bool = False, set_partial: bool = True) -> dict:
    """전체 ETF 지표 계산. set_partial=True 면 scanner.KEEP_PARTIAL 을 직접 관리(단독 실행), False 면 run_scan 안에서 호출된 것."""
    if set_partial and not scanner.SCAN_LOCK.acquire(blocking=False):  # run_scan 과 같은 가격 캐시를 쓰므로 동시 실행 금지 (훅 호출은 이미 락 안)
        raise RuntimeError("전체 스캔이 실행 중이라 ETF 스캔을 시작할 수 없습니다")
    with _lock:
        if STATE["running"]:
            if set_partial:
                scanner.SCAN_LOCK.release()
            raise RuntimeError("ETF 스캔이 이미 실행 중")
        STATE.update(running=True, stage="목록", done=0, total=0, error=None,
                     started_at=datetime.now(scanner.KST).isoformat(timespec="seconds"), finished_at=None)
    if set_partial:
        scanner.KEEP_PARTIAL = intraday
    try:
        uni = universe()
        _set(total=len(uni), stage=f"ETF 가격 수집 ({len(uni)})")
        rows, failed = [], []
        for u in uni:
            df = scanner.fetch_kr_one(u["ticker"])
            m = compute_metrics(df) if df is not None else None
            with _lock:
                STATE["done"] += 1
            if m is None:
                failed.append(u["ticker"])
                continue
            pm = compute_metrics(df.iloc[:-1]) if len(df) > 61 else None
            row = {**u, **m}
            if pm:
                row["prev"] = {k: pm.get(k) for k in ("rsi_d", "rsi_w", "stoch_k", "stoch_k_w", "band_pos", "close", "date")}
            if intraday and scanner.is_partial_bar(df, "KR"):
                row["partial"] = True
            rows.append(row)
        now_s = datetime.now(scanner.KST).isoformat(timespec="seconds")
        result = {"scanned_at": now_s, "partial": intraday and any(r.get("partial") for r in rows),
                  "universe_at": (_load_universe_cache() or {}).get("fetched_at"), "etfs": rows, "failed": failed}
        scanner.atomic_write_text(RESULTS, json.dumps(result, ensure_ascii=False))
        _set(stage="완료")
        return result
    except Exception as e:  # noqa: BLE001
        log.exception("ETF scan failed")
        _set(error=str(e), stage="실패")
        raise
    finally:
        if set_partial:
            scanner.KEEP_PARTIAL = False
        _set(running=False, finished_at=datetime.now(scanner.KST).isoformat(timespec="seconds"))
        if set_partial:
            scanner.SCAN_LOCK.release()


def after_scan(regions, intraday: bool) -> None:
    """scanner.run_scan 끝에 호출되는 훅: 국내 스캔이 있었으면 ETF 도 같이 갱신."""
    if "KR" not in regions:
        return
    scanner._set(stage="ETF 동기화")
    run(intraday=intraday, set_partial=False)
