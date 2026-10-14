"""가격 수집 + 지표 계산 + 결과 저장.

- 국내: FinanceDataReader(네이버 금융) 종목별 수집, 스레드 병렬
- 해외: yfinance 배치 다운로드
- 지수: 나스닥(^IXIC), 코스피(^KS11), 코스닥(^KQ11), S&P500(^GSPC)
"""
from __future__ import annotations

import json
import logging
import os
import socket
import threading
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import FinanceDataReader as fdr
import numpy as np
import pandas as pd
import yfinance as yf

from .indicators import compute_metrics, rsi, slow_stochastic
from .universe import DATA_DIR, build_universe

warnings.filterwarnings("ignore")
socket.setdefaulttimeout(20)  # FDR(네이버) 요청이 무한 대기하지 않도록
log = logging.getLogger(__name__)
KST = ZoneInfo("Asia/Seoul")
RESULTS = DATA_DIR / "results.json"
HISTORY_DIR = DATA_DIR / "history"
HISTORY_DIR.mkdir(exist_ok=True)
PRICE_DIR = DATA_DIR / "prices"  # 종목별 일봉 캐시 (parquet, 증분 갱신)
PRICE_DIR.mkdir(exist_ok=True)
CACHE_KEEP = 900  # 보관 행 수

LOOKBACK_DAYS = 600  # 200일선 + 주봉 RSI 워밍업 + 백필(6월~) 여유
INDICES = {"IXIC": "^IXIC", "GSPC": "^GSPC", "KS11": "^KS11", "KQ11": "^KQ11", "USDKRW": "KRW=X"}
# 스캔 완료 후 호출되는 훅 (regions, intraday) — ETF 전용 스캔 등. main 에서 등록.
AFTER_SCAN: list = []

STATE = {
    "running": False, "stage": "", "done": 0, "total": 0,
    "started_at": None, "finished_at": None, "error": None, "failed": [],
}
_lock = threading.Lock()


def _set(**kw):
    with _lock:
        STATE.update(kw)


def atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


SCAN_LOCK = threading.Lock()  # run_scan 과 etf.run 이 공유: 같은 가격 캐시를 동시에 쓰지 않도록
KEEP_PARTIAL = False  # 장중 동기화 모드: 미완성 봉을 지표 계산에는 쓰되 캐시/히스토리에는 저장하지 않음 (run_scan 이 설정)


def region_of(ticker: str) -> str:
    return "KR" if (ticker.isdigit() or ticker in ("^KS11", "^KQ11")) else "US"


def is_partial_bar(df: pd.DataFrame, region: str) -> bool:
    """마지막 봉이 아직 확정되지 않은 당일 봉인가 (KR: 18:05 KST 이전, US: 16:05 NY 이전).

    네이버 일봉은 16:00~18:00 시간외 단일가 체결가를 당일 종가로 내보내므로 국내는 18:00 이후에만 확정으로 본다.
    """
    if df is None or not len(df):
        return False
    if region == "KR":
        now = datetime.now(KST)
        return now.strftime("%H:%M") < "18:05" and df.index[-1].date() == now.date()
    now = datetime.now(ZoneInfo("America/New_York"))
    return now.strftime("%H:%M") < "16:05" and df.index[-1].date() == now.date()


def drop_partial_bar(df: pd.DataFrame, region: str, force: bool = False) -> pd.DataFrame:
    """장중에 스캔하면 마지막 봉이 미완성이므로 제거한다. KEEP_PARTIAL(장중 동기화) 중에는 force=True 일 때만 제거."""
    if df is None or not len(df):
        return df
    if KEEP_PARTIAL and not force:
        return df
    return df.iloc[:-1] if is_partial_bar(df, region) else df


# ---------- 로컬 가격 캐시 ----------
def _cache_path(ticker: str) -> Path:
    return PRICE_DIR / f"{ticker.replace('^', '_')}.parquet"


def load_cache(ticker: str) -> pd.DataFrame | None:
    p = _cache_path(ticker)
    if not p.exists():
        return None
    try:
        df = pd.read_parquet(p)
        return df if len(df) else None
    except Exception:  # noqa: BLE001
        return None


def merge_cache(ticker: str, new: pd.DataFrame | None) -> pd.DataFrame | None:
    """캐시 + 신규 데이터를 합쳐 저장하고 합친 결과를 반환. 신규가 없으면 캐시 그대로.

    겹치는 날짜의 종가가 1% 이상 다르면(액면분할·수정주가 변경) 캐시를 버리고 None을 반환한다.
    호출자는 캐시 파일이 사라진 것을 보고 전체 구간을 다시 받는다.
    """
    old = load_cache(ticker)
    if new is None or not len(new):
        return old
    new = new[[c for c in ("Open", "High", "Low", "Close", "Volume") if c in new.columns]].copy()
    new.index = pd.to_datetime(new.index).tz_localize(None).normalize()
    new = new[~new.index.duplicated(keep="last")]
    if old is not None:
        common = old.index.intersection(new.index)
        if len(common):
            ratio = (old.loc[common, "Close"] / new.loc[common, "Close"]).replace([np.inf, -np.inf], np.nan).dropna()
            if len(ratio) and (ratio - 1).abs().max() > 0.01:
                log.warning("%s: cached prices differ from fresh data (max %.1f%%) → cache reset", ticker, (ratio - 1).abs().max() * 100)
                _cache_path(ticker).unlink(missing_ok=True)
                return None
        df = pd.concat([old, new])
        df = df[~df.index.duplicated(keep="last")].sort_index()
    else:
        df = new.sort_index()
    df = df.tail(CACHE_KEEP)
    # 장중 미완성 봉은 캐시에 남기지 않는다 (장중 동기화 모드에서는 반환값에만 붙여 지표 계산에 사용)
    region = region_of(ticker)
    partial = df.iloc[-1:] if (KEEP_PARTIAL and ticker != "KRW=X" and is_partial_bar(df, region)) else None
    df = drop_partial_bar(df, region, force=True)
    try:
        tmp = _cache_path(ticker).with_suffix(".parquet.tmp")
        df.to_parquet(tmp)
        os.replace(tmp, _cache_path(ticker))  # 읽는 쪽이 반쯤 쓰인 파일을 보지 않도록
    except Exception as e:  # noqa: BLE001
        log.warning("cache write failed %s: %s", ticker, e)
    if partial is not None:
        df = pd.concat([df, partial])
    return df


def _cache_fresh(ticker: str, min_rows: int = 380) -> bool:
    """캐시가 충분히 길고 최근 것이면 증분 수집만 하면 된다."""
    df = load_cache(ticker)
    if df is None or len(df) < min_rows:
        return False
    return (datetime.now(KST).date() - df.index[-1].date()).days <= 14


# ---------- 수집 ----------
def _start_date(days: int = LOOKBACK_DAYS) -> str:
    return (datetime.now(KST) - timedelta(days=days)).strftime("%Y-%m-%d")


INCR_DAYS = 12  # 증분 수집 범위 (주말/휴일 여유)


def _incr_start(ticker: str) -> str:
    """캐시가 신선하면 캐시 마지막 날짜 - 5일부터, 아니면 전체 구간."""
    if _cache_fresh(ticker):
        last = load_cache(ticker).index[-1]
        return (last - timedelta(days=5)).strftime("%Y-%m-%d")
    return _start_date()


def _fdr_get(code: str, start: str, retries: int = 2) -> pd.DataFrame | None:
    for i in range(retries + 1):
        try:
            df = fdr.DataReader(code, start)
            if df is not None and len(df):
                return df
            return None
        except Exception as e:  # noqa: BLE001
            if i == retries:
                log.warning("KR fetch failed %s: %s", code, e)
            time.sleep(0.5 * (i + 1))
    return None


def fetch_kr_one(code: str) -> pd.DataFrame | None:
    new = drop_partial_bar(_fdr_get(code, _incr_start(code)), "KR")
    df = merge_cache(code, new)
    if df is None and new is not None and not _cache_path(code).exists():
        # 캐시가 리셋됨(분할 등) → 전체 구간 재수집
        df = merge_cache(code, drop_partial_bar(_fdr_get(code, _start_date()), "KR"))
    return df


def fetch_kr(codes: list[str]) -> dict[str, pd.DataFrame]:
    """네이버는 동시 요청 시 연결이 멈추는 경우가 있어 순차 수집 (실측 약 0.06초/종목)."""
    out: dict[str, pd.DataFrame] = {}
    for c in codes:
        df = fetch_kr_one(c)
        if df is not None:
            out[c] = df
        with _lock:
            STATE["done"] += 1
    return out


def _yf_batch(part: list[str], start: str) -> dict[str, pd.DataFrame]:
    got: dict[str, pd.DataFrame] = {}
    try:
        d = yf.download(part, start=start, group_by="ticker", progress=False, auto_adjust=True, threads=True)
    except Exception as e:  # noqa: BLE001
        log.warning("US chunk download failed: %s", e)
        return got
    if d is None or not len(d):
        return got
    for s in part:
        try:
            if isinstance(d.columns, pd.MultiIndex):
                if s in d.columns.get_level_values(0):
                    df = d[s]
                elif len(part) == 1:
                    df = d.droplevel(0, axis=1)
                else:
                    continue  # 배치에서 빠진 종목: 다른 종목 열을 합친 엉뚱한 표가 되지 않도록 건너뜀
            else:
                df = d
            df = df.dropna(subset=["Close"])
            if len(df):
                got[s] = drop_partial_bar(df, "US")
        except KeyError:
            pass
    return got


def fetch_us(symbols: list[str], chunk: int = 300) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    fresh = [s for s in symbols if _cache_fresh(s)]
    full = [s for s in symbols if s not in set(fresh)]
    groups: list[tuple[list[str], str | None]] = [(full, _start_date())]
    # 증분 대상은 캐시 마지막 날짜별로 묶어 시작일을 맞춘다
    by_start: dict[str, list[str]] = {}
    for s in fresh:
        by_start.setdefault(_incr_start(s), []).append(s)
    groups += list((v, k) for k, v in by_start.items())
    reset: list[str] = []
    for group, start in groups:
        for i in range(0, len(group), chunk):
            part = group[i:i + chunk]
            got = _yf_batch(part, start)
            for s in part:
                df = merge_cache(s, got.get(s))
                if df is None and s in got and not _cache_path(s).exists():
                    reset.append(s)
                elif df is not None:
                    out[s] = df
            with _lock:
                STATE["done"] += len(part)
    if reset:  # 분할 등으로 캐시가 리셋된 종목은 전체 재수집
        got = _yf_batch(reset, _start_date())
        for s in reset:
            df = merge_cache(s, got.get(s))
            if df is not None:
                out[s] = df
    return out


def fetch_index(sym: str) -> pd.DataFrame | None:
    start = _incr_start(sym)
    new = None
    try:
        df = yf.download(sym, start=start, progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if len(df):
            new = df if sym == "KRW=X" else drop_partial_bar(df, "KR" if sym in ("^KS11", "^KQ11") else "US")
    except Exception as e:  # noqa: BLE001
        log.warning("index fetch failed %s: %s", sym, e)
    if new is None:
        try:  # fallback: FDR
            new = fdr.DataReader(sym.lstrip("^"), start)
        except Exception:  # noqa: BLE001
            new = None
    return merge_cache(sym, new)


def index_metrics(df: pd.DataFrame) -> dict | None:
    m = compute_metrics(df)
    if m is None:
        return None
    c = df["Close"]
    r = rsi(c)
    sk, sd = slow_stochastic(df["High"], df["Low"], c)
    # 최근 60거래일 RSI / 스토캐스틱 경로 (차트용)
    tail = r.tail(60)
    m["rsi_series"] = [[d.strftime("%Y-%m-%d"), round(float(v), 1)] for d, v in tail.items() if pd.notna(v)]
    m["stoch_series"] = [[d.strftime("%Y-%m-%d"), round(float(v), 1)] for d, v in sk.tail(60).items() if pd.notna(v)]
    return m


# ---------- 히스토리 (봉 날짜 기준 스냅샷) ----------
SNAP_KEYS = ("ticker", "region", "rsi_d", "rsi_w", "stoch_k", "stoch_k_w", "band_pos", "close", "suspect")
IDX_KEYS = ("date", "close", "rsi_d", "rsi_w", "stoch_k", "stoch_d", "band_pos")


def write_history(rows_by_date: dict[str, list[dict]], indices_by_date: dict[str, dict], backfilled: bool = False) -> list[str]:
    """날짜별 파일에 종목 행/지수 값을 머지한다 (같은 티커는 덮어씀). 반환: 갱신된 날짜 목록."""
    touched = []
    for ds in sorted(set(rows_by_date) | set(indices_by_date)):
        path = HISTORY_DIR / f"{ds}.json"
        cur = {"date": ds, "indices": {}, "stocks": []}
        if path.exists():
            try:
                cur = json.loads(path.read_text())
            except Exception:  # noqa: BLE001
                pass
        by_t = {r["ticker"]: r for r in cur.get("stocks", [])}
        for r in rows_by_date.get(ds, []):
            by_t[r["ticker"]] = {k: r.get(k) for k in SNAP_KEYS}
        cur["stocks"] = sorted(by_t.values(), key=lambda r: r["ticker"])
        cur.setdefault("indices", {}).update({k: {kk: v.get(kk) for kk in IDX_KEYS} for k, v in indices_by_date.get(ds, {}).items()})
        cur["updated_at"] = datetime.now(KST).isoformat(timespec="seconds")
        if backfilled:
            cur["backfilled"] = True
        atomic_write_text(path, json.dumps(cur, ensure_ascii=False))
        touched.append(ds)
    return touched


# ---------- 반등 점수 ----------
DEFENSIVE = {"Health Care", "Consumer Staples", "Utilities"}


def _pts(v, table):
    """table: [(threshold, points), ...] 내림차순/오름차순 모두 허용. v가 None이면 0."""
    if v is None:
        return 0
    for th, pt, op in table:
        if (op == ">=" and v >= th) or (op == "<=" and v <= th):
            return pt
    return 0


def rebound_score(rows: list[dict]) -> None:
    """게이트가 열렸을 때 큰 반등을 보인 특성(2026-01~10 검증)으로 0~100 점수. rows 를 제자리 수정."""
    # 유동성(거래대금) 지역 내 백분위 — 시가총액 대용
    for region in ("KR", "US"):
        grp = [r for r in rows if r["region"] == region and r.get("turnover20")]
        grp.sort(key=lambda r: r["turnover20"])
        n = max(1, len(grp))
        for i, r in enumerate(grp):
            r["liq_pct"] = round(i / n, 2)
    for r in rows:
        parts = {
            "slope": _pts(r.get("ma_slope200"), [(6, 25, ">="), (3, 20, ">="), (2, 15, ">="), (0, 8, ">=")]),
            "dist50": _pts(r.get("dist_ma50"), [(-18, 25, "<="), (-9, 20, "<="), (-5, 10, "<=")]),
            "drop1m": _pts(r.get("ret_1m"), [(-20, 15, "<="), (-12, 12, "<="), (-8, 7, "<=")]),
            "rsi": _pts(r.get("rsi_d"), [(33, 15, "<="), (40, 12, "<="), (42, 6, "<=")]),
            "liq": _pts(r.get("liq_pct"), [(0.6, 10, ">="), (0.4, 5, ">=")]),
        }
        sec = r.get("sector") or ""
        parts["sector"] = 0 if sec in DEFENSIVE else (10 if sec else 5)  # 섹터 정보 없으면(국내) 중립 5
        r["score"] = sum(parts.values())
        r["score_parts"] = parts
        r["rule_a"] = bool((r.get("ma_slope200") or -99) >= 2 and (r.get("dist_ma50") or 0) <= -9 and (r.get("rsi_d") or 99) <= 40)


# ---------- 스캔 ----------
INDEX_REGION = {"IXIC": "US", "GSPC": "US", "USDKRW": "US", "KS11": "KR", "KQ11": "KR"}


def run_scan(kosdaq_top: int = 100, regions: tuple[str, ...] = ("KR", "US"), intraday: bool = False) -> dict:
    """전체 스캔. regions 를 일부만 주면 그 지역만 다시 수집하고 나머지 지역은 이전 결과를 유지한다.

    intraday=True(장중 동기화): 당일 미완성 봉을 포함해 지표를 계산하되 캐시·히스토리에는 저장하지 않는다.
    결과 행에는 partial=True, 결과에는 지역별 synced_at 이 기록된다.
    """
    global KEEP_PARTIAL
    if not SCAN_LOCK.acquire(blocking=False):
        raise RuntimeError("scan already running")
    with _lock:
        if STATE["running"]:
            SCAN_LOCK.release()
            raise RuntimeError("scan already running")
        STATE.update(running=True, stage="유니버스 구성", done=0, total=0, error=None, failed=[],
                     started_at=datetime.now(KST).isoformat(timespec="seconds"), finished_at=None,
                     mode=("장중 " if intraday else "") + "/".join(regions))
        KEEP_PARTIAL = intraday
    try:
        uni = build_universe(kosdaq_top)
        uni = [u for u in uni if u["region"] in regions]
        kr = [u for u in uni if u["region"] == "KR"]
        us = [u for u in uni if u["region"] == "US"]
        _set(total=len(uni), stage=f"가격 수집 (국내 {len(kr)} / 해외 {len(us)})")

        prices: dict[str, pd.DataFrame] = {}
        if kr:
            prices.update(fetch_kr([u["ticker"] for u in kr]))
        if us:
            prices.update(fetch_us([u["ticker"] for u in us]))

        _set(stage="지수 수집")
        prev = load_results() if len(regions) < 2 else None
        indices = {k: v for k, v in (prev or {}).get("indices", {}).items() if k in INDEX_REGION and INDEX_REGION[k] not in regions}
        for key, sym in INDICES.items():
            if INDEX_REGION[key] not in regions:
                continue
            df = fetch_index(sym)
            im = index_metrics(df) if df is not None else None
            if im is not None:
                indices[key] = im

        _set(stage="지표 계산")
        rows, failed = [], []
        rows_by_date: dict[str, list[dict]] = {}
        for u in uni:
            df = prices.get(u["ticker"])
            m = compute_metrics(df) if df is not None else None
            if m is None:
                failed.append(u["ticker"])
                continue
            pm = compute_metrics(df.iloc[:-1]) if len(df) > 61 else None
            row = {**u, **m}
            if pm:
                row["prev"] = {k: pm.get(k) for k in ("rsi_d", "rsi_w", "stoch_k", "stoch_k_w", "band_pos", "close", "date")}
            if intraday and is_partial_bar(df, u["region"]):
                row["partial"] = True
            rows.append(row)
            rows_by_date.setdefault(m["date"], []).append(row)

        now_s = datetime.now(KST).isoformat(timespec="seconds")
        synced = dict((prev or {}).get("synced_at", {}))
        if prev:  # 이번에 수집하지 않은 지역은 이전 결과 유지
            rows = [r for r in prev.get("stocks", []) if r.get("region") not in regions] + rows
            failed = [t for t in prev.get("failed", []) if t not in {u["ticker"] for u in uni}] + failed
            universe = dict(prev.get("universe", {}))
        else:
            universe = {}
        universe.update({"kosdaq_top": kosdaq_top, **({"kr": len(kr)} if kr else {}), **({"us": len(us)} if us else {})})
        for reg in regions:
            synced[reg] = {"at": now_s, "partial": intraday and any(r.get("partial") for r in rows if r.get("region") == reg)}
        rebound_score(rows)
        result = {
            "scanned_at": now_s,
            "synced_at": synced,
            "universe": universe,
            "indices": indices,
            "stocks": rows,
            "failed": failed,
        }
        atomic_write_text(RESULTS, json.dumps(result, ensure_ascii=False))
        if not intraday:  # 장중 잠정값은 히스토리(체류·백필)에 남기지 않는다
            idx_by_date: dict[str, dict] = {}
            for k, v in indices.items():
                if "date" in v and INDEX_REGION.get(k, "US") in regions:
                    idx_by_date.setdefault(v["date"], {})[k] = v
            write_history(rows_by_date, idx_by_date)
        for hook in AFTER_SCAN:
            try:
                hook(regions, intraday)
            except Exception:  # noqa: BLE001
                log.exception("after-scan hook failed")
        _set(failed=failed, stage="완료")
        return result
    except Exception as e:  # noqa: BLE001
        log.exception("scan failed")
        _set(error=str(e), stage="실패")
        raise
    finally:
        KEEP_PARTIAL = False
        _set(running=False, finished_at=datetime.now(KST).isoformat(timespec="seconds"))
        SCAN_LOCK.release()


def load_results() -> dict | None:
    if RESULTS.exists():
        return json.loads(RESULTS.read_text())
    return None


# ---------- 자동 스케줄 ----------
# 장중 동기화 간격(분). 기본 60. 국내: KST 09:05~15:35 (장 09:00~15:30), 미국: 뉴욕 09:35~16:00 (서머타임 자동 반영).
# 수집이 미국 약 30초 / 국내 약 1분이라 10분 밑으로는 권하지 않는다 (네이버·야후 호출 제한).
INTRADAY_MINUTES = max(5, min(60, int(os.environ.get("INTRADAY_MINUTES", "60"))))


def _times(start: str, end: str, step: int) -> list[str]:
    h, m = map(int, start.split(":"))
    eh, em = map(int, end.split(":"))
    out = []
    while (h, m) <= (eh, em):
        out.append(f"{h:02d}:{m:02d}")
        m += step
        h, m = h + m // 60, m % 60
    return out


INTRADAY_KR = sorted(set(_times("09:05", "15:30", INTRADAY_MINUTES) + ["15:35"]))
INTRADAY_US = sorted(set(_times("09:35", "15:55", INTRADAY_MINUTES) + ["16:00"]))
NY = ZoneInfo("America/New_York")


def in_session(now: datetime, region: str) -> bool:
    """정규장 중인가 (휴장일은 고려하지 않음)."""
    if region == "KR":
        return now.weekday() < 5 and "09:00" <= now.strftime("%H:%M") <= "15:30"
    ny = now.astimezone(NY)
    return ny.weekday() < 5 and "09:30" <= ny.strftime("%H:%M") <= "16:00"


def due_intraday(now: datetime) -> tuple[str, ...] | None:
    """지금 장중 동기화할 지역. 평일 KST 분 단위 일치 → ("KR",), 뉴욕 평일 분 단위 일치 → ("US",)."""
    if now.weekday() < 5 and now.strftime("%H:%M") in INTRADAY_KR:
        return ("KR",)
    ny = now.astimezone(NY)
    if ny.weekday() < 5 and ny.strftime("%H:%M") in INTRADAY_US:
        return ("US",)
    return None


SCAN_RUNS = DATA_DIR / "scan_runs.json"  # 정시 전체 스캔을 날짜별로 기록 (재시작·지연 후 따라잡기용)
CATCHUP_HOURS = 4  # 정시를 놓쳤어도 이 시간 안이면 늦게라도 전체 스캔


def _load_runs() -> dict[str, str]:
    try:
        return json.loads(SCAN_RUNS.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def due_full_scan(now: datetime, times: list[str], runs: dict[str, str]) -> str | None:
    """오늘 아직 안 돈 정시 스캔 중 시각이 지났고 CATCHUP_HOURS 이내인 것."""
    if now.weekday() >= 5:
        return None
    today = now.strftime("%Y-%m-%d")
    for t in times:
        if runs.get(t) == today:
            continue
        hh, mm = (int(x) for x in t.split(":"))
        at = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
        if at <= now < at + timedelta(hours=CATCHUP_HOURS):
            return t
    return None


def scheduler_loop(times: list[str], kosdaq_top: int = 100, intraday: bool = True):
    """매 30초 체크: 평일 지정 시각(KST)에 전체 스캔(놓쳤으면 4시간 내 따라잡기), 장중에는 시간별로 해당 지역만 잠정 동기화."""
    last_run: dict[str, str] = {}
    runs = _load_runs()
    while True:
        now = datetime.now(KST)
        key = now.strftime("%Y-%m-%d %H:%M")
        if not STATE["running"] and not SCAN_LOCK.locked():
            if (t := due_full_scan(now, times, runs)):
                runs[t] = now.strftime("%Y-%m-%d")
                try:
                    atomic_write_text(SCAN_RUNS, json.dumps(runs))
                except OSError:
                    pass
                try:
                    run_scan(kosdaq_top)
                except Exception:  # noqa: BLE001
                    pass
            elif intraday and (regs := due_intraday(now)) and last_run.get("intraday") != key:
                last_run["intraday"] = key
                try:
                    run_scan(kosdaq_top, regions=regs, intraday=True)
                except Exception:  # noqa: BLE001
                    pass
        time.sleep(30)
