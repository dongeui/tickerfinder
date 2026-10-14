"""10년 백테스트: 매수 조건(시장 게이트 + 종목 조건)이 성립한 날 매수했을 때의 이후 성과와 추천 확률.

종목별로
- 진입 이벤트: 전날 미충족 → 당일 충족 (시장 게이트 + 스토캐스틱 + 150~200일선 밴드)
- 각 이벤트의 20/60/90/180/360 거래일 뒤 수익률, 구간 내 최고/최저 종가 대비 수익률
- 구간별: 승률(종가 수익 > 0), 반등확률(구간 내 최고가 +10% 이상 도달), 평균/중앙값
- 추천확률: 구간별 (0.6×승률 + 0.4×반등확률)을 지역 기준율로 수축(shrink)한 뒤 가중 평균
  (표본이 적은 종목은 기준율 쪽으로 끌려간다. k=8 이벤트 가중)
"""
from __future__ import annotations

import hashlib
import json
import logging
import threading
from datetime import datetime

import numpy as np
import pandas as pd

from .analysis import Thresholds
from .backfill import compute_series
from .indicators import rsi, stoch_point_in_time
from .longhist import INDEX_SYMS, load_long, signature
from .scanner import KST, atomic_write_text
from .universe import DATA_DIR

log = logging.getLogger(__name__)
HORIZONS = (20, 60, 90, 180, 360)
WEIGHTS = {20: 0.30, 60: 0.25, 90: 0.20, 180: 0.15, 360: 0.10}
REBOUND_PCT = 10.0  # 구간 내 최고가가 이만큼 오르면 '반등'으로 본다
SHRINK_K = 8
BT_DIR = DATA_DIR / "backtest"
BT_DIR.mkdir(exist_ok=True)

STATE = {"running": False, "stage": "", "done": 0, "total": 0, "error": None}
_lock = threading.Lock()
_mem: dict[str, dict] = {}


TF_FREQ = {"d": None, "w": "W-FRI", "m": "M"}


ENTRY_MODES = ("next_open", "close")  # 다음날 시가 매수(현실적) | 신호일 종가 매수


def _key(th: Thresholds, use_kr_index: bool, entry: str = "next_open") -> str:
    s = json.dumps({"st": th.stoch_max, "tol": th.band_tol, "lo": th.mkt_lo, "hi": th.mkt_hi, "kr": use_kr_index,
                    "tf_kr": th.stoch_tf_kr, "tf_us": th.stoch_tf_us, "entry": entry, "sig": signature()}, sort_keys=True)
    return hashlib.md5(s.encode()).hexdigest()[:12]


def _gate_series(th: Thresholds, use_kr_index: bool) -> dict[str, pd.Series]:
    """지역별 날짜 → 게이트 충족 여부. 국내에 나스닥을 쓸 때는 국내 장 마감 시점에 아는 '전일' 나스닥."""
    out: dict[str, pd.Series] = {}
    ix = load_long("IXIC")
    ks = load_long("KS11")
    if ix is None:
        raise RuntimeError("no IXIC long history; run long fetch first")
    r_ix = rsi(ix["Close"].dropna())
    g_ix = (r_ix >= th.mkt_lo) & (r_ix <= th.mkt_hi)
    out["US"] = g_ix
    if use_kr_index and ks is not None:
        r_ks = rsi(ks["Close"].dropna())
        out["KR"] = (r_ks >= th.mkt_lo) & (r_ks <= th.mkt_hi)
    else:
        out["KR"] = g_ix.shift(1)  # 날짜 정렬은 종목 쪽에서 asof 로 처리
    return out


def _fwd_table(c: pd.Series, o: pd.Series | None = None, entry: str = "next_open") -> dict[str, pd.Series]:
    """각 신호일 기준 H일 뒤 수익률, H일 내 최고/최저 종가 수익률 (%).

    entry="next_open": 신호일 다음 거래일 시가에 매수, H일은 그 매수일부터 센다 (실제 운용과 동일).
    entry="close":     신호일 종가에 매수 (이론값).
    """
    out: dict[str, pd.Series] = {}
    n = len(c)
    arr = c.to_numpy(dtype=float)
    if entry == "next_open" and o is not None:
        base = o.shift(-1).to_numpy(dtype=float)  # 다음날 시가
        lag = 1
    else:
        base = arr
        lag = 0
    out["entry_px"] = pd.Series(base, index=c.index)
    for h in HORIZONS:
        fwd = np.full(n, np.nan)
        k = h + lag
        if n > k:
            fwd[:-k] = arr[k:] / base[:-k] - 1
        out[f"fwd{h}"] = pd.Series(fwd * 100, index=c.index)
        rev = c[::-1]
        mx = rev.rolling(h, min_periods=h).max()[::-1].shift(-(1 + lag))
        mn = rev.rolling(h, min_periods=h).min()[::-1].shift(-(1 + lag))
        out[f"max{h}"] = (mx / base - 1) * 100
        out[f"min{h}"] = (mn / base - 1) * 100
    return out


def _events_for(ticker: str, region: str, th: Thresholds, gate: dict[str, pd.Series], entry: str = "next_open", df: pd.DataFrame | None = None) -> list[dict] | None:
    if df is None:
        df = load_long(ticker)
    if df is None or len(df) < 260:
        return None
    ser = compute_series(df)
    c = ser["close"]
    if th.stoch_tf(region) == "w":
        ser["stoch_k"] = ser["stoch_k_w"]
    elif TF_FREQ.get(th.stoch_tf(region)):
        ser["stoch_k"] = stoch_point_in_time(df.dropna(subset=["Close"]), TF_FREQ[th.stoch_tf(region)])
    g = gate[region].reindex(c.index, method="ffill").fillna(False).astype(bool)
    hit = (ser["stoch_k"] <= th.stoch_max) & (ser["band_pos"].abs() <= th.band_tol) & g & ~ser["suspect"].astype(bool)
    hit = hit.fillna(False)
    entry_days = hit & ~hit.shift(1, fill_value=False)
    if not entry_days.any():
        return []
    o = df["Open"].reindex(c.index) if "Open" in df else None
    fw = _fwd_table(c, o, entry)
    evs = []
    for t in c.index[entry_days.to_numpy()]:
        e = {"date": t.strftime("%Y-%m-%d"), "close": round(float(c[t]), 4), "entry_px": _r(fw["entry_px"][t], 4),
             "rsi_d": _r(ser.at[t, "rsi_d"]), "stoch_k": _r(ser.at[t, "stoch_k"]), "band_pos": _r(ser.at[t, "band_pos"])}
        for h in HORIZONS:
            e[f"fwd{h}"] = _r(fw[f"fwd{h}"][t])
            e[f"max{h}"] = _r(fw[f"max{h}"][t])
            e[f"min{h}"] = _r(fw[f"min{h}"][t])
        evs.append(e)
    return evs


def _r(v, nd=2):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(v) else round(v, nd)


def _hstats(evs: list[dict]) -> dict[str, dict]:
    out = {}
    for h in HORIZONS:
        f = [e[f"fwd{h}"] for e in evs if e.get(f"fwd{h}") is not None]
        m = [e[f"max{h}"] for e in evs if e.get(f"max{h}") is not None]
        lo = [e[f"min{h}"] for e in evs if e.get(f"min{h}") is not None]
        if not f:
            out[str(h)] = {"n": 0}
            continue
        out[str(h)] = {
            "n": len(f), "wins": int(sum(v > 0 for v in f)), "win": round(100 * sum(v > 0 for v in f) / len(f), 1),
            "mean": round(float(np.mean(f)), 2), "median": round(float(np.median(f)), 2),
            "rebounds": int(sum(v >= REBOUND_PCT for v in m)), "rebound": round(100 * sum(v >= REBOUND_PCT for v in m) / len(m), 1),
            "avg_max": round(float(np.mean(m)), 2), "avg_min": round(float(np.mean(lo)), 2),
        }
    return out


def _prob(hs: dict[str, dict], base: dict[str, dict]) -> tuple[float | None, dict]:
    """구간별 확률을 지역 기준율로 수축한 뒤 가중 평균. 반환: (추천확률 %, 구간별 수축 확률)."""
    parts, total_w, acc = {}, 0.0, 0.0
    for h in HORIZONS:
        s, b = hs.get(str(h), {}), base.get(str(h), {})
        if not b.get("n"):
            continue
        p0 = 0.6 * b["win"] + 0.4 * b["rebound"]
        n = s.get("n", 0)
        p_raw = (0.6 * s["win"] + 0.4 * s["rebound"]) if n else p0
        p = (n * p_raw + SHRINK_K * p0) / (n + SHRINK_K)
        parts[str(h)] = round(p, 1)
        acc += WEIGHTS[h] * p
        total_w += WEIGHTS[h]
    return (round(acc / total_w, 1) if total_w else None), parts


def compute(th: Thresholds, use_kr_index: bool = True, entry: str = "next_open") -> dict:
    entry = entry if entry in ENTRY_MODES else "next_open"
    key = _key(th, use_kr_index, entry)
    if key in _mem:
        return _mem[key]
    path = BT_DIR / f"{key}.json"
    if path.exists():
        try:
            _mem[key] = json.loads(path.read_text())
            for k in list(_mem)[:-3]:
                _mem.pop(k, None)
            return _mem[key]
        except Exception:  # noqa: BLE001
            pass
    with _lock:
        if STATE["running"]:
            raise RuntimeError("backtest already running")
        STATE.update(running=True, stage="백테스트 계산", done=0, total=0, error=None)
    try:
        uni_path = DATA_DIR / "universe.json"
        uni = json.loads(uni_path.read_text()) if uni_path.exists() else []
        gate = _gate_series(th, use_kr_index)
        _lock_set(total=len(uni))
        per: dict[str, dict] = {}
        all_evs: dict[str, list[dict]] = {"KR": [], "US": []}
        first_dates = []
        for u in uni:
            df = load_long(u["ticker"])
            evs = _events_for(u["ticker"], u["region"], th, gate, entry, df=df)
            _lock_set(done=STATE["done"] + 1)
            if df is not None and len(df):
                first_dates.append(df.index[0])
            if evs is None:
                continue
            per[u["ticker"]] = {"region": u["region"], "events": evs}
            all_evs[u["region"]].extend(evs)
        base = {reg: _hstats(evs) for reg, evs in all_evs.items()}
        base["ALL"] = _hstats(all_evs["KR"] + all_evs["US"])
        for reg in ("KR", "US", "ALL"):
            base[reg]["prob"], base[reg]["prob_parts"] = _prob({}, base[reg])
        tickers: dict[str, dict] = {}
        for t, d in per.items():
            hs = _hstats(d["events"])
            prob, parts = _prob(hs, base[d["region"]])
            tickers[t] = {"region": d["region"], "n": len(d["events"]), "h": hs, "prob": prob, "prob_parts": parts,
                          "events": d["events"][-40:],
                          "last_event": d["events"][-1]["date"] if d["events"] else None}
        gate_days = {reg: int(s.fillna(False).sum()) for reg, s in gate.items()}
        res = {"key": key, "computed_at": datetime.now(KST).isoformat(timespec="seconds"),
               "params": {"stoch_max": th.stoch_max, "band_tol": th.band_tol, "mkt_lo": th.mkt_lo, "mkt_hi": th.mkt_hi,
                          "use_kr_index": use_kr_index, "stoch_tf_kr": th.stoch_tf_kr, "stoch_tf_us": th.stoch_tf_us, "entry": entry, "rebound_pct": REBOUND_PCT, "shrink_k": SHRINK_K, "weights": WEIGHTS},
               "span": [min(first_dates).strftime("%Y-%m-%d") if first_dates else None, datetime.now(KST).strftime("%Y-%m-%d")],
               "gate_days": gate_days, "n_tickers": len(tickers), "base": base, "tickers": tickers}
        res["sig"] = list(signature())
        atomic_write_text(path, json.dumps(res, ensure_ascii=False))
        _mem[key] = res
        for k in list(_mem)[:-3]:  # 결과 하나가 10~20MB → 최근 3개만 메모리에
            _mem.pop(k, None)
        _prune(res["sig"])
        _lock_set(stage="완료")
        return res
    except Exception as e:  # noqa: BLE001
        log.exception("backtest failed")
        _lock_set(error=str(e), stage="실패")
        raise
    finally:
        _lock_set(running=False)


def _prune(sig: list) -> None:
    """장기 캐시가 바뀌어 더 이상 쓰이지 않는 결과 파일 삭제 (파일당 수 MB)."""
    for p in BT_DIR.glob("*.json"):
        try:
            if json.loads(p.read_text()).get("sig") != sig:
                p.unlink()
        except Exception:  # noqa: BLE001
            pass


def _lock_set(**kw):
    with _lock:
        STATE.update(kw)


def cached(th: Thresholds, use_kr_index: bool, entry: str = "next_open") -> dict | None:
    """계산 없이 캐시만 조회."""
    key = _key(th, use_kr_index, entry if entry in ENTRY_MODES else "next_open")
    if key in _mem:
        return _mem[key]
    path = BT_DIR / f"{key}.json"
    if path.exists():
        try:
            _mem[key] = json.loads(path.read_text())
            for k in list(_mem)[:-3]:
                _mem.pop(k, None)
            return _mem[key]
        except Exception:  # noqa: BLE001
            return None
    return None


def compact(res: dict) -> dict:
    """표용 요약: 종목별 확률/표본/20·60일 중앙값."""
    t = {}
    for k, v in res["tickers"].items():
        h20, h60 = v["h"].get("20", {}), v["h"].get("60", {})
        t[k] = {"prob": v["prob"], "n": v["n"], "last": v["last_event"],
                "win20": h20.get("win"), "med20": h20.get("median"), "reb20": h20.get("rebound"),
                "win60": h60.get("win"), "med60": h60.get("median")}
    return {k: v for k, v in res.items() if k != "tickers"} | {"tickers": t}
