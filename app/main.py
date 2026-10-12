"""FastAPI 서버. `uv run uvicorn app.main:app --port 8765`"""
from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import analysis, autosell, backtest, kis, longhist, positions, scanner
from .universe import DATA_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("tickerfinder")

ROOT = Path(__file__).resolve().parent.parent
STATIC = ROOT / "static"
WATCHLIST = DATA_DIR / "watchlist.json"
KOSDAQ_TOP = int(os.environ.get("KOSDAQ_TOP", "100"))
# 평일 자동 스캔 시각(KST). 국내는 시간외 단일가(~18:00) 종료 후, 미국은 마감 후. 빈 문자열이면 비활성.
AUTO_SCAN_TIMES = [t.strip() for t in os.environ.get("AUTO_SCAN_TIMES", "18:10,06:40").split(",") if t.strip()]
# 장중 시간별 잠정 동기화 (국내 KST 09:05~15:35, 미국 뉴욕 10:05~15:05). INTRADAY_SYNC=0 이면 끔.
INTRADAY_SYNC = os.environ.get("INTRADAY_SYNC", "1") not in ("0", "false", "no")

app = FastAPI(title="tickerfinder")


@app.on_event("startup")
def _start_scheduler():
    if AUTO_SCAN_TIMES:
        threading.Thread(target=scanner.scheduler_loop, args=(AUTO_SCAN_TIMES, KOSDAQ_TOP, INTRADAY_SYNC), daemon=True).start()
        log.info("auto scan at %s KST (weekdays), intraday sync %s", AUTO_SCAN_TIMES, "on" if INTRADAY_SYNC else "off")
    autosell.KOSDAQ_TOP = KOSDAQ_TOP
    threading.Thread(target=autosell.loop, daemon=True).start()
    cfg = autosell.config()
    log.info("auto sell %s (dry=%s) KR %s KST / US %s NY", "on" if cfg["enabled"] else "off", cfg["dry"], cfg["time_kr"], cfg["time_us"])


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.post("/api/scan")
def start_scan(region: str | None = None, intraday: bool = False):
    """전체 스캔. region=KR|US 이면 그 지역만, intraday=true 면 장중 잠정 동기화(미완성 봉 포함, 히스토리 미기록)."""
    if scanner.STATE["running"]:
        return JSONResponse({"ok": False, "msg": "already running"}, status_code=409)
    regions = (region.upper(),) if region and region.upper() in ("KR", "US") else ("KR", "US")

    def _run():
        try:
            scanner.run_scan(KOSDAQ_TOP, regions=regions, intraday=intraday)
        except Exception:  # noqa: BLE001  (RuntimeError: already running 등은 STATE에 기록됨)
            pass

    threading.Thread(target=_run, daemon=True).start()
    return {"ok": True, "regions": regions, "intraday": intraday}


@app.get("/api/status")
def status():
    now = datetime.now(scanner.KST)
    return {**scanner.STATE, "auto_scan_times": AUTO_SCAN_TIMES, "kosdaq_top": KOSDAQ_TOP, "intraday_sync": INTRADAY_SYNC,
            "intraday_times": {"KR": scanner.INTRADAY_KR, "US_ny": scanner.INTRADAY_US, "minutes": scanner.INTRADAY_MINUTES},
            "session": {"KR": scanner.in_session(now, "KR"), "US": scanner.in_session(now, "US")}}


@app.get("/api/results")
def results():
    r = scanner.load_results()
    if r is None:
        raise HTTPException(404, "no results yet; run a scan")
    return r


class Watch(BaseModel):
    tickers: list[str]


@app.get("/api/watchlist")
def get_watchlist():
    if WATCHLIST.exists():
        return json.loads(WATCHLIST.read_text())
    return {"tickers": []}


@app.put("/api/watchlist")
def put_watchlist(w: Watch):
    tickers = sorted({t.strip().upper() for t in w.tickers if t.strip()})
    scanner.atomic_write_text(WATCHLIST, json.dumps({"tickers": tickers}, ensure_ascii=False))
    return {"tickers": tickers}


def _tf(v: str) -> str:
    return v if v in ("d", "w", "m") else "d"


def _th(stoch_max: float = 25, band_tol: float = 3, mkt_lo: float = 30, mkt_hi: float = 40,
        rsi_d_min: float = 75, rsi_w_lo: float = 65, rsi_w_hi: float = 85,
        stoch_tf_kr: str = "d", stoch_tf_us: str = "w") -> analysis.Thresholds:
    return analysis.Thresholds(stoch_max, band_tol, mkt_lo, mkt_hi, rsi_d_min, rsi_w_lo, rsi_w_hi, _tf(stoch_tf_kr), _tf(stoch_tf_us))


@app.get("/api/history")
def history(
    limit: int = Query(200, le=2000),
    stoch_max: float = 25, band_tol: float = 3, mkt_lo: float = 30, mkt_hi: float = 40,
    rsi_d_min: float = 75, rsi_w_lo: float = 65, rsi_w_hi: float = 85, stoch_tf_kr: str = "d", stoch_tf_us: str = "w",
):
    """날짜별 지수 지표 + 임계값 기준 매수/매도 시그널 종목."""
    th = _th(stoch_max, band_tol, mkt_lo, mkt_hi, rsi_d_min, rsi_w_lo, rsi_w_hi, stoch_tf_kr, stoch_tf_us)
    return {"days": analysis.daily_counts(th, limit)}


@app.get("/api/signals")
def signals(
    mode: str = "buy", require_mkt: bool = False, use_kr_index: bool = False, limit: int = Query(600, le=5000),
    stoch_max: float = 25, band_tol: float = 3, mkt_lo: float = 30, mkt_hi: float = 40,
    rsi_d_min: float = 75, rsi_w_lo: float = 65, rsi_w_hi: float = 85, stoch_tf_kr: str = "d", stoch_tf_us: str = "w",
):
    """시그널 진입 이벤트 + 이후 5/10/20거래일 수익률 요약."""
    if mode not in ("buy", "sell"):
        raise HTTPException(400, "mode must be buy or sell")
    th = _th(stoch_max, band_tol, mkt_lo, mkt_hi, rsi_d_min, rsi_w_lo, rsi_w_hi, stoch_tf_kr, stoch_tf_us)
    return analysis.signal_events(mode, th, require_mkt, use_kr_index, limit)


@app.get("/api/streaks")
def streaks(
    mode: str = "buy",
    stoch_max: float = 25, band_tol: float = 3, mkt_lo: float = 30, mkt_hi: float = 40,
    rsi_d_min: float = 75, rsi_w_lo: float = 65, rsi_w_hi: float = 85, stoch_tf_kr: str = "d", stoch_tf_us: str = "w",
):
    """종목별 연속 충족 일수 / 오늘 이탈 정보."""
    if mode not in ("buy", "sell"):
        raise HTTPException(400, "mode must be buy or sell")
    return analysis.streaks(mode, _th(stoch_max, band_tol, mkt_lo, mkt_hi, rsi_d_min, rsi_w_lo, rsi_w_hi, stoch_tf_kr, stoch_tf_us))


@app.get("/api/ticker/{ticker}")
def ticker(ticker: str):
    return analysis.ticker_series(ticker.upper())


# ---------- 최근 60일 종가 (표의 미니 추세선) ----------
_SPARK: dict = {"key": None, "data": {}}


@app.get("/api/sparks")
def sparks(n: int = Query(60, le=250)):
    files = sorted(scanner.PRICE_DIR.glob("*.parquet"))
    key = (n, len(files), max((f.stat().st_mtime_ns for f in files), default=0))
    if _SPARK["key"] != key:
        out = {}
        for f in files:
            if f.name.startswith("_"):
                continue
            try:
                c = pd.read_parquet(f, columns=["Close"])["Close"].dropna().tail(n)
                out[f.stem] = [round(float(v), 4) for v in c]
            except Exception:  # noqa: BLE001
                pass
        _SPARK.update(key=key, data=out)
    return _SPARK["data"]


# ---------- 10년 백테스트 ----------
def _bt_th(stoch_max: float, band_tol: float, mkt_lo: float, mkt_hi: float, stoch_tf_kr: str = "d", stoch_tf_us: str = "w") -> analysis.Thresholds:
    return analysis.Thresholds(stoch_max=stoch_max, band_tol=band_tol, mkt_lo=mkt_lo, mkt_hi=mkt_hi,
                               stoch_tf_kr=_tf(stoch_tf_kr), stoch_tf_us=_tf(stoch_tf_us))


def _bt_run(th: analysis.Thresholds, use_kr_index: bool, fetch: bool, years: int | None = None, entry: str = "next_open"):
    try:
        if fetch:
            uni_path = DATA_DIR / "universe.json"
            uni = json.loads(uni_path.read_text()) if uni_path.exists() else []
            longhist.fetch_all(uni, years=years or longhist.YEARS, force=bool(years))
        backtest.compute(th, use_kr_index, entry)
    except Exception:  # noqa: BLE001
        log.exception("backtest run failed")


@app.post("/api/backtest/run")
def backtest_run(
    fetch: bool = True, use_kr_index: bool = True, stoch_tf_kr: str = "d", stoch_tf_us: str = "w",
    stoch_max: float = 25, band_tol: float = 3, mkt_lo: float = 30, mkt_hi: float = 40, years: int | None = Query(None, ge=1, le=30),
    entry: str = "next_open",
):
    """장기 데이터 수집(fetch=true) 후 백테스트 계산을 백그라운드로 실행. years 를 주면 그 기간으로 전체 재수집."""
    if longhist.STATE["running"] or backtest.STATE["running"]:
        return JSONResponse({"ok": False, "msg": "already running"}, status_code=409)
    threading.Thread(target=_bt_run, args=(_bt_th(stoch_max, band_tol, mkt_lo, mkt_hi, stoch_tf_kr, stoch_tf_us), use_kr_index, fetch, years, entry), daemon=True).start()
    return {"ok": True}


@app.get("/api/backtest/status")
def backtest_status():
    n = len(list(longhist.LONG_DIR.glob("*.parquet")))
    return {"fetch": longhist.STATE, "compute": backtest.STATE, "long_files": n}


@app.get("/api/backtest")
def backtest_summary(
    use_kr_index: bool = True, stoch_tf_kr: str = "d", stoch_tf_us: str = "w", entry: str = "next_open",
    stoch_max: float = 25, band_tol: float = 3, mkt_lo: float = 30, mkt_hi: float = 40,
):
    """종목별 추천확률 요약. 캐시가 없으면 계산을 시작하고 202를 돌려준다."""
    th = _bt_th(stoch_max, band_tol, mkt_lo, mkt_hi, stoch_tf_kr, stoch_tf_us)
    res = backtest.cached(th, use_kr_index, entry)
    if res is None:
        if not longhist.load_long("IXIC") is None and not (longhist.STATE["running"] or backtest.STATE["running"]):
            threading.Thread(target=_bt_run, args=(th, use_kr_index, False, None, entry), daemon=True).start()
        return JSONResponse({"ready": False, "status": backtest_status()}, status_code=202)
    return {"ready": True, **backtest.compact(res)}


@app.get("/api/backtest/{ticker}")
def backtest_ticker(
    ticker: str, use_kr_index: bool = True, stoch_tf_kr: str = "d", stoch_tf_us: str = "w", entry: str = "next_open",
    stoch_max: float = 25, band_tol: float = 3, mkt_lo: float = 30, mkt_hi: float = 40,
):
    res = backtest.cached(_bt_th(stoch_max, band_tol, mkt_lo, mkt_hi, stoch_tf_kr, stoch_tf_us), use_kr_index, entry)
    if res is None:
        raise HTTPException(404, "backtest not computed yet")
    t = res["tickers"].get(ticker.upper())
    if t is None:
        raise HTTPException(404, "no long history for ticker")
    return {"ticker": ticker.upper(), **t, "base": res["base"][t["region"]], "params": res["params"], "span": res["span"], "gate_days": res["gate_days"]}


# ---------- 지수 RSI 장기 분포 (참고 지표) ----------
_RSI_DIST: dict = {"key": None, "data": None}


@app.get("/api/rsi_dist")
def rsi_dist():
    """나스닥/코스피 RSI(14)의 장기(캐시 전체) 분포: 평균·분위·구간 비율·현재 백분위."""
    from .indicators import rsi as _rsi
    key = longhist.signature()
    if _RSI_DIST["key"] == key and _RSI_DIST["data"]:
        return _RSI_DIST["data"]
    out = {}
    for sym in ("IXIC", "KS11"):
        df = longhist.load_long(sym)
        if df is None:
            continue
        r = _rsi(df["Close"].dropna()).dropna()
        cur = float(r.iloc[-1])
        out[sym] = {
            "span": [r.index[0].strftime("%Y-%m-%d"), r.index[-1].strftime("%Y-%m-%d")], "days": int(len(r)),
            "mean": round(float(r.mean()), 1), "median": round(float(r.median()), 1), "std": round(float(r.std()), 1),
            "cur": round(cur, 1), "cur_pctile": round(float((r <= cur).mean() * 100), 1),
            "pct": {str(int(q * 100)): round(float(r.quantile(q)), 1) for q in (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)},
            "le": {str(t): round(float((r <= t).mean() * 100), 1) for t in (30, 35, 40, 45, 50)},
            "ge": {str(t): round(float((r >= t).mean() * 100), 1) for t in (60, 65, 70, 75, 80)},
            "year_mean": {str(y): round(float(v), 1) for y, v in r.groupby(r.index.year).mean().items()},
        }
    _RSI_DIST.update(key=key, data=out)
    return out


# ---------- 한국투자증권 연동 ----------
@app.get("/api/kis/status")
def kis_status():
    c = kis.client()
    out = {"configured": c.configured, "mode": c.mode, "account": (c.cano[:4] + "****-" + c.prdt) if c.cano else None, "token_ok": None, "error": None}
    if c.configured:
        try:
            c.token()
            out["token_ok"] = True
        except Exception as e:  # noqa: BLE001
            out["token_ok"] = False
            out["error"] = str(e)
    return out


@app.get("/api/kis/balance")
def kis_balance():
    c = kis.client()
    if not c.configured:
        raise HTTPException(400, ".env 에 KIS 설정이 없습니다")
    out = {"mode": c.mode, "kr": None, "us": None, "errors": []}
    for reg, fn in (("kr", c.balance_kr), ("us", c.balance_us)):
        try:
            out[reg] = fn()
        except Exception as e:  # noqa: BLE001
            out["errors"].append(f"{reg}: {e}")
    out["orders"] = kis.order_history(50)
    return out


@app.get("/api/kis/trades")
def kis_trades(days: int = 30):
    """한투 주문·체결 내역 (국내+미국, 최근 days일). 이 앱에서 보낸 주문은 app=True 로 표시."""
    c = kis.client()
    if not c.configured:
        raise HTTPException(400, ".env 에 KIS 설정이 없습니다")
    days = max(1, min(90, days))
    end = datetime.now(scanner.KST)
    start = (end - timedelta(days=days)).strftime("%Y%m%d")
    out = {"mode": c.mode, "days": days, "rows": [], "errors": []}
    for fn in (c.trades_kr, c.trades_us):
        try:
            out["rows"] += fn(start, end.strftime("%Y%m%d"))
        except Exception as e:  # noqa: BLE001
            out["errors"].append(str(e))
    app_odno = {o.get("odno") for o in kis.order_history(1000) if o.get("odno")}
    for r in out["rows"]:
        r["app"] = r.get("odno") in app_odno or r.get("orgn_odno") in app_odno
    out["rows"].sort(key=lambda r: (r["date"], r["time"]), reverse=True)
    out["app_orders"] = kis.order_history(100)
    return out


@app.get("/api/kis/price/{ticker}")
def kis_price(ticker: str):
    c = kis.client()
    if not c.configured:
        raise HTTPException(400, ".env 에 KIS 설정이 없습니다")
    ticker = ticker.upper()
    try:
        if ticker.isdigit():
            return {"ticker": ticker, "region": "KR", "price": c.price_kr(ticker), "currency": "KRW"}
        px, excg = c.price_us(ticker)
        return {"ticker": ticker, "region": "US", "price": px, "currency": "USD", "excg": excg}
    except Exception as e:  # noqa: BLE001
        raise HTTPException(502, str(e))


class Order(BaseModel):
    ticker: str
    side: str  # buy | sell
    qty: int
    price: float


@app.post("/api/kis/order")
def kis_order(o: Order):
    """지정가 주문 (화면에서 사용자가 확인 버튼을 누른 뒤에만 호출)."""
    c = kis.client()
    if not c.configured:
        raise HTTPException(400, ".env 에 KIS 설정이 없습니다")
    if o.side not in ("buy", "sell"):
        raise HTTPException(400, "side must be buy or sell")
    t = o.ticker.upper()
    try:
        return c.order("KR" if t.isdigit() else "US", t, o.side, o.qty, o.price)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(502, str(e))


# ---------- 보유 포지션 / 매도 규칙 ----------
class Position(BaseModel):
    ticker: str
    entry_date: str | None = None
    entry_px: float | None = None
    qty: int | None = None
    note: str | None = None
    half_done: bool | None = None


@app.get("/api/positions")
def get_positions():
    return positions.load_positions()


@app.put("/api/positions")
def put_position(p: Position):
    pos = positions.load_positions()
    t = p.ticker.strip().upper()
    cur = pos.get(t, {})
    for k in ("entry_date", "entry_px", "qty", "note", "half_done"):
        v = getattr(p, k)
        if v is not None:
            cur[k] = v
    if p.half_done:
        cur.setdefault("half_date", datetime.now().strftime("%Y-%m-%d"))
    elif p.half_done is False:
        cur.pop("half_date", None)
    pos[t] = cur
    positions.save_positions(pos)
    return pos


@app.delete("/api/positions/{ticker}")
def del_position(ticker: str):
    pos = positions.load_positions()
    pos.pop(ticker.upper(), None)
    positions.save_positions(pos)
    return pos


@app.get("/api/exit_rule")
def get_exit_rule():
    return positions.load_rule().__dict__


class RuleIn(BaseModel):
    rsi_d_min: float = 75
    rsi_w_lo: float = 65
    rsi_w_hi: float = 85
    arm_us: float = 10
    trail_us: float = 15
    arm_kr: float = 5
    trail_kr: float = 10
    max_hold: int = 360


@app.put("/api/exit_rule")
def put_exit_rule(r: RuleIn):
    """매도 규칙 저장. 계좌 탭 판정과 자동매도가 같은 값을 쓴다."""
    rule = positions.ExitRule(**r.model_dump())
    positions.save_rule(rule)
    return rule.__dict__


@app.get("/api/exit_signals")
def exit_signals(with_broker: bool = True):
    """포지션별 매도 규칙 판정(저장된 규칙). with_broker=true 이고 KIS 설정이 있으면 한투 보유 종목도 합친다."""
    rule = positions.load_rule()
    holdings, errors = [], []
    c = kis.client()
    if with_broker and c.configured:
        for fn in (c.balance_kr, c.balance_us):
            try:
                holdings += fn()["holdings"]
            except Exception as e:  # noqa: BLE001
                errors.append(str(e))
    return {"positions": positions.evaluate_all(rule, holdings), "rule": rule.__dict__, "errors": errors, "broker": bool(with_broker and c.configured)}


# ---------- 자동 매도 ----------
@app.get("/api/autosell/status")
def autosell_status():
    cfg = autosell.config()
    hist = autosell.history(200)
    return {**cfg, "running": autosell.STATE["running"], "next": autosell.next_runs(), "last": autosell.STATE["last"],
            "today_orders": {r: len(autosell.today_orders(r, autosell.trade_date(r), hist)) for r in ("KR", "US")}, "history": hist[:30]}


@app.post("/api/autosell/run")
def autosell_run(region: str = Query(..., pattern="^(KR|US)$"), dry: bool = True):
    """수동 실행(테스트). dry=false 는 실제 주문 — AUTO_SELL=1 이고 정규장 중일 때만."""
    if autosell.STATE["running"] or scanner.STATE["running"]:
        raise HTTPException(409, "스캔 또는 자동매도가 실행 중")
    return autosell.run(region, dry=dry, force=dry)


app.mount("/static", StaticFiles(directory=STATIC), name="static")
