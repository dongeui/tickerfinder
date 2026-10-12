"""자동 매도. 평일 장 마감 전 지역별 1회(국내 KST 15:10, 미국 뉴욕 15:30) 매도 규칙을 판정해 지정가 매도 주문을 보낸다.

.env:
  AUTO_SELL=1            켬 (기본 0)
  AUTO_SELL_DRY=1        드라이런: 주문은 안 보내고 기록·알림만 (기본 1)
  AUTO_SELL_MAX_PER_DAY  거래일·지역별 자동 주문 상한 (기본 3)
  AUTO_SELL_LIMIT_PCT    지정가 = 현재가 × (1 − 이 값%) (기본 0.5)
  AUTO_SELL_TIME_KR / AUTO_SELL_TIME_US  판정 시각 (기본 15:10 KST / 15:30 뉴욕)

안전장치(모두 건너뛰면 기록·알림만):
  정규장 중일 때만 · 당일 데이터만 · 전일 대비 ±20% 또는 suspect 표시 종목 제외 · 한투 현재가와 스캔 종가 3% 이상 차이 제외
  · 매수일 기록된 종목만 · 한투 잔고 수량 이내 · 같은 종목 미체결 매도 주문 있으면 제외 · 그 거래일에 이미 자동 주문한 종목 제외 · 거래일·지역별 상한
"""
from __future__ import annotations

import json
import logging
import math
import os
import subprocess
import threading
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from . import kis, positions, scanner
from .scanner import KST, NY, in_session
from .universe import DATA_DIR

log = logging.getLogger("tickerfinder.autosell")
LOG_FILE = DATA_DIR / "autosell.jsonl"
STATE: dict = {"running": False, "last": {}, "error": None}
_lock = threading.Lock()


KOSDAQ_TOP = 100  # main 이 시작 시 덮어씀 (스케줄 스캔과 같은 유니버스)


def _num(v, default, cast):
    try:
        return cast(v)
    except (TypeError, ValueError):
        log.warning("autosell: 잘못된 설정값 %r → 기본값 %s", v, default)
        return default


def _hm(v: str, default: str) -> str:
    try:
        h, m = map(int, str(v).split(":"))
        return f"{h:02d}:{m:02d}" if 0 <= h < 24 and 0 <= m < 60 else default
    except ValueError:
        return default


def config() -> dict:
    env = kis.load_env()
    g = lambda k, d: os.environ.get(k) or env.get(k) or d  # noqa: E731
    return {
        "enabled": str(g("AUTO_SELL", "0")).lower() in ("1", "true", "yes", "on"),
        "dry": str(g("AUTO_SELL_DRY", "1")).lower() in ("1", "true", "yes", "on"),
        "max_per_day": max(0, _num(g("AUTO_SELL_MAX_PER_DAY", "3"), 3, int)),
        "limit_pct": _num(g("AUTO_SELL_LIMIT_PCT", "0.5"), 0.5, float),
        "time_kr": _hm(g("AUTO_SELL_TIME_KR", "15:10"), "15:10"),
        "time_us": _hm(g("AUTO_SELL_TIME_US", "15:30"), "15:30"),
        "mode": env.get("KIS_MODE", "paper"),
    }


def _append(rec: dict) -> None:
    with LOG_FILE.open("a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def history(limit: int = 50) -> list[dict]:
    """최근 실행 기록(최신순). 깨진 줄은 건너뛴다."""
    if not LOG_FILE.exists():
        return []
    rows = []
    for line in LOG_FILE.read_text().splitlines():
        try:
            if line.strip():
                rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows[-limit:][::-1]


def trade_date(region: str, now: datetime | None = None) -> str:
    """지역 기준 거래일 (국내 KST, 미국 뉴욕)."""
    now = now or datetime.now(KST)
    return (now if region == "KR" else now.astimezone(NY)).strftime("%Y-%m-%d")


def today_orders(region: str, tdate: str, hist: list[dict] | None = None) -> list[dict]:
    """그 지역 거래일에 실제 전송된 자동 주문(드라이런 제외)."""
    out = []
    for run_ in hist if hist is not None else history(200):
        if run_.get("region") == region and run_.get("trade_date") == tdate:
            out += [d for d in run_.get("decisions", []) if d.get("status") == "ordered"]
    return out


def notify(title: str, msg: str) -> None:
    log.info("%s: %s", title, msg)
    try:
        subprocess.run(["osascript", "-e", f'display notification "{msg.replace(chr(34), chr(39))}" with title "{title.replace(chr(34), chr(39))}"'],
                       timeout=5, capture_output=True)
    except Exception:  # noqa: BLE001
        pass


def kr_tick(px: float) -> float:
    """KRX 호가 단위로 내림."""
    for lim, tick in ((2000, 1), (5000, 5), (20000, 10), (50000, 50), (200000, 100), (500000, 500)):
        if px < lim:
            return math.floor(px / tick) * tick
    return math.floor(px / 1000) * 1000


def _sync(region: str, rec: dict) -> None:
    """그 지역 장중 동기화. 다른 스캔이 돌고 있으면 기다렸다가(최대 약 3분) 재시도."""
    for _attempt in range(3):
        for _ in range(18):
            if not scanner.STATE["running"]:
                break
            time.sleep(10)
        try:
            scanner.run_scan(KOSDAQ_TOP, regions=(region,), intraday=True)
            rec.pop("sync_error", None)
            return
        except RuntimeError as e:  # 같은 순간 스케줄 스캔이 먼저 잡은 경우 → 끝나길 기다렸다 재시도
            rec["sync_error"] = str(e)
            if "already running" not in str(e):
                return
        except Exception as e:  # noqa: BLE001
            rec["sync_error"] = str(e)
            return


def run(region: str, dry: bool | None = None, force: bool = False, wait: float = 0) -> dict:
    """한 지역 판정·주문. force=True 면 AUTO_SELL=0 이어도 실행(수동 테스트). wait: 다른 실행이 끝날 때까지 기다릴 초.
    결과 레코드를 반환하고 기록한다."""
    cfg = config()
    if dry is None:
        dry = cfg["dry"]
    now = datetime.now(KST)
    local_today = trade_date(region, now)
    rec = {"at": now.isoformat(timespec="seconds"), "region": region, "trade_date": local_today, "dry": dry, "mode": cfg["mode"], "decisions": [], "error": None}
    got = _lock.acquire(timeout=wait) if wait > 0 else _lock.acquire(blocking=False)
    if not got:
        rec["error"] = "이미 실행 중 (다른 자동매도/드라이런이 끝나지 않음)"
        _append(rec)
        notify("자동매도 건너뜀", f"{region}: {rec['error']}")
        return rec
    STATE["running"] = True
    try:
        if not cfg["enabled"] and not force:
            rec["error"] = "AUTO_SELL=0"
            return rec
        c = kis.client()
        if not c.configured:
            rec["error"] = "KIS 설정 없음"
            notify("자동매도 오류", rec["error"])
            return rec
        if not in_session(now, region) and not force:
            rec["error"] = f"{region} 정규장 아님"
            return rec
        # 1) 최신 가격·지표로 갱신 (장중 잠정 동기화)
        _sync(region, rec)
        # 2) 잔고·미체결
        try:
            bal = (c.balance_kr if region == "KR" else c.balance_us)()
        except Exception as e:  # noqa: BLE001
            rec["error"] = f"잔고 조회 실패: {e}"
            notify("자동매도 오류", rec["error"])
            return rec
        held = {h["ticker"]: h for h in bal["holdings"]}
        try:
            d0 = (now - timedelta(days=7)).strftime("%Y%m%d")
            trades = (c.trades_kr if region == "KR" else c.trades_us)(d0, now.strftime("%Y%m%d"))
        except Exception as e:  # noqa: BLE001
            rec["error"] = f"거래내역 조회 실패: {e}"
            notify("자동매도 오류", rec["error"])
            return rec
        open_sell = {t["ticker"] for t in trades if t["side"] == "sell" and t["remaining"] > 0 and not t["cancelled"] and not t["rejected"]}
        ordered_today = today_orders(region, local_today)
        done_today = {d["ticker"] for d in ordered_today}
        n_today = len(ordered_today)
        # 3) 판정 (화면과 같은 저장 규칙)
        res = scanner.load_results() or {}
        cur = {s["ticker"]: s for s in res.get("stocks", [])}
        pos = positions.load_positions()
        for e in positions.evaluate_all(positions.load_rule(), bal["holdings"]):
            t = e["ticker"]
            if e["region"] != region or e["action"] not in ("SELL", "HALF"):
                continue
            d = {"ticker": t, "name": e.get("name"), "action": e["action"], "reasons": e["reasons"], "ret": e.get("ret"), "status": "skipped", "why": None}
            rec["decisions"].append(d)
            row = cur.get(t) or {}
            h = held.get(t)
            if not pos.get(t, {}).get("entry_date"):
                d["why"] = "매수일 미입력"; continue
            if not h or (h.get("qty") or 0) <= 0:
                d["why"] = "한투 잔고 없음"; continue
            if rec.get("sync_error"):
                d["why"] = f"장중 동기화 실패 ({rec['sync_error']}) — 오래된 지표로는 주문 안 함"; continue
            if e.get("last_date") != local_today or row.get("date") != local_today:
                d["why"] = f"당일 데이터 아님 ({e.get('last_date')})"; continue
            if row.get("suspect") or abs(row.get("chg") or 0) >= 20:
                d["why"] = f"가격 이상 의심 (전일 대비 {row.get('chg')}%)"; continue
            if t in open_sell:
                d["why"] = "미체결 매도 주문 있음"; continue
            if t in done_today:
                d["why"] = "오늘 이미 자동 주문"; continue
            try:
                px = c.price_kr(t) if region == "KR" else c.price_us(t)[0]
            except Exception as ex:  # noqa: BLE001
                d["why"] = f"현재가 조회 실패: {ex}"; continue
            if not px or abs(px / float(row.get("close") or px) - 1) > 0.03:
                d["why"] = f"가격 불일치 (한투 {px} vs 스캔 {row.get('close')})"; continue
            qty = int(h["qty"]) if pos.get(t, {}).get("qty") is None else min(int(h["qty"]), int(pos[t]["qty"]))
            if e["action"] == "HALF":
                qty = qty // 2
                if qty < 1:
                    d["why"] = "1주라 절반 매도 불가 (전량 신호까지 보유)"; continue
            if qty < 1:
                d["why"] = "수량 0"; continue
            if n_today >= cfg["max_per_day"]:
                d["why"] = f"하루 상한 {cfg['max_per_day']}건 도달"; continue
            limit = px * (1 - cfg["limit_pct"] / 100)
            limit = kr_tick(limit) if region == "KR" else round(limit, 2)
            d.update({"qty": qty, "price": limit, "cur_px": px})
            label = f"{d['name'] or t} {qty}주 @ {limit:,.2f}" if region == "US" else f"{d['name'] or t} {qty}주 @ {limit:,.0f}원"
            if dry:
                d["status"] = "dry"
                notify("자동매도 (드라이런)", f"{'절반' if e['action']=='HALF' else '전량'} 매도 했을 주문: {label}")
                continue
            try:
                o = c.order(region, t, "sell", qty, limit)
                d["status"] = "ordered"; d["odno"] = o.get("odno"); d["msg"] = o.get("msg")
                n_today += 1
                notify("자동매도 주문 전송", f"{label} · 주문번호 {o.get('odno')}")
            except Exception as ex:  # noqa: BLE001
                d["status"] = "error"; d["why"] = str(ex)
                notify("자동매도 주문 실패", f"{label}: {ex}")
        return rec
    except Exception as e:  # noqa: BLE001
        rec["error"] = f"{type(e).__name__}: {e}"
        notify("자동매도 오류", rec["error"])
        return rec
    finally:
        STATE["running"] = False
        STATE["last"][region] = rec
        STATE["error"] = rec.get("error")
        _append(rec)
        _lock.release()


def next_runs() -> dict:
    cfg = config()
    now = datetime.now(KST)
    def nxt(tz, hm):
        n = now.astimezone(tz)
        h, m = map(int, hm.split(":"))
        t = n.replace(hour=h, minute=m, second=0, microsecond=0)
        while t <= n or t.weekday() >= 5:
            t += timedelta(days=1)
        return t.astimezone(KST).strftime("%Y-%m-%d %H:%M")
    return {"KR": nxt(KST, cfg["time_kr"]), "US": nxt(NY, cfg["time_us"])}


def loop() -> None:
    """매 20초 체크. 평일 지정 시각(국내 KST, 미국 뉴욕)에 지역별 1회."""
    last: dict[str, str] = {}
    while True:
        try:
            cfg = config()
            if cfg["enabled"]:
                now = datetime.now(KST)
                ny = now.astimezone(NY)
                for region, t, hm in (("KR", now, cfg["time_kr"]), ("US", ny, cfg["time_us"])):
                    day = t.strftime("%Y-%m-%d")
                    if t.weekday() < 5 and t.strftime("%H:%M") == hm and last.get(region) != day:
                        last[region] = day
                        run(region, wait=600)  # 드라이런 등 다른 실행이 돌고 있으면 최대 10분 기다렸다 실행
        except Exception as e:  # noqa: BLE001
            log.exception("autosell loop: %s", e)
        time.sleep(20)
