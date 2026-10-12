"""주문 체결 대조.

이 앱이 보낸 주문(data/orders.jsonl)은 한투의 "접수" 응답만 받는다. 여기서 한투 체결내역(trades_kr / trades_us)과 주문번호로 대조해
체결수량·체결가·상태(fill_status)를 주문 기록에 적고, 체결된 수량만 포지션(positions.json)에 반영한다.
지정가 주문은 당일만 유효하므로 다음 거래일에도 남은 미체결은 expired 로 본다 (거래소가 마감 시 자동 취소).

fill_status: accepted(대조 전) | open(미체결) | partial(부분체결) | filled | expired | cancelled | rejected
"""
from __future__ import annotations

import json
import logging
import threading
import time
from datetime import datetime

from . import kis
from .notify import notify
from .scanner import KST, NY, atomic_write_text
from .universe import DATA_DIR

log = logging.getLogger("tickerfinder.fills")
ORDER_LOG = DATA_DIR / "orders.jsonl"
FINAL = {"filled", "expired", "cancelled", "rejected"}
STATE: dict = {"last_at": None, "last_error": None, "last_changed": []}
_lock = threading.Lock()


def load_orders() -> list[dict]:
    if not ORDER_LOG.exists():
        return []
    rows = []
    for line in ORDER_LOG.read_text().splitlines():
        try:
            if line.strip():
                rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def save_orders(rows: list[dict]) -> None:
    atomic_write_text(ORDER_LOG, "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def trade_date_of(o: dict) -> str:
    """주문의 지역 기준 거래일."""
    at = datetime.fromisoformat(o["at"])
    at = at.replace(tzinfo=KST) if at.tzinfo is None else at
    return (at if o["region"] == "KR" else at.astimezone(NY)).strftime("%Y-%m-%d")


def pending() -> list[dict]:
    return [o for o in load_orders() if o.get("odno") and o.get("fill_status", "accepted") not in FINAL]


def reconcile() -> dict:
    """미확정 주문을 한투 체결내역과 대조. 변경된 주문 목록과 오류를 돌려준다."""
    if not _lock.acquire(blocking=False):
        return {"checked": 0, "changed": [], "errors": ["이미 대조 중"]}
    try:
        orders = load_orders()
        todo = [o for o in orders if o.get("odno") and o.get("fill_status", "accepted") not in FINAL]
        out = {"checked": len(todo), "changed": [], "errors": []}
        if not todo:
            return out
        c = kis.client()
        if not c.configured:
            out["errors"].append("KIS 설정 없음")
            return out
        now = datetime.now(KST)
        by_region: dict[str, dict[str, dict]] = {}
        for region in {o["region"] for o in todo}:
            d0 = min(o["at"][:10] for o in todo if o["region"] == region).replace("-", "")
            try:
                rows = (c.trades_kr if region == "KR" else c.trades_us)(d0, now.strftime("%Y%m%d"))
                by_region[region] = {r["odno"]: r for r in rows}
            except Exception as e:  # noqa: BLE001
                out["errors"].append(f"{region} 체결내역 조회 실패: {e}")
        today = {"KR": now.strftime("%Y-%m-%d"), "US": now.astimezone(NY).strftime("%Y-%m-%d")}
        for o in todo:
            if o["region"] not in by_region:
                continue
            r = by_region[o["region"]].get(o["odno"])
            old = o.get("fill_status", "accepted")
            stale = trade_date_of(o) < today[o["region"]]
            if r is None:
                if not stale:
                    continue  # 아직 내역에 안 올라왔을 수 있음
                new = {"fill_status": "expired", "filled_qty": o.get("filled_qty", 0), "filled_px": o.get("filled_px"), "remaining": o["qty"]}
            else:
                filled, remaining = r["filled_qty"], r["remaining"]
                if r["rejected"]:
                    st = "rejected"
                elif r["cancelled"]:
                    st = "cancelled"
                elif filled >= o["qty"] > 0:
                    st = "filled"
                elif remaining > 0 and stale:
                    st = "expired"
                elif filled > 0:
                    st = "partial"
                else:
                    st = "open"
                new = {"fill_status": st, "filled_qty": filled, "filled_px": r["filled_px"] or None, "remaining": remaining}
                if r.get("reject_reason"):
                    new["reject_reason"] = r["reject_reason"]
            new["checked_at"] = now.isoformat(timespec="seconds")
            changed = any(o.get(k) != v for k, v in new.items() if k != "checked_at")
            o.update(new)
            if changed:
                out["changed"].append({"ticker": o["ticker"], "side": o["side"], "odno": o["odno"], "from": old, **{k: new[k] for k in ("fill_status", "filled_qty", "filled_px")}})
        save_orders(orders)
        from . import positions  # noqa: PLC0415  (순환 import 회피)
        positions.load_positions()  # 체결 반영
        for ch in out["changed"]:
            if ch["fill_status"] in FINAL or ch["fill_status"] == "partial":
                side = "매수" if ch["side"] == "buy" else "매도"
                label = {"filled": "체결", "partial": "부분체결", "expired": "미체결 만료", "cancelled": "취소", "rejected": "거부"}[ch["fill_status"]]
                notify(f"{side} {label}", f"{ch['ticker']} {ch['filled_qty']:g}주" + (f" @ {ch['filled_px']:,.2f}" if ch["filled_px"] else "") + f" · 주문 {ch['odno']}")
        STATE.update(last_at=now.isoformat(timespec="seconds"), last_error=out["errors"][0] if out["errors"] else None, last_changed=out["changed"])
        return out
    except Exception as e:  # noqa: BLE001
        log.exception("reconcile")
        STATE["last_error"] = f"{type(e).__name__}: {e}"
        return {"checked": 0, "changed": [], "errors": [STATE["last_error"]]}
    finally:
        _lock.release()


def reconcile_later(delay: float = 20) -> None:
    """주문 직후 호출: delay 초 뒤 한 번, 그 뒤 loop 가 이어서 본다."""
    threading.Timer(delay, reconcile).start()


def loop() -> None:
    """미확정 주문이 있으면 2분마다, 없으면 30분마다(만료 처리) 대조."""
    last = 0.0
    while True:
        try:
            interval = 120 if pending() else 1800
            if time.time() - last >= interval:
                last = time.time()
                reconcile()
        except Exception as e:  # noqa: BLE001
            log.exception("fills loop: %s", e)
        time.sleep(30)
