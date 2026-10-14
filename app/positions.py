"""보유 포지션 기록과 매도 규칙 판정.

포지션(data/positions.json): {ticker: {"entry_date", "entry_px", "qty", "note", "fills": {odno: 반영수량}}}
- 이 앱에서 보낸 주문(data/orders.jsonl)은 체결 대조(fills.py)로 확인된 체결수량만큼만 포지션에 반영된다 (접수만으론 반영 안 함).
- 한투 잔고에는 평단만 있고 매수일이 없으므로 매수일은 주문 기록 또는 직접 입력.

매도 규칙(scripts/exit_rules.py 검증, 2026-10-12):
- 과열 조건: 일봉 RSI ≥ 75 and 주봉 RSI 65~85 → 매도
- 트레일: 보유 중 최고 수익이 arm% 를 넘은 뒤, 종가가 고점 대비 trail% 하락 → 매도
- 최대 보유 360거래일 → 매도. 고정 손절 없음.
미국 arm 10 / trail 15, 국내 arm 5 / trail 10.
"""
from __future__ import annotations

import json
import threading
from dataclasses import dataclass

import pandas as pd

from .scanner import atomic_write_text, load_cache, load_results
from .universe import DATA_DIR

POS_FILE = DATA_DIR / "positions.json"
ORDER_LOG = DATA_DIR / "orders.jsonl"
MAX_HOLD = 360


@dataclass(frozen=True)
class ExitRule:
    rsi_d_min: float = 75
    rsi_w_lo: float = 65
    rsi_w_hi: float = 85
    arm_us: float = 10
    trail_us: float = 15
    arm_kr: float = 5
    trail_kr: float = 10
    max_hold: int = MAX_HOLD


RULE_FILE = DATA_DIR / "exit_rule.json"


def load_rule() -> ExitRule:
    """저장된 매도 규칙(화면에서 바꾼 값). 없거나 깨졌으면 기본값. 화면 판정과 자동매도가 같은 규칙을 쓴다."""
    try:
        d = json.loads(RULE_FILE.read_text()) if RULE_FILE.exists() else {}
        return ExitRule(**{k: type(getattr(ExitRule, k))(v) for k, v in d.items() if hasattr(ExitRule, k)})
    except Exception:  # noqa: BLE001
        return ExitRule()


def save_rule(rule: ExitRule) -> None:
    atomic_write_text(RULE_FILE, json.dumps(rule.__dict__, ensure_ascii=False, indent=1))


_plock = threading.RLock()  # load(체결 반영 포함)·save 가 여러 스레드(화면·자동매도·체결대조)에서 겹치지 않도록


def load_positions() -> dict[str, dict]:
    with _plock:
        return _load_positions()


def _load_positions() -> dict[str, dict]:
    """포지션 파일을 읽고, 앱 주문 중 체결 대조(fills.reconcile)로 확인된 체결수량을 반영한다.

    주문 기록(orders.jsonl)의 filled_qty 중 아직 반영 안 된 만큼만 더하거나 뺀다 (pos[t]["fills"][odno] = 반영한 수량).
    매수 체결 → 포지션 등록/수량·평단 갱신. 매도 체결 → 수량 차감, 0이면 삭제, 남으면 절반 매도 완료로 기록.
    """
    pos = json.loads(POS_FILE.read_text()) if POS_FILE.exists() else {}
    if not ORDER_LOG.exists():
        return pos
    changed = False
    orders = []
    for line in ORDER_LOG.read_text().splitlines():
        try:
            if line.strip():
                orders.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    for o in orders:
        odno, filled = o.get("odno"), float(o.get("filled_qty") or 0)
        if not odno or filled <= 0:
            continue
        t = o["ticker"]
        applied = float(pos.get(t, {}).get("fills", {}).get(odno, 0) or pos.get("_applied", {}).get(odno, 0))
        delta = filled - applied
        if delta <= 0:
            continue
        px = float(o.get("filled_px") or o.get("price") or 0)
        if o["side"] == "buy":
            p = pos.setdefault(t, {"entry_date": o["at"][:10], "entry_px": px, "qty": 0, "note": f"앱 주문 체결 ({o.get('mode')})"})
            oldq = float(p.get("qty") or 0)
            if oldq:
                p["entry_px"] = round((float(p.get("entry_px") or px) * oldq + px * delta) / (oldq + delta), 4)
            elif not p.get("entry_px"):  # 수량 없이 평단만 직접 입력해 둔 포지션은 평단을 덮어쓰지 않는다
                p["entry_px"] = px
            p["qty"] = int(oldq + delta)
            p.setdefault("entry_date", o["at"][:10])
        else:
            if t not in pos:
                # 추적 중이 아닌 종목의 매도 체결: 반영할 것은 없지만 '처리됨'으로 남겨 나중에 포지션을 추가했을 때 다시 차감되지 않게 한다
                pos.setdefault("_applied", {})[odno] = int(filled)
                changed = True
                continue
            p = pos[t]
            if p.get("qty") is not None:
                p["qty"] = max(0, int(float(p["qty"]) - delta))
                if p["qty"] == 0:
                    pos.pop(t)
                    changed = True
                    continue
            p["half_done"] = True
            p["half_date"] = o["at"][:10]
        p.setdefault("fills", {})[odno] = int(applied + delta)
        changed = True
    if changed:
        save_positions(pos)
    return pos


def save_positions(pos: dict[str, dict]) -> None:
    with _plock:
        atomic_write_text(POS_FILE, json.dumps(pos, ensure_ascii=False, indent=1))


def evaluate(ticker: str, entry_date: str | None, entry_px: float | None, rule: ExitRule, cur: dict | None = None,
             half_done: bool = False) -> dict:
    """한 포지션의 매도 규칙 상태. cur: results.json 의 종목 행(최신 지표).

    action: SELL(전량 또는 잔량 매도) | HALF(+arm% 도달, 절반 매도 시점) | HOLD
    """
    region = "KR" if ticker.isdigit() else "US"
    arm, trail = (rule.arm_kr, rule.trail_kr) if region == "KR" else (rule.arm_us, rule.trail_us)
    out = {"ticker": ticker, "region": region, "entry_date": entry_date, "entry_px": entry_px, "action": None, "reasons": [],
           "ret": None, "peak_ret": None, "from_peak": None, "days": None, "armed": False, "arm": arm, "trail": trail}
    df = load_cache(ticker)
    if df is None or not len(df):
        out["reasons"].append("가격 데이터 없음")
        return out
    c = df["Close"].dropna()
    if cur and cur.get("close") and cur.get("date") and cur["date"] > c.index[-1].strftime("%Y-%m-%d"):
        # 장중 동기화로 results.json 에 더 새로운(잠정) 종가가 있으면 그것을 현재가로 쓴다
        c = pd.concat([c, pd.Series([float(cur["close"])], index=[pd.Timestamp(cur["date"])])])
    last = float(c.iloc[-1])
    out["last"] = last
    out["last_date"] = c.index[-1].strftime("%Y-%m-%d")
    out["partial"] = bool(cur and cur.get("partial"))
    if entry_date:
        seg = c[c.index >= pd.Timestamp(entry_date)]
        if len(seg):
            out["days"] = int(len(seg))
            peak = float(seg.max())
            if entry_px:
                out["peak_ret"] = round((peak / entry_px - 1) * 100, 2)
            out["from_peak"] = round((last / peak - 1) * 100, 2)
    if entry_px:
        out["ret"] = round((last / entry_px - 1) * 100, 2)
    sig = []
    if cur and cur.get("rsi_d") is not None and cur.get("rsi_w") is not None:
        out["rsi_d"], out["rsi_w"] = cur["rsi_d"], cur["rsi_w"]
        if cur["rsi_d"] >= rule.rsi_d_min and rule.rsi_w_lo <= cur["rsi_w"] <= rule.rsi_w_hi:
            sig.append(f"과열 조건 (일RSI {cur['rsi_d']:.0f} · 주RSI {cur['rsi_w']:.0f})")
    if out["peak_ret"] is not None and out["peak_ret"] >= arm:
        out["armed"] = True
        if out["from_peak"] is not None and out["from_peak"] <= -trail:
            sig.append(f"트레일 발동 (고점 +{out['peak_ret']:.0f}% 대비 {out['from_peak']:.0f}%)")
    if out["days"] is not None and out["days"] >= rule.max_hold:
        sig.append(f"보유 {out['days']}일 ≥ {rule.max_hold}일")
    out["reasons"] = sig
    out["half_done"] = half_done
    if sig:
        out["action"] = "SELL"
    elif out["armed"] and not half_done:
        out["action"] = "HALF"
        out["reasons"] = [f"최고수익 +{out['peak_ret']:.1f}% ≥ +{arm:g}% — 절반 익절 시점 (나머지는 과열조건 / 고점 -{trail:g}% 트레일)"]
    else:
        out["action"] = "HOLD"
    if not sig and not entry_date:
        out["reasons"].append("매수일 미입력 — 트레일·기한 판정 불가")
    return out


def evaluate_all(rule: ExitRule, holdings: list[dict] | None = None) -> list[dict]:
    """포지션 파일 + (있으면) 한투 보유 종목을 합쳐 판정. 한투 보유 종목은 평단을 매수가로 쓴다."""
    pos = load_positions()
    res = load_results() or {}
    cur = {s["ticker"]: s for s in res.get("stocks", [])}
    merged: dict[str, dict] = {t: dict(v) for t, v in pos.items()}
    for h in holdings or []:
        m = merged.setdefault(h["ticker"], {})
        m.setdefault("entry_px", h.get("avg"))
        m["qty"] = h.get("qty", m.get("qty"))
        m["name"] = h.get("name")
        m["broker"] = True
    out = []
    for t, p in merged.items():
        e = evaluate(t, p.get("entry_date"), p.get("entry_px"), rule, cur.get(t), bool(p.get("half_done")))
        e.update({"qty": p.get("qty"), "note": p.get("note"), "name": p.get("name") or cur.get(t, {}).get("name", t), "broker": p.get("broker", False),
                  "half_date": p.get("half_date")})
        out.append(e)
    order = {"SELL": 0, "HALF": 1, "HOLD": 2, None: 3}
    out.sort(key=lambda e: (order.get(e["action"], 2), e["ticker"]))
    return out
