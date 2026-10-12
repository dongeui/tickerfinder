"""매수 신호(게이트 + 종목 조건, 국내 일봉/미국 주봉 스토캐스틱) 이후 매도 규칙 비교 (약 20년).

진입: 신호 다음 거래일 시가. 청산: 규칙이 발동한 날의 다음 거래일 시가 (보유기간 만료는 그날 종가).
사용: uv run python scripts/exit_rules.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.analysis import Thresholds  # noqa: E402
from app.backfill import compute_series  # noqa: E402
from app.backtest import _gate_series  # noqa: E402
from app.indicators import rsi  # noqa: E402
from app.longhist import load_long  # noqa: E402
from app.universe import DATA_DIR  # noqa: E402

MAXH = 360


def first_true(mask: np.ndarray) -> int | None:
    idx = np.flatnonzero(mask)
    return int(idx[0]) if len(idx) else None


def simulate(ev: dict, rule: str) -> tuple[float, int, str] | None:
    """반환: (수익률 %, 보유일, 청산 사유)."""
    c, o, ep = ev["close"], ev["open"], ev["entry"]
    n = len(c)
    if n < 2:
        return None
    ret_c = c / ep - 1  # 각 일자 종가 기준 수익률 (0 = 매수일)
    peak = np.maximum.accumulate(c)
    name, *args = rule.split("_")
    maxh = MAXH
    trig = np.zeros(n, dtype=bool)
    if name == "hold":
        maxh = int(args[0])
    elif name == "tpsl":  # tpsl_10_10_90 : +10% 목표, -10% 손절, 최대 90일
        tp, sl, maxh = float(args[0]), float(args[1]), int(args[2])
        trig = (ret_c >= tp / 100) | (ret_c <= -sl / 100)
    elif name == "sellsig":  # sellsig_75_180 : 일봉 RSI≥75 & 주봉 RSI 65~85
        lv, maxh = float(args[0]), int(args[1])
        trig = (ev["rsi_d"] >= lv) & (ev["rsi_w"] >= 65) & (ev["rsi_w"] <= 85)
    elif name == "rsi":  # rsi_70_180
        lv, maxh = float(args[0]), int(args[1])
        trig = ev["rsi_d"] >= lv
    elif name == "stoch":  # stoch_80_180 (일봉 %K)
        lv, maxh = float(args[0]), int(args[1])
        trig = ev["stoch_k"] >= lv
    elif name == "trail":  # trail_10_360 : 고점 대비 -10%
        dd, maxh = float(args[0]), int(args[1])
        trig = c <= peak * (1 - dd / 100)
    elif name == "trailtp":  # trailtp_10_5_360 : +5% 이상 수익 난 뒤에만 고점 대비 -10% 트레일
        dd, arm, maxh = float(args[0]), float(args[1]), int(args[2])
        armed = np.maximum.accumulate(ret_c) >= arm / 100
        trig = armed & (c <= peak * (1 - dd / 100))
    elif name == "mkt":  # mkt_60_180 : 자국 지수 RSI ≥ 60
        lv, maxh = float(args[0]), int(args[1])
        trig = ev["mkt_rsi"] >= lv
    elif name == "band":  # band_10_180 : 150~200일선 밴드 하단 -10% 이탈(손절) 아니면 보유
        dd, maxh = float(args[0]), int(args[1])
        trig = ev["band_pos"] <= -dd
    elif name == "combo":  # combo_75_15_360 : 매도조건(RSI≥lv) 또는 고점 대비 -dd% 트레일
        lv, dd, maxh = float(args[0]), float(args[1]), int(args[2])
        trig = ((ev["rsi_d"] >= lv) & (ev["rsi_w"] >= 65) & (ev["rsi_w"] <= 85)) | (c <= peak * (1 - dd / 100))
    elif name == "sigtrail":  # sigtrail_75_15_10_360 : 매도조건(RSI≥lv) 또는 (+arm% 이상 수익 후 고점 대비 -dd%)
        lv, dd, arm, maxh = float(args[0]), float(args[1]), float(args[2]), int(args[3])
        armed = np.maximum.accumulate(ret_c) >= arm / 100
        trig = ((ev["rsi_d"] >= lv) & (ev["rsi_w"] >= 65) & (ev["rsi_w"] <= 85)) | (armed & (c <= peak * (1 - dd / 100)))
    elif name == "half":  # half_10_75_15_360 : +tp%에 절반 매도, 나머지는 매도조건(RSI≥lv) 또는 고점 대비 -dd% 트레일
        tp, lv, dd, maxh = float(args[0]), float(args[1]), float(args[2]), int(args[3])
        t1 = first_true((ret_c >= tp / 100) & (np.arange(n) > 0))
        armed = np.maximum.accumulate(ret_c) >= tp / 100  # 절반 판 뒤(= +tp% 도달 뒤)부터만 트레일
        trig = ((ev["rsi_d"] >= lv) & (ev["rsi_w"] >= 65) & (ev["rsi_w"] <= 85)) | (armed & (c <= peak * (1 - dd / 100)))
        trig[0] = False
        j = first_true(trig[: maxh + 1])
        if j is not None and j + 1 < n:
            r2, d2 = (o[j + 1] / ep - 1) * 100, j + 1
        else:
            k = min(maxh, n - 1)
            r2, d2 = ret_c[k] * 100, k
        if t1 is not None and t1 + 1 < n and t1 < d2:
            r1 = (o[t1 + 1] / ep - 1) * 100
            return (r1 + r2) / 2, d2, "rule"
        return r2, d2, "rule" if j is not None else "time"
    elif name == "combosl":  # combosl_75_10_360 : 매도조건 또는 -10% 손절
        lv, sl, maxh = float(args[0]), float(args[1]), int(args[2])
        trig = ((ev["rsi_d"] >= lv) & (ev["rsi_w"] >= 65) & (ev["rsi_w"] <= 85)) | (ret_c <= -sl / 100)
    trig[0] = False  # 매수 당일은 청산 안 함
    j = first_true(trig[: maxh + 1])
    if j is not None and j + 1 < n:
        return (o[j + 1] / ep - 1) * 100, j + 1, "rule"
    k = min(maxh, n - 1)
    return ret_c[k] * 100, k, "time"


RULES = [
    "hold_20", "hold_60", "hold_90", "hold_180", "hold_360",
    "tpsl_10_10_90", "tpsl_15_10_180", "tpsl_20_15_360",
    "sellsig_75_180", "sellsig_75_360", "sellsig_80_360", "rsi_70_180", "stoch_80_180",
    "trail_10_360", "trail_15_360", "trailtp_10_5_360", "trailtp_15_10_360",
    "mkt_60_180", "mkt_65_360", "band_10_180",
    "combo_75_15_360", "combosl_75_10_360", "combosl_80_15_360",
    "sigtrail_75_15_10_360", "sigtrail_75_10_5_360", "sigtrail_80_15_10_360", "sigtrail_75_15_10_180",
]
NAMES = {
    "hold_20": "20일 보유", "hold_60": "60일 보유", "hold_90": "90일 보유", "hold_180": "180일 보유", "hold_360": "360일 보유",
    "tpsl_10_10_90": "+10%/-10%, 90일", "tpsl_15_10_180": "+15%/-10%, 180일", "tpsl_20_15_360": "+20%/-15%, 360일",
    "sellsig_75_180": "매도조건(RSI75), 180일", "sellsig_75_360": "매도조건(RSI75), 360일", "sellsig_80_360": "매도조건(RSI80), 360일",
    "rsi_70_180": "일RSI≥70, 180일", "stoch_80_180": "스토K≥80, 180일",
    "trail_10_360": "고점-10% 트레일, 360일", "trail_15_360": "고점-15% 트레일, 360일",
    "trailtp_10_5_360": "+5% 후 고점-10% 트레일", "trailtp_15_10_360": "+10% 후 고점-15% 트레일",
    "mkt_60_180": "지수RSI≥60, 180일", "mkt_65_360": "지수RSI≥65, 360일", "band_10_180": "밴드-10% 손절, 180일",
    "sigtrail_75_15_10_360": "매도조건75 or +10%후 고점-15%", "sigtrail_75_10_5_360": "매도조건75 or +5%후 고점-10%",
    "sigtrail_80_15_10_360": "매도조건80 or +10%후 고점-15%", "sigtrail_75_15_10_180": "매도조건75 or +10%후 고점-15%, 180일",
    "half_10_75_15_360": "+10%에 절반, 나머지 매도조건 or 고점-15%", "half_10_75_10_360": "+10%에 절반, 나머지 매도조건 or 고점-10%",
    "half_15_75_15_360": "+15%에 절반, 나머지 매도조건 or 고점-15%",
    "combo_75_15_360": "매도조건 or 고점-15%", "combosl_75_10_360": "매도조건 or -10%손절", "combosl_80_15_360": "매도조건80 or -15%손절",
}


def main():
    global RULES
    if len(sys.argv) > 1:
        RULES = sys.argv[1].split(",")
    th = Thresholds(stoch_tf_kr="d", stoch_tf_us="w")
    gate = _gate_series(th, use_kr_index=True)
    idx_rsi = {"US": rsi(load_long("IXIC")["Close"].dropna()), "KR": rsi(load_long("KS11")["Close"].dropna())}
    uni = json.loads((DATA_DIR / "universe.json").read_text())
    events: list[dict] = []
    for i, u in enumerate(uni):
        df = load_long(u["ticker"])
        if df is None or len(df) < 260:
            continue
        df = df.dropna(subset=["Close"])
        ser = compute_series(df)
        c = ser["close"]
        sk = ser["stoch_k_w"] if th.stoch_tf(u["region"]) == "w" else ser["stoch_k"]
        g = gate[u["region"]].reindex(c.index, method="ffill").fillna(False).astype(bool)
        hit = ((sk <= th.stoch_max) & (ser["band_pos"].abs() <= th.band_tol) & g & ~ser["suspect"].astype(bool)).fillna(False)
        entry = hit & ~hit.shift(1, fill_value=False)
        if not entry.any():
            continue
        o = df["Open"].reindex(c.index).to_numpy(float)
        arrs = {k: ser[k].to_numpy(float) for k in ("close", "rsi_d", "rsi_w", "stoch_k", "band_pos")}
        mk = idx_rsi[u["region"]].reindex(c.index, method="ffill").to_numpy(float)
        for p in np.flatnonzero(entry.to_numpy()):
            s = p + 1  # 매수일 (다음 거래일)
            if s + 2 >= len(c) or np.isnan(o[s]):
                continue
            e = s + MAXH + 2
            ev = {"region": u["region"], "ticker": u["ticker"], "date": c.index[p].strftime("%Y-%m-%d"), "entry": o[s], "open": o[s:e]}
            for k, a in arrs.items():
                ev[k] = a[s:e]
            ev["mkt_rsi"] = mk[s:e]
            events.append(ev)
        if i % 200 == 0:
            print(f"... {i}/{len(uni)} events={len(events)}", file=sys.stderr)
    print(f"이벤트 {len(events)}건 (US {sum(e['region']=='US' for e in events)}, KR {sum(e['region']=='KR' for e in events)})")

    rows = []
    for ev in events:
        for rule in RULES:
            r = simulate(ev, rule)
            if r:
                rows.append({"region": ev["region"], "rule": rule, "ret": r[0], "days": r[1], "why": r[2]})
    R = pd.DataFrame(rows)
    R.to_csv(DATA_DIR / "exit_rules.csv", index=False)
    for reg in ("US", "KR"):
        print(f"\n== {reg} ==")
        print(f"{'규칙':<24}{'평균':>7}{'중앙':>7}{'승률':>6}{'보유일':>7}{'연환산':>8}{'규칙청산':>8}{'손익비':>7}{'하위10%':>8}")
        for rule in RULES:
            g = R[(R.region == reg) & (R.rule == rule)]
            if not len(g):
                continue
            win = g.ret > 0
            pf = g.ret[win].sum() / max(1e-9, -g.ret[~win].sum())
            ann = g.ret.mean() / g.days.mean() * 252
            print(f"{NAMES.get(rule, rule):<24}{g.ret.mean():+7.2f}{g.ret.median():+7.2f}{win.mean()*100:5.0f}%{g.days.mean():7.0f}{ann:+8.1f}{(g.why=='rule').mean()*100:7.0f}%{pf:7.2f}{g.ret.quantile(.1):+8.1f}")


if __name__ == "__main__":
    main()
