"""시장이 매수 환경(지수 RSI 30~40)에 들어온 국면에서 분할매수 방식 비교 (지수 기준, 약 20년).

국면(episode) = 게이트 충족일이 10거래일 안에 이어지는 묶음. 국면 첫날을 기준일(d0)로 두고 각 전략이
자본 1을 어떻게 투입하는지에 따라 d0 기준 +90/180/360 거래일 시점의 총자본 수익률(미투입 현금은 0%)과
투입 자본 대비 수익률, 보유 중 최대 평가손실을 비교한다.

사용: uv run python scripts/dca_gate.py [mkt_lo=30] [mkt_hi=40]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.indicators import rsi  # noqa: E402
from app.longhist import load_long  # noqa: E402

HZ = (90, 180, 360)
WINDOW = 60  # 분할 투입을 허용하는 기간 (d0 이후 거래일)


def episodes(gate: pd.Series, gap: int = 10) -> list[int]:
    """게이트 충족일 인덱스 위치 → 국면 시작 위치 목록."""
    pos = np.flatnonzero(gate.to_numpy())
    starts, last = [], -10**9
    for p in pos:
        if p - last > gap:
            starts.append(p)
        last = p
    return starts


def simulate(close: np.ndarray, r: np.ndarray, gate: np.ndarray, s: int, plan: str) -> dict | None:
    """plan 에 따라 (투입일 위치, 비중) 목록을 만들고 성과를 계산."""
    n = len(close)
    if s + max(HZ) >= n:
        return None
    c0 = close[s]
    legs: list[tuple[int, float]] = []
    if plan == "lump":
        legs = [(s, 1.0)]
    elif plan.startswith("time"):  # time3_5 = 3회, 5일 간격
        k, step = map(int, plan[4:].split("_"))
        legs = [(s + i * step, 1 / k) for i in range(k)]
    elif plan.startswith("price"):  # price3_5 = 3회, 5% 하락마다 추가
        k, drop = map(int, plan[5:].split("_"))
        legs = [(s, 1 / k)]
        lvl = 1
        for i in range(s + 1, min(n, s + WINDOW + 1)):
            if lvl < k and close[i] <= c0 * (1 - drop * lvl / 100):
                legs.append((i, 1 / k))
                lvl += 1
    elif plan == "rsi3":  # 1/3 진입, RSI ≤35 추가, ≤30 추가
        legs = [(s, 1 / 3)]
        done = {35: False, 30: False}
        for i in range(s + 1, min(n, s + WINDOW + 1)):
            for lv in (35, 30):
                if not done[lv] and r[i] <= lv:
                    done[lv] = True
                    legs.append((i, 1 / 3))
    elif plan == "regate3":  # 게이트 충족일마다 5일 이상 간격으로 최대 3회
        legs = [(s, 1 / 3)]
        last = s
        for i in range(s + 1, min(n, s + WINDOW + 1)):
            if len(legs) < 3 and gate[i] and i - last >= 5:
                legs.append((i, 1 / 3))
                last = i
    elif plan == "lump_rsi30":  # 전액이지만 RSI 30 이하 올 때까지 기다림(60일 내 안 오면 60일째 전액)
        t = next((i for i in range(s, min(n, s + WINDOW + 1)) if r[i] <= 30), s + WINDOW)
        legs = [(t, 1.0)]
    shares = sum(w / close[i] for i, w in legs)
    deployed = sum(w for _, w in legs)
    out = {"plan": plan, "deployed": deployed, "legs": len(legs)}
    for h in HZ:
        val = shares * close[s + h] + (1 - deployed)
        out[f"tot{h}"] = (val - 1) * 100
        out[f"inv{h}"] = (shares * close[s + h] / deployed - 1) * 100 if deployed else 0.0
    # 보유 중 최대 평가손실 (투입 자본 대비, d0~d360)
    worst = 0.0
    sh, dep = 0.0, 0.0
    li = dict(legs)
    for i in range(s, s + 360 + 1):
        if i in li:
            sh += li[i] / close[i]
            dep += li[i]
        if dep:
            worst = min(worst, (sh * close[i] / dep - 1) * 100)
    out["mdd"] = worst
    return out


def run(sym: str, lo: float, hi: float) -> pd.DataFrame:
    df = load_long(sym)
    close = df["Close"].dropna()
    r = rsi(close)
    gate = (r >= lo) & (r <= hi)
    rows = []
    for s in episodes(gate):
        for plan in ("lump", "time3_5", "time3_10", "time4_10", "price3_5", "price3_10", "rsi3", "regate3", "lump_rsi30"):
            o = simulate(close.to_numpy(), r.to_numpy(), gate.to_numpy(), s, plan)
            if o:
                o["date"] = close.index[s].strftime("%Y-%m-%d")
                rows.append(o)
    return pd.DataFrame(rows)


NAMES = {"lump": "일시 전액(d0)", "time3_5": "시간 3회·5일", "time3_10": "시간 3회·10일", "time4_10": "시간 4회·10일",
         "price3_5": "가격 3회·-5%마다", "price3_10": "가격 3회·-10%마다", "rsi3": "RSI 3회(40/35/30)", "regate3": "게이트 재충족 3회",
         "lump_rsi30": "RSI30 기다려 전액"}


def main():
    lo = float(sys.argv[1]) if len(sys.argv) > 1 else 30
    hi = float(sys.argv[2]) if len(sys.argv) > 2 else 40
    for sym, label in (("IXIC", "나스닥"), ("KS11", "코스피")):
        R = run(sym, lo, hi)
        eps = R.date.nunique()
        print(f"\n== {label} (지수 RSI {lo:g}~{hi:g}, 국면 {eps}개: {R.date.min()} ~ {R.date.max()}) ==")
        print(f"{'전략':<18}{'투입':>5}{'회':>3} | " + " | ".join(f"{h:>3}일 총자본 평균/중앙/승률 · 투입대비" for h in HZ) + " | 최대평가손실")
        for plan, g in R.groupby("plan", sort=False):
            cells = []
            for h in HZ:
                t, v = g[f"tot{h}"], g[f"inv{h}"]
                cells.append(f"{t.mean():+6.1f}/{t.median():+6.1f}/{(t > 0).mean() * 100:3.0f}% · {v.mean():+6.1f}")
            print(f"{NAMES[plan]:<18}{g.deployed.mean() * 100:4.0f}%{g.legs.mean():3.1f} | " + " | ".join(cells) + f" | {g.mdd.mean():+6.1f}")
        # 일시 매수 대비 분할이 이긴 국면 비율
        base = R[R.plan == "lump"].set_index("date")
        print("  일시매수 대비 180일 총자본이 더 좋았던 국면 비율:", end=" ")
        for plan in ("time3_10", "price3_5", "rsi3", "regate3"):
            g = R[R.plan == plan].set_index("date")
            print(f"{NAMES[plan]} {(g.tot180 > base.tot180.reindex(g.index)).mean() * 100:.0f}%", end=" · ")
        print()
        print("  국면 목록:", ", ".join(f"{d}({v:+.0f}%)" for d, v in base.tot180.items()))


if __name__ == "__main__":
    main()
