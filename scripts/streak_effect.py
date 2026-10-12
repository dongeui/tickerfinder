"""체류(종목 조건 연속 충족 일수)가 길어질수록 이후 성과가 달라지는지 (약 20년).

종목 조건(스토캐스틱 + 밴드)을 충족한 모든 날에 대해 "오늘이 체류 k일째"를 구하고,
시장 환경(지수 RSI ≤ 40)까지 맞는 날만 골라 체류 구간별로 다음날 시가 매수 → 20/60/90일 수익률을 비교한다.
또 각 체류 구간이 어떻게 끝났는지(위로 이탈 = 반등, 아래로 이탈 = 밴드 붕괴)도 센다.

사용: uv run python scripts/streak_effect.py
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
from app.longhist import load_long  # noqa: E402
from app.universe import DATA_DIR  # noqa: E402

BUCKETS = [(1, 1), (2, 5), (6, 10), (11, 20), (21, 40), (41, 80), (81, 10**6)]
HZ = (20, 60, 90)


def bucket(k: int) -> str:
    for lo, hi in BUCKETS:
        if lo <= k <= hi:
            return f"{lo}일" if lo == hi else (f"{lo}~{hi}일" if hi < 10**6 else f"{lo}일+")
    return "?"


def main():
    th = Thresholds(mkt_lo=0, mkt_hi=40, stoch_tf_kr="d", stoch_tf_us="w")
    gate = _gate_series(th, use_kr_index=True)
    uni = json.loads((DATA_DIR / "universe.json").read_text())
    rows, ends = [], []
    for i, u in enumerate(uni):
        df = load_long(u["ticker"])
        if df is None or len(df) < 260:
            continue
        df = df.dropna(subset=["Close"])
        ser = compute_series(df)
        c = ser["close"].to_numpy(float)
        o = df["Open"].reindex(ser.index).to_numpy(float)
        sk = ser["stoch_k_w"] if th.stoch_tf(u["region"]) == "w" else ser["stoch_k"]
        hit = ((sk <= th.stoch_max) & (ser["band_pos"].abs() <= th.band_tol) & ~ser["suspect"].astype(bool)).fillna(False).to_numpy()
        g = gate[u["region"]].reindex(ser.index, method="ffill").fillna(False).astype(bool).to_numpy()
        n = len(c)
        k = 0
        start = None
        for t in range(n):
            if hit[t]:
                k += 1
                if k == 1:
                    start = t
                if g[t] and t + 1 < n and not np.isnan(o[t + 1]):
                    ep = o[t + 1]
                    r = {"region": u["region"], "k": k, "b": bucket(k)}
                    for h in HZ:
                        r[f"f{h}"] = (c[t + 1 + h] / ep - 1) * 100 if t + 1 + h < n else np.nan
                    rows.append(r)
            else:
                if k > 0 and start is not None:
                    # 체류 종료: 마지막 충족일 대비 이탈일 종가 등락, 체류 시작가 대비 이탈일 종가
                    ends.append({"region": u["region"], "k": k, "b": bucket(k), "exit_chg": (c[t] / c[t - 1] - 1) * 100,
                                 "from_start": (c[t] / c[start] - 1) * 100,
                                 "f60": (c[t + 60] / c[t] - 1) * 100 if t + 60 < n else np.nan})
                k = 0
                start = None
        if i % 300 == 0:
            print(f"... {i}/{len(uni)}", file=sys.stderr)
    R, E = pd.DataFrame(rows), pd.DataFrame(ends)
    order = [bucket(lo) for lo, _ in BUCKETS]
    for reg in ("US", "KR"):
        print(f"\n== {reg}: 시장 환경 충족일에 '체류 k일째' 종목을 샀을 때 ==")
        print(f"{'체류':<10}{'표본':>7} | " + " | ".join(f"{h:>3}일 승률 / 평균 / 중앙" for h in HZ))
        for b in order:
            g = R[(R.region == reg) & (R.b == b)]
            if not len(g):
                continue
            cells = [f"{(g[f'f{h}'].dropna() > 0).mean() * 100:4.0f}% / {g[f'f{h}'].mean():+5.1f} / {g[f'f{h}'].median():+5.1f}" for h in HZ]
            print(f"{b:<10}{len(g):>7} | " + " | ".join(cells))
        print(f"\n   {reg}: 체류가 어떻게 끝났나 (시장 환경 무관, 모든 체류 구간)")
        print(f"{'체류':<10}{'건수':>7}{'위로 이탈(반등)':>14}{'아래로 이탈':>10}{'시작가 대비 이탈일':>14}{'이탈 후 60일':>12}")
        for b in order:
            g = E[(E.region == reg) & (E.b == b)]
            if not len(g):
                continue
            up = (g.exit_chg > 0).mean() * 100
            print(f"{b:<10}{len(g):>7}{up:13.0f}%{100 - up:9.0f}%{g.from_start.median():+13.1f}%{g.f60.median():+11.1f}%")


if __name__ == "__main__":
    main()
