"""게이트 진입 종목의 반등을 나스닥/코스피 같은 기간 수익률과 비교 (20/60/90거래일).

선행: uv run python scripts/rebound_factors.py  (data/rebound_events.csv 생성)
사용: uv run python scripts/rebound_vs_index.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.scanner import load_cache  # noqa: E402
from app.universe import DATA_DIR  # noqa: E402

HORIZONS = (20, 60, 90)


def fwd(series: pd.Series, t: pd.Timestamp, h: int) -> float:
    s = series.loc[t:]
    return (s.iloc[h] / s.iloc[0] - 1) * 100 if len(s) > h else np.nan


def main():
    X = pd.read_csv(DATA_DIR / "rebound_events.csv")
    idx = {"IXIC": load_cache("^IXIC")["Close"], "KS11": load_cache("^KS11")["Close"]}
    cache: dict[str, pd.DataFrame | None] = {}
    rows = []
    for e in X.itertuples():
        if e.ticker not in cache:
            cache[e.ticker] = load_cache(e.ticker)
        df = cache[e.ticker]
        if df is None:
            continue
        t = pd.Timestamp(e.date)
        c = df["Close"]
        if t not in c.index:
            continue
        r = {"date": e.date, "episode": e.episode, "region": e.region, "ticker": e.ticker,
             "ruleA": bool(e.ma_slope200 >= 2 and e.dist_ma50 <= -9 and e.rsi_d <= 40)}
        for h in HORIZONS:
            r[f"s{h}"] = fwd(c, t, h)
            r[f"nq{h}"] = fwd(idx["IXIC"], t, h)
            r[f"ks{h}"] = fwd(idx["KS11"], t, h)
            r[f"ex{h}"] = r[f"s{h}"] - (r[f"ks{h}"] if e.region == "KR" else r[f"nq{h}"])  # 자국 지수 대비
            r[f"exnq{h}"] = r[f"s{h}"] - r[f"nq{h}"]  # 나스닥 대비
        rows.append(r)
    R = pd.DataFrame(rows)

    def line(g: pd.DataFrame, label: str):
        out = [label.ljust(26)]
        for h in HORIZONS:
            v = g[f"s{h}"].dropna()
            if len(v) == 0:
                out.append(f"{h}일: -")
                continue
            out.append(f"{h}일 종목{v.mean():+6.1f}% 나스닥{g.loc[v.index, f'nq{h}'].mean():+5.1f}% "
                       f"초과(나스닥){g.loc[v.index, f'exnq{h}'].mean():+6.1f}% 초과(자국){g.loc[v.index, f'ex{h}'].mean():+6.1f}% n={len(v)}")
        print(" | ".join(out))

    print("== 전체 / 지역별 ==")
    line(R, "전체"); line(R[R.region == "KR"], "국내"); line(R[R.region == "US"], "미국")
    print("\n== 국면별 ==")
    for ep, g in R.groupby("episode"):
        line(g, f"ep{ep} {g.date.min()}~{g.date.max()}")
    print("\n== 규칙A(200일선 상승·50일선 -9%·RSI≤40) vs 나머지 ==")
    line(R[R.ruleA], "규칙A"); line(R[~R.ruleA], "나머지")
    for reg in ("KR", "US"):
        line(R[R.ruleA & (R.region == reg)], f"규칙A {reg}"); line(R[~R.ruleA & (R.region == reg)], f"나머지 {reg}")
    print("\n== 지수 자체의 반등 (게이트 진입일 기준) ==")
    for ep, g in R.groupby("episode"):
        d = g.drop_duplicates("date")
        print(f"ep{ep}: 나스닥 20/60/90일 {d.nq20.mean():+.1f}/{d.nq60.mean():+.1f}/{d.nq90.mean():+.1f}%  "
              f"코스피 {d.ks20.mean():+.1f}/{d.ks60.mean():+.1f}/{d.ks90.mean():+.1f}%")
    print("\n== 자국 지수를 이긴 비율 ==")
    for h in HORIZONS:
        v = R[f"ex{h}"].dropna(); a = R.loc[v.index]
        print(f"{h}일: 전체 {(v > 0).mean() * 100:.0f}% (n={len(v)}) | 규칙A {(a[a.ruleA][f'ex{h}'] > 0).mean() * 100:.0f}% | 나머지 {(a[~a.ruleA][f'ex{h}'] > 0).mean() * 100:.0f}%")
    R.to_csv(DATA_DIR / "rebound_vs_index.csv", index=False)


if __name__ == "__main__":
    main()
