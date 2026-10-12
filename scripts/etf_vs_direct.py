"""나스닥 게이트(RSI 30~40)가 열린 날, 원화 기준으로 무엇이 유리했나:
  A) 미국 직투(QQQ, 각 시점 환율로 환전)  B) 국내상장 환노출 ETF  C) 국내상장 환헤지 ETF

사용: uv run python scripts/etf_vs_direct.py
"""
from __future__ import annotations

import glob
import json
import sys
import warnings
from pathlib import Path

import FinanceDataReader as fdr
import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")
START = "2025-01-01"
ETFS = {"TIGER나스닥100(환노출)": "133690", "KODEX나스닥100(H)": "449190"}


def yfc(sym: str) -> pd.Series:
    d = yf.download(sym, start=START, progress=False, auto_adjust=True)
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = d.columns.get_level_values(0)
    return d["Close"].tz_localize(None)


def kr_next(us_date: pd.Timestamp, s: pd.Series):
    """미국 날짜 D 이후 첫 국내 거래일 (D 종가 정보를 반영한 첫 국내 세션)."""
    t = s.loc[us_date + pd.Timedelta(days=1):]
    return t.index[0] if len(t) else None


def main():
    qqq, ixic, fx = yfc("QQQ"), yfc("^IXIC"), yfc("KRW=X")
    etf = {k: fdr.DataReader(c, START)["Close"] for k, c in ETFS.items()}
    gate = []
    for f in sorted(glob.glob("data/history/*.json")):
        d = json.load(open(f))
        ix = d.get("indices", {}).get("IXIC")
        if ix and ix.get("date") == Path(f).stem and 30 <= ix["rsi_d"] <= 40:
            gate.append(pd.Timestamp(ix["date"]))
    print("게이트 날짜:", [str(g.date()) for g in gate])
    rows = []
    for g in gate:
        for h in (20, 60):
            q = qqq.loc[g:]
            if len(q) <= h:
                continue
            d1, d2 = q.index[0], q.index[h]
            f1 = fx.reindex([d1], method="ffill").iloc[0]
            f2 = fx.reindex([d2], method="ffill").iloc[0]
            r = {"gate": str(g.date()), "h": h, "QQQ달러": (q.iloc[h] / q.iloc[0] - 1) * 100, "환율변화": (f2 / f1 - 1) * 100,
                 "A_미국직투(원화)": ((q.iloc[h] / q.iloc[0]) * (f2 / f1) - 1) * 100}
            for k, s in etf.items():
                k1, k2 = kr_next(d1, s), kr_next(d2, s)
                r[k] = (s.loc[k2] / s.loc[k1] - 1) * 100 if k1 is not None and k2 is not None and k2 > k1 else np.nan
            rows.append(r)
    R = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    for h in (20, 60):
        g = R[R.h == h].drop(columns="h").round(1)
        print(f"\n== +{h}거래일 (원화 수익률 %) ==")
        print(g.to_string(index=False))
        print("평균:", g.drop(columns="gate").mean().round(2).to_dict())
    nr, fr = ixic.pct_change(), fx.pct_change().reindex(ixic.index, method="ffill")
    print("\n[환율] 나스닥 일수익률 vs 원/달러 일변화 상관:", round(nr.corr(fr), 2),
          "| 나스닥 -2% 이하인 날 원/달러 평균 변화:", f"{fr[nr <= -0.02].mean() * 100:+.2f}%")
    e, eh = etf["TIGER나스닥100(환노출)"], etf["KODEX나스닥100(H)"]
    a = e.index[-250]
    bench = qqq * fx.reindex(qqq.index, method="ffill")
    ba = bench.reindex([a - pd.Timedelta(days=1)], method="ffill").iloc[0]
    bb = bench.reindex([e.index[-1] - pd.Timedelta(days=1)], method="ffill").iloc[0]
    print(f"[1년 추적] 환노출 ETF {(e.iloc[-1] / e.loc[a] - 1) * 100:+.1f}% vs QQQ×환율 {(bb / ba - 1) * 100:+.1f}%")
    print(f"[1년 추적] 환헤지 ETF {(eh.iloc[-1] / eh.iloc[-250] - 1) * 100:+.1f}% vs QQQ 달러 {(qqq.iloc[-1] / qqq.iloc[-250] - 1) * 100:+.1f}%")


if __name__ == "__main__":
    main()
