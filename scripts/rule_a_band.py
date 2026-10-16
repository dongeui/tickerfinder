"""A 표시 × 눌림목(150~200일선 밴드 ±3%) × 주봉 스토캐 ≤25 조합의 게이트(나스닥 RSI ≤40) 국면 성과 (미국 2007~).
진입 = 조합이 새로 켜진 날 다음날 시가. 사용: uv run python scripts/rule_a_band.py
"""
import sys, json, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "/Users/dongeui/Desktop/code/tickerfinder")
import numpy as np, pandas as pd
from app.longhist import load_long
from app.indicators import rsi, slow_stochastic
from app.universe import DATA_DIR
pd.set_option("display.width", 220)
ix = load_long("IXIC").dropna(subset=["Close"]); ix.index = pd.to_datetime(ix.index).tz_localize(None).normalize()
nrsi = rsi(ix["Close"])
tickers = [s["ticker"] for s in json.load(open(DATA_DIR / "results.json"))["stocks"] if s["region"] == "US"]
ev = []
for t in tickers:
    df = load_long(t)
    if df is None or len(df) < 300:
        continue
    df = df.dropna(subset=["Close"]); df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    c, o = df["Close"], df["Open"]
    ma50, ma150, ma200 = (c.rolling(n).mean() for n in (50, 150, 200))
    lo, hi = np.minimum(ma150, ma200), np.maximum(ma150, ma200)
    band = np.where(c > hi, (c / hi - 1) * 100, np.where(c < lo, (c / lo - 1) * 100, 0.0))
    band = pd.Series(band, index=c.index).where(lo.notna())
    A = (((ma200 / ma200.shift(20) - 1) * 100) >= 2) & ((c / ma50 - 1) * 100 <= -9) & (rsi(c) <= 40)
    w = df.resample("W-FRI").agg({"High": "max", "Low": "min", "Close": "last"}).dropna()
    skw, _ = slow_stochastic(w["High"], w["Low"], w["Close"])
    skw = skw.shift(1).reindex(c.index, method="ffill")  # 확정된 직전 주봉 (룩어헤드 방지)
    gate = (nrsi.reindex(c.index, method="ffill") <= 40)
    inb = band.abs() <= 3
    st = skw <= 25
    ent = o.shift(-1); f60 = (c.shift(-61) / ent - 1) * 100; f120 = (c.shift(-121) / ent - 1) * 100
    mn60 = (c[::-1].rolling(60, min_periods=1).min()[::-1].shift(-1) / ent - 1) * 100
    combos = {"A ∧ 밴드 안": A & inb, "A ∧ 밴드 안 ∧ 스토캐": A & inb & st, "A ∧ 밴드 이탈": A & (band < -3), "A ∧ 밴드 위": A & (band > 3),
              "A 전체": A, "밴드 안 ∧ 스토캐 (정규 C)": inb & st, "밴드 안 ∧ 스토캐 ∧ A 아님": inb & st & ~A}
    for name, m in combos.items():
        m = (m & gate).fillna(False)
        new = m & ~m.shift(1, fill_value=False) & (c.index >= "2007-01-01")
        for d in c.index[new.to_numpy()]:
            ev.append({"combo": name, "ticker": t, "date": d, "f60": f60.get(d), "f120": f120.get(d), "min60": mn60.get(d), "band": band.get(d)})
E = pd.DataFrame(ev).dropna(subset=["f60"])
order = ["A ∧ 밴드 안 ∧ 스토캐", "A ∧ 밴드 안", "A ∧ 밴드 이탈", "A ∧ 밴드 위", "A 전체", "밴드 안 ∧ 스토캐 (정규 C)", "밴드 안 ∧ 스토캐 ∧ A 아님"]
T = E.groupby("combo").agg(n=("f60", "size"), 종목=("ticker", "nunique"), 국면일=("date", "nunique"), 승률60=("f60", lambda x: (x > 0).mean() * 100), 평균60=("f60", "mean"),
                           중앙60=("f60", "median"), 하위10_60=("f60", lambda x: x.quantile(.1)), 중간최저60=("min60", "mean"),
                           승률120=("f120", lambda x: (x.dropna() > 0).mean() * 100), 평균120=("f120", "mean")).reindex(order).round(1)
print("게이트(나스닥 RSI ≤40) 국면, 미국 2007~ · 다음날 시가 진입"); print(T.to_string())
best = E[E.combo == "A ∧ 밴드 안"].copy(); best["yr"] = best.date.dt.year
print("\n[A ∧ 밴드 안] 연도별 60일:"); print(best.groupby("yr").f60.agg(n="size", 승률=lambda x: (x > 0).mean() * 100, 평균="mean").round(1).T.to_string())
