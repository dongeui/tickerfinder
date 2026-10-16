"""A 표시(200일선 20일 기울기 ≥ +2% & 50일선 대비 -9% 이하 & 일봉 RSI ≤ 40) 종목의 게이트 국면 성과 (미국 유니버스, 2007~).

진입 = A 조건이 새로 켜진 날(전날 미충족)의 다음날 시가. 게이트 = 그날 나스닥 RSI(14) ≤ 40 (모델) 또는 ≤ 46 (사용자 설정).
비교: 같은 게이트에서 모든 종목 아무 날(기준율), 정규 매수 조건 C(scripts/pit_prob 이벤트), 게이트 밖 A.
사용: uv run python scripts/rule_a_gate.py
"""
import sys, json, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "/Users/dongeui/Desktop/code/tickerfinder")
import numpy as np, pandas as pd
from app.longhist import load_long
from app.indicators import rsi
from app.universe import DATA_DIR
pd.set_option("display.width", 220)

ix = load_long("IXIC").dropna(subset=["Close"]); ix.index = pd.to_datetime(ix.index).tz_localize(None).normalize()
nrsi = rsi(ix["Close"])
res = json.load(open(DATA_DIR / "results.json"))
tickers = [s["ticker"] for s in res["stocks"] if s["region"] == "US"]
H = (20, 60, 120)
ev, base = [], []
for t in tickers:
    df = load_long(t)
    if df is None or len(df) < 300:
        continue
    df = df.dropna(subset=["Close"]); df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    c, o = df["Close"], df["Open"]
    ma200, ma50 = c.rolling(200).mean(), c.rolling(50).mean()
    slope = (ma200 / ma200.shift(20) - 1) * 100
    a = (slope >= 2) & ((c / ma50 - 1) * 100 <= -9) & (rsi(c) <= 40)
    g = nrsi.reindex(c.index, method="ffill")
    new = a & ~a.shift(1, fill_value=False)
    idx = np.arange(len(c))
    ent = o.shift(-1)  # 다음날 시가
    fw = {h: (c.shift(-(h + 1)) / ent - 1) * 100 for h in H}
    mn60 = (c[::-1].rolling(60, min_periods=1).min()[::-1].shift(-1) / ent - 1) * 100
    sel = new & (c.index >= "2007-01-01")
    for d in c.index[sel.to_numpy()]:
        ev.append({"ticker": t, "date": d, "nrsi": g.get(d), **{f"f{h}": fw[h].get(d) for h in H}, "min60": mn60.get(d)})
    # 기준율: 게이트 날 아무 종목 (10일 간격 표본)
    for d in c.index[(c.index >= "2007-01-01")][::10]:
        base.append({"nrsi": g.get(d), **{f"f{h}": fw[h].get(d) for h in H}})
E = pd.DataFrame(ev).dropna(subset=["f60"]); B = pd.DataFrame(base).dropna(subset=["f60"])

def st(x, name):
    f = lambda h: x[f"f{h}"].dropna()
    return {"구분": name, "n": len(x), "종목": x["ticker"].nunique() if "ticker" in x else None,
            "20일 승률": (f(20) > 0).mean() * 100, "20일 평균": f(20).mean(),
            "60일 승률": (f(60) > 0).mean() * 100, "60일 평균": f(60).mean(), "60일 하위10%": f(60).quantile(.1), "60일 내 최저 평균": x["min60"].mean() if "min60" in x else np.nan,
            "120일 승률": (f(120) > 0).mean() * 100, "120일 평균": f(120).mean()}
rows = [st(E[E.nrsi <= 40], "A · 나스닥 RSI ≤40 (모델 게이트)"), st(E[(E.nrsi > 40) & (E.nrsi <= 46)], "A · 40~46 (사용자 설정 추가분)"),
        st(E[E.nrsi <= 46], "A · ≤46 합계"), st(E[E.nrsi > 46], "A · 게이트 밖 (>46)"), st(E, "A · 전체"),
        st(B[B.nrsi <= 40], "기준율: 게이트 ≤40 아무 종목"), st(B, "기준율: 아무 날 아무 종목")]
try:
    P = pd.read_parquet(DATA_DIR / "pit_prob_events.parquet"); P = P[(P.region == "US") & (P.combo == "C")].dropna(subset=["fwd60"])
    rows.append({"구분": "정규 매수 조건 C (스토캐+밴드, ≤40)", "n": len(P), "종목": P.ticker.nunique(), "20일 승률": (P.fwd20 > 0).mean() * 100, "20일 평균": P.fwd20.mean(),
                 "60일 승률": (P.fwd60 > 0).mean() * 100, "60일 평균": P.fwd60.mean(), "60일 하위10%": P.fwd60.quantile(.1), "60일 내 최저 평균": P.min60.mean(),
                 "120일 승률": np.nan, "120일 평균": np.nan})
except Exception:  # noqa: BLE001
    pass
print(pd.DataFrame(rows).set_index("구분").round(1).to_string())
G = E[E.nrsi <= 40].copy(); G["yr"] = G.date.dt.year
print("\n[A · ≤40] 연도별 60일:"); print(G.groupby("yr").f60.agg(n="size", 승률=lambda x: (x > 0).mean() * 100, 평균="mean").round(1).T.to_string())
# 같은 날 A 가 여러 개면 동일비중 묶음으로 (날짜 단위 승률)
day = G.groupby("date").f60.mean()
print(f"\n[A · ≤40] 진입일 단위 동일비중: {len(day)}일, 날짜 승률 {(day > 0).mean()*100:.0f}%, 평균 {day.mean():+.1f}%, 하위10% {day.quantile(.1):+.1f}%")
