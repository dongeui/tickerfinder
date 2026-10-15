"""하락장(나스닥 RSI 게이트 국면)에서 버틴 종목 vs 같이 맞은 종목 — 이후 회복 탄력 비교 (미국 유니버스 20년).

국면: 나스닥 RSI(14) 가 45 위에서 40 이하로 처음 내려온 날(게이트 첫날). 하락 구간 = 그 전 20거래일.
상대강도 RS = 종목의 하락 구간 수익 - 나스닥의 하락 구간 수익. 이후 수익 = 게이트 첫날 종가 → +20/60/120 거래일.
사용: uv run python scripts/rs_in_drawdown.py
"""
import sys, json, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "/Users/dongeui/Desktop/code/tickerfinder")
import numpy as np, pandas as pd
from app.longhist import load_long
from app.indicators import rsi, slow_stochastic
from app.universe import DATA_DIR
pd.set_option("display.width", 220)

ix = load_long("IXIC").dropna(subset=["Close"]); ix.index = pd.to_datetime(ix.index).tz_localize(None).normalize()
c = ix["Close"]; r = rsi(c)
gate_days = [c.index[i] for i in range(21, len(c) - 1) if r.iloc[i] <= 40 and (r.iloc[i-20:i] > 45).any() and not (r.iloc[i-5:i] <= 40).any()]
gate_days = [d for d in gate_days if d >= pd.Timestamp("2007-01-01")]
print(f"게이트 첫날(나스닥 RSI≤40 진입) {len(gate_days)}회, {gate_days[0].date()}~{gate_days[-1].date()}")

res = json.load(open(DATA_DIR / "results.json"))
tickers = [s["ticker"] for s in res["stocks"] if s["region"] == "US"]
rows = []
for t in tickers:
    df = load_long(t)
    if df is None or len(df) < 300:
        continue
    df = df.dropna(subset=["Close"]); df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    p = df["Close"]
    w = df.resample("W-FRI").last().dropna(); skw, _ = slow_stochastic(w["High"], w["Low"], w["Close"])
    ma200 = p.rolling(200).mean()
    for g in gate_days:
        if g not in p.index:
            continue
        i = p.index.get_loc(g); gi = c.index.get_loc(g)
        if i < 220 or i + 120 >= len(p):
            continue
        win = (p.iloc[i] / p.iloc[i-20] - 1) * 100; mkt = (c.iloc[gi] / c.iloc[gi-20] - 1) * 100
        rows.append({"date": g, "ticker": t, "ret20": win, "rs": win - mkt, "mkt20": mkt, "stoch_w": float(skw[skw.index <= g].iloc[-1]) if (skw.index <= g).any() else np.nan,
                     "above200": p.iloc[i] > ma200.iloc[i], "f20": (p.iloc[i+20] / p.iloc[i] - 1) * 100, "f60": (p.iloc[i+60] / p.iloc[i] - 1) * 100, "f120": (p.iloc[i+120] / p.iloc[i] - 1) * 100,
                     "mkt60": (c.iloc[gi+60] / c.iloc[gi] - 1) * 100})
E = pd.DataFrame(rows)
print(f"종목×국면 {len(E):,}건, 종목 {E.ticker.nunique()}, 국면 {E.date.nunique()}  (하락 구간 나스닥 중앙 {E.groupby('date').mkt20.first().median():+.1f}%)")

def tab(g, key):
    return g.groupby(key, observed=True).agg(n=("f60", "size"), 하락구간수익=("ret20", "mean"), 이후20일=("f20", "mean"), 이후60일=("f60", "mean"), 승률60=("f60", lambda x: (x > 0).mean() * 100),
                                             하위10_60=("f60", lambda x: x.quantile(.1)), 이후120일=("f120", "mean"), 승률120=("f120", lambda x: (x > 0).mean() * 100)).round(1)
E["grp"] = pd.cut(E.ret20, [-np.inf, -20, -10, -5, 0, np.inf], labels=["-20% 이하", "-20~-10%", "-10~-5%", "-5~0%", "올랐음(>0)"])
print("\n[A] 하락 구간(게이트 전 20거래일) 종목의 절대 수익별 → 이후 성과"); print(tab(E, "grp").to_string())
E["rsq"] = pd.qcut(E.rs, 5, labels=["RS 최하(같이 더 맞음)", "RS 하", "RS 중", "RS 상", "RS 최상(버팀)"])
print("\n[B] 상대강도(종목 - 나스닥) 5분위 → 이후 성과"); print(tab(E, "rsq").to_string())
# 국면별로 묶어서 (한 국면이 좌우하지 않게): 국면마다 '버틴 상위 20%' 평균 - '맞은 하위 20%' 평균
per = E.groupby("date").apply(lambda g: pd.Series({"top": g[g.rsq == "RS 최상(버팀)"].f60.mean(), "bot": g[g.rsq == "RS 최하(같이 더 맞음)"].f60.mean(), "top120": g[g.rsq == "RS 최상(버팀)"].f120.mean(), "bot120": g[g.rsq == "RS 최하(같이 더 맞음)"].f120.mean()})).dropna()
print(f"\n[C] 국면별 비교 ({len(per)}국면): 60일 — 버틴 상위20% 평균 {per.top.mean():+.1f}% vs 맞은 하위20% {per.bot.mean():+.1f}%, 버틴 쪽이 이긴 국면 {(per.top > per.bot).mean()*100:.0f}%")
print(f"                      120일 — {per.top120.mean():+.1f}% vs {per.bot120.mean():+.1f}%, 버틴 쪽이 이긴 국면 {(per.top120 > per.bot120).mean()*100:.0f}%")
# 매수 후보(주봉 스토캐 ≤25) 안에서만
cand = E[E.stoch_w <= 25]
print(f"\n[D] 매수 후보(주봉 스토캐 ≤25, {len(cand):,}건) 안에서 상대강도 3분위")
cand = cand.assign(rs3=pd.qcut(cand.rs, 3, labels=["같이 맞음", "중간", "버팀"]))
print(tab(cand, "rs3").to_string())
print("\n[E] 200일선 위/아래 × 버팀/맞음")
E["pos"] = np.where(E.above200, "200일선 위", "200일선 아래")
E["rs2"] = np.where(E.rs > 0, "지수보다 버팀", "지수보다 더 맞음")
print(E.groupby(["pos", "rs2"]).agg(n=("f60", "size"), 이후60일=("f60", "mean"), 승률60=("f60", lambda x: (x > 0).mean() * 100), 이후120일=("f120", "mean")).round(1).to_string())
