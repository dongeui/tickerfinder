"""게이트 열린 날 조건(C) 충족 종목이 몇 개나 되고, 그중 무엇으로 고르면 성과가 갈리는가 (미국 20년 PIT 이벤트)."""
import sys, json, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "/Users/dongeui/Desktop/code/tickerfinder")
import numpy as np, pandas as pd
from app.longhist import load_long
from app.indicators import rsi
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
E = pd.read_parquet("data/pit_prob_events.parquet"); E = E[(E.region=="US")&(E.combo=="C")].copy(); E["date"]=pd.to_datetime(E.date)
res = json.load(open("data/results.json")); meta = {r["ticker"]: r for r in res["stocks"]}
E["sector"] = E.ticker.map(lambda t: meta.get(t,{}).get("sector")); E["marcap_now"] = E.ticker.map(lambda t: meta.get(t,{}).get("marcap"))
# 이벤트 시점 추가 지표
feats = {}
for t, g in E.groupby("ticker"):
    df = load_long(t)
    if df is None: continue
    df = df.dropna(subset=["Close"]); df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    c = df["Close"]; v = df["Volume"].fillna(0)
    f = pd.DataFrame({"rsi_d": rsi(c), "turn20": (c*v).rolling(20).mean(), "ma50": c.rolling(50).mean(), "ma200": c.rolling(200).mean()})
    f["slope50"] = (f.ma50/f.ma50.shift(20)-1)*100; f["dist50"] = (c/f.ma50-1)*100; f["ret6m"] = (c/c.shift(126)-1)*100; f["ret1m"] = (c/c.shift(21)-1)*100
    f["vol60"] = c.pct_change().rolling(60).std()*np.sqrt(252)*100
    sub = f.reindex(g.date.values)
    for col in ("rsi_d","turn20","slope50","dist50","ret6m","ret1m","vol60"): E.loc[g.index, col] = sub[col].values
E = E.dropna(subset=["fwd60"])
print(f"미국 C(둘 다 충족 + 게이트) 진입 {len(E):,}건, {E.date.min().date()}~{E.date.max().date()}, 종목 {E.ticker.nunique()}")

# A. 하루에 몇 개?
per_day = E.groupby("date").size()
print(f"\n[A] 게이트 열린 날의 신규 진입(첫 충족일) 종목 수: 진입일 {len(per_day)}일 · 중앙 {per_day.median():.0f}개 · 75% {per_day.quantile(.75):.0f} · 90% {per_day.quantile(.9):.0f} · 최대 {per_day.max()} ({per_day.idxmax().date()})")
# 에피소드(30일 이내 묶음)
d = per_day.index.to_series(); ep = (d.diff().dt.days > 30).cumsum()
epi = E.groupby(ep.reindex(E.date).values).agg(start=("date","min"), end=("date","max"), n=("ticker","size"), names=("ticker","nunique"), fwd60=("fwd60","mean"), win60=("fwd60", lambda x:(x>0).mean()*100))
print(f"게이트 국면(30일 이내 묶음) {len(epi)}번 · 국면당 고유 종목 중앙 {epi.names.median():.0f}개 · 75% {epi.names.quantile(.75):.0f} · 최대 {epi.names.max()}")
print(epi.sort_values("start").tail(14).assign(start=lambda x:x.start.dt.date, end=lambda x:x.end.dt.date).round(1).to_string())

# B. 요인별 5분위
def q(g, col, k=5, labels=None):
    try: return pd.qcut(g[col], k, labels=labels or [f"Q{i+1}" for i in range(k)], duplicates="drop")
    except ValueError: return None
def tab(g, key):
    return g.groupby(key, observed=True).agg(n=("fwd60","size"), win60=("fwd60", lambda x:(x>0).mean()*100), avg60=("fwd60","mean"), med60=("fwd60","median"), p10_60=("fwd60", lambda x:x.quantile(.1)), win180=("fwd180", lambda x:(x.dropna()>0).mean()*100), avg180=("fwd180","mean")).round(1)
print("\n[B] 요인별 5분위 (Q1=낮음 → Q5=높음). 60일 승률/평균/중앙/하위10%, 180일 승률/평균")
for col, name in [("turn20","20일 거래대금(시총 대용)"),("rsi_d","일봉 RSI"),("stoch_k","주봉 스토캐 깊이"),("from_high52","52주 고점 대비"),("slope200","200일선 기울기"),("slope50","50일선 기울기"),("dist50","50일선 이격"),("ret6m","6개월 수익"),("ret1m","1개월 수익"),("vol60","60일 변동성"),("band_pos","밴드 위치")]:
    k = q(E, col)
    if k is None: continue
    print(f"--- {name}"); print(tab(E.assign(k=k), "k").to_string())
E["cap_now"] = pd.qcut(E.marcap_now, 3, labels=["소형(현재)","중형","대형(현재)"])
print("--- 현재 시총 3분위 (생존·현재 기준이라 참고만)"); print(tab(E, "cap_now").to_string())
E["tech"] = E.sector.isin(["Information Technology","Communication Services"])
print("--- 테크 여부"); print(tab(E, "tech").to_string())
print("--- 섹터 (n≥300)"); t = tab(E, "sector"); print(t[t.n>=300].sort_values("avg60", ascending=False).to_string())

# C. 기존 필터(상위×횡보) 위에 요인 추가
core = E[(E.slope200>=2.2)&(E.slope50>=0)]
print(f"\n[C] 기존 필터(200일선 상위 × 50일선 횡보 이상) n={len(core)}: 60일 승률 {(core.fwd60>0).mean()*100:.0f}% 평균 {core.fwd60.mean():+.1f}% p10 {core.fwd60.quantile(.1):+.1f} · 180일 {(core.fwd180.dropna()>0).mean()*100:.0f}% {core.fwd180.mean():+.1f}%")
for col, name in [("turn20","거래대금"),("rsi_d","일봉 RSI"),("from_high52","고점 대비"),("dist50","50일선 이격"),("vol60","변동성")]:
    k = q(core, col, 3, ["하","중","상"])
    if k is None: continue
    print(f"--- 필터 안에서 {name} 3분위"); print(tab(core.assign(k=k), "k").to_string())

# D. 같은 날 후보 중 상위 N 고르기 vs 전부
print("\n[D] 같은 진입일에 후보가 5개 이상일 때: 전부 동일비중 vs 요인 상위 5개 (날짜별 평균 60일 수익, 날짜 승률)")
days = [g for _, g in E.groupby("date") if len(g) >= 5]
print(f"대상 진입일 {len(days)}일 (후보 5개 이상), 후보 합계 {sum(len(g) for g in days)}")
def sim(pick):
    r = np.array([pick(g).fwd60.mean() for g in days]); return f"평균 {r.mean():+.1f}% · 중앙 {np.median(r):+.1f}% · 날짜 승률 {(r>0).mean()*100:.0f}% · 하위10% {np.quantile(r,.1):+.1f}%"
print(f"  전부 동일비중             : {sim(lambda g: g)}")
print(f"  거래대금 큰 5 (대형)       : {sim(lambda g: g.nlargest(5,'turn20'))}")
print(f"  거래대금 작은 5 (소형)     : {sim(lambda g: g.nsmallest(5,'turn20'))}")
print(f"  200일선 기울기 높은 5      : {sim(lambda g: g.nlargest(5,'slope200'))}")
print(f"  200일선 기울기 낮은 5      : {sim(lambda g: g.nsmallest(5,'slope200'))}")
print(f"  고점 대비 덜 빠진 5        : {sim(lambda g: g.nlargest(5,'from_high52'))}")
print(f"  고점 대비 많이 빠진 5      : {sim(lambda g: g.nsmallest(5,'from_high52'))}")
print(f"  RSI 낮은 5               : {sim(lambda g: g.nsmallest(5,'rsi_d'))}")
print(f"  RSI 높은 5               : {sim(lambda g: g.nlargest(5,'rsi_d'))}")
print(f"  변동성 낮은 5             : {sim(lambda g: g.nsmallest(5,'vol60'))}")
print(f"  50일선 기울기 높은 5       : {sim(lambda g: g.nlargest(5,'slope50'))}")
print(f"  기울기 상위×횡보 있으면 그것, 없으면 기울기 상위 5: {sim(lambda g: (lambda c: c if len(c)>=1 else g.nlargest(5,'slope200'))(g[(g.slope200>=2.2)&(g.slope50>=0)]))}")
