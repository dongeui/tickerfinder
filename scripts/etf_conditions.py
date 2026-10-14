"""ETF 에 현재 매수조건(미국 설정: 주봉 스토캐 ≤25, 150~200일선 밴드 ±3, 나스닥 RSI ≤40)이 의미 있는지 비교."""
import sys, pathlib, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/Users/dongeui/Desktop/code/tickerfinder")
import numpy as np, pandas as pd
from app.backfill import compute_series
from app.indicators import rsi
from app.longhist import load_long
D = pathlib.Path(__file__).resolve().parent.parent / "data" / "etf_bt"  # scripts/etf_conditions_dl.py 가 받은 파일
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 400)

ix = load_long("IXIC")["Close"].dropna(); ix.index = pd.to_datetime(ix.index).tz_localize(None).normalize()
gate_us = (rsi(ix) <= 40)
krw = pd.read_parquet(D/"KRW_X.parquet")["Close"].dropna(); krw.index = pd.to_datetime(krw.index).tz_localize(None).normalize()

def load(sym):
    df = pd.read_parquet(D/(sym.replace("=","_").replace(".","_")+".parquet")).dropna(subset=["Close"])
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    return df

def to_krw(df):
    k = krw.reindex(df.index, method="ffill")
    out = df.copy()
    for c in ("Open","High","Low","Close"): out[c] = df[c]*k
    return out.dropna(subset=["Close"])

H = (20, 60, 180)
def fwd(c, o):
    n=len(c); arr=c.to_numpy(float); base=o.shift(-1).to_numpy(float)
    out={}
    for h in H:
        f=np.full(n,np.nan); k=h+1
        if n>k: f[:-k]=arr[k:]/base[:-k]-1
        out[h]=pd.Series(f*100,index=c.index)
    return out

def prep(df, kr_listed):
    ser = compute_series(df)
    c = ser["close"]
    ser["ma50"] = c.rolling(50).mean()
    ser["slope200"] = (ser["ma200"]/ser["ma200"].shift(20)-1)*100
    ser["slope50"] = (ser["ma50"]/ser["ma50"].shift(20)-1)*100
    ser["hi52"] = (c/c.rolling(250).max()-1)*100
    g = gate_us.reindex(c.index, method="ffill")
    if kr_listed: g = gate_us.shift(1).reindex(c.index, method="ffill")
    ser["gate"] = g.fillna(False).astype(bool)
    ser["gate35"] = (rsi(ix)<=35).shift(1 if kr_listed else 0).reindex(c.index, method="ffill").fillna(False).astype(bool)
    ser["stoch_dw"] = ser["stoch_k_w"]
    f = fwd(c, df["Open"].reindex(c.index).fillna(c))
    for h in H: ser[f"f{h}"] = f[h]
    return ser

RULES = {
  "C 현재(주스토≤25&밴드±3&게이트)": lambda s: (s.stoch_k_w<=25)&(s.band_pos.abs()<=3)&s.gate,
  "A 주스토≤25&게이트":           lambda s: (s.stoch_k_w<=25)&s.gate,
  "B 밴드±3&게이트":              lambda s: (s.band_pos.abs()<=3)&s.gate,
  "게이트만(나스닥RSI≤40)":         lambda s: s.gate,
  "C 밴드±5":                      lambda s: (s.stoch_k_w<=25)&(s.band_pos.abs()<=5)&s.gate,
  "C 게이트 없이":                  lambda s: (s.stoch_k_w<=25)&(s.band_pos.abs()<=3),
  "C & 200일선 상승(>0)":           lambda s: (s.stoch_k_w<=25)&(s.band_pos.abs()<=3)&s.gate&(s.slope200>0),
  "C & 200일선 하락(≤0)":           lambda s: (s.stoch_k_w<=25)&(s.band_pos.abs()<=3)&s.gate&(s.slope200<=0),
  "일RSI≤30":                      lambda s: s.rsi_d<=30,
  "일RSI≤30&200일선 위":            lambda s: (s.rsi_d<=30)&(s.close>s.ma200),
  "일스토≤20&게이트":               lambda s: (s.stoch_k<=20)&s.gate,
  "고점-10%이상&200일선 상승":       lambda s: (s.hi52<=-10)&(s.slope200>0),
  "게이트≤35&200일선 위":           lambda s: s.gate35&(s.close>s.ma200),
  "게이트&200일선 상승":             lambda s: s.gate&(s.slope200>0),
}

def stats(ser, hit, first_only=True):
    hit = hit.fillna(False).astype(bool)
    e = hit & ~hit.shift(1, fill_value=False) if first_only else hit
    sub = ser[e]
    r = {"n": int(len(sub))}
    for h in H:
        v = sub[f"f{h}"].dropna()
        if len(v): r[f"win{h}"]=round((v>0).mean()*100); r[f"avg{h}"]=round(v.mean(),1); r[f"p10_{h}"]=round(v.quantile(.1),1)
    return r

def base(ser):
    r={"n": int(ser.f60.notna().sum())}
    for h in H:
        v=ser[f"f{h}"].dropna(); r[f"win{h}"]=round((v>0).mean()*100); r[f"avg{h}"]=round(v.mean(),1); r[f"p10_{h}"]=round(v.quantile(.1),1)
    return r

groups = {
  "미국 ETF 달러(환헤지 가정) 2000~": [(s, load(s), False) for s in ["QQQ","SPY","DIA","IWM","SMH","XLK","SCHD","VIG","XLV","XLF","XLE"]],
  "미국 ETF 원화환산(환노출 가정) 2004~": [(s+"₩", to_krw(load(s)), True) for s in ["QQQ","SPY","DIA","IWM","SMH","XLK","SCHD","VIG"]],
  "국내상장 실제 ETF": [(s, load(s), True) for s in ["133690.KS","143850.KS","245340.KS","360750.KS","379810.KS","381180.KS"]],
}
allrows=[]
for gname, items in groups.items():
    print("\n=====", gname)
    sers = {s: prep(df, kr) for s, df, kr in items}
    # 기준율(모든 날)
    pooled = pd.concat(sers.values())
    print("기준율(모든 날 진입):", base(pooled))
    tab=[]
    for rn, fn in RULES.items():
        agg = pd.concat([ser[(fn(ser).fillna(False).astype(bool)) & ~(fn(ser).fillna(False).astype(bool)).shift(1, fill_value=False)] for ser in sers.values()])
        r={"rule":rn, "n":len(agg)}
        for h in H:
            v=agg[f"f{h}"].dropna()
            if len(v): r[f"win{h}"]=round((v>0).mean()*100); r[f"avg{h}"]=round(v.mean(),1); r[f"med{h}"]=round(v.median(),1); r[f"p10_{h}"]=round(v.quantile(.1),1)
        tab.append(r); allrows.append({"group":gname, **r})
    print(pd.DataFrame(tab).to_string(index=False))
    # 종목별 현재 조건
    print("-- 종목별 현재 조건 C:")
    for s, ser in sers.items():
        st = stats(ser, RULES["C 현재(주스토≤25&밴드±3&게이트)"](ser)); b=base(ser)
        print(f"  {s:10s} n={st['n']:3d} 60일 승률 {st.get('win60','-')}% 평균 {st.get('avg60','-')}% p10 {st.get('p10_60','-')} | 180일 {st.get('win180','-')}% {st.get('avg180','-')}% | 기준(모든날) 60일 {b['win60']}% {b['avg60']}% / 180일 {b['win180']}% {b['avg180']}%  ({ser.index[0].year}~)")
# 시대별 (달러 그룹, 현재 조건)
print("\n===== 달러 그룹 현재 조건 C, 시대별")
sers = {s: prep(load(s), False) for s in ["QQQ","SPY","DIA","IWM","SMH","XLK","XLV","XLF","XLE"]}
fn = RULES["C 현재(주스토≤25&밴드±3&게이트)"]
agg = pd.concat([ser[(fn(ser).fillna(False).astype(bool)) & ~(fn(ser).fillna(False).astype(bool)).shift(1, fill_value=False)] for ser in sers.values()])
agg["era"] = pd.cut(agg.index.year, [1999,2007,2012,2019,2026], labels=["2000-07","2008-12","2013-19","2020-26"])
print(agg.groupby("era").agg(n=("f60","size"), win60=("f60", lambda v: round((v.dropna()>0).mean()*100)), avg60=("f60","mean"), win180=("f180", lambda v: round((v.dropna()>0).mean()*100)), avg180=("f180","mean")).round(1))
print("\n===== 달러 그룹: 현재 조건 C 를 200일선 기울기 3분위(미국 개별주 기준 1.0/2.2)로")
agg["tier"] = pd.cut(agg.slope200, [-99, 1.0, 2.2, 99], labels=["하위","중위","상위"])
print(agg.groupby("tier").agg(n=("f60","size"), win60=("f60", lambda v: round((v.dropna()>0).mean()*100)), avg60=("f60","mean"), p10_60=("f60", lambda v: v.quantile(.1)), win180=("f180", lambda v: round((v.dropna()>0).mean()*100)), avg180=("f180","mean")).round(1))
print("\n===== 달러 그룹: 신호 빈도 (연평균 종목당 진입 횟수)")
for s, ser in sers.items():
    h = fn(ser).fillna(False).astype(bool); e = h & ~h.shift(1, fill_value=False); yrs = (ser.index[-1]-ser.index[0]).days/365
    print(f"  {s}: {e.sum()}회 / {yrs:.0f}년 = 연 {e.sum()/yrs:.1f}회, 충족일 비율 {h.mean()*100:.1f}%")
