import sys, runpy, pathlib, io, contextlib, warnings; warnings.filterwarnings("ignore")
sys.argv=["x"]; sys.path.insert(0,"/Users/dongeui/Desktop/code/tickerfinder")
with contextlib.redirect_stdout(io.StringIO()):
    g = runpy.run_path(str(pathlib.Path(__file__).parent/"etf_conditions.py"))
import pandas as pd, numpy as np
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
prep, load, to_krw, H = g["prep"], g["load"], g["to_krw"], g["H"]

def prep2(df, kr):
    ser = prep(df, kr); c = ser["close"]
    ser["dist200"] = (c/ser["ma200"]-1)*100
    ser["rsi_d_max20"] = ser["rsi_d"].rolling(20).max()
    for h in H:  # 신호 다음날 시가 매도 가정이 아니라, 보유자가 H일 내 겪는 최저/최고 (종가 기준, 신호일 종가 대비)
        rev = c[::-1]
        ser[f"min{h}"] = (rev.rolling(h, min_periods=h).min()[::-1].shift(-1)/c-1)*100
        ser[f"max{h}"] = (rev.rolling(h, min_periods=h).max()[::-1].shift(-1)/c-1)*100
    return ser

RULES = {
  "현재(일RSI≥75 & 주RSI 65~85)": lambda s: (s.rsi_d>=75)&(s.rsi_w>=65)&(s.rsi_w<=85),
  "일RSI≥70":                     lambda s: s.rsi_d>=70,
  "일RSI≥75":                     lambda s: s.rsi_d>=75,
  "일RSI≥80":                     lambda s: s.rsi_d>=80,
  "일RSI≥80 & 주RSI≥70":          lambda s: (s.rsi_d>=80)&(s.rsi_w>=70),
  "주RSI≥70":                     lambda s: s.rsi_w>=70,
  "주RSI≥75":                     lambda s: s.rsi_w>=75,
  "주RSI≥80":                     lambda s: s.rsi_w>=80,
  "주스토≥85":                    lambda s: s.stoch_k_w>=85,
  "200일선 이격 ≥+10%":            lambda s: s.dist200>=10,
  "200일선 이격 ≥+15%":            lambda s: s.dist200>=15,
  "200일선 이격 ≥+20%":            lambda s: s.dist200>=20,
  "일RSI≥75 & 이격≥+10%":          lambda s: (s.rsi_d>=75)&(s.dist200>=10),
  "일RSI≥75 & 이격≥+15%":          lambda s: (s.rsi_d>=75)&(s.dist200>=15),
  "나스닥RSI≥70(게이트 반대)":       lambda s: s.ixrsi>=70,
  "일RSI≥75 & 나스닥RSI≥70":        lambda s: (s.rsi_d>=75)&(s.ixrsi>=70),
}
from app.indicators import rsi
from app.longhist import load_long
ix = load_long("IXIC")["Close"].dropna(); ix.index = pd.to_datetime(ix.index).tz_localize(None).normalize(); ixr = rsi(ix)

def table(sers, hz=(20,60,180)):
    pooled = pd.concat(sers.values())
    rows=[]
    b={"rule":"기준율(모든 날)", "n":int(pooled.f60.notna().sum())}
    for h in hz:
        v=pooled[f"f{h}"].dropna(); b[f"win{h}"]=round((v>0).mean()*100); b[f"avg{h}"]=round(v.mean(),1); b[f"dd{h}"]=round(pooled[f"min{h}"].dropna().mean(),1)
    rows.append(b)
    for rn, fn in RULES.items():
        parts=[]
        for ser in sers.values():
            h=fn(ser).fillna(False).astype(bool); e=h&~h.shift(1, fill_value=False); parts.append(ser[e])
        agg=pd.concat(parts); r={"rule":rn,"n":len(agg)}
        for h in hz:
            v=agg[f"f{h}"].dropna()
            if len(v): r[f"win{h}"]=round((v>0).mean()*100); r[f"avg{h}"]=round(v.mean(),1); r[f"dd{h}"]=round(agg[f"min{h}"].dropna().mean(),1)
        rows.append(r)
    return pd.DataFrame(rows)

groups = {
  "미국 ETF 달러 2000~ (QQQ SPY DIA IWM SMH XLK SCHD VIG XLV XLF XLE)": {s: load(s) for s in ["QQQ","SPY","DIA","IWM","SMH","XLK","SCHD","VIG","XLV","XLF","XLE"]},
  "미국 ETF 원화환산 2004~": {s+"₩": to_krw(load(s)) for s in ["QQQ","SPY","DIA","IWM","SMH","XLK","SCHD","VIG"]},
  "국내상장 실제 ETF 2010~": {s: load(s) for s in ["133690.KS","143850.KS","245340.KS","360750.KS","379810.KS","381180.KS"]},
}
for gname, dfs in groups.items():
    kr = "원화" in gname or "국내" in gname
    sers={}
    for s, df in dfs.items():
        ser = prep2(df, kr); ser["ixrsi"] = ixr.shift(1 if kr else 0).reindex(ser.index, method="ffill"); sers[s]=ser
    print("\n=====", gname, " (신호 후 H일 수익: 낮을수록 매도 신호로 유효 · dd = H일 내 최저 종가까지 평균 하락)")
    print(table(sers).to_string(index=False))

# 개별주와 비교: 20년 pit 이벤트? → 대신 미국 개별주 샘플 (prices10y 에서 200종목)로 같은 표
print("\n===== 참고: 미국 개별주 150종목 (data/prices10y) 같은 규칙")
import random, json
uni = [u for u in json.loads(pathlib.Path("/Users/dongeui/Desktop/code/tickerfinder/data/universe.json").read_text()) if u["region"]=="US"]
random.seed(1); pick = random.sample(uni, 150)
sers={}
for u in pick:
    df = load_long(u["ticker"])
    if df is None or len(df)<1500: continue
    df = df.copy(); df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    ser = prep2(df, False); ser["ixrsi"] = ixr.reindex(ser.index, method="ffill"); sers[u["ticker"]]=ser
print(len(sers), "종목")
print(table(sers).to_string(index=False))
