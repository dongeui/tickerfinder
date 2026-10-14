"""매도 신호에 팔고 다음 매수 신호에 되사는 '사이클'이 그냥 들고 있는 것보다 나은가."""
import sys, runpy, pathlib, io, contextlib, warnings; warnings.filterwarnings("ignore")
sys.argv=["x"]; sys.path.insert(0,"/Users/dongeui/Desktop/code/tickerfinder")
with contextlib.redirect_stdout(io.StringIO()):
    g = runpy.run_path(str(pathlib.Path(__file__).parent/"etf_conditions.py"))
import pandas as pd, numpy as np
prep, load, to_krw = g["prep"], g["load"], g["to_krw"]
from app.indicators import rsi
from app.longhist import load_long
ix = load_long("IXIC")["Close"].dropna(); ix.index = pd.to_datetime(ix.index).tz_localize(None).normalize(); ixr = rsi(ix)

SELLS = {
  "현재(일RSI≥75&주RSI65~85)": lambda s: (s.rsi_d>=75)&(s.rsi_w>=65)&(s.rsi_w<=85),
  "일RSI≥80":                 lambda s: s.rsi_d>=80,
  "주RSI≥75":                 lambda s: s.rsi_w>=75,
  "주RSI≥80":                 lambda s: s.rsi_w>=80,
  "이격≥+20%":                lambda s: s.dist200>=20,
  "일RSI≥75&이격≥+15%":        lambda s: (s.rsi_d>=75)&(s.dist200>=15),
}
BUYS = {"H": lambda s: (s.stoch_k_w<=25)&(s.band_pos.abs()<=3)&s.gate, "open": lambda s: (s.stoch_k_w<=25)&s.gate}

def cycles(ser, sell_fn, buy_fn, max_days=250):
    c = ser["close"]; o = ser["open"]
    sh = sell_fn(ser).fillna(False).astype(bool); se = sh & ~sh.shift(1, fill_value=False)
    bh = buy_fn(ser).fillna(False).astype(bool); be = bh & ~bh.shift(1, fill_value=False)
    bidx = np.flatnonzero(be.to_numpy()); out=[]
    pos = np.flatnonzero(se.to_numpy())
    for i in pos:
        if i+1 >= len(c): continue
        sell_px = o.iloc[i+1]  # 신호 다음날 시가에 매도
        j = bidx[bidx > i]
        if len(j)==0: continue
        j = j[0]
        if j+1 >= len(c): continue
        if j - i > max_days:  # 1년 내 재매수 신호 없음 → 1년 뒤 종가에 재진입했다고 가정
            k = min(i+max_days, len(c)-1); rebuy = c.iloc[k]; days = k-i; timeout=True
        else:
            rebuy = o.iloc[j+1]; days = j-i; timeout=False
        out.append({"gain": (sell_px/rebuy-1)*100, "days": days, "timeout": timeout, "missed_max": (c.iloc[i+1:i+1+days].max()/sell_px-1)*100 if days>0 else 0})
    return pd.DataFrame(out)

groups = {
  "달러(환헤지)": ({s: load(s) for s in ["QQQ","SPY","DIA","IWM","SMH","XLK","SCHD","VIG","XLV","XLF","XLE"]}, False, "H"),
  "원화환산(환노출)": ({s+"₩": to_krw(load(s)) for s in ["QQQ","SPY","DIA","IWM","SMH","XLK","SCHD","VIG"]}, True, "open"),
  "국내 실제 ETF": ({s: load(s) for s in ["133690.KS","143850.KS","245340.KS","360750.KS","379810.KS","381180.KS"]}, True, "open"),
}
print("사이클 = 매도 신호 다음날 시가에 팔고 → 다음 매수 신호 다음날 시가에 되삼 (1년 내 매수 신호 없으면 1년 뒤 종가). 이득 = 판 가격 / 되산 가격 - 1 (양수면 싸게 되삼)")
for gname, (dfs, kr, fx) in groups.items():
    sers={}
    for s, df in dfs.items():
        ser = prep(df, kr); ser["open"] = df["Open"].reindex(ser.index).fillna(ser["close"]); ser["dist200"]=(ser.close/ser.ma200-1)*100; sers[s]=ser
    print(f"\n===== {gname} (되사기 규칙: {'스토캐+밴드+게이트' if fx=='H' else '스토캐+게이트'})")
    for rn, fn in SELLS.items():
        d = pd.concat([cycles(ser, fn, BUYS[fx]) for ser in sers.values()], ignore_index=True)
        if not len(d): print(f"  {rn}: 없음"); continue
        print(f"  {rn:24s} n={len(d):4d} 싸게 되산 비율 {(d.gain>0).mean()*100:3.0f}% · 이득 평균 {d.gain.mean():+5.1f}% 중앙 {d.gain.median():+5.1f}% 하위10% {d.gain.quantile(.1):+5.1f}% · 되사기까지 중앙 {d.days.median():3.0f}일 · 1년 내 매수신호 없음 {d.timeout.mean()*100:2.0f}% · 그 사이 놓친 최고점 평균 +{d.missed_max.mean():.1f}%")
