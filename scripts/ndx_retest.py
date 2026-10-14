import pandas as pd, numpy as np, pathlib, warnings; warnings.filterwarnings("ignore")
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
D = pathlib.Path(__file__).resolve().parent.parent / "data" / "etf_bt"  # scripts/ndx_retest_dl.py 가 받은 NDX.parquet
df = pd.read_parquet(D/"NDX.parquet"); c = df["Close"].astype(float); c.index = pd.to_datetime(c.index).tz_localize(None).normalize()
ath_prev = c.cummax().shift(1)
dist = (c/ath_prev-1)*100            # 전고점 대비 %
dd60 = ((c/c.rolling(60).max())-1)   # 참고
# 현재 상태
last = c.index[-1]
print(f"현재 {last.date()} 종가 {c.iloc[-1]:,.0f} · 사상 최고 종가 {c.max():,.0f} ({c.idxmax().date()}) · 고점 대비 {(c.iloc[-1]/c.max()-1)*100:+.2f}%")
low60 = c.iloc[-60:].min(); print(f"최근 60거래일 최저 {low60:,.0f} ({c.iloc[-60:].idxmin().date()}) → 고점 대비 {(low60/c.loc[:c.iloc[-60:].idxmin()].max()-1)*100:+.1f}% 눌림 후 복귀")
# 이벤트: 전고점 ±1% 안에 처음 들어온 날, 직전 60일 안에 전고점 대비 -5% 이하가 있었던 경우(눌림 후 재도전). 같은 재도전은 40일 묶음.
near = dist.abs() <= 1.0
pulled = (dist.rolling(60).min() <= -5)
elig = near & pulled
events = []; last_ev = None
for t in c.index[elig.to_numpy()]:
    if last_ev is not None and (c.index.get_loc(t) - c.index.get_loc(last_ev)) < 40: continue
    last_ev = t; events.append(t)
rows=[]
for t in events:
    i = c.index.get_loc(t); hi = ath_prev.iloc[i]; px = c.iloc[i]
    pull = dist.iloc[max(0,i-60):i].min()
    r = {"date": t.date(), "전고점": round(hi), "직전눌림%": round(pull,1)}
    for h in (20, 60, 120, 250):
        seg = c.iloc[i+1:i+1+h]
        if len(seg) < h: r[f"up{h}"]=np.nan; r[f"dd{h}"]=np.nan; r[f"ret{h}"]=np.nan; continue
        r[f"up{h}"] = (seg.max()/hi-1)*100     # 전고점 위로 최대 얼마나
        r[f"dd{h}"] = (seg.min()/px-1)*100     # 이벤트 종가 대비 최대 하락
        r[f"ret{h}"] = (seg.iloc[-1]/px-1)*100
    # 먼저 닿는 쪽: 전고점 +5% vs 이벤트가 -5% (120일 내)
    seg = c.iloc[i+1:i+1+120]
    up_i = next((k for k,v in enumerate(seg) if v >= hi*1.05), None); dn_i = next((k for k,v in enumerate(seg) if v <= px*0.95), None)
    r["먼저(±5%)"] = "상방" if up_i is not None and (dn_i is None or up_i < dn_i) else ("하방" if dn_i is not None else "둘다아님")
    up_i = next((k for k,v in enumerate(seg) if v >= hi*1.10), None); dn_i = next((k for k,v in enumerate(seg) if v <= px*0.90), None)
    r["먼저(±10%)"] = "상방" if up_i is not None and (dn_i is None or up_i < dn_i) else ("하방" if dn_i is not None else "둘다아님")
    rows.append(r)
E = pd.DataFrame(rows)
print(f"\n재도전 이벤트 {len(E)}건 (1985~). up=전고점 위로 최대 %, dd=이벤트 종가 대비 최대 하락 %, ret=기간 수익 %")
print(E.round(1).to_string(index=False))
def summ(E, label):
    print(f"\n--- {label}: n={len(E)}")
    for h in (20,60,120,250):
        s = E.dropna(subset=[f"up{h}"])
        if not len(s): continue
        print(f"  {h:3d}일: 새 고점 경신 {(s[f'up{h}']>0).mean()*100:3.0f}% · +5% 이상 돌파 {(s[f'up{h}']>=5).mean()*100:3.0f}% · +10% 이상 {(s[f'up{h}']>=10).mean()*100:3.0f}% | 돌파폭 중앙 +{s[f'up{h}'].clip(lower=0).median():.1f}% (상위25% +{s[f'up{h}'].quantile(.75):.1f}%) | 최대하락 중앙 {s[f'dd{h}'].median():.1f}% · -5% 이하 {(s[f'dd{h}']<=-5).mean()*100:3.0f}% · -10% 이하 {(s[f'dd{h}']<=-10).mean()*100:3.0f}% · 하위25% {s[f'dd{h}'].quantile(.25):.1f}% | 기간수익 중앙 {s[f'ret{h}'].median():+.1f}% 승률 {(s[f'ret{h}']>0).mean()*100:.0f}%")
    s = E.dropna(subset=["up120"])
    print(f"  120일 내 먼저 닿는 쪽 ±5%: 상방 {(s['먼저(±5%)']=='상방').mean()*100:.0f}% / 하방 {(s['먼저(±5%)']=='하방').mean()*100:.0f}% / 둘다아님 {(s['먼저(±5%)']=='둘다아님').mean()*100:.0f}%  ·  ±10%: 상방 {(s['먼저(±10%)']=='상방').mean()*100:.0f}% / 하방 {(s['먼저(±10%)']=='하방').mean()*100:.0f}% / 둘다아님 {(s['먼저(±10%)']=='둘다아님').mean()*100:.0f}%")
summ(E, "전체")
summ(E[pd.to_datetime(E.date) >= "2009-01-01"], "2009년 이후")
summ(E[pd.to_datetime(E.date) < "2009-01-01"], "2008년 이전(닷컴·금융위기 포함)")
summ(E[E["직전눌림%"] <= -10], "직전 눌림 -10% 이상이었던 경우")
summ(E[E["직전눌림%"] > -10], "직전 눌림 -5~-10%")
# 기준율: 아무 날
print("\n--- 기준율(모든 날, 1985~): 120일 뒤 수익 승률 %.0f%%, 120일 내 최대하락 중앙 %.1f%%" % (((c.shift(-120)/c-1)>0).mean()*100, ((c[::-1].rolling(120).min()[::-1].shift(-1)/c-1)*100).median()))

# 같은 전고점을 두 번째 이상 재도전(앞 재도전이 못 뚫고 밀린 뒤 다시 온 경우)
E["prev_hi"] = E["전고점"].shift(1)
again = E[(E["전고점"] - E["prev_hi"]).abs() / E["전고점"] < 0.005]
print("\n=== 같은 전고점 2번째 이상 재도전 (지금 상황과 유사):", len(again), "건")
print(again[["date","전고점","직전눌림%","up60","dd60","up120","dd120","ret120","먼저(±5%)","먼저(±10%)"]].round(1).to_string(index=False))
summ(again, "같은 고점 재재도전")
first = E[~E.index.isin(again.index)]
summ(first, "첫 재도전")
