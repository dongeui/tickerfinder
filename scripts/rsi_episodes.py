"""나스닥 RSI 가 45 아래로 내려온 국면: 40 까지 가는 비율, 45 진입 vs 40 대기 vs 45/40 분할의 지수 수익 비교 (20년)."""
import sys, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "/Users/dongeui/Desktop/code/tickerfinder")
import numpy as np, pandas as pd
from app.longhist import load_long
from app.indicators import rsi
pd.set_option("display.width", 250)
df = load_long("IXIC").dropna(subset=["Close"]); df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
c = df["Close"]; r = rsi(c); o = df["Open"].reindex(c.index)
print(f"나스닥 종합 {c.index[0].date()}~{c.index[-1].date()} {len(c)}일 · RSI≤45 {int((r<=45).sum())}일 · ≤40 {int((r<=40).sum())}일 · ≤35 {int((r<=35).sum())}일")
def fwd(i, h):  # i 번째 날 신호 → 다음날 시가 매수, h 거래일 뒤 종가
    if i+1 >= len(c) or i+1+h >= len(c): return np.nan
    return (c.iloc[i+1+h] / o.iloc[i+1] - 1) * 100
# 국면: RSI 가 45 이하로 처음 내려온 날 시작, 55 위로 회복하면 종료
eps = []; i = 0; n = len(c); rv = r.values
while i < n:
    if rv[i] <= 45:
        j = i; first40 = None; first35 = None; lo = rv[i]; lo_i = i
        while j < n and not rv[j] > 55:
            if first40 is None and rv[j] <= 40: first40 = j
            if first35 is None and rv[j] <= 35: first35 = j
            if rv[j] < lo: lo, lo_i = rv[j], j
            j += 1
        end = min(j, n-1)
        e = {"start": c.index[i].date(), "end": c.index[end].date(), "days": end - i, "min_rsi": round(lo, 1), "hit40": first40 is not None, "hit35": first35 is not None,
             "d45_to_40": (first40 - i) if first40 is not None else None, "drop45_40": (c.iloc[first40] / c.iloc[i] - 1) * 100 if first40 is not None else None,
             "drawdown_after45": (c.iloc[i:end+1].min() / c.iloc[i] - 1) * 100}
        for h in (20, 60, 120): e[f"f45_{h}"] = fwd(i, h); e[f"f40_{h}"] = fwd(first40, h) if first40 is not None else np.nan
        eps.append(e); i = j
    else: i += 1
E = pd.DataFrame(eps)
print(f"\n[국면] RSI≤45 진입 국면 {len(E)}번(55 회복까지 한 국면) · 그중 40 도달 {int(E.hit40.sum())}번({E.hit40.mean()*100:.0f}%) · 35 도달 {int(E.hit35.sum())}번({E.hit35.mean()*100:.0f}%)")
h40 = E[E.hit40]; print(f"  40 도달한 국면: 45→40 까지 중앙 {h40.d45_to_40.median():.0f}거래일(최대 {h40.d45_to_40.max():.0f}), 그 사이 지수 하락 중앙 {h40.drop45_40.median():+.1f}% (평균 {h40.drop45_40.mean():+.1f}%)")
print(f"  40 못 간 국면(반등): 45 진입 뒤 최대낙폭 중앙 {E[~E.hit40].drawdown_after45.median():+.1f}%, 60일 뒤 지수 중앙 {E[~E.hit40].f45_60.median():+.1f}% / 120일 {E[~E.hit40].f45_120.median():+.1f}%")
def stat(x): x = x.dropna(); return f"n={len(x)} 승률 {(x>0).mean()*100:.0f}% 평균 {x.mean():+.1f}% 중앙 {x.median():+.1f}% 하위10% {x.quantile(.1):+.1f}%"
print("\n[지수 수익, 다음날 시가 매수] 국면별 첫 신호일 기준")
for h in (20, 60, 120):
    print(f"  {h:>3}일 | 45 첫날 전부 매수 (전 국면)      : {stat(E[f'f45_{h}'])}")
    print(f"  {h:>3}일 | 40 첫날 전부 매수 (40 도달 국면만): {stat(E[f'f40_{h}'])}")
    half = E.apply(lambda e: 0.5*e[f"f45_{h}"] + 0.5*e[f"f40_{h}"] if e.hit40 else e[f"f45_{h}"], axis=1)  # 분할: 45 에서 절반, 40 오면 나머지 (안 오면 절반만 투입)
    print(f"  {h:>3}일 | 45 절반 + 40 절반 분할 (투입자본 기준): {stat(half)}")
    # 자본 전체 기준: 40 안 오면 절반은 현금(0%)
    half_cap = E.apply(lambda e: 0.5*e[f"f45_{h}"] + 0.5*e[f"f40_{h}"] if e.hit40 else 0.5*e[f"f45_{h}"], axis=1)
    print(f"  {h:>3}일 | 같은 분할, 총자본 기준(미투입 절반=0%) : {stat(half_cap)}")
print("\n[기회 횟수] 연도별 국면 수 (45 진입 / 40 도달)")
E["yr"] = pd.to_datetime(E.start).dt.year
print(E.groupby("yr").agg(n45=("start","size"), n40=("hit40","sum")).T.to_string())
print("\n[최근 12 국면]")
print(E.tail(12)[["start","end","days","min_rsi","hit40","d45_to_40","drop45_40","drawdown_after45","f45_60","f40_60","f45_120","f40_120"]].round(1).to_string(index=False))
print("\n현재 RSI", round(float(r.iloc[-1]),1), c.index[-1].date(), "| 이번 국면 시작:", E.iloc[-1].start if E.iloc[-1].end >= c.index[-1].date() - pd.Timedelta(days=3) else "(현재 국면 아님)")
