"""뉴스 함의(긍정/우려 판정)와 나스닥 실제 움직임의 관계 (data/news_log.json × 나스닥 종합).

- 일별 점수 = (긍정 - 우려) / 기사 수 (발행일 KST 기준). 월별 점수도 같은 식.
- 동행: 그 날/달의 지수 수익과의 상관. 선행: 다음 날·5일·20일, 다음 달 수익과의 상관. 후행: 직전 5일 수익과의 상관 (기사는 지난 일을 쓴다).
- 백필 표본은 Google News 가 '관련도' 순으로 주는 상위 기사라 큰 뉴스가 있던 날에 치우친다는 점을 감안.
사용: uv run python scripts/news_vs_market.py
"""
import sys, json, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "/Users/dongeui/Desktop/code/tickerfinder")
import numpy as np, pandas as pd
from app.longhist import load_long
from app.universe import DATA_DIR
pd.set_option("display.width", 220)

log = json.load(open(DATA_DIR / "news_log.json"))
R = pd.DataFrame([{"date": r["published"][:10], "tone": r.get("tone") or ""} for r in log.values() if r.get("published")])
R["pos"] = (R.tone == "긍정").astype(int); R["neg"] = (R.tone == "우려").astype(int)
D = R.groupby("date").agg(n=("tone", "size"), pos=("pos", "sum"), neg=("neg", "sum")); D.index = pd.to_datetime(D.index)
D["score"] = (D.pos - D.neg) / D.n
print(f"기사 {len(R):,}건 · 날짜 {len(D)}일 ({D.index.min().date()}~{D.index.max().date()}) · 판정 분포: {R.tone.value_counts().to_dict()}")

ix = load_long("IXIC").dropna(subset=["Close"]); ix.index = pd.to_datetime(ix.index).tz_localize(None).normalize()
try:
    rec = pd.read_parquet(DATA_DIR / "prices" / "_IXIC.parquet"); rec.index = pd.to_datetime(rec.index).normalize()
    ix = pd.concat([ix, rec[rec.index > ix.index[-1]]])
except Exception:  # noqa: BLE001
    pass
c = ix["Close"]
# 날짜 → 그 날(또는 직전 거래일) 종가 위치
pos_idx = c.index.searchsorted(D.index, side="right") - 1
D = D[pos_idx >= 20].copy(); pos_idx = pos_idx[pos_idx >= 20]
def ret(k_from, k_to):
    a = np.clip(pos_idx + k_from, 0, len(c) - 1); b = np.clip(pos_idx + k_to, 0, len(c) - 1)
    r = (c.values[b] / c.values[a] - 1) * 100
    r[(pos_idx + k_to) >= len(c)] = np.nan
    return r
D["same_day"] = ret(-1, 0); D["prev5"] = ret(-5, 0); D["next1"] = ret(0, 1); D["next5"] = ret(0, 5); D["next20"] = ret(0, 20)
Dn = D[D.n >= 5]
print(f"\n[일별, 기사 5건 이상인 {len(Dn)}일] 점수 = (긍정-우려)/n")
for col, name in [("prev5", "직전 5일 수익(후행)"), ("same_day", "당일 수익(동행)"), ("next1", "다음 1일(선행)"), ("next5", "다음 5일"), ("next20", "다음 20일")]:
    x = Dn[["score", col]].dropna()
    print(f"  {name:14s} 피어슨 {x.score.corr(x[col]):+.2f}  스피어먼 {x.score.rank().corr(x[col].rank()):+.2f}  n={len(x)}")
Dn = Dn.assign(bucket=pd.cut(Dn.score, [-1.01, -0.4, 0, 1.01], labels=["우려 우세(≤-0.4)", "약우려(-0.4~0)", "긍정 우세(>0)"]))
print(Dn.groupby("bucket", observed=True).agg(n=("score", "size"), prev5=("prev5", "mean"), same=("same_day", "mean"), next1=("next1", "mean"), next5=("next5", "mean"), next5_win=("next5", lambda x: (x.dropna() > 0).mean() * 100), next20=("next20", "mean"), next20_win=("next20", lambda x: (x.dropna() > 0).mean() * 100)).round(2).to_string())

# 월별
R["month"] = R.date.str[:7]
M = R.groupby("month").agg(n=("tone", "size"), pos=("pos", "sum"), neg=("neg", "sum")); M["score"] = (M.pos - M.neg) / M.n
mc = c.resample("ME").last(); mr = (mc / mc.shift(1) - 1) * 100; mr.index = mr.index.strftime("%Y-%m")
M["same_month"] = mr.reindex(M.index); M["next_month"] = mr.shift(-1).reindex(M.index); M["prev_month"] = mr.shift(1).reindex(M.index)
M["next3"] = ((mc.shift(-3) / mc - 1) * 100).set_axis(mc.index.strftime("%Y-%m")).reindex(M.index)
Mn = M[M.n >= 20]
print(f"\n[월별, 기사 20건 이상인 {len(Mn)}개월]")
for col, name in [("prev_month", "직전 달 수익(후행)"), ("same_month", "같은 달 수익(동행)"), ("next_month", "다음 달(선행)"), ("next3", "다음 3개월")]:
    x = Mn[["score", col]].dropna()
    if len(x) >= 6:
        print(f"  {name:14s} 피어슨 {x.score.corr(x[col]):+.2f}  스피어먼 {x.score.rank().corr(x[col].rank()):+.2f}  n={len(x)}")
Mn = Mn.assign(bucket=pd.cut(Mn.score, [-1.01, -0.3, 0, 1.01], labels=["하락 우위(≤-0.3)", "혼조(-0.3~0)", "상승 우위(>0)"]))
if len(Mn) >= 6:
    print(Mn.groupby("bucket", observed=True).agg(n=("score", "size"), prev=("prev_month", "mean"), same=("same_month", "mean"), next=("next_month", "mean"), next_win=("next_month", lambda x: (x.dropna() > 0).mean() * 100), next3=("next3", "mean")).round(2).to_string())
print("\n월별 표 (점수, 같은 달 / 다음 달 나스닥 %):")
print(M.assign(score=M.score.round(2)).round(1).to_string())
