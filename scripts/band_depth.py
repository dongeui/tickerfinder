"""밴드(150·200일선) 아래로 과하게 이탈한 종목을 사면 어떤가 — 조합 A(스토캐 ≤25 + 게이트, 밴드 무관) 이벤트를 밴드 위치별로 비교 (2026-10-15).
사용: uv run python scripts/band_depth.py  (data/pit_prob_events.parquet 필요: scripts/pit_prob.py --rebuild)"""
import sys; sys.path.insert(0, "/Users/dongeui/Desktop/code/tickerfinder")
import numpy as np, pandas as pd
pd.set_option("display.width", 250)
E = pd.read_parquet("data/pit_prob_events.parquet"); E["date"] = pd.to_datetime(E.date)
def tab(g):
    return g.groupby("bk", observed=True).agg(n=("fwd60","size"), win60=("fwd60", lambda x:(x>0).mean()*100), avg60=("fwd60","mean"), med60=("fwd60","median"),
        p10_60=("fwd60", lambda x:x.quantile(.1)), mdd60=("min60","mean"), win180=("fwd180", lambda x:(x.dropna()>0).mean()*100), avg180=("fwd180","mean"),
        p10_180=("fwd180", lambda x:x.dropna().quantile(.1)), win360=("fwd360", lambda x:(x.dropna()>0).mean()*100), avg360=("fwd360","mean")).round(1)
bins = [-np.inf, -15, -7, -3, -1e-9, 1e-9, 3, 7, np.inf]
labels = ["< -15%", "-15~-7%", "-7~-3%", "-3~0% (밴드 아래 조금)", "밴드 안 (0)", "0~+3%", "+3~+7%", "> +7%"]
for reg in ("US", "KR"):
    A = E[(E.region == reg) & (E.combo == "A")].dropna(subset=["fwd60"]).copy()
    A["bk"] = pd.cut(A.band_pos, bins, labels=labels)
    print(f"\n=== {reg} 조합 A(스토캐 ≤25 + 지수 RSI 게이트, 밴드 무관) {len(A):,}건 {A.date.min().date()}~{A.date.max().date()} — 밴드 위치별 ===")
    print(tab(A).to_string())
    base = A.assign(bk="전체"); print(tab(base).to_string().split("\n")[-1])
    # 과대낙폭 중에서도 200일선 기울기(상승 추세) 유무
    deep = A[A.band_pos < -3].copy()
    deep["bk"] = np.where(deep.slope200 >= 2.2, "밴드 -3% 이하 & 200일선 상승(기울기≥2.2)", "밴드 -3% 이하 & 200일선 평평/하락")
    print("--- 밴드 아래 -3% 이하를 200일선 기울기로 나누면"); print(tab(deep).to_string())
    deep["bk"] = pd.cut(deep.from_high52, [-np.inf, -40, -25, -15, 0], labels=["고점 -40% 이하", "-40~-25%", "-25~-15%", "-15~0%"])
    print("--- 밴드 아래 -3% 이하를 52주 고점 대비로 나누면"); print(tab(deep).to_string())
    # 연도별: 밴드 안 vs -3% 이하 (약세장 구분)
    A["yr"] = A.date.dt.year
    yr = A.assign(grp=np.where(A.band_pos.abs() <= 3, "밴드±3", np.where(A.band_pos < -3, "아래>3%", "위>3%"))).groupby(["yr","grp"]).fwd60.agg(["size","mean", lambda x:(x>0).mean()*100]).round(1)
    yr.columns = ["n","avg60","win60"]; print("--- 연도별 60일 (밴드±3 vs 아래>3%)"); print(yr.unstack("grp").to_string())
import warnings; warnings.filterwarnings("ignore")
# --- 밴드 안(C) 과 아래 이탈 비교, 눌림목→이탈 전환 ---
E = pd.read_parquet("data/pit_prob_events.parquet"); E["date"] = pd.to_datetime(E.date)
def st(g, label):
    f = g.dropna(subset=["fwd60"])
    return dict(bk=label, n=len(f), win60=(f.fwd60>0).mean()*100, avg60=f.fwd60.mean(), med60=f.fwd60.median(), p10_60=f.fwd60.quantile(.1), mdd60=f.min60.mean(),
                win180=(f.fwd180.dropna()>0).mean()*100, avg180=f.fwd180.mean(), p10_180=f.fwd180.dropna().quantile(.1), win360=(f.fwd360.dropna()>0).mean()*100, avg360=f.fwd360.mean())
for reg in ("US","KR"):
    X = E[E.region==reg]; A = X[X.combo=="A"]; C = X[X.combo=="C"]
    rows = [st(C, "C 밴드 ±3% 안 (현재 매수 조건)"), st(A[A.band_pos.between(-7,-3)], "A 밴드 아래 3~7% (브로드컴 지금)"), st(A[A.band_pos.between(-15,-7)], "A 아래 7~15%"), st(A[A.band_pos<-15], "A 아래 15% 초과"), st(A[A.band_pos>3], "A 밴드 위 3% 초과")]
    # 전환: 10거래일(14일) 안에 C 가 있었던 종목이 밴드 아래로 이탈한 A → '눌림목 → 이탈' 사례
    cdates = C.groupby("ticker").date.apply(lambda s: s.sort_values().values)
    def recent_c(r):
        d = cdates.get(r.ticker)
        if d is None: return False
        prev = d[(d < np.datetime64(r.date)) & (d >= np.datetime64(r.date) - np.timedelta64(14, "D"))]
        return len(prev) > 0
    Ab = A[(A.band_pos < -3)].copy(); Ab["trans"] = Ab.apply(recent_c, axis=1)
    rows.append(st(Ab[Ab.trans], "A 아래 이탈 & 2주 안에 밴드 안(C) 이었던 것 = 눌림목→이탈"))
    rows.append(st(Ab[Ab.trans & Ab.band_pos.between(-7,-3)], "  그중 이탈 3~7%"))
    rows.append(st(Ab[~Ab.trans], "A 아래 이탈 & 직전 2주 C 없음"))
    print(f"\n=== {reg} ==="); print(pd.DataFrame(rows).set_index("bk").round(1).to_string())
