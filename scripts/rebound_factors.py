"""게이트(시장 RSI 30~40)가 열린 날 매수 조건에 진입한 종목 중, 어떤 특성이 큰 반등과 연결되는지 분석.

사용: uv run python scripts/rebound_factors.py [--gate-hi 40] [--horizon 20]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app import analysis as A  # noqa: E402
from app.scanner import load_cache  # noqa: E402
from app.universe import DATA_DIR  # noqa: E402


def features_at(df: pd.DataFrame, t: pd.Timestamp, idx: pd.DataFrame | None, band_tol: float) -> dict | None:
    d = df.loc[:t]
    if len(d) < 220 or d.index[-1] != t:
        return None
    c = d["Close"]
    v = d["Volume"].fillna(0)
    r = c.pct_change()
    ma20, ma50 = c.rolling(20).mean().iloc[-1], c.rolling(50).mean().iloc[-1]
    ma150, ma200 = c.rolling(150).mean().iloc[-1], c.rolling(200).mean().iloc[-1]
    lo, hi = min(ma150, ma200), max(ma150, ma200)
    # 밴드 체류일: 최근 며칠 연속 |band_pos|<=tol
    m150s, m200s = c.rolling(150).mean(), c.rolling(200).mean()
    los, his = np.minimum(m150s, m200s), np.maximum(m150s, m200s)
    inband = ((c >= los * (1 - band_tol / 100)) & (c <= his * (1 + band_tol / 100))).to_numpy()
    stay = 0
    for x in inband[::-1]:
        if not x:
            break
        stay += 1
    last = c.iloc[-1]
    f = {
        "ret_5d": (last / c.iloc[-6] - 1) * 100,
        "ret_1m": (last / c.iloc[-21] - 1) * 100,
        "ret_3m": (last / c.iloc[-61] - 1) * 100,
        "ret_6m": (last / c.iloc[-127] - 1) * 100,
        "from_high52": (last / c.tail(252).max() - 1) * 100,
        "dist_ma20": (last / ma20 - 1) * 100,
        "dist_ma50": (last / ma50 - 1) * 100,
        "ma_slope200": (ma200 / c.rolling(200).mean().iloc[-21] - 1) * 100,  # 200일선 기울기(1개월)
        "vol20_ann": r.tail(20).std() * np.sqrt(252) * 100,
        "vol_spike": v.tail(5).mean() / max(v.tail(60).mean(), 1),
        "turnover20_log": np.log10(max((c * v).tail(20).mean(), 1)),
        "band_stay_days": stay,
    }
    if idx is not None:
        ic = idx["Close"].reindex(c.index, method="ffill")
        f["rs_1m"] = f["ret_1m"] - (ic.iloc[-1] / ic.iloc[-21] - 1) * 100
        f["rs_3m"] = f["ret_3m"] - (ic.iloc[-1] / ic.iloc[-61] - 1) * 100
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate-lo", type=float, default=30)
    ap.add_argument("--gate-hi", type=float, default=40)
    ap.add_argument("--horizon", type=int, default=20)
    ap.add_argument("--no-gate", action="store_true", help="게이트 없이 전체 진입 이벤트로 분석")
    a = ap.parse_args()
    th = A.Thresholds(mkt_lo=a.gate_lo, mkt_hi=a.gate_hi)
    ev = A.signal_events("buy", th, require_mkt=not a.no_gate, use_kr_index=True, limit=100000)["events"]
    H = f"fwd{a.horizon}"
    ev = [e for e in ev if e.get(H) is not None]
    uni = {u["ticker"]: u for u in json.loads((DATA_DIR / "universe.json").read_text())}
    idx = {"KR": load_cache("^KS11"), "US": load_cache("^IXIC")}

    rows = []
    cache: dict[str, pd.DataFrame] = {}
    for e in ev:
        df = cache.get(e["ticker"])
        if df is None:
            df = load_cache(e["ticker"])
            cache[e["ticker"]] = df
        if df is None:
            continue
        t = pd.Timestamp(e["date"])
        f = features_at(df, t, idx.get(e["region"]), th.band_tol)
        if f is None:
            continue
        u = uni.get(e["ticker"], {})
        ix = idx[e["region"]]
        rows.append({**e, **f, "market": u.get("market", e.get("market")), "sector": u.get("sector", ""),
                     "marcap_log": np.log10(u["marcap"]) if u.get("marcap") else np.nan, "y": e[H]})
    X = pd.DataFrame(rows)
    if X.empty:
        print("no events")
        return
    # 국면(episode) 정의: 진입일 간격이 10거래일 넘게 비면 새 국면
    dates = sorted(X["date"].unique())
    ep, cur, prev = {}, 0, None
    hist_dates = [d["date"] for d in A.load_history()]
    pos = {d: i for i, d in enumerate(hist_dates)}
    for d in dates:
        if prev is not None and pos.get(d, 0) - pos.get(prev, 0) > 10:
            cur += 1
        ep[d] = cur
        prev = d
    X["episode"] = X["date"].map(ep)
    X["y_adj"] = X["y"] - X.groupby(["episode", "region"])["y"].transform("mean")  # 국면·지역 평균 제거

    print(f"이벤트 {len(X)}건, 국면 {X.episode.nunique()}개, 평균 +{a.horizon}일 {X.y.mean():+.2f}% (중앙값 {X.y.median():+.2f}%, 승률 {(X.y>0).mean()*100:.0f}%)")
    print(X.groupby("episode").agg(n=("y", "size"), start=("date", "min"), end=("date", "max"), mean=("y", "mean"), win=("y", lambda s: (s > 0).mean() * 100)).round(2).to_string())

    feats = ["rsi_d", "rsi_w", "stoch_k", "band_pos", "ret_5d", "ret_1m", "ret_3m", "ret_6m", "from_high52", "dist_ma20", "dist_ma50",
             "ma_slope200", "rs_1m", "rs_3m", "vol20_ann", "vol_spike", "turnover20_log", "marcap_log", "band_stay_days"]
    print("\n== 특성별 순위상관(Spearman) with 반등률: raw / 국면·지역 보정 ==")
    out = []
    for f in feats:
        if f not in X or X[f].notna().sum() < 50:
            continue
        s = X[[f, "y", "y_adj"]].dropna()
        out.append((f, s[f].rank().corr(s["y"].rank()), s[f].rank().corr(s["y_adj"].rank()), len(s)))
    out.sort(key=lambda r: -abs(r[2]))
    for f, r1, r2, n in out:
        print(f"{f:15s} raw {r1:+.3f}  adj {r2:+.3f}  (n={n})")

    print("\n== 상위 특성 5분위별 평균 반등(보정 전) / 승률 ==")
    for f, _, _, _ in out[:8]:
        s = X[[f, "y"]].dropna()
        try:
            q = pd.qcut(s[f], 5, labels=False, duplicates="drop")
        except ValueError:
            continue
        g = s.groupby(q).agg(lo=(f, "min"), hi=(f, "max"), n=("y", "size"), mean=("y", "mean"), win=("y", lambda v: (v > 0).mean() * 100)).round(2)
        print(f"-- {f}")
        print(g.to_string())

    print("\n== 범주별 ==")
    for col in ("region", "market", "sector"):
        if col in X and X[col].notna().any():
            g = X.groupby(col)["y"].agg(n="size", mean="mean", median="median", win=lambda v: (v > 0).mean() * 100).round(2)
            g = g[g.n >= 8].sort_values("mean", ascending=False)
            print(f"-- {col}")
            print(g.to_string())

    # 간단 조합 규칙: 상위 3개 특성 각각 '유리한 쪽 절반'에 모두 속하는 종목
    print("\n== 조합 규칙 테스트 (상위 특성 3개, 유리한 절반 교집합) ==")
    top = out[:3]
    mask = pd.Series(True, index=X.index)
    desc = []
    for f, _, r, _ in top:
        med = X[f].median()
        cond = (X[f] >= med) if r > 0 else (X[f] <= med)
        mask &= cond.fillna(False)
        desc.append(f"{f} {'>=' if r > 0 else '<='} {med:.2f}")
    sel = X[mask]
    print(" & ".join(desc))
    print(f"선택 {len(sel)}건: 평균 {sel.y.mean():+.2f}% 승률 {(sel.y>0).mean()*100:.0f}%  vs 전체 {X.y.mean():+.2f}% / {(X.y>0).mean()*100:.0f}%")
    X.to_csv(DATA_DIR / "rebound_events.csv", index=False)
    print(f"\n이벤트별 특성 저장: data/rebound_events.csv")


if __name__ == "__main__":
    main()
