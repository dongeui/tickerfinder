"""매수 진입 이벤트(시장 게이트 + 주봉 스토캐스틱 + 150~200일선 밴드)에서 '추세'와 '추천확률'의 연관성 분석 (미국, 약 20년).

- 이벤트: app/backtest.py 의 _events_for (전날 미충족 → 당일 충족, 다음날 시가 매수) 를 그대로 재사용
- 추세 특성(이벤트 당일 종가 기준): 200일선 기울기(20일), 종가/200일선 괴리, 50일선/200일선(골든·데드), 6개월 수익률, 52주 고점 대비 낙폭
- 추천확률: (1) 종목 전체 값 = backtest.py 캐시(app 과 동일 값, 미래 이벤트 포함 → 룩어헤드)
            (2) 시점 기준(PIT) 값 = 이벤트일 기준 360거래일 이전에 끝난 과거 이벤트만으로 같은 공식(수축 k=8) 계산
- 결과: 추세 구간별 성과, 확률 구간 × 추세 구간 성과 → data/trend_vs_prob.csv

사용: uv run python scripts/trend_vs_prob.py [--gate-lo 30 --gate-hi 40 --band-tol 3 --stoch-max 25]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app import backtest as B  # noqa: E402
from app.analysis import Thresholds  # noqa: E402
from app.longhist import load_long  # noqa: E402
from app.universe import DATA_DIR  # noqa: E402

H = B.HORIZONS
PIT_GAP_DAYS = 540  # 360거래일 ≈ 1.45년: 이 만큼 이전 이벤트만 '결과를 아는' 이벤트로 본다
FEATS = {
    "slope200": "200일선 기울기(20일,%)",
    "dist200": "종가/200일선 괴리(%)",
    "gc50_200": "50일선/200일선(%)",
    "ret6m": "6개월 수익률(%)",
    "from_high52": "52주 고점 대비(%)",
}
PROB_BINS = [(0, 55, "<55"), (55, 65, "55~65"), (65, 75, "65~75"), (75, 101, "75+")]
MIN_N = 100


def features(df: pd.DataFrame) -> pd.DataFrame:
    c = df["Close"].dropna()
    ma50, ma200 = c.rolling(50).mean(), c.rolling(200).mean()
    return pd.DataFrame({
        "slope200": (ma200 / ma200.shift(20) - 1) * 100,
        "dist200": (c / ma200 - 1) * 100,
        "gc50_200": (ma50 / ma200 - 1) * 100,
        "ret6m": (c / c.shift(126) - 1) * 100,
        "from_high52": (c / c.rolling(252).max() - 1) * 100,
    })


def pit_prob(evs: list[dict], base: dict) -> list[tuple[float | None, int]]:
    """각 이벤트 시점에서 결과를 알 수 있는 과거 이벤트만으로 계산한 추천확률과 그 표본 수."""
    out = []
    dates = [pd.Timestamp(e["date"]) for e in evs]
    for i, d in enumerate(dates):
        prior = [evs[j] for j in range(i) if (d - dates[j]).days >= PIT_GAP_DAYS]
        p, _ = B._prob(B._hstats(prior) if prior else {}, base)
        out.append((p, len(prior)))
    return out


def tercile(s: pd.Series, labels: tuple[str, str, str]) -> tuple[pd.Series, list[float]]:
    q = s.quantile([1 / 3, 2 / 3]).tolist()
    return pd.cut(s, [-np.inf, q[0], q[1], np.inf], labels=list(labels)), q


def perf(g: pd.DataFrame) -> dict:
    r = {"n": len(g)}
    for h in H:
        f = g[f"fwd{h}"].dropna()
        m = g.loc[f.index, f"max{h}"]
        if not len(f):
            continue
        r[f"n{h}"] = len(f)
        r[f"win{h}"] = (f > 0).mean() * 100
        r[f"reb{h}"] = (m >= B.REBOUND_PCT).mean() * 100
        r[f"mean{h}"] = f.mean()
        r[f"med{h}"] = f.median()
        r[f"adj{h}"] = g.loc[f.index, f"adj{h}"].mean()
    return r


def fmt_row(label: str, r: dict, hs=(20, 60, 90, 180, 360)) -> str:
    flag = "*" if r["n"] < MIN_N else " "
    cells = []
    for h in hs:
        if f"win{h}" in r:
            cells.append(f"{r[f'win{h}']:3.0f}/{r[f'reb{h}']:3.0f}/{r[f'mean{h}']:+5.1f}/{r[f'med{h}']:+5.1f}")
        else:
            cells.append("        -        ")
    return f"{label:<22}{r['n']:>6}{flag} | " + " | ".join(cells)


def diff_p(a: pd.Series, b: pd.Series) -> tuple[float, float]:
    """평균 차이와 Welch z-검정 양측 p값 (대표본 정규근사)."""
    a, b = a.dropna().to_numpy(float), b.dropna().to_numpy(float)
    if len(a) < 2 or len(b) < 2:
        return float("nan"), float("nan")
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    z = (a.mean() - b.mean()) / se if se else 0.0
    return float(a.mean() - b.mean()), 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))


def header(hs=(20, 60, 90, 180, 360)) -> str:
    return f"{'구간':<22}{'건수':>6}  | " + " | ".join(f"{h:>3}일 승률/반등/평균/중앙" for h in hs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate-lo", type=float, default=Thresholds.mkt_lo)
    ap.add_argument("--gate-hi", type=float, default=Thresholds.mkt_hi)
    ap.add_argument("--band-tol", type=float, default=Thresholds.band_tol)
    ap.add_argument("--stoch-max", type=float, default=Thresholds.stoch_max)
    a = ap.parse_args()
    t0 = time.time()
    th = Thresholds(stoch_max=float(a.stoch_max), band_tol=float(a.band_tol), mkt_lo=float(a.gate_lo), mkt_hi=float(a.gate_hi))  # 앱과 같은 캐시 키(float)
    print(f"임계값: 스토캐스틱≤{th.stoch_max:g}(미국 주봉) 밴드±{th.band_tol:g}% 게이트 나스닥RSI {th.mkt_lo:g}~{th.mkt_hi:g}, 매수=다음날 시가")

    # 종목별 추천확률: 앱과 동일한 backtest.compute (캐시 있으면 즉시, 없으면 전체 계산)
    res = B.compute(th, use_kr_index=True, entry="next_open")
    base_us = res["base"]["US"]
    print(f"백테스트 캐시 {res['key']} ({res['computed_at']}), 미국 기준 추천확률 {base_us['prob']}%  [{time.time() - t0:.0f}s]", file=sys.stderr)

    gate = B._gate_series(th, use_kr_index=True)
    uni = [u for u in json.loads((DATA_DIR / "universe.json").read_text()) if u["region"] == "US"]
    rows = []
    for i, u in enumerate(uni):
        t = u["ticker"]
        evs = B._events_for(t, "US", th, gate, "next_open")
        if not evs:
            continue
        df = load_long(t)
        ft = features(df)
        tp = res["tickers"].get(t, {})
        pits = pit_prob(evs, base_us)
        for e, (pp, pn) in zip(evs, pits):
            d = pd.Timestamp(e["date"])
            rows.append({**e, "ticker": t, "name": u.get("name", ""), **ft.loc[d].to_dict(),
                         "prob_full": tp.get("prob"), "n_full": tp.get("n"), "prob_pit": pp, "n_pit": pn})
        if (i + 1) % 100 == 0:
            print(f"... {i + 1}/{len(uni)} 종목, 이벤트 {len(rows)}건 [{time.time() - t0:.0f}s]", file=sys.stderr)
    X = pd.DataFrame(rows)
    X["date"] = pd.to_datetime(X["date"])
    X["ym"] = X["date"].dt.to_period("M")
    for h in H:  # 같은 달 진입 이벤트들의 평균을 뺀 값 (시장 국면 효과 제거)
        X[f"adj{h}"] = X[f"fwd{h}"] - X.groupby("ym")[f"fwd{h}"].transform("mean")
    print(f"미국 {X.ticker.nunique()}종목, 진입 이벤트 {len(X)}건 ({X.date.min().date()} ~ {X.date.max().date()})  [{time.time() - t0:.0f}s]")

    # 추세 구간 (3분위) + 부호 구간
    bins: dict[str, pd.Series] = {}
    edges: dict[str, list[float]] = {}
    for f in FEATS:
        bins[f], edges[f] = tercile(X[f], ("하위", "중위", "상위"))
    bins["slope200_sign"] = pd.Series(np.where(X["slope200"] >= 0, "상승(≥0)", "하락(<0)"), index=X.index)
    bins["gc_sign"] = pd.Series(np.where(X["gc50_200"] >= 0, "골든(50≥200)", "데드(50<200)"), index=X.index)
    bins["above200"] = pd.Series(np.where(X["dist200"] >= 0, "200일선 위", "200일선 아래"), index=X.index)
    score = ((X["slope200"] >= 0).astype(int) + (X["gc50_200"] >= 0).astype(int) + (X["ret6m"] >= 0).astype(int) + (X["from_high52"] >= -15).astype(int))
    X["trend_score"] = score
    bins["trend_score"] = score.map({0: "0(약세)", 1: "1", 2: "2", 3: "3", 4: "4(강세)"})
    order = {"slope200_sign": ["하락(<0)", "상승(≥0)"], "gc_sign": ["데드(50<200)", "골든(50≥200)"], "above200": ["200일선 아래", "200일선 위"],
             "trend_score": ["0(약세)", "1", "2", "3", "4(강세)"]}
    names = dict(FEATS, slope200_sign="200일선 기울기 부호", gc_sign="50일선 vs 200일선", above200="종가 vs 200일선", trend_score="추세점수(0~4)")

    out_rows = []

    def emit(table: str, prob_bin: str, feat: str, label: str, r: dict):
        out_rows.append({"table": table, "prob_bin": prob_bin, "feature": feat, "bin": label, "low_conf": r["n"] < MIN_N, **{k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()}})

    print("\n==== 1) 추세 구간별 성과 (전체 이벤트) — 승률%/반등확률%(+10%)/평균%/중앙% ; * = 표본<100 ====")
    print(f"전체: {fmt_row('ALL', perf(X))}")
    print(header())
    for f, b in bins.items():
        labs = order.get(f) or ["하위", "중위", "상위"]
        e = edges.get(f)
        print(f"-- {names[f]}" + (f"  (3분위 경계 {e[0]:+.1f} / {e[1]:+.1f})" if e else ""))
        for lab in labs:
            g = X[b == lab]
            if not len(g):
                continue
            r = perf(g)
            emit("trend", "ALL", f, lab, r)
            print(fmt_row(f"  {lab}", r))

    # 국면 보정 (같은 달 평균 제거) — 추세 구간별 60/180일 초과 수익
    print("\n==== 1b) 국면 보정(같은 달 이벤트 평균 대비 초과수익 %p): 60일 / 180일 ====")
    for f in ("slope200", "gc50_200", "ret6m", "from_high52", "trend_score"):
        labs = order.get(f) or ["하위", "중위", "상위"]
        cells = []
        for lab in labs:
            g = X[bins[f] == lab]
            if len(g):
                cells.append(f"{lab} {g['adj60'].mean():+.1f}/{g['adj180'].mean():+.1f}(n={len(g)})")
        print(f"{names[f]:<20}: " + "  ".join(cells))

    # 확률 구간 × 추세
    for which, col in (("full", "prob_full"), ("pit", "prob_pit")):
        title = "종목 전체 추천확률(앱 값, 룩어헤드 포함)" if which == "full" else f"시점 기준 추천확률(이벤트일 {PIT_GAP_DAYS}일 이전 이벤트만)"
        print(f"\n==== 2{'a' if which == 'full' else 'b'}) {title} 구간 × 추세 구간 ====")
        pb = pd.cut(X[col], [b[0] for b in PROB_BINS] + [101], labels=[b[2] for b in PROB_BINS], right=False)
        X[f"pbin_{which}"] = pb
        print("확률 구간별 분포: " + ", ".join(f"{k}: {v}건" for k, v in pb.value_counts().sort_index().items()))
        if which == "pit":
            print(f"(PIT 과거 이벤트 0건=기준율 그대로인 이벤트 {int((X['n_pit'] == 0).sum())}건, 과거 이벤트 중앙값 {X['n_pit'].median():.0f}건)")
        print(header(hs=(20, 60, 180, 360)))
        for f in ("slope200_sign", "gc_sign", "slope200", "ret6m", "from_high52", "trend_score"):
            labs = order.get(f) or ["하위", "중위", "상위"]
            print(f"-- {names[f]}")
            for _, _, pl in PROB_BINS:
                sub = X[pb == pl]
                if not len(sub):
                    continue
                r_all = perf(sub)
                emit(f"prob_{which}", pl, "ALL", "ALL", r_all)
                print(fmt_row(f"  확률 {pl} 전체", r_all, hs=(20, 60, 180, 360)))
                for lab in labs:
                    g = sub[bins[f].loc[sub.index] == lab]
                    if not len(g):
                        continue
                    r = perf(g)
                    emit(f"prob_{which}", pl, f, lab, r)
                    print(fmt_row(f"      {lab}", r, hs=(20, 60, 180, 360)))

        print(f"  [요약] 확률 구간별 200일선 기울기 상위−하위 3분위 차이 (평균 %p, p값 / 국면보정 %p / 승률 %p / 반등 %p)")
        for _, _, pl in PROB_BINS:
            sub = X[pb == pl]
            hi_, lo_ = sub[bins["slope200"].loc[sub.index] == "상위"], sub[bins["slope200"].loc[sub.index] == "하위"]
            if len(hi_) < 2 or len(lo_) < 2:
                continue
            cells = []
            for h in (60, 180):
                d, p = diff_p(hi_[f"fwd{h}"], lo_[f"fwd{h}"])
                da, _ = diff_p(hi_[f"adj{h}"], lo_[f"adj{h}"])
                dw = (hi_[f"fwd{h}"] > 0).mean() * 100 - (lo_[f"fwd{h}"] > 0).mean() * 100
                dr = (hi_[f"max{h}"] >= B.REBOUND_PCT).mean() * 100 - (lo_[f"max{h}"] >= B.REBOUND_PCT).mean() * 100
                cells.append(f"{h}일 {d:+.1f}(p={p:.3f}) / {da:+.1f} / {dw:+.1f} / {dr:+.1f}")
                emit(f"prob_{which}_slopediff", pl, "slope200 상위-하위", f"{h}", {"n": min(len(hi_), len(lo_)), f"diff{h}": d, f"p{h}": p, f"adjdiff{h}": da, f"windiff{h}": dw, f"rebdiff{h}": dr})
            flag = "*" if min(len(hi_), len(lo_)) < MIN_N else ""
            print(f"    확률 {pl:<6} n={len(hi_)}/{len(lo_)}{flag}: " + " | ".join(cells))

    print("\n==== 2c) 200일선 기울기 3분위 × 52주 고점 대비 3분위 (전체 이벤트) ====")
    print(header(hs=(60, 180, 360)))
    for sl in ("하위", "중위", "상위"):
        for fh in ("하위", "중위", "상위"):
            g = X[(bins["slope200"] == sl) & (bins["from_high52"] == fh)]
            if not len(g):
                continue
            r = perf(g)
            emit("slope_x_drawdown", "ALL", "slope200×from_high52", f"기울기{sl}×낙폭{'큼' if fh == '하위' else '중간' if fh == '중위' else '작음'}", r)
            print(fmt_row(f"  기울기{sl} × 낙폭{'큼' if fh == '하위' else '중간' if fh == '중위' else '작음'}", r, hs=(60, 180, 360)))

    # 추천확률과 추세 특성의 상관 (종목 확률이 특정 추세에 몰려 있는지)
    print("\n==== 3) 추천확률 ↔ 추세 특성 순위상관(Spearman) ====")
    for col in ("prob_full", "prob_pit"):
        s = X[[col] + list(FEATS)].dropna()
        print(f"{col:10}: " + "  ".join(f"{f} {s[col].rank().corr(s[f].rank()):+.3f}" for f in FEATS))
    print("fwd60 ↔ : " + "  ".join(f"{f} {X['fwd60'].rank().corr(X[f].rank()):+.3f}" for f in list(FEATS) + ["prob_full", "prob_pit"]))

    pd.DataFrame(out_rows).to_csv(DATA_DIR / "trend_vs_prob.csv", index=False)
    X.drop(columns=["ym"]).to_csv(DATA_DIR / "trend_vs_prob_events.csv", index=False)
    print(f"\n저장: data/trend_vs_prob.csv (요약 {len(out_rows)}행), data/trend_vs_prob_events.csv (이벤트 {len(X)}건)  총 {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
