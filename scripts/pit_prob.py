"""룩어헤드 없는 '보수적 추천확률' 모델 설계·검증 (미국 중심, 국내 요약).

1) 조건 조합 백테스트 (시장 게이트는 항상): A=스토캐스틱만, B=밴드만, C=둘 다 (C 는 app/backtest.py 캐시와 일치 확인)
2) 시점 기준(PIT, walk-forward) 확률: 이벤트일 − PIT_GAP 일 이전에 끝난 이벤트만으로 종목/지역/셀 통계
3) 모델 비교: M0 현재(종목 전체값, 룩어헤드) / M1 지역×조합 기준율 / M2 M1+추세 셀(Wilson 하한, 수축) / M3 M2+종목 효과(큰 k)
4) 최종 모델 → data/prob_model.json, 오늘 스캔 종목 예시

이벤트 생성은 backtest._events_for 와 동일(전날 미충족 → 당일 충족, 다음날 시가 매수)하되 조건 조합만 바꾼다.
산출물: data/pit_prob_events.parquet (이벤트 캐시), data/pit_prob_scores.parquet (이벤트별 모델 점수),
        data/pit_prob_summary.csv (표), data/prob_model.json (앱용 모델)

사용: uv run python scripts/pit_prob.py [--rebuild] [--limit N] [--z 1.28] [--k-cell 50] [--k-ticker 40]
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
from app.analysis import Thresholds, buy_hit  # noqa: E402
from app.backfill import compute_series  # noqa: E402
from app.indicators import stoch_point_in_time  # noqa: E402
from app.longhist import load_long  # noqa: E402
from app.universe import DATA_DIR  # noqa: E402

H = B.HORIZONS
W = B.WEIGHTS
PIT_GAP_DAYS = 540  # 360거래일 ≈ 1.45년: 이 만큼 이전 이벤트만 '결과를 아는' 이벤트로 본다 (trend_vs_prob 과 동일)
COMBOS = ("A", "B", "C")
COMBO_NAME = {"A": "스토캐스틱만", "B": "밴드만", "C": "둘 다(앱 규칙)"}
SLOPE_LAB = ("lo", "mid", "hi")          # 200일선 기울기 3분위
DD_LAB = ("deep", "mid", "shallow")      # 52주 고점 대비 낙폭 3분위 (deep = 많이 빠짐)
EVAL_HS = (20, 60, 180)
MIN_BASE_N = 100  # 지역×조합 PIT 기준율 표본이 이보다 적으면 평가에서 제외(초기 burn-in)
EV_PATH = DATA_DIR / "pit_prob_events.parquet"
SC_PATH = DATA_DIR / "pit_prob_scores.parquet"
SUM_PATH = DATA_DIR / "pit_prob_summary.csv"
MODEL_PATH = DATA_DIR / "prob_model.json"


# ---------------------------------------------------------------- 1) 이벤트 생성 (조합별)
def events_combo(ticker: str, region: str, th: Thresholds, gate: dict[str, pd.Series], entry: str = "next_open") -> pd.DataFrame | None:
    """backtest._events_for 와 같은 로직, 조건 조합(A/B/C)별 진입 이벤트 + 추세 특성."""
    df = load_long(ticker)
    if df is None or len(df) < 260:
        return None
    ser = compute_series(df)
    c = ser["close"]
    if th.stoch_tf(region) == "w":
        ser["stoch_k"] = ser["stoch_k_w"]
    elif B.TF_FREQ.get(th.stoch_tf(region)):
        ser["stoch_k"] = stoch_point_in_time(df.dropna(subset=["Close"]), B.TF_FREQ[th.stoch_tf(region)])
    g = gate[region].reindex(c.index, method="ffill").fillna(False).astype(bool)
    ok = g & ~ser["suspect"].astype(bool) & ser["stoch_k"].notna() & ser["band_pos"].notna()
    st = (ser["stoch_k"] <= th.stoch_max).fillna(False)
    bd = (ser["band_pos"].abs() <= th.band_tol).fillna(False)
    hits = {"A": ok & st & ~bd, "B": ok & bd & ~st, "C": ok & st & bd}
    o = df["Open"].reindex(c.index) if "Open" in df else None
    fw = None
    ma200 = c.rolling(200).mean()
    slope = (ma200 / ma200.shift(20) - 1) * 100
    dd = (c / c.rolling(252).max() - 1) * 100
    rows = []
    for combo, hit in hits.items():
        entry_days = hit & ~hit.shift(1, fill_value=False)
        if not entry_days.any():
            continue
        if fw is None:
            fw = B._fwd_table(c, o, entry)
        idx = c.index[entry_days.to_numpy()]
        d = {"ticker": ticker, "region": region, "combo": combo, "date": idx, "close": c[idx].to_numpy(), "entry_px": fw["entry_px"][idx].to_numpy(),
             "stoch_k": ser["stoch_k"][idx].to_numpy(), "band_pos": ser["band_pos"][idx].to_numpy(),
             "slope200": slope[idx].to_numpy(), "from_high52": dd[idx].to_numpy()}
        for h in H:
            for k in ("fwd", "max", "min"):
                d[f"{k}{h}"] = fw[f"{k}{h}"][idx].to_numpy()
        rows.append(pd.DataFrame(d))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def build_events(th: Thresholds, limit: int | None) -> pd.DataFrame:
    gate = B._gate_series(th, use_kr_index=True)
    uni = json.loads((DATA_DIR / "universe.json").read_text())
    if limit:
        uni = [u for r in ("US", "KR") for u in [x for x in uni if x["region"] == r][:limit]]
    t0 = time.time()
    parts = []
    for i, u in enumerate(uni):
        ev = events_combo(u["ticker"], u["region"], th, gate)
        if ev is not None and len(ev):
            parts.append(ev)
        if (i + 1) % 200 == 0:
            print(f"... 이벤트 생성 {i + 1}/{len(uni)} [{time.time() - t0:.0f}s]", file=sys.stderr)
    X = pd.concat(parts, ignore_index=True)
    X["date"] = pd.to_datetime(X["date"])
    for h in H:  # 수치는 캐시와 같은 소수 2자리로 (일치 확인용)
        for k in ("fwd", "max", "min"):
            X[f"{k}{h}"] = X[f"{k}{h}"].round(2)
    return X


def verify_against_cache(X: pd.DataFrame, th: Thresholds) -> None:
    """조합 C 가 backtest.compute 캐시(앱 값)와 같은지: 이벤트 수, 지역 기준율, 몇 종목의 이벤트 날짜/수익률."""
    res = B.cached(th, True, "next_open")
    if res is None:
        print("!! 기본 조합 캐시 없음 — 일치 확인 생략", file=sys.stderr)
        return
    C = X[X.combo == "C"]
    ok = True
    for reg in ("US", "KR"):
        sub = C[C.region == reg]
        n_cache = res["base"][reg]["20"]["n"]
        n_mine = int(sub["fwd20"].notna().sum())
        hs = B._hstats(records(sub))
        p_mine, _ = B._prob({}, hs)
        p_cache = res["base"][reg]["prob"]
        flag = "OK" if (n_cache == n_mine and abs(p_mine - p_cache) < 0.051) else "MISMATCH"
        ok &= flag == "OK"
        print(f"[일치 확인 {reg}] 20일 이벤트 수 캐시 {n_cache} vs 스크립트 {n_mine}; 기준 추천확률 캐시 {p_cache} vs 스크립트 {p_mine}  → {flag}")
    bad = 0
    for t, v in list(res["tickers"].items()):
        mine = C[C.ticker == t]
        if v["n"] != len(mine):
            bad += 1
            continue
        ev = v["events"][-40:]
        m = mine.tail(len(ev))
        if [e["date"] for e in ev] != [d.strftime("%Y-%m-%d") for d in m["date"]]:
            bad += 1
            continue
        for e, (_, r) in zip(ev, m.iterrows()):
            for h in H:
                a, b = e.get(f"fwd{h}"), r[f"fwd{h}"]
                if (a is None) != (pd.isna(b)) or (a is not None and abs(a - b) > 0.011):
                    bad += 1
                    break
            else:
                continue
            break
    print(f"[일치 확인 종목] 캐시 {len(res['tickers'])}종목 중 이벤트 수/날짜/수익률 불일치 {bad}종목 → {'OK' if bad == 0 else 'MISMATCH'}")
    if not ok or bad:
        print("!! 기본 조합(C) 결과가 캐시와 다릅니다. 이벤트 생성 로직 확인 필요", file=sys.stderr)


# ---------------------------------------------------------------- 공통 통계
def records(g: pd.DataFrame) -> list[dict]:
    """backtest._hstats 용 dict 목록 (NaN → None)."""
    cols = [f"{k}{h}" for h in H for k in ("fwd", "max", "min")]
    return [{k: (None if pd.isna(v) else float(v)) for k, v in zip(cols, row)} for row in g[cols].to_numpy()]


def perf(g: pd.DataFrame) -> dict:
    r = {"n": len(g)}
    for h in H:
        f = g[f"fwd{h}"].dropna()
        if not len(f):
            continue
        m = g.loc[f.index, f"max{h}"]
        r[f"n{h}"] = len(f)
        r[f"win{h}"] = (f > 0).mean() * 100
        r[f"reb{h}"] = (m >= B.REBOUND_PCT).mean() * 100
        r[f"mean{h}"] = f.mean()
        r[f"med{h}"] = f.median()
        r[f"p10_{h}"] = f.quantile(0.1)
    return r


def target(g: pd.DataFrame) -> pd.Series:
    """모델이 예측하려는 값: Σ_h w_h (0.6·승 + 0.4·반등) ×100, 결과가 있는 구간만 가중 재정규화."""
    acc, tw = pd.Series(0.0, index=g.index), pd.Series(0.0, index=g.index)
    for h in H:
        f = g[f"fwd{h}"]
        ok = f.notna()
        v = 0.6 * (f > 0).astype(float) + 0.4 * (g[f"max{h}"] >= B.REBOUND_PCT).astype(float)
        acc[ok] += W[h] * v[ok]
        tw[ok] += W[h]
    return (acc / tw.replace(0, np.nan)) * 100


def wilson_lb(p: np.ndarray, n: np.ndarray, z: float) -> np.ndarray:
    """비율 p(0~1), 표본 n 의 Wilson 신뢰구간 하한. z=0 이면 p 그대로. n=0 이면 nan."""
    p = np.asarray(p, float)
    n = np.asarray(n, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        if z == 0:
            return np.where(n > 0, p, np.nan)
        den = 1 + z * z / n
        cen = p + z * z / (2 * n)
        rad = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
        return np.where(n > 0, (cen - rad) / den, np.nan)


class PitStats:
    """그룹(지역×조합[×셀|×종목])별로 정렬된 이벤트의 누적합을 만들어, 질의 시점 이전(결과 확정) 이벤트의 구간별 n/승/반등을 돌려준다."""

    def __init__(self, X: pd.DataFrame, keys: list[str]):
        self.keys = keys
        self.groups: dict = {}
        for k, g in X.groupby(keys, sort=False, observed=True):
            g = g.sort_values("date")
            d = {"dates": g["date"].to_numpy()}
            for h in H:
                f = g[f"fwd{h}"].to_numpy(float)
                m = g[f"max{h}"].to_numpy(float)
                okh = ~np.isnan(f)
                d[f"n{h}"] = np.concatenate([[0], np.cumsum(okh)])
                d[f"w{h}"] = np.concatenate([[0], np.cumsum(okh & (f > 0))])
                d[f"r{h}"] = np.concatenate([[0], np.cumsum(okh & (m >= B.REBOUND_PCT))])
            self.groups[k if isinstance(k, tuple) else (k,)] = d

    def query(self, X: pd.DataFrame, gap_days: int) -> dict[str, np.ndarray]:
        """X 의 각 행(같은 키의 그룹)에 대해 date − gap 이전 이벤트의 구간별 n/승/반등."""
        out = {f"{k}{h}": np.zeros(len(X)) for h in H for k in ("n", "w", "r")}
        cutoff = (X["date"] - pd.Timedelta(days=gap_days)).to_numpy()
        pos = np.arange(len(X))
        for k, idx in X.groupby(self.keys, sort=False, observed=True).indices.items():
            g = self.groups.get(k if isinstance(k, tuple) else (k,))
            if g is None:
                continue
            j = np.searchsorted(g["dates"], cutoff[idx], side="right")
            for h in H:
                for kk in ("n", "w", "r"):
                    out[f"{kk}{h}"][pos[idx]] = g[f"{kk}{h}"][j]
        return out


def composite(st: dict[str, np.ndarray], z: float, deff: float, prior: dict[str, np.ndarray] | None, k: float) -> tuple[np.ndarray, np.ndarray]:
    """구간별 p_h = 0.6·승률 + 0.4·반등확률 → Wilson 하한(z, 유효표본 n/deff) → prior 쪽으로 수축(k) → 가중평균. 반환 (확률%, 표본수)."""
    acc = np.zeros(len(st["n20"]))
    tw = np.zeros(len(acc))
    for h in H:
        n = st[f"n{h}"]
        with np.errstate(divide="ignore", invalid="ignore"):
            p = np.where(n > 0, (0.6 * st[f"w{h}"] + 0.4 * st[f"r{h}"]) / np.maximum(n, 1), np.nan)
        lb = wilson_lb(p, n / deff, z)
        if prior is None:
            ph = lb
        else:
            p0 = prior[f"p{h}"]
            ne = n / deff
            ph = np.where(n > 0, (ne * np.nan_to_num(lb) + k * p0) / (ne + k), p0)
        ok = ~np.isnan(ph)
        acc[ok] += W[h] * ph[ok]
        tw[ok] += W[h]
    with np.errstate(divide="ignore", invalid="ignore"):
        prob = np.where(tw > 0, acc / tw, np.nan) * 100
    return prob, st["n20"]


def horizon_probs(st: dict[str, np.ndarray], z: float, deff: float) -> dict[str, np.ndarray]:
    out = {}
    for h in H:
        n = st[f"n{h}"]
        with np.errstate(divide="ignore", invalid="ignore"):
            p = np.where(n > 0, (0.6 * st[f"w{h}"] + 0.4 * st[f"r{h}"]) / np.maximum(n, 1), np.nan)
        out[f"p{h}"] = wilson_lb(p, n / deff, z)
    return out


def design_effect(X: pd.DataFrame, h: int = 60) -> float:
    """같은 달 이벤트 묶음(클러스터)의 설계효과 1+(m−1)ρ. ρ = 월 클러스터 ANOVA 급내상관(승패 지표)."""
    g = X.dropna(subset=[f"fwd{h}"]).copy()
    g["y"] = (g[f"fwd{h}"] > 0).astype(float)
    g["ym"] = g["date"].dt.to_period("M")
    grp = g.groupby("ym")["y"]
    m = grp.size()
    if len(m) < 2:
        return 1.0
    gm = g["y"].mean()
    ssb = float((m * (grp.mean() - gm) ** 2).sum())
    ssw = float(((g["y"] - grp.transform("mean")) ** 2).sum())
    k = len(m)
    n = len(g)
    msb, msw = ssb / (k - 1), ssw / max(n - k, 1)
    m0 = (n - (m ** 2).sum() / n) / (k - 1)
    rho = max(0.0, (msb - msw) / (msb + (m0 - 1) * msw)) if (msb + (m0 - 1) * msw) > 0 else 0.0
    return float(1 + (m.mean() - 1) * rho)


# ---------------------------------------------------------------- 표 출력
def fmt_perf(label: str, r: dict, hs=H) -> str:
    cells = []
    for h in hs:
        if f"win{h}" in r:
            cells.append(f"{r[f'win{h}']:3.0f}/{r[f'reb{h}']:3.0f}/{r[f'mean{h}']:+5.1f}/{r[f'med{h}']:+5.1f}/{r[f'p10_{h}']:+5.1f}")
        else:
            cells.append("           -            ")
    return f"{label:<26}{r['n']:>6} | " + " | ".join(cells)


def perf_header(hs=H) -> str:
    return f"{'구간':<26}{'건수':>6} | " + " | ".join(f"{h:>3}일 승률/반등/평균/중앙/하위10%" for h in hs)


def spearman(a: pd.Series, b: pd.Series) -> float:
    s = pd.concat([a, b], axis=1).dropna()
    return float(s.iloc[:, 0].rank().corr(s.iloc[:, 1].rank())) if len(s) > 10 else float("nan")


def spread(g: pd.DataFrame, score: str, h: int, q: float = 0.2) -> dict:
    """상위 q 분위 vs 하위 q 분위 점수 집단의 승률·평균·반등 차이 (%p)."""
    s = g[score].dropna()
    if len(s) < 20:
        return {}
    lo, hi = s.quantile(q), s.quantile(1 - q)
    top, bot = g[g[score] >= hi], g[g[score] <= lo]
    ft, fb = top[f"fwd{h}"].dropna(), bot[f"fwd{h}"].dropna()
    if len(ft) < 5 or len(fb) < 5:
        return {}
    return {"n_top": len(ft), "n_bot": len(fb),
            "win": (ft > 0).mean() * 100 - (fb > 0).mean() * 100,
            "mean": ft.mean() - fb.mean(),
            "reb": (top.loc[ft.index, f"max{h}"] >= B.REBOUND_PCT).mean() * 100 - (bot.loc[fb.index, f"max{h}"] >= B.REBOUND_PCT).mean() * 100}


def adj_spread(g: pd.DataFrame, score: str, h: int, q: float = 0.2) -> dict:
    """상위/하위 20% 점수 집단의 국면보정 수익(adj) 평균 차이와 '같은 달 평균 상회' 비율 차이 (%p)."""
    s = g[score].dropna()
    if len(s) < 20:
        return {}
    lo, hi = s.quantile(q), s.quantile(1 - q)
    top, bot = g.loc[g[score] >= hi, f"adj{h}"].dropna(), g.loc[g[score] <= lo, f"adj{h}"].dropna()
    if len(top) < 5 or len(bot) < 5:
        return {}
    return {"mean": top.mean() - bot.mean(), "beat": (top > 0).mean() * 100 - (bot > 0).mean() * 100}


def quantile_table(g: pd.DataFrame, score: str, q: int, hs=EVAL_HS) -> pd.DataFrame:
    s = g[score]
    try:
        bins = pd.qcut(s.rank(method="first"), q, labels=[f"Q{i + 1}" for i in range(q)])
    except ValueError:
        return pd.DataFrame()
    rows = []
    for lab in bins.cat.categories:
        sub = g[bins == lab]
        r = {"bin": lab, "n": len(sub), "score_mean": sub[score].mean(), "score_min": sub[score].min(), "score_max": sub[score].max(), "target": sub["target"].mean()}
        for h in hs:
            f = sub[f"fwd{h}"].dropna()
            r[f"win{h}"] = (f > 0).mean() * 100 if len(f) else np.nan
            r[f"mean{h}"] = f.mean() if len(f) else np.nan
            r[f"reb{h}"] = (sub.loc[f.index, f"max{h}"] >= B.REBOUND_PCT).mean() * 100 if len(f) else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


def calib_table(g: pd.DataFrame, score: str, width: float = 5.0) -> pd.DataFrame:
    s = g[score].dropna()
    lo = math.floor(s.min() / width) * width
    hi = math.ceil(s.max() / width) * width + width
    edges = np.arange(lo, hi + 1e-9, width)
    b = pd.cut(g[score], edges, right=False)
    rows = []
    for iv, sub in g.groupby(b, observed=True):
        if not len(sub):
            continue
        f60 = sub["fwd60"].dropna()
        rows.append({"score_bin": f"{iv.left:.0f}~{iv.right:.0f}", "n": len(sub), "score_mean": sub[score].mean(), "actual_target": sub["target"].mean(),
                     "gap": sub["target"].mean() - sub[score].mean(),
                     "win60": (f60 > 0).mean() * 100 if len(f60) else np.nan, "mean60": f60.mean() if len(f60) else np.nan,
                     "over": sub[score].mean() > sub["target"].mean() + 1e-9})
    return pd.DataFrame(rows)


def print_df(df: pd.DataFrame, cols: list[str] | None = None, fmt: str = "{:.1f}") -> None:
    if df is None or not len(df):
        print("  (표본 부족)")
        return
    d = df[cols] if cols else df
    with pd.option_context("display.width", 250, "display.max_columns", 50, "display.float_format", fmt.format):
        print(d.to_string(index=False))


# ---------------------------------------------------------------- 메인
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true", help="이벤트 캐시 무시하고 다시 생성")
    ap.add_argument("--limit", type=int, default=None, help="지역별 종목 수 제한(테스트용)")
    ap.add_argument("--z", type=float, default=1.28, help="Wilson 하한 z (1.28 = 단측 90%%)")
    ap.add_argument("--deff", type=float, default=None, help="설계효과(유효표본 = n/deff). 기본: 데이터에서 추정")
    ap.add_argument("--k-cell", type=float, default=500.0, help="셀 델타 수축 k (n/(n+k))")
    ap.add_argument("--k-ticker", type=float, default=40.0, help="종목 → 셀 확률 수축 k (M3)")
    ap.add_argument("--gap", type=int, default=PIT_GAP_DAYS)
    a = ap.parse_args()
    t0 = time.time()
    th = Thresholds(stoch_max=float(Thresholds.stoch_max), band_tol=float(Thresholds.band_tol), mkt_lo=float(Thresholds.mkt_lo), mkt_hi=float(Thresholds.mkt_hi))
    print(f"임계값: 스토캐스틱≤{th.stoch_max:g}(미국 주봉/국내 일봉) 밴드±{th.band_tol:g}% 게이트 지수RSI {th.mkt_lo:g}~{th.mkt_hi:g}, 매수=다음날 시가, PIT 간격 {a.gap}일")

    # ---- 1) 이벤트
    if EV_PATH.exists() and not a.rebuild and not a.limit:
        X = pd.read_parquet(EV_PATH)
        print(f"이벤트 캐시 로드 {EV_PATH.name}: {len(X)}건 [{time.time() - t0:.0f}s]")
    else:
        X = build_events(th, a.limit)
        if not a.limit:
            X.to_parquet(EV_PATH, index=False)
        print(f"이벤트 생성 {len(X)}건 → {EV_PATH.name} [{time.time() - t0:.0f}s]")
    X["year"] = X["date"].dt.year
    X["target"] = target(X)
    ym = X["date"].dt.to_period("M")
    for h in EVAL_HS:  # 같은 지역·같은 달 진입 이벤트(모든 조합) 평균을 뺀 값: 시장 국면 효과 제거(횡단면 평가용)
        X[f"adj{h}"] = X[f"fwd{h}"] - X.groupby([X["region"], ym])[f"fwd{h}"].transform("mean")
    if not a.limit:
        verify_against_cache(X, th)

    summary_rows: list[dict] = []

    def emit(table: str, **kw):
        summary_rows.append({"table": table, **{k: (round(v, 3) if isinstance(v, float) else v) for k, v in kw.items()}})

    print("\n==== 1) 조건 조합별 성과 (시장 게이트 항상 적용) — 승률%/반등%(+10%)/평균%/중앙%/하위10% ====")
    for reg in ("US", "KR"):
        print(f"-- {reg}")
        print(perf_header())
        for combo in COMBOS:
            g = X[(X.region == reg) & (X.combo == combo)]
            r = perf(g)
            emit("combo_perf", region=reg, combo=combo, **r)
            print(fmt_perf(f"  {combo} {COMBO_NAME[combo]}", r))
        g = X[X.region == reg]
        print(f"  (참고) 조합별 이벤트 비중: " + ", ".join(f"{c} {int((g.combo == c).sum())}건" for c in COMBOS) + f", 종목 수 {g.ticker.nunique()}")

    # ---- 셀 임계값: 지역별 조합 C 이벤트의 특성 3분위 (결과와 무관한 분포 → 고정값으로 사용)
    thresholds: dict[str, dict] = {}
    for reg in ("US", "KR"):
        g = X[(X.region == reg) & (X.combo == "C")]
        thresholds[reg] = {"slope200": g["slope200"].quantile([1 / 3, 2 / 3]).round(1).tolist(),
                           "from_high52": g["from_high52"].quantile([1 / 3, 2 / 3]).round(1).tolist()}
    print("\n셀 임계값(지역별 조합 C 3분위, 고정): " + json.dumps(thresholds))

    X["scell"] = None  # 기울기 3분위
    X["dcell"] = None  # 낙폭 3분위
    for reg in ("US", "KR"):
        m = X.region == reg
        t = thresholds[reg]
        sc = pd.cut(X.loc[m, "slope200"], [-np.inf, t["slope200"][0], t["slope200"][1], np.inf], labels=SLOPE_LAB, right=False)
        dc = pd.cut(X.loc[m, "from_high52"], [-np.inf, t["from_high52"][0], t["from_high52"][1], np.inf], labels=DD_LAB, right=False)
        X.loc[m, "scell"] = ("s_" + sc.astype(str)).where(sc.notna(), None)
        X.loc[m, "dcell"] = ("dd_" + dc.astype(str)).where(dc.notna(), None)
    X["cell"] = (X["scell"] + "|" + X["dcell"]).where(X["scell"].notna() & X["dcell"].notna(), None)

    # ---- 설계효과 (같은 달 클러스터) — 지역 수준(level) 불확실성에만 적용
    deff = {reg: (a.deff or design_effect(X[(X.region == reg) & (X.combo == "C")])) for reg in ("US", "KR")}
    print(f"설계효과(월 클러스터, 60일 승패, 조합 C): " + ", ".join(f"{r} {v:.2f}" for r, v in deff.items()) + (" (지정값)" if a.deff else " (추정)"))

    # ---- 1b) 셀 효과의 기간 이전(transfer): <2016 로 추정한 셀 델타가 2016+ 에서 재현되는가
    print("\n==== 1b) 셀 효과 이전 검증: 2016년 이전 이벤트로 추정한 셀 델타(셀−조합 평균, 목표값 %p) vs 2016년 이후 실제 델타 ====")
    print("  beta = 이후 델타를 이전 델타에 회귀한 기울기(1=그대로 재현, 0=무관, 음수=반전), corr = 상관")
    for reg in ("US", "KR"):
        for combo in COMBOS:
            g = X[(X.region == reg) & (X.combo == combo) & X.cell.notna()]
            tr, te = g[g.year < 2016], g[g.year >= 2016]
            cells = []
            for key in ("cell", "scell", "dcell"):
                d = tr.groupby(key).agg(n_tr=("target", "size"), t_tr=("target", "mean")).join(te.groupby(key).agg(n_te=("target", "size"), t_te=("target", "mean")))
                d["d_tr"], d["d_te"] = d.t_tr - tr.target.mean(), d.t_te - te.target.mean()
                w = d.n_te
                beta = float((w * d.d_tr * d.d_te).sum() / (w * d.d_tr ** 2).sum()) if (w * d.d_tr ** 2).sum() else np.nan
                corr = float(np.corrcoef(d.d_tr, d.d_te)[0, 1]) if len(d) > 2 else np.nan
                cells.append(f"{key} beta {beta:+.2f} corr {corr:+.2f}")
                emit("transfer", region=reg, combo=combo, key=key, beta=beta, corr=corr, n_train=len(tr), n_test=len(te))
                if key == "scell":
                    sdelta = " ".join(f"{i}: {r.d_tr:+.1f}→{r.d_te:+.1f}" for i, r in d.iterrows())
            print(f"  {reg} {combo}: " + " | ".join(cells) + f"   [기울기 3분위 델타 이전→이후: {sdelta}]")

    # ---- 2) PIT 통계 (walk-forward)
    X = X.sort_values(["region", "combo", "date"]).reset_index(drop=True)
    ps_base = PitStats(X, ["region", "combo"])
    st_base = ps_base.query(X, a.gap)
    dz = np.array([deff[r] for r in X["region"]])

    def pit_cell_stats(col: str) -> dict[str, np.ndarray]:
        Xc = X.dropna(subset=[col])
        stc = PitStats(Xc, ["region", "combo", col]).query(Xc, a.gap)
        out = {k: np.zeros(len(X)) for k in stc}
        for k in stc:
            out[k][Xc.index.to_numpy()] = stc[k]
        return out

    st_cell, st_scell, st_dcell = pit_cell_stats("cell"), pit_cell_stats("scell"), pit_cell_stats("dcell")
    st_tick = PitStats(X, ["region", "combo", "ticker"]).query(X, a.gap)
    XC = X[X.combo == "C"]
    st_tick_c = PitStats(XC, ["region", "ticker"]).query(XC, a.gap)

    X["n_base_pit"] = st_base["n20"]
    X["n_cell_pit"], X["n_scell_pit"], X["n_tick_pit"] = st_cell["n20"], st_scell["n20"], st_tick["n20"]
    X["M1_raw"] = level_prob(st_base, 0.0, np.ones(len(X)))                 # 기준율 그대로
    X["M1"] = level_prob(st_base, a.z, dz)                                  # 기준율 하한(유효표본 n/deff)
    X["M2"] = X["M1"] + delta_prob(st_cell, st_base, a.k_cell)              # + 2차원 셀(기울기×낙폭) 델타
    X["M2s"] = X["M1"] + delta_prob(st_scell, st_base, a.k_cell)            # + 기울기 셀 델타
    X["M2dd"] = X["M1"] + delta_prob(st_dcell, st_base, a.k_cell)           # + 낙폭 셀 델타
    X["M2s_half"] = X["M1"] + 0.5 * delta_prob(st_scell, st_base, a.k_cell)  # 기울기 델타 절반
    X["M3"] = X["M2s"] + delta_prob(st_tick, st_scell, a.k_ticker)          # + 종목 델타(셀 대비, k_ticker)
    # 종목 PIT 확률 (현재 앱 공식: k=8, 조합 C, 룩어헤드 없음) — 조합 C 이벤트만
    X["prob_pit_app"] = np.nan
    base_h_c = horizon_probs(ps_base.query(XC, a.gap), 0.0, 1.0)
    X.loc[XC.index, "n_tick_pit_app"] = st_tick_c["n20"]
    X.loc[XC.index, "prob_pit_app"], _ = composite(st_tick_c, 0.0, 1.0, base_h_c, B.SHRINK_K)
    # M0: 현재 앱 값 (종목 전체 이벤트, 룩어헤드)
    res = B.cached(th, True, "next_open")
    full = {t: v["prob"] for t, v in res["tickers"].items()} if res else {}
    X["M0"] = X["ticker"].map(full)

    if not a.limit:
        X.to_parquet(SC_PATH, index=False)
        print(f"이벤트별 점수 저장 {SC_PATH.name} [{time.time() - t0:.0f}s]")

    E = X[(X.n_base_pit >= MIN_BASE_N) & X.cell.notna()].copy()
    print(f"평가 대상 이벤트(기준율 PIT 표본≥{MIN_BASE_N}, 셀 있음): {len(E)} / {len(X)}  (" + ", ".join(f"{r} {int((E.region == r).sum())}" for r in ("US", "KR")) + ")")

    # ---- 2b) 종목 PIT 확률 예측력 (조합 C, 미국)
    print("\n==== 2) 종목 PIT 추천확률(앱 공식 k=8, 이벤트일 이전 이벤트만)의 예측력 — 미국, 조합 C ====")
    EC = E[(E.region == "US") & (E.combo == "C")]
    print(f"이벤트 {len(EC)}건, 종목 과거 이벤트 0건 {int((EC.n_tick_pit_app == 0).sum())}건, 중앙값 {EC.n_tick_pit_app.median():.0f}건")
    print("십분위 (점수 → 20/60/180일 승률·평균):")
    qt = quantile_table(EC, "prob_pit_app", 10)
    print_df(qt, ["bin", "n", "score_mean", "target", "win20", "mean20", "win60", "mean60", "win180", "mean180"])
    for h in EVAL_HS:
        sp, asp = spread(EC, "prob_pit_app", h), adj_spread(EC, "prob_pit_app", h)
        print(f"  fwd{h}: 스피어만 {spearman(EC['prob_pit_app'], EC[f'fwd{h}']):+.3f}, 상위20%−하위20% 승률 {sp.get('win', np.nan):+.1f}p 평균 {sp.get('mean', np.nan):+.1f}p 반등 {sp.get('reb', np.nan):+.1f}p | 국면보정 평균 {asp.get('mean', np.nan):+.1f}p")
        emit("pit_app_pred", region="US", combo="C", h=h, spearman=spearman(EC['prob_pit_app'], EC[f'fwd{h}']), **{f"spread_{k}": v for k, v in sp.items()}, adj_mean=asp.get("mean"))
    print("  (비교) M0 현재 앱 값(룩어헤드): " + ", ".join(f"fwd{h} 스피어만 {spearman(EC['M0'], EC[f'fwd{h}']):+.3f} 스프레드 승률 {spread(EC, 'M0', h).get('win', np.nan):+.1f}p" for h in EVAL_HS))

    # ---- 3) 모델 비교
    MODELS = ["M0", "M1_raw", "M1", "M2", "M2s", "M2dd", "M2s_half", "M3"]
    MNAME = {"M0": "현재(종목 전체값, 룩어헤드)", "M1_raw": "지역×조합 기준율 PIT(그대로)", "M1": "기준율 하한(z, n/deff)", "M2": "M1+기울기×낙폭 셀 델타",
             "M2s": "M1+기울기 셀 델타", "M2dd": "M1+낙폭 셀 델타", "M2s_half": "M1+기울기 델타×0.5", "M3": "M2s+종목 델타(k_ticker)"}
    print(f"\n==== 3) 모델 비교 (walk-forward 점수 vs 이후 성과)  z={a.z}, k_cell={a.k_cell:g}, k_ticker={a.k_ticker:g} ====")
    print("모델: " + "; ".join(f"{m}={MNAME[m]}" for m in MODELS))
    for reg in ("US", "KR"):
        scopes = [("ALL", E[E.region == reg]), ("C", E[(E.region == reg) & (E.combo == "C")]),
                  ("ALL 2016+", E[(E.region == reg) & (E.year >= 2016)]), ("C 2016+", E[(E.region == reg) & (E.combo == "C") & (E.year >= 2016)])]
        for scope, g in scopes:
            if reg == "KR" and scope not in ("C", "C 2016+"):
                continue
            print(f"\n-- {reg} / 조합 {scope} ({len(g)}건): 스피어만(fwd20/60/180 | adj60) | 상위20%−하위20% 60일 승률/평균/반등 | 180일 승률/평균 | 국면보정 60일 평균/상회율, 180일 평균 | 점수 평균(실제) | 연도별 60일 국면보정 스프레드 양(+)/전체")
            for m in MODELS:
                if g[m].notna().sum() < 100 or g[m].nunique() < 5:
                    continue
                cors = [spearman(g[m], g[f"fwd{h}"]) for h in EVAL_HS] + [spearman(g[m], g["adj60"])]
                s60, s180 = spread(g, m, 60), spread(g, m, 180)
                a60, a180 = adj_spread(g, m, 60), adj_spread(g, m, 180)
                ys = []
                for y, gy in g.groupby("year"):
                    if gy[m].nunique() < 5 or len(gy) < 40:
                        continue
                    sy = adj_spread(gy, m, 60)
                    if sy:
                        ys.append(sy["mean"] > 0)
                emit("model_cmp", region=reg, scope=scope, model=m, n=len(g), sp20=cors[0], sp60=cors[1], sp180=cors[2], sp_adj60=cors[3],
                     spread60_win=s60.get("win"), spread60_mean=s60.get("mean"), spread60_reb=s60.get("reb"), spread180_win=s180.get("win"), spread180_mean=s180.get("mean"),
                     adj60_mean=a60.get("mean"), adj60_beat=a60.get("beat"), adj180_mean=a180.get("mean"),
                     score_mean=g[m].mean(), target_mean=g["target"].mean(), years_pos=sum(ys), years=len(ys))
                print(f"  {m:<8} {cors[0]:+.3f}/{cors[1]:+.3f}/{cors[2]:+.3f}|{cors[3]:+.3f} | {s60.get('win', np.nan):+5.1f}p/{s60.get('mean', np.nan):+5.1f}p/{s60.get('reb', np.nan):+5.1f}p | "
                      f"{s180.get('win', np.nan):+5.1f}p/{s180.get('mean', np.nan):+5.1f}p | {a60.get('mean', np.nan):+5.1f}p/{a60.get('beat', np.nan):+5.1f}p, {a180.get('mean', np.nan):+5.1f}p | {g[m].mean():5.1f} ({g['target'].mean():5.1f}) | {sum(ys)}/{len(ys)}")
            if reg == "US" and scope in ("C", "C 2016+"):
                for m in ("M0", "M2s"):
                    print(f"  5분위 — {m}")
                    qt = quantile_table(g, m, 5)
                    print_df(qt, ["bin", "n", "score_mean", "score_min", "score_max", "target", "win20", "win60", "mean60", "reb60", "win180", "mean180"])
                    for _, r in qt.iterrows():
                        emit("quintile", region=reg, scope=scope, model=m, **r.to_dict())

    # 캘리브레이션
    print("\n==== 3b) 캘리브레이션 — 점수 5점 구간별 실제 목표값(Σw(0.6승+0.4반등)) / 60일 승률; over=예측>실제 ====")
    print("보수적 = n≥100 인 모든 구간에서 예측 ≤ 실제(과대평가 없음)")
    for reg in ("US", "KR"):
        for scope, g in (("ALL", E[E.region == reg]), ("C", E[(E.region == reg) & (E.combo == "C")]), ("ALL 2016+", E[(E.region == reg) & (E.year >= 2016)]), ("C 2016+", E[(E.region == reg) & (E.combo == "C") & (E.year >= 2016)])):
            if reg == "KR" and scope not in ("C", "C 2016+"):
                continue
            for m in (["M0", "M1_raw", "M1", "M2", "M2s", "M3"] if reg == "US" else ["M1_raw", "M1", "M2s"]):
                ct = calib_table(g, m)
                big = ct[ct.n >= 100]
                n_over = int(big["over"].sum())
                worst = float(big["gap"].min()) if len(big) else np.nan
                print(f"-- {reg}/{scope} {m}: n≥100 구간 {len(big)}개 중 과대평가 {n_over}개, 최악 (실제−예측) {worst:+.1f}p, 전체 평균 예측 {g[m].mean():.1f} vs 실제 {g['target'].mean():.1f}")
                if reg == "US" and m in ("M0", "M2s") and scope in ("ALL", "C", "C 2016+"):
                    print_df(ct, ["score_bin", "n", "score_mean", "actual_target", "gap", "win60", "mean60", "over"])
                for _, r in ct.iterrows():
                    emit("calibration", region=reg, scope=scope, model=m, **r.to_dict())

    print("\n==== 3c) 조합별 PIT 기준율 평균과 보수화 폭 ====")
    for reg in ("US", "KR"):
        g = E[E.region == reg]
        print(f"-- {reg}: " + " | ".join(f"{c}: M1_raw {g[g.combo == c]['M1_raw'].mean():.1f} M1 {g[g.combo == c]['M1'].mean():.1f} 실제 {g[g.combo == c]['target'].mean():.1f} (n={int((g.combo == c).sum())})" for c in COMBOS))
        print(f"   보수화 폭(평균 점수 차): M1_raw→M1(수준 하한) {g['M1'].mean() - g['M1_raw'].mean():+.1f}p, M1→M2s(셀) {g['M2s'].mean() - g['M1'].mean():+.1f}p, M0 대비 M2s {g['M2s'].mean() - g['M0'].mean():+.1f}p")

    # ---- 4) 최종 모델 (전체 표본, 현재 시점): 기준율 하한 + 기울기 셀 델타 → prob_model.json
    print("\n==== 4) 최종 모델 (M2s) 확률표 — 전체 20년, 현재 시점 값 ====")
    print(f"  prob = base_LB + delta_cell;  base_LB = Σw·Wilson하한(p_h, n_h/deff, z={a.z}),  delta_cell = Σw·[n_ch/(n_ch+k)]·(p_ch − p_h), k={a.k_cell:g}")
    model = {"version": 2, "computed_at": pd.Timestamp.now(tz="Asia/Seoul").isoformat(timespec="seconds"),
             "formula": "prob = base[region][combo].prob + region[region][combo][cell].delta ; cell = slope200 tercile (s_lo/s_mid/s_hi) by thresholds[region].slope200; no cell → delta 0",
             "params": {"z": a.z, "k_cell": a.k_cell, "deff": {r: round(v, 3) for r, v in deff.items()}, "weights": {str(h): W[h] for h in H},
                        "rebound_pct": B.REBOUND_PCT, "stoch_max": th.stoch_max, "band_tol": th.band_tol, "mkt_lo": th.mkt_lo, "mkt_hi": th.mkt_hi, "span": [str(X.date.min().date()), str(X.date.max().date())]},
             "thresholds": {r: {"slope200": thresholds[r]["slope200"]} for r in thresholds}, "cell_labels": ["s_lo", "s_mid", "s_hi"],
             "combo": COMBO_NAME, "base": {}, "region": {}}
    for reg in ("US", "KR"):
        model["base"][reg], model["region"][reg] = {}, {}
        for combo in COMBOS:
            g = X[(X.region == reg) & (X.combo == combo)]
            hs = B._hstats(records(g))
            stb = {f"{k}{h}": np.array([hs[str(h)].get({"n": "n", "w": "wins", "r": "rebounds"}[k], 0)], float) for h in H for k in ("n", "w", "r")}
            p_raw = float(level_prob(stb, 0.0, np.ones(1))[0])
            p_lb = float(level_prob(stb, a.z, np.array([deff[reg]]))[0])
            model["base"][reg][combo] = {"prob": round(p_lb, 1), "prob_raw": round(p_raw, 1), "n": int(hs["20"]["n"]), "n_eff": int(hs["20"]["n"] / deff[reg]),
                                        "win60": hs["60"].get("win"), "reb60": hs["60"].get("rebound"), "mean60": hs["60"].get("mean"), "win180": hs["180"].get("win"),
                                        "p_h": {str(h): {"n": int(hs[str(h)]["n"]), "wins": int(hs[str(h)].get("wins", 0)), "rebounds": int(hs[str(h)].get("rebounds", 0))} for h in H}}
            model["region"][reg][combo] = {}
            for cell in ("s_lo", "s_mid", "s_hi"):
                gc = g[g.scell == cell]
                hc = B._hstats(records(gc)) if len(gc) else {str(h): {"n": 0} for h in H}
                stc = {f"{k}{h}": np.array([hc[str(h)].get({"n": "n", "w": "wins", "r": "rebounds"}[k], 0)], float) for h in H for k in ("n", "w", "r")}
                delta = float(delta_prob(stc, stb, a.k_cell)[0])
                p_cell_raw = float(level_prob(stc, 0.0, np.ones(1))[0]) if len(gc) else None
                model["region"][reg][combo][cell] = {"prob": round(p_lb + delta, 1), "delta": round(delta, 1), "prob_raw": (round(p_cell_raw, 1) if p_cell_raw is not None else None), "n": int(len(gc)),
                                                     "win60": hc["60"].get("win"), "reb60": hc["60"].get("rebound"), "mean60": hc["60"].get("mean"), "win180": hc["180"].get("win"), "mean180": hc["180"].get("mean")}
                emit("final_cell", region=reg, combo=combo, cell=cell, **{k: v for k, v in model["region"][reg][combo][cell].items()})
            # 참고: 2차원 셀(기울기×낙폭) 원시 통계도 요약 CSV 에
            for cell, gc in g.groupby("cell"):
                hc = B._hstats(records(gc))
                emit("ref_cell2d", region=reg, combo=combo, cell=cell, n=len(gc), target=gc.target.mean(), win60=hc["60"].get("win"), mean60=hc["60"].get("mean"), win180=hc["180"].get("win"))
        print(f"-- {reg}: 기준율 하한/원시/n/유효n: " + ", ".join(f"{c} {model['base'][reg][c]['prob']}/{model['base'][reg][c]['prob_raw']}/{model['base'][reg][c]['n']}/{model['base'][reg][c]['n_eff']}" for c in COMBOS))
        for combo in COMBOS:
            line = []
            for cell in ("s_lo", "s_mid", "s_hi"):
                v = model["region"][reg][combo][cell]
                line.append(f"{cell:<5} {v['prob']:5.1f} (Δ{v['delta']:+4.1f}, raw {v['prob_raw']:5.1f}, n={v['n']:>5}, 60일 {v['win60']:3.0f}%·{v['mean60']:+5.1f}, 180일 {v['win180']:3.0f}%·{v['mean180']:+5.1f})")
            print(f"   {combo} {COMBO_NAME[combo]:<10} " + " | ".join(line))

    if not a.limit:
        MODEL_PATH.write_text(json.dumps(model, ensure_ascii=False, indent=1))
        pd.DataFrame(summary_rows).to_csv(SUM_PATH, index=False)

    # 최종 모델(전체 표본)을 이벤트에 그대로 적용했을 때의 캘리브레이션 (in-sample, 수준 하한이 얼마나 여유를 주는지)
    print("\n==== 4a) 최종 모델을 전체 이벤트에 적용(in-sample) — 셀별 예측 vs 실제 목표값 (예측 ≤ 실제 확인) ====")
    for reg in ("US", "KR"):
        rows = []
        for combo in COMBOS:
            for cell in ("s_lo", "s_mid", "s_hi"):
                g = X[(X.region == reg) & (X.combo == combo) & (X.scell == cell)]
                v = model["region"][reg][combo][cell]
                rows.append({"combo": combo, "cell": cell, "n": len(g), "pred": v["prob"], "actual": g.target.mean(), "gap": g.target.mean() - v["prob"],
                             "actual_2016+": g[g.year >= 2016].target.mean(), "worst_year": g.groupby("year").target.mean().min()})
        print(f"-- {reg}")
        print_df(pd.DataFrame(rows))

    # ---- 4b) 오늘 스캔 결과 예시 (미국, 규칙 충족 종목)
    print("\n==== 4b) 오늘 스캔(미국, 매수 규칙 충족 종목)을 최종 모델로 점수 매긴 예시 ====")
    try:
        today = json.loads((DATA_DIR / "results.json").read_text())
    except Exception:  # noqa: BLE001
        today = None
    if today and res:
        th_app = Thresholds()
        hits = [s for s in today["stocks"] if s["region"] == "US" and buy_hit(s, th_app)]
        ex = []
        for s in hits:
            t = s["ticker"]
            tv = res["tickers"].get(t, {})
            sc = score_today(model, "US", "C", s.get("ma_slope200"))
            ex.append({"ticker": t, "name": s.get("name", ""), "slope200": s.get("ma_slope200"), "from_high52": s.get("from_high52"), "cell": sc["cell"],
                       "n_ticker": tv.get("n"), "old_prob": tv.get("prob"), "new_prob": sc["prob"]})
        D = pd.DataFrame(ex)
        if len(D):
            D["diff"] = D["new_prob"] - D["old_prob"]
            print(f"규칙 충족 {len(D)}종목: 기존 확률 평균 {D.old_prob.mean():.1f} → 새 확률 평균 {D.new_prob.mean():.1f} (차이 {D['diff'].mean():+.1f}p), 셀 분포 {D.cell.value_counts().to_dict()}")
            show = pd.concat([D.sort_values("old_prob", ascending=False).head(5), D.sort_values("old_prob").head(5)]).drop_duplicates("ticker")
            print_df(show, ["ticker", "name", "slope200", "from_high52", "cell", "n_ticker", "old_prob", "new_prob", "diff"])
            if not a.limit:
                D.to_csv(DATA_DIR / "pit_prob_today_us.csv", index=False)
    print(f"\n저장: {MODEL_PATH.name}, {SUM_PATH.name} ({len(summary_rows)}행), {SC_PATH.name}, {EV_PATH.name}  총 {time.time() - t0:.0f}s")


def level_prob(st: dict[str, np.ndarray], z: float, deff: np.ndarray) -> np.ndarray:
    """기준율(%): Σ_h w_h · Wilson하한(p_h, n_h/deff, z),  p_h = (0.6·승 + 0.4·반등)/n_h. z=0 이면 원시값."""
    acc, tw = np.zeros(len(st["n20"])), np.zeros(len(st["n20"]))
    for h in H:
        n = st[f"n{h}"]
        with np.errstate(divide="ignore", invalid="ignore"):
            p = np.where(n > 0, (0.6 * st[f"w{h}"] + 0.4 * st[f"r{h}"]) / np.maximum(n, 1), np.nan)
        ph = wilson_lb(p, n / deff, z)
        ok = ~np.isnan(ph)
        acc[ok] += W[h] * ph[ok]
        tw[ok] += W[h]
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(tw > 0, acc / tw, np.nan) * 100


def delta_prob(st_cell: dict[str, np.ndarray], st_base: dict[str, np.ndarray], k: float) -> np.ndarray:
    """셀 델타(%p): Σ_h w_h · [n_ch/(n_ch+k)] · (p_ch − p_h). 셀 표본 0 이면 0."""
    acc, tw = np.zeros(len(st_base["n20"])), np.zeros(len(st_base["n20"]))
    for h in H:
        n, nc = st_base[f"n{h}"], st_cell[f"n{h}"]
        with np.errstate(divide="ignore", invalid="ignore"):
            p = np.where(n > 0, (0.6 * st_base[f"w{h}"] + 0.4 * st_base[f"r{h}"]) / np.maximum(n, 1), np.nan)
            pc = np.where(nc > 0, (0.6 * st_cell[f"w{h}"] + 0.4 * st_cell[f"r{h}"]) / np.maximum(nc, 1), np.nan)
        d = np.where((n > 0) & (nc > 0), nc / (nc + k) * (pc - p), 0.0)
        ok = n > 0
        acc[ok] += W[h] * d[ok]
        tw[ok] += W[h]
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(tw > 0, acc / tw, 0.0) * 100


def score_today(model: dict, region: str, combo: str, slope200: float | None) -> dict:
    """앱에 넣을 최종 공식 (참조 구현): 셀 = slope200 3분위 → prob = base.prob + cell.delta (= cell.prob). 셀 없으면 base.prob."""
    t = model["thresholds"][region]["slope200"]
    base = model["base"][region][combo]
    if slope200 is None:
        return {"cell": None, "prob": base["prob"]}
    cell = "s_lo" if slope200 < t[0] else "s_mid" if slope200 < t[1] else "s_hi"
    return {"cell": cell, "prob": model["region"][region][combo][cell]["prob"]}


if __name__ == "__main__":
    main()
