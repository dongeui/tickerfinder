"""과거 구간 백필: 로컬 가격 캐시로 날짜별 지표 스냅샷(data/history/YYYY-MM-DD.json)을 생성.

실시간 스캔(scanner.run_scan)이 하루 1개 스냅샷을 쌓는 것과 같은 포맷이므로, 과거 구간을 한 번에 채워두면
리포트/시그널 분석이 바로 가능하다.

사용:  uv run python -m app.backfill 2026-06-01 2026-10-11
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime

import numpy as np
import pandas as pd

from .indicators import rsi, slow_stochastic, stoch_point_in_time
from .scanner import HISTORY_DIR, INDICES, KST, load_cache, write_history
from .universe import DATA_DIR, build_universe

log = logging.getLogger(__name__)


def weekly_rsi_daily(close: pd.Series, n: int = 14) -> pd.Series:
    """각 일자 t 기준의 주봉 RSI (t가 속한 주는 t 종가로 마감한 미완성 봉으로 취급).

    완성된 직전 주까지의 Wilder 평균을 구한 뒤, t 종가로 한 스텝만 더 진행시키면 정확히 일치한다.
    """
    w = close.resample("W-FRI").last().dropna()
    delta = w.diff()
    up = delta.clip(lower=0)
    dn = -delta.clip(upper=0)
    au = up.ewm(alpha=1 / n, adjust=False).mean()
    ad = dn.ewm(alpha=1 / n, adjust=False).mean()
    # 일자 t → t가 속한 주의 금요일 라벨 → 직전 주 상태
    fri = close.index + pd.to_timedelta((4 - close.index.weekday) % 7, unit="D")
    prev_close = w.shift(1).reindex(fri).to_numpy()
    prev_au = au.shift(1).reindex(fri).to_numpy()
    prev_ad = ad.shift(1).reindex(fri).to_numpy()
    d = close.to_numpy() - prev_close
    au_t = prev_au * (n - 1) / n + np.clip(d, 0, None) / n
    ad_t = prev_ad * (n - 1) / n + np.clip(-d, 0, None) / n
    with np.errstate(divide="ignore", invalid="ignore"):
        out = 100 - 100 / (1 + au_t / ad_t)
    out = np.where(ad_t == 0, 100.0, out)
    out = np.where((au_t == 0) & (ad_t == 0), 50.0, out)
    return pd.Series(out, index=close.index)


def compute_series(df: pd.DataFrame) -> pd.DataFrame:
    """일자별 지표 테이블 (compute_metrics 와 같은 정의)."""
    df = df.dropna(subset=["Close"])
    c = df["Close"]
    h = df["High"].fillna(c)
    l = df["Low"].fillna(c)
    sk, sd = slow_stochastic(h, l, c)
    ma150 = c.rolling(150).mean()
    ma200 = c.rolling(200).mean()
    lo = np.minimum(ma150, ma200)
    hi = np.maximum(ma150, ma200)
    band = np.where(c > hi, (c / hi - 1) * 100, np.where(c < lo, (c / lo - 1) * 100, 0.0))
    band = pd.Series(band, index=c.index).where(lo.notna())
    jump = c.pct_change().abs()
    return pd.DataFrame({
        "close": c, "rsi_d": rsi(c), "rsi_w": weekly_rsi_daily(c),
        "suspect": jump.rolling(250, min_periods=1).max() > 0.5,
        "stoch_k": sk, "stoch_d": sd, "stoch_k_w": stoch_point_in_time(df, "W-FRI"), "band_pos": band,
        "ma150": ma150, "ma200": ma200,
        "turnover20": (c * df["Volume"].fillna(0)).rolling(20).mean(),
    })


def _r(v, nd=2):
    return None if v is None or (isinstance(v, float) and np.isnan(v)) else round(float(v), nd)


def backfill(start: str, end: str, overwrite: bool = False) -> list[str]:
    uni_path = DATA_DIR / "universe.json"
    uni = json.loads(uni_path.read_text()) if uni_path.exists() else build_universe()
    series: dict[str, pd.DataFrame] = {}
    for u in uni:
        df = load_cache(u["ticker"])
        if df is not None and len(df) >= 220:
            series[u["ticker"]] = compute_series(df)
    idx_series = {}
    for key, sym in INDICES.items():
        df = load_cache(sym)
        if df is not None and len(df) >= 220:
            idx_series[key] = compute_series(df)
    log.info("series ready: %d stocks, %d indices", len(series), len(idx_series))

    rows_by_date: dict[str, list[dict]] = {}
    for u in uni:
        ser = series.get(u["ticker"])
        if ser is None:
            continue
        sub = ser.loc[start:end].dropna(subset=["rsi_d"])
        for day, row in sub.iterrows():
            rows_by_date.setdefault(day.strftime("%Y-%m-%d"), []).append({
                "ticker": u["ticker"], "region": u["region"],
                "rsi_d": _r(row["rsi_d"]), "rsi_w": _r(row["rsi_w"]), "stoch_k": _r(row["stoch_k"]), "stoch_k_w": _r(row["stoch_k_w"]),
                "band_pos": _r(row["band_pos"]), "close": _r(row["close"], 4), "suspect": bool(row["suspect"])})
    idx_by_date: dict[str, dict] = {}
    for key, ser in idx_series.items():
        for day, row in ser.loc[start:end].iterrows():
            ds = day.strftime("%Y-%m-%d")
            idx_by_date.setdefault(ds, {})[key] = {"date": ds, "close": _r(row["close"], 2), "rsi_d": _r(row["rsi_d"]),
                                                   "rsi_w": _r(row["rsi_w"]), "stoch_k": _r(row["stoch_k"]),
                                                   "stoch_d": _r(row["stoch_d"]), "band_pos": _r(row["band_pos"])}
    if overwrite:
        for ds in set(rows_by_date) | set(idx_by_date):
            p = HISTORY_DIR / f"{ds}.json"
            if p.exists():
                p.unlink()
    written = write_history(rows_by_date, idx_by_date, backfilled=True)
    log.info("backfilled %d days (%s ~ %s)", len(written), written[0] if written else "-", written[-1] if written else "-")
    return written


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    a = sys.argv[1:]
    start = a[0] if a else "2026-06-01"
    end = a[1] if len(a) > 1 else datetime.now(KST).strftime("%Y-%m-%d")
    backfill(start, end, overwrite="--overwrite" in a)
