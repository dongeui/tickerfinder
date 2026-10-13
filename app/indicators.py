"""기술적 지표 계산. 입력은 Open/High/Low/Close/Volume 컬럼을 가진 일봉 DataFrame."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    """Wilder 방식 RSI (대부분의 HTS/TradingView 기본값과 동일)."""
    delta = close.diff()
    up = delta.clip(lower=0)
    dn = -delta.clip(upper=0)
    avg_up = up.ewm(alpha=1 / n, adjust=False).mean()
    avg_dn = dn.ewm(alpha=1 / n, adjust=False).mean()
    rs = avg_up / avg_dn.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    out = out.where(avg_dn != 0, 100.0)  # 하락이 전혀 없으면 100
    out = out.where(~((avg_up == 0) & (avg_dn == 0)), 50.0)  # 완전 횡보면 50
    return out


def slow_stochastic(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14, k: int = 3, d: int = 3):
    """슬로우 스토캐스틱 (n, k, d). 반환: (slow %K, slow %D)."""
    ll = low.rolling(n).min()
    hh = high.rolling(n).max()
    rng = (hh - ll).replace(0, np.nan)
    fast_k = 100 * (close - ll) / rng
    slow_k = fast_k.rolling(k).mean()
    slow_d = slow_k.rolling(d).mean()
    return slow_k, slow_d


def to_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """일봉 → 주봉(금요일 기준). 진행 중인 이번 주도 하나의 봉으로 포함 (차트 프로그램과 동일)."""
    agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    w = df.resample("W-FRI").agg(agg)
    return w.dropna(subset=["Close"])


def _f(x, nd: int = 2):
    """JSON 직렬화 가능한 float (NaN → None)."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if math.isnan(v) or math.isinf(v):
        return None
    return round(v, nd)


def compute_metrics(df: pd.DataFrame) -> dict | None:
    """한 종목의 모든 지표를 계산해 dict로 반환. 데이터가 부족하면 None."""
    df = df.dropna(subset=["Close"]).copy()
    if len(df) < 60:
        return None
    for c in ("Open", "High", "Low", "Volume"):
        if c not in df:
            df[c] = df["Close"] if c != "Volume" else 0
    df["High"] = df["High"].fillna(df["Close"])
    df["Low"] = df["Low"].fillna(df["Close"])

    c = df["Close"]
    rsi_d = rsi(c)
    sk_d, sd_d = slow_stochastic(df["High"], df["Low"], c)
    ma20 = c.rolling(20).mean()
    ma50 = c.rolling(50).mean()
    ma150 = c.rolling(150).mean()
    ma200 = c.rolling(200).mean()

    w = to_weekly(df)
    rsi_w = rsi(w["Close"]) if len(w) >= 15 else pd.Series(dtype=float)
    sk_w, sd_w = slow_stochastic(w["High"], w["Low"], w["Close"]) if len(w) >= 20 else (pd.Series(dtype=float),) * 2

    last = c.iloc[-1]
    prev = c.iloc[-2] if len(c) > 1 else np.nan
    m150, m200 = ma150.iloc[-1], ma200.iloc[-1]
    # 두 이평선이 모두 있어야 밴드를 정의 (backfill.compute_series 와 동일)
    band_lo = min(m150, m200) if not (np.isnan(m150) or np.isnan(m200)) else np.nan
    band_hi = max(m150, m200) if not (np.isnan(m150) or np.isnan(m200)) else np.nan
    # 밴드 대비 위치(%): 밴드 안이면 0, 위면 +, 아래면 -
    if np.isnan(band_lo):
        band_pos = np.nan
    elif last > band_hi:
        band_pos = (last / band_hi - 1) * 100
    elif last < band_lo:
        band_pos = (last / band_lo - 1) * 100
    else:
        band_pos = 0.0

    turnover20 = (c * df["Volume"]).tail(20).mean()
    # 하루 ±50% 넘는 점프가 최근 250봉 안에 있으면 수정주가 미반영(분할·감자) 의심
    max_jump = c.pct_change().abs().tail(250).max()
    suspect = bool(max_jump is not None and not np.isnan(max_jump) and max_jump > 0.5)
    hi52 = c.tail(252).max()
    idx_6m = max(0, len(c) - 126)
    idx_1m = max(0, len(c) - 21)
    m50 = ma50.iloc[-1]
    ma200_prev = ma200.iloc[-21] if len(ma200) > 21 else np.nan
    ma50_prev = ma50.iloc[-21] if len(ma50) > 21 else np.nan

    return {
        "date": c.index[-1].strftime("%Y-%m-%d"),
        "close": _f(last, 4),
        "chg": _f((last / prev - 1) * 100) if prev and not np.isnan(prev) else None,
        "rsi_d": _f(rsi_d.iloc[-1]),
        "rsi_d_max20": _f(rsi_d.tail(20).max()),
        "rsi_w": _f(rsi_w.iloc[-1]) if len(rsi_w) else None,
        "stoch_k": _f(sk_d.iloc[-1]),
        "stoch_d": _f(sd_d.iloc[-1]),
        "stoch_k_w": _f(sk_w.iloc[-1]) if len(sk_w) else None,
        "stoch_d_w": _f(sd_w.iloc[-1]) if len(sd_w) else None,
        "ma20": _f(ma20.iloc[-1], 4),
        "ma50": _f(ma50.iloc[-1], 4),
        "ma150": _f(m150, 4),
        "ma200": _f(m200, 4),
        "band_pos": _f(band_pos),
        "turnover20": _f(turnover20, 0),
        "from_high52": _f((last / hi52 - 1) * 100),
        "ret_6m": _f((last / c.iloc[idx_6m] - 1) * 100),
        "ret_1m": _f((last / c.iloc[idx_1m] - 1) * 100),
        "dist_ma50": _f((last / m50 - 1) * 100) if not np.isnan(m50) else None,
        "ma_slope200": _f((m200 / ma200_prev - 1) * 100) if not (np.isnan(m200) or np.isnan(ma200_prev)) else None,
        "ma_slope50": _f((m50 / ma50_prev - 1) * 100) if not (np.isnan(m50) or np.isnan(ma50_prev)) else None,
        "bars": int(len(c)),
        "max_jump": _f(max_jump * 100) if not np.isnan(max_jump) else None,
        "suspect": suspect,
    }


def stoch_point_in_time(df: pd.DataFrame, freq: str, n: int = 14, k: int = 3) -> pd.Series:
    """일자별 상위 시간대(freq: 'W-FRI' | 'M') 슬로우 %K. 각 일자 t가 속한 주/월은 t까지의 미완성 봉으로 취급 (차트와 동일)."""
    h, l, c = df["High"].fillna(df["Close"]), df["Low"].fillna(df["Close"]), df["Close"]
    per = df.index.to_period(freq)
    bars = pd.DataFrame({"High": h, "Low": l, "Close": c}).groupby(per).agg({"High": "max", "Low": "min", "Close": "last"})
    hh, ll = bars["High"].rolling(n).max(), bars["Low"].rolling(n).min()
    fast = 100 * (bars["Close"] - ll) / (hh - ll).replace(0, np.nan)
    prev_hh = bars["High"].rolling(n - 1).max().shift(1)
    prev_ll = bars["Low"].rolling(n - 1).min().shift(1)
    g = pd.DataFrame({"h": h, "l": l}).groupby(per)
    idx = pd.PeriodIndex(per)
    HH = np.maximum(g["h"].cummax().to_numpy(), prev_hh.reindex(idx).to_numpy())
    LL = np.minimum(g["l"].cummin().to_numpy(), prev_ll.reindex(idx).to_numpy())
    rng = HH - LL
    with np.errstate(divide="ignore", invalid="ignore"):
        fk = np.where(rng > 0, 100 * (c.to_numpy() - LL) / rng, np.nan)
    acc = fk.copy()
    for i in range(1, k):
        acc = acc + fast.shift(i).reindex(idx).to_numpy()
    return pd.Series(acc / k, index=df.index)
