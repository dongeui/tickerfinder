"""종목 상세 차트 데이터: 20년 일봉(prices10y) + 최근 일봉 캐시(prices) 병합 → 일/주/월 봉 + 50·150·200일선."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from . import positions
from .longhist import load_long
from .scanner import load_cache

TF_RULE = {"w": "W-FRI", "m": "ME"}


def _merged(ticker: str) -> pd.DataFrame | None:
    long = load_long(ticker)
    recent = load_cache(ticker)
    parts = []
    if long is not None and len(long):
        long = long[[c for c in ("Open", "High", "Low", "Close", "Volume") if c in long.columns]].copy()
        parts.append(long)
    if recent is not None and len(recent):
        recent = recent[[c for c in ("Open", "High", "Low", "Close", "Volume") if c in recent.columns]].copy()
        recent.index = pd.to_datetime(recent.index).tz_localize(None).normalize()
        if parts:
            ov = long.index.intersection(recent.index)
            if len(ov) >= 5:  # 수정주가 기준이 다르면(분할 등) 옛 구간을 최근 기준으로 맞춘다
                ratio = float((recent.loc[ov, "Close"] / long.loc[ov, "Close"]).median())
                if math.isfinite(ratio) and ratio > 0 and abs(ratio - 1) > 0.02:
                    for c in ("Open", "High", "Low", "Close"):
                        long[c] = long[c] * ratio
        parts.append(recent)
    if not parts:
        return None
    df = pd.concat(parts)
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    df = df[~df.index.duplicated(keep="last")].sort_index().dropna(subset=["Close"])
    for c in ("Open", "High", "Low"):
        if c not in df.columns:
            df[c] = df["Close"]
        df[c] = df[c].fillna(df["Close"])
    if "Volume" not in df.columns:
        df["Volume"] = 0
    df["Volume"] = df["Volume"].fillna(0)
    return df if len(df) else None


def series(ticker: str, tf: str = "d", region: str | None = None) -> dict | None:
    df = _merged(ticker)
    if df is None:
        return None
    c = df["Close"]
    for n in (50, 150, 200):
        df[f"ma{n}"] = c.rolling(n).mean()
    if tf in TF_RULE:
        g = df.groupby(pd.Grouper(freq=TF_RULE[tf]))
        out = g.agg(Open=("Open", "first"), High=("High", "max"), Low=("Low", "min"), Close=("Close", "last"), Volume=("Volume", "sum"),
                    ma50=("ma50", "last"), ma150=("ma150", "last"), ma200=("ma200", "last"))
        last_day = g.apply(lambda x: x.index[-1] if len(x) else pd.NaT)
        out.index = pd.DatetimeIndex(last_day.reindex(out.index).to_numpy())
        out = out[out["Close"].notna()]
    else:
        out = df
    nd = 0 if region == "KR" else 2
    bars = []
    for t, r in out.iterrows():
        def f(v, d=nd):
            return None if v is None or (isinstance(v, float) and not math.isfinite(v)) else round(float(v), d)
        bars.append({"t": t.strftime("%Y-%m-%d"), "o": f(r["Open"]), "h": f(r["High"]), "l": f(r["Low"]), "c": f(r["Close"]),
                     "v": int(r["Volume"]) if math.isfinite(float(r["Volume"])) else 0,
                     "ma50": f(r["ma50"]), "ma150": f(r["ma150"]), "ma200": f(r["ma200"])})
    marks = []
    try:
        pos = positions.load_positions().get(ticker)
    except Exception:  # noqa: BLE001
        pos = None
    if pos and pos.get("entry_date"):
        marks.append({"t": pos["entry_date"], "px": pos.get("entry_px"), "qty": pos.get("qty"), "kind": "entry"})
    return {"ticker": ticker, "tf": tf, "bars": bars, "first": bars[0]["t"], "last": bars[-1]["t"], "marks": marks}
