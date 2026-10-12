"""누적 스냅샷(data/history)을 기반으로 한 시그널 분석.

- 날짜별 시그널 수
- 시그널 진입 이벤트 (전날 미충족 → 당일 충족) 와 이후 N거래일 수익률
- 종목별 지표 흐름
"""
from __future__ import annotations

import json
import statistics
from dataclasses import dataclass
from functools import lru_cache

from .scanner import HISTORY_DIR, load_results

HORIZONS = (5, 10, 20)


@dataclass(frozen=True)
class Thresholds:
    stoch_max: float = 25
    band_tol: float = 3
    mkt_lo: float = 30
    mkt_hi: float = 40
    rsi_d_min: float = 75
    rsi_w_lo: float = 65
    rsi_w_hi: float = 85
    stoch_tf_kr: str = "d"  # 국내 스토캐스틱 시간대 (d=일봉, w=주봉)
    stoch_tf_us: str = "w"  # 미국

    def stoch_tf(self, region: str) -> str:
        return self.stoch_tf_kr if region == "KR" else self.stoch_tf_us


def _history_key() -> tuple:
    files = sorted(HISTORY_DIR.glob("*.json"))
    return tuple((f.name, f.stat().st_mtime_ns) for f in files)


@lru_cache(maxsize=2)
def _load(key: tuple) -> list[dict]:
    days = []
    for name, _ in key:
        try:
            d = json.loads((HISTORY_DIR / name).read_text())
        except Exception:  # noqa: BLE001
            continue
        d["date"] = name[:-5]
        days.append(d)
    # 휴장 등으로 지수가 없는 날은 직전 값을 이어서 사용 (각 지수의 date 필드로 원래 날짜 확인 가능)
    last: dict = {}
    for d in days:
        idx = d.setdefault("indices", {})
        d["prev_indices"] = dict(last)  # 전일(직전 스냅샷) 지수 — 국내 종목은 나스닥 전일 종가로 판단
        for k, v in last.items():
            idx.setdefault(k, v)
        last.update(idx)
    return days


def load_history() -> list[dict]:
    return _load(_history_key())


def names_map() -> dict[str, dict]:
    cur = load_results() or {}
    return {s["ticker"]: s for s in cur.get("stocks", [])}


def mkt_ok(day: dict, th: Thresholds, region: str = "US", use_kr_index: bool = False) -> bool | None:
    """시장 환경 판단. 국내 종목에 나스닥을 쓸 때는 국내 장 마감 시점에 알 수 있는 '전일' 나스닥을 사용한다."""
    key = "KS11" if (use_kr_index and region == "KR") else "IXIC"
    src = day.get("prev_indices", {}) if (region == "KR" and key == "IXIC") else day.get("indices", {})
    ix = src.get(key, {})
    if ix.get("rsi_d") is None:
        return None
    return th.mkt_lo <= ix["rsi_d"] <= th.mkt_hi


def buy_hit(s: dict, th: Thresholds) -> bool:
    if s.get("suspect"):
        return False
    sk, bp = s.get("stoch_k"), s.get("band_pos")
    if th.stoch_tf(s.get("region", "US")) == "w":
        sk = s.get("stoch_k_w", sk)  # 주봉 값이 없는 옛 스냅샷은 일봉으로 대체
    return sk is not None and bp is not None and sk <= th.stoch_max and abs(bp) <= th.band_tol


def sell_hit(s: dict, th: Thresholds) -> bool:
    if s.get("suspect"):
        return False
    rd, rw = s.get("rsi_d"), s.get("rsi_w")
    return rd is not None and rw is not None and rd >= th.rsi_d_min and th.rsi_w_lo <= rw <= th.rsi_w_hi


def daily_counts(th: Thresholds, limit: int = 200) -> list[dict]:
    days = load_history()[-limit:]
    names = names_map()
    out = []
    for d in days:
        buy = [s["ticker"] for s in d["stocks"] if buy_hit(s, th)]
        sell = [s["ticker"] for s in d["stocks"] if sell_hit(s, th)]
        out.append({
            "date": d["date"], "updated_at": d.get("updated_at"), "backfilled": d.get("backfilled", False),
            "indices": {k: {kk: vv for kk, vv in v.items() if not str(kk).endswith("_series")} for k, v in d.get("indices", {}).items()},
            "mkt_ok": mkt_ok(d, th),
            "buy_count": len(buy), "sell_count": len(sell),
            "buy": [{"ticker": t, "name": names.get(t, {}).get("name", t)} for t in buy[:60]],
            "sell": [{"ticker": t, "name": names.get(t, {}).get("name", t)} for t in sell[:60]],
        })
    return out


def _series_by_ticker(days: list[dict]) -> dict[str, list[tuple[str, dict]]]:
    out: dict[str, list[tuple[str, dict]]] = {}
    for d in days:
        for s in d["stocks"]:
            out.setdefault(s["ticker"], []).append((d["date"], s))
    return out


def signal_events(mode: str, th: Thresholds, require_mkt: bool = False, use_kr_index: bool = False,
                  limit: int = 600) -> dict:
    """진입 이벤트와 이후 수익률. mode: buy | sell."""
    days = load_history()
    names = names_map()
    hit = buy_hit if mode == "buy" else sell_hit
    by_day = {d["date"]: d for d in days}
    events = []
    for ticker, rows in _series_by_ticker(days).items():
        prev = None  # 첫 스냅샷에서 이미 충족 중인 종목은 '진입'으로 세지 않는다
        meta = names.get(ticker, {})
        region = rows[0][1].get("region", meta.get("region", "US"))
        closes = [r[1].get("close") for r in rows]
        for i, (date, s) in enumerate(rows):
            cur = hit(s, th)
            if cur and require_mkt and mode == "buy":
                cur = bool(mkt_ok(by_day[date], th, region, use_kr_index))
            if cur and prev is False:
                c0 = s.get("close")
                ev = {"date": date, "ticker": ticker, "name": meta.get("name", ticker), "region": region,
                      "market": meta.get("market", ""), "close": c0,
                      "rsi_d": s.get("rsi_d"), "rsi_w": s.get("rsi_w"), "stoch_k": s.get("stoch_k"), "band_pos": s.get("band_pos"),
                      "mkt_ok": mkt_ok(by_day[date], th, region, use_kr_index)}
                for h in HORIZONS:
                    j = i + h
                    ev[f"fwd{h}"] = round((closes[j] / c0 - 1) * 100, 2) if c0 and j < len(closes) and closes[j] else None
                # 지금까지의 최대 상승/하락 (20일 내)
                seg = [c for c in closes[i + 1:i + 21] if c]
                ev["max20"] = round((max(seg) / c0 - 1) * 100, 2) if seg and c0 else None
                ev["min20"] = round((min(seg) / c0 - 1) * 100, 2) if seg and c0 else None
                events.append(ev)
            prev = cur
    events.sort(key=lambda e: (e["date"], e["ticker"]), reverse=True)

    def summarize(evs: list[dict]) -> dict:
        out = {"n": len(evs)}
        for h in HORIZONS:
            vals = [e[f"fwd{h}"] for e in evs if e.get(f"fwd{h}") is not None]
            if vals:
                out[f"fwd{h}"] = {"n": len(vals), "mean": round(statistics.mean(vals), 2),
                                   "median": round(statistics.median(vals), 2),
                                   "win": round(100 * sum(v > 0 for v in vals) / len(vals), 1)}
        return out

    summary = {"all": summarize(events),
               "KR": summarize([e for e in events if e["region"] == "KR"]),
               "US": summarize([e for e in events if e["region"] == "US"]),
               "mkt_ok": summarize([e for e in events if e["mkt_ok"]]),
               "mkt_no": summarize([e for e in events if e["mkt_ok"] is False])}
    return {"mode": mode, "events": events[:limit], "total": len(events), "summary": summary,
            "span": [days[0]["date"], days[-1]["date"]] if days else None}


def ticker_series(ticker: str) -> dict:
    days = load_history()
    meta = names_map().get(ticker, {})
    rows = [{"date": d["date"], **s} for d in days for s in d["stocks"] if s["ticker"] == ticker]
    idx = [{"date": d["date"], "ixic_rsi": d.get("indices", {}).get("IXIC", {}).get("rsi_d"),
            "ks11_rsi": d.get("indices", {}).get("KS11", {}).get("rsi_d")} for d in days]
    return {"ticker": ticker, "name": meta.get("name", ticker), "region": meta.get("region"), "rows": rows, "market": idx}


def streaks(mode: str, th: Thresholds) -> dict[str, dict]:
    """종목별: 최신 스냅샷 기준 연속 충족 일수(streak), 오늘 이탈 여부(exited, prev_streak), 최근 60일 충족 일수."""
    days = load_history()
    hit = buy_hit if mode == "buy" else sell_hit
    out: dict[str, dict] = {}
    for ticker, rows in _series_by_ticker(days).items():
        hits = [hit(s, th) for _, s in rows]
        n = len(hits)
        streak = 0
        for h in reversed(hits):
            if not h:
                break
            streak += 1
        exited, prev_streak = False, 0
        if n >= 2 and not hits[-1] and hits[-2]:
            exited = True
            for h in reversed(hits[:-1]):
                if not h:
                    break
                prev_streak += 1
        first_date = rows[n - streak][0] if streak else None
        out[ticker] = {"streak": streak, "since": first_date, "last_date": rows[-1][0], "exited": exited,
                       "prev_streak": prev_streak, "hits60": int(sum(hits[-60:])), "chg_last": None}
        if exited and rows[-1][1].get("close") and rows[-2][1].get("close"):
            out[ticker]["chg_last"] = round((rows[-1][1]["close"] / rows[-2][1]["close"] - 1) * 100, 2)
    return out
