"""스토캐스틱을 일봉/주봉/월봉으로 계산했을 때 매수 조건 백테스트 비교.

주봉·월봉은 각 일자 t 기준으로 "t가 속한 주/월을 t까지의 미완성 봉"으로 취급한다 (차트 프로그램과 동일, 미래 정보 없음).
사용: uv run python scripts/stoch_timeframe.py [stoch_max=25]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.analysis import Thresholds  # noqa: E402
from app.backfill import compute_series  # noqa: E402
from app.backtest import HORIZONS, _fwd_table, _gate_series  # noqa: E402
from app.indicators import stoch_point_in_time as stoch_tf_daily  # noqa: E402
from app.longhist import load_long  # noqa: E402
from app.universe import DATA_DIR  # noqa: E402



def main():
    st_max = float(sys.argv[1]) if len(sys.argv) > 1 else 25
    th = Thresholds(stoch_max=st_max)
    gate = _gate_series(th, use_kr_index=True)
    uni = json.loads((DATA_DIR / "universe.json").read_text())
    rows = []
    for i, u in enumerate(uni):
        df = load_long(u["ticker"])
        if df is None or len(df) < 400:
            continue
        ser = compute_series(df)
        c = ser["close"]
        g = gate[u["region"]].reindex(c.index, method="ffill").fillna(False).astype(bool)
        base = (ser["band_pos"].abs() <= th.band_tol) & g & ~ser["suspect"].astype(bool)
        fw = _fwd_table(c)
        sk = {"d": ser["stoch_k"], "w": stoch_tf_daily(df, "W-FRI"), "m": stoch_tf_daily(df, "M")}
        for tf, s in sk.items():
            hit = (base & (s <= st_max)).fillna(False)
            entry = hit & ~hit.shift(1, fill_value=False)
            for t in c.index[entry.to_numpy()]:
                r = {"tf": tf, "region": u["region"], "ticker": u["ticker"], "date": t}
                for hz in HORIZONS:
                    r[f"fwd{hz}"] = fw[f"fwd{hz}"][t]
                    r[f"max{hz}"] = fw[f"max{hz}"][t]
                rows.append(r)
        if i % 200 == 0:
            print(f"... {i}/{len(uni)}", file=sys.stderr)
    R = pd.DataFrame(rows)
    R.to_csv(DATA_DIR / "stoch_timeframe_events.csv", index=False)
    names = {"d": "일봉", "w": "주봉", "m": "월봉"}
    for reg in ("US", "KR"):
        print(f"\n== {reg} (스토캐스틱 ≤{st_max:g}, 밴드 ±3%, 게이트 적용) ==")
        print(f"{'':6}{'건수':>7}{'종목수':>6} | " + " | ".join(f"{hz:>3}일 승률/반등/평균/중앙" for hz in HORIZONS))
        for tf in ("d", "w", "m"):
            g = R[(R.region == reg) & (R.tf == tf)]
            cells = []
            for hz in HORIZONS:
                f = g[f"fwd{hz}"].dropna()
                m = g.loc[f.index, f"max{hz}"]
                cells.append(f"{(f > 0).mean() * 100:4.0f}%/{(m >= 10).mean() * 100:3.0f}%/{f.mean():+5.1f}/{f.median():+5.1f}" if len(f) else "        -        ")
            print(f"{names[tf]:6}{len(g):>7}{g.ticker.nunique():>6} | " + " | ".join(cells))
    # 겹침: 주봉 신호 중 같은 날 일봉도 충족한 비율
    for reg in ("US", "KR"):
        d = set(map(tuple, R[(R.region == reg) & (R.tf == "d")][["ticker", "date"]].values))
        for tf in ("w", "m"):
            s = R[(R.region == reg) & (R.tf == tf)]
            ov = sum((t, dt) in d for t, dt in s[["ticker", "date"]].values)
            print(f"{reg} {names[tf]} 진입 {len(s)}건 중 같은 날 일봉도 진입인 비율 {ov / max(1, len(s)) * 100:.0f}%")


if __name__ == "__main__":
    main()
