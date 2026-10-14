import yfinance as yf, pandas as pd, pathlib
OUT = pathlib.Path(__file__).resolve().parent.parent / "data" / "etf_bt"; OUT.mkdir(parents=True, exist_ok=True)
syms = ["QQQ","SPY","DIA","IWM","SMH","XLK","SCHD","VIG","XLV","XLF","XLE","KRW=X","133690.KS","360750.KS","143850.KS","245340.KS","379810.KS","381180.KS","458730.KS"]
df = yf.download(syms, start="2000-01-01", auto_adjust=True, group_by="ticker", threads=True, progress=False)
for s in syms:
    try:
        d = df[s].dropna(subset=["Close"])
        d.to_parquet(OUT / (s.replace("=","_").replace(".","_") + ".parquet"))
        print(s, len(d), d.index[0].date(), d.index[-1].date())
    except Exception as e:
        print(s, "FAIL", e)
