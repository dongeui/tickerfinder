import yfinance as yf, pathlib
D = pathlib.Path(__file__).resolve().parent.parent / "data" / "etf_bt"  # scripts/ndx_retest_dl.py 가 받은 NDX.parquet
df = yf.download("^NDX", start="1985-01-01", auto_adjust=False, progress=False)
if hasattr(df.columns, "levels"): df.columns = df.columns.get_level_values(0)
df = df.dropna(subset=["Close"]); df.to_parquet(D/"NDX.parquet"); print(len(df), df.index[0].date(), df.index[-1].date(), float(df["Close"].iloc[-1]))
