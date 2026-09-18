"""Market data loading helpers."""
from __future__ import annotations
import pandas as pd
import yfinance as yf

def fetch_market_data(tickers, start_date, end_date):
    """Fetch daily adjusted prices and return simple percentage returns."""
    tickers = list(tickers)
    if not tickers:
        raise ValueError("tickers cannot be empty")
    data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=False,
                       progress=False, group_by="ticker")
    if data.empty:
        raise ValueError("No market data returned for the requested period")
    if len(tickers) == 1:
        prices = data[(tickers[0], "Adj Close")] if isinstance(data.columns, pd.MultiIndex) else data["Adj Close"]
        prices = prices.to_frame(name=tickers[0])
    else:
        prices = pd.DataFrame({t: data[(t, "Adj Close")] for t in tickers})
    prices = prices.dropna(how="all").ffill().dropna()
    returns = prices.pct_change().dropna()
    if returns.empty:
        raise ValueError("Not enough price data to calculate returns")
    return returns
