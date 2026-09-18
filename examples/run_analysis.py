from src.var_calculator.data_loader import fetch_market_data
from src.var_calculator.engine import VaREngine
from src.var_calculator.visualization import plot_distribution

if __name__=="__main__":
    tickers=["AAPL","MSFT","GOOGL"]
    weights=[0.4,0.4,0.2]
    returns=fetch_market_data(tickers,"2020-01-01","2023-12-31")
    engine=VaREngine(returns.values,weights,investment=1000000,confidence=0.99,horizon=10)
    print("10-Day 99% VaR (Historical):  $"+format(engine.historical(),",.2f"))
    print("10-Day 99% VaR (Parametric):  $"+format(engine.parametric(),",.2f"))
    print("10-Day 99% VaR (Monte Carlo): $"+format(engine.monte_carlo(seed=42),",.2f"))
    print("10-Day 99% CVaR:               $"+format(engine.cvar_historical(),",.2f"))
    plot_distribution(engine.port_returns,engine.confidence)
