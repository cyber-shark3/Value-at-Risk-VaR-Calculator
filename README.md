# Value at Risk (VaR) Calculator

A modular Python risk tool for portfolio Value at Risk using historical simulation, parametric variance-covariance and Monte Carlo methods. It also includes historical Expected Shortfall (CVaR).

## Methods

- Historical simulation: lower (1 - confidence) percentile of observed portfolio returns.
- Parametric: portfolio mean, volatility and Normal or Student-t quantiles.
- Monte Carlo: correlated synthetic return scenarios from estimated means and covariance.
- CVaR: average loss in the historical tail beyond the VaR cutoff.

## Horizon scaling

Multi-day results use the common square-root-of-time approximation: daily VaR multiplied by sqrt(horizon). This is an approximation and depends on return assumptions.

## Run

    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    pytest -q

See examples/run_analysis.py for a Yahoo Finance example.

## Risk note

VaR is a loss threshold, not a maximum-loss guarantee. Results depend on the data window, estimation method, distribution assumptions and data quality.
