import numpy as np
from src.var_calculator.engine import VaREngine

def test_var_is_positive():
    rng=np.random.default_rng(42)
    returns=rng.normal(0,0.01,(1000,2))
    engine=VaREngine(returns,[0.5,0.5],investment=100,confidence=0.95)
    assert engine.historical()>0

def test_parametric_and_historical_are_close():
    rng=np.random.default_rng(42)
    returns=rng.normal(0,0.01,(10000,2))
    engine=VaREngine(returns,[0.5,0.5],investment=1000,confidence=0.95)
    h,p=engine.historical(),engine.parametric()
    assert abs(h-p)/h<0.10

def test_monte_carlo_seed_is_reproducible():
    rng=np.random.default_rng(7)
    returns=rng.normal(0,0.01,(1000,2))
    engine=VaREngine(returns,[0.6,0.4],investment=1000,confidence=0.99)
    assert engine.monte_carlo(seed=123)==engine.monte_carlo(seed=123)
