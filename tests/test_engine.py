import numpy as np
import pytest
from src.var_calculator.engine import VaREngine

def test_historical_var_is_positive():
    rng=np.random.default_rng(42)
    returns=rng.normal(0,0.01,(1000,2))
    engine=VaREngine(returns,[0.5,0.5],investment=100,confidence=0.95)
    assert 0 < engine.historical() < 5

def test_parametric_close_to_historical_for_normal_data():
    rng=np.random.default_rng(42)
    returns=rng.normal(0,0.01,(10000,2))
    engine=VaREngine(returns,[0.5,0.5],investment=1000,confidence=0.95)
    h_var,p_var=engine.historical(),engine.parametric()
    assert abs(h_var-p_var)/h_var < 0.10

def test_monte_carlo_is_reproducible():
    rng=np.random.default_rng(7)
    returns=rng.normal(0,0.01,(1000,2))
    engine=VaREngine(returns,[0.6,0.4],investment=1000,confidence=0.99)
    assert engine.monte_carlo(seed=123)==engine.monte_carlo(seed=123)

def test_invalid_weights_raise():
    with pytest.raises(ValueError):
        VaREngine(np.zeros((10,2)),[1.0])
