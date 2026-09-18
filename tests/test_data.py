import pandas as pd

def test_returns_shape_and_no_missing_values():
    prices=pd.DataFrame({"A":[100,102,101],"B":[50,51,52]})
    returns=prices.pct_change().dropna()
    assert returns.shape==(2,2)
    assert not returns.isna().any().any()
